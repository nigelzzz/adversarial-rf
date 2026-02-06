#!/usr/bin/env python
"""
Create side-by-side visualizations for Intact, CW, and Top-K FFT signals:

- I/Q constellation distributions (scatter)
- Frequency-domain average magnitude spectra

Per modulation: BPSK, QPSK, QAM16, QAM64 at SNR=18 by default.

Usage (defaults work with repo layout):
    python visualize_cw_iq_freq_grid.py \
        --dataset_path ./data/RML2016.10a_dict.pkl \
        --model_path ./2016.10a_AWN.pkl \
        --snr 18 --samples_per_mod 500 \
        --cw_steps 100 --topk 50 \
        --ta_box minmax  # use per-sample min–max mapping to avoid clipping

Outputs:
    - cw_analysis/iq_grid.png (and .pdf)
    - cw_analysis/freq_grid.png (and .pdf)
"""

from typing import List, Tuple, Dict
import os
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from util.utils import create_AWN_model, recover_constellation
from util.config import Config
from util.defense import normalize_iq_data, denormalize_iq_data, fft_topk_denoise
from util.adv_attack import (
    Model01Wrapper,
    iq_to_ta_input,
    ta_output_to_iq,
    iq_to_ta_input_minmax,
    ta_output_to_iq_minmax,
)


def _load_mod_subset_single(
    ds_path: str,
    snr: int,
    mods: List[bytes],
    n_per_mod: int,
    label_map: Dict[bytes, int],
) -> Tuple[Dict[bytes, torch.Tensor], Dict[bytes, torch.Tensor], List[bytes], List[int]]:
    with open(ds_path, "rb") as f:
        Xd = pickle.load(f, encoding="bytes")
    _, all_mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1, 0])
    X = {}
    y = {}
    for m in mods:
        data = Xd[(m, snr)][:n_per_mod]
        X[m] = torch.from_numpy(data).float()
        y[m] = torch.full((len(data),), label_map[m], dtype=torch.long)
    return X, y, mods, [snr]


def _load_mod_subset_snr_gt(
    ds_path: str,
    snr_threshold: int,
    mods: List[bytes],
    n_per_mod: int,
    label_map: Dict[bytes, int],
) -> Tuple[Dict[bytes, torch.Tensor], Dict[bytes, torch.Tensor], List[bytes], List[int]]:
    """Load up to n_per_mod samples per modulation, stratified across SNR >= threshold.

    If n_per_mod <= 0, use all available samples for the selected SNR bins.
    """
    with open(ds_path, "rb") as f:
        Xd = pickle.load(f, encoding="bytes")
    snrs, all_mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1, 0])
    snr_bins = [s for s in snrs if s >= snr_threshold]
    if not snr_bins:
        raise ValueError("No SNR bins at or above threshold")
    use_all = n_per_mod is None or n_per_mod <= 0
    per = None if use_all else max(1, int(np.ceil(n_per_mod / len(snr_bins))))
    X: Dict[bytes, torch.Tensor] = {}
    y: Dict[bytes, torch.Tensor] = {}
    for m in mods:
        chunks = []
        for s in snr_bins:
            arr = Xd[(m, s)]
            take = arr.shape[0] if use_all else min(per, arr.shape[0])
            chunks.append(torch.from_numpy(arr[:take]).float())
        x_cat = torch.cat(chunks, dim=0)
        X[m] = x_cat if use_all else x_cat[:n_per_mod]
        y[m] = torch.full((X[m].shape[0],), label_map[m], dtype=torch.long)
    return X, y, mods, snr_bins


def _avg_mag_spectrum(batch: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    """Return (freq, avg_mag) for combined I/Q magnitude using rFFT.

    batch: [N,2,T] in real domain.
    """
    N, C, T = batch.shape
    F = T // 2 + 1
    # rFFT over time for both channels; average magnitude across N and channels
    Xi = torch.fft.rfft(batch[:, 0, :], n=T, dim=1)
    Xq = torch.fft.rfft(batch[:, 1, :], n=T, dim=1)
    mag = (Xi.abs() + Xq.abs()) / 2.0  # [N, F]
    avg = mag.mean(dim=0).detach().cpu().numpy()
    # Normalized frequency [0, 0.5]
    freq = np.linspace(0.0, 0.5, F)
    return freq, avg


def main():
    import argparse

    parser = argparse.ArgumentParser(description="CW vs Top-K FFT visualization (IQ & spectrum)")
    parser.add_argument("--dataset_path", type=str, default="./data/RML2016.10a_dict.pkl")
    parser.add_argument("--model_path", type=str, default="./2016.10a_AWN.pkl")
    parser.add_argument("--snr", type=int, default=18)
    parser.add_argument("--snr_mode", type=str, default="single", choices=["single", "gt"],
                        help="single: use exact SNR; gt: aggregate SNR > threshold")
    parser.add_argument("--snr_threshold", type=int, default=0,
                        help="Threshold used when snr_mode=gt; for SNR>=0 use --snr_mode gt --snr_threshold 0")
    parser.add_argument("--samples_per_mod", type=int, default=500,
                        help="Per-mod limit; set <=0 to use all samples for selected SNR bins")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cw_steps", type=int, default=100)
    parser.add_argument("--cw_c", type=float, default=1.0)
    parser.add_argument("--cw_kappa", type=float, default=0.0)
    parser.add_argument("--cw_lr", type=float, default=1e-2)
    parser.add_argument("--lowpass", action=argparse.BooleanOptionalAction, default=True,
                        help="Apply lowpass smoothing to CW perturbation")
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument("--topk_list", type=str, default=None,
                        help="Comma-separated Top-K values to sweep; picks best for plots")
    parser.add_argument("--mods", type=str, default="BPSK,QPSK,QAM16,QAM64")
    parser.add_argument("--ta_box", type=str, default="minmax", choices=["unit", "minmax"],
                        help="Mapping for torchattacks input box-constraint")
    args = parser.parse_args()

    device = torch.device(args.device)
    mods = [m.encode() for m in args.mods.split(",")]

    # Build model early to get label mapping
    cfg = Config("2016.10a", train=False)
    cfg.device = device
    model = create_AWN_model(cfg)
    ckpt = torch.load(args.model_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt)
    model.eval()
    label_map = cfg.classes

    # Load subset with AWN label mapping
    if args.snr_mode == "gt":
        X_dict, y_dict, mods, snr_bins = _load_mod_subset_snr_gt(
            args.dataset_path, args.snr_threshold, mods, args.samples_per_mod, label_map
        )
        snr_desc = f"SNR>{args.snr_threshold} ({','.join(map(str, snr_bins))})"
    else:
        X_dict, y_dict, mods, snr_bins = _load_mod_subset_single(
            args.dataset_path, args.snr, mods, args.samples_per_mod, label_map
        )
        snr_desc = f"SNR={args.snr}"

    # Attack setup (torchattacks + wrapper)
    import torchattacks

    wrapped = Model01Wrapper(model).eval()
    atk = torchattacks.CW(wrapped, c=args.cw_c, kappa=args.cw_kappa, steps=args.cw_steps, lr=args.cw_lr)

    X_adv: Dict[bytes, torch.Tensor] = {}
    X_rec_best: Dict[bytes, torch.Tensor] = {}
    acc: Dict[bytes, Dict[str, float]] = {}
    per_k: Dict[int, Dict[bytes, float]] = {}
    sweep_topk: List[int] = []
    if args.topk_list:
        try:
            sweep_topk = [int(x) for x in args.topk_list.split(',') if x.strip()]
        except Exception:
            sweep_topk = []
    if not sweep_topk:
        sweep_topk = [args.topk]

    for m in mods:
        x = X_dict[m].to(device)
        y = y_dict[m].to(device)

        if args.ta_box == "minmax":
            x01_4d, a, b = iq_to_ta_input_minmax(x)
            wrapped.set_minmax(a, b)
            adv01_4d = atk(x01_4d, y)
            adv = ta_output_to_iq_minmax(adv01_4d, a, b)
            wrapped.clear_minmax()
        else:
            x01_4d = iq_to_ta_input(x)
            adv01_4d = atk(x01_4d, y)
            adv = ta_output_to_iq(adv01_4d)
        if args.lowpass:
            delta = adv - x
            delta = _lowpass_filter(delta, kernel_size=17)
            clip_min = x.amin(dim=(1, 2), keepdim=True)
            clip_max = x.amax(dim=(1, 2), keepdim=True)
            adv = _batch_clip(x + delta, clip_min, clip_max)

        X_adv[m] = adv.detach().cpu()

        # Sweep Top-K values and pick the best by accuracy
        best_k = None
        best_rec = None
        best_rec_acc = -1.0
        per_k_mod: Dict[int, float] = {}
        for k in sweep_topk:
            xn = normalize_iq_data(adv, 0.02, 0.04)
            xr = fft_topk_denoise(xn, topk=int(k))
            xr = denormalize_iq_data(xr, 0.02, 0.04)
            with torch.no_grad():
                lr, _ = model(xr)
            yr = lr.argmax(dim=1).cpu()
            y_true = y.cpu()
            rec_acc = float((yr == y_true).float().mean().item() * 100.0)
            per_k_mod[int(k)] = rec_acc
            if rec_acc > best_rec_acc:
                best_rec_acc = rec_acc
                best_k = int(k)
                best_rec = xr.detach().cpu()
        X_rec_best[m] = best_rec
        # accumulate per-k
        for k, val in per_k_mod.items():
            per_k.setdefault(k, {})[m] = val

        # Base accuracies (clean and adv)
        with torch.no_grad():
            lc, _ = model(x)
            la, _ = model(adv)
        yc = lc.argmax(dim=1).cpu()
        ya = la.argmax(dim=1).cpu()
        y_true = y.cpu()
        acc[m] = {
            'clean': float((yc == y_true).float().mean().item() * 100.0),
            'adv': float((ya == y_true).float().mean().item() * 100.0),
            'rec': best_rec_acc,
            'best_topk': best_k,
        }

    # Build I/Q grid figure (raw IQ + recovered constellation)
    os.makedirs("cw_analysis", exist_ok=True)

    # Helper: flatten and subsample raw IQ
    def _flat(x: np.ndarray, n: int = 12000):
        I = x[:, 0, :].reshape(-1)
        Q = x[:, 1, :].reshape(-1)
        if len(I) > n:
            sel = np.random.choice(len(I), n, replace=False)
            return I[sel], Q[sel]
        return I, Q

    # Helper: recover constellation from all samples and subsample
    def _flat_constellation(x: np.ndarray, sps: int = 8, n: int = 12000):
        Is, Qs = [], []
        for j in range(x.shape[0]):
            ic, qc = recover_constellation(x[j, 0, :], x[j, 1, :], sps=sps)
            Is.append(ic)
            Qs.append(qc)
        I = np.concatenate(Is)
        Q = np.concatenate(Qs)
        if len(I) > n:
            sel = np.random.choice(len(I), n, replace=False)
            return I[sel], Q[sel]
        return I, Q

    # --- Raw IQ grid (4 mods x 3 columns) ---
    fig_iq = plt.figure(figsize=(16, 12))
    gs_iq = GridSpec(4, 3, figure=fig_iq, hspace=0.35, wspace=0.25)
    fig_iq.suptitle(
        f"I/Q Raw Trajectory ({snr_desc}, CW steps={args.cw_steps}, c={args.cw_c}, TopK={args.topk}, box={args.ta_box})",
        fontsize=14,
        fontweight="bold",
    )

    for idx, m in enumerate(mods):
        name = m.decode()
        clean = X_dict[m].numpy()
        adv = X_adv[m].numpy()
        rec = X_rec_best[m].numpy()

        I0, Q0 = _flat(clean)
        I1, Q1 = _flat(adv)
        I2, Q2 = _flat(rec)

        ax = fig_iq.add_subplot(gs_iq[idx, 0])
        ax.scatter(I0, Q0, s=2, alpha=0.35, c="#1f77b4", edgecolors="none")
        ax.set_title(f"{name}\nIntact")
        ax.set_xlabel("In-phase (I)")
        ax.set_ylabel("Quadrature (Q)")
        ax.grid(True, alpha=0.2)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-0.02, 0.02)
        ax.set_ylim(-0.02, 0.02)

        ax = fig_iq.add_subplot(gs_iq[idx, 1])
        ax.scatter(I1, Q1, s=2, alpha=0.35, c="#d62728", edgecolors="none")
        ax.set_title(f"{name}\nCW Attack")
        ax.set_xlabel("In-phase (I)")
        ax.set_ylabel("Quadrature (Q)")
        ax.grid(True, alpha=0.2)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-0.02, 0.02)
        ax.set_ylim(-0.02, 0.02)

        ax = fig_iq.add_subplot(gs_iq[idx, 2])
        ax.scatter(I2, Q2, s=2, alpha=0.35, c="#2ca02c", edgecolors="none")
        ax.set_title(f"{name}\nTop-K FFT Recovered")
        ax.set_xlabel("In-phase (I)")
        ax.set_ylabel("Quadrature (Q)")
        ax.grid(True, alpha=0.2)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-0.02, 0.02)
        ax.set_ylim(-0.02, 0.02)

    out_iq_png = "cw_analysis/iq_grid.png"
    out_iq_pdf = "cw_analysis/iq_grid.pdf"
    fig_iq.savefig(out_iq_png, dpi=200, bbox_inches="tight")
    fig_iq.savefig(out_iq_pdf, bbox_inches="tight")
    plt.close(fig_iq)
    print(f"Saved I/Q raw grid: {out_iq_png}")

    # --- Recovered Constellation grid (4 mods x 3 columns) ---
    fig_const = plt.figure(figsize=(16, 12))
    gs_const = GridSpec(4, 3, figure=fig_const, hspace=0.35, wspace=0.25)
    fig_const.suptitle(
        f"Recovered Constellation ({snr_desc}, CW steps={args.cw_steps}, c={args.cw_c}, TopK={args.topk}, box={args.ta_box})",
        fontsize=14,
        fontweight="bold",
    )

    for idx, m in enumerate(mods):
        name = m.decode()
        clean = X_dict[m].numpy()
        adv = X_adv[m].numpy()
        rec = X_rec_best[m].numpy()

        I0, Q0 = _flat_constellation(clean)
        I1, Q1 = _flat_constellation(adv)
        I2, Q2 = _flat_constellation(rec)

        for col, (Ip, Qp, color, label) in enumerate([
            (I0, Q0, "#1f77b4", "Intact"),
            (I1, Q1, "#d62728", "CW Attack"),
            (I2, Q2, "#2ca02c", "Top-K FFT Recovered"),
        ]):
            ax = fig_const.add_subplot(gs_const[idx, col])
            ax.scatter(Ip, Qp, s=5, alpha=0.4, c=color, edgecolors="none")
            ax.set_title(f"{name}\n{label}")
            ax.set_xlabel("I")
            ax.set_ylabel("Q")
            ax.grid(True, alpha=0.2)
            ax.set_aspect("equal", adjustable="box")

    out_const_png = "cw_analysis/constellation_grid.png"
    out_const_pdf = "cw_analysis/constellation_grid.pdf"
    fig_const.savefig(out_const_png, dpi=200, bbox_inches="tight")
    fig_const.savefig(out_const_pdf, bbox_inches="tight")
    plt.close(fig_const)
    print(f"Saved constellation grid: {out_const_png}")

    # Build frequency grid figure
    fig_f = plt.figure(figsize=(16, 12))
    gs_f = GridSpec(4, 3, figure=fig_f, hspace=0.35, wspace=0.25)
    fig_f.suptitle(
        f"Frequency Magnitude (avg) ({snr_desc}, CW steps={args.cw_steps}, c={args.cw_c}, TopK={args.topk}, box={args.ta_box})",
        fontsize=14,
        fontweight="bold",
    )

    for idx, m in enumerate(mods):
        name = m.decode()
        f0, mag0 = _avg_mag_spectrum(X_dict[m])
        f1, mag1 = _avg_mag_spectrum(X_adv[m])
        f2, mag2 = _avg_mag_spectrum(X_rec_best[m])

        ax = fig_f.add_subplot(gs_f[idx, 0])
        ax.plot(f0, mag0, "#1f77b4", lw=0.9)
        ax.set_title(f"{name}\nIntact (avg)")
        ax.set_xlabel("Normalized Frequency")
        ax.set_ylabel("|FFT| (avg)")
        ax.grid(True, alpha=0.3)
        ax.set_yscale("log")

        ax = fig_f.add_subplot(gs_f[idx, 1])
        ax.plot(f1, mag1, "#d62728", lw=0.9)
        ax.set_title(f"{name}\nCW Attack (avg)")
        ax.set_xlabel("Normalized Frequency")
        ax.set_ylabel("|FFT| (avg)")
        ax.grid(True, alpha=0.3)
        ax.set_yscale("log")

        ax = fig_f.add_subplot(gs_f[idx, 2])
        ax.plot(f2, mag2, "#2ca02c", lw=0.9)
        ax.axvline(args.topk / (2.0 * X_dict[m].shape[-1]), color="orange", ls="--", lw=1.2, alpha=0.8,
                   label=f"Top-{args.topk} cutoff")
        ax.set_title(f"{name}\nTop-K FFT (avg)")
        ax.set_xlabel("Normalized Frequency")
        ax.set_ylabel("|FFT| (avg)")
        ax.grid(True, alpha=0.3)
        ax.set_yscale("log")
        ax.legend(fontsize=8)

    out_f_png = "cw_analysis/freq_grid.png"
    out_f_pdf = "cw_analysis/freq_grid.pdf"
    fig_f.savefig(out_f_png, dpi=200, bbox_inches="tight")
    fig_f.savefig(out_f_pdf, bbox_inches="tight")
    plt.close(fig_f)
    print(f"Saved frequency grid: {out_f_png}")

    # Print accuracy table and save JSON
    import json
    print("\nAccuracy by modulation (%):")
    print(f"{'Mod':<8} {'Clean':>7} {'CW':>7} {'Rec':>7} {'Gain':>7}")
    print('-'*40)
    overall = {'clean': [], 'adv': [], 'rec': []}
    for m in mods:
        a = acc[m]
        gain = a['rec'] - a['adv']
        print(f"{m.decode():<8} {a['clean']:7.2f} {a['adv']:7.2f} {a['rec']:7.2f} {gain:7.2f}")
        overall['clean'].append(a['clean'])
        overall['adv'].append(a['adv'])
        overall['rec'].append(a['rec'])
    o_clean = float(np.mean(overall['clean']))
    o_adv = float(np.mean(overall['adv']))
    o_rec = float(np.mean(overall['rec']))
    print('-'*40)
    print(f"Overall  {o_clean:7.2f} {o_adv:7.2f} {o_rec:7.20f} {o_rec-o_adv:7.2f}")

    res = {
        'config': {
            'snr_mode': args.snr_mode,
            'snr': args.snr,
            'snr_threshold': args.snr_threshold,
            'snr_bins': snr_bins,
            'samples_per_mod': args.samples_per_mod,
            'cw_steps': args.cw_steps,
            'cw_c': args.cw_c,
            'topk': args.topk,
            'ta_box': args.ta_box,
            'mods': [m.decode() for m in mods],
        },
        'per_mod': {m.decode(): acc[m] for m in mods},
        'per_topk': {str(k): {m.decode(): per_k[k][m] for m in mods if m in per_k[k]} for k in sorted(per_k)},
        'overall': {'clean': o_clean, 'adv': o_adv, 'rec': o_rec, 'gain': o_rec - o_adv},
        'figures': {'iq_grid': out_iq_png, 'freq_grid': out_f_png},
    }
    out_json = 'cw_analysis/accuracy_diff.json'
    with open(out_json, 'w') as f:
        json.dump(res, f, indent=2)
    print(f"Saved accuracy summary: {out_json}")


if __name__ == "__main__":
    main()
