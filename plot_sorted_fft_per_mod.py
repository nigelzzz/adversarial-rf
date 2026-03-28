#!/usr/bin/env python3
"""
Plot sorted FFT magnitude curves per modulation.
Shows clean vs CW-attacked, with the 5% knee point marked.
"""

import os, sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

from data_loader.data_loader import Load_Dataset, Dataset_Split
from util.config import Config
from util.utils import fix_seed, create_model
from util.logger import create_logger
from util.adv_attack import (
    Model01Wrapper, iq_to_ta_input_minmax, ta_output_to_iq_minmax,
)
import torchattacks


def sorted_mag_stats(x):
    """Compute sorted FFT magnitude (mean over I/Q). Returns [N, T] sorted descending."""
    X = torch.fft.fft(x, dim=2)
    mag = X.abs().mean(dim=1)  # avg I/Q → [N, T]
    sorted_mag, _ = mag.sort(dim=1, descending=True)
    return sorted_mag


def find_knee(sorted_mag_avg, thresh=0.05):
    """Find index where mag drops below thresh * peak."""
    peak = sorted_mag_avg[0]
    if peak < 1e-12:
        return len(sorted_mag_avg)
    ratio = sorted_mag_avg / peak
    below = (ratio < thresh).nonzero(as_tuple=True)[0]
    if len(below) == 0:
        return len(sorted_mag_avg)
    return int(below[0].item())


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--ckpt_path', type=str, default='./checkpoint')
    parser.add_argument('--seed', type=int, default=2022)
    parser.add_argument('--cw_steps', type=int, default=100)
    parser.add_argument('--cw_c', type=float, default=1.0)
    parser.add_argument('--cw_kappa', type=float, default=1.0)
    parser.add_argument('--cw_lr', type=float, default=0.001)
    parser.add_argument('--n_samples', type=int, default=200,
                        help='Samples per mod for averaging')
    args = parser.parse_args()

    fix_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Load
    cfg = Config('2016.10a', train=False)
    cfg.init_dir('sorted_fft_plots')
    cfg.device = device
    logger = create_logger(os.path.join(cfg.log_dir, 'log.txt'))

    model = create_model(cfg, 'awn')
    model.load_state_dict(torch.load(
        os.path.join(args.ckpt_path, '2016.10a_AWN.pkl'), map_location=device))
    model.to(device).eval()

    idx_to_mod = {}
    for k, v in cfg.classes.items():
        idx_to_mod[v] = k.decode() if isinstance(k, bytes) else str(k)

    Signals, Labels, SNRs, snrs, mods = Load_Dataset('2016.10a', logger)
    _, test_set, _, test_idx = Dataset_Split(Signals, Labels, snrs, mods, logger)
    sig_test, lab_test = test_set
    test_snrs = np.array([SNRs[i] for i in test_idx])

    # Filter SNR >= 0
    mask = test_snrs >= 0
    sig_test, lab_test = sig_test[mask], lab_test[mask]

    sig_test = sig_test.to(device)
    lab_test = lab_test.to(device)

    mod_order = ['BPSK', 'QPSK', '8PSK', 'QAM16', 'QAM64',
                 'PAM4', 'CPFSK', 'GFSK', 'AM-DSB', 'AM-SSB', 'WBFM']
    mod_best_k = {
        'AM-DSB': 2, 'QAM64': 10, 'PAM4': 15, 'GFSK': 15,
        'QAM16': 20, 'BPSK': 64, 'QPSK': 50, '8PSK': 64,
        'CPFSK': 64, 'AM-SSB': 64, 'WBFM': 10,
    }

    out_dir = os.path.join(cfg.result_dir, 'sorted_fft')
    os.makedirs(out_dir, exist_ok=True)

    # Generate CW attack per modulation and plot
    wrapped = Model01Wrapper(model).to(device)
    attack = torchattacks.CW(wrapped, c=args.cw_c, kappa=args.cw_kappa,
                              steps=args.cw_steps, lr=args.cw_lr)

    T = 128
    all_clean_curves = {}
    all_adv_curves = {}
    all_knees_clean = {}
    all_knees_adv = {}

    print("Generating per-mod plots...")
    for m in mod_order:
        midx = None
        for ci, cn in idx_to_mod.items():
            if cn == m:
                midx = ci
                break
        if midx is None:
            continue

        mod_mask = (lab_test == midx)
        x_mod = sig_test[mod_mask][:args.n_samples]
        y_mod = lab_test[mod_mask][:args.n_samples]
        n = len(y_mod)
        if n == 0:
            continue

        # CW attack
        chunks = []
        bs = 64
        for i in range(0, n, bs):
            xb, yb = x_mod[i:i+bs], y_mod[i:i+bs]
            x_ta, a, b = iq_to_ta_input_minmax(xb)
            wrapped.set_minmax(a, b)
            try:
                adv_ta = attack(x_ta, yb)
                adv_iq = ta_output_to_iq_minmax(adv_ta, a, b).detach()
            finally:
                wrapped.clear_minmax()
            chunks.append(adv_iq)
        x_adv = torch.cat(chunks)

        # Sorted magnitudes
        sm_clean = sorted_mag_stats(x_mod)  # [n, T]
        sm_adv = sorted_mag_stats(x_adv)

        # Average curves
        avg_clean = sm_clean.mean(dim=0).cpu().numpy()
        avg_adv = sm_adv.mean(dim=0).cpu().numpy()

        # Normalize to peak=1
        avg_clean_norm = avg_clean / max(avg_clean[0], 1e-12)
        avg_adv_norm = avg_adv / max(avg_adv[0], 1e-12)

        # Knee points
        knee_clean = find_knee(torch.tensor(avg_clean_norm), 0.05)
        knee_adv = find_knee(torch.tensor(avg_adv_norm), 0.05)

        all_clean_curves[m] = avg_clean_norm
        all_adv_curves[m] = avg_adv_norm
        all_knees_clean[m] = knee_clean
        all_knees_adv[m] = knee_adv

        # ── Individual plot per mod ─────────────────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f'{m} — Sorted FFT Magnitude (avg {n} samples, SNR>=0)',
                     fontsize=14, fontweight='bold')

        bins = np.arange(1, T + 1)

        # Left: absolute magnitude
        axes[0].semilogy(bins, avg_clean, color='#2d6a4f', linewidth=2, label='Clean')
        axes[0].semilogy(bins, avg_adv, color='#ae2012', linewidth=2, label='CW attacked')
        axes[0].set_xlabel('Rank (sorted by magnitude)', fontsize=11)
        axes[0].set_ylabel('FFT Magnitude (log)', fontsize=11)
        axes[0].set_title('Absolute Magnitude')
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xlim(1, T)

        # Right: normalized (peak=1) with knee
        axes[1].plot(bins, avg_clean_norm, color='#2d6a4f', linewidth=2, label='Clean')
        axes[1].plot(bins, avg_adv_norm, color='#ae2012', linewidth=2, label='CW attacked')
        axes[1].axhline(y=0.05, color='gray', linestyle='--', linewidth=1, label='5% threshold')

        # Mark knee points
        if knee_clean < T:
            axes[1].axvline(x=knee_clean, color='#2d6a4f', linestyle=':', linewidth=2)
            axes[1].plot(knee_clean, 0.05, 'o', color='#2d6a4f', markersize=10, zorder=5)
            axes[1].annotate(f'K={knee_clean}',
                            xy=(knee_clean, 0.05), xytext=(knee_clean + 5, 0.15),
                            fontsize=11, fontweight='bold', color='#2d6a4f',
                            arrowprops=dict(arrowstyle='->', color='#2d6a4f'))
        if knee_adv < T:
            axes[1].axvline(x=knee_adv, color='#ae2012', linestyle=':', linewidth=2)
            axes[1].plot(knee_adv, 0.05, 's', color='#ae2012', markersize=10, zorder=5)
            axes[1].annotate(f'K={knee_adv}',
                            xy=(knee_adv, 0.05), xytext=(knee_adv + 5, 0.25),
                            fontsize=11, fontweight='bold', color='#ae2012',
                            arrowprops=dict(arrowstyle='->', color='#ae2012'))

        axes[1].set_xlabel('Rank (sorted by magnitude)', fontsize=11)
        axes[1].set_ylabel('Normalized Magnitude (peak=1)', fontsize=11)
        axes[1].set_title(f'Normalized + Knee (best recovery K={mod_best_k.get(m, "?")})')
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xlim(1, T)
        axes[1].set_ylim(-0.02, 1.05)

        plt.tight_layout()
        path = os.path.join(out_dir, f'{m}_sorted_fft.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  {m}: knee_clean={knee_clean}, knee_adv={knee_adv} → {path}")

    # ── Combined overview: all mods in one figure ───────────────────
    fig, axes = plt.subplots(2, 1, figsize=(16, 12))
    fig.suptitle('Sorted FFT Magnitude — All Modulations (SNR >= 0, avg 200 samples)',
                 fontsize=15, fontweight='bold')

    colors = plt.cm.tab20(np.linspace(0, 1, len(mod_order)))
    bins = np.arange(1, T + 1)

    # Top: clean signals
    for i, m in enumerate(mod_order):
        if m not in all_clean_curves:
            continue
        axes[0].plot(bins, all_clean_curves[m], color=colors[i], linewidth=1.8,
                    label=f'{m} (K={all_knees_clean[m]})')
    axes[0].axhline(y=0.05, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    axes[0].set_xlabel('Rank', fontsize=11)
    axes[0].set_ylabel('Normalized Magnitude', fontsize=11)
    axes[0].set_title('Clean Signals', fontsize=13)
    axes[0].legend(fontsize=8, ncol=3, loc='upper right')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(1, T)
    axes[0].set_ylim(-0.02, 1.05)

    # Bottom: CW attacked
    for i, m in enumerate(mod_order):
        if m not in all_adv_curves:
            continue
        axes[1].plot(bins, all_adv_curves[m], color=colors[i], linewidth=1.8,
                    label=f'{m} (K={all_knees_adv[m]})')
    axes[1].axhline(y=0.05, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    axes[1].set_xlabel('Rank', fontsize=11)
    axes[1].set_ylabel('Normalized Magnitude', fontsize=11)
    axes[1].set_title('CW Attacked (minmax, c=1, kappa=1)', fontsize=13)
    axes[1].legend(fontsize=8, ncol=3, loc='upper right')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(1, T)
    axes[1].set_ylim(-0.02, 1.05)

    plt.tight_layout()
    path = os.path.join(out_dir, 'ALL_sorted_fft_overview.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\nOverview: {path}")

    # ── Combined: clean vs adv per mod (4x3 grid) ──────────────────
    fig, axes = plt.subplots(3, 4, figsize=(22, 14))
    fig.suptitle('Sorted FFT Magnitude per Modulation — Clean (green) vs CW Attacked (red)',
                 fontsize=15, fontweight='bold')

    for idx, m in enumerate(mod_order):
        row, col = idx // 4, idx % 4
        ax = axes[row][col]
        if m not in all_clean_curves:
            ax.set_visible(False)
            continue

        ax.plot(bins, all_clean_curves[m], color='#2d6a4f', linewidth=2, label='Clean')
        ax.plot(bins, all_adv_curves[m], color='#ae2012', linewidth=2, label='CW')
        ax.axhline(y=0.05, color='gray', linestyle='--', linewidth=0.8)

        kc = all_knees_clean[m]
        ka = all_knees_adv[m]
        if kc < T:
            ax.axvline(x=kc, color='#2d6a4f', linestyle=':', linewidth=1.5, alpha=0.7)
        if ka < T:
            ax.axvline(x=ka, color='#ae2012', linestyle=':', linewidth=1.5, alpha=0.7)

        ax.set_title(f'{m}  (clean K={kc}, adv K={ka})', fontsize=11, fontweight='bold')
        ax.set_xlim(1, T)
        ax.set_ylim(-0.02, 1.05)
        ax.grid(True, alpha=0.2)
        if idx == 0:
            ax.legend(fontsize=8)
        if row == 2:
            ax.set_xlabel('Rank', fontsize=9)
        if col == 0:
            ax.set_ylabel('Norm. Mag.', fontsize=9)

    # Hide unused subplot
    axes[2][3].set_visible(False)

    plt.tight_layout()
    path = os.path.join(out_dir, 'ALL_per_mod_grid.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Grid: {path}")

    # Print summary table
    print(f"\n{'Mod':<8s}  {'Knee(clean)':>12s}  {'Knee(adv)':>10s}  {'BestK':>6s}  {'Shift':>6s}")
    print("-" * 50)
    for m in mod_order:
        if m not in all_knees_clean:
            continue
        kc = all_knees_clean[m]
        ka = all_knees_adv[m]
        bk = mod_best_k.get(m, '?')
        shift = ka - kc
        print(f"{m:<8s}  {kc:>12d}  {ka:>10d}  {bk:>6}  {shift:>+6d}")


if __name__ == '__main__':
    main()
