#!/usr/bin/env python
"""
Compare IQ distributions and frequency spectra:
  Intact vs FGSM-unit vs FGSM-minmax for BPSK @ SNR=18.

Produces a 2x3 figure:
  Top row:    IQ scatter (Intact / unit / minmax)
  Bottom row: FFT magnitude spectrum (average + individual)

Usage:
    python plot_fgsm_unit_vs_minmax.py
    python plot_fgsm_unit_vs_minmax.py --eps 0.05 --n_samples 200
"""

import os
import sys
import pickle
import argparse
import numpy as np

import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Project imports
from util.utils import create_model
from util.config import Config, merge_args2cfg
from util.adv_attack import (
    Model01Wrapper,
    iq_to_ta_input, ta_output_to_iq,
    iq_to_ta_input_minmax, ta_output_to_iq_minmax,
)

try:
    import torchattacks
except ImportError:
    sys.exit("torchattacks not installed. Run: pip install torchattacks")


def load_bpsk_snr18(data_path, n_samples=200):
    """Load BPSK samples at SNR=18 from RML2016.10a."""
    with open(data_path, 'rb') as f:
        Xd = pickle.load(f, encoding='bytes')

    key = (b'BPSK', 18)
    if key not in Xd:
        raise KeyError(f"Key {key} not found in dataset")

    data = Xd[key]  # [N, 2, 128]
    n = min(n_samples, data.shape[0])
    rng = np.random.RandomState(42)
    idx = rng.choice(data.shape[0], size=n, replace=False)
    samples = data[idx].astype(np.float32)
    print(f"Loaded {n} BPSK samples @ SNR=18, shape {samples.shape}")
    return samples


def load_model(ckpt_path, device):
    """Load pretrained AWN model."""
    cfg = Config('2016.10a', train=False)
    cfg.device = device
    model = create_model(cfg, 'awn')
    ckpt = os.path.join(ckpt_path, '2016.10a_AWN.pkl')
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    print(f"Loaded AWN from {ckpt}")
    return model


def run_fgsm(model, x_iq, labels, eps, mode, device):
    """Run FGSM attack with unit or minmax normalization.

    Returns adversarial IQ tensor [N, 2, 128].
    """
    wrapped = Model01Wrapper(model).to(device)
    wrapped.eval()
    atk = torchattacks.FGSM(wrapped, eps=eps)

    if mode == 'unit':
        x_ta = iq_to_ta_input(x_iq)
        x_adv_ta = atk(x_ta, labels)
        x_adv_iq = ta_output_to_iq(x_adv_ta)
    elif mode == 'minmax':
        x_ta, a, b = iq_to_ta_input_minmax(x_iq)
        wrapped.set_minmax(a, b)
        try:
            x_adv_ta = atk(x_ta, labels)
            x_adv_iq = ta_output_to_iq_minmax(x_adv_ta, a, b)
        finally:
            wrapped.clear_minmax()
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return x_adv_iq


def iq_to_scatter(x_np):
    """Flatten [N, 2, 128] -> (I_flat, Q_flat)."""
    return x_np[:, 0, :].ravel(), x_np[:, 1, :].ravel()


def compute_avg_spectrum(x_np):
    """Compute average FFT magnitude spectrum across all samples.

    x_np: [N, 2, 128]  (I/Q channels)
    Returns: (freqs, avg_mag_I, avg_mag_Q, avg_mag_combined)
        where freqs are normalized frequencies [0, 0.5].
    """
    N, _, T = x_np.shape
    # Complex IQ signal
    iq = x_np[:, 0, :] + 1j * x_np[:, 1, :]  # [N, 128]
    # FFT per sample
    spectra = np.fft.fft(iq, axis=1)           # [N, 128]
    mag = np.abs(spectra)                       # [N, 128]
    # Average magnitude across samples
    avg_mag = np.mean(mag, axis=0)              # [128]
    # Shift so DC is in the center
    avg_mag_shifted = np.fft.fftshift(avg_mag)
    freqs = np.fft.fftshift(np.fft.fftfreq(T))  # normalized [-0.5, 0.5)

    # Also compute per-channel (I-only, Q-only) spectra
    spec_I = np.abs(np.fft.fft(x_np[:, 0, :], axis=1))
    spec_Q = np.abs(np.fft.fft(x_np[:, 1, :], axis=1))
    avg_I = np.fft.fftshift(np.mean(spec_I, axis=0))
    avg_Q = np.fft.fftshift(np.mean(spec_Q, axis=0))

    return freqs, avg_mag_shifted, avg_I, avg_Q


def compute_perturbation_spectrum(clean_np, adv_np):
    """Compute average FFT magnitude of the perturbation (delta) signal."""
    delta = adv_np - clean_np  # [N, 2, 128]
    delta_iq = delta[:, 0, :] + 1j * delta[:, 1, :]
    spectra = np.fft.fft(delta_iq, axis=1)
    avg_mag = np.fft.fftshift(np.mean(np.abs(spectra), axis=0))
    freqs = np.fft.fftshift(np.fft.fftfreq(delta.shape[2]))
    return freqs, avg_mag


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='./data/RML2016.10a_dict.pkl')
    parser.add_argument('--ckpt', default='./checkpoint')
    parser.add_argument('--eps', type=float, default=0.03)
    parser.add_argument('--n_samples', type=int, default=200)
    parser.add_argument('--out', default='iq_fgsm_unit_vs_minmax.png')
    parser.add_argument('--alpha', type=float, default=0.25)
    parser.add_argument('--point_size', type=float, default=2)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- Load data & model ---
    samples_np = load_bpsk_snr18(args.data, args.n_samples)
    model = load_model(args.ckpt, device)

    x_iq = torch.from_numpy(samples_np).to(device)
    # Get true labels from model
    with torch.no_grad():
        logits, _ = model(x_iq)
        labels = logits.argmax(dim=1)

    # --- Run FGSM with both modes ---
    print(f"\nRunning FGSM (eps={args.eps})...")
    x_adv_unit = run_fgsm(model, x_iq, labels, args.eps, 'unit', device)
    x_adv_minmax = run_fgsm(model, x_iq, labels, args.eps, 'minmax', device)

    # Move to numpy
    clean_np = x_iq.cpu().numpy()
    unit_np = x_adv_unit.detach().cpu().numpy()
    minmax_np = x_adv_minmax.detach().cpu().numpy()

    # --- Compute perturbation stats ---
    delta_unit = unit_np - clean_np
    delta_minmax = minmax_np - clean_np
    sig_amp = np.mean(np.abs(clean_np))
    print(f"\nSignal mean |amplitude|:  {sig_amp:.6f}")
    print(f"Unit   perturbation L2:  {np.sqrt(np.mean(delta_unit**2)):.6f}  "
          f"(mean |delta|={np.mean(np.abs(delta_unit)):.6f})")
    print(f"Minmax perturbation L2:  {np.sqrt(np.mean(delta_minmax**2)):.6f}  "
          f"(mean |delta|={np.mean(np.abs(delta_minmax)):.6f})")

    # --- Check accuracy ---
    with torch.no_grad():
        clean_pred = model(x_iq)[0].argmax(1)
        unit_pred = model(x_adv_unit)[0].argmax(1)
        minmax_pred = model(x_adv_minmax)[0].argmax(1)
    clean_acc = (clean_pred == labels).float().mean().item()
    unit_acc = (unit_pred == labels).float().mean().item()
    minmax_acc = (minmax_pred == labels).float().mean().item()
    print(f"\nAccuracy:  clean={clean_acc:.1%}  unit={unit_acc:.1%}  minmax={minmax_acc:.1%}")

    # --- Compute spectra ---
    freqs_c, mag_c, _, _ = compute_avg_spectrum(clean_np)
    freqs_u, mag_u, _, _ = compute_avg_spectrum(unit_np)
    freqs_m, mag_m, _, _ = compute_avg_spectrum(minmax_np)
    _, delta_mag_u = compute_perturbation_spectrum(clean_np, unit_np)
    _, delta_mag_m = compute_perturbation_spectrum(clean_np, minmax_np)

    # --- IQ scatter data ---
    I_c, Q_c = iq_to_scatter(clean_np)
    I_u, Q_u = iq_to_scatter(unit_np)
    I_m, Q_m = iq_to_scatter(minmax_np)

    # Axis limits from all IQ data (percentile to exclude outliers)
    all_vals = np.concatenate([I_c, Q_c, I_u, Q_u, I_m, Q_m])
    lo = np.percentile(all_vals, 0.5)
    hi = np.percentile(all_vals, 99.5)
    pad = (hi - lo) * 0.05
    lim = (lo - pad, hi + pad)

    # --- 3x3 figure: IQ | Spectrum (signal) | Spectrum (perturbation) ---
    fig, axes = plt.subplots(3, 3, figsize=(14, 12))
    scatter_kw = dict(s=args.point_size, alpha=args.alpha,
                      edgecolors='none', rasterized=True)

    colors = ['#1f77b4', '#d62728', '#2ca02c']
    titles = [
        f'Intact  (acc={clean_acc:.0%})',
        f'FGSM unit  eps={args.eps}  (acc={unit_acc:.0%})',
        f'FGSM minmax  eps={args.eps}  (acc={minmax_acc:.0%})',
    ]
    iq_data = [(I_c, Q_c), (I_u, Q_u), (I_m, Q_m)]
    sig_spectra = [(freqs_c, mag_c), (freqs_u, mag_u), (freqs_m, mag_m)]
    delta_spectra = [None, (freqs_u, delta_mag_u), (freqs_m, delta_mag_m)]

    # Shared y-limit for signal spectra
    sig_ymax = max(mag_c.max(), mag_u.max(), mag_m.max()) * 1.1
    # Shared y-limit for perturbation spectra
    delta_ymax = max(delta_mag_u.max(), delta_mag_m.max()) * 1.1

    for col in range(3):
        # --- Row 0: IQ scatter ---
        ax = axes[0, col]
        I, Q = iq_data[col]
        ax.scatter(I, Q, c=colors[col], **scatter_kw)
        ax.set_title(titles[col], fontsize=10, fontweight='bold')
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_aspect('equal')
        ax.set_xlabel('In-phase (I)', fontsize=9)
        if col == 0:
            ax.set_ylabel('Quadrature (Q)', fontsize=9)
        ax.grid(True, alpha=0.2, linewidth=0.5)
        ax.tick_params(labelsize=8)

        # --- Row 1: Signal spectrum (clean overlaid with adv) ---
        ax = axes[1, col]
        # Always show clean as gray reference
        ax.plot(freqs_c, 20 * np.log10(mag_c + 1e-10),
                color='gray', alpha=0.5, linewidth=1, label='Clean')
        freqs_s, mag_s = sig_spectra[col]
        ax.plot(freqs_s, 20 * np.log10(mag_s + 1e-10),
                color=colors[col], linewidth=1.2,
                label='Intact' if col == 0 else 'Adversarial')
        ax.set_xlim(-0.5, 0.5)
        ax.set_xlabel('Normalized Frequency', fontsize=9)
        if col == 0:
            ax.set_ylabel('Magnitude (dB)', fontsize=9)
        ax.set_title('Signal Spectrum', fontsize=9)
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(True, alpha=0.2, linewidth=0.5)
        ax.tick_params(labelsize=8)

        # --- Row 2: Perturbation spectrum ---
        ax = axes[2, col]
        if delta_spectra[col] is not None:
            freqs_d, mag_d = delta_spectra[col]
            ax.plot(freqs_d, 20 * np.log10(mag_d + 1e-10),
                    color=colors[col], linewidth=1.2)
            ax.set_title('Perturbation Spectrum', fontsize=9)
        else:
            ax.text(0.5, 0.5, 'No perturbation\n(clean signal)',
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=10, color='gray')
            ax.set_title('Perturbation Spectrum', fontsize=9)
        ax.set_xlim(-0.5, 0.5)
        ax.set_xlabel('Normalized Frequency', fontsize=9)
        if col == 0:
            ax.set_ylabel('Magnitude (dB)', fontsize=9)
        ax.grid(True, alpha=0.2, linewidth=0.5)
        ax.tick_params(labelsize=8)

    # Align y-limits across rows
    # Row 1: signal spectra share limits
    all_sig_db = [20 * np.log10(m + 1e-10) for _, m in sig_spectra]
    sig_db_min = min(s.min() for s in all_sig_db)
    sig_db_max = max(s.max() for s in all_sig_db)
    sig_pad = (sig_db_max - sig_db_min) * 0.05
    for col in range(3):
        axes[1, col].set_ylim(sig_db_min - sig_pad, sig_db_max + sig_pad)

    # Row 2: perturbation spectra share limits
    all_delta_db = [20 * np.log10(delta_spectra[c][1] + 1e-10)
                    for c in [1, 2]]
    delta_db_min = min(s.min() for s in all_delta_db)
    delta_db_max = max(s.max() for s in all_delta_db)
    delta_pad = (delta_db_max - delta_db_min) * 0.05
    for col in range(3):
        axes[2, col].set_ylim(delta_db_min - delta_pad, delta_db_max + delta_pad)

    fig.suptitle('BPSK @ SNR=18 — FGSM unit vs minmax: IQ + Frequency Domain',
                 fontsize=13, fontweight='bold', y=1.01)
    fig.tight_layout()
    fig.savefig(args.out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved: {args.out}")


if __name__ == '__main__':
    main()
