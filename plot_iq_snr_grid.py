#!/usr/bin/env python
"""
Plot IQ distribution grid: SNR (0-18 dB) x Modulation (BPSK, QPSK, QAM16, QAM64)

Shows how constellation clarity improves with increasing SNR.

Usage:
    python plot_iq_snr_grid.py
"""

import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATASET_PATH = './data/RML2016.10a_dict.pkl'
MODULATIONS = [b'BPSK', b'QPSK', b'QAM16', b'QAM64']
SNRS = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]  # 10 SNR levels
N_SAMPLES = 300  # samples per cell
N_POINTS = 8000  # points per subplot

# Load dataset
print(f'Loading {DATASET_PATH} ...')
with open(DATASET_PATH, 'rb') as f:
    Xd = pickle.load(f, encoding='bytes')

# Create figure: rows=SNR, cols=modulation
n_rows = len(SNRS)
n_cols = len(MODULATIONS)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 28))

fig.suptitle('IQ Distribution: SNR (0 → 18 dB) × Modulation',
             fontsize=16, fontweight='bold', y=0.995)

for row_idx, snr in enumerate(SNRS):
    for col_idx, mod in enumerate(MODULATIONS):
        ax = axes[row_idx, col_idx]
        name = mod.decode()

        key = (mod, snr)
        if key not in Xd:
            ax.text(0.5, 0.5, 'N/A', ha='center', va='center', transform=ax.transAxes)
            continue

        data = Xd[key][:N_SAMPLES].copy()  # [N, 2, 128]
        N = data.shape[0]

        all_I, all_Q = [], []
        for k in range(N):
            I_k = data[k, 0, :]
            Q_k = data[k, 1, :]
            iq = I_k + 1j * Q_k

            # Per-signal RMS normalization
            rms = np.sqrt(np.mean(np.abs(iq)**2))
            if rms > 0:
                iq = iq / rms

            # Per-signal phase alignment
            if mod == b'BPSK':
                phase_est = np.angle(np.mean(iq**2)) / 2
            else:
                phase_est = np.angle(np.mean(iq**4)) / 4
            iq = iq * np.exp(-1j * phase_est)

            all_I.append(iq.real)
            all_Q.append(iq.imag)

        I_raw = np.concatenate(all_I)
        Q_raw = np.concatenate(all_Q)

        # Subsample
        if len(I_raw) > N_POINTS:
            sel = np.random.choice(len(I_raw), N_POINTS, replace=False)
            I_raw = I_raw[sel]
            Q_raw = Q_raw[sel]

        # Scale to [-0.02, 0.02]
        scale = 0.015
        I_plot = I_raw * scale
        Q_plot = Q_raw * scale

        ax.scatter(I_plot, Q_plot, s=1.5, alpha=0.5, c='#1f77b4',
                   edgecolors='none', rasterized=True)
        ax.set_xlim(-0.02, 0.02)
        ax.set_ylim(-0.02, 0.02)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.2, linewidth=0.5)
        ax.tick_params(labelsize=7)

        # Row labels (SNR) on left
        if col_idx == 0:
            ax.set_ylabel(f'SNR={snr} dB', fontsize=10, fontweight='bold')

        # Column titles on top
        if row_idx == 0:
            ax.set_title(name, fontsize=13, fontweight='bold')

        # X label only on bottom row
        if row_idx == n_rows - 1:
            ax.set_xlabel('I', fontsize=9)

plt.tight_layout(rect=[0, 0, 1, 0.99])
out = 'iq_snr_grid.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'\nSaved: {out}')
