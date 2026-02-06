#!/usr/bin/env python
"""
Plot intact (clean) IQ scatter distributions for BPSK, QPSK, QAM16, QAM64.

Style matching common RML2016.10a visualizations:
  - Raw IQ scatter with per-signal phase alignment
  - Per-signal RMS normalization
  - Axis range [-0.02, 0.02]
  - ~15,000 points per modulation

Expected appearance:
  - BPSK: thin horizontal band (2 clusters on I-axis)
  - QPSK: square-shaped cloud (4 corners)
  - QAM16: fuzzy 4x4 grid structure
  - QAM64: dense fuzzy square

Usage:
    python plot_intact_iq.py
    python plot_intact_iq.py --snr 10 --samples 500
"""

import pickle
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATASET_PATH = './data/RML2016.10a_dict.pkl'
MODULATIONS = [b'BPSK', b'QPSK', b'QAM16', b'QAM64']

parser = argparse.ArgumentParser()
parser.add_argument('--snr', type=int, default=10)
parser.add_argument('--samples', type=int, default=500)
parser.add_argument('--points', type=int, default=15000)
args = parser.parse_args()

# Load dataset
print(f'Loading {DATASET_PATH} ...')
with open(DATASET_PATH, 'rb') as f:
    Xd = pickle.load(f, encoding='bytes')

# --- Figure: 2x2 grid ---
fig, axes = plt.subplots(2, 2, figsize=(6, 6))

for idx, mod in enumerate(MODULATIONS):
    row, col = divmod(idx, 2)
    ax = axes[row, col]
    name = mod.decode()

    key = (mod, args.snr)
    if key not in Xd:
        print(f'  WARNING: {name} @ SNR={args.snr} not found')
        ax.set_title(name)
        continue

    data = Xd[key][:args.samples].copy()  # [N, 2, 128]
    N = data.shape[0]
    print(f'  {name}: {N} samples')

    all_I, all_Q = [], []
    for k in range(N):
        I_k = data[k, 0, :]
        Q_k = data[k, 1, :]
        iq = I_k + 1j * Q_k

        # 1) Per-signal RMS normalization
        rms = np.sqrt(np.mean(np.abs(iq)**2))
        if rms > 0:
            iq = iq / rms

        # 2) Per-signal phase alignment (M-th power method)
        if mod == b'BPSK':
            phase_est = np.angle(np.mean(iq**2)) / 2
        else:
            phase_est = np.angle(np.mean(iq**4)) / 4
        iq = iq * np.exp(-1j * phase_est)

        all_I.append(iq.real)
        all_Q.append(iq.imag)

    # Flatten to raw IQ points
    I_raw = np.concatenate(all_I)
    Q_raw = np.concatenate(all_Q)

    # Subsample to target points
    if len(I_raw) > args.points:
        sel = np.random.choice(len(I_raw), args.points, replace=False)
        I_raw = I_raw[sel]
        Q_raw = Q_raw[sel]

    # Scale to fit axis range [-0.02, 0.02]
    scale = 0.015  # typical RMS-normalized range is ~[-1.5, 1.5]
    I_plot = I_raw * scale
    Q_plot = Q_raw * scale

    ax.scatter(I_plot, Q_plot, s=2, alpha=0.35, c='#1f77b4', edgecolors='none',
               rasterized=True)
    ax.set_title(name, fontsize=11)
    ax.set_xlim(-0.02, 0.02)
    ax.set_ylim(-0.02, 0.02)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2, linewidth=0.5)
    ax.tick_params(labelsize=8)

    # Only show axis labels on edges
    if row == 1:
        ax.set_xlabel('In-phase (I)', fontsize=9)
    if col == 0:
        ax.set_ylabel('Quadrature (Q)', fontsize=9)

plt.tight_layout()
out = 'intact_iq_distribution.png'
fig.savefig(out, dpi=200, bbox_inches='tight')
plt.close(fig)
print(f'\nSaved: {out}')

# === Explanation ===
print("""
Why this matches common RML2016.10a visualizations:
--------------------------------------------------
• Per-signal RMS normalization equalizes power across samples, making
  distributions comparable and fitting within a fixed axis range.

• Per-signal phase alignment (M-th power method) removes random carrier
  phase offset, aligning constellation points to expected positions.

• Raw IQ scatter (all 128 time samples) shows the "cloud" shape because
  of pulse shaping and residual carrier frequency offset within each frame.

• Unlike textbook constellations, we don't apply matched filtering or
  symbol timing recovery—this preserves the inter-symbol transitions
  that create the characteristic fuzzy/cloud appearance.
""")
