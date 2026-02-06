#!/usr/bin/env python
"""
Generate IQ constellation scatter plots from RadioML RML2016.10a data.

Produces 2x2 grids (BPSK / QPSK / QAM16 / QAM64) with configurable
SNR selection, weighted multi-SNR mixing, and optional phase/CFO correction.

A --make_comparison convenience mode outputs two figures in one run:
  iq_snr18.png  (SNR=18 only, clean constellation)
  iq_mixed.png  (SNR 0/6/12/18 weighted mix, fuzzier cloud)
"""

import sys
import pickle
import argparse
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_snr_weights(tokens):
    """Parse weight tokens: either 'snr=w' pairs or a plain list of floats."""
    weights = {}
    try:
        if any('=' in t for t in tokens):
            for t in tokens:
                snr_str, w_str = t.split('=', 1)
                weights[int(snr_str)] = float(w_str)
        else:
            # Plain float list -- caller must zip with --snrs
            weights = [float(t) for t in tokens]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Cannot parse --snr_weights '{' '.join(tokens)}': {exc}"
        )
    return weights


def build_parser():
    p = argparse.ArgumentParser(
        description='Plot IQ constellation distributions from RML2016.10a',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Data
    p.add_argument('--data', type=str, required=True,
                   help='Path to RML2016.10a_dict.pkl')

    # Modulations
    p.add_argument('--mods', nargs='+', default=['BPSK', 'QPSK', 'QAM16', 'QAM64'],
                   help='Modulation types to plot (default: BPSK QPSK QAM16 QAM64)')

    # SNR selection (single-figure mode)
    p.add_argument('--snrs', nargs='+', type=int, default=[18],
                   help='SNR values to use (default: 18)')
    p.add_argument('--snr_weights', nargs='+', default=None,
                   help='Weights per SNR: either snr=w pairs (0=0.5 6=0.25) '
                        'or aligned floats (0.5 0.25 0.15 0.10)')

    # Sampling
    p.add_argument('--n_examples_total_per_mod', type=int, default=200,
                   help='Total frames per modulation (default: 200)')
    p.add_argument('--seed', type=int, default=0,
                   help='Random seed (default: 0)')

    # Output (single-figure mode)
    p.add_argument('--out', type=str, default='iq_distribution.png',
                   help='Output file (default: iq_distribution.png)')

    # Appearance
    p.add_argument('--alpha', type=float, default=0.20,
                   help='Scatter point alpha (default: 0.20)')
    p.add_argument('--point_size', type=float, default=3,
                   help='Scatter point size (default: 3)')
    p.add_argument('--max_points', type=int, default=0,
                   help='Max points per subplot; 0 = no cap (default: 0)')
    p.add_argument('--equalize_axes', type=_str2bool, default=True,
                   help='Same x/y limits across subplots (default: true)')

    # Corrections
    p.add_argument('--phase_correct', action='store_true', default=False,
                   help='Per-frame constant phase derotation')
    p.add_argument('--cfo_correct', action='store_true', default=False,
                   help='Simple CFO compensation per frame')

    # Strictness
    p.add_argument('--strict', action='store_true', default=False,
                   help='Error if requested (mod, SNR) is missing in dataset')

    # Comparison mode
    p.add_argument('--make_comparison', action='store_true', default=False,
                   help='Produce iq_snr18.png and iq_mixed.png in one run')
    p.add_argument('--out_snr18', type=str, default='iq_snr18.png')
    p.add_argument('--out_mixed', type=str, default='iq_mixed.png')
    p.add_argument('--snr18_value', type=int, default=18)
    p.add_argument('--snr18_total', type=int, default=400)
    p.add_argument('--mixed_snrs', nargs='+', type=int, default=[0, 6, 12, 18])
    p.add_argument('--mixed_weights', nargs='+',
                   default=['0=0.5', '6=0.25', '12=0.15', '18=0.10'])
    p.add_argument('--mixed_total', type=int, default=300)

    return p


def _str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', '1'):
        return True
    if v.lower() in ('no', 'false', '0'):
        return False
    raise argparse.ArgumentTypeError(f'Boolean value expected, got {v!r}')


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_dataset(path):
    """Load the RML2016.10a pickle dict. Keys are (mod_bytes, snr_int)."""
    print(f'Loading {path} ...')
    with open(path, 'rb') as f:
        data = pickle.load(f, encoding='bytes')
    # Summarize available keys
    mods = sorted(set(k[0] for k in data))
    snrs = sorted(set(k[1] for k in data))
    print(f'  Modulations: {[m.decode() if isinstance(m, bytes) else m for m in mods]}')
    print(f'  SNRs: {snrs}')
    return data


# ---------------------------------------------------------------------------
# Sampling with largest-remainder allocation
# ---------------------------------------------------------------------------

def largest_remainder_alloc(weights, total):
    """Allocate *total* items among bins proportional to *weights*.

    Uses the largest-remainder method for deterministic, stable rounding
    that guarantees the counts sum to exactly *total*.
    """
    s = sum(weights)
    exact = [w / s * total for w in weights]
    floors = [int(np.floor(e)) for e in exact]
    remainders = [e - f for e, f in zip(exact, floors)]
    deficit = total - sum(floors)
    # Award extra units to bins with the largest remainders
    indices = sorted(range(len(remainders)),
                     key=lambda i: remainders[i], reverse=True)
    for i in indices[:deficit]:
        floors[i] += 1
    return floors


def resolve_weights(snrs, raw_weights):
    """Return a dict {snr: weight} from parsed weight input."""
    if raw_weights is None:
        # Uniform
        return {s: 1.0 / len(snrs) for s in snrs}

    if isinstance(raw_weights, dict):
        # Already snr=weight pairs; keep only relevant SNRs
        out = {s: raw_weights.get(s, 0.0) for s in snrs}
    elif isinstance(raw_weights, list) and len(raw_weights) == len(snrs):
        out = dict(zip(snrs, raw_weights))
    else:
        raise ValueError(
            f'--snr_weights length ({len(raw_weights)}) does not match '
            f'--snrs length ({len(snrs)}). Use snr=weight pairs or '
            f'provide one weight per SNR.'
        )
    total = sum(out.values())
    if total <= 0:
        raise ValueError('Sum of weights must be positive')
    return {s: w / total for s, w in out.items()}


def sample_frames(dataset, mod_str, snrs, weight_dict, total, rng, strict):
    """Sample *total* frames for one modulation across multiple SNRs.

    Returns np.ndarray of shape [total, 2, frame_len].
    """
    mod_bytes = mod_str.encode() if isinstance(mod_str, str) else mod_str

    # Determine usable SNRs
    usable = [s for s in snrs if (mod_bytes, s) in dataset]
    missing = set(snrs) - set(usable)
    if missing:
        msg = f'{mod_str}: SNR(s) {sorted(missing)} not found in dataset'
        if strict:
            raise KeyError(msg)
        else:
            warnings.warn(msg)
    if not usable:
        return None

    # Reweight over usable SNRs only
    weights = [weight_dict.get(s, 0.0) for s in usable]
    wsum = sum(weights)
    if wsum <= 0:
        weights = [1.0] * len(usable)
    counts = largest_remainder_alloc(weights, total)

    frames = []
    for snr, n in zip(usable, counts):
        avail = dataset[(mod_bytes, snr)]
        n_avail = avail.shape[0]
        if n <= 0:
            continue
        if n > n_avail:
            warnings.warn(
                f'{mod_str} @ SNR={snr}: requested {n} frames but only '
                f'{n_avail} available; sampling with replacement'
            )
            idx = rng.choice(n_avail, size=n, replace=True)
        else:
            idx = rng.choice(n_avail, size=n, replace=False)
        frames.append(avail[idx])

    if not frames:
        return None
    return np.concatenate(frames, axis=0)


# ---------------------------------------------------------------------------
# IQ extraction with optional corrections
# ---------------------------------------------------------------------------

def extract_iq(frame):
    """Extract I, Q arrays from a single frame.

    Handles both (2, N) and (N, 2) layouts.
    """
    frame = np.asarray(frame, dtype=np.float64)
    if frame.ndim == 1:
        raise ValueError(f'Frame is 1-D with shape {frame.shape}')
    if frame.shape[0] == 2:
        return frame[0], frame[1]
    if frame.shape[1] == 2:
        return frame[:, 0], frame[:, 1]
    raise ValueError(f'Cannot determine I/Q layout from shape {frame.shape}')


def phase_correct_frame(I, Q):
    """Remove constant phase offset via 4th-power estimator.

    Falls back gracefully if estimation is unstable.
    """
    iq = I + 1j * Q
    # 4th-power works for BPSK/QPSK/QAM; good enough for visualization
    moment = np.mean(iq ** 4)
    if np.abs(moment) < 1e-12:
        return I, Q  # skip if degenerate
    phi = np.angle(moment) / 4.0
    iq_corr = iq * np.exp(-1j * phi)
    return iq_corr.real, iq_corr.imag


def cfo_correct_frame(I, Q):
    """Simple CFO compensation: estimate linear phase slope and derotate.

    Uses consecutive-sample phase differences to estimate a constant
    frequency offset, then removes it.
    """
    iq = I + 1j * Q
    # Phase differences between consecutive samples
    prod = iq[1:] * np.conj(iq[:-1])
    # Filter out near-zero samples to avoid noise
    mag = np.abs(prod)
    mask = mag > np.median(mag) * 0.1
    if mask.sum() < 2:
        return I, Q  # not enough good samples
    avg_rot = np.mean(prod[mask] / mag[mask])
    freq_est = np.angle(avg_rot)  # radians per sample
    if np.abs(freq_est) < 1e-8:
        return I, Q  # negligible CFO
    n = np.arange(len(iq))
    iq_corr = iq * np.exp(-1j * freq_est * n)
    return iq_corr.real, iq_corr.imag


def frames_to_iq(frames, do_phase, do_cfo):
    """Convert array of frames to flat I, Q arrays with optional corrections."""
    I_all, Q_all = [], []
    for k in range(frames.shape[0]):
        I_k, Q_k = extract_iq(frames[k])
        if do_cfo:
            I_k, Q_k = cfo_correct_frame(I_k, Q_k)
        if do_phase:
            I_k, Q_k = phase_correct_frame(I_k, Q_k)
        I_all.append(I_k)
        Q_all.append(Q_k)
    return np.concatenate(I_all), np.concatenate(Q_all)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_grid(mod_iq, mods, alpha, point_size, max_points, equalize_axes,
              rng, caption='(a) Intact Data'):
    """Create a 2x2 constellation scatter grid.

    Parameters
    ----------
    mod_iq : dict  {mod_str: (I_array, Q_array)}
    mods : list of str  (order for subplots)
    caption : str  centered below figure

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(7, 7))

    # Pre-compute global axis limits if equalize_axes is on.
    # Use 99.5th percentile to exclude outliers that would stretch axes.
    global_lim = None
    if equalize_axes:
        all_vals = []
        for mod in mods:
            if mod in mod_iq and mod_iq[mod] is not None:
                I, Q = mod_iq[mod]
                all_vals.append(I)
                all_vals.append(Q)
        if all_vals:
            combined = np.concatenate(all_vals)
            lo = np.percentile(combined, 0.5)
            hi = np.percentile(combined, 99.5)
            span = hi - lo
            pad = span * 0.05
            global_lim = (lo - pad, hi + pad)

    for idx, mod in enumerate(mods):
        row, col = divmod(idx, 2)
        ax = axes[row, col]
        ax.set_title(mod, fontsize=11, fontweight='bold')

        if mod not in mod_iq or mod_iq[mod] is None:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                    transform=ax.transAxes, fontsize=10, color='gray')
            ax.set_xlabel('In-phase (I)', fontsize=9)
            ax.set_ylabel('Quadrature (Q)', fontsize=9)
            continue

        I, Q = mod_iq[mod]

        # Optional subsampling for speed
        if max_points > 0 and len(I) > max_points:
            sel = rng.choice(len(I), size=max_points, replace=False)
            I, Q = I[sel], Q[sel]

        ax.scatter(I, Q, s=point_size, alpha=alpha, c='#1f77b4',
                   edgecolors='none', rasterized=True)

        if global_lim is not None:
            ax.set_xlim(global_lim)
            ax.set_ylim(global_lim)

        ax.set_aspect('equal')
        ax.grid(True, alpha=0.2, linewidth=0.5)
        ax.tick_params(labelsize=8)

        if row == 1:
            ax.set_xlabel('In-phase (I)', fontsize=9)
        if col == 0:
            ax.set_ylabel('Quadrature (Q)', fontsize=9)

    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.text(0.5, 0.005, caption, ha='center', va='bottom', fontsize=10,
             style='italic')
    return fig


def generate_figure(dataset, mods, snrs, weight_dict, total_per_mod,
                    rng, alpha, point_size, max_points, equalize_axes,
                    phase_correct, cfo_correct, strict,
                    caption='(a) Intact Data'):
    """Sample data and produce a 2x2 figure. Returns the Figure object."""
    mod_iq = {}
    for mod in mods:
        frames = sample_frames(dataset, mod, snrs, weight_dict,
                               total_per_mod, rng, strict)
        if frames is None:
            mod_iq[mod] = None
            continue
        I, Q = frames_to_iq(frames, phase_correct, cfo_correct)
        mod_iq[mod] = (I, Q)
        n_pts = len(I)
        print(f'  {mod}: {frames.shape[0]} frames -> {n_pts} points')

    return plot_grid(mod_iq, mods, alpha, point_size, max_points,
                     equalize_axes, rng, caption=caption)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = build_parser()
    args = parser.parse_args()

    dataset = load_dataset(args.data)

    if args.make_comparison:
        # --- SNR=18 figure ---
        print(f'\n--- SNR={args.snr18_value} figure ---')
        rng = np.random.RandomState(args.seed)
        snrs_18 = [args.snr18_value]
        w18 = {args.snr18_value: 1.0}
        fig18 = generate_figure(
            dataset, args.mods, snrs_18, w18, args.snr18_total,
            rng, args.alpha, args.point_size, args.max_points,
            args.equalize_axes, args.phase_correct, args.cfo_correct,
            args.strict, caption='(a) Intact Data',
        )
        fig18.savefig(args.out_snr18, dpi=200, bbox_inches='tight')
        plt.close(fig18)
        print(f'Saved: {args.out_snr18}')

        # --- Mixed-SNR figure ---
        print(f'\n--- Mixed SNR figure ---')
        rng = np.random.RandomState(args.seed)
        w_mixed = parse_snr_weights(args.mixed_weights)
        w_mixed_resolved = resolve_weights(args.mixed_snrs, w_mixed)
        fig_mix = generate_figure(
            dataset, args.mods, args.mixed_snrs, w_mixed_resolved,
            args.mixed_total, rng, args.alpha, args.point_size,
            args.max_points, args.equalize_axes, args.phase_correct,
            args.cfo_correct, args.strict, caption='(a) Intact Data',
        )
        fig_mix.savefig(args.out_mixed, dpi=200, bbox_inches='tight')
        plt.close(fig_mix)
        print(f'Saved: {args.out_mixed}')

    else:
        # --- Single-figure mode ---
        rng = np.random.RandomState(args.seed)
        raw_w = parse_snr_weights(args.snr_weights) if args.snr_weights else None
        weight_dict = resolve_weights(args.snrs, raw_w)
        fig = generate_figure(
            dataset, args.mods, args.snrs, weight_dict,
            args.n_examples_total_per_mod, rng, args.alpha,
            args.point_size, args.max_points, args.equalize_axes,
            args.phase_correct, args.cfo_correct, args.strict,
            caption='(a) Intact Data',
        )
        fig.savefig(args.out, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved: {args.out}')


if __name__ == '__main__':
    main()
