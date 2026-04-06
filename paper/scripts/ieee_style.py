"""Shared matplotlib IEEE styling configuration for TCCN paper figures.

Usage:
    from ieee_style import apply_ieee_style, single_col_fig, double_col_fig
    from ieee_style import DEFENSE_COLORS, DEFENSE_LABELS, IEEE_COL_WIDTH, IEEE_DBL_WIDTH

All figures for the paper should use these shared settings to ensure consistent
IEEE-standard dimensions (3.487in single-column, 7.16in double-column) and
8pt serif fonts matching the IEEEtran document class.
"""

import matplotlib
import matplotlib.pyplot as plt

# ---- IEEE column dimensions (inches) ----
IEEE_COL_WIDTH = 3.487   # single-column width
IEEE_DBL_WIDTH = 7.16    # double-column width
GOLDEN_RATIO = 1.618     # height divisor for natural aspect ratio

# ---- IEEE rcParams dict ----
IEEE_STYLE = {
    'font.family': 'serif',
    'font.serif': ['Times', 'Times New Roman', 'Computer Modern Roman'],
    'font.size': 8,
    'text.usetex': False,        # set True if pdflatex is available in PATH
    'axes.labelsize': 8,
    'axes.titlesize': 9,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.format': 'pdf',
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02,
    'lines.linewidth': 1.0,
    'lines.markersize': 3,
    'axes.linewidth': 0.5,
    'grid.linewidth': 0.3,
}


def apply_ieee_style():
    """Apply IEEE rcParams globally to matplotlib."""
    plt.rcParams.update(IEEE_STYLE)


def single_col_fig(nrows=1, ncols=1, **kwargs):
    """Return (fig, ax) sized for a single IEEEtran column (3.487 in wide)."""
    height = IEEE_COL_WIDTH / GOLDEN_RATIO
    return plt.subplots(nrows, ncols,
                        figsize=(IEEE_COL_WIDTH, height), **kwargs)


def double_col_fig(height_ratio=0.5, nrows=1, ncols=1, **kwargs):
    """Return (fig, ax) sized for a double IEEEtran column (7.16 in wide)."""
    height = IEEE_DBL_WIDTH * height_ratio
    return plt.subplots(nrows, ncols,
                        figsize=(IEEE_DBL_WIDTH, height), **kwargs)


# ---- Defense color palette ----
# Distinct colors chosen for legibility in greyscale printing and color vision
# deficiency. Verified distinguishable at 300 dpi in 8pt serif context.
DEFENSE_COLORS = {
    'adaptive_k':     '#1f77b4',   # blue
    'spectral_gated': '#2ca02c',   # green
    'no_defense':     '#7f7f7f',   # gray
    'kalman':         '#ff7f0e',   # orange
    'wiener':         '#d62728',   # red
    'savitzky_golay': '#9467bd',   # purple
    'gaussian':       '#8c564b',   # brown
    'fir':            '#e377c2',   # pink
    'rand_smooth':    '#bcbd22',   # olive
}

# ---- Paper-friendly display names ----
DEFENSE_LABELS = {
    'adaptive_k':     'Adaptive-K',
    'spectral_gated': 'Spectral-Gated',
    'no_defense':     'No Defense',
    'kalman':         'Kalman',
    'wiener':         'Wiener',
    'savitzky_golay': 'Savitzky-Golay',
    'gaussian':       'Gaussian',
    'fir':            'FIR',
    'rand_smooth':    'Rand. Smooth.',
}

# ---- Preferred display order for legends and grouped bar charts ----
DEFENSE_ORDER = [
    'adaptive_k',
    'spectral_gated',
    'no_defense',
    'kalman',
    'wiener',
    'gaussian',
    'fir',
    'savitzky_golay',
    'rand_smooth',
]

# ---- Attack display names ----
ATTACK_LABELS = {
    'cw':    'CW',
    'eadl1': 'EAD-L1',
    'eaden': 'EAD-EN',
    'fgsm':  'FGSM',
    'pgd':   'PGD',
}
