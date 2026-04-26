"""Phase 7: Render attack_bench.csv as a grouped bar chart (5 attacks x CPU/GPU).

Usage:
    cd paper/scripts
    python plot_attack_bench_latency.py \
        [--csv ../../inference/2016.10a_<idx>/result/attack_bench.csv] \
        [--out ../latex/figures/attack_bench_latency.pdf]

Defaults:
    --csv: most recent inference/2016.10a_*/result/attack_bench.csv
    --out: paper/latex/figures/attack_bench_latency.pdf
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Local import (script lives in paper/scripts/, ieee_style.py is its sibling).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ieee_style import (
    apply_ieee_style, single_col_fig, IEEE_COL_WIDTH, DEFENSE_COLORS,
)


# Phase 7 paper-fixed attack list and display labels (D-05, D-10).
ATTACK_ORDER = ['fgsm', 'pgd', 'cw', 'eadl1', 'eaden']
ATTACK_DISPLAY = {
    'fgsm':  'FGSM',
    'pgd':   'PGD',
    'cw':    'CW',
    'eadl1': 'EAD-L1',
    'eaden': 'EAD-EN',
}
DEVICE_ORDER = ['cpu', 'cuda']
DEVICE_DISPLAY = {'cpu': 'CPU', 'cuda': 'GPU'}
# Pair gray (CPU) with blue (GPU) — distinguishable in greyscale printing.
DEVICE_COLORS = {
    'cpu':  DEFENSE_COLORS['no_defense'],
    'cuda': DEFENSE_COLORS['adaptive_k'],
}


def _repo_root() -> str:
    """Resolve repo root: paper/scripts/<file> -> paper/scripts -> paper -> repo."""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(os.path.dirname(here))


def _resolve_default_csv() -> str:
    """Return path to the most recent attack_bench.csv under inference/."""
    repo = _repo_root()
    pattern = os.path.join(repo, 'inference', '2016.10a_*', 'result', 'attack_bench.csv')
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(
            "No attack_bench.csv found under inference/2016.10a_*/result/. "
            "Run the bench first:\n"
            "  python main.py --mode attack_bench --dataset 2016.10a "
            "--ckpt_path ./checkpoint"
        )
    # Lexicographic sort matches chronological order because the index
    # auto-increments.
    return paths[-1]


def _load_and_validate(csv_path: str) -> pd.DataFrame:
    """Load the CSV (skipping the env comment line) and validate its shape."""
    df = pd.read_csv(csv_path, comment='#')
    got_attacks = sorted(set(df['attack']))
    expected_attacks = sorted(ATTACK_ORDER)
    if got_attacks != expected_attacks:
        raise ValueError(
            f"CSV attack set mismatch: got {got_attacks} expected {expected_attacks}"
        )
    got_devices = sorted(set(df['device']))
    expected_devices = sorted(DEVICE_ORDER)
    if got_devices != expected_devices:
        raise ValueError(
            f"CSV device set mismatch: got {got_devices} expected {expected_devices}"
        )
    if len(df) == 10:
        pass
    else:
        raise ValueError(f"Expected 10 rows, got {len(df)}")
    return df


def _plot(df: pd.DataFrame, out_path: str) -> None:
    """Render the grouped bar chart and write it to ``out_path``."""
    apply_ieee_style()
    fig, ax = single_col_fig()

    x = np.arange(len(ATTACK_ORDER))
    w = 0.38

    for d in DEVICE_ORDER:
        means = [
            df[(df.attack == a) & (df.device == d)]['mean_ms_per_sample'].iloc[0]
            for a in ATTACK_ORDER
        ]
        stds = [
            df[(df.attack == a) & (df.device == d)]['std_ms_per_sample'].iloc[0]
            for a in ATTACK_ORDER
        ]
        dx = -w / 2 if d == 'cpu' else +w / 2
        ax.bar(
            x + dx, means, width=w, yerr=stds, capsize=2,
            color=DEVICE_COLORS[d], label=DEVICE_DISPLAY[d],
            edgecolor='black', linewidth=0.4,
            error_kw={'linewidth': 0.5},
        )

    ax.set_yscale('log')
    ax.set_xticks(x)
    ax.set_xticklabels([ATTACK_DISPLAY[a] for a in ATTACK_ORDER])
    ax.set_ylabel('Latency (ms/sample)')
    ax.set_xlabel('Attack')
    ax.grid(True, axis='y', which='both', alpha=0.3, linewidth=0.3)
    ax.legend(loc='upper left', frameon=False)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close(fig)
    print(f"Wrote {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render Phase 7 attack_bench.csv as a grouped CPU/GPU bar chart."
    )
    parser.add_argument(
        '--csv', type=str, default=None,
        help='Path to attack_bench.csv (defaults to newest inference/2016.10a_*).',
    )
    parser.add_argument(
        '--out', type=str,
        default='paper/latex/figures/attack_bench_latency.pdf',
        help='Output PDF path (relative paths anchored to repo root).',
    )
    args = parser.parse_args()

    repo = _repo_root()

    csv_path = args.csv if args.csv is not None else _resolve_default_csv()
    if not os.path.isabs(csv_path):
        csv_path = os.path.join(repo, csv_path)

    out_path = args.out
    if not os.path.isabs(out_path):
        out_path = os.path.join(repo, out_path)

    df = _load_and_validate(csv_path)
    _plot(df, out_path)
    return 0


if __name__ == '__main__':
    sys.exit(main())
