"""
Per-modulation CW attack evaluation with adaptive-K defense recovery.

Runs CW attack (torchattacks) on the retrained SNR>=0 model, reports
per-modulation accuracy before and after adaptive-K defense.

Supports two normalization modes:
  - paper:  (x + 0.02) / 0.04   (fixed, matches AWN_All.py)
  - minmax: per-sample min-max to [0,1]

Usage:
    # Paper normalization (recommended for submission)
    python eval_cw_per_mod.py --norm paper

    # Minmax normalization
    python eval_cw_per_mod.py --norm minmax

    # Both side by side
    python eval_cw_per_mod.py --norm both

    # Custom checkpoint and sample count
    python eval_cw_per_mod.py --norm paper --ckpt training/2016.10a_7/models/2016.10a_AWN.pkl \
        --samples_per_mod 500 --batch_size 128

    # Save CSV output
    python eval_cw_per_mod.py --norm paper --out_csv results_cw_per_mod.csv
"""

import argparse
import sys

import numpy as np
import torch
import torch.nn as nn

from data_loader.data_loader import Load_Dataset, Dataset_Split
from util.config import Config
from util.utils import fix_seed, create_AWN_model
from util.defense import (
    adaptive_k_defense,
    adaptive_k_v2_defense,
    adaptive_k_v2_snr_defense,
    spectral_gated_defense,
)

try:
    import torchattacks
except ImportError:
    sys.exit("torchattacks not installed. Run: pip install torchattacks")


# ---- Normalization wrappers for torchattacks ----------------------------

class PaperNormWrapper(nn.Module):
    """Maps [0,1] (torchattacks space) -> IQ via  x * 0.04 - 0.02"""
    def __init__(self, base_model):
        super().__init__()
        self.base = base_model

    def forward(self, x):
        iq = x * 0.04 - 0.02
        if iq.dim() == 4:
            iq = iq.squeeze(-1)
        logits, _ = self.base(iq)
        return logits


def run_attack_paper(model, wrapper, atk, xb, yb):
    """Attack + recover using paper normalization."""
    x_01 = (xb + 0.02) / 0.04
    x_adv_01 = atk(x_01, yb)
    x_adv = x_adv_01 * 0.04 - 0.02
    return x_adv


def run_attack_minmax(model, wrapper, atk, xb, yb):
    """Attack + recover using per-sample minmax normalization."""
    from util.adv_attack import iq_to_ta_input_minmax, ta_output_to_iq_minmax
    x_01, a, b = iq_to_ta_input_minmax(xb)
    wrapper.set_minmax(a, b)
    try:
        x_adv_01 = atk(x_01, yb)
        x_adv = ta_output_to_iq_minmax(x_adv_01, a, b)
    finally:
        wrapper.clear_minmax()
    return x_adv


# ---- Main evaluation loop -----------------------------------------------

def evaluate(model, X_sub, Y_sub, norm_mode, cw_c, cw_kappa, cw_steps, cw_lr, bs):
    """
    Returns dict with keys 'raw' and 'adaptive_k', each mapping to
    an array of predictions [N].
    """
    if norm_mode == 'paper':
        wrapper = PaperNormWrapper(model).to(next(model.parameters()).device)
        atk = torchattacks.CW(wrapper, c=cw_c, kappa=cw_kappa, steps=cw_steps, lr=cw_lr)
        attack_fn = lambda xb, yb: run_attack_paper(model, wrapper, atk, xb, yb)
    else:
        from util.adv_attack import Model01Wrapper
        wrapper = Model01Wrapper(model)
        atk = torchattacks.CW(wrapper, c=cw_c, kappa=cw_kappa, steps=cw_steps, lr=cw_lr)
        attack_fn = lambda xb, yb: run_attack_minmax(model, wrapper, atk, xb, yb)

    defense_fns = {
        'adaptive_k':     adaptive_k_defense,
        'spectral_gated': spectral_gated_defense,
        'v2_fixed':       lambda x: adaptive_k_v2_defense(x, k_max=20),
        'v2_snr':         adaptive_k_v2_snr_defense,
    }

    preds = {name: [] for name in ['raw'] + list(defense_fns.keys())}

    for i in range(0, len(X_sub), bs):
        xb = X_sub[i:i+bs].to(next(model.parameters()).device)
        yb = torch.tensor(Y_sub[i:i+bs], dtype=torch.long).to(xb.device)

        x_adv = attack_fn(xb, yb)
        model.eval()

        with torch.no_grad():
            logits, _ = model(x_adv)
            preds['raw'].append(logits.argmax(dim=1).cpu().numpy())

        for dname, dfn in defense_fns.items():
            x_rec = dfn(x_adv)
            with torch.no_grad():
                logits, _ = model(x_rec)
                preds[dname].append(logits.argmax(dim=1).cpu().numpy())

        done = min(i + bs, len(X_sub))
        sys.stderr.write(f'\r  [{norm_mode}] {done}/{len(X_sub)}')

    sys.stderr.write('\n')
    return {k: np.concatenate(v) for k, v in preds.items()}


def print_table(Y, results_dict, cls_names):
    """Print a formatted comparison table for each norm mode."""
    modes = list(results_dict.keys())
    # Defense names are all keys except 'raw' in the first mode's results
    defense_names = [k for k in results_dict[modes[0]].keys() if k != 'raw']

    rows = []
    for mode in modes:
        res = results_dict[mode]
        print(f'\n=== Normalization: {mode} ===')
        # Header
        hdr = f'{"Mod":<10} {"no_def":>8}'
        for dname in defense_names:
            label = dname[:10]
            hdr += f' {label:>10}'
        print(hdr)
        print('-' * len(hdr))

        for ci, name in enumerate(cls_names):
            mask = Y == ci
            n = int(mask.sum())
            if n == 0:
                continue
            raw_acc = (res['raw'][mask] == ci).mean() * 100
            line = f'{name:<10} {raw_acc:>7.1f}%'
            row_data = {'norm': mode, 'mod': name, 'n': n, 'no_def': raw_acc}
            for dname in defense_names:
                acc = (res[dname][mask] == ci).mean() * 100
                line += f' {acc:>9.1f}%'
                row_data[dname] = acc
            print(line)
            rows.append(row_data)

        # Overall
        raw_acc = (res['raw'] == Y).mean() * 100
        line = f'{"Overall":<10} {raw_acc:>7.1f}%'
        row_data = {'norm': mode, 'mod': 'Overall', 'n': len(Y), 'no_def': raw_acc}
        for dname in defense_names:
            acc = (res[dname] == Y).mean() * 100
            line += f' {acc:>9.1f}%'
            row_data[dname] = acc
        print('-' * len(hdr))
        print(line)
        rows.append(row_data)

    return rows


def main():
    parser = argparse.ArgumentParser(description='Per-modulation CW attack + adaptive-K evaluation')
    parser.add_argument('--norm', type=str, default='paper', choices=['paper', 'minmax', 'both'],
                        help='Normalization mode (default: paper)')
    parser.add_argument('--ckpt', type=str, default='training/2016.10a_7/models/2016.10a_AWN.pkl',
                        help='Path to model checkpoint')
    parser.add_argument('--dataset', type=str, default='2016.10a')
    parser.add_argument('--snr_min', type=int, default=0, help='Minimum SNR filter')
    parser.add_argument('--samples_per_mod', type=int, default=200,
                        help='Samples per modulation (default: 200)')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=2022)
    parser.add_argument('--cw_c', type=float, default=1.0)
    parser.add_argument('--cw_kappa', type=float, default=1.0)
    parser.add_argument('--cw_steps', type=int, default=100)
    parser.add_argument('--cw_lr', type=float, default=0.001)
    parser.add_argument('--out_csv', type=str, default=None, help='Save results to CSV')
    args = parser.parse_args()

    fix_seed(args.seed)

    import logging
    logger = logging.getLogger('eval_cw')
    logger.setLevel(logging.WARNING)

    cfg = Config(args.dataset, train=False)
    cfg.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    Signals, Labels, SNRs, snrs, mods = Load_Dataset(args.dataset, logger, snr_min=args.snr_min)
    _, _, _, test_idx = Dataset_Split(Signals, Labels, snrs, mods, logger)
    X_test = Signals[test_idx].clone().detach().float()
    Y_test = np.array(Labels[test_idx])

    model = create_AWN_model(cfg)
    model.load_state_dict(torch.load(args.ckpt, map_location=cfg.device))
    model.eval()

    cls_names = ['QAM16', 'QAM64', '8PSK', 'WBFM', 'BPSK', 'CPFSK',
                 'AM-DSB', 'GFSK', 'PAM4', 'QPSK', 'AM-SSB']

    # Subsample
    indices = []
    for ci in range(len(cls_names)):
        mod_idx = np.where(Y_test == ci)[0]
        np.random.seed(42)
        sel = np.random.choice(mod_idx, min(args.samples_per_mod, len(mod_idx)), replace=False)
        indices.append(sel)
    indices = np.concatenate(indices)
    X_sub = X_test[indices]
    Y_sub = Y_test[indices]

    print(f'CW attack: c={args.cw_c}, kappa={args.cw_kappa}, steps={args.cw_steps}, lr={args.cw_lr}')
    print(f'Samples: {len(X_sub)} ({args.samples_per_mod}/mod), SNR >= {args.snr_min}')
    print(f'Checkpoint: {args.ckpt}')
    print()

    # Determine which norms to run
    norms = ['paper', 'minmax'] if args.norm == 'both' else [args.norm]
    results = {}
    for norm in norms:
        results[norm] = evaluate(
            model, X_sub, Y_sub, norm,
            args.cw_c, args.cw_kappa, args.cw_steps, args.cw_lr, args.batch_size,
        )

    print()
    rows = print_table(Y_sub, results, cls_names)

    # Save CSV if requested
    if args.out_csv and rows:
        import csv
        fieldnames = list(rows[0].keys())
        with open(args.out_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f'\nSaved to {args.out_csv}')


if __name__ == '__main__':
    main()
