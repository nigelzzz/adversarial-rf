#!/usr/bin/env python3
"""Per-modulation CW recovery accuracy across K values."""

import os, sys, time
import numpy as np
import torch
from collections import defaultdict

from data_loader.data_loader import Load_Dataset, Dataset_Split
from util.config import Config
from util.utils import fix_seed, create_model
from util.logger import create_logger
from util.defense import fft_topk_denoise
from util.adv_attack import (
    Model01Wrapper, iq_to_ta_input, ta_output_to_iq,
    iq_to_ta_input_minmax, ta_output_to_iq_minmax,
)

import torchattacks


@torch.no_grad()
def per_mod_accuracy(model, x, y, mod_indices, idx_to_mod):
    """Return dict {mod_name: accuracy}."""
    model.eval()
    logits, _ = model(x)
    preds = logits.argmax(1)
    correct = (preds == y)

    result = {}
    for idx, name in idx_to_mod.items():
        mask = (y == idx)
        if mask.sum() > 0:
            result[name] = (correct[mask].float().mean().item(), int(mask.sum().item()))
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--ckpt_path', type=str, default='./checkpoint')
    parser.add_argument('--seed', type=int, default=2022)
    parser.add_argument('--eval_limit', type=int, default=None)
    parser.add_argument('--cw_steps', type=int, default=100)
    parser.add_argument('--cw_c', type=float, default=1.0)
    parser.add_argument('--cw_lr', type=float, default=0.001)
    parser.add_argument('--cw_kappa', type=float, default=1.0)
    parser.add_argument('--cw_batch', type=int, default=64)
    parser.add_argument('--ta_box', type=str, default='minmax')
    args = parser.parse_args()

    fix_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Load
    cfg = Config('2016.10a', train=False)
    cfg.init_dir('per_mod_k')
    cfg.device = device
    logger = create_logger(os.path.join(cfg.log_dir, 'log.txt'))

    model = create_model(cfg, 'awn')
    model.load_state_dict(torch.load(
        os.path.join(args.ckpt_path, '2016.10a_AWN.pkl'), map_location=device))
    model.to(device).eval()

    # idx <-> mod mapping
    idx_to_mod = {}
    for k, v in cfg.classes.items():
        idx_to_mod[v] = k.decode() if isinstance(k, bytes) else str(k)

    Signals, Labels, SNRs, snrs, mods = Load_Dataset('2016.10a', logger)
    _, test_set, _, test_idx = Dataset_Split(Signals, Labels, snrs, mods, logger)
    sig_test, lab_test = test_set
    test_snrs = np.array([SNRs[i] for i in test_idx])

    # Filter SNR >= 0
    mask = test_snrs >= 0
    sig_test = sig_test[mask]
    lab_test = lab_test[mask]
    test_snrs_f = test_snrs[mask]

    if args.eval_limit and args.eval_limit < len(lab_test):
        perm = torch.randperm(len(lab_test))[:args.eval_limit]
        sig_test = sig_test[perm]
        lab_test = lab_test[perm]
        test_snrs_f = test_snrs_f[perm.numpy()]

    sig_test = sig_test.to(device)
    lab_test = lab_test.to(device)
    n = len(lab_test)
    print(f"Samples: {n}, Device: {device}")

    # Modulation order for display
    mod_order = ['BPSK', 'QPSK', '8PSK', 'QAM16', 'QAM64',
                 'PAM4', 'CPFSK', 'GFSK', 'AM-DSB', 'AM-SSB', 'WBFM']

    # ── Clean accuracy per mod ──────────────────────────────────────────
    print("\n=== Clean Accuracy (per mod) ===")
    clean_pm = per_mod_accuracy(model, sig_test, lab_test, None, idx_to_mod)
    for m in mod_order:
        if m in clean_pm:
            acc, cnt = clean_pm[m]
            print(f"  {m:<8s}: {acc*100:6.2f}%  (n={cnt})")

    # ── CW attack ───────────────────────────────────────────────────────
    box = args.ta_box
    print(f"\nGenerating CW (c={args.cw_c}, kappa={args.cw_kappa}, steps={args.cw_steps}, lr={args.cw_lr}, box={box})...")
    wrapped = Model01Wrapper(model).to(device)
    attack = torchattacks.CW(wrapped, c=args.cw_c, kappa=args.cw_kappa,
                              steps=args.cw_steps, lr=args.cw_lr)
    bs = args.cw_batch
    chunks = []
    for i in range(0, n, bs):
        xb = sig_test[i:i+bs]
        yb = lab_test[i:i+bs]
        if box == 'minmax':
            x_ta, a, b = iq_to_ta_input_minmax(xb)
            wrapped.set_minmax(a, b)
            try:
                x_adv_ta = attack(x_ta, yb)
                x_adv_iq = ta_output_to_iq_minmax(x_adv_ta, a, b).detach()
            finally:
                wrapped.clear_minmax()
        else:
            x_adv_iq = ta_output_to_iq(attack(iq_to_ta_input(xb), yb)).detach()
        chunks.append(x_adv_iq)
        if (i // bs) % 10 == 0:
            print(f"  batch {i//bs+1}/{(n+bs-1)//bs}")
    x_adv = torch.cat(chunks)

    print("\n=== CW Attacked (no defense) ===")
    atk_pm = per_mod_accuracy(model, x_adv, lab_test, None, idx_to_mod)
    for m in mod_order:
        if m in atk_pm:
            acc, cnt = atk_pm[m]
            print(f"  {m:<8s}: {acc*100:6.2f}%")

    # ── K sweep per modulation ──────────────────────────────────────────
    k_values = [2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 64]

    # Collect: recovery_acc[mod][k] = accuracy
    recovery_acc = defaultdict(dict)
    clean_topk_acc = defaultdict(dict)

    print("\nSweeping K per modulation...")
    for k in k_values:
        x_def = fft_topk_denoise(x_adv, topk=k)
        x_clean_def = fft_topk_denoise(sig_test, topk=k)

        pm = per_mod_accuracy(model, x_def, lab_test, None, idx_to_mod)
        pm_clean = per_mod_accuracy(model, x_clean_def, lab_test, None, idx_to_mod)

        for m in mod_order:
            if m in pm:
                recovery_acc[m][k] = pm[m][0]
            if m in pm_clean:
                clean_topk_acc[m][k] = pm_clean[m][0]

    # ── Print table: CW recovery per mod per K ──────────────────────────
    print("\n" + "=" * 120)
    print("CW RECOVERY ACCURACY (%) — per modulation, per K")
    print("=" * 120)
    header = f"{'Mod':<8s} {'Clean':>7s} {'CW':>6s}"
    for k in k_values:
        header += f" {'K='+str(k):>6s}"
    header += f" {'BestK':>6s}"
    print(header)
    print("-" * 120)

    overall_best_k = defaultdict(int)
    for m in mod_order:
        if m not in recovery_acc:
            continue
        clean_a = clean_pm[m][0] * 100 if m in clean_pm else 0
        atk_a = atk_pm[m][0] * 100 if m in atk_pm else 0
        row = f"{m:<8s} {clean_a:>6.1f}% {atk_a:>5.1f}%"

        best_k_val = 0
        best_k_acc = 0
        for k in k_values:
            acc = recovery_acc[m].get(k, 0) * 100
            row += f" {acc:>5.1f}%"
            if acc > best_k_acc:
                best_k_acc = acc
                best_k_val = k
        row += f"   K={best_k_val}"
        overall_best_k[m] = best_k_val
        print(row)

    # Overall row
    print("-" * 120)
    BS = 256
    correct_clean = correct_atk = 0
    for i in range(0, n, BS):
        lo, _ = model(sig_test[i:i+BS])
        correct_clean += (lo.argmax(1) == lab_test[i:i+BS]).sum().item()
        lo2, _ = model(x_adv[i:i+BS])
        correct_atk += (lo2.argmax(1) == lab_test[i:i+BS]).sum().item()
    clean_all = correct_clean / n * 100
    atk_all = correct_atk / n * 100
    row = f"{'ALL':<8s} {clean_all:>6.1f}% {atk_all:>5.1f}%"
    best_overall_k = 0
    best_overall_acc = 0
    for k in k_values:
        correct_k = 0
        for i in range(0, n, BS):
            x_def = fft_topk_denoise(x_adv[i:i+BS], topk=k)
            lo, _ = model(x_def)
            correct_k += (lo.argmax(1) == lab_test[i:i+BS]).sum().item()
        acc = correct_k / n * 100
        row += f" {acc:>5.1f}%"
        if acc > best_overall_acc:
            best_overall_acc = acc
            best_overall_k = k
    row += f"   K={best_overall_k}"
    print(row)
    print("=" * 120)

    # ── Print table: Clean acc after top-K (collateral damage) ──────────
    print("\n" + "=" * 120)
    print("CLEAN ACCURACY AFTER TOP-K (%) — collateral damage per modulation")
    print("=" * 120)
    header = f"{'Mod':<8s} {'NoFilt':>7s}"
    for k in k_values:
        header += f" {'K='+str(k):>6s}"
    print(header)
    print("-" * 120)
    for m in mod_order:
        if m not in clean_topk_acc:
            continue
        clean_a = clean_pm[m][0] * 100
        row = f"{m:<8s} {clean_a:>6.1f}%"
        for k in k_values:
            acc = clean_topk_acc[m].get(k, 0) * 100
            row += f" {acc:>5.1f}%"
        print(row)
    print("=" * 120)

    # ── Best K per modulation ───────────────────────────────────────────
    print("\n=== Recommended K per modulation (for CW recovery) ===")
    for m in mod_order:
        if m in overall_best_k:
            k = overall_best_k[m]
            rec = recovery_acc[m].get(k, 0) * 100
            cln = clean_topk_acc[m].get(k, 0) * 100
            print(f"  {m:<8s}: K={k:<3d}  recovery={rec:5.1f}%  clean@K={cln:5.1f}%")

    # ── Per-SNR per-mod breakdown ──────────────────────────────────────
    snr_tensor = torch.tensor(test_snrs_f, device=device)
    unique_snrs = sorted(set(test_snrs_f.tolist()))
    # Pick a few key K values for the SNR table
    k_snr_list = [5, 8, 10, 15, 20, 30, 50]

    print("\n" + "=" * 140)
    print("PER-SNR PER-MOD: CW recovery at selected K values")
    print("=" * 140)
    for snr_val in unique_snrs:
        snr_mask = (snr_tensor == snr_val)
        sig_snr = x_adv[snr_mask]
        lab_snr = lab_test[snr_mask]
        sig_clean_snr = sig_test[snr_mask]

        print(f"\n--- SNR = {int(snr_val)} dB ({snr_mask.sum().item()} samples) ---")
        header = f"  {'Mod':<8s} {'Clean':>7s} {'CW':>6s}"
        for k in k_snr_list:
            header += f" {'K='+str(k):>6s}"
        header += f" {'BestK':>7s}"
        print(header)

        for m in mod_order:
            midx = None
            for ci, cn in idx_to_mod.items():
                if cn == m:
                    midx = ci
                    break
            if midx is None:
                continue
            mod_mask = (lab_snr == midx)
            if mod_mask.sum() == 0:
                continue

            xm_adv = sig_snr[mod_mask]
            xm_clean = sig_clean_snr[mod_mask]
            ym = lab_snr[mod_mask]
            nm = len(ym)

            # Clean
            lo, _ = model(xm_clean)
            ca = (lo.argmax(1) == ym).float().mean().item() * 100
            # CW attacked
            lo2, _ = model(xm_adv)
            aa = (lo2.argmax(1) == ym).float().mean().item() * 100

            row = f"  {m:<8s} {ca:>6.1f}% {aa:>5.1f}%"
            best_k_val, best_k_acc = 0, 0
            for k in k_snr_list:
                xd = fft_topk_denoise(xm_adv, topk=k)
                lo3, _ = model(xd)
                racc = (lo3.argmax(1) == ym).float().mean().item() * 100
                row += f" {racc:>5.1f}%"
                if racc > best_k_acc:
                    best_k_acc = racc
                    best_k_val = k
            row += f"  K={best_k_val}({best_k_acc:.0f}%)"
            print(row)
    print("=" * 140)

    # ── Save CSV ────────────────────────────────────────────────────────
    os.makedirs(cfg.result_dir, exist_ok=True)
    csv_path = os.path.join(cfg.result_dir, 'per_mod_k_recovery.csv')
    with open(csv_path, 'w') as f:
        f.write("modulation,n_samples,clean_acc,cw_acc")
        for k in k_values:
            f.write(f",recovery_k{k},clean_k{k}")
        f.write(",best_k\n")
        for m in mod_order:
            if m not in recovery_acc:
                continue
            cnt = clean_pm[m][1] if m in clean_pm else 0
            ca = clean_pm[m][0] if m in clean_pm else 0
            aa = atk_pm[m][0] if m in atk_pm else 0
            f.write(f"{m},{cnt},{ca:.4f},{aa:.4f}")
            for k in k_values:
                ra = recovery_acc[m].get(k, 0)
                cka = clean_topk_acc[m].get(k, 0)
                f.write(f",{ra:.4f},{cka:.4f}")
            f.write(f",{overall_best_k[m]}\n")
    print(f"\nSaved: {csv_path}")


if __name__ == '__main__':
    main()
