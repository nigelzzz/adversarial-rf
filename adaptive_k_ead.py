#!/usr/bin/env python3
"""
Adaptive-K defense evaluation for EADL1 and EADEN attacks.
Uses the winning approach: sorted magnitude knee (5% threshold).
"""

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
    Model01Wrapper, iq_to_ta_input_minmax, ta_output_to_iq_minmax,
)
import torchattacks


# ── Helpers ────────────────────────────────────────────────────────────────

@torch.no_grad()
def batch_accuracy(model, x, y, bs=256):
    model.eval()
    correct = 0
    for i in range(0, len(y), bs):
        lo, _ = model(x[i:i+bs])
        correct += (lo.argmax(1) == y[i:i+bs]).sum().item()
    return correct / len(y)


@torch.no_grad()
def per_mod_accuracy(model, x, y, idx_to_mod, bs=256):
    model.eval()
    preds = []
    for i in range(0, len(y), bs):
        lo, _ = model(x[i:i+bs])
        preds.append(lo.argmax(1))
    preds = torch.cat(preds)
    correct = (preds == y)
    result = {}
    for idx, name in idx_to_mod.items():
        mask = (y == idx)
        if mask.sum() > 0:
            result[name] = (correct[mask].float().mean().item(), int(mask.sum().item()))
    return result


def compute_psd(x):
    X = torch.fft.fft(x, dim=2)
    psd = (X.abs() ** 2).mean(dim=1)
    return psd


def sorted_magnitude_knee(psd, ratio_thresh=0.05):
    sorted_mag, _ = psd.sqrt().sort(dim=1, descending=True)
    peak = sorted_mag[:, 0:1].clamp(min=1e-12)
    ratio = sorted_mag / peak
    below = ratio < ratio_thresh
    knee = below.float().argmax(dim=1)
    never = ~below.any(dim=1)
    knee[never] = psd.shape[1]
    knee = knee.clamp(min=1)
    return knee


def apply_adaptive_k(x, selected_k):
    k_candidates = sorted(set(selected_k.cpu().tolist()))
    result = x.clone()
    for k in k_candidates:
        mask = (selected_k == k)
        if mask.any():
            result[mask] = fft_topk_denoise(x[mask], topk=int(k))
    return result


def generate_attack(attack_name, model, x, y, device, bs=64):
    """Generate adversarial examples for EADL1 or EADEN with minmax."""
    wrapped = Model01Wrapper(model).to(device)
    if attack_name == 'eadl1':
        attack = torchattacks.EADL1(wrapped, kappa=1, lr=0.01,
                                     max_iterations=200, binary_search_steps=1,
                                     initial_const=10.0, beta=0.01)
    elif attack_name == 'eaden':
        attack = torchattacks.EADEN(wrapped, kappa=1, lr=0.01,
                                     max_iterations=200, binary_search_steps=1,
                                     initial_const=10.0, beta=0.01)
    else:
        raise ValueError(f"Unknown attack: {attack_name}")

    chunks = []
    n = len(y)
    t0 = time.time()
    for i in range(0, n, bs):
        xb, yb = x[i:i+bs], y[i:i+bs]
        x_ta, a, b = iq_to_ta_input_minmax(xb)
        wrapped.set_minmax(a, b)
        try:
            adv_ta = attack(x_ta, yb)
            adv_iq = ta_output_to_iq_minmax(adv_ta, a, b).detach()
        finally:
            wrapped.clear_minmax()
        chunks.append(adv_iq)
        if (i // bs) % 20 == 0:
            elapsed = time.time() - t0
            print(f"  {attack_name} batch {i//bs+1}/{(n+bs-1)//bs} ({elapsed:.0f}s)")
    return torch.cat(chunks)


def oracle_per_sample_best_k(model, x_adv, y, k_values, bs=256):
    """Find the per-sample best K (upper bound)."""
    n = len(y)
    best_k = torch.zeros(n, dtype=torch.long, device=x_adv.device)
    best_correct = torch.zeros(n, dtype=torch.bool, device=x_adv.device)
    for k in k_values:
        x_def = fft_topk_denoise(x_adv, topk=k)
        preds = []
        for i in range(0, n, bs):
            lo, _ = model(x_def[i:i+bs])
            preds.append(lo.argmax(1))
        preds = torch.cat(preds)
        correct_k = (preds == y)
        improve = correct_k & ~best_correct
        best_k[improve] = k
        best_correct = best_correct | correct_k
    return best_correct.float().mean().item(), best_k


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=2022)
    parser.add_argument('--ckpt_path', type=str, default='./checkpoint')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--eval_limit', type=int, default=None,
                        help='Limit samples for faster eval')
    args = parser.parse_args()

    fix_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Load model + data
    cfg = Config('2016.10a', train=False)
    cfg.init_dir('ead_adaptive_k')
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
    print(f"Samples (SNR>=0): {n}, Device: {device}")

    mod_order = ['BPSK', 'QPSK', '8PSK', 'QAM16', 'QAM64',
                 'PAM4', 'CPFSK', 'GFSK', 'AM-DSB', 'AM-SSB', 'WBFM']

    # Clean accuracy
    clean_acc = batch_accuracy(model, sig_test, lab_test)
    clean_pm = per_mod_accuracy(model, sig_test, lab_test, idx_to_mod)
    print(f"\nClean accuracy: {clean_acc*100:.2f}%")

    k_values = [2, 3, 5, 8, 10, 15, 20, 30, 50, 64]

    # ── Run for each attack ────────────────────────────────────────────────
    for attack_name in ['eadl1', 'eaden']:
        print(f"\n{'='*80}")
        print(f"  {attack_name.upper()} Attack + Adaptive-K Defense")
        print(f"{'='*80}")

        # Generate adversarial
        t0 = time.time()
        x_adv = generate_attack(attack_name, model, sig_test, lab_test,
                                device, bs=args.batch_size)
        print(f"  Attack generation: {time.time()-t0:.1f}s")

        # Attack accuracy (no defense)
        atk_acc = batch_accuracy(model, x_adv, lab_test)
        atk_pm = per_mod_accuracy(model, x_adv, lab_test, idx_to_mod)
        print(f"\n  {attack_name.upper()} accuracy (no defense): {atk_acc*100:.2f}%")

        # Fixed K sweep
        print(f"\n  Fixed K sweep:")
        fixed_results = {}
        for k in k_values:
            x_def = fft_topk_denoise(x_adv, topk=k)
            acc = batch_accuracy(model, x_def, lab_test)
            fixed_results[k] = acc
            print(f"    K={k:<3d}: {acc*100:.2f}%")

        best_fixed_k = max(fixed_results, key=fixed_results.get)
        print(f"  Best fixed K={best_fixed_k}: {fixed_results[best_fixed_k]*100:.2f}%")

        # Adaptive-K (magnitude knee 5%)
        psd = compute_psd(x_adv)
        selected_k = sorted_magnitude_knee(psd, ratio_thresh=0.05)
        x_adaptive = apply_adaptive_k(x_adv, selected_k)
        adaptive_acc = batch_accuracy(model, x_adaptive, lab_test)
        adaptive_pm = per_mod_accuracy(model, x_adaptive, lab_test, idx_to_mod)
        avg_k = selected_k.float().mean().item()
        print(f"\n  Adaptive-K (knee 5%): {adaptive_acc*100:.2f}% (avg K={avg_k:.1f})")

        # Oracle
        oracle_acc, oracle_k = oracle_per_sample_best_k(
            model, x_adv, lab_test, k_values)
        print(f"  Oracle: {oracle_acc*100:.2f}%")

        # ── Per-modulation table ───────────────────────────────────────────
        print(f"\n  {'Mod':<8s} {'Clean':>7s} {attack_name.upper():>7s} {'AdaptK':>7s} {'Oracle':>7s} {'AvgK':>6s}")
        print(f"  {'-'*50}")

        # compute oracle per-mod
        oracle_def = apply_adaptive_k(x_adv, oracle_k)
        oracle_pm = per_mod_accuracy(model, oracle_def, lab_test, idx_to_mod)

        for m in mod_order:
            ca = clean_pm[m][0]*100 if m in clean_pm else 0
            aa = atk_pm[m][0]*100 if m in atk_pm else 0
            ada = adaptive_pm[m][0]*100 if m in adaptive_pm else 0
            ora = oracle_pm[m][0]*100 if m in oracle_pm else 0

            # avg K for this mod
            midx = None
            for ci, cn in idx_to_mod.items():
                if cn == m:
                    midx = ci
                    break
            mod_mask = (lab_test == midx) if midx is not None else None
            mk = selected_k[mod_mask].float().mean().item() if mod_mask is not None and mod_mask.any() else 0
            print(f"  {m:<8s} {ca:>6.1f}% {aa:>6.1f}% {ada:>6.1f}% {ora:>6.1f}% {mk:>5.0f}")

        # ── Per-modulation K sweep table ───────────────────────────────────
        print(f"\n  {attack_name.upper()} RECOVERY per mod per K:")
        header = f"  {'Mod':<8s} {'Clean':>7s} {'Atk':>6s}"
        for k in k_values:
            header += f" {'K='+str(k):>6s}"
        header += f" {'BestK':>7s}"
        print(header)
        print(f"  {'-'*110}")

        for m in mod_order:
            midx = None
            for ci, cn in idx_to_mod.items():
                if cn == m:
                    midx = ci
                    break
            if midx is None:
                continue
            mod_mask = (lab_test == midx)
            if mod_mask.sum() == 0:
                continue

            xm_adv = x_adv[mod_mask]
            ym = lab_test[mod_mask]
            ca = clean_pm[m][0]*100 if m in clean_pm else 0
            aa = atk_pm[m][0]*100 if m in atk_pm else 0
            row = f"  {m:<8s} {ca:>6.1f}% {aa:>5.1f}%"

            best_k_val, best_k_acc = 0, 0
            for k in k_values:
                xd = fft_topk_denoise(xm_adv, topk=k)
                lo, _ = model(xd)
                racc = (lo.argmax(1) == ym).float().mean().item() * 100
                row += f" {racc:>5.1f}%"
                if racc > best_k_acc:
                    best_k_acc = racc
                    best_k_val = k
            row += f"   K={best_k_val}"
            print(row)

        # ── Per-SNR breakdown ──────────────────────────────────────────────
        snr_tensor = torch.tensor(test_snrs_f, device=device)
        unique_snrs = sorted(set(test_snrs_f.tolist()))

        print(f"\n  Per-SNR summary ({attack_name.upper()}):")
        print(f"  {'SNR':>5s} {'Clean':>7s} {'Atk':>7s} {'AdaptK':>7s} {'BestFixK':>9s}")
        print(f"  {'-'*45}")
        for snr_val in unique_snrs:
            smask = (snr_tensor == snr_val)
            xm = x_adv[smask]
            ym = lab_test[smask]
            xc = sig_test[smask]

            lo_c, _ = model(xc)
            ca = (lo_c.argmax(1) == ym).float().mean().item() * 100
            lo_a, _ = model(xm)
            aa = (lo_a.argmax(1) == ym).float().mean().item() * 100

            # adaptive
            psd_s = compute_psd(xm)
            sk = sorted_magnitude_knee(psd_s, 0.05)
            xd = apply_adaptive_k(xm, sk)
            lo_d, _ = model(xd)
            ada = (lo_d.argmax(1) == ym).float().mean().item() * 100

            # best fixed K for this SNR
            best_k, best_acc = 0, 0
            for k in k_values:
                xdk = fft_topk_denoise(xm, topk=k)
                lo_k, _ = model(xdk)
                acc_k = (lo_k.argmax(1) == ym).float().mean().item() * 100
                if acc_k > best_acc:
                    best_acc = acc_k
                    best_k = k
            print(f"  {int(snr_val):>5d} {ca:>6.1f}% {aa:>6.1f}% {ada:>6.1f}% K={best_k}({best_acc:.1f}%)")

    # ── Summary comparison ─────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"  SUMMARY")
    print(f"{'='*80}")
    print(f"  Clean accuracy (SNR>=0): {clean_acc*100:.2f}%")
    print(f"  See per-attack results above.")


if __name__ == '__main__':
    main()
