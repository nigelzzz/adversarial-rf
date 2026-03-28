#!/usr/bin/env python3
"""
Adaptive-K Experiment: Find the best IFFT top-K strategy to recover CW-attacked signals.

Steps:
  1. Load AWN + RML2016.10a, filter SNR >= 0
  2. Measure clean accuracy (baseline)
  3. Generate CW adversarial examples
  4. Sweep K = 1..128 to build the full recovery curve
  5. Test 6 adaptive-K approaches:
     a) Energy knee (cumulative spectral energy threshold)
     b) Confidence sweep (try K small→large, stop when classifier confident)
     c) Classify-then-filter (predict mod → look up K)
     d) Spectral shape (significant bin count)
     e) Oracle per-sample best K (upper bound)
     f) NEW: Gradient-of-accuracy elbow (pick K at steepest accuracy gain)
  6. Print summary table
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict

# -- project imports --
from data_loader.data_loader import Load_Dataset, Dataset_Split
from util.config import Config, merge_args2cfg
from util.utils import fix_seed, create_model
from util.logger import create_logger
from util.defense import fft_topk_denoise, cumulative_energy_knee, fft_adaptive_topk_denoise
from util.adaptive_defense import (
    confidence_sweep_topk_denoise,
    classify_then_filter_topk_denoise,
    spectral_shape_topk_denoise,
)
from util.adv_attack import Model01Wrapper, iq_to_ta_input, ta_output_to_iq

try:
    import torchattacks
except ImportError:
    print("ERROR: pip install torchattacks")
    sys.exit(1)


# ── helpers ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def batch_accuracy(model, x, y, batch_size=256):
    """Compute accuracy over a dataset in batches."""
    model.eval()
    correct = 0
    n = len(y)
    for i in range(0, n, batch_size):
        xb = x[i:i+batch_size]
        yb = y[i:i+batch_size]
        logits, _ = model(xb)
        correct += (logits.argmax(1) == yb).sum().item()
    return correct / n


@torch.no_grad()
def batch_topk_accuracy(model, x_adv, y, topk, batch_size=256):
    """Apply FFT top-K then compute accuracy."""
    model.eval()
    correct = 0
    n = len(y)
    for i in range(0, n, batch_size):
        xb = fft_topk_denoise(x_adv[i:i+batch_size], topk=topk)
        yb = y[i:i+batch_size]
        logits, _ = model(xb)
        correct += (logits.argmax(1) == yb).sum().item()
    return correct / n


def generate_cw_adversarial(model, x_clean, y, device, batch_size=64,
                            cw_c=10.0, cw_steps=200, cw_lr=0.005):
    """Generate CW adversarial examples using torchattacks."""
    wrapped = Model01Wrapper(model)
    wrapped.to(device)
    attack = torchattacks.CW(wrapped, c=cw_c, steps=cw_steps, lr=cw_lr)

    n = len(y)
    adv_chunks = []
    for i in range(0, n, batch_size):
        xb = x_clean[i:i+batch_size].to(device)
        yb = y[i:i+batch_size].to(device)
        # IQ [-1,1] → [0,1] for torchattacks
        x_ta = iq_to_ta_input(xb)
        x_adv_ta = attack(x_ta, yb)
        x_adv_iq = ta_output_to_iq(x_adv_ta)
        adv_chunks.append(x_adv_iq.detach())
        if (i // batch_size) % 10 == 0:
            print(f"  CW batch {i//batch_size + 1}/{(n + batch_size - 1)//batch_size}")
    return torch.cat(adv_chunks, dim=0)


@torch.no_grad()
def oracle_per_sample_best_k(model, x_adv, y, k_list, batch_size=256):
    """For each sample, find the K that gives correct classification (oracle)."""
    model.eval()
    n = len(y)
    device = x_adv.device
    best_k = torch.full((n,), k_list[-1], dtype=torch.int, device=device)
    oracle_correct = torch.zeros(n, dtype=torch.bool, device=device)

    for k in sorted(k_list):
        for i in range(0, n, batch_size):
            xb = fft_topk_denoise(x_adv[i:i+batch_size], topk=k)
            yb = y[i:i+batch_size]
            logits, _ = model(xb)
            preds = logits.argmax(1)
            correct_mask = (preds == yb)
            # For samples not yet correct, if this K makes them correct, record it
            batch_idx = torch.arange(i, min(i+batch_size, n), device=device)
            newly_correct = correct_mask & ~oracle_correct[i:i+batch_size]
            best_k[batch_idx[newly_correct]] = k
            oracle_correct[batch_idx[newly_correct]] = True

    acc = oracle_correct.float().mean().item()
    return best_k, acc


@torch.no_grad()
def elbow_k_approach(model, x_adv, y, batch_size=256):
    """
    NEW approach: For each sample, compute accuracy at K=5,10,...,60
    and pick the K where marginal accuracy gain drops below threshold.
    This is a per-sample version looking at confidence gain.
    """
    model.eval()
    n = len(y)
    device = x_adv.device
    k_values = list(range(5, 65, 5))  # 5, 10, 15, ..., 60

    # For each sample, collect confidence of correct class at each K
    conf_matrix = torch.zeros(n, len(k_values), device=device)

    for ki, k in enumerate(k_values):
        for i in range(0, n, batch_size):
            xb = fft_topk_denoise(x_adv[i:i+batch_size], topk=k)
            yb = y[i:i+batch_size]
            logits, _ = model(xb)
            probs = torch.softmax(logits, dim=1)
            # confidence for the TRUE class
            true_conf = probs.gather(1, yb.unsqueeze(1)).squeeze(1)
            conf_matrix[i:i+batch_size, ki] = true_conf

    # For each sample find elbow: K where adding more bins gives < 0.05 gain
    selected_k = torch.full((n,), k_values[-1], dtype=torch.int, device=device)
    for si in range(n):
        confs = conf_matrix[si]
        for ki in range(1, len(k_values)):
            gain = confs[ki] - confs[ki-1]
            if confs[ki] >= 0.5 and gain < 0.03:
                selected_k[si] = k_values[ki]
                break

    # Apply selected K
    result = x_adv.clone()
    for k in k_values:
        mask = (selected_k == k)
        if mask.any():
            result[mask] = fft_topk_denoise(x_adv[mask], topk=k)

    # Compute accuracy
    correct = 0
    for i in range(0, n, batch_size):
        logits, _ = model(result[i:i+batch_size])
        correct += (logits.argmax(1) == y[i:i+batch_size]).sum().item()
    acc = correct / n

    return result, selected_k, acc


# ── main ────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--ckpt_path', type=str, default='./checkpoint')
    parser.add_argument('--seed', type=int, default=2022)
    parser.add_argument('--eval_limit', type=int, default=None,
                        help='Limit test samples (None = all)')
    parser.add_argument('--cw_steps', type=int, default=200)
    parser.add_argument('--cw_c', type=float, default=10.0)
    parser.add_argument('--cw_lr', type=float, default=0.005)
    parser.add_argument('--cw_batch', type=int, default=64)
    args = parser.parse_args()

    fix_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ── 1. Load model & data ────────────────────────────────────────────
    print("\n=== Loading model and dataset ===")
    cfg = Config('2016.10a', train=False)
    cfg.init_dir('adaptive_k_exp')
    cfg.device = device

    logger = create_logger(os.path.join(cfg.log_dir, 'log.txt'))

    model = create_model(cfg, 'awn')
    ckpt = os.path.join(args.ckpt_path, '2016.10a_AWN.pkl')
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.to(device)
    model.eval()
    print(f"Loaded checkpoint: {ckpt}")

    Signals, Labels, SNRs, snrs, mods = Load_Dataset('2016.10a', logger)
    _, test_set, _, test_idx = Dataset_Split(Signals, Labels, snrs, mods, logger)
    sig_test, lab_test = test_set

    # Filter SNR >= 0
    test_snrs = np.array([SNRs[i] for i in test_idx])
    snr_mask = test_snrs >= 0
    sig_test = sig_test[snr_mask]
    lab_test = lab_test[snr_mask]
    test_snrs_filtered = test_snrs[snr_mask]
    print(f"Test samples (SNR >= 0): {len(lab_test)}")
    print(f"SNR values: {sorted(set(test_snrs_filtered.tolist()))}")

    if args.eval_limit and args.eval_limit < len(lab_test):
        perm = torch.randperm(len(lab_test))[:args.eval_limit]
        sig_test = sig_test[perm]
        lab_test = lab_test[perm]
        test_snrs_filtered = test_snrs_filtered[perm.numpy()]
        print(f"Limited to {args.eval_limit} samples")

    sig_test = sig_test.to(device)
    lab_test = lab_test.to(device)

    # ── 2. Clean accuracy ───────────────────────────────────────────────
    print("\n=== Clean Accuracy ===")
    t0 = time.time()
    clean_acc = batch_accuracy(model, sig_test, lab_test)
    print(f"Clean accuracy (SNR>=0): {clean_acc*100:.2f}%  ({time.time()-t0:.1f}s)")

    # ── 3. Generate CW adversarial examples ─────────────────────────────
    print(f"\n=== Generating CW adversarial (c={args.cw_c}, steps={args.cw_steps}) ===")
    t0 = time.time()
    x_adv = generate_cw_adversarial(
        model, sig_test, lab_test, device,
        batch_size=args.cw_batch,
        cw_c=args.cw_c, cw_steps=args.cw_steps, cw_lr=args.cw_lr,
    )
    cw_time = time.time() - t0
    print(f"CW generation: {cw_time:.1f}s")

    adv_acc = batch_accuracy(model, x_adv, lab_test)
    print(f"CW attack accuracy: {adv_acc*100:.2f}%")

    # ── 4. Full K sweep (K = 1..128) ────────────────────────────────────
    print("\n=== K Sweep (1..128) ===")
    k_sweep = list(range(1, 129))
    k_accs = {}
    t0 = time.time()
    for k in k_sweep:
        acc = batch_topk_accuracy(model, x_adv, lab_test, topk=k)
        k_accs[k] = acc
        if k <= 10 or k % 10 == 0 or k == 128:
            print(f"  K={k:3d}: {acc*100:.2f}%")
    print(f"K sweep time: {time.time()-t0:.1f}s")

    # Find best fixed K
    best_k = max(k_accs, key=k_accs.get)
    best_k_acc = k_accs[best_k]
    print(f"\nBest fixed K = {best_k}, accuracy = {best_k_acc*100:.2f}%")

    # Also check what clean acc looks like after fixed top-K
    clean_topk_acc = {}
    for k in [best_k, 10, 20, 30, 50]:
        ca = batch_topk_accuracy(model, sig_test, lab_test, topk=k)
        clean_topk_acc[k] = ca
        print(f"  Clean acc after top-{k}: {ca*100:.2f}%")

    # ── 5. Adaptive-K approaches ────────────────────────────────────────
    print("\n=== Adaptive-K Approaches ===")
    results = {}

    k_candidates = [5, 10, 15, 20, 25, 30, 40, 50, 60]

    # --- A: Energy knee ---
    print("\n[A] Energy Knee (cumulative spectral energy)")
    for thresh in [0.80, 0.85, 0.90, 0.95, 0.99]:
        t0 = time.time()
        x_def, sel_k = fft_adaptive_topk_denoise(
            x_adv, threshold=thresh,
            k_candidates=k_candidates, k_min=5, k_max=60,
        )
        acc = batch_accuracy(model, x_def, lab_test)
        k_mean = sel_k.float().mean().item()
        k_std = sel_k.float().std().item()
        elapsed = time.time() - t0
        label = f"energy_knee_{thresh:.2f}"
        results[label] = {'acc': acc, 'k_mean': k_mean, 'k_std': k_std, 'time': elapsed}
        print(f"  thresh={thresh:.2f}: acc={acc*100:.2f}%, K_mean={k_mean:.1f}+-{k_std:.1f} ({elapsed:.1f}s)")

    # --- B: Confidence sweep ---
    print("\n[B] Confidence Sweep (try K small→large)")
    for conf_thresh in [0.5, 0.6, 0.7, 0.8, 0.9]:
        t0 = time.time()
        x_def, sel_k = confidence_sweep_topk_denoise(
            x_adv, model, k_candidates=k_candidates,
            confidence_threshold=conf_thresh,
        )
        acc = batch_accuracy(model, x_def, lab_test)
        k_mean = sel_k.float().mean().item()
        k_std = sel_k.float().std().item()
        elapsed = time.time() - t0
        label = f"confidence_{conf_thresh:.1f}"
        results[label] = {'acc': acc, 'k_mean': k_mean, 'k_std': k_std, 'time': elapsed}
        print(f"  conf={conf_thresh:.1f}: acc={acc*100:.2f}%, K_mean={k_mean:.1f}+-{k_std:.1f} ({elapsed:.1f}s)")

    # --- C: Classify-then-filter ---
    print("\n[C] Classify-Then-Filter (mod-specific K)")
    # Try multiple K maps
    k_maps = {
        'conservative': {
            'QAM64': 10, 'PAM4': 10, 'AM-DSB': 10,
            'GFSK': 15, 'BPSK': 15, 'QPSK': 15, 'CPFSK': 15,
            'QAM16': 15, '8PSK': 15,
            'AM-SSB': 40, 'WBFM': 40,
        },
        'default': {
            'QAM64': 10, 'PAM4': 10, 'AM-DSB': 10,
            'GFSK': 20, 'BPSK': 20, 'QPSK': 20, 'CPFSK': 20,
            'QAM16': 20, '8PSK': 20,
            'AM-SSB': 50, 'WBFM': 50,
        },
        'aggressive': {
            'QAM64': 5, 'PAM4': 5, 'AM-DSB': 5,
            'GFSK': 10, 'BPSK': 10, 'QPSK': 10, 'CPFSK': 10,
            'QAM16': 10, '8PSK': 10,
            'AM-SSB': 30, 'WBFM': 30,
        },
    }
    for name, kmap in k_maps.items():
        t0 = time.time()
        x_def, sel_k = classify_then_filter_topk_denoise(x_adv, model, cfg, mod_k_map=kmap)
        acc = batch_accuracy(model, x_def, lab_test)
        k_mean = sel_k.float().mean().item()
        k_std = sel_k.float().std().item()
        elapsed = time.time() - t0
        label = f"classify_{name}"
        results[label] = {'acc': acc, 'k_mean': k_mean, 'k_std': k_std, 'time': elapsed}
        print(f"  {name}: acc={acc*100:.2f}%, K_mean={k_mean:.1f}+-{k_std:.1f} ({elapsed:.1f}s)")

    # --- D: Spectral shape ---
    print("\n[D] Spectral Shape (significant bin count)")
    for sig_pct in [0.05, 0.10, 0.15, 0.20]:
        t0 = time.time()
        x_def, sel_k = spectral_shape_topk_denoise(
            x_adv, sig_pct=sig_pct,
            k_candidates=k_candidates, k_min=5, k_max=60,
        )
        acc = batch_accuracy(model, x_def, lab_test)
        k_mean = sel_k.float().mean().item()
        k_std = sel_k.float().std().item()
        elapsed = time.time() - t0
        label = f"spectral_{sig_pct:.2f}"
        results[label] = {'acc': acc, 'k_mean': k_mean, 'k_std': k_std, 'time': elapsed}
        print(f"  sig_pct={sig_pct:.2f}: acc={acc*100:.2f}%, K_mean={k_mean:.1f}+-{k_std:.1f} ({elapsed:.1f}s)")

    # --- E: Oracle per-sample best K ---
    print("\n[E] Oracle (per-sample best K)")
    t0 = time.time()
    oracle_k, oracle_acc = oracle_per_sample_best_k(
        model, x_adv, lab_test,
        k_list=list(range(1, 65)),
    )
    k_mean = oracle_k.float().mean().item()
    k_std = oracle_k.float().std().item()
    elapsed = time.time() - t0
    results['oracle'] = {'acc': oracle_acc, 'k_mean': k_mean, 'k_std': k_std, 'time': elapsed}
    print(f"  Oracle: acc={oracle_acc*100:.2f}%, K_mean={k_mean:.1f}+-{k_std:.1f} ({elapsed:.1f}s)")

    # --- F: Elbow-based K ---
    print("\n[F] Confidence Elbow (per-sample confidence plateau)")
    t0 = time.time()
    x_def_elbow, sel_k_elbow, elbow_acc = elbow_k_approach(model, x_adv, lab_test)
    k_mean = sel_k_elbow.float().mean().item()
    k_std = sel_k_elbow.float().std().item()
    elapsed = time.time() - t0
    results['elbow'] = {'acc': elbow_acc, 'k_mean': k_mean, 'k_std': k_std, 'time': elapsed}
    print(f"  Elbow: acc={elbow_acc*100:.2f}%, K_mean={k_mean:.1f}+-{k_std:.1f} ({elapsed:.1f}s)")

    # ── 6. Summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("SUMMARY: Adaptive-K Experiment (RML2016.10a, SNR >= 0, CW attack)")
    print("=" * 80)
    print(f"{'Metric':<35s}  {'Accuracy':>10s}  {'K_mean':>8s}  {'K_std':>7s}  {'Time':>6s}")
    print("-" * 80)
    print(f"{'Clean (no attack)':<35s}  {clean_acc*100:>9.2f}%")
    print(f"{'CW attacked (no defense)':<35s}  {adv_acc*100:>9.2f}%")
    print(f"{'Best fixed K=' + str(best_k):<35s}  {best_k_acc*100:>9.2f}%")
    print("-" * 80)

    # Sort by accuracy
    sorted_results = sorted(results.items(), key=lambda x: x[1]['acc'], reverse=True)
    for label, r in sorted_results:
        print(f"{label:<35s}  {r['acc']*100:>9.2f}%  {r['k_mean']:>7.1f}  {r['k_std']:>7.1f}  {r['time']:>5.1f}s")
    print("=" * 80)

    # ── 7. Save K sweep curve to CSV ────────────────────────────────────
    csv_path = os.path.join(cfg.result_dir, 'k_sweep_curve.csv')
    os.makedirs(cfg.result_dir, exist_ok=True)
    with open(csv_path, 'w') as f:
        f.write("k,accuracy\n")
        for k in sorted(k_accs.keys()):
            f.write(f"{k},{k_accs[k]:.6f}\n")
    print(f"\nK sweep curve saved to: {csv_path}")

    # Save summary
    summary_path = os.path.join(cfg.result_dir, 'adaptive_k_summary.csv')
    with open(summary_path, 'w') as f:
        f.write("approach,accuracy,k_mean,k_std,time_s\n")
        f.write(f"clean,{clean_acc:.6f},,,\n")
        f.write(f"cw_attacked,{adv_acc:.6f},,,\n")
        f.write(f"best_fixed_k{best_k},{best_k_acc:.6f},{best_k},,\n")
        for label, r in sorted_results:
            f.write(f"{label},{r['acc']:.6f},{r['k_mean']:.1f},{r['k_std']:.1f},{r['time']:.1f}\n")
    print(f"Summary saved to: {summary_path}")

    # ── 8. Recommend best approach ──────────────────────────────────────
    # Find best non-oracle approach
    non_oracle = [(l, r) for l, r in sorted_results if l != 'oracle']
    if non_oracle:
        best_label, best_r = non_oracle[0]
        print(f"\n>>> BEST ADAPTIVE APPROACH: {best_label}")
        print(f"    Accuracy: {best_r['acc']*100:.2f}%  (vs oracle {oracle_acc*100:.2f}%)")
        print(f"    Mean K:   {best_r['k_mean']:.1f} +/- {best_r['k_std']:.1f}")
        gap = (oracle_acc - best_r['acc']) * 100
        print(f"    Gap to oracle: {gap:.2f}pp")


if __name__ == '__main__':
    main()
