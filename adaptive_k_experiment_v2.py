#!/usr/bin/env python3
"""
Adaptive-K Experiment v2: Better approaches for CW recovery.

Key insight from v1: CW spreads perturbation across many bins, making
energy-based K selection choose K too large. Best fixed K=4 gives 17%
but clean acc@K=4 is only 32.6%. The oracle shows 35.6% is reachable.

New approaches:
  1. Dual-pass: Apply aggressive K first, classify, then apply mod-specific K
  2. Reconstruction residual: Pick K that minimizes reconstruction anomaly
  3. Cross-validation K: Split signal into halves, pick K with best agreement
  4. Iterative refinement: Start K=4, classify, adjust K based on confidence
  5. Spectral flatness: CW makes spectrum flatter; use flatness to detect attack
     strength and pick K inversely proportional to flatness
  6. Phase coherence: True signal has structured phase; pick K where phase
     coherence is maximized
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict

from data_loader.data_loader import Load_Dataset, Dataset_Split
from util.config import Config, merge_args2cfg
from util.utils import fix_seed, create_model
from util.logger import create_logger
from util.defense import fft_topk_denoise
from util.adv_attack import Model01Wrapper, iq_to_ta_input, ta_output_to_iq

try:
    import torchattacks
except ImportError:
    sys.exit("pip install torchattacks")


# ── helpers ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def batch_accuracy(model, x, y, bs=256):
    model.eval()
    correct = 0
    for i in range(0, len(y), bs):
        logits, _ = model(x[i:i+bs])
        correct += (logits.argmax(1) == y[i:i+bs]).sum().item()
    return correct / len(y)


def generate_cw(model, x, y, device, bs=64, c=10.0, steps=200, lr=0.005):
    wrapped = Model01Wrapper(model).to(device)
    attack = torchattacks.CW(wrapped, c=c, steps=steps, lr=lr)
    chunks = []
    for i in range(0, len(y), bs):
        xb, yb = x[i:i+bs].to(device), y[i:i+bs].to(device)
        x_ta = iq_to_ta_input(xb)
        x_adv = attack(x_ta, yb)
        chunks.append(ta_output_to_iq(x_adv).detach())
        if (i // bs) % 10 == 0:
            print(f"  CW {i//bs+1}/{(len(y)+bs-1)//bs}")
    return torch.cat(chunks)


# ── new adaptive-K approaches ──────────────────────────────────────────────

@torch.no_grad()
def approach_dual_pass(model, x_adv, y, cfg, bs=256):
    """
    Dual-pass: First apply aggressive K=3 to get rough classification,
    then use class-specific K for the second pass.
    """
    model.eval()
    n = len(y)
    device = x_adv.device

    # Build idx→mod mapping
    idx_to_mod = {}
    for k, v in cfg.classes.items():
        name = k.decode() if isinstance(k, bytes) else str(k)
        idx_to_mod[v] = name

    # Pass 1: aggressive denoise to classify
    preds = []
    for i in range(0, n, bs):
        xb = fft_topk_denoise(x_adv[i:i+bs], topk=3)
        logits, _ = model(xb)
        preds.append(logits.argmax(1))
    preds = torch.cat(preds)

    # Mod-specific K (tuned for CW: very small K works best)
    mod_k = {
        'QAM64': 3, 'PAM4': 3, 'QAM16': 4, '8PSK': 4,
        'BPSK': 3, 'QPSK': 4, 'CPFSK': 5, 'GFSK': 5,
        'AM-DSB': 6, 'AM-SSB': 8, 'WBFM': 10,
    }

    # Pass 2: apply per-class K
    selected_k = torch.full((n,), 4, dtype=torch.int, device=device)
    for i in range(n):
        mod = idx_to_mod.get(int(preds[i].item()), '')
        selected_k[i] = mod_k.get(mod, 4)

    result = x_adv.clone()
    for k in set(mod_k.values()):
        mask = (selected_k == k)
        if mask.any():
            result[mask] = fft_topk_denoise(x_adv[mask], topk=k)

    acc = batch_accuracy(model, result, y)
    return acc, selected_k


@torch.no_grad()
def approach_spectral_flatness(model, x_adv, y, bs=256):
    """
    Spectral flatness detects CW attack strength.
    Clean signals have structured spectra (low flatness).
    CW makes spectrum flatter. Use flatness to pick K:
    - High flatness (strong attack) → small K
    - Low flatness (weak/no attack) → large K
    """
    model.eval()
    n = len(y)
    device = x_adv.device

    # Compute spectral flatness per sample
    X = torch.fft.fft(x_adv, dim=2)
    mag = X.abs()  # [N, 2, T]
    # Geometric mean / Arithmetic mean (in log domain for stability)
    log_mag = torch.log(mag.clamp(min=1e-10))
    geo_mean = log_mag.mean(dim=2).exp()  # [N, 2]
    arith_mean = mag.mean(dim=2)           # [N, 2]
    flatness = (geo_mean / arith_mean.clamp(min=1e-10)).mean(dim=1)  # [N]

    # Map flatness to K: higher flatness → lower K
    # Flatness ~ 0.3-0.5 for attacked, ~0.1-0.3 for clean
    k_values = [3, 4, 5, 6, 8, 10, 15, 20]
    flatness_thresholds = [0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.0]

    selected_k = torch.full((n,), k_values[-1], dtype=torch.int, device=device)
    for k, thresh in zip(k_values, flatness_thresholds):
        mask = flatness >= thresh
        selected_k[mask] = k

    result = x_adv.clone()
    for k in set(k_values):
        mask = (selected_k == k)
        if mask.any():
            result[mask] = fft_topk_denoise(x_adv[mask], topk=k)

    acc = batch_accuracy(model, result, y)
    return acc, selected_k, flatness


@torch.no_grad()
def approach_iterative_refinement(model, x_adv, y, bs=256):
    """
    Start with K=3, get prediction + confidence.
    If confidence low, try K=5, K=8.
    Pick the K that gave highest confidence.
    """
    model.eval()
    n = len(y)
    device = x_adv.device

    k_schedule = [3, 4, 5, 6, 8, 10]
    best_conf = torch.zeros(n, device=device)
    best_k = torch.full((n,), k_schedule[0], dtype=torch.int, device=device)
    best_result = fft_topk_denoise(x_adv, topk=k_schedule[0])

    for k in k_schedule:
        x_def = fft_topk_denoise(x_adv, topk=k)
        for i in range(0, n, bs):
            xb = x_def[i:i+bs]
            logits, _ = model(xb)
            conf = torch.softmax(logits, dim=1).max(dim=1).values
            better = conf > best_conf[i:i+bs]
            best_conf[i:i+bs] = torch.where(better, conf, best_conf[i:i+bs])
            best_k[i:i+bs] = torch.where(better, k, best_k[i:i+bs])
            best_result[i:i+bs] = torch.where(
                better.unsqueeze(1).unsqueeze(2).expand_as(xb),
                xb, best_result[i:i+bs]
            )

    acc = batch_accuracy(model, best_result, y)
    return acc, best_k


@torch.no_grad()
def approach_phase_coherence(model, x_adv, y, bs=256):
    """
    Digital signals have structured phase relationships.
    For each candidate K, compute phase variance of top-K bins.
    Pick K that has most coherent phase (lowest phase variance).
    """
    model.eval()
    n = len(y)
    device = x_adv.device

    k_candidates = [3, 4, 5, 6, 8, 10, 15]

    X = torch.fft.fft(x_adv, dim=2)  # [N, 2, T]
    mags = X.abs()
    # Use atan2 for phase (more compatible than .angle())
    phases = torch.atan2(X.imag, X.real)  # [N, 2, T]

    best_coherence = torch.full((n,), -1.0, device=device)
    selected_k = torch.full((n,), k_candidates[0], dtype=torch.int, device=device)

    for k in k_candidates:
        _, idx = mags.topk(k=k, dim=2)  # [N, 2, k]
        top_phases = phases.gather(2, idx)  # [N, 2, k]

        # Phase coherence: |mean(e^{j*phase})| — 1.0 = perfectly coherent
        cos_p = torch.cos(top_phases)
        sin_p = torch.sin(top_phases)
        coherence = torch.sqrt(cos_p.mean(dim=2)**2 + sin_p.mean(dim=2)**2).mean(dim=1)  # [N]

        better = coherence > best_coherence
        best_coherence = torch.where(better, coherence, best_coherence)
        selected_k = torch.where(better, k, selected_k)

    result = x_adv.clone()
    for k in k_candidates:
        mask = (selected_k == k)
        if mask.any():
            result[mask] = fft_topk_denoise(x_adv[mask], topk=k)

    acc = batch_accuracy(model, result, y)
    return acc, selected_k


@torch.no_grad()
def approach_cross_validate(model, x_adv, y, bs=256):
    """
    Split each signal into two halves (even/odd time samples).
    For each K, apply top-K to both halves independently.
    Pick K where the two halves' predictions agree most.
    """
    model.eval()
    n = len(y)
    device = x_adv.device
    T = x_adv.shape[2]  # 128

    # Can't easily split and keep model input shape, so use a different approach:
    # Apply top-K, then check prediction stability across small K perturbations.
    # K with highest prediction agreement between K-1 and K+1 is most stable.
    k_candidates = [3, 4, 5, 6, 8, 10]

    # Get predictions at each K
    pred_at_k = {}
    conf_at_k = {}
    for k in k_candidates:
        x_def = fft_topk_denoise(x_adv, topk=k)
        preds = []
        confs = []
        for i in range(0, n, bs):
            logits, _ = model(x_def[i:i+bs])
            probs = torch.softmax(logits, dim=1)
            preds.append(logits.argmax(1))
            confs.append(probs.max(1).values)
        pred_at_k[k] = torch.cat(preds)
        conf_at_k[k] = torch.cat(confs)

    # For each sample, pick K where prediction matches neighbors AND has good confidence
    selected_k = torch.full((n,), 4, dtype=torch.int, device=device)
    best_score = torch.zeros(n, device=device)

    for i, k in enumerate(k_candidates):
        # Stability: agree with at least one neighbor
        agree_count = torch.zeros(n, device=device)
        if i > 0:
            agree_count += (pred_at_k[k] == pred_at_k[k_candidates[i-1]]).float()
        if i < len(k_candidates) - 1:
            agree_count += (pred_at_k[k] == pred_at_k[k_candidates[i+1]]).float()
        # Score = stability * confidence
        score = agree_count * conf_at_k[k]
        better = score > best_score
        best_score = torch.where(better, score, best_score)
        selected_k = torch.where(better, k, selected_k)

    result = x_adv.clone()
    for k in k_candidates:
        mask = (selected_k == k)
        if mask.any():
            result[mask] = fft_topk_denoise(x_adv[mask], topk=k)

    acc = batch_accuracy(model, result, y)
    return acc, selected_k


@torch.no_grad()
def approach_energy_ratio(model, x_adv, y, bs=256):
    """
    Key insight: In CW attack, the perturbation adds energy to ALL bins.
    Compare energy in top-K bins vs remaining bins.
    For clean signals, top-K should capture most energy.
    For attacked signals, even non-top bins have significant energy.

    Use the energy concentration ratio to estimate attack strength
    and pick K accordingly.
    """
    model.eval()
    n = len(y)
    device = x_adv.device
    T = x_adv.shape[2]

    X = torch.fft.fft(x_adv, dim=2)
    energy = X.abs() ** 2  # [N, 2, T]
    total_energy = energy.sum(dim=2)  # [N, 2]

    # Energy in top-3 bins vs total
    sorted_e, _ = energy.sort(dim=2, descending=True)
    top3_energy = sorted_e[:, :, :3].sum(dim=2)  # [N, 2]
    ratio = (top3_energy / total_energy.clamp(min=1e-10)).mean(dim=1)  # [N]

    # High ratio = most energy in few bins = less attacked or narrowband signal → use small K
    # Low ratio = energy spread = heavily attacked → use very small K (aggressive filtering)
    # Counter-intuitive but correct: CW spreads energy, so we want FEWER bins to remove attack

    k_values = [3, 4, 5, 6, 8, 10, 15]
    # ratio < 0.2: heavily attacked → K=3
    # ratio 0.2-0.4: moderate → K=4-5
    # ratio > 0.4: clean-ish → K=8-15
    thresholds = [0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.0]

    selected_k = torch.full((n,), k_values[0], dtype=torch.int, device=device)
    for k, thresh in zip(k_values, thresholds):
        mask = ratio >= thresh
        selected_k[mask] = k

    result = x_adv.clone()
    for k in set(k_values):
        mask = (selected_k == k)
        if mask.any():
            result[mask] = fft_topk_denoise(x_adv[mask], topk=k)

    acc = batch_accuracy(model, result, y)
    return acc, selected_k, ratio


@torch.no_grad()
def approach_best_of_small_k(model, x_adv, y, bs=256):
    """
    Since K=3-5 are clearly the best range, just try each and pick
    the one with highest classifier confidence per sample.
    Simple and focused.
    """
    model.eval()
    n = len(y)
    device = x_adv.device

    k_candidates = [2, 3, 4, 5, 6]
    best_conf = torch.full((n,), -1.0, device=device)
    selected_k = torch.full((n,), 4, dtype=torch.int, device=device)
    best_result = fft_topk_denoise(x_adv, topk=4)

    for k in k_candidates:
        x_def = fft_topk_denoise(x_adv, topk=k)
        for i in range(0, n, bs):
            xb = x_def[i:i+bs]
            logits, _ = model(xb)
            conf = torch.softmax(logits, dim=1).max(dim=1).values
            better = conf > best_conf[i:i+bs]
            idx = slice(i, min(i+bs, n))
            best_conf[idx] = torch.where(better, conf, best_conf[idx])
            selected_k[idx] = torch.where(better, k, selected_k[idx])
            best_result[idx] = torch.where(
                better.unsqueeze(1).unsqueeze(2).expand_as(xb),
                xb, best_result[idx]
            )

    acc = batch_accuracy(model, best_result, y)
    return acc, selected_k


# ── main ────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--ckpt_path', type=str, default='./checkpoint')
    parser.add_argument('--seed', type=int, default=2022)
    parser.add_argument('--eval_limit', type=int, default=None)
    parser.add_argument('--cw_steps', type=int, default=200)
    parser.add_argument('--cw_c', type=float, default=10.0)
    parser.add_argument('--cw_lr', type=float, default=0.005)
    parser.add_argument('--cw_batch', type=int, default=64)
    args = parser.parse_args()

    fix_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load model & data
    print("\n=== Loading ===")
    cfg = Config('2016.10a', train=False)
    cfg.init_dir('adaptive_k_v2')
    cfg.device = device
    logger = create_logger(os.path.join(cfg.log_dir, 'log.txt'))

    model = create_model(cfg, 'awn')
    ckpt = os.path.join(args.ckpt_path, '2016.10a_AWN.pkl')
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.to(device).eval()

    Signals, Labels, SNRs, snrs, mods = Load_Dataset('2016.10a', logger)
    _, test_set, _, test_idx = Dataset_Split(Signals, Labels, snrs, mods, logger)
    sig_test, lab_test = test_set
    test_snrs = np.array([SNRs[i] for i in test_idx])
    mask = test_snrs >= 0
    sig_test, lab_test = sig_test[mask], lab_test[mask]

    if args.eval_limit and args.eval_limit < len(lab_test):
        perm = torch.randperm(len(lab_test))[:args.eval_limit]
        sig_test, lab_test = sig_test[perm], lab_test[perm]
    sig_test, lab_test = sig_test.to(device), lab_test.to(device)
    print(f"Samples: {len(lab_test)}")

    # Baselines
    clean_acc = batch_accuracy(model, sig_test, lab_test)
    print(f"Clean acc: {clean_acc*100:.2f}%")

    # Also compute clean acc at small K values (to measure collateral damage)
    print("\nClean accuracy at various K:")
    for k in [2, 3, 4, 5, 6, 8, 10, 15, 20]:
        ca = batch_accuracy(model, fft_topk_denoise(sig_test, topk=k), lab_test)
        print(f"  Clean K={k}: {ca*100:.2f}%")

    # CW attack
    print(f"\n=== CW attack (c={args.cw_c}, steps={args.cw_steps}) ===")
    x_adv = generate_cw(model, sig_test, lab_test, device,
                        bs=args.cw_batch, c=args.cw_c,
                        steps=args.cw_steps, lr=args.cw_lr)
    adv_acc = batch_accuracy(model, x_adv, lab_test)
    print(f"CW acc: {adv_acc*100:.2f}%")

    # Fixed K baselines
    print("\n=== Fixed K recovery ===")
    for k in [2, 3, 4, 5, 6, 8, 10, 15, 20]:
        acc = batch_accuracy(model, fft_topk_denoise(x_adv, topk=k), lab_test)
        print(f"  K={k}: {acc*100:.2f}%")

    # ── New approaches ──────────────────────────────────────────────────
    results = {}

    print("\n=== [1] Dual-Pass (K=3 classify → mod-specific K) ===")
    t0 = time.time()
    acc, sel_k = approach_dual_pass(model, x_adv, lab_test, cfg)
    results['dual_pass'] = {'acc': acc, 'k_mean': sel_k.float().mean().item(),
                            'k_std': sel_k.float().std().item(), 'time': time.time()-t0}
    print(f"  acc={acc*100:.2f}%, K_mean={sel_k.float().mean():.1f}")

    print("\n=== [2] Spectral Flatness ===")
    t0 = time.time()
    acc, sel_k, flatness = approach_spectral_flatness(model, x_adv, lab_test)
    results['spectral_flatness'] = {'acc': acc, 'k_mean': sel_k.float().mean().item(),
                                    'k_std': sel_k.float().std().item(), 'time': time.time()-t0}
    print(f"  acc={acc*100:.2f}%, K_mean={sel_k.float().mean():.1f}")
    print(f"  flatness: mean={flatness.mean():.3f}, std={flatness.std():.3f}")

    print("\n=== [3] Iterative Refinement (best confidence across K=3..10) ===")
    t0 = time.time()
    acc, sel_k = approach_iterative_refinement(model, x_adv, lab_test)
    results['iterative_refine'] = {'acc': acc, 'k_mean': sel_k.float().mean().item(),
                                   'k_std': sel_k.float().std().item(), 'time': time.time()-t0}
    print(f"  acc={acc*100:.2f}%, K_mean={sel_k.float().mean():.1f}")

    print("\n=== [4] Phase Coherence ===")
    t0 = time.time()
    acc, sel_k = approach_phase_coherence(model, x_adv, lab_test)
    results['phase_coherence'] = {'acc': acc, 'k_mean': sel_k.float().mean().item(),
                                  'k_std': sel_k.float().std().item(), 'time': time.time()-t0}
    print(f"  acc={acc*100:.2f}%, K_mean={sel_k.float().mean():.1f}")

    print("\n=== [5] Prediction Stability (cross-K agreement) ===")
    t0 = time.time()
    acc, sel_k = approach_cross_validate(model, x_adv, lab_test)
    results['pred_stability'] = {'acc': acc, 'k_mean': sel_k.float().mean().item(),
                                 'k_std': sel_k.float().std().item(), 'time': time.time()-t0}
    print(f"  acc={acc*100:.2f}%, K_mean={sel_k.float().mean():.1f}")

    print("\n=== [6] Energy Ratio ===")
    t0 = time.time()
    acc, sel_k, ratio = approach_energy_ratio(model, x_adv, lab_test)
    results['energy_ratio'] = {'acc': acc, 'k_mean': sel_k.float().mean().item(),
                               'k_std': sel_k.float().std().item(), 'time': time.time()-t0}
    print(f"  acc={acc*100:.2f}%, K_mean={sel_k.float().mean():.1f}")
    print(f"  energy ratio: mean={ratio.mean():.3f}, std={ratio.std():.3f}")

    print("\n=== [7] Best-of-Small-K (K=2..6, pick highest confidence) ===")
    t0 = time.time()
    acc, sel_k = approach_best_of_small_k(model, x_adv, lab_test)
    results['best_small_k'] = {'acc': acc, 'k_mean': sel_k.float().mean().item(),
                               'k_std': sel_k.float().std().item(), 'time': time.time()-t0}
    print(f"  acc={acc*100:.2f}%, K_mean={sel_k.float().mean():.1f}")

    # ── Summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("SUMMARY v2: Adaptive-K for CW Recovery (RML2016.10a, SNR >= 0)")
    print("=" * 80)
    print(f"  Clean accuracy:        {clean_acc*100:.2f}%")
    print(f"  CW attacked:           {adv_acc*100:.2f}%")
    # Print fixed K reference
    for k in [3, 4, 5]:
        acc_k = batch_accuracy(model, fft_topk_denoise(x_adv, topk=k), lab_test)
        print(f"  Fixed K={k}:             {acc_k*100:.2f}%")
    print("-" * 80)
    print(f"{'Approach':<30s}  {'Acc':>8s}  {'K_mean':>7s}  {'K_std':>7s}  {'Time':>6s}")
    print("-" * 80)
    for label, r in sorted(results.items(), key=lambda x: x[1]['acc'], reverse=True):
        print(f"  {label:<28s}  {r['acc']*100:>7.2f}%  {r['k_mean']:>6.1f}  {r['k_std']:>7.1f}  {r['time']:>5.1f}s")
    print("=" * 80)

    # Save
    os.makedirs(cfg.result_dir, exist_ok=True)
    csv_path = os.path.join(cfg.result_dir, 'adaptive_k_v2_summary.csv')
    with open(csv_path, 'w') as f:
        f.write("approach,accuracy,k_mean,k_std,time_s\n")
        for label, r in sorted(results.items(), key=lambda x: x[1]['acc'], reverse=True):
            f.write(f"{label},{r['acc']:.6f},{r['k_mean']:.1f},{r['k_std']:.1f},{r['time']:.1f}\n")
    print(f"\nSaved: {csv_path}")

    # Best approach
    best_label = max(results, key=lambda x: results[x]['acc'])
    best = results[best_label]
    print(f"\n>>> BEST: {best_label} — {best['acc']*100:.2f}%, K_mean={best['k_mean']:.1f}")


if __name__ == '__main__':
    main()
