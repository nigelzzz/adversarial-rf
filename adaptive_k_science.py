#!/usr/bin/env python3
"""
Adaptive-K Science Experiment: Find the best per-sample K using
frequency-domain features from signal processing theory.

From the per-mod data we know:
  - AM-DSB: K=2  (DC-concentrated)
  - QAM64:  K=10 (narrowband constellation)
  - PAM4:   K=15, GFSK: K=15
  - QAM16:  K=20
  - BPSK:   K=64, QPSK: K=50, CPFSK: K=64, AM-SSB: K=64
  - 8PSK:   K=64

Approaches to test:
  A. Spectral Entropy (Shannon entropy of normalized PSD)
  B. Spectral Kurtosis (peakedness of spectrum)
  C. Energy Containment Bandwidth (min bins for X% energy)
  D. Spectral Roll-off (frequency below which X% energy)
  E. Sorted-Magnitude Knee (elbow on sorted |FFT| curve)
  F. MDL/AIC Model Order (information-theoretic # of signal components)
  G. Eigenvalue Threshold (covariance matrix eigenvalue gap)
  H. Spectral Centroid + Spread → K mapping
  I. Hybrid: Entropy + Bandwidth combined
  J. Two-stage: Feature → classify mod → lookup K (no model needed)
  K. Oracle per-sample best K (upper bound)
"""

import os, sys, time, math
import numpy as np
import torch
import torch.nn as nn
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


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

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
            result[name] = correct[mask].float().mean().item()
    return result


def generate_cw(model, x, y, device, bs=64, c=1.0, kappa=1.0,
                steps=100, lr=0.001, ta_box='minmax'):
    wrapped = Model01Wrapper(model).to(device)
    attack = torchattacks.CW(wrapped, c=c, kappa=kappa, steps=steps, lr=lr)
    chunks = []
    n = len(y)
    for i in range(0, n, bs):
        xb, yb = x[i:i+bs], y[i:i+bs]
        if ta_box == 'minmax':
            x_ta, a, b = iq_to_ta_input_minmax(xb)
            wrapped.set_minmax(a, b)
            try:
                adv_ta = attack(x_ta, yb)
                adv_iq = ta_output_to_iq_minmax(adv_ta, a, b).detach()
            finally:
                wrapped.clear_minmax()
        else:
            adv_iq = ta_output_to_iq(attack(iq_to_ta_input(xb), yb)).detach()
        chunks.append(adv_iq)
        if (i // bs) % 20 == 0:
            print(f"  CW {i//bs+1}/{(n+bs-1)//bs}")
    return torch.cat(chunks)


def apply_adaptive_k(x_adv, selected_k, k_candidates=None):
    """Apply per-sample K from selected_k tensor."""
    if k_candidates is None:
        k_candidates = sorted(set(selected_k.cpu().tolist()))
    result = x_adv.clone()
    for k in k_candidates:
        mask = (selected_k == k)
        if mask.any():
            result[mask] = fft_topk_denoise(x_adv[mask], topk=k)
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Spectral Feature Extraction (all pure signal processing, no model)
# ═══════════════════════════════════════════════════════════════════════════

def compute_psd(x):
    """Compute power spectral density. x: [N,2,T] -> psd: [N,T]"""
    X = torch.fft.fft(x, dim=2)       # [N, 2, T]
    psd = (X.abs() ** 2).mean(dim=1)   # avg over I/Q -> [N, T]
    return psd


def spectral_entropy(psd):
    """Shannon entropy of normalized PSD. Higher = more spread. [N]"""
    p = psd / psd.sum(dim=1, keepdim=True).clamp(min=1e-12)
    entropy = -(p * torch.log2(p.clamp(min=1e-12))).sum(dim=1)
    return entropy  # range: 0 (single bin) to log2(T) (flat)


def spectral_kurtosis(psd):
    """Excess kurtosis of PSD. Higher = more peaked. [N]"""
    mu = psd.mean(dim=1, keepdim=True)
    var = ((psd - mu) ** 2).mean(dim=1, keepdim=True).clamp(min=1e-12)
    kurt = ((psd - mu) ** 4).mean(dim=1) / (var.squeeze(1) ** 2) - 3.0
    return kurt  # high = few peaks, low = flat


def energy_containment_bandwidth(psd, fraction=0.95):
    """Min number of bins to contain `fraction` of total energy. [N]"""
    sorted_psd, _ = psd.sort(dim=1, descending=True)
    cumsum = sorted_psd.cumsum(dim=1)
    total = psd.sum(dim=1, keepdim=True)
    reached = cumsum >= fraction * total
    # argmax finds first True
    k = reached.float().argmax(dim=1) + 1  # [N]
    # handle never-reached
    never = ~reached.any(dim=1)
    k[never] = psd.shape[1]
    return k  # [N] int-like


def spectral_rolloff(psd, fraction=0.85):
    """Bin index below which `fraction` of energy lies (cumulative from bin 0). [N]"""
    cumsum = psd.cumsum(dim=1)
    total = psd.sum(dim=1, keepdim=True)
    reached = cumsum >= fraction * total
    rolloff = reached.float().argmax(dim=1) + 1
    never = ~reached.any(dim=1)
    rolloff[never] = psd.shape[1]
    return rolloff


def sorted_magnitude_knee(psd, ratio_thresh=0.05):
    """
    Elbow detection on sorted |FFT| magnitudes.
    Find first index where mag[i]/mag[0] < ratio_thresh.
    """
    sorted_mag, _ = psd.sqrt().sort(dim=1, descending=True)
    peak = sorted_mag[:, 0:1].clamp(min=1e-12)
    ratio = sorted_mag / peak
    below = ratio < ratio_thresh
    # argmax finds first True
    knee = below.float().argmax(dim=1)
    # if never below threshold, use all bins
    never = ~below.any(dim=1)
    knee[never] = psd.shape[1]
    # knee=0 means even bin 0 is below (shouldn't happen), clamp to 1
    knee = knee.clamp(min=1)
    return knee


def mdl_model_order(x, max_order=30):
    """
    MDL (Minimum Description Length) to estimate number of signal components.
    Based on eigenvalues of the sample covariance matrix.
    x: [N, 2, T]
    Returns: [N] estimated model order (= suggested K)
    """
    N, C, T = x.shape
    device = x.device

    # Build sample covariance from FFT magnitudes
    X = torch.fft.fft(x, dim=2)  # [N, 2, T]
    # Use magnitude spectrum as "observations"
    mag = X.abs()  # [N, 2, T]
    # Reshape to [N, T, 2] for covariance, but T>>2, so use freq-domain cov
    # Instead: treat each sample's PSD as data, compute eigenvalues of
    # the Toeplitz-like structure via sorted PSD values
    psd = (mag ** 2).mean(dim=1)  # [N, T]

    # Sort eigenvalues (PSD values sorted descending ≈ eigenvalues)
    eigvals, _ = psd.sort(dim=1, descending=True)
    eigvals = eigvals[:, :max_order].clamp(min=1e-12)

    # MDL criterion: for each candidate order k
    # MDL(k) = -N*sum(log(eig[k:])) + N*(M-k)*log(arithmetic_mean(eig[k:])/geometric_mean(eig[k:]))
    #        + 0.5*k*(2M-k)*log(N)
    # Simplified: pick k where the eigenvalue ratio drops below a threshold
    M = max_order
    orders = torch.zeros(N, dtype=torch.long, device=device)

    for ni in range(N):
        eig = eigvals[ni]  # [max_order]
        best_mdl = float('inf')
        best_k = 1
        for k in range(1, M):
            noise_eig = eig[k:]
            if len(noise_eig) == 0:
                break
            arith = noise_eig.mean()
            geo = noise_eig.log().mean().exp()
            if arith < 1e-12:
                break
            ratio = geo / arith  # 1.0 = all equal (pure noise), <1 = signal present
            mdl_val = -(M - k) * ratio.log() + 0.5 * k * (2*M - k) * math.log(max(N, 2)) / M
            if mdl_val < best_mdl:
                best_mdl = mdl_val
                best_k = k
        orders[ni] = best_k

    return orders


def eigenvalue_gap(x, max_eig=40):
    """
    Find K from the largest eigenvalue gap in the PSD.
    Sorted PSD values act as approximate eigenvalues.
    The gap between "signal" and "noise" eigenvalues suggests K.
    """
    psd = compute_psd(x)  # [N, T]
    sorted_psd, _ = psd.sort(dim=1, descending=True)
    sorted_psd = sorted_psd[:, :max_eig].clamp(min=1e-12)

    # Compute ratios between consecutive eigenvalues
    ratios = sorted_psd[:, :-1] / sorted_psd[:, 1:].clamp(min=1e-12)  # [N, max_eig-1]

    # The largest ratio indicates the signal/noise boundary
    gap_idx = ratios.argmax(dim=1) + 1  # [N], +1 because we want K (number of signal components)
    return gap_idx


def spectral_centroid_spread(psd):
    """
    Spectral centroid and spread (2nd moment).
    Narrow spread → small K, wide spread → large K.
    """
    T = psd.shape[1]
    freqs = torch.arange(T, device=psd.device, dtype=psd.dtype)
    p = psd / psd.sum(dim=1, keepdim=True).clamp(min=1e-12)

    centroid = (p * freqs.unsqueeze(0)).sum(dim=1)  # [N]
    spread = ((p * (freqs.unsqueeze(0) - centroid.unsqueeze(1)) ** 2).sum(dim=1)).sqrt()  # [N]
    return centroid, spread


# ═══════════════════════════════════════════════════════════════════════════
# Adaptive-K Mapping Functions
# ═══════════════════════════════════════════════════════════════════════════

def feature_to_k_quantile(feature, k_candidates, invert=False):
    """
    Map a continuous feature to K candidates using quantile binning.
    If invert=True, higher feature → lower K.
    """
    N = len(feature)
    n_bins = len(k_candidates)
    k_sorted = sorted(k_candidates, reverse=invert)

    # Rank-based: sort samples by feature, assign K by quantile
    _, sort_idx = feature.sort()
    selected_k = torch.zeros(N, dtype=torch.int, device=feature.device)
    bin_size = N // n_bins
    for i, k in enumerate(k_sorted):
        start = i * bin_size
        end = (i + 1) * bin_size if i < n_bins - 1 else N
        selected_k[sort_idx[start:end]] = k

    return selected_k


def feature_to_k_linear(feature, k_min, k_max, invert=False):
    """
    Linearly map feature range to [k_min, k_max].
    If invert=True, higher feature → lower K.
    """
    fmin = feature.min()
    fmax = feature.max()
    if fmax - fmin < 1e-8:
        return torch.full_like(feature, (k_min + k_max) // 2, dtype=torch.int)
    normalized = (feature - fmin) / (fmax - fmin)  # [0, 1]
    if invert:
        normalized = 1.0 - normalized
    k_float = k_min + normalized * (k_max - k_min)
    return k_float.round().int().clamp(k_min, k_max)


def feature_to_k_thresholds(feature, thresholds, k_values):
    """
    Map feature to K using threshold breakpoints.
    thresholds: ascending list, k_values: corresponding K for each bin.
    len(k_values) = len(thresholds) + 1
    """
    N = len(feature)
    selected_k = torch.full((N,), k_values[-1], dtype=torch.int, device=feature.device)
    for i, thresh in enumerate(thresholds):
        selected_k[feature <= thresh] = k_values[i]
        # only assign to samples not yet assigned a lower threshold
    # re-do properly: ascending thresholds
    selected_k = torch.full((N,), k_values[0], dtype=torch.int, device=feature.device)
    for i, thresh in enumerate(thresholds):
        selected_k[feature > thresh] = k_values[i + 1]
    return selected_k


# ═══════════════════════════════════════════════════════════════════════════
# Oracle
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def oracle_best_k(model, x_adv, y, k_candidates, bs=256):
    """For each sample, try all K and pick the one giving correct prediction
    with highest confidence."""
    model.eval()
    N = len(y)
    device = x_adv.device
    best_k = torch.full((N,), k_candidates[-1], dtype=torch.int, device=device)
    best_conf = torch.full((N,), -1.0, device=device)
    is_correct = torch.zeros(N, dtype=torch.bool, device=device)

    for k in k_candidates:
        for i in range(0, N, bs):
            xb = fft_topk_denoise(x_adv[i:i+bs], topk=k)
            yb = y[i:i+bs]
            lo, _ = model(xb)
            probs = torch.softmax(lo, dim=1)
            pred = lo.argmax(1)
            conf = probs.max(1).values
            correct = (pred == yb)

            idx = slice(i, min(i+bs, N))
            # Prefer: correct with highest confidence
            # If not correct, prefer: highest confidence (might still be useful)
            upgrade = (correct & ~is_correct[idx]) | \
                      (correct & is_correct[idx] & (conf > best_conf[idx])) | \
                      (~correct & ~is_correct[idx] & (conf > best_conf[idx]))

            best_k[idx] = torch.where(upgrade, k, best_k[idx])
            best_conf[idx] = torch.where(upgrade, conf, best_conf[idx])
            is_correct[idx] = is_correct[idx] | correct

    acc = is_correct.float().mean().item()
    return best_k, acc


# ═══════════════════════════════════════════════════════════════════════════
# Main Experiment
# ═══════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--ckpt_path', type=str, default='./checkpoint')
    parser.add_argument('--seed', type=int, default=2022)
    parser.add_argument('--eval_limit', type=int, default=None)
    parser.add_argument('--cw_steps', type=int, default=100)
    parser.add_argument('--cw_c', type=float, default=1.0)
    parser.add_argument('--cw_kappa', type=float, default=1.0)
    parser.add_argument('--cw_lr', type=float, default=0.001)
    parser.add_argument('--ta_box', type=str, default='minmax')
    args = parser.parse_args()

    fix_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # ── Load ────────────────────────────────────────────────────────────
    print("=== Loading ===")
    cfg = Config('2016.10a', train=False)
    cfg.init_dir('adaptive_k_science')
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
    mask = test_snrs >= 0
    sig_test, lab_test = sig_test[mask], lab_test[mask]

    if args.eval_limit and args.eval_limit < len(lab_test):
        perm = torch.randperm(len(lab_test))[:args.eval_limit]
        sig_test, lab_test = sig_test[perm], lab_test[perm]

    sig_test, lab_test = sig_test.to(device), lab_test.to(device)
    N = len(lab_test)
    T = sig_test.shape[2]  # 128
    print(f"Samples: {N}, T={T}, Device: {device}")

    # ── Baselines ───────────────────────────────────────────────────────
    clean_acc = batch_accuracy(model, sig_test, lab_test)
    print(f"\nClean acc: {clean_acc*100:.2f}%")

    # ── CW Attack ───────────────────────────────────────────────────────
    print(f"\nGenerating CW (box={args.ta_box}, c={args.cw_c}, kappa={args.cw_kappa})...")
    t0 = time.time()
    x_adv = generate_cw(model, sig_test, lab_test, device,
                        c=args.cw_c, kappa=args.cw_kappa,
                        steps=args.cw_steps, lr=args.cw_lr, ta_box=args.ta_box)
    print(f"CW time: {time.time()-t0:.0f}s")

    adv_acc = batch_accuracy(model, x_adv, lab_test)
    print(f"CW acc: {adv_acc*100:.2f}%")

    # Fixed-K baselines
    print("\nFixed-K baselines:")
    fixed_k_acc = {}
    for k in [2, 5, 8, 10, 15, 20, 30, 50, 64]:
        acc = batch_accuracy(model, fft_topk_denoise(x_adv, topk=k), lab_test)
        fixed_k_acc[k] = acc
        print(f"  K={k:2d}: {acc*100:.2f}%")
    best_fixed_k = max(fixed_k_acc, key=fixed_k_acc.get)
    print(f"  Best fixed: K={best_fixed_k} → {fixed_k_acc[best_fixed_k]*100:.2f}%")

    # ── Compute all spectral features ───────────────────────────────────
    print("\n=== Computing spectral features ===")
    psd = compute_psd(x_adv)  # [N, T]

    feat = {}
    feat['entropy'] = spectral_entropy(psd)
    feat['kurtosis'] = spectral_kurtosis(psd)
    feat['ecb_90'] = energy_containment_bandwidth(psd, fraction=0.90).float()
    feat['ecb_95'] = energy_containment_bandwidth(psd, fraction=0.95).float()
    feat['ecb_99'] = energy_containment_bandwidth(psd, fraction=0.99).float()
    feat['rolloff_85'] = spectral_rolloff(psd, fraction=0.85).float()
    feat['rolloff_95'] = spectral_rolloff(psd, fraction=0.95).float()
    feat['knee_5pct'] = sorted_magnitude_knee(psd, ratio_thresh=0.05).float()
    feat['knee_10pct'] = sorted_magnitude_knee(psd, ratio_thresh=0.10).float()
    feat['eig_gap'] = eigenvalue_gap(x_adv, max_eig=40).float()
    centroid, spread = spectral_centroid_spread(psd)
    feat['centroid'] = centroid
    feat['spread'] = spread

    # MDL (slow, do in batches)
    print("  Computing MDL model order...")
    mdl_orders = []
    mdl_bs = 500
    for i in range(0, N, mdl_bs):
        mdl_orders.append(mdl_model_order(x_adv[i:i+mdl_bs], max_order=30))
    feat['mdl'] = torch.cat(mdl_orders).float()

    # Print feature stats
    print("\nFeature statistics:")
    print(f"  {'Feature':<14s}  {'Mean':>8s}  {'Std':>8s}  {'Min':>8s}  {'Max':>8s}")
    for name, f in feat.items():
        print(f"  {name:<14s}  {f.mean():>8.2f}  {f.std():>8.2f}  {f.min():>8.2f}  {f.max():>8.2f}")

    # ── Per-mod feature analysis ────────────────────────────────────────
    print("\nPer-mod feature means (on adversarial signals):")
    mod_order = ['BPSK', 'QPSK', '8PSK', 'QAM16', 'QAM64',
                 'PAM4', 'CPFSK', 'GFSK', 'AM-DSB', 'AM-SSB', 'WBFM']
    mod_best_k = {
        'AM-DSB': 2, 'QAM64': 10, 'PAM4': 15, 'GFSK': 15,
        'QAM16': 20, 'BPSK': 64, 'QPSK': 50, '8PSK': 64,
        'CPFSK': 64, 'AM-SSB': 64, 'WBFM': 10,
    }

    header = f"  {'Mod':<8s} {'OptK':>5s}"
    key_feats = ['entropy', 'kurtosis', 'ecb_95', 'knee_5pct', 'eig_gap', 'spread', 'mdl']
    for fn in key_feats:
        header += f"  {fn:>10s}"
    print(header)
    for m in mod_order:
        midx = None
        for ci, cn in idx_to_mod.items():
            if cn == m:
                midx = ci
                break
        if midx is None:
            continue
        mask = (lab_test == midx)
        if mask.sum() == 0:
            continue
        row = f"  {m:<8s} {mod_best_k.get(m, '?'):>5}"
        for fn in key_feats:
            val = feat[fn][mask].mean().item()
            row += f"  {val:>10.2f}"
        print(row)

    # ═══════════════════════════════════════════════════════════════════
    # Test all approaches
    # ═══════════════════════════════════════════════════════════════════
    K_CANDIDATES = [2, 5, 8, 10, 15, 20, 30, 50, 64]

    results = {}  # name -> {acc, k_mean, k_std, per_mod}

    def evaluate_approach(name, selected_k):
        """Apply adaptive K, compute overall + per-mod accuracy."""
        x_def = apply_adaptive_k(x_adv, selected_k, K_CANDIDATES)
        acc = batch_accuracy(model, x_def, lab_test)
        pm = per_mod_accuracy(model, x_def, lab_test, idx_to_mod)
        km = selected_k.float().mean().item()
        ks = selected_k.float().std().item()
        results[name] = {'acc': acc, 'k_mean': km, 'k_std': ks, 'per_mod': pm}
        return acc

    # ── A: Spectral Entropy ─────────────────────────────────────────
    print("\n=== [A] Spectral Entropy ===")
    # High entropy = spread spectrum = large K, low entropy = concentrated = small K
    for mapping in ['linear', 'quantile']:
        if mapping == 'linear':
            sk = feature_to_k_linear(feat['entropy'], k_min=2, k_max=64, invert=False)
        else:
            sk = feature_to_k_quantile(feat['entropy'], K_CANDIDATES, invert=False)
        name = f'A_entropy_{mapping}'
        acc = evaluate_approach(name, sk)
        print(f"  {name}: {acc*100:.2f}%, K_mean={sk.float().mean():.1f}")

    # Threshold-based with tuned breakpoints
    # From per-mod: low entropy (AM-DSB ~4.5) → K=2, high entropy (BPSK ~6.5) → K=64
    for thresholds, kvalues, tag in [
        ([4.5, 5.0, 5.3, 5.6, 5.9, 6.1, 6.3, 6.5], [2, 5, 8, 10, 15, 20, 30, 50, 64], 'fine'),
        ([5.0, 5.5, 6.0, 6.5], [2, 8, 15, 30, 64], 'coarse'),
    ]:
        sk = feature_to_k_thresholds(feat['entropy'], thresholds, kvalues)
        name = f'A_entropy_thresh_{tag}'
        acc = evaluate_approach(name, sk)
        print(f"  {name}: {acc*100:.2f}%, K_mean={sk.float().mean():.1f}")

    # ── B: Spectral Kurtosis ────────────────────────────────────────
    print("\n=== [B] Spectral Kurtosis ===")
    # High kurtosis = peaked = small K, low kurtosis = flat = large K
    for mapping in ['linear', 'quantile']:
        if mapping == 'linear':
            sk = feature_to_k_linear(feat['kurtosis'], k_min=2, k_max=64, invert=True)
        else:
            sk = feature_to_k_quantile(feat['kurtosis'], K_CANDIDATES, invert=True)
        name = f'B_kurtosis_{mapping}'
        acc = evaluate_approach(name, sk)
        print(f"  {name}: {acc*100:.2f}%, K_mean={sk.float().mean():.1f}")

    # ── C: Energy Containment Bandwidth ─────────────────────────────
    print("\n=== [C] Energy Containment Bandwidth ===")
    # ECB directly estimates how many bins needed → use as K
    for frac_name, ecb_feat in [('ecb_90', feat['ecb_90']),
                                 ('ecb_95', feat['ecb_95']),
                                 ('ecb_99', feat['ecb_99'])]:
        # Direct: K = ECB (clamp to candidates)
        sk = ecb_feat.round().int().clamp(2, 64)
        # Snap to nearest candidate
        cands = torch.tensor(K_CANDIDATES, device=device)
        dist = (sk.unsqueeze(1) - cands.unsqueeze(0)).abs()
        sk_snap = cands[dist.argmin(dim=1)]
        name = f'C_{frac_name}_direct'
        acc = evaluate_approach(name, sk_snap)
        print(f"  {name}: {acc*100:.2f}%, K_mean={sk_snap.float().mean():.1f}")

        # Scaled: K = ECB * multiplier
        for mult in [0.5, 0.75, 1.0, 1.25, 1.5]:
            sk = (ecb_feat * mult).round().int().clamp(2, 64)
            sk_snap = cands[((sk.unsqueeze(1) - cands.unsqueeze(0)).abs()).argmin(dim=1)]
            name = f'C_{frac_name}_x{mult}'
            acc = evaluate_approach(name, sk_snap)
            print(f"  {name}: {acc*100:.2f}%, K_mean={sk_snap.float().mean():.1f}")

    # ── D: Spectral Roll-off ────────────────────────────────────────
    print("\n=== [D] Spectral Roll-off ===")
    for ro_name, ro_feat in [('rolloff_85', feat['rolloff_85']),
                              ('rolloff_95', feat['rolloff_95'])]:
        cands = torch.tensor(K_CANDIDATES, device=device)
        for mult in [0.5, 0.75, 1.0]:
            sk = (ro_feat * mult).round().int().clamp(2, 64)
            sk_snap = cands[((sk.unsqueeze(1) - cands.unsqueeze(0)).abs()).argmin(dim=1)]
            name = f'D_{ro_name}_x{mult}'
            acc = evaluate_approach(name, sk_snap)
            print(f"  {name}: {acc*100:.2f}%, K_mean={sk_snap.float().mean():.1f}")

    # ── E: Sorted Magnitude Knee (Elbow) ────────────────────────────
    print("\n=== [E] Sorted Magnitude Knee ===")
    cands = torch.tensor(K_CANDIDATES, device=device)
    for knee_name, knee_feat in [('knee_5pct', feat['knee_5pct']),
                                  ('knee_10pct', feat['knee_10pct'])]:
        for mult in [0.75, 1.0, 1.5, 2.0]:
            sk = (knee_feat * mult).round().int().clamp(2, 64)
            sk_snap = cands[((sk.unsqueeze(1) - cands.unsqueeze(0)).abs()).argmin(dim=1)]
            name = f'E_{knee_name}_x{mult}'
            acc = evaluate_approach(name, sk_snap)
            print(f"  {name}: {acc*100:.2f}%, K_mean={sk_snap.float().mean():.1f}")

    # ── F: MDL Model Order ──────────────────────────────────────────
    print("\n=== [F] MDL Model Order ===")
    cands = torch.tensor(K_CANDIDATES, device=device)
    for mult in [1.0, 2.0, 3.0, 5.0]:
        sk = (feat['mdl'] * mult).round().int().clamp(2, 64)
        sk_snap = cands[((sk.unsqueeze(1) - cands.unsqueeze(0)).abs()).argmin(dim=1)]
        name = f'F_mdl_x{mult}'
        acc = evaluate_approach(name, sk_snap)
        print(f"  {name}: {acc*100:.2f}%, K_mean={sk_snap.float().mean():.1f}")

    # ── G: Eigenvalue Gap ───────────────────────────────────────────
    print("\n=== [G] Eigenvalue Gap ===")
    cands = torch.tensor(K_CANDIDATES, device=device)
    for mult in [1.0, 1.5, 2.0, 3.0]:
        sk = (feat['eig_gap'] * mult).round().int().clamp(2, 64)
        sk_snap = cands[((sk.unsqueeze(1) - cands.unsqueeze(0)).abs()).argmin(dim=1)]
        name = f'G_eiggap_x{mult}'
        acc = evaluate_approach(name, sk_snap)
        print(f"  {name}: {acc*100:.2f}%, K_mean={sk_snap.float().mean():.1f}")

    # ── H: Spectral Spread ──────────────────────────────────────────
    print("\n=== [H] Spectral Spread ===")
    # Wide spread → large K
    sk = feature_to_k_linear(feat['spread'], k_min=2, k_max=64, invert=False)
    name = 'H_spread_linear'
    acc = evaluate_approach(name, sk)
    print(f"  {name}: {acc*100:.2f}%, K_mean={sk.float().mean():.1f}")

    sk = feature_to_k_quantile(feat['spread'], K_CANDIDATES, invert=False)
    name = 'H_spread_quantile'
    acc = evaluate_approach(name, sk)
    print(f"  {name}: {acc*100:.2f}%, K_mean={sk.float().mean():.1f}")

    # ── I: Hybrid (Entropy + ECB) ───────────────────────────────────
    print("\n=== [I] Hybrid: Entropy + ECB95 ===")
    # Normalize both to [0,1], average, map to K
    ent_norm = (feat['entropy'] - feat['entropy'].min()) / \
               (feat['entropy'].max() - feat['entropy'].min()).clamp(min=1e-8)
    ecb_norm = (feat['ecb_95'] - feat['ecb_95'].min()) / \
               (feat['ecb_95'].max() - feat['ecb_95'].min()).clamp(min=1e-8)

    for w_ent, w_ecb in [(0.5, 0.5), (0.7, 0.3), (0.3, 0.7)]:
        hybrid = w_ent * ent_norm + w_ecb * ecb_norm
        sk = feature_to_k_linear(hybrid, k_min=2, k_max=64, invert=False)
        name = f'I_hybrid_ent{w_ent:.1f}_ecb{w_ecb:.1f}'
        acc = evaluate_approach(name, sk)
        print(f"  {name}: {acc*100:.2f}%, K_mean={sk.float().mean():.1f}")

    # Also: entropy + knee
    knee_norm = (feat['knee_5pct'] - feat['knee_5pct'].min()) / \
                (feat['knee_5pct'].max() - feat['knee_5pct'].min()).clamp(min=1e-8)
    hybrid2 = 0.5 * ent_norm + 0.5 * knee_norm
    sk = feature_to_k_linear(hybrid2, k_min=2, k_max=64, invert=False)
    name = 'I_hybrid_ent_knee'
    acc = evaluate_approach(name, sk)
    print(f"  {name}: {acc*100:.2f}%, K_mean={sk.float().mean():.1f}")

    # ── J: Feature-based mod classification → K lookup ──────────────
    print("\n=== [J] Feature-classify → K lookup (no model) ===")
    # Use (entropy, ecb_95, kurtosis) to cluster samples into mod-groups
    # Then assign K based on the mod's known best K
    # Simple: threshold on entropy alone to split into 3 groups
    # Group 1 (low entropy, narrow): AM-DSB, QAM64, PAM4 → small K
    # Group 2 (medium entropy): GFSK, QAM16, WBFM → medium K
    # Group 3 (high entropy, wide): BPSK, QPSK, 8PSK, CPFSK, AM-SSB → large K
    for t_low, t_high, k_low, k_mid, k_high in [
        (5.0, 6.0, 8, 20, 50),
        (5.0, 5.8, 8, 15, 50),
        (5.2, 6.0, 10, 20, 64),
        (4.8, 5.5, 5, 15, 50),
    ]:
        sk = torch.full((N,), k_mid, dtype=torch.int, device=device)
        sk[feat['entropy'] <= t_low] = k_low
        sk[feat['entropy'] > t_high] = k_high
        name = f'J_ent3grp_{t_low}_{t_high}_{k_low}_{k_mid}_{k_high}'
        acc = evaluate_approach(name, sk)
        print(f"  {name}: {acc*100:.2f}%, K_mean={sk.float().mean():.1f}")

    # More refined: use entropy + ecb together for 5 groups
    print("\n  5-group (entropy + ecb):")
    ent = feat['entropy']
    ecb = feat['ecb_95']
    # DC-concentrated: low entropy AND low ecb → K=2 (AM-DSB)
    # Narrowband digital: medium entropy, low ecb → K=10 (QAM64, PAM4)
    # Medium BW digital: medium entropy, medium ecb → K=15-20 (GFSK, QAM16)
    # Wideband digital: high entropy → K=50-64 (BPSK, QPSK, CPFSK)
    # Wideband analog: high entropy, high ecb → K=64 (AM-SSB)
    sk = torch.full((N,), 30, dtype=torch.int, device=device)
    sk[(ent < 4.8)] = 2                              # AM-DSB type
    sk[(ent >= 4.8) & (ent < 5.3) & (ecb < 15)] = 10  # QAM64 type
    sk[(ent >= 4.8) & (ent < 5.3) & (ecb >= 15)] = 15  # PAM4/GFSK type
    sk[(ent >= 5.3) & (ent < 6.0)] = 20               # QAM16 / medium
    sk[(ent >= 6.0)] = 50                              # BPSK/QPSK/CPFSK/AM-SSB
    name = 'J_5group_ent_ecb'
    acc = evaluate_approach(name, sk)
    print(f"  {name}: {acc*100:.2f}%, K_mean={sk.float().mean():.1f}")

    # ── K: Oracle ───────────────────────────────────────────────────
    print("\n=== [K] Oracle (per-sample best K) ===")
    t0 = time.time()
    oracle_k, oracle_acc = oracle_best_k(model, x_adv, lab_test, K_CANDIDATES)
    results['K_oracle'] = {
        'acc': oracle_acc,
        'k_mean': oracle_k.float().mean().item(),
        'k_std': oracle_k.float().std().item(),
        'per_mod': per_mod_accuracy(
            model, apply_adaptive_k(x_adv, oracle_k, K_CANDIDATES), lab_test, idx_to_mod),
    }
    print(f"  Oracle: {oracle_acc*100:.2f}%, K_mean={oracle_k.float().mean():.1f} ({time.time()-t0:.0f}s)")

    # ═══════════════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 120)
    print("GRAND SUMMARY — All Adaptive-K Approaches")
    print("=" * 120)
    print(f"  Clean accuracy:         {clean_acc*100:.2f}%")
    print(f"  CW attacked:            {adv_acc*100:.2f}%")
    print(f"  Best fixed K={best_fixed_k}:        {fixed_k_acc[best_fixed_k]*100:.2f}%")
    print(f"  Oracle:                 {oracle_acc*100:.2f}%")
    print("-" * 120)

    # Sort by accuracy
    sorted_res = sorted(results.items(), key=lambda x: x[1]['acc'], reverse=True)
    print(f"\n  {'#':>3s}  {'Approach':<42s}  {'Acc':>8s}  {'K_mean':>7s}  {'K_std':>7s}  | ", end='')
    for m in mod_order:
        print(f"{m[:5]:>6s}", end='')
    print()
    print("  " + "-" * 116)

    for rank, (name, r) in enumerate(sorted_res, 1):
        row = f"  {rank:>3d}  {name:<42s}  {r['acc']*100:>7.2f}%  {r['k_mean']:>6.1f}  {r['k_std']:>7.1f}  | "
        for m in mod_order:
            pm_acc = r.get('per_mod', {}).get(m, 0)
            row += f"{pm_acc*100:>5.1f}%"
        print(row)
        if rank == 10:
            print("  " + "·" * 116)

    print("=" * 120)

    # Save CSV
    os.makedirs(cfg.result_dir, exist_ok=True)
    csv_path = os.path.join(cfg.result_dir, 'adaptive_k_science_summary.csv')
    with open(csv_path, 'w') as f:
        f.write("rank,approach,accuracy,k_mean,k_std")
        for m in mod_order:
            f.write(f",{m}")
        f.write("\n")
        for rank, (name, r) in enumerate(sorted_res, 1):
            f.write(f"{rank},{name},{r['acc']:.6f},{r['k_mean']:.1f},{r['k_std']:.1f}")
            for m in mod_order:
                f.write(f",{r.get('per_mod', {}).get(m, 0):.4f}")
            f.write("\n")
    print(f"\nSaved: {csv_path}")

    # Best non-oracle
    non_oracle = [(n, r) for n, r in sorted_res if n != 'K_oracle']
    if non_oracle:
        bn, br = non_oracle[0]
        print(f"\n>>> BEST APPROACH: {bn}")
        print(f"    Accuracy: {br['acc']*100:.2f}%  (oracle: {oracle_acc*100:.2f}%)")
        print(f"    K_mean:   {br['k_mean']:.1f} +/- {br['k_std']:.1f}")
        print(f"    Gap to oracle: {(oracle_acc - br['acc'])*100:.2f}pp")


if __name__ == '__main__':
    main()
