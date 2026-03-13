"""
Calibration mode for adaptive Top-K selection.

For each sample in the clean test set:
1. Compute cumulative_energy_knee at multiple alpha thresholds
2. Find ground-truth minimum safe K (smallest K where classifier is correct)
3. Report per-modulation knee distributions and mapping accuracy

Output: calibration JSON with recommended threshold.
"""

import os
import json
from typing import List, Dict, Optional
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

from util.defense import cumulative_energy_knee, fft_topk_denoise


@torch.no_grad()
def run_adaptive_k_calibration(
    model: nn.Module,
    sig_test: torch.Tensor,
    lab_test: torch.Tensor,
    SNRs: List[int],
    test_idx: np.ndarray,
    cfg,
    logger,
    thresholds: List[float] = None,
    k_candidates: List[int] = None,
    eval_limit: Optional[int] = None,
    batch_size: int = 256,
) -> Dict:
    """
    Calibrate adaptive K selection on clean data.

    For each sample:
    - Compute spectral energy knee at each threshold
    - Find ground-truth minimum safe K (smallest candidate where classifier correct)
    - Report per-modulation statistics

    Args:
        model: AWN classification model
        sig_test: Test signals [N, 2, T]
        lab_test: Test labels [N]
        SNRs: SNR values for all samples
        test_idx: Indices of test samples in original dataset
        cfg: Config object
        logger: Logger instance
        thresholds: List of alpha values to test (default: [0.80, 0.85, 0.90, 0.95])
        k_candidates: Candidate K values (default: [10, 15, 20, 30, 50])
        eval_limit: Limit number of samples
        batch_size: Batch size for processing

    Returns:
        Calibration results dict
    """
    if thresholds is None:
        thresholds = [0.80, 0.85, 0.90, 0.95]
    if k_candidates is None:
        k_candidates = [10, 15, 20, 30, 50]
    k_cands = sorted(k_candidates)

    model.eval()
    device = cfg.device

    # Build class index -> mod name mapping
    idx_to_mod = {}
    for k, v in cfg.classes.items():
        if isinstance(k, bytes):
            idx_to_mod[v] = k.decode()
        else:
            idx_to_mod[v] = str(k)

    # Apply eval limit
    if eval_limit is not None and eval_limit < len(sig_test):
        sig_test = sig_test[:eval_limit]
        lab_test = lab_test[:eval_limit]
        test_idx = test_idx[:eval_limit]

    n_samples = len(sig_test)
    logger.info(f"Calibrating adaptive K on {n_samples} clean samples")
    logger.info(f"Thresholds: {thresholds}")
    logger.info(f"K candidates: {k_cands}")

    # Storage for per-sample results
    all_knees = {alpha: [] for alpha in thresholds}  # alpha -> [N] knee values
    all_gt_safe_k = []   # [N] ground-truth minimum safe K
    all_labels = []      # [N] modulation labels
    all_snrs = []        # [N] SNR values

    for i in range(0, n_samples, batch_size):
        x_batch = sig_test[i:i+batch_size].to(device)
        y_batch = lab_test[i:i+batch_size].to(device)
        batch_idx = test_idx[i:i+batch_size]

        # 1. Compute knee at each threshold
        for alpha in thresholds:
            knee = cumulative_energy_knee(x_batch, threshold=alpha)
            all_knees[alpha].append(knee.cpu())

        # 2. Find ground-truth minimum safe K for each sample
        # Try each K candidate from smallest to largest
        correct_at_k = {}
        for kc in k_cands:
            x_topk = fft_topk_denoise(x_batch, topk=kc)
            logits, _ = model(x_topk)
            preds = logits.argmax(dim=1)
            correct_at_k[kc] = (preds == y_batch).cpu()

        # For each sample, find smallest K where correct
        bs = len(y_batch)
        gt_safe_k = torch.full((bs,), k_cands[-1], dtype=torch.int)
        for j in range(bs):
            for kc in k_cands:
                if correct_at_k[kc][j]:
                    gt_safe_k[j] = kc
                    break
        all_gt_safe_k.append(gt_safe_k)
        all_labels.append(y_batch.cpu())

        # Get SNR values for this batch
        for idx in batch_idx:
            all_snrs.append(SNRs[idx])

    # Concatenate results
    for alpha in thresholds:
        all_knees[alpha] = torch.cat(all_knees[alpha])
    all_gt_safe_k = torch.cat(all_gt_safe_k)
    all_labels = torch.cat(all_labels)
    all_snrs = np.array(all_snrs)

    # --- Analysis ---
    logger.info("\n=== Per-Modulation Knee Distribution ===")

    calibration = {
        'thresholds': thresholds,
        'k_candidates': k_cands,
        'n_samples': n_samples,
        'per_modulation': {},
        'per_threshold': {},
    }

    # Per-modulation analysis
    unique_labels = sorted(torch.unique(all_labels).numpy().astype(int))
    for label_idx in unique_labels:
        mod_name = idx_to_mod.get(label_idx, f'class_{label_idx}')
        mask = (all_labels == label_idx)
        n_mod = mask.sum().item()
        gt_k = all_gt_safe_k[mask].float()

        mod_info = {
            'n_samples': n_mod,
            'gt_safe_k_mean': float(gt_k.mean()),
            'gt_safe_k_median': float(gt_k.median()),
            'gt_safe_k_dist': {},
            'knee_stats': {},
        }

        # GT safe K distribution
        for kc in k_cands:
            cnt = int((all_gt_safe_k[mask] == kc).sum().item())
            mod_info['gt_safe_k_dist'][kc] = cnt

        # Knee stats at each threshold
        for alpha in thresholds:
            knees = all_knees[alpha][mask].float()
            mod_info['knee_stats'][str(alpha)] = {
                'mean': float(knees.mean()),
                'median': float(knees.median()),
                'p25': float(knees.quantile(0.25)),
                'p75': float(knees.quantile(0.75)),
                'max': int(knees.max()),
            }

        calibration['per_modulation'][mod_name] = mod_info

        logger.info(f"\n  {mod_name} (n={n_mod}):")
        logger.info(f"    GT safe K: mean={gt_k.mean():.1f}, median={gt_k.median():.0f}, "
                    f"dist={mod_info['gt_safe_k_dist']}")
        for alpha in thresholds:
            stats = mod_info['knee_stats'][str(alpha)]
            logger.info(f"    alpha={alpha}: knee mean={stats['mean']:.1f}, "
                        f"median={stats['median']:.0f}, "
                        f"p25={stats['p25']:.0f}, p75={stats['p75']:.0f}")

    # Per-threshold mapping accuracy
    logger.info("\n=== Mapping Accuracy (knee -> K candidate) ===")
    best_threshold = thresholds[0]
    best_mapping_acc = 0.0

    for alpha in thresholds:
        knees = all_knees[alpha]
        # Map knee to smallest candidate >= knee
        selected_k = torch.full_like(knees, k_cands[-1])
        for kc in reversed(k_cands):
            selected_k = torch.where(knees <= kc, kc, selected_k)
        selected_k = selected_k.clamp(k_cands[0], k_cands[-1])

        # Check if selected_k >= gt_safe_k (safe = won't under-filter)
        safe = (selected_k >= all_gt_safe_k).float().mean().item()
        # Check if selected_k == gt_safe_k (exact match)
        exact = (selected_k == all_gt_safe_k).float().mean().item()
        avg_k = selected_k.float().mean().item()

        calibration['per_threshold'][str(alpha)] = {
            'safe_rate': safe,
            'exact_match_rate': exact,
            'avg_selected_k': avg_k,
        }

        logger.info(f"  alpha={alpha}: safe_rate={safe*100:.1f}%, "
                    f"exact_match={exact*100:.1f}%, avg_K={avg_k:.1f}")

        if safe > best_mapping_acc:
            best_mapping_acc = safe
            best_threshold = alpha

    calibration['recommended_threshold'] = best_threshold
    logger.info(f"\nRecommended threshold: {best_threshold} (safe_rate={best_mapping_acc*100:.1f}%)")

    # Save calibration JSON
    os.makedirs(cfg.result_dir, exist_ok=True)
    out_path = os.path.join(cfg.result_dir, 'adaptive_k_calibration.json')
    with open(out_path, 'w') as f:
        json.dump(calibration, f, indent=2)
    logger.info(f"Saved calibration to: {out_path}")

    return calibration
