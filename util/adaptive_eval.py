"""
Adaptive attack evaluation: compares 3 conditions for each (attack, topk):

1. Standard attack, no defense          -> std_no_def_acc
2. Standard attack + post-hoc defense   -> std_with_def_acc
3. Adaptive attack (through defense)    -> adaptive_acc

If adaptive_acc ~= std_no_def_acc, the defense is broken (key paper result).
"""

import os
from typing import List, Optional

import pandas as pd
import torch
import torch.nn as nn

from util.adv_attack import Model01Wrapper
from util.adaptive_attack import DefendedModel01Wrapper
from util.defense import fft_topk_denoise
from util.sigguard_eval import (
    create_attack, generate_adversarial, compute_accuracy,
    FIXED_BATCH_ATTACKS,
)


def run_adaptive_eval(
    model: nn.Module,
    sig_test: torch.Tensor,
    lab_test: torch.Tensor,
    cfg,
    logger,
    topk_values: Optional[List[int]] = None,
    attacks: Optional[List[str]] = None,
    eval_limit: Optional[int] = None,
    batch_size: int = 128,
) -> pd.DataFrame:
    """
    Run adaptive attack evaluation.

    Args:
        model: Base classification model
        sig_test: Test signals [N, 2, T]
        lab_test: Test labels [N]
        cfg: Configuration object
        logger: Logger instance
        topk_values: List of Top-K values to evaluate (default: [20, 50])
        attacks: List of attack names (default: ['fgsm', 'pgd', 'cw'])
        eval_limit: Limit number of test samples
        batch_size: Batch size

    Returns:
        DataFrame with columns: attack, topk, std_no_def_acc, std_with_def_acc, adaptive_acc
    """
    if topk_values is None:
        topk_values = [20, 50]
    if attacks is None:
        attacks = ['fgsm', 'pgd', 'cw']

    device = cfg.device
    ta_box = str(getattr(cfg, 'ta_box', 'unit')).lower()
    model.eval()

    # Apply eval limit
    if eval_limit is not None and eval_limit < len(sig_test):
        indices = torch.randperm(len(sig_test))[:eval_limit]
        sig_test = sig_test[indices]
        lab_test = lab_test[indices]

    n_samples = len(sig_test)
    logger.info(f"Adaptive evaluation: {n_samples} samples, attacks={attacks}, "
                f"topk={topk_values}")

    # Compute clean accuracy
    clean_accs = []
    for i in range(0, n_samples, batch_size):
        x_b = sig_test[i:i+batch_size].to(device)
        y_b = lab_test[i:i+batch_size].to(device)
        clean_accs.append(compute_accuracy(model, x_b, y_b) * len(y_b))
    clean_acc = sum(clean_accs) / n_samples
    logger.info(f"Clean accuracy: {clean_acc*100:.2f}%")

    # Standard wrapper (no defense)
    wrapped_std = Model01Wrapper(model)
    wrapped_std.to(device)
    wrapped_std.eval()

    results = []

    for attack_name in attacks:
        logger.info(f"\n{'='*60}")
        logger.info(f"Attack: {attack_name.upper()}")
        logger.info(f"{'='*60}")

        needs_padding = attack_name.lower() in FIXED_BATCH_ATTACKS

        # --- Condition 1: Standard attack, no defense ---
        try:
            atk_std = create_attack(attack_name, wrapped_std, cfg)
        except Exception as e:
            logger.warning(f"Failed to create {attack_name}: {e}")
            continue

        all_adv_std = []
        for i in range(0, n_samples, batch_size):
            x_b = sig_test[i:i+batch_size].to(device)
            y_b = lab_test[i:i+batch_size].to(device)
            try:
                x_adv = generate_adversarial(
                    atk_std, x_b, y_b,
                    wrapped_model=wrapped_std, ta_box=ta_box,
                    pad_to_batch_size=batch_size if needs_padding else None,
                )
                all_adv_std.append(x_adv.cpu())
            except Exception as e:
                logger.warning(f"Standard attack batch failed: {e}")
                all_adv_std.append(x_b.cpu())

        adv_std = torch.cat(all_adv_std, dim=0)

        # Evaluate standard attack without defense
        std_no_def_correct = 0
        for i in range(0, n_samples, batch_size):
            x_b = adv_std[i:i+batch_size].to(device)
            y_b = lab_test[i:i+batch_size].to(device)
            std_no_def_correct += compute_accuracy(model, x_b, y_b) * len(y_b)
        std_no_def_acc = std_no_def_correct / n_samples
        logger.info(f"  Standard attack, no defense: {std_no_def_acc*100:.2f}%")

        for topk in topk_values:
            logger.info(f"\n  --- Top-K = {topk} ---")

            # --- Condition 2: Standard attack + post-hoc defense ---
            std_with_def_correct = 0
            for i in range(0, n_samples, batch_size):
                x_b = adv_std[i:i+batch_size].to(device)
                y_b = lab_test[i:i+batch_size].to(device)
                x_def = fft_topk_denoise(x_b, topk=topk)
                std_with_def_correct += compute_accuracy(model, x_def, y_b) * len(y_b)
            std_with_def_acc = std_with_def_correct / n_samples
            logger.info(f"  Standard attack + Top-{topk} defense: "
                        f"{std_with_def_acc*100:.2f}%")

            # --- Condition 3: Adaptive attack (through defense) ---
            wrapped_defended = DefendedModel01Wrapper(model, topk=topk)
            wrapped_defended.to(device)
            wrapped_defended.eval()

            try:
                atk_adaptive = create_attack(attack_name, wrapped_defended, cfg)
            except Exception as e:
                logger.warning(f"Failed to create adaptive {attack_name}: {e}")
                results.append({
                    'attack': attack_name.upper(),
                    'topk': topk,
                    'clean_acc': clean_acc,
                    'std_no_def_acc': std_no_def_acc,
                    'std_with_def_acc': std_with_def_acc,
                    'adaptive_acc': float('nan'),
                })
                continue

            adaptive_correct = 0
            for i in range(0, n_samples, batch_size):
                x_b = sig_test[i:i+batch_size].to(device)
                y_b = lab_test[i:i+batch_size].to(device)
                try:
                    x_adv_adapt = generate_adversarial(
                        atk_adaptive, x_b, y_b,
                        wrapped_model=wrapped_defended, ta_box=ta_box,
                        pad_to_batch_size=batch_size if needs_padding else None,
                    )
                    # Evaluate WITH defense applied (adaptive attack should bypass it)
                    x_def = fft_topk_denoise(x_adv_adapt, topk=topk)
                    adaptive_correct += compute_accuracy(model, x_def, y_b) * len(y_b)
                except Exception as e:
                    logger.warning(f"Adaptive attack batch failed: {e}")
                    x_def = fft_topk_denoise(x_b, topk=topk)
                    adaptive_correct += compute_accuracy(model, x_def, y_b) * len(y_b)

            adaptive_acc = adaptive_correct / n_samples
            logger.info(f"  Adaptive attack + Top-{topk} defense: "
                        f"{adaptive_acc*100:.2f}%")

            # Summary for this combination
            defense_gain = std_with_def_acc - std_no_def_acc
            adaptive_bypass = std_with_def_acc - adaptive_acc
            logger.info(f"  Defense gain (std): +{defense_gain*100:.2f}pp")
            logger.info(f"  Adaptive bypass:    -{adaptive_bypass*100:.2f}pp")
            if adaptive_acc <= std_no_def_acc + 0.05:
                logger.info(f"  >> DEFENSE BROKEN: adaptive ~= no-defense")
            else:
                logger.info(f"  >> Defense partially effective against adaptive attack")

            results.append({
                'attack': attack_name.upper(),
                'topk': topk,
                'clean_acc': clean_acc,
                'std_no_def_acc': std_no_def_acc,
                'std_with_def_acc': std_with_def_acc,
                'adaptive_acc': adaptive_acc,
            })

    df = pd.DataFrame(results)

    # Print summary table
    logger.info(f"\n{'='*80}")
    logger.info(f"Adaptive Attack Summary (Clean acc: {clean_acc*100:.2f}%)")
    logger.info(f"{'='*80}")
    header = f"{'Attack':<10} {'TopK':>5} {'No Defense':>12} {'With Defense':>13} {'Adaptive':>12} {'Broken?':>8}"
    logger.info(header)
    logger.info("-" * 80)
    for _, row in df.iterrows():
        broken = "YES" if row['adaptive_acc'] <= row['std_no_def_acc'] + 0.05 else "no"
        logger.info(
            f"{row['attack']:<10} {row['topk']:>5d} "
            f"{row['std_no_def_acc']*100:>11.2f}% "
            f"{row['std_with_def_acc']*100:>12.2f}% "
            f"{row['adaptive_acc']*100:>11.2f}% "
            f"{broken:>8}"
        )
    logger.info(f"{'='*80}\n")

    # Save
    os.makedirs(cfg.result_dir, exist_ok=True)
    csv_path = os.path.join(cfg.result_dir, 'adaptive_eval.csv')
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved adaptive eval results to: {csv_path}")

    return df
