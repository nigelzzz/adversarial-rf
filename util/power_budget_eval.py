"""
Power Budget Evaluation: sweep epsilon and compute attack accuracy + PSR (dB).

Compares adversarial perturbation efficiency against equivalent AWGN jamming
at the same power level, producing accuracy-vs-PSR curves.
"""

import os
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from util.adv_attack import Model01Wrapper
from util.sigguard_eval import (
    create_attack, generate_adversarial, compute_accuracy,
    FIXED_BATCH_ATTACKS,
)


def _compute_psr_db(x_clean: torch.Tensor, x_adv: torch.Tensor) -> float:
    """
    Compute Perturbation-to-Signal Ratio in dB.
    PSR = 10 * log10( ||delta||^2 / ||x||^2 )
    """
    delta = x_adv - x_clean
    power_delta = delta.pow(2).mean().item()
    power_signal = x_clean.pow(2).mean().item()
    if power_signal < 1e-12:
        return float('nan')
    return 10.0 * np.log10(power_delta / power_signal)


def _awgn_attack(x: torch.Tensor, psr_db: float) -> torch.Tensor:
    """Add AWGN noise to achieve a given PSR (dB)."""
    power_signal = x.pow(2).mean()
    power_noise = power_signal * (10.0 ** (psr_db / 10.0))
    std = power_noise.sqrt()
    noise = torch.randn_like(x) * std
    return x + noise


def run_power_budget_eval(
    model: nn.Module,
    sig_test: torch.Tensor,
    lab_test: torch.Tensor,
    cfg,
    logger,
    attacks: Optional[List[str]] = None,
    epsilons: Optional[List[float]] = None,
    eval_limit: Optional[int] = None,
    batch_size: int = 128,
) -> pd.DataFrame:
    """
    Sweep epsilon values and measure attack accuracy + PSR.

    Args:
        model: Classification model
        sig_test: Test signals [N, 2, T]
        lab_test: Test labels [N]
        cfg: Configuration object
        logger: Logger instance
        attacks: Attack names (default: ['fgsm', 'pgd'])
        epsilons: List of epsilon values to sweep
        eval_limit: Limit test samples
        batch_size: Batch size

    Returns:
        DataFrame with columns: attack, epsilon, psr_db, attack_acc, awgn_acc
    """
    if attacks is None:
        attacks = ['fgsm', 'pgd']
    if epsilons is None:
        epsilons = [0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2]

    device = cfg.device
    ta_box = str(getattr(cfg, 'ta_box', 'unit')).lower()
    model.eval()

    if eval_limit is not None and eval_limit < len(sig_test):
        indices = torch.randperm(len(sig_test))[:eval_limit]
        sig_test = sig_test[indices]
        lab_test = lab_test[indices]

    n_samples = len(sig_test)
    logger.info(f"Power budget evaluation: {n_samples} samples, "
                f"attacks={attacks}, epsilons={epsilons}")

    # Clean accuracy
    clean_correct = 0
    for i in range(0, n_samples, batch_size):
        x_b = sig_test[i:i+batch_size].to(device)
        y_b = lab_test[i:i+batch_size].to(device)
        clean_correct += compute_accuracy(model, x_b, y_b) * len(y_b)
    clean_acc = clean_correct / n_samples
    logger.info(f"Clean accuracy: {clean_acc*100:.2f}%")

    wrapped = Model01Wrapper(model)
    wrapped.to(device)
    wrapped.eval()

    results = []

    for attack_name in attacks:
        logger.info(f"\n=== {attack_name.upper()} ===")

        for eps in epsilons:
            # Override epsilon for this sweep point
            orig_eps = getattr(cfg, 'attack_eps', 0.03)
            cfg.attack_eps = eps

            try:
                atk = create_attack(attack_name, wrapped, cfg)
            except Exception as e:
                logger.warning(f"Failed to create {attack_name} eps={eps}: {e}")
                cfg.attack_eps = orig_eps
                continue

            needs_padding = attack_name.lower() in FIXED_BATCH_ATTACKS

            # Generate adversarial + compute accuracy and PSR
            total_correct = 0
            total_psr_sum = 0.0
            n_processed = 0

            for i in range(0, n_samples, batch_size):
                x_b = sig_test[i:i+batch_size].to(device)
                y_b = lab_test[i:i+batch_size].to(device)
                try:
                    x_adv = generate_adversarial(
                        atk, x_b, y_b,
                        wrapped_model=wrapped, ta_box=ta_box,
                        pad_to_batch_size=batch_size if needs_padding else None,
                    )
                    total_correct += compute_accuracy(model, x_adv, y_b) * len(y_b)
                    psr = _compute_psr_db(x_b, x_adv)
                    if not np.isnan(psr):
                        total_psr_sum += psr * len(y_b)
                    n_processed += len(y_b)
                except Exception as e:
                    logger.warning(f"Batch failed: {e}")

            cfg.attack_eps = orig_eps

            if n_processed == 0:
                continue

            attack_acc = total_correct / n_processed
            avg_psr = total_psr_sum / n_processed

            # AWGN baseline at same PSR
            awgn_correct = 0
            for i in range(0, n_samples, batch_size):
                x_b = sig_test[i:i+batch_size].to(device)
                y_b = lab_test[i:i+batch_size].to(device)
                x_noisy = _awgn_attack(x_b, avg_psr)
                awgn_correct += compute_accuracy(model, x_noisy, y_b) * len(y_b)
            awgn_acc = awgn_correct / n_samples

            results.append({
                'attack': attack_name.upper(),
                'epsilon': eps,
                'psr_db': avg_psr,
                'attack_acc': attack_acc,
                'awgn_acc': awgn_acc,
                'clean_acc': clean_acc,
            })

            logger.info(
                f"  eps={eps:.4f}: PSR={avg_psr:.2f}dB, "
                f"Adv={attack_acc*100:.2f}%, AWGN={awgn_acc*100:.2f}%"
            )

    df = pd.DataFrame(results)

    # Plot accuracy vs PSR
    if not df.empty:
        os.makedirs(cfg.result_dir, exist_ok=True)

        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        for attack_name in attacks:
            atk_df = df[df['attack'] == attack_name.upper()].sort_values('psr_db')
            if atk_df.empty:
                continue
            ax.plot(atk_df['psr_db'], atk_df['attack_acc'] * 100,
                    'o-', label=f'{attack_name.upper()} (adversarial)')
            ax.plot(atk_df['psr_db'], atk_df['awgn_acc'] * 100,
                    's--', label=f'AWGN @ same PSR', alpha=0.7)

        ax.axhline(y=clean_acc * 100, color='green', linestyle=':',
                    label=f'Clean ({clean_acc*100:.1f}%)')
        ax.set_xlabel('Perturbation-to-Signal Ratio (dB)')
        ax.set_ylabel('Classification Accuracy (%)')
        ax.set_title('Adversarial vs AWGN Jamming Efficiency')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plot_path = os.path.join(cfg.result_dir, 'power_budget_eval.png')
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved plot to: {plot_path}")

        csv_path = os.path.join(cfg.result_dir, 'power_budget_eval.csv')
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved results to: {csv_path}")

    return df
