"""
Transfer Attack Evaluation: NxN matrix of source->target model attack transferability.

For each source model, generates adversarial examples and evaluates on ALL target models.
Produces CSV with columns: attack, source_model, target_model, clean_acc, attack_acc, is_whitebox.
"""

import os
from typing import List, Optional

import pandas as pd
import torch
import torch.nn as nn

from util.adv_attack import Model01Wrapper
from util.sigguard_eval import (
    create_attack, generate_adversarial, compute_accuracy, ALL_ATTACKS,
    FIXED_BATCH_ATTACKS,
)
from util.utils import create_model


def _load_model(cfg, model_name, ckpt_path, logger):
    """Load a model from checkpoint."""
    ckpt_file = os.path.join(ckpt_path, f"{cfg.dataset}_{model_name.upper()}.pkl")
    if not os.path.exists(ckpt_file):
        logger.warning(f"Checkpoint not found: {ckpt_file}")
        return None
    model = create_model(cfg, model_name)
    model.load_state_dict(torch.load(ckpt_file, map_location=cfg.device))
    model.eval()
    return model


def run_transfer_eval(
    cfg,
    logger,
    sig_test: torch.Tensor,
    lab_test: torch.Tensor,
    model_names: List[str],
    ckpt_path: str,
    attacks: Optional[List[str]] = None,
    eval_limit: Optional[int] = None,
    batch_size: int = 128,
) -> pd.DataFrame:
    """
    Run NxN transfer attack evaluation.

    Args:
        cfg: Configuration object
        logger: Logger instance
        sig_test: Test signals [N, 2, T]
        lab_test: Test labels [N]
        model_names: List of model names (e.g., ['awn', 'vtcnn2', 'resnet1d', 'lstm'])
        ckpt_path: Path to checkpoint directory
        attacks: List of attack names (default: ['fgsm', 'pgd', 'cw'])
        eval_limit: Limit number of test samples
        batch_size: Batch size for attack generation

    Returns:
        DataFrame with transfer results
    """
    if attacks is None:
        attacks = ['fgsm', 'pgd', 'cw']

    device = cfg.device
    ta_box = str(getattr(cfg, 'ta_box', 'unit')).lower()

    # Apply eval limit
    if eval_limit is not None and eval_limit < len(sig_test):
        indices = torch.randperm(len(sig_test))[:eval_limit]
        sig_test = sig_test[indices]
        lab_test = lab_test[indices]

    n_samples = len(sig_test)
    logger.info(f"Transfer evaluation: {n_samples} samples, models={model_names}, "
                f"attacks={attacks}")

    # Load all models
    models = {}
    for name in model_names:
        m = _load_model(cfg, name, ckpt_path, logger)
        if m is not None:
            models[name] = m
            logger.info(f"Loaded {name.upper()}")

    if len(models) < 2:
        logger.warning("Need at least 2 models for transfer evaluation")
        return pd.DataFrame()

    # Compute clean accuracy for all models
    clean_accs = {}
    for name, model in models.items():
        accs = []
        for i in range(0, n_samples, batch_size):
            x_batch = sig_test[i:i+batch_size].to(device)
            y_batch = lab_test[i:i+batch_size].to(device)
            acc = compute_accuracy(model, x_batch, y_batch)
            accs.append(acc * len(y_batch))
        clean_accs[name] = sum(accs) / n_samples
        logger.info(f"  {name.upper()} clean accuracy: {clean_accs[name]*100:.2f}%")

    results = []

    # For each source model, generate adversarial examples and evaluate on all targets
    for source_name, source_model in models.items():
        wrapped_source = Model01Wrapper(source_model)
        wrapped_source.to(device)
        wrapped_source.eval()

        for attack_name in attacks:
            logger.info(f"\n=== {attack_name.upper()}: source={source_name.upper()} ===")

            try:
                atk = create_attack(attack_name, wrapped_source, cfg)
            except Exception as e:
                logger.warning(f"Failed to create {attack_name} for {source_name}: {e}")
                continue

            needs_padding = attack_name.lower() in FIXED_BATCH_ATTACKS

            # Generate adversarial examples from source model
            all_adv = []
            for i in range(0, n_samples, batch_size):
                x_batch = sig_test[i:i+batch_size].to(device)
                y_batch = lab_test[i:i+batch_size].to(device)
                try:
                    x_adv = generate_adversarial(
                        atk, x_batch, y_batch,
                        wrapped_model=wrapped_source,
                        ta_box=ta_box,
                        pad_to_batch_size=batch_size if needs_padding else None,
                    )
                    all_adv.append(x_adv.cpu())
                except Exception as e:
                    logger.warning(f"Attack batch failed: {e}")
                    all_adv.append(x_batch.cpu())

            adv_signals = torch.cat(all_adv, dim=0)

            # Evaluate adversarial examples on all target models
            for target_name, target_model in models.items():
                target_accs = []
                for i in range(0, n_samples, batch_size):
                    x_adv_batch = adv_signals[i:i+batch_size].to(device)
                    y_batch = lab_test[i:i+batch_size].to(device)
                    acc = compute_accuracy(target_model, x_adv_batch, y_batch)
                    target_accs.append(acc * len(y_batch))

                attack_acc = sum(target_accs) / n_samples
                is_wb = source_name == target_name

                results.append({
                    'attack': attack_name.upper(),
                    'source_model': source_name.upper(),
                    'target_model': target_name.upper(),
                    'clean_acc': clean_accs[target_name],
                    'attack_acc': attack_acc,
                    'is_whitebox': is_wb,
                })

                tag = "WB" if is_wb else "BB"
                logger.info(f"  [{tag}] {source_name.upper()} -> {target_name.upper()}: "
                            f"{attack_acc*100:.2f}%")

    df = pd.DataFrame(results)

    # Print pivot tables per attack
    for attack_name in attacks:
        atk_df = df[df['attack'] == attack_name.upper()]
        if atk_df.empty:
            continue
        pivot = atk_df.pivot(
            index='source_model', columns='target_model', values='attack_acc'
        )
        logger.info(f"\n{'='*60}")
        logger.info(f"Transfer Matrix: {attack_name.upper()} (accuracy under attack)")
        logger.info(f"Rows=source (attacker), Cols=target (victim)")
        logger.info(f"{'='*60}")
        # Format as percentage table
        header = f"{'Source':<12}" + "".join(f"{c:>12}" for c in pivot.columns)
        logger.info(header)
        logger.info("-" * len(header))
        for idx, row in pivot.iterrows():
            vals = "".join(f"{v*100:>11.2f}%" for v in row)
            logger.info(f"{idx:<12}{vals}")
        logger.info("")

    # Save CSV
    os.makedirs(cfg.result_dir, exist_ok=True)
    csv_path = os.path.join(cfg.result_dir, 'transfer_eval.csv')
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved transfer results to: {csv_path}")

    return df
