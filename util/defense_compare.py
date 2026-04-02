"""
util/defense_compare.py

Core defense comparison evaluation framework for the real-time AMC defense paper.

Provides:
  - run_defense_compare(): Iterates 9 defenses x 5 attacks x 10 SNR points,
    applies each defense, computes per-SNR accuracy, and saves results to CSV.
  - ATTACKS: List of 5 attacks for evaluation (CW, EAD-L1, EAD-EN, FGSM, PGD)
  - SNR_POINTS: List of 10 SNR evaluation points (0..18 dB in steps of 2)
  - DEFENSE_CONFIGS: Dict of 9 defenses mapping name to cfg override

Design decisions (from 02-CONTEXT.md):
  D-01: 5 attacks: CW (L2), EAD-L1, EAD-EN, FGSM (Linf), PGD (Linf)
  D-02: SNR >= 0 dB only — 10 points: 0, 2, 4, 6, 8, 10, 12, 14, 16, 18
  D-03: 9 defense rows: no_defense, ae_fft_topk, spectral_gated, kalman, wiener,
        savitzky_golay, gaussian, fir, rand_smooth
  D-04: 200 samples per (SNR, modulation) cell
  D-05: eps=0.03 (minmax) for Linf attacks; c=1.0 for CW/EAD
  D-09: One comparison table per attack; rows=9 defenses, cols=10 SNR + weighted avg
  D-10: Single run; report accuracy percentages
  D-11: Weighted average column weighted by samples per SNR
"""

import os
import logging
from typing import List, Optional, Any
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import accuracy_score

from util.adv_attack import (
    Model01Wrapper,
    iq_to_ta_input,
    ta_output_to_iq,
    iq_to_ta_input_minmax,
    ta_output_to_iq_minmax,
)
from util.defense_registry import (
    DEFENSE_REGISTRY,
    defend,
    randomized_smoothing_predict,
    _apply_filter,
)

try:
    import torchattacks
except ImportError:
    torchattacks = None

__all__ = ['run_defense_compare', 'ATTACKS', 'SNR_POINTS', 'DEFENSE_CONFIGS']

# ---------------------------------------------------------------------------
# Module-level constants (D-01, D-02, D-03)
# ---------------------------------------------------------------------------

ATTACKS = ['cw', 'eadl1', 'eaden', 'fgsm', 'pgd']

SNR_POINTS = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]

DEFENSE_CONFIGS = {
    'no_defense':      {'defense': 'none'},
    'ae_fft_topk':     {'defense': 'fft_topk'},  # unified pipeline uses detector gate
    'spectral_gated':  {'defense': 'spectral_gated'},
    'kalman':          {'defense': 'kalman'},
    'wiener':          {'defense': 'wiener'},
    'savitzky_golay':  {'defense': 'savitzky_golay'},
    'gaussian':        {'defense': 'gaussian'},
    'fir':             {'defense': 'fir'},
    'rand_smooth':     {'defense': 'rand_smooth'},
}

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Attack creation (mirrors sigguard_eval.py create_attack, limited to 5)
# ---------------------------------------------------------------------------

def create_attack(
    attack_name: str,
    wrapped_model: nn.Module,
    cfg,
) -> Any:
    """
    Create torchattacks attack object for the 5 attacks in ATTACKS.

    Reads cfg.attack_eps for Linf attacks (FGSM, PGD) — default 0.03.
    Reads cfg.cw_c for CW/EAD attacks — default 1.0.
    Reads cfg.cw_steps (default 200), cfg.cw_lr (default 0.005) for CW.
    Reads cfg.ead_* params for EAD attacks.
    """
    if torchattacks is None:
        raise ImportError("torchattacks not installed. Run: pip install torchattacks")

    name = attack_name.lower()
    eps = float(getattr(cfg, 'attack_eps', 0.03))
    alpha = eps / 4.0
    steps = 10

    if name == 'fgsm':
        return torchattacks.FGSM(wrapped_model, eps=eps)

    elif name == 'pgd':
        return torchattacks.PGD(wrapped_model, eps=eps, alpha=alpha, steps=steps)

    elif name == 'cw':
        c = float(getattr(cfg, 'cw_c', 1.0))
        cw_steps = int(getattr(cfg, 'cw_steps', 200))
        cw_lr = float(getattr(cfg, 'cw_lr', 0.005))
        return torchattacks.CW(wrapped_model, c=c, steps=cw_steps, lr=cw_lr)

    elif name == 'eadl1':
        kappa = float(getattr(cfg, 'ead_kappa', 0))
        lr = float(getattr(cfg, 'ead_lr', 0.01))
        max_iterations = int(getattr(cfg, 'ead_max_iterations', 100))
        binary_search_steps = int(getattr(cfg, 'ead_binary_search_steps', 9))
        initial_const = float(getattr(cfg, 'ead_initial_const', 0.001))
        beta = float(getattr(cfg, 'ead_beta', 0.001))
        return torchattacks.EADL1(
            wrapped_model,
            kappa=kappa,
            lr=lr,
            max_iterations=max_iterations,
            binary_search_steps=binary_search_steps,
            initial_const=initial_const,
            beta=beta,
        )

    elif name == 'eaden':
        kappa = float(getattr(cfg, 'ead_kappa', 0))
        lr = float(getattr(cfg, 'ead_lr', 0.01))
        max_iterations = int(getattr(cfg, 'ead_max_iterations', 100))
        binary_search_steps = int(getattr(cfg, 'ead_binary_search_steps', 9))
        initial_const = float(getattr(cfg, 'ead_initial_const', 0.001))
        beta = float(getattr(cfg, 'ead_beta', 0.001))
        return torchattacks.EADEN(
            wrapped_model,
            kappa=kappa,
            lr=lr,
            max_iterations=max_iterations,
            binary_search_steps=binary_search_steps,
            initial_const=initial_const,
            beta=beta,
        )

    else:
        raise ValueError(f"Unknown attack for defense_compare: {attack_name}. "
                         f"Must be one of: {ATTACKS}")


# ---------------------------------------------------------------------------
# Defense application helpers
# ---------------------------------------------------------------------------

def _get_filter_kwargs(defense_name: str, cfg) -> dict:
    """
    Extract filter-specific kwargs from cfg for classical filters.

    Mirrors the _apply_filter dispatch logic in defense_registry.py so that
    defense_compare uses the same parameter defaults.
    """
    if defense_name == 'fft_topk':
        return {'topk': int(getattr(cfg, 'def_topk', 50))}
    elif defense_name == 'spectral_gated':
        return {'topk': int(getattr(cfg, 'def_topk', 20))}
    elif defense_name == 'kalman':
        return {
            'process_noise': float(getattr(cfg, 'kalman_process_noise', 1e-4)),
            'meas_noise':    float(getattr(cfg, 'kalman_meas_noise', 0.01)),
        }
    elif defense_name == 'wiener':
        return {
            'noise':      float(getattr(cfg, 'wiener_noise', 0.01)),
            'filter_len': int(getattr(cfg, 'wiener_filter_len', 5)),
        }
    elif defense_name == 'savitzky_golay':
        return {
            'window_length': int(getattr(cfg, 'sg_window_length', 11)),
            'polyorder':     int(getattr(cfg, 'sg_polyorder', 3)),
        }
    elif defense_name == 'gaussian':
        return {'sigma': float(getattr(cfg, 'gaussian_sigma', 1.0))}
    elif defense_name == 'fir':
        return {
            'cutoff':  float(getattr(cfg, 'fir_cutoff', 0.1)),
            'numtaps': int(getattr(cfg, 'fir_numtaps', 31)),
        }
    else:
        return {}


def _apply_defense(
    defense_name: str,
    x_adv: torch.Tensor,
    model: nn.Module,
    detector,
    cfg,
    logger,
) -> torch.Tensor:
    """
    Apply a defense to adversarial signals and return integer class predictions.

    Dispatch logic:
    - 'no_defense':     classify adversarial directly without any filtering
    - 'ae_fft_topk':    use defend() pipeline (detector gate + fft_topk recovery)
    - 'rand_smooth':    use randomized_smoothing_predict() majority-vote wrapper
    - all others:       look up DEFENSE_REGISTRY[name], apply filter, then classify

    Args:
        defense_name: One of the 9 keys in DEFENSE_CONFIGS
        x_adv:        Adversarial signal tensor [N, 2, T] in raw IQ scale
        model:        AWN model: model(x) -> (logits [N, C], regu_sum)
        detector:     RFSignalAutoEncoder or None (for ae_fft_topk gate)
        cfg:          Config object with defense parameters
        logger:       Logger for progress messages

    Returns:
        predictions: [N] numpy array of int class predictions
    """
    device = x_adv.device

    if defense_name == 'no_defense':
        with torch.no_grad():
            logits, _ = model(x_adv)
        preds = logits.argmax(dim=1).cpu().numpy()

    elif defense_name == 'ae_fft_topk':
        # Unified detect->recover->classify pipeline
        # Temporarily set cfg.defense to 'fft_topk' for the defend() call
        orig_defense = getattr(cfg, 'defense', 'fft_topk')
        cfg.defense = 'fft_topk'
        with torch.no_grad():
            predictions_tensor, _ = defend(x_adv, model, detector, cfg)
        cfg.defense = orig_defense
        preds = predictions_tensor.cpu().numpy()

    elif defense_name == 'rand_smooth':
        rs_k     = int(getattr(cfg, 'rs_k', 20))
        rs_sigma = float(getattr(cfg, 'rs_sigma', 0.01))
        with torch.no_grad():
            predictions_tensor = randomized_smoothing_predict(
                model, x_adv, k=rs_k, sigma=rs_sigma
            )
        preds = predictions_tensor.cpu().numpy()

    else:
        # Classical filters + fft_topk + spectral_gated:
        # apply filter then classify
        filter_fn = DEFENSE_REGISTRY.get(defense_name)
        if filter_fn is None:
            logger.warning(
                "Defense '%s' not found in DEFENSE_REGISTRY (None). "
                "Running undefended classifier.", defense_name
            )
            with torch.no_grad():
                logits, _ = model(x_adv)
            preds = logits.argmax(dim=1).cpu().numpy()
        else:
            with torch.no_grad():
                x_filtered = _apply_filter(filter_fn, x_adv, defense_name, cfg)
                logits, _ = model(x_filtered)
            preds = logits.argmax(dim=1).cpu().numpy()

    return preds


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------

def run_defense_compare(
    model: nn.Module,
    sig_test: torch.Tensor,
    lab_test: torch.Tensor,
    SNRs: np.ndarray,
    test_idx: np.ndarray,
    cfg,
    logger,
    detector=None,
    attacks: Optional[List[str]] = None,
    snr_points: Optional[List[int]] = None,
    max_per_cell: int = 200,
    batch_size: int = 64,
) -> pd.DataFrame:
    """
    Run the defense comparison matrix: 9 defenses x 5 attacks x 10 SNR points.

    For each (attack, SNR) pair:
      1. Generate adversarial examples using torchattacks with minmax normalization
      2. For each of the 9 defenses, apply defense and compute accuracy

    Results are saved to:
      <cfg.result_dir>/defense_compare/defense_compare.csv   (full matrix)
      <cfg.result_dir>/defense_compare/defense_compare_<attack>.csv  (per-attack pivot)

    Args:
        model:       AWN model in eval mode
        sig_test:    Test signals tensor [N, 2, T] (raw IQ scale)
        lab_test:    Test labels tensor [N] (int)
        SNRs:        SNR array for all signals (length = total test dataset, not just test_idx)
        test_idx:    Indices into SNRs that correspond to test set rows
        cfg:         Config object with attack/defense parameters
        logger:      Python logger
        detector:    RFSignalAutoEncoder or None (loaded from cfg.detector_ckpt if None)
        attacks:     Subset of ATTACKS to run (default: all 5)
        snr_points:  Subset of SNR_POINTS to evaluate (default: all 10)
        max_per_cell: Max samples per modulation class at each SNR (D-04: default 200)
        batch_size:  Attack batch size (default: 64)

    Returns:
        pd.DataFrame with columns: attack, snr, defense, accuracy, n_samples
    """
    if attacks is None:
        attacks = list(ATTACKS)
    if snr_points is None:
        snr_points = list(SNR_POINTS)

    device = getattr(cfg, 'device', torch.device('cpu'))
    model = model.to(device)
    model.eval()

    # --- Load detector if not provided but checkpoint path is set ---
    if detector is None and getattr(cfg, 'detector_ckpt', None) is not None:
        try:
            from util.detector import RFSignalAutoEncoder
            detector = RFSignalAutoEncoder().to(device)
            det_state = torch.load(
                cfg.detector_ckpt, map_location=device, weights_only=True
            )
            detector.load_state_dict(det_state)
            detector.eval()
            logger.info(f"Loaded detector from {cfg.detector_ckpt}")
        except Exception as e:
            logger.warning(f"Could not load detector from {cfg.detector_ckpt}: {e}")
            detector = None

    # --- Build output directory ---
    out_dir = os.path.join(cfg.result_dir, 'defense_compare')
    os.makedirs(out_dir, exist_ok=True)

    # --- Set up Model01Wrapper for torchattacks (minmax normalization) ---
    wrapped_model = Model01Wrapper(model).to(device)

    # --- Pre-compute per-SNR test masks ---
    # test_SNRs[i] = SNR of the i-th element in the test set (row in sig_test)
    test_SNRs = np.array([SNRs[i] for i in test_idx])
    lab_test_np = lab_test.cpu().numpy() if isinstance(lab_test, torch.Tensor) else np.array(lab_test)

    results = []

    for attack_name in attacks:
        logger.info(f"=== Starting attack: {attack_name.upper()} ===")

        # Create attack object once per attack (wrapped model re-used)
        attack_obj = create_attack(attack_name, wrapped_model, cfg)

        for snr in snr_points:
            # --- Filter test data to this SNR ---
            snr_mask = test_SNRs == snr
            snr_indices = np.where(snr_mask)[0]  # indices into test set (sig_test, lab_test)

            if len(snr_indices) == 0:
                logger.warning(f"No test samples found for SNR={snr}, skipping.")
                continue

            # --- Sub-sample up to max_per_cell samples per modulation class (D-04) ---
            labels_at_snr = lab_test_np[snr_indices]
            unique_classes = np.unique(labels_at_snr)
            selected_indices = []
            for cls in unique_classes:
                cls_mask = labels_at_snr == cls
                cls_indices = snr_indices[cls_mask]
                # Cap at max_per_cell samples per modulation
                if len(cls_indices) > max_per_cell:
                    cls_indices = cls_indices[:max_per_cell]
                selected_indices.extend(cls_indices.tolist())
            selected_indices = np.array(selected_indices)

            if len(selected_indices) == 0:
                continue

            sigs_snr = sig_test[selected_indices].to(device)
            labs_snr = lab_test[selected_indices].to(device)
            labs_snr_np = labs_snr.cpu().numpy()

            # --- Generate adversarial examples in batches ---
            n_samples = len(selected_indices)
            adv_batches = []

            for batch_start in range(0, n_samples, batch_size):
                batch_end = min(batch_start + batch_size, n_samples)
                batch_sig = sigs_snr[batch_start:batch_end]
                batch_lab = labs_snr[batch_start:batch_end]

                # Convert to torchattacks minmax input
                x_01, a, b = iq_to_ta_input_minmax(batch_sig)
                wrapped_model.set_minmax(a, b)

                try:
                    x_adv_01 = attack_obj(x_01, batch_lab)
                    x_adv = ta_output_to_iq_minmax(x_adv_01, a, b)
                finally:
                    wrapped_model.clear_minmax()

                adv_batches.append(x_adv.detach())

            x_adv_all = torch.cat(adv_batches, dim=0)

            # --- Apply each defense and compute accuracy ---
            for defense_name in DEFENSE_CONFIGS:
                preds = _apply_defense(
                    defense_name, x_adv_all, model, detector, cfg, logger
                )
                acc = accuracy_score(labs_snr_np, preds)

                logger.info(
                    f"Attack={attack_name} SNR={snr:+3d} Defense={defense_name:<20s} "
                    f"Acc={acc:.4f} N={n_samples}"
                )

                results.append({
                    'attack':    attack_name,
                    'snr':       snr,
                    'defense':   defense_name,
                    'accuracy':  acc,
                    'n_samples': n_samples,
                })

    # --- Build DataFrame ---
    df = pd.DataFrame(results)

    if df.empty:
        logger.warning("No results collected — DataFrame is empty.")
        return df

    # --- Compute weighted average accuracy per (attack, defense) ---
    def _weighted_avg(group):
        weights = group['n_samples'].values
        accs = group['accuracy'].values
        return np.average(accs, weights=weights)

    wt_avg_rows = []
    for (atk, def_), grp in df.groupby(['attack', 'defense']):
        wt_avg_rows.append({
            'attack':    atk,
            'snr':       'weighted_avg',
            'defense':   def_,
            'accuracy':  _weighted_avg(grp),
            'n_samples': grp['n_samples'].sum(),
        })
    df_wt = pd.DataFrame(wt_avg_rows)
    df_full = pd.concat([df, df_wt], ignore_index=True)

    # --- Save full CSV ---
    full_csv_path = os.path.join(out_dir, 'defense_compare.csv')
    df_full.to_csv(full_csv_path, index=False)
    logger.info(f"Saved full results to {full_csv_path}")

    # --- Save per-attack pivot tables (D-09) ---
    for attack_name in attacks:
        df_atk = df_full[df_full['attack'] == attack_name].copy()
        if df_atk.empty:
            continue

        # Pivot: rows=defense, columns=snr (numeric + weighted_avg)
        pivot = df_atk.pivot_table(
            index='defense',
            columns='snr',
            values='accuracy',
            aggfunc='first',
        )
        # Reorder columns: SNR points first, then weighted_avg
        snr_cols = [s for s in (snr_points + ['weighted_avg']) if s in pivot.columns]
        pivot = pivot[snr_cols]

        # Reorder rows to match DEFENSE_CONFIGS key order
        ordered_defenses = [d for d in DEFENSE_CONFIGS if d in pivot.index]
        pivot = pivot.loc[ordered_defenses]

        pivot_path = os.path.join(out_dir, f'defense_compare_{attack_name}.csv')
        pivot.to_csv(pivot_path)
        logger.info(f"Saved {attack_name} pivot table to {pivot_path}")

        # Log best-performing defense per column
        best_per_col = pivot.idxmax(axis=0)
        best_str = ', '.join([f"SNR{c}={v}" for c, v in best_per_col.items()])
        logger.info(f"Best defense per column ({attack_name}): {best_str}")

    return df_full
