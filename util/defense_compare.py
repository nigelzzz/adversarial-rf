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
from typing import Dict, List, Optional, Any
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import accuracy_score, confusion_matrix

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

__all__ = [
    'run_defense_compare',
    'generate_confusion_matrices',
    'generate_budget_curves',
    'ATTACKS',
    'SNR_POINTS',
    'DEFENSE_CONFIGS',
    'CONFMAT_ATTACKS',
    'CONFMAT_SNRS',
    'LINF_EPSILONS',
    'OPT_C_VALUES',
    'LINF_ATTACKS',
    'OPT_ATTACKS',
    '_CALIB_TO_CFG',
    '_set_calibrated_params',
    '_restore_cfg_params',
]

# ---------------------------------------------------------------------------
# Module-level constants (D-01, D-02, D-03)
# ---------------------------------------------------------------------------

ATTACKS = ['cw', 'eadl1', 'eaden', 'fgsm', 'pgd']

SNR_POINTS = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]

# Perturbation budget curve constants (D-06, D-07, D-08)
LINF_EPSILONS = [0.01, 0.03, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
OPT_C_VALUES = [0.01, 0.1, 1.0, 10.0]
LINF_ATTACKS = ['fgsm', 'pgd']
OPT_ATTACKS = ['cw', 'eadl1', 'eaden']

# Confusion matrix generation scope (D-12, D-13)
CONFMAT_ATTACKS = ['cw', 'eadl1', 'eaden']   # 3 optimization attacks
CONFMAT_SNRS = [0, 10, 18]                    # 3 representative SNR points

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


# ---------------------------------------------------------------------------
# Calibration parameter helpers (Gap 4 fix)
# ---------------------------------------------------------------------------

# Mapping from calibration_params.json filter keys to cfg attribute names
_CALIB_TO_CFG = {
    'kalman':          {'process_noise': 'kalman_process_noise', 'meas_noise': 'kalman_meas_noise'},
    'wiener':          {'noise': 'wiener_noise', 'filter_len': 'wiener_filter_len'},
    'savitzky_golay':  {'window_length': 'sg_window_length', 'polyorder': 'sg_polyorder'},
    'gaussian':        {'sigma': 'gaussian_sigma'},
    'fir':             {'cutoff': 'fir_cutoff', 'numtaps': 'fir_numtaps'},
}


def _set_calibrated_params(cfg, calib_params: dict, defense_name: str, snr: int, logger) -> dict:
    """
    Set cfg attributes from calibration_params.json for a specific (defense, SNR).
    Returns dict of original values for restoration.

    If defense_name is not in calib_params or SNR not found, does nothing (falls through
    to cfg defaults, which is the existing behavior).
    """
    originals: dict = {}
    if defense_name not in calib_params:
        return originals
    snr_key = str(snr)  # JSON keys are strings
    if snr_key not in calib_params[defense_name]:
        return originals
    mapping = _CALIB_TO_CFG.get(defense_name, {})
    best = calib_params[defense_name][snr_key]
    for param_name, cfg_attr in mapping.items():
        if param_name in best:
            originals[cfg_attr] = getattr(cfg, cfg_attr, None)
            setattr(cfg, cfg_attr, best[param_name])
    return originals


def _restore_cfg_params(cfg, originals: dict):
    """Restore cfg attributes saved by _set_calibrated_params."""
    for attr, val in originals.items():
        if val is None:
            try:
                delattr(cfg, attr)
            except AttributeError:
                pass
        else:
            setattr(cfg, attr, val)


def _load_calib_params(calibration_path, logger) -> dict:
    """Load calibration_params.json; return empty dict on error or if path is None."""
    if calibration_path is None:
        return {}
    import json as _json
    if not os.path.isfile(calibration_path):
        if logger:
            logger.warning("Calibration file not found: %s — using cfg defaults", calibration_path)
        return {}
    try:
        with open(calibration_path, 'r') as f:
            calib = _json.load(f)
        if logger:
            logger.info("Loaded calibration params from %s (%d filters)", calibration_path, len(calib))
        return calib
    except Exception as exc:
        if logger:
            logger.warning("Could not load calibration params from %s: %s", calibration_path, exc)
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
    calibration_path: Optional[str] = None,
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

    # --- Load per-SNR calibrated filter parameters (Gap 4 fix) ---
    calib_params = _load_calib_params(calibration_path, logger)

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
                # Apply per-SNR calibrated params if available (Gap 4 fix)
                originals = _set_calibrated_params(cfg, calib_params, defense_name, snr, logger)
                try:
                    preds = _apply_defense(
                        defense_name, x_adv_all, model, detector, cfg, logger
                    )
                finally:
                    _restore_cfg_params(cfg, originals)
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


# ---------------------------------------------------------------------------
# Confusion matrix generation (EVAL-03, D-12 through D-16)
# ---------------------------------------------------------------------------

def generate_confusion_matrices(
    model: nn.Module,
    sig_test: torch.Tensor,
    lab_test: torch.Tensor,
    SNRs: np.ndarray,
    test_idx: np.ndarray,
    cfg,
    logger,
    detector=None,
    confmat_attacks: Optional[List[str]] = None,
    confmat_snrs: Optional[List[int]] = None,
    max_per_cell: int = 200,
    batch_size: int = 64,
    calibration_path: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """
    Generate 18 confusion matrices: 3 attacks x 3 SNRs x before/after defense (D-12, D-13, D-14).

    For each (attack, SNR) pair:
      - Before defense: classify adversarial examples directly (no filtering)
      - After defense:  apply unified FFT Top-K pipeline, then classify

    Each matrix is 11x11 (D-15), row-normalized to percentages (D-16).

    Saves to <cfg.result_dir>/defense_compare/confmat/:
      {attack}_snr{snr}_before.npy        — raw integer confusion matrix
      {attack}_snr{snr}_before_pct.csv    — row-normalized percentages with mod labels
      {attack}_snr{snr}_after.npy         — raw integer confusion matrix
      {attack}_snr{snr}_after_pct.csv     — row-normalized percentages with mod labels
      confmat_summary.csv                 — diagonal accuracy for all 18 matrices

    Args:
        model:           AWN model in eval mode
        sig_test:        Test signals tensor [N, 2, T] (raw IQ scale)
        lab_test:        Test labels tensor [N] (int)
        SNRs:            SNR array for all signals (full dataset, not just test_idx)
        test_idx:        Indices into SNRs that correspond to test set rows
        cfg:             Config object with attack/defense parameters
        logger:          Python logger
        detector:        RFSignalAutoEncoder or None
        confmat_attacks: Attacks to evaluate (default: CONFMAT_ATTACKS)
        confmat_snrs:    SNR points to evaluate (default: CONFMAT_SNRS)
        max_per_cell:    Max samples per modulation class at each SNR (D-04: default 200)
        batch_size:      Attack generation batch size (default: 64)

    Returns:
        Dict mapping '{attack}_snr{snr}_{before|after}' -> raw np.ndarray confusion matrix
    """
    if confmat_attacks is None:
        confmat_attacks = list(CONFMAT_ATTACKS)
    if confmat_snrs is None:
        confmat_snrs = list(CONFMAT_SNRS)

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

    # --- Build output directory: defense_compare/confmat/ ---
    confmat_dir = os.path.join(cfg.result_dir, 'defense_compare', 'confmat')
    os.makedirs(confmat_dir, exist_ok=True)

    # --- Load per-SNR calibrated filter parameters (Gap 4 fix) ---
    calib_params = _load_calib_params(calibration_path, logger)

    # --- Class label info (per D-15: 11x11 for RML2016.10a) ---
    mod_names = [
        k.decode() if isinstance(k, bytes) else str(k)
        for k in cfg.classes.keys()
    ]
    n_classes = len(mod_names)
    all_class_labels = list(range(n_classes))

    # --- Pre-compute per-SNR test masks ---
    test_SNRs = np.array([SNRs[i] for i in test_idx])
    lab_test_np = lab_test.cpu().numpy() if isinstance(lab_test, torch.Tensor) else np.array(lab_test)

    # --- Model01Wrapper for torchattacks (minmax normalization, D-05) ---
    wrapped_model = Model01Wrapper(model).to(device)

    results: Dict[str, np.ndarray] = {}
    summary_rows = []

    for attack_name in confmat_attacks:
        logger.info(f"=== Confmat: starting attack {attack_name.upper()} ===")

        attack_obj = create_attack(attack_name, wrapped_model, cfg)

        for snr in confmat_snrs:
            # --- Filter test data to this SNR ---
            snr_mask = test_SNRs == snr
            snr_indices = np.where(snr_mask)[0]

            if len(snr_indices) == 0:
                logger.warning(f"Confmat: no test samples at SNR={snr}, skipping.")
                continue

            # --- Sub-sample up to max_per_cell per modulation class (D-04) ---
            labels_at_snr = lab_test_np[snr_indices]
            unique_classes = np.unique(labels_at_snr)
            selected_indices = []
            for cls in unique_classes:
                cls_mask = labels_at_snr == cls
                cls_indices = snr_indices[cls_mask]
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

                x_01, a, b = iq_to_ta_input_minmax(batch_sig)
                wrapped_model.set_minmax(a, b)

                try:
                    x_adv_01 = attack_obj(x_01, batch_lab)
                    x_adv = ta_output_to_iq_minmax(x_adv_01, a, b)
                finally:
                    wrapped_model.clear_minmax()

                adv_batches.append(x_adv.detach())

            x_adv_all = torch.cat(adv_batches, dim=0)

            # ------------------------------------------------------------------
            # BEFORE defense: classify adversarial directly
            # ------------------------------------------------------------------
            with torch.no_grad():
                logits, _ = model(x_adv_all)
            preds_before = logits.argmax(dim=1).cpu().numpy()

            cm_before = confusion_matrix(
                labs_snr_np, preds_before, labels=all_class_labels
            )
            # Row-normalize to percentages (D-16)
            cm_before_pct = cm_before.astype(np.float64)
            row_sums = cm_before_pct.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            cm_before_pct = cm_before_pct / row_sums * 100.0

            before_acc = np.diag(cm_before).sum() / max(cm_before.sum(), 1)

            # Save raw .npy (D-16 pattern)
            before_npy = os.path.join(confmat_dir, f'{attack_name}_snr{snr}_before.npy')
            np.save(before_npy, cm_before)

            # Save row-normalized CSV
            before_csv = os.path.join(confmat_dir, f'{attack_name}_snr{snr}_before_pct.csv')
            df_before = pd.DataFrame(cm_before_pct, index=mod_names, columns=mod_names)
            df_before.index.name = 'true\\pred'
            df_before.to_csv(before_csv, float_format='%.2f')

            key_before = f'{attack_name}_snr{snr}_before'
            results[key_before] = cm_before

            # ------------------------------------------------------------------
            # AFTER defense: apply unified FFT Top-K pipeline (ae_fft_topk, D-14)
            # ------------------------------------------------------------------
            orig_defense = getattr(cfg, 'defense', 'none')
            orig_topk = getattr(cfg, 'def_topk', 50)

            cfg.defense = 'fft_topk'
            cfg.def_topk = int(getattr(cfg, 'def_topk', 50))

            try:
                with torch.no_grad():
                    preds_after_tensor, _ = defend(x_adv_all, model, detector, cfg)
                preds_after = preds_after_tensor.cpu().numpy()
            finally:
                # Restore original cfg values
                cfg.defense = orig_defense
                cfg.def_topk = orig_topk

            cm_after = confusion_matrix(
                labs_snr_np, preds_after, labels=all_class_labels
            )
            # Row-normalize to percentages (D-16)
            cm_after_pct = cm_after.astype(np.float64)
            row_sums = cm_after_pct.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            cm_after_pct = cm_after_pct / row_sums * 100.0

            after_acc = np.diag(cm_after).sum() / max(cm_after.sum(), 1)

            # Save raw .npy
            after_npy = os.path.join(confmat_dir, f'{attack_name}_snr{snr}_after.npy')
            np.save(after_npy, cm_after)

            # Save row-normalized CSV
            after_csv = os.path.join(confmat_dir, f'{attack_name}_snr{snr}_after_pct.csv')
            df_after = pd.DataFrame(cm_after_pct, index=mod_names, columns=mod_names)
            df_after.index.name = 'true\\pred'
            df_after.to_csv(after_csv, float_format='%.2f')

            key_after = f'{attack_name}_snr{snr}_after'
            results[key_after] = cm_after

            logger.info(
                f"Confmat {attack_name} SNR={snr:+3d}: "
                f"before_acc={before_acc:.4f} after_acc={after_acc:.4f} "
                f"N={n_samples}"
            )

            summary_rows.append({'attack': attack_name, 'snr': snr, 'condition': 'before',
                                  'accuracy': round(before_acc, 4), 'n_samples': n_samples})
            summary_rows.append({'attack': attack_name, 'snr': snr, 'condition': 'after',
                                  'accuracy': round(after_acc, 4), 'n_samples': n_samples})

    # --- Save summary CSV listing all 18 matrices with diagonal accuracy ---
    if summary_rows:
        df_summary = pd.DataFrame(summary_rows, columns=['attack', 'snr', 'condition', 'accuracy', 'n_samples'])
        summary_csv = os.path.join(confmat_dir, 'confmat_summary.csv')
        df_summary.to_csv(summary_csv, index=False)
        logger.info(f"Saved confmat summary to {summary_csv}")

    logger.info(f"Generated {len(results)} confusion matrices in {confmat_dir}")
    return results


# ---------------------------------------------------------------------------
# Perturbation budget curve generation (EVAL-04, D-06, D-07, D-08)
# ---------------------------------------------------------------------------

def generate_budget_curves(
    model: nn.Module,
    sig_test: torch.Tensor,
    lab_test: torch.Tensor,
    SNRs: np.ndarray,
    test_idx: np.ndarray,
    cfg,
    logger,
    detector=None,
    linf_epsilons: Optional[List[float]] = None,
    opt_c_values: Optional[List[float]] = None,
    target_snrs: Optional[List[int]] = None,
    max_per_cell: int = 200,
    batch_size: int = 64,
    calibration_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Generate perturbation budget curves for paper EVAL-04.

    Sweeps:
      - Linf epsilon for FGSM and PGD (8 points per D-06)
      - Optimization attack confidence c for CW, EAD-L1, EAD-EN (4 points per D-07, D-08)

    For each (attack, perturbation strength, SNR, defense) combination, runs the attack
    at that strength and measures accuracy under each of the 9 defenses.

    Saves to <cfg.result_dir>/defense_compare/budget_curves/:
      budget_curves_detail.csv  — full row-level data
      budget_curves_agg.csv     — weighted average accuracy per (attack, param_value, defense)
      budget_{attack}.csv       — pivot table: param_value rows x defense columns

    Args:
        model:          AWN model in eval mode
        sig_test:       Test signals tensor [N, 2, T] (raw IQ scale)
        lab_test:       Test labels tensor [N] (int)
        SNRs:           SNR array for all signals (full dataset length)
        test_idx:       Indices into SNRs for test set rows
        cfg:            Config object with attack/defense parameters
        logger:         Python logger
        detector:       RFSignalAutoEncoder or None
        linf_epsilons:  Epsilon values for Linf attacks (default: LINF_EPSILONS)
        opt_c_values:   Confidence c values for optimization attacks (default: OPT_C_VALUES)
        target_snrs:    SNR points to average over (default: SNR_POINTS)
        max_per_cell:   Max samples per modulation class per SNR point (D-04: default 200)
        batch_size:     Attack generation batch size (default: 64)

    Returns:
        pd.DataFrame with columns: attack, param_name, param_value, snr, defense, accuracy, n_samples
    """
    if linf_epsilons is None:
        linf_epsilons = list(LINF_EPSILONS)
    if opt_c_values is None:
        opt_c_values = list(OPT_C_VALUES)
    if target_snrs is None:
        target_snrs = list(SNR_POINTS)

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
    budget_dir = os.path.join(cfg.result_dir, 'defense_compare', 'budget_curves')
    os.makedirs(budget_dir, exist_ok=True)

    # --- Load per-SNR calibrated filter parameters (Gap 4 fix) ---
    calib_params = _load_calib_params(calibration_path, logger)

    # --- Set up Model01Wrapper for torchattacks (minmax normalization, D-05) ---
    wrapped_model = Model01Wrapper(model).to(device)

    # --- Pre-compute per-SNR test masks ---
    test_SNRs = np.array([SNRs[i] for i in test_idx])
    lab_test_np = lab_test.cpu().numpy() if isinstance(lab_test, torch.Tensor) else np.array(lab_test)

    results = []

    # ------------------------------------------------------------------
    # Helper: generate adversarial examples and evaluate all defenses
    # ------------------------------------------------------------------
    def _run_attack_snr(attack_name, attack_obj, snr):
        """
        Generate adversarial examples at the given SNR and evaluate all 9 defenses.
        Returns list of result dicts (one per defense), or empty list if no samples.
        """
        # Filter test data to this SNR
        snr_mask = test_SNRs == snr
        snr_indices = np.where(snr_mask)[0]

        if len(snr_indices) == 0:
            return []

        # Sub-sample up to max_per_cell per modulation class (D-04)
        labels_at_snr = lab_test_np[snr_indices]
        unique_classes = np.unique(labels_at_snr)
        selected_indices = []
        for cls in unique_classes:
            cls_mask = labels_at_snr == cls
            cls_indices = snr_indices[cls_mask]
            if len(cls_indices) > max_per_cell:
                cls_indices = cls_indices[:max_per_cell]
            selected_indices.extend(cls_indices.tolist())
        selected_indices = np.array(selected_indices)

        if len(selected_indices) == 0:
            return []

        sigs_snr = sig_test[selected_indices].to(device)
        labs_snr = lab_test[selected_indices].to(device)
        labs_snr_np = labs_snr.cpu().numpy()
        n_samples = len(selected_indices)

        # Generate adversarial examples in batches
        adv_batches = []
        for batch_start in range(0, n_samples, batch_size):
            batch_end = min(batch_start + batch_size, n_samples)
            batch_sig = sigs_snr[batch_start:batch_end]
            batch_lab = labs_snr[batch_start:batch_end]

            x_01, a, b = iq_to_ta_input_minmax(batch_sig)
            wrapped_model.set_minmax(a, b)

            try:
                x_adv_01 = attack_obj(x_01, batch_lab)
                x_adv = ta_output_to_iq_minmax(x_adv_01, a, b)
            finally:
                wrapped_model.clear_minmax()

            adv_batches.append(x_adv.detach())

        x_adv_all = torch.cat(adv_batches, dim=0)

        # Apply each defense and compute accuracy
        snr_results = []
        for defense_name in DEFENSE_CONFIGS:
            # Apply per-SNR calibrated params if available (Gap 4 fix)
            originals = _set_calibrated_params(cfg, calib_params, defense_name, snr, logger)
            try:
                preds = _apply_defense(
                    defense_name, x_adv_all, model, detector, cfg, logger
                )
            finally:
                _restore_cfg_params(cfg, originals)
            acc = accuracy_score(labs_snr_np, preds)
            snr_results.append({
                'attack':    attack_name,
                'snr':       snr,
                'defense':   defense_name,
                'accuracy':  acc,
                'n_samples': n_samples,
            })

        return snr_results

    # ------------------------------------------------------------------
    # Part A: Linf budget curves (D-06)
    # ------------------------------------------------------------------
    logger.info("=== Budget curves: Linf attacks (FGSM, PGD) ===")
    logger.info(f"Epsilon values: {linf_epsilons}")

    # Save original cfg.attack_eps
    orig_attack_eps = getattr(cfg, 'attack_eps', 0.03)

    try:
        for eps in linf_epsilons:
            cfg.attack_eps = eps
            logger.info(f"  Linf eps={eps:.4f}")

            for attack_name in LINF_ATTACKS:
                attack_obj = create_attack(attack_name, wrapped_model, cfg)

                for snr in target_snrs:
                    snr_results = _run_attack_snr(attack_name, attack_obj, snr)
                    for row in snr_results:
                        row['param_name'] = 'eps'
                        row['param_value'] = eps
                        results.append(row)
                        logger.info(
                            f"    Attack={attack_name} eps={eps:.4f} SNR={snr:+3d} "
                            f"Defense={row['defense']:<20s} Acc={row['accuracy']:.4f}"
                        )
    finally:
        cfg.attack_eps = orig_attack_eps

    # ------------------------------------------------------------------
    # Part B: Optimization attack budget curves (D-07, D-08)
    # ------------------------------------------------------------------
    logger.info("=== Budget curves: Optimization attacks (CW, EAD-L1, EAD-EN) ===")
    logger.info(f"c values: {opt_c_values}")

    # Save original cfg values for CW and EAD
    orig_cw_c = getattr(cfg, 'cw_c', 1.0)
    orig_ead_initial_const = getattr(cfg, 'ead_initial_const', 0.001)

    try:
        for c_val in opt_c_values:
            cfg.cw_c = c_val
            cfg.ead_initial_const = c_val
            logger.info(f"  Optimization c={c_val:.4f}")

            for attack_name in OPT_ATTACKS:
                attack_obj = create_attack(attack_name, wrapped_model, cfg)

                for snr in target_snrs:
                    snr_results = _run_attack_snr(attack_name, attack_obj, snr)
                    for row in snr_results:
                        row['param_name'] = 'c'
                        row['param_value'] = c_val
                        results.append(row)
                        logger.info(
                            f"    Attack={attack_name} c={c_val:.4f} SNR={snr:+3d} "
                            f"Defense={row['defense']:<20s} Acc={row['accuracy']:.4f}"
                        )
    finally:
        cfg.cw_c = orig_cw_c
        cfg.ead_initial_const = orig_ead_initial_const

    # ------------------------------------------------------------------
    # Build DataFrame and save outputs
    # ------------------------------------------------------------------
    df = pd.DataFrame(results)

    if df.empty:
        logger.warning("generate_budget_curves: no results collected — DataFrame is empty.")
        return df

    # Reorder columns for clarity
    col_order = ['attack', 'param_name', 'param_value', 'snr', 'defense', 'accuracy', 'n_samples']
    df = df[col_order]

    # Save full detail CSV
    detail_csv = os.path.join(budget_dir, 'budget_curves_detail.csv')
    df.to_csv(detail_csv, index=False)
    logger.info(f"Saved budget curve detail to {detail_csv}")

    # Compute aggregated budget curves: weighted average accuracy per (attack, param_name, param_value, defense)
    agg_rows = []
    for (atk, p_name, p_val, def_), grp in df.groupby(['attack', 'param_name', 'param_value', 'defense']):
        weights = grp['n_samples'].values
        accs = grp['accuracy'].values
        wt_acc = float(np.average(accs, weights=weights)) if weights.sum() > 0 else 0.0
        agg_rows.append({
            'attack':      atk,
            'param_name':  p_name,
            'param_value': p_val,
            'defense':     def_,
            'accuracy':    wt_acc,
        })
    df_agg = pd.DataFrame(agg_rows, columns=['attack', 'param_name', 'param_value', 'defense', 'accuracy'])
    agg_csv = os.path.join(budget_dir, 'budget_curves_agg.csv')
    df_agg.to_csv(agg_csv, index=False)
    logger.info(f"Saved aggregated budget curves to {agg_csv}")

    # Save per-attack pivot tables (param_value as rows, defenses as columns)
    all_attacks = LINF_ATTACKS + OPT_ATTACKS
    for attack_name in all_attacks:
        df_atk = df_agg[df_agg['attack'] == attack_name].copy()
        if df_atk.empty:
            continue

        try:
            pivot = df_atk.pivot_table(
                index='param_value',
                columns='defense',
                values='accuracy',
                aggfunc='first',
            )
            # Reorder defense columns to match DEFENSE_CONFIGS key order
            ordered_defenses = [d for d in DEFENSE_CONFIGS if d in pivot.columns]
            pivot = pivot[ordered_defenses]
            pivot.index.name = 'param_value'

            pivot_csv = os.path.join(budget_dir, f'budget_{attack_name}.csv')
            pivot.to_csv(pivot_csv)
            logger.info(f"Saved {attack_name} budget pivot table to {pivot_csv}")
        except Exception as e:
            logger.warning(f"Could not create pivot table for {attack_name}: {e}")

    # Log summary: first vs last perturbation strength for no_defense and ae_fft_topk
    for attack_name in all_attacks:
        df_atk = df_agg[df_agg['attack'] == attack_name]
        if df_atk.empty:
            continue
        param_values = sorted(df_atk['param_value'].unique())
        if len(param_values) < 2:
            continue
        first_val, last_val = param_values[0], param_values[-1]
        for defense_name in ['no_defense', 'ae_fft_topk']:
            rows_first = df_atk[
                (df_atk['param_value'] == first_val) & (df_atk['defense'] == defense_name)
            ]
            rows_last = df_atk[
                (df_atk['param_value'] == last_val) & (df_atk['defense'] == defense_name)
            ]
            if rows_first.empty or rows_last.empty:
                continue
            acc_first = rows_first['accuracy'].values[0]
            acc_last = rows_last['accuracy'].values[0]
            logger.info(
                f"Budget summary | {attack_name:<8s} {defense_name:<20s} "
                f"first({first_val})={acc_first:.4f} last({last_val})={acc_last:.4f}"
            )

    logger.info(f"Budget curves complete. Results in {budget_dir}")
    return df
