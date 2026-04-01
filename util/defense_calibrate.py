"""
util/defense_calibrate.py

Parameter calibration sweeps, latency benchmarking, and clean accuracy validation
for the classical filter defense baselines.

Provides:
  - PARAM_GRIDS: per-filter parameter grids for calibration sweeps (D-04)
  - calibrate_filter(): composite-score grid search for a single filter at one SNR (D-01/D-02/D-03)
  - run_calibration_sweep(): per-SNR calibration loop for all filters (D-03)
  - run_latency_benchmark(): per-component latency measurement with warmup (D-11/D-12/D-13/D-14)
  - validate_clean_accuracy(): clean accuracy drop check with 2% threshold (PIPE-03)

Design decisions (from RESEARCH.md):
  D-01: Auto-calibrate filter parameters on validation set
  D-02: Composite score = alpha * clean_acc + (1 - alpha) * defended_acc
  D-03: Per-SNR calibration — separate optimal parameters at each SNR point
  D-04: Parameter ranges per filter (Kalman, Wiener, SG, Gaussian, FIR)
  D-11: GPU ops timed with torch.cuda.Event; CPU ops with time.perf_counter
  D-12: Per-sample latency (divide by batch size)
  D-13: Warmup iterations before measurement (10 for GPU, 3 for CPU)
  D-14: Measure detector, filter, and classifier separately
  PIPE-03: Clean accuracy drop must be < 2%

Pitfall notes (from RESEARCH.md):
  - Pitfall 5: Limit val set to ~200 samples per SNR to keep CW tractable
  - Pitfall 3: SG window must be odd and > polyorder (enforced in defense_baselines.py)
"""

import json
import logging
import time
from pathlib import Path
from typing import Optional

import torch

from util.defense_registry import (
    DEFENSE_REGISTRY,
    randomized_smoothing_predict,
    _GPU_DEFENSES,
    _CPU_DEFENSES,
)

__all__ = [
    'calibrate_filter',
    'run_calibration_sweep',
    'run_latency_benchmark',
    'validate_clean_accuracy',
    'PARAM_GRIDS',
]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Parameter grids (D-04: locked decisions from CONTEXT.md)
# ---------------------------------------------------------------------------

PARAM_GRIDS: dict = {
    'kalman': [
        {'process_noise': q, 'meas_noise': r}
        for q in [1e-5, 1e-4, 1e-3, 5e-3]
        for r in [1e-3, 5e-3, 0.01, 0.05, 0.1]
    ],
    'wiener': [
        {'noise': n, 'filter_len': l}
        for n in [1e-3, 5e-3, 0.01, 0.05, 0.1]
        for l in [3, 5, 7, 11, 15]
    ],
    'savitzky_golay': [
        {'window_length': w, 'polyorder': p}
        for w in [5, 7, 11, 15, 21, 31]
        for p in [1, 2, 3]
        if p < w   # Pitfall 3: enforce p < w at grid construction
    ],
    'gaussian': [
        {'sigma': s}
        for s in [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
    ],
    'fir': [
        {'cutoff': c, 'numtaps': n}
        for c in [0.05, 0.1, 0.15, 0.2, 0.3]
        for n in [15, 31, 63]
    ],
}


# ---------------------------------------------------------------------------
# Internal timing helpers (D-11, D-13)
# ---------------------------------------------------------------------------

def _time_gpu_op(fn, *args, n_warmup: int = 10, n_measure: int = 50) -> float:
    """
    Time a GPU operation using torch.cuda.Event (D-11, D-13).

    Args:
        fn:        Callable to time; called as fn(*args).
        *args:     Arguments forwarded to fn.
        n_warmup:  Number of warmup calls before measurement starts (D-13).
        n_measure: Number of measurement iterations.

    Returns:
        Per-batch wall time in milliseconds (averaged over n_measure runs).
    """
    for _ in range(n_warmup):
        fn(*args)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_measure):
        fn(*args)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / n_measure  # ms per batch


def _time_cpu_op(fn, *args, n_warmup: int = 3, n_measure: int = 20) -> float:
    """
    Time a CPU operation using time.perf_counter (D-11, D-13).

    Args:
        fn:        Callable to time; called as fn(*args).
        *args:     Arguments forwarded to fn.
        n_warmup:  Number of warmup calls (D-13).
        n_measure: Number of measurement iterations.

    Returns:
        Per-batch wall time in milliseconds (averaged over n_measure runs).
    """
    for _ in range(n_warmup):
        fn(*args)
    t0 = time.perf_counter()
    for _ in range(n_measure):
        fn(*args)
    return (time.perf_counter() - t0) * 1000.0 / n_measure  # ms per batch


# ---------------------------------------------------------------------------
# calibrate_filter
# ---------------------------------------------------------------------------

def calibrate_filter(
    defense_name: str,
    param_grid: list,
    val_signals: torch.Tensor,
    val_labels: torch.Tensor,
    model,
    attack_fn,
    snr_value: int,
    alpha: float = 0.5,
) -> tuple:
    """
    Grid search over param_grid to find best parameters for defense_name at one SNR.

    For each parameter set in param_grid (D-01/D-02/D-03):
      1. Apply filter to clean val signals → compute clean_acc.
      2. Generate adversarial examples via attack_fn.
      3. Apply filter to adversarial → compute defended_acc.
      4. composite score = alpha * clean_acc + (1 - alpha) * defended_acc.

    Per Pitfall 5 (RESEARCH.md): limit val set to 200 samples per SNR call to
    keep CW attack tractable. The caller should pass a pre-sliced val_signals.

    Args:
        defense_name:  Key in DEFENSE_REGISTRY (e.g., 'kalman', 'gaussian').
        param_grid:    List of dicts with parameter kwargs for the filter function.
        val_signals:   Tensor [N, 2, T] of validation IQ signals at this SNR.
                       Limit to ≤200 samples to keep sweep tractable.
        val_labels:    Tensor [N] of integer class labels.
        model:         AWN model callable: model(x) -> (logits [N, C], regu_sum).
        attack_fn:     Callable attack_fn(x, y) -> x_adv [N, 2, T].
        snr_value:     SNR in dB (used only for logging).
        alpha:         Weight for clean accuracy in composite score (default 0.5).
                       score = alpha * clean_acc + (1 - alpha) * defended_acc.

    Returns:
        (best_params, best_score) where best_params is the param dict and
        best_score is the composite float score in [0, 1].
    """
    filter_fn = DEFENSE_REGISTRY.get(defense_name)
    if filter_fn is None:
        raise ValueError(
            f"Defense '{defense_name}' not found in DEFENSE_REGISTRY or is None. "
            f"Available: {list(DEFENSE_REGISTRY.keys())}"
        )

    device = val_signals.device
    val_labels = val_labels.to(device)

    best_score = -1.0
    best_params: dict = {}

    logger.debug(
        "Calibrating '%s' at SNR=%d dB — %d param sets, %d val samples",
        defense_name, snr_value, len(param_grid), val_signals.shape[0]
    )

    for params in param_grid:
        # --- Clean accuracy with these params ---
        with torch.no_grad():
            x_filtered = filter_fn(val_signals, **params)
            logits, _ = model(x_filtered)
        clean_acc = (logits.argmax(1) == val_labels).float().mean().item()

        # --- Adversarial accuracy with these params ---
        x_adv = attack_fn(val_signals, val_labels)
        with torch.no_grad():
            x_adv_filtered = filter_fn(x_adv, **params)
            adv_logits, _ = model(x_adv_filtered)
        defended_acc = (adv_logits.argmax(1) == val_labels).float().mean().item()

        # Composite score (D-02)
        score = alpha * clean_acc + (1.0 - alpha) * defended_acc

        if score > best_score:
            best_score = score
            best_params = params

        logger.debug(
            "  %s  clean=%.3f  defended=%.3f  score=%.3f",
            params, clean_acc, defended_acc, score
        )

    logger.info(
        "Best params for '%s' at SNR=%d dB: %s  (score=%.4f)",
        defense_name, snr_value, best_params, best_score
    )
    return best_params, best_score


# ---------------------------------------------------------------------------
# run_calibration_sweep
# ---------------------------------------------------------------------------

def run_calibration_sweep(
    model,
    dataset_loader,
    cfg,
    attack_fn,
    alpha: float = 0.5,
    max_samples_per_snr: int = 200,
) -> dict:
    """
    Per-SNR calibration sweep for all filters in PARAM_GRIDS (D-03).

    Loops over every SNR in the dataset and every filter in PARAM_GRIDS.
    Calls calibrate_filter() for each (SNR, filter) pair and stores results.
    Saves calibration results as JSON to cfg.result_dir/calibration_params.json.

    Args:
        model:               AWN model.
        dataset_loader:      PyTorch DataLoader or iterable yielding (signals, labels, SNRs).
                             Alternatively, pass a tuple (signals, labels, snr_list) directly.
        cfg:                 Config object with cfg.result_dir attribute.
        attack_fn:           Callable attack_fn(x, y) -> x_adv for adversarial evaluation.
                             Typically a torchattacks CW instance.
        alpha:               Composite score weight for clean accuracy (default 0.5).
        max_samples_per_snr: Max val samples per SNR cell to limit CW attack time (default 200).
                             Reduces sweep time from hours to minutes (Pitfall 5).

    Returns:
        results: dict {filter_name: {snr: best_params_dict}}
    """
    # Unpack dataset: accept (signals, labels, snrs) tuple or DataLoader
    if isinstance(dataset_loader, tuple) and len(dataset_loader) == 3:
        all_signals, all_labels, all_snrs = dataset_loader
    else:
        # Aggregate from DataLoader
        all_sigs_list, all_labs_list, all_snrs_list = [], [], []
        for batch in dataset_loader:
            if len(batch) == 3:
                sigs, labs, snrs = batch
            else:
                sigs, labs = batch
                snrs = torch.zeros(sigs.shape[0], dtype=torch.long)
            all_sigs_list.append(sigs)
            all_labs_list.append(labs)
            all_snrs_list.append(snrs)
        all_signals = torch.cat(all_sigs_list, dim=0)
        all_labels  = torch.cat(all_labs_list, dim=0)
        all_snrs    = torch.cat(all_snrs_list, dim=0)

    device = next(model.parameters()).device
    all_signals = all_signals.to(device)
    all_labels  = all_labels.to(device)

    if not isinstance(all_snrs, torch.Tensor):
        all_snrs = torch.tensor(all_snrs)
    all_snrs = all_snrs.to(device)

    unique_snrs = sorted(all_snrs.unique().cpu().tolist())

    results: dict = {name: {} for name in PARAM_GRIDS}

    logger.info(
        "Starting calibration sweep: %d SNRs × %d filters",
        len(unique_snrs), len(PARAM_GRIDS)
    )

    for snr in unique_snrs:
        snr_int = int(snr)
        mask = (all_snrs == snr_int)
        snr_signals = all_signals[mask][:max_samples_per_snr]
        snr_labels  = all_labels[mask][:max_samples_per_snr]

        if snr_signals.shape[0] == 0:
            logger.warning("No samples at SNR=%d dB — skipping.", snr_int)
            continue

        logger.info(
            "SNR=%+3d dB  |  %d samples (capped at %d)",
            snr_int, snr_signals.shape[0], max_samples_per_snr
        )

        for filter_name, param_grid in PARAM_GRIDS.items():
            if DEFENSE_REGISTRY.get(filter_name) is None:
                logger.warning(
                    "Defense '%s' is None in DEFENSE_REGISTRY — skipping calibration.",
                    filter_name
                )
                continue

            best_params, best_score = calibrate_filter(
                defense_name=filter_name,
                param_grid=param_grid,
                val_signals=snr_signals,
                val_labels=snr_labels,
                model=model,
                attack_fn=attack_fn,
                snr_value=snr_int,
                alpha=alpha,
            )
            results[filter_name][snr_int] = best_params
            logger.info(
                "  %-18s  SNR=%+3d dB  score=%.4f  params=%s",
                filter_name, snr_int, best_score, best_params
            )

    # Save calibration results to JSON
    out_path = Path(cfg.result_dir) / 'calibration_params.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info("Calibration results saved to %s", out_path)

    # Print summary table
    print("\n" + "=" * 72)
    print("  Calibration Sweep Results")
    print("=" * 72)
    print(f"  {'Filter':<18}  {'SNR':>5}  {'Best Params'}")
    print("-" * 72)
    for filter_name, snr_dict in results.items():
        for snr_int, params in sorted(snr_dict.items()):
            print(f"  {filter_name:<18}  {snr_int:>+4}  {params}")
    print("=" * 72)
    print(f"  Saved to: {out_path}")
    print("=" * 72 + "\n")

    return results


# ---------------------------------------------------------------------------
# run_latency_benchmark
# ---------------------------------------------------------------------------

def run_latency_benchmark(
    model,
    detector,
    cfg,
    batch_size: int = 32,
) -> dict:
    """
    Per-component latency benchmark for all defenses in DEFENSE_REGISTRY (D-11/D-12/D-13/D-14).

    For each defense:
      - GPU-native filters (gaussian, fir, fft_topk, spectral_gated): torch.cuda.Event timing,
        10 warmup + 50 measure iterations.
      - CPU filters (kalman, wiener, savitzky_golay): time.perf_counter timing,
        3 warmup + 20 measure iterations.
    Rand_smooth: time full randomized_smoothing_predict(model, x, k=20, sigma=0.01).
    Detector: time detector(normalize_for_detector(x)).
    Classifier: time model(x)[0].

    Per-sample latency = batch_time_ms / (batch_size * n_iters) or / batch_size
    (D-12: divide by batch size).

    Also benchmarks at batch_size=1 for single-sample latency.

    Args:
        model:      AWN model.
        detector:   RFSignalAutoEncoder (or None to skip detector benchmark).
        cfg:        Config object (used for device detection only).
        batch_size: Primary benchmark batch size (default 32).

    Returns:
        results: dict with keys 'batch{N}' -> {defense_name: per_sample_ms}
                 including 'batch1' for single-sample latency.
    """
    device = next(model.parameters()).device
    use_cuda = device.type == 'cuda' and torch.cuda.is_available()

    x32  = torch.randn(batch_size, 2, 128, device=device)
    x1   = torch.randn(1, 2, 128, device=device)

    def _per_sample_ms(batch_ms: float, bs: int) -> float:
        return batch_ms / bs

    results: dict = {}

    for tag, x_batch in [('batch32', x32), ('batch1', x1)]:
        bs = x_batch.shape[0]
        entry: dict = {}

        # --- Classifier baseline ---
        def _cls():
            with torch.no_grad():
                return model(x_batch)[0]

        if use_cuda:
            cls_ms = _time_gpu_op(_cls, n_warmup=10, n_measure=50)
        else:
            cls_ms = _time_cpu_op(_cls, n_warmup=3, n_measure=20)
        entry['classifier'] = _per_sample_ms(cls_ms, bs)

        # --- Detector baseline ---
        if detector is not None:
            from util.detector import normalize_for_detector

            def _det():
                with torch.no_grad():
                    x_norm = normalize_for_detector(x_batch)
                    return detector(x_norm)

            if use_cuda:
                det_ms = _time_gpu_op(_det, n_warmup=10, n_measure=50)
            else:
                det_ms = _time_cpu_op(_det, n_warmup=3, n_measure=20)
            entry['detector'] = _per_sample_ms(det_ms, bs)
        else:
            entry['detector'] = None

        # --- Signal filters ---
        for defense_name, filter_fn in DEFENSE_REGISTRY.items():
            if defense_name == 'rand_smooth':
                continue  # handled separately below
            if filter_fn is None:
                entry[defense_name] = None
                continue

            # Build a no-arg callable for timing helpers
            def _make_filter_call(fn, params):
                def _call():
                    return fn(x_batch, **params)
                return _call

            # Use default params (no calibration injection needed for latency benchmark)
            default_params = _get_default_params(defense_name)
            _filter_call = _make_filter_call(filter_fn, default_params)

            if defense_name in _GPU_DEFENSES and use_cuda:
                filt_ms = _time_gpu_op(_filter_call, n_warmup=10, n_measure=50)
            else:
                filt_ms = _time_cpu_op(_filter_call, n_warmup=3, n_measure=20)
            entry[defense_name] = _per_sample_ms(filt_ms, bs)

        # --- Randomized smoothing ---
        def _rs():
            return randomized_smoothing_predict(model, x_batch, k=20, sigma=0.01)

        if use_cuda:
            rs_ms = _time_gpu_op(_rs, n_warmup=10, n_measure=50)
        else:
            rs_ms = _time_cpu_op(_rs, n_warmup=3, n_measure=20)
        entry['rand_smooth'] = _per_sample_ms(rs_ms, bs)

        results[tag] = entry

    # Print formatted table
    _print_latency_table(results, batch_size)
    return results


def _get_default_params(defense_name: str) -> dict:
    """Return default parameter dict for a given defense (used during latency benchmark)."""
    defaults = {
        'kalman':          {'process_noise': 1e-4, 'meas_noise': 0.01},
        'wiener':          {'noise': 0.01, 'filter_len': 5},
        'savitzky_golay':  {'window_length': 11, 'polyorder': 3},
        'gaussian':        {'sigma': 1.0},
        'fir':             {'cutoff': 0.1, 'numtaps': 31},
        'fft_topk':        {'topk': 50},
        'spectral_gated':  {'topk': 20},
    }
    return defaults.get(defense_name, {})


def _print_latency_table(results: dict, batch_size: int) -> None:
    """Print a formatted latency comparison table to stdout."""
    print("\n" + "=" * 64)
    print("  Latency Benchmark (per-sample ms)")
    print("=" * 64)
    print(f"  {'Defense':<18}  {'batch=%d' % batch_size:>12}  {'batch=1':>12}")
    print("-" * 64)

    key32 = 'batch32'
    key1  = 'batch1'

    # Collect all defense names from results
    all_names = set()
    for v in results.values():
        all_names.update(v.keys())

    gpu_group = sorted(n for n in all_names if n in _GPU_DEFENSES or n == 'rand_smooth')
    cpu_group = sorted(n for n in all_names if n in _CPU_DEFENSES)
    other     = sorted(n for n in all_names if n not in gpu_group and n not in cpu_group)

    def _fmt(val):
        if val is None:
            return 'N/A'.rjust(12)
        return f'{val:.4f}'.rjust(12)

    print("  GPU-native:")
    for name in gpu_group:
        v32 = results.get(key32, {}).get(name)
        v1  = results.get(key1,  {}).get(name)
        print(f"    {name:<16}  {_fmt(v32)}  {_fmt(v1)}")

    print("  CPU-path:")
    for name in cpu_group:
        v32 = results.get(key32, {}).get(name)
        v1  = results.get(key1,  {}).get(name)
        print(f"    {name:<16}  {_fmt(v32)}  {_fmt(v1)}")

    if other:
        print("  Other:")
        for name in other:
            v32 = results.get(key32, {}).get(name)
            v1  = results.get(key1,  {}).get(name)
            print(f"    {name:<16}  {_fmt(v32)}  {_fmt(v1)}")

    print("=" * 64 + "\n")


# ---------------------------------------------------------------------------
# validate_clean_accuracy
# ---------------------------------------------------------------------------

def validate_clean_accuracy(
    model,
    detector,
    cfg,
    test_signals: torch.Tensor,
    test_labels: torch.Tensor,
    SNRs=None,
    test_idx=None,
    threshold_pct: float = 2.0,
) -> dict:
    """
    Validate that each defense does not degrade clean signal accuracy by more than
    threshold_pct percent (PIPE-03 requirement: <2% degradation on unperturbed signals).

    For each defense in DEFENSE_REGISTRY + 'rand_smooth':
      1. Set cfg.defense = defense_name.
      2. Run defend(test_signals, model, detector, cfg) -> predictions.
      3. Compute accuracy.
    Compare each defense accuracy to no-defense baseline.
    Flag any defense with drop > threshold_pct.

    Args:
        model:         AWN model.
        detector:      RFSignalAutoEncoder (or None to skip detector gating).
        cfg:           Config object with cfg.defense settable attribute.
        test_signals:  Tensor [N, 2, T] of clean test IQ signals.
        test_labels:   Tensor [N] of integer class labels.
        SNRs:          Optional array of per-sample SNR values (not used directly;
                       reserved for future per-SNR breakdown).
        test_idx:      Optional array of test indices (unused; kept for API compatibility).
        threshold_pct: Accuracy drop threshold in percentage points (default 2.0, PIPE-03).

    Returns:
        accuracy_by_defense: dict {defense_name: accuracy_float} including 'no_defense'.
    """
    from util.defense_registry import defend

    device = next(model.parameters()).device
    test_signals = test_signals.to(device)
    test_labels  = test_labels.to(device)

    # --- Baseline: no defense ---
    with torch.no_grad():
        logits, _ = model(test_signals)
    baseline_preds = logits.argmax(1)
    baseline_acc = (baseline_preds == test_labels).float().mean().item()

    accuracy_by_defense: dict = {'no_defense': baseline_acc}
    logger.info("Clean baseline accuracy (no defense): %.4f", baseline_acc)

    # --- Per-defense evaluation ---
    all_defenses = [k for k in DEFENSE_REGISTRY.keys()] + ['rand_smooth']

    # Preserve original cfg.defense so we can restore it after the loop
    original_defense = getattr(cfg, 'defense', 'fft_topk')

    for defense_name in all_defenses:
        if DEFENSE_REGISTRY.get(defense_name) is None and defense_name != 'rand_smooth':
            logger.warning(
                "Defense '%s' is None in DEFENSE_REGISTRY — skipping clean accuracy check.",
                defense_name
            )
            accuracy_by_defense[defense_name] = None
            continue

        cfg.defense = defense_name
        try:
            predictions, _ = defend(test_signals, model, detector, cfg)
            acc = (predictions == test_labels).float().mean().item()
        except Exception as e:
            logger.error(
                "Error running defense '%s' on clean signals: %s", defense_name, e
            )
            accuracy_by_defense[defense_name] = None
            continue

        accuracy_by_defense[defense_name] = acc
        drop = (baseline_acc - acc) * 100.0

        if drop > threshold_pct:
            logger.warning(
                "PIPE-03 VIOLATION: defense '%s' drops clean accuracy by %.2f%% "
                "(threshold: %.1f%%)  baseline=%.4f  defended=%.4f",
                defense_name, drop, threshold_pct, baseline_acc, acc
            )
        else:
            logger.info(
                "PIPE-03 OK: '%s'  drop=%.2f%%  defended=%.4f",
                defense_name, drop, acc
            )

    # Restore original defense
    cfg.defense = original_defense

    # Print comparison table
    _print_accuracy_table(accuracy_by_defense, baseline_acc, threshold_pct)

    return accuracy_by_defense


def _print_accuracy_table(
    accuracy_by_defense: dict,
    baseline_acc: float,
    threshold_pct: float,
) -> None:
    """Print a formatted clean accuracy comparison table to stdout."""
    print("\n" + "=" * 72)
    print(f"  Clean Accuracy Validation (PIPE-03: drop < {threshold_pct:.1f}%)")
    print("=" * 72)
    print(f"  {'Defense':<20}  {'Accuracy':>10}  {'Drop (pp)':>10}  {'Status':>10}")
    print("-" * 72)
    for name, acc in accuracy_by_defense.items():
        if acc is None:
            print(f"  {name:<20}  {'N/A':>10}  {'N/A':>10}  {'SKIPPED':>10}")
            continue
        drop = (baseline_acc - acc) * 100.0
        status = 'FAIL' if name != 'no_defense' and drop > threshold_pct else 'OK'
        print(f"  {name:<20}  {acc:>10.4f}  {drop:>+10.2f}  {status:>10}")
    print("=" * 72 + "\n")
