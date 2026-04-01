"""
util/defense_registry.py

Unified defense registry and pipeline for the real-time AMC defense framework.

Provides:
  - DEFENSE_REGISTRY: dict mapping string defense names to callable signal filters
  - defend(): unified detect->recover->classify pipeline with per-component latency breakdown
  - randomized_smoothing_predict(): majority-vote classifier wrapper (k=20, sigma=0.01)

Design decisions (from RESEARCH.md):
  D-05: DEFENSE_REGISTRY maps names to callables
  D-07: defend() signature is (x, model, detector, cfg) -> (predictions, latency_breakdown)
  D-08: k=20 noisy copies for randomized smoothing
  D-09: sigma=0.01 fixed (matches IQ signal scale)
  D-10: RS is a classifier wrapper, NOT a signal filter -- separate dispatch path
  D-11: GPU ops timed with torch.cuda.Event; CPU ops with time.perf_counter
  D-12: Per-sample latency (divide batch time by N)
  D-13: Warmup runs before measurement
"""

import time
import logging
import warnings
import functools
from typing import Optional

import torch
import torch.nn.functional as F

from util.defense import (
    fft_topk_denoise_normalized,
    spectral_gated_defense,
    normalize_iq_data,
    denormalize_iq_data,
)
from util.detector import normalize_for_detector, kl_divergence_timewise

__all__ = ['DEFENSE_REGISTRY', 'defend', 'randomized_smoothing_predict']

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Import classical filter baselines (Plan 01 -- may not exist yet)
# ---------------------------------------------------------------------------
try:
    from util.defense_baselines import (
        kalman_filter,
        wiener_filter,
        sg_filter,
        gaussian_filter,
        fir_filter,
    )
    _BASELINES_AVAILABLE = True
except ImportError:
    warnings.warn(
        "util.defense_baselines not found. Classical filter baselines (kalman, wiener, "
        "savitzky_golay, gaussian, fir) will be None in DEFENSE_REGISTRY. "
        "Run Plan 01 to generate util/defense_baselines.py.",
        ImportWarning,
        stacklevel=2,
    )
    kalman_filter = None
    wiener_filter = None
    sg_filter = None
    gaussian_filter = None
    fir_filter = None
    _BASELINES_AVAILABLE = False


# ---------------------------------------------------------------------------
# FFT defense wrappers (thin adapters matching registry callable signature)
# ---------------------------------------------------------------------------

def fft_topk_defense_wrapper(x: torch.Tensor, *, topk: int = 50) -> torch.Tensor:
    """
    Wraps fft_topk_denoise_normalized from util/defense.py.
    Operates in normalized domain: normalize -> Top-K -> denormalize.

    Args:
        x: Input tensor [N, 2, T] (raw IQ scale, ~±0.02)
        topk: Number of FFT bins to keep per channel (default: 50)

    Returns:
        Denoised tensor in original IQ scale.
    """
    return fft_topk_denoise_normalized(x, topk=topk)


def spectral_gated_wrapper(x: torch.Tensor, *, topk: int = 20) -> torch.Tensor:
    """
    Wraps spectral_gated_defense from util/defense.py.
    Routes each sample through Top-K (narrowband) or quantization (wideband).

    Args:
        x: Input tensor [N, 2, T] (raw IQ scale)
        topk: Number of FFT bins for narrowband path (default: 20)

    Returns:
        Defended tensor in original IQ scale.
    """
    return spectral_gated_defense(x, topk=topk)


# ---------------------------------------------------------------------------
# DEFENSE_REGISTRY
# ---------------------------------------------------------------------------

DEFENSE_REGISTRY: dict = {
    'kalman':         kalman_filter,
    'wiener':         wiener_filter,
    'savitzky_golay': sg_filter,
    'gaussian':       gaussian_filter,
    'fir':            fir_filter,
    'fft_topk':       fft_topk_defense_wrapper,
    'spectral_gated': spectral_gated_wrapper,
    'rand_smooth':    None,   # special case: classifier wrapper, not signal filter
}

# GPU-native defenses: time with torch.cuda.Event
_GPU_DEFENSES = {'gaussian', 'fir', 'fft_topk', 'spectral_gated'}

# CPU-resident defenses: time with time.perf_counter
_CPU_DEFENSES = {'kalman', 'wiener', 'savitzky_golay'}


# ---------------------------------------------------------------------------
# Warmup helper (D-13)
# ---------------------------------------------------------------------------

def _warmup(fn, *args, n: int = 3) -> None:
    """
    Run fn(*args) n times to warm up CUDA kernel compilation and JIT caches
    before starting latency measurement. Synchronizes GPU after warmup.
    """
    for _ in range(n):
        fn(*args)
    if torch.cuda.is_available():
        torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# Randomized smoothing (D-08, D-09, D-10)
# ---------------------------------------------------------------------------

def randomized_smoothing_predict(
    model,
    x: torch.Tensor,
    *,
    k: int = 20,
    sigma: float = 0.01,
) -> torch.Tensor:
    """
    Majority-vote randomized smoothing classifier wrapper (D-08/D-09/D-10).

    NOT a signal filter -- returns class predictions, not a denoised signal.
    This is the separate dispatch path for 'rand_smooth' in DEFENSE_REGISTRY.

    Args:
        model: AWN model, callable as model(x) -> (logits [N, C], regu_sum)
        x:     Input tensor [N, 2, T] (raw IQ scale)
        k:     Number of noisy copies for majority vote (default: 20, per D-08)
        sigma: Gaussian noise std dev (default: 0.01, per D-09)

    Returns:
        predictions: [N] int64 tensor of majority-vote predicted class indices
    """
    N, C, T = x.shape
    device = x.device

    def _forward_noisy(x_in: torch.Tensor) -> torch.Tensor:
        """Forward k noisy copies through model, return votes [N, num_classes]."""
        x_rep = x_in.unsqueeze(0).expand(k, -1, -1, -1).reshape(k * N, C, T)
        x_noisy = x_rep + torch.randn_like(x_rep) * sigma
        with torch.no_grad():
            logits, _ = model(x_noisy)  # [k*N, num_classes]
        num_classes = logits.shape[1]
        preds = logits.argmax(dim=1).reshape(k, N)  # [k, N]
        votes = torch.zeros(N, num_classes, device=device)
        # Accumulate votes via scatter_add
        ones = torch.ones(N, 1, device=device)
        for i in range(k):
            votes.scatter_add_(1, preds[i].unsqueeze(1), ones)
        return votes  # [N, num_classes]

    try:
        votes = _forward_noisy(x)
    except RuntimeError as e:
        if 'out of memory' in str(e).lower():
            # OOM fallback: process one noisy copy at a time, accumulate votes
            logger.warning(
                "Randomized smoothing OOM with k=%d, N=%d batch. "
                "Falling back to micro-batch (1 copy at a time).", k, N
            )
            torch.cuda.empty_cache()
            with torch.no_grad():
                logits_0, _ = model(x)  # single forward to get num_classes
            num_classes = logits_0.shape[1]
            votes = torch.zeros(N, num_classes, device=device)
            for _ in range(k):
                x_noisy = x + torch.randn_like(x) * sigma
                with torch.no_grad():
                    logits, _ = model(x_noisy)
                preds = logits.argmax(dim=1)  # [N]
                votes.scatter_add_(1, preds.unsqueeze(1),
                                   torch.ones(N, 1, device=device))
        else:
            raise

    return votes.argmax(dim=1)  # [N]


# ---------------------------------------------------------------------------
# Unified defense pipeline (D-07, D-11 through D-14)
# ---------------------------------------------------------------------------

def defend(
    x: torch.Tensor,
    model,
    detector,
    cfg,
) -> tuple:
    """
    Unified detect->recover->classify pipeline (D-07).

    Reads cfg.defense to dispatch to the appropriate defense strategy.
    Times each component separately (D-14):
      - 'detector_ms': AE detector inference (GPU event timing)
      - 'filter_ms':   Signal recovery / filter application
      - 'classifier_ms': AWN classifier inference (GPU event timing)
      - 'total_ms':    End-to-end wall time per sample

    Latency values are per-sample milliseconds (D-12).
    Warmup runs are performed before measurement (D-13).

    Args:
        x:        Input tensor [N, 2, T] (raw IQ scale ~±0.02)
        model:    AWN model callable: model(x) -> (logits [N, C], regu_sum)
        detector: RFSignalAutoEncoder (or None to skip detection gate)
        cfg:      Config object with at least:
                    cfg.defense (str): defense name from DEFENSE_REGISTRY
                  Optional fields:
                    cfg.detector_threshold (float): KL threshold for AE gate
                    cfg.def_topk (int): top-K for fft_topk / spectral_gated
                    cfg.rs_k (int): copies for rand_smooth (default 20)
                    cfg.rs_sigma (float): noise sigma for rand_smooth (default 0.01)

    Returns:
        (predictions, latency_breakdown)
          predictions:       [N] int64 tensor of predicted class indices
          latency_breakdown: dict with keys 'detector_ms', 'filter_ms',
                             'classifier_ms', 'total_ms' (all per-sample ms)
    """
    defense_name = getattr(cfg, 'defense', 'fft_topk')
    N = x.shape[0]
    latency_breakdown: dict = {
        'detector_ms':   0.0,
        'filter_ms':     0.0,
        'classifier_ms': 0.0,
        'total_ms':      0.0,
    }

    wall_start = time.perf_counter()

    # ------------------------------------------------------------------
    # Special path: Randomized Smoothing (D-10)
    # ------------------------------------------------------------------
    if defense_name == 'rand_smooth':
        rs_k     = int(getattr(cfg, 'rs_k', 20))
        rs_sigma = float(getattr(cfg, 'rs_sigma', 0.01))

        # Warmup (D-13): one noisy forward pass
        def _rs_call():
            return randomized_smoothing_predict(model, x, k=rs_k, sigma=rs_sigma)
        _warmup(_rs_call, n=3)

        if torch.cuda.is_available():
            t0 = torch.cuda.Event(enable_timing=True)
            t1 = torch.cuda.Event(enable_timing=True)
            t0.record()
            predictions = randomized_smoothing_predict(
                model, x, k=rs_k, sigma=rs_sigma
            )
            t1.record()
            torch.cuda.synchronize()
            total_gpu_ms = t0.elapsed_time(t1)
        else:
            t_cpu = time.perf_counter()
            predictions = randomized_smoothing_predict(
                model, x, k=rs_k, sigma=rs_sigma
            )
            total_gpu_ms = (time.perf_counter() - t_cpu) * 1000.0

        per_sample_ms = total_gpu_ms / N
        latency_breakdown['filter_ms']     = per_sample_ms
        latency_breakdown['classifier_ms'] = per_sample_ms
        latency_breakdown['total_ms']      = (time.perf_counter() - wall_start) * 1000.0 / N
        return predictions, latency_breakdown

    # ------------------------------------------------------------------
    # Standard path: signal filter -> classifier
    # ------------------------------------------------------------------
    x_filtered = x  # start with input; may be replaced by filter

    # --- Step 1: Detector gate (optional) ---
    has_detector   = detector is not None
    has_threshold  = hasattr(cfg, 'detector_threshold') and cfg.detector_threshold is not None
    topk_val       = int(getattr(cfg, 'def_topk', 50))

    if has_detector and has_threshold:
        threshold = float(cfg.detector_threshold)

        # Warmup detector
        def _det_call():
            x_norm = normalize_for_detector(x)
            return detector(x_norm)
        _warmup(_det_call, n=3)

        if torch.cuda.is_available():
            t0 = torch.cuda.Event(enable_timing=True)
            t1 = torch.cuda.Event(enable_timing=True)
            t0.record()

            x_norm = normalize_for_detector(x)
            with torch.no_grad():
                recon  = detector(x_norm)
            kl = kl_divergence_timewise(x_norm, recon)

            t1.record()
            torch.cuda.synchronize()
            latency_breakdown['detector_ms'] = t0.elapsed_time(t1) / N
        else:
            t_cpu = time.perf_counter()
            x_norm = normalize_for_detector(x)
            with torch.no_grad():
                recon  = detector(x_norm)
            kl = kl_divergence_timewise(x_norm, recon)
            latency_breakdown['detector_ms'] = (time.perf_counter() - t_cpu) * 1000.0 / N

        # Gate: only denoise samples above threshold
        to_denoise = kl > threshold
        if to_denoise.any():
            x_subset = x[to_denoise]
            if defense_name in DEFENSE_REGISTRY and DEFENSE_REGISTRY[defense_name] is not None:
                filter_fn = DEFENSE_REGISTRY[defense_name]
                x_filtered = x.clone()
                x_filtered[to_denoise] = _apply_filter(
                    filter_fn, x_subset, defense_name, cfg
                )
            # else: no filter; x_filtered stays as x
        # If no samples above threshold, x_filtered = x (no denoising needed)

        # Filter latency is already folded into detector gate timing above;
        # report it as combined for detector-gated path.
        latency_breakdown['filter_ms'] = 0.0  # included in detector_ms

    else:
        # No detector gate -- apply filter to all samples
        if defense_name in DEFENSE_REGISTRY and DEFENSE_REGISTRY[defense_name] is not None:
            filter_fn = DEFENSE_REGISTRY[defense_name]

            # Warmup filter
            _warmup(functools.partial(_apply_filter, filter_fn, x, defense_name, cfg), n=3)

            if defense_name in _GPU_DEFENSES and torch.cuda.is_available():
                t0 = torch.cuda.Event(enable_timing=True)
                t1 = torch.cuda.Event(enable_timing=True)
                t0.record()
                x_filtered = _apply_filter(filter_fn, x, defense_name, cfg)
                t1.record()
                torch.cuda.synchronize()
                latency_breakdown['filter_ms'] = t0.elapsed_time(t1) / N
            else:
                # CPU filter timing with perf_counter (D-11)
                t_cpu = time.perf_counter()
                x_filtered = _apply_filter(filter_fn, x, defense_name, cfg)
                latency_breakdown['filter_ms'] = (time.perf_counter() - t_cpu) * 1000.0 / N
        else:
            logger.warning(
                "Defense '%s' not found in DEFENSE_REGISTRY or is None. "
                "Running classifier on unfiltered input.", defense_name
            )

    # --- Step 2: Classifier inference ---
    # Warmup classifier
    def _cls_call():
        with torch.no_grad():
            return model(x_filtered)
    _warmup(_cls_call, n=3)

    if torch.cuda.is_available():
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        with torch.no_grad():
            logits, _ = model(x_filtered)
        t1.record()
        torch.cuda.synchronize()
        latency_breakdown['classifier_ms'] = t0.elapsed_time(t1) / N
    else:
        t_cpu = time.perf_counter()
        with torch.no_grad():
            logits, _ = model(x_filtered)
        latency_breakdown['classifier_ms'] = (time.perf_counter() - t_cpu) * 1000.0 / N

    predictions = logits.argmax(dim=1)
    latency_breakdown['total_ms'] = (time.perf_counter() - wall_start) * 1000.0 / N

    return predictions, latency_breakdown


# ---------------------------------------------------------------------------
# Internal helper: apply a registry filter with cfg-derived kwargs
# ---------------------------------------------------------------------------

def _apply_filter(
    filter_fn,
    x: torch.Tensor,
    defense_name: str,
    cfg,
) -> torch.Tensor:
    """
    Call the appropriate filter function with parameters extracted from cfg.
    Each filter has its own parameter names (from D-04 locked decisions).
    """
    if defense_name == 'fft_topk':
        topk = int(getattr(cfg, 'def_topk', 50))
        return filter_fn(x, topk=topk)

    elif defense_name == 'spectral_gated':
        topk = int(getattr(cfg, 'def_topk', 20))
        return filter_fn(x, topk=topk)

    elif defense_name == 'kalman':
        q = float(getattr(cfg, 'kalman_process_noise', 1e-4))
        r = float(getattr(cfg, 'kalman_meas_noise', 0.01))
        return filter_fn(x, process_noise=q, meas_noise=r)

    elif defense_name == 'wiener':
        noise      = float(getattr(cfg, 'wiener_noise', 0.01))
        filter_len = int(getattr(cfg, 'wiener_filter_len', 5))
        return filter_fn(x, noise=noise, filter_len=filter_len)

    elif defense_name == 'savitzky_golay':
        window_length = int(getattr(cfg, 'sg_window_length', 11))
        polyorder     = int(getattr(cfg, 'sg_polyorder', 3))
        return filter_fn(x, window_length=window_length, polyorder=polyorder)

    elif defense_name == 'gaussian':
        sigma = float(getattr(cfg, 'gaussian_sigma', 1.0))
        return filter_fn(x, sigma=sigma)

    elif defense_name == 'fir':
        cutoff  = float(getattr(cfg, 'fir_cutoff', 0.1))
        numtaps = int(getattr(cfg, 'fir_numtaps', 31))
        return filter_fn(x, cutoff=cutoff, numtaps=numtaps)

    else:
        # Unknown filter -- call with no extra args (pass-through if it accepts x only)
        return filter_fn(x)
