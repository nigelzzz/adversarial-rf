"""
Adaptive Top-K defense strategies that select K per-sample.

Three model-aware approaches beyond the pure signal-processing energy knee:
1. Confidence sweep: try K from small to large, stop when classifier is confident
2. Classify-then-filter: classify raw signal, look up K for predicted modulation
3. Spectral shape: use significant bin count to estimate spectral width -> K
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple, List

from util.defense import (
    fft_topk_denoise,
    normalize_iq_data,
    denormalize_iq_data,
)


# Default modulation -> K mapping derived from calibration experiments.
# K=10: spectrally compact (few bins capture the signal)
# K=20: moderate bandwidth (digital modulations with symbol shaping)
# K=50: wideband (AM-SSB, WBFM occupy most of the band)
DEFAULT_MOD_K_MAP = {
    'QAM64': 10, 'PAM4': 10, 'AM-DSB': 10,
    'GFSK': 20, 'BPSK': 20, 'QPSK': 20, 'CPFSK': 20, 'QAM16': 20, '8PSK': 20,
    'AM-SSB': 50, 'WBFM': 50,
}

DEFAULT_K_FALLBACK = 20


# ---------------------------------------------------------------------------
# Approach 1: Confidence Sweep
# ---------------------------------------------------------------------------

@torch.no_grad()
def confidence_sweep_topk_denoise(
    x: torch.Tensor,
    model: nn.Module,
    k_candidates: list = None,
    confidence_threshold: float = 0.8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Try K values from smallest to largest, stop when classifier is confident.

    For each sample, apply fft_topk_denoise with increasing K. At each step
    run the classifier and check max softmax probability. Accept the first K
    where confidence >= threshold.

    Args:
        x: Tensor [N, 2, T] (I/Q)
        model: AWN model returning (logits, regu_sum)
        k_candidates: Sorted K values to try (default [10, 20, 30, 50])
        confidence_threshold: Min softmax probability to accept (default 0.8)

    Returns:
        (x_denoised [N,2,T], selected_k [N] int tensor)
    """
    if k_candidates is None:
        k_candidates = [10, 20, 30, 50]
    k_cands = sorted(k_candidates)

    N = x.shape[0]
    device = x.device
    result = x.clone()
    selected_k = torch.full((N,), k_cands[-1], dtype=torch.int, device=device)
    decided = torch.zeros(N, dtype=torch.bool, device=device)

    model.eval()
    for kc in k_cands:
        undecided = ~decided
        if not undecided.any():
            break

        x_topk = fft_topk_denoise(x[undecided], topk=kc)
        logits, _ = model(x_topk)
        probs = torch.softmax(logits, dim=1)
        max_conf = probs.max(dim=1).values

        confident = max_conf >= confidence_threshold
        undecided_idx = torch.where(undecided)[0]
        newly_decided = undecided_idx[confident]

        result[newly_decided] = x_topk[confident]
        selected_k[newly_decided] = kc
        decided[newly_decided] = True

    # Remaining undecided samples: use largest K
    if (~decided).any():
        x_last = fft_topk_denoise(x[~decided], topk=k_cands[-1])
        result[~decided] = x_last

    return result, selected_k


# ---------------------------------------------------------------------------
# Approach 2: Classify-Then-Filter
# ---------------------------------------------------------------------------

@torch.no_grad()
def classify_then_filter_topk_denoise(
    x: torch.Tensor,
    model: nn.Module,
    cfg,
    mod_k_map: dict = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Two-pass defense: classify raw signal, look up K for predicted modulation.

    Pass 1: Run classifier on unfiltered signal to estimate modulation type.
    Pass 2: Apply fft_topk_denoise with the modulation-specific K.

    Even under attack, confusion is often within modulation families that share
    similar K requirements (e.g., BPSK confused with QPSK, both need K=20).

    Args:
        x: Tensor [N, 2, T] (I/Q)
        model: AWN model returning (logits, regu_sum)
        cfg: Config object with cfg.classes mapping
        mod_k_map: Dict mapping modulation name -> K value
                   (default: DEFAULT_MOD_K_MAP)

    Returns:
        (x_denoised [N,2,T], selected_k [N] int tensor)
    """
    if mod_k_map is None:
        mod_k_map = DEFAULT_MOD_K_MAP

    # Build class index -> modulation name mapping
    idx_to_mod = {}
    for k, v in cfg.classes.items():
        name = k.decode() if isinstance(k, bytes) else str(k)
        idx_to_mod[v] = name

    N = x.shape[0]
    device = x.device

    # Pass 1: classify raw signal
    model.eval()
    logits, _ = model(x)
    preds = logits.argmax(dim=1)  # [N]

    # Map predicted class -> K
    selected_k = torch.full((N,), DEFAULT_K_FALLBACK, dtype=torch.int, device=device)
    for i in range(N):
        mod_name = idx_to_mod.get(int(preds[i].item()), '')
        selected_k[i] = mod_k_map.get(mod_name, DEFAULT_K_FALLBACK)

    # Pass 2: apply per-sample K (batch by unique K for efficiency)
    result = x.clone()
    unique_ks = torch.unique(selected_k)
    for kc in unique_ks:
        mask = (selected_k == kc)
        if mask.any():
            result[mask] = fft_topk_denoise(x[mask], topk=int(kc.item()))

    return result, selected_k


# ---------------------------------------------------------------------------
# Approach 3: Spectral Shape (Significant Bin Count)
# ---------------------------------------------------------------------------

def significant_bin_count(x: torch.Tensor, sig_pct: float = 0.10) -> torch.Tensor:
    """
    Count FFT bins with magnitude above sig_pct * peak_magnitude per sample.

    Unlike cumulative energy knee, this measures spectral width relative to
    the peak — narrowband signals have few significant bins regardless of SNR,
    while wideband signals have many.

    Args:
        x: Tensor [N, 2, T] (I/Q)
        sig_pct: Fraction of peak magnitude (default 0.10 = 10%)

    Returns:
        bin_count: Int tensor [N], max across I/Q channels
    """
    assert x.dim() == 3 and x.size(1) == 2
    N, C, T = x.shape

    X = torch.fft.fft(x, n=T, dim=2)
    mag = X.abs()  # [N, 2, T]

    peak = mag.max(dim=2, keepdim=True).values  # [N, 2, 1]
    threshold = sig_pct * peak
    sig_mask = mag > threshold  # [N, 2, T]
    count_per_ch = sig_mask.sum(dim=2)  # [N, 2]
    count = count_per_ch.max(dim=1).values  # [N]
    return count.int()


def spectral_shape_topk_denoise(
    x: torch.Tensor,
    sig_pct: float = 0.10,
    k_candidates: list = None,
    k_min: int = 10,
    k_max: int = 50,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Adaptive Top-K based on spectral shape (significant bin count).

    Counts bins above sig_pct * peak, maps to smallest K candidate >= count.
    Narrowband signals get small K (aggressive filtering).
    Wideband signals get large K (preserve bandwidth).

    Args:
        x: Tensor [N, 2, T] (I/Q)
        sig_pct: Fraction of peak magnitude for significant bins (default 0.10)
        k_candidates: Sorted K candidates (default [10, 15, 20, 30, 50])
        k_min: Minimum K
        k_max: Maximum K

    Returns:
        (x_denoised [N,2,T], selected_k [N] int tensor)
    """
    if k_candidates is None:
        k_candidates = [10, 15, 20, 30, 50]
    k_cands = sorted(k_candidates)

    bin_count = significant_bin_count(x, sig_pct)  # [N]

    # Map bin_count to smallest candidate >= bin_count
    selected_k = torch.full_like(bin_count, k_cands[-1])
    for kc in reversed(k_cands):
        selected_k = torch.where(bin_count <= kc, kc, selected_k)
    selected_k = selected_k.clamp(k_min, k_max)

    # Group by K and batch-process
    result = x.clone()
    for kc in k_cands:
        mask = (selected_k == kc)
        if mask.any():
            result[mask] = fft_topk_denoise(x[mask], topk=kc)
    return result, selected_k


# ---------------------------------------------------------------------------
# Approach 4: Concentration + Distortion (pure signal-processing)
# ---------------------------------------------------------------------------

def concentration_distortion_topk_denoise(
    x: torch.Tensor,
    k_candidates: list = None,
    conc_thresholds: list = None,
    dist_thresholds: list = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Adaptive Top-K using spectral concentration AND reconstruction distortion.

    For each candidate K (smallest first):
      C_K = sum(top-K energy) / total_energy   (concentration)
      D_K = ||x - topK(x)||^2 / ||x||^2        (distortion)
    Accept smallest K where C_K >= conc_threshold AND D_K <= dist_threshold.
    If no K qualifies → reject → return raw x (no filtering), selected_k = 0.

    Args:
        x: Tensor [N, 2, T] (I/Q)
        k_candidates: Sorted K values to try (default [10, 20, 30])
        conc_thresholds: Min concentration per K (default [0.85, 0.80, 0.75])
        dist_thresholds: Max distortion per K (default [0.15, 0.20, 0.25])

    Returns:
        (x_denoised [N,2,T], selected_k [N] int tensor where 0=rejected)
    """
    if k_candidates is None:
        k_candidates = [10, 20, 30]
    if conc_thresholds is None:
        conc_thresholds = [0.85, 0.80, 0.75]
    if dist_thresholds is None:
        dist_thresholds = [0.15, 0.20, 0.25]

    k_cands = sorted(k_candidates)
    assert len(conc_thresholds) == len(k_cands)
    assert len(dist_thresholds) == len(k_cands)

    N, C, T = x.shape
    device = x.device

    # Step 1: FFT and total energy
    X = torch.fft.fft(x, n=T, dim=2)
    energy = X.abs() ** 2  # [N, 2, T]
    total_energy = energy.sum(dim=2, keepdim=True)  # [N, 2, 1]

    # Sort energy descending per (sample, channel)
    sorted_energy, _ = energy.sort(dim=2, descending=True)

    # Pre-compute signal power for distortion denominator
    x_power = (x ** 2).sum(dim=(1, 2))  # [N]
    x_power = x_power.clamp(min=1e-12)

    result = x.clone()  # default: raw (no filtering)
    selected_k = torch.zeros(N, dtype=torch.int, device=device)  # 0 = rejected
    decided = torch.zeros(N, dtype=torch.bool, device=device)

    for i, kc in enumerate(k_cands):
        undecided = ~decided
        if not undecided.any():
            break

        # Step 2: Concentration for this K
        topk_energy = sorted_energy[undecided, :, :kc].sum(dim=2)  # [M, 2]
        total_e = total_energy[undecided].squeeze(2)  # [M, 2]
        concentration = topk_energy / total_e.clamp(min=1e-12)  # [M, 2]
        # Use max across I/Q channels (if either channel is concentrated, OK)
        conc_max = concentration.max(dim=1).values  # [M]

        # Step 3: Reconstruct with this K
        x_topk = fft_topk_denoise(x[undecided], topk=kc)

        # Step 4: Distortion
        diff = x[undecided] - x_topk
        dist = (diff ** 2).sum(dim=(1, 2)) / x_power[undecided]  # [M]

        # Step 5: Decision
        accept = (conc_max >= conc_thresholds[i]) & (dist <= dist_thresholds[i])

        undecided_idx = torch.where(undecided)[0]
        newly_decided = undecided_idx[accept]

        result[newly_decided] = x_topk[accept]
        selected_k[newly_decided] = kc
        decided[newly_decided] = True

    # Step 6: Rejected samples keep raw x (result already = x.clone())
    return result, selected_k


# ---------------------------------------------------------------------------
# Normalized wrappers (for multi_attack_eval which uses normalized FFT domain)
# ---------------------------------------------------------------------------

def confidence_sweep_topk_denoise_normalized(
    x: torch.Tensor,
    model: nn.Module,
    k_candidates: list = None,
    confidence_threshold: float = 0.8,
    norm_offset: float = 0.02,
    norm_scale: float = 0.04,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Normalized wrapper for confidence sweep."""
    x_norm = normalize_iq_data(x, norm_offset, norm_scale)
    # Denoise candidates in normalized space, but run model on denormalized
    if k_candidates is None:
        k_candidates = [10, 20, 30, 50]
    k_cands = sorted(k_candidates)

    N = x.shape[0]
    device = x.device
    result = x_norm.clone()
    selected_k = torch.full((N,), k_cands[-1], dtype=torch.int, device=device)
    decided = torch.zeros(N, dtype=torch.bool, device=device)

    model.eval()
    for kc in k_cands:
        undecided = ~decided
        if not undecided.any():
            break

        x_topk_norm = fft_topk_denoise(x_norm[undecided], topk=kc)
        x_topk_raw = denormalize_iq_data(x_topk_norm, norm_offset, norm_scale)
        logits, _ = model(x_topk_raw)
        probs = torch.softmax(logits, dim=1)
        max_conf = probs.max(dim=1).values

        confident = max_conf >= confidence_threshold
        undecided_idx = torch.where(undecided)[0]
        newly_decided = undecided_idx[confident]

        result[newly_decided] = x_topk_norm[confident]
        selected_k[newly_decided] = kc
        decided[newly_decided] = True

    if (~decided).any():
        result[~decided] = fft_topk_denoise(x_norm[~decided], topk=k_cands[-1])

    return denormalize_iq_data(result, norm_offset, norm_scale), selected_k


def classify_then_filter_topk_denoise_normalized(
    x: torch.Tensor,
    model: nn.Module,
    cfg,
    mod_k_map: dict = None,
    norm_offset: float = 0.02,
    norm_scale: float = 0.04,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Normalized wrapper for classify-then-filter."""
    if mod_k_map is None:
        mod_k_map = DEFAULT_MOD_K_MAP

    idx_to_mod = {}
    for k, v in cfg.classes.items():
        name = k.decode() if isinstance(k, bytes) else str(k)
        idx_to_mod[v] = name

    N = x.shape[0]
    device = x.device

    # Classify on raw signal (model expects raw IQ)
    model.eval()
    logits, _ = model(x)
    preds = logits.argmax(dim=1)

    selected_k = torch.full((N,), DEFAULT_K_FALLBACK, dtype=torch.int, device=device)
    for i in range(N):
        mod_name = idx_to_mod.get(int(preds[i].item()), '')
        selected_k[i] = mod_k_map.get(mod_name, DEFAULT_K_FALLBACK)

    # Filter in normalized space
    x_norm = normalize_iq_data(x, norm_offset, norm_scale)
    result_norm = x_norm.clone()
    unique_ks = torch.unique(selected_k)
    for kc in unique_ks:
        mask = (selected_k == kc)
        if mask.any():
            result_norm[mask] = fft_topk_denoise(x_norm[mask], topk=int(kc.item()))

    return denormalize_iq_data(result_norm, norm_offset, norm_scale), selected_k


def spectral_shape_topk_denoise_normalized(
    x: torch.Tensor,
    sig_pct: float = 0.10,
    k_candidates: list = None,
    norm_offset: float = 0.02,
    norm_scale: float = 0.04,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Normalized wrapper for spectral shape."""
    x_norm = normalize_iq_data(x, norm_offset, norm_scale)
    x_denoised_norm, selected_k = spectral_shape_topk_denoise(
        x_norm, sig_pct, k_candidates,
    )
    return denormalize_iq_data(x_denoised_norm, norm_offset, norm_scale), selected_k


def concentration_distortion_topk_denoise_normalized(
    x: torch.Tensor,
    k_candidates: list = None,
    conc_thresholds: list = None,
    dist_thresholds: list = None,
    norm_offset: float = 0.02,
    norm_scale: float = 0.04,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Normalized wrapper for concentration+distortion."""
    x_norm = normalize_iq_data(x, norm_offset, norm_scale)
    x_denoised_norm, selected_k = concentration_distortion_topk_denoise(
        x_norm, k_candidates, conc_thresholds, dist_thresholds,
    )
    return denormalize_iq_data(x_denoised_norm, norm_offset, norm_scale), selected_k
