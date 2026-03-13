import torch
from typing import Optional, Tuple
import torch


def _band_to_bins(T: int, f_low: float, f_high: float) -> Tuple[int, int]:
    """
    Convert normalized band edges in [0, 0.5] to one-sided rFFT bin indices.
    Returns inclusive (k_low, k_high).
    """
    F = T // 2 + 1
    # Map normalized frequency to nearest bin; clamp to valid range
    k_low = int(torch.ceil(torch.tensor(f_low * T)).item())
    k_high = int(torch.floor(torch.tensor(f_high * T)).item())
    k_low = max(0, min(k_low, F - 1))
    k_high = max(0, min(k_high, F - 1))
    if k_high < k_low:
        k_high = k_low
    return k_low, k_high


def fft_notch_denoise(x: torch.Tensor,
                      f_low: float,
                      f_high: float) -> torch.Tensor:
    """
    Remove a band of frequencies from each I/Q channel using rFFT + iRFFT.

    Args:
        x: Tensor of shape [N, 2, T] (I/Q real-valued channels)
        f_low: Lower edge in normalized digital frequency [0, 0.5]
        f_high: Upper edge in normalized digital frequency [0, 0.5]

    Returns:
        x_denoised: Tensor of same shape as x.
    """
    assert x.dim() == 3 and x.size(1) == 2, "Expected input of shape [N, 2, T] (I/Q)."
    N, C, T = x.shape
    device = x.device
    dtype = x.dtype

    k_low, k_high = _band_to_bins(T, f_low, f_high)
    F = T // 2 + 1

    # rFFT along time for each channel separately
    X_i = torch.fft.rfft(x[:, 0, :], n=T, dim=1)
    X_q = torch.fft.rfft(x[:, 1, :], n=T, dim=1)

    # Build binary mask that zeros out the [k_low, k_high] band
    mask = torch.ones(F, device=device, dtype=dtype)
    mask[k_low:k_high + 1] = 0.0

    X_i = X_i * mask[None, :]
    X_q = X_q * mask[None, :]

    yi = torch.fft.irfft(X_i, n=T, dim=1)
    yq = torch.fft.irfft(X_q, n=T, dim=1)

    y = torch.stack([yi, yq], dim=1)
    return y


def fft_mask_denoise(x: torch.Tensor,
                     keep_mask_1s: torch.Tensor) -> torch.Tensor:
    """
    Apply a one-sided frequency mask (length T//2+1) per sample/channel.

    keep_mask_1s: shape [F] or [N, F] with F=T//2+1, values in {0,1} (or [0,1]).
    """
    assert x.dim() == 3 and x.size(1) == 2, "Expected input of shape [N, 2, T] (I/Q)."
    N, C, T = x.shape
    device = x.device
    dtype = x.dtype
    F = T // 2 + 1

    if keep_mask_1s.dim() == 1:
        assert keep_mask_1s.numel() == F
        mask = keep_mask_1s.to(device=device, dtype=dtype)[None, :].repeat(N, 1)
    elif keep_mask_1s.dim() == 2:
        assert keep_mask_1s.shape[1] == F
        mask = keep_mask_1s.to(device=device, dtype=dtype)
        if mask.shape[0] == 1:
            mask = mask.repeat(N, 1)
        else:
            assert mask.shape[0] == N
    else:
        raise ValueError("keep_mask_1s must be rank-1 or rank-2")

    X_i = torch.fft.rfft(x[:, 0, :], n=T, dim=1)
    X_q = torch.fft.rfft(x[:, 1, :], n=T, dim=1)

    X_i = X_i * mask
    X_q = X_q * mask

    yi = torch.fft.irfft(X_i, n=T, dim=1)
    yq = torch.fft.irfft(X_q, n=T, dim=1)

    y = torch.stack([yi, yq], dim=1)
    return y


def _soft_notch_mask(F: int,
                     k_low: int,
                     k_high: int,
                     depth: float = 1.0,
                     trans: int = 3,
                     device=None,
                     dtype=None) -> torch.Tensor:
    """
    Build a soft notch mask of length F with attenuation `depth` (0..1) inside [k_low,k_high]
    and raised-cosine transitions of width `trans` bins.
    depth=1.0 means full suppression (0 gain), depth=0.0 means no suppression (1 gain).
    """
    g_in = 1.0 - float(depth)
    m = torch.ones(F, device=device, dtype=dtype)
    # Main flat bottom
    m[k_low:k_high + 1] = g_in
    # Left transition
    if trans > 0:
        for i in range(1, trans + 1):
            k = k_low - i
            if k < 0:
                break
            t = 0.5 * (1 + torch.cos(torch.tensor(i / (trans + 1) * torch.pi, device=device)))
            m[k] = t + (1 - t) * g_in
        # Right transition
        for i in range(1, trans + 1):
            k = k_high + i
            if k >= F:
                break
            t = 0.5 * (1 + torch.cos(torch.tensor(i / (trans + 1) * torch.pi, device=device)))
            m[k] = t + (1 - t) * g_in
    return m


def fft_soft_notch_denoise(x: torch.Tensor,
                           f_low: float,
                           f_high: float,
                           *,
                           depth: float = 1.0,
                           trans: int = 3) -> torch.Tensor:
    """
    Apply a soft (tapered) spectral notch in [f_low,f_high] with attenuation `depth` and
    raised-cosine transitions of width `trans` bins, then inverse RFFT.
    """
    assert x.dim() == 3 and x.size(1) == 2, "Expected input of shape [N, 2, T] (I/Q)."
    N, C, T = x.shape
    device = x.device
    dtype = x.dtype
    F = T // 2 + 1
    k_low, k_high = _band_to_bins(T, f_low, f_high)

    mask = _soft_notch_mask(F, k_low, k_high, depth=depth, trans=trans, device=device, dtype=dtype)

    X_i = torch.fft.rfft(x[:, 0, :], n=T, dim=1)
    X_q = torch.fft.rfft(x[:, 1, :], n=T, dim=1)
    X_i = X_i * mask[None, :]
    X_q = X_q * mask[None, :]
    yi = torch.fft.irfft(X_i, n=T, dim=1)
    yq = torch.fft.irfft(X_q, n=T, dim=1)
    return torch.stack([yi, yq], dim=1)


def fft_topk_denoise(x: torch.Tensor, topk: int) -> torch.Tensor:
    """
    Keep only the top-|k| FFT components (by magnitude) per sample/channel and
    reconstruct via inverse FFT. This mirrors the behavior used in AWN_All.py
    (full complex FFT, not rFFT), applied independently to I and Q.

    Args:
        x: Tensor of shape [N, 2, T] (I/Q real-valued channels)
        topk: Number of frequency bins to retain per channel (0 < topk <= T)

    Returns:
        x_denoised: Tensor of same shape as x.
    """
    assert x.dim() == 3 and x.size(1) == 2, "Expected input of shape [N, 2, T] (I/Q)."
    N, C, T = x.shape
    if topk is None or topk <= 0:
        return x
    k = min(int(topk), T)

    # Full complex FFT along time for both channels at once
    X = torch.fft.fft(x, n=T, dim=2)
    mags = X.abs()
    # Top-k indices per (N,C)
    _, idx = mags.topk(k=k, dim=2)
    mask = torch.zeros_like(mags, dtype=torch.bool)
    mask.scatter_(2, idx, True)
    X_filt = X * mask.to(X.dtype)
    y = torch.fft.ifft(X_filt, n=T, dim=2).real
    return y


def fft_topk_percent_denoise(x: torch.Tensor, percent: float) -> torch.Tensor:
    """
    Variant of `fft_topk_denoise` where `percent` (0..1] controls how many bins to keep.
    For example, percent=0.1 keeps roughly 10% of FFT bins per channel.
    """
    assert 0.0 < percent <= 1.0, "percent must be in (0,1]."
    T = x.size(-1)
    topk = max(1, int(round(percent * T)))
    return fft_topk_denoise(x, topk)


def cumulative_energy_knee(x: torch.Tensor, threshold: float = 0.90) -> torch.Tensor:
    """
    Per-sample K where top-K FFT bins capture >= threshold of total energy.

    Computes the full complex FFT, sorts bins by energy descending, and finds
    the minimum K such that the top-K bins capture at least `threshold` fraction
    of total spectral energy. Returns the max across I/Q channels.

    Args:
        x: Tensor of shape [N, 2, T] (I/Q real-valued channels)
        threshold: Fraction of total energy to capture (default 0.90)

    Returns:
        knee: Int tensor of shape [N] with per-sample K values
    """
    assert x.dim() == 3 and x.size(1) == 2, "Expected input of shape [N, 2, T] (I/Q)."
    N, C, T = x.shape

    X = torch.fft.fft(x, n=T, dim=2)          # [N, 2, T]
    energy = X.abs() ** 2                       # [N, 2, T]
    sorted_e, _ = energy.sort(dim=2, descending=True)
    cumsum = sorted_e.cumsum(dim=2)             # [N, 2, T]
    total = energy.sum(dim=2, keepdim=True)     # [N, 2, 1]

    # Find first index where cumsum >= threshold * total, per (N, C)
    reached = cumsum >= threshold * total       # [N, 2, T] bool
    # argmax on bool returns first True index; if never reached, returns 0
    # Add 1 because K=index+1 (we need at least index+1 bins)
    knee_per_ch = reached.float().argmax(dim=2) + 1  # [N, 2]
    # Handle edge case: if threshold is never reached (e.g., all zeros), use T
    never_reached = ~reached.any(dim=2)         # [N, 2]
    knee_per_ch[never_reached] = T
    # Take max across I/Q channels
    knee = knee_per_ch.max(dim=1).values        # [N]
    return knee.int()


def fft_adaptive_topk_denoise(
    x: torch.Tensor,
    threshold: float = 0.90,
    k_candidates: list = None,
    k_min: int = 10,
    k_max: int = 50,
) -> tuple:
    """
    Per-sample adaptive Top-K FFT denoising.

    For each sample, computes the spectral energy knee (minimum K where top-K
    bins capture >= threshold of total energy), maps it to the smallest
    candidate >= knee, then applies fft_topk_denoise with that K.

    Args:
        x: Tensor of shape [N, 2, T] (I/Q real-valued channels)
        threshold: Energy fraction threshold for knee detection (default 0.90)
        k_candidates: Sorted list of candidate K values (default [10, 15, 20, 30, 50])
        k_min: Minimum allowed K
        k_max: Maximum allowed K

    Returns:
        (x_denoised, selected_k) where selected_k is [N] int tensor of chosen K values
    """
    if k_candidates is None:
        k_candidates = [10, 15, 20, 30, 50]
    k_cands = sorted(k_candidates)

    knee = cumulative_energy_knee(x, threshold)  # [N]

    # Map knee to smallest candidate >= knee
    selected_k = torch.full_like(knee, k_cands[-1])
    for kc in reversed(k_cands):
        selected_k = torch.where(knee <= kc, kc, selected_k)
    selected_k = selected_k.clamp(k_min, k_max)

    # Group by K and batch-process for efficiency
    result = x.clone()
    for kc in k_cands:
        mask = (selected_k == kc)
        if mask.any():
            result[mask] = fft_topk_denoise(x[mask], topk=kc)
    return result, selected_k


def fft_adaptive_topk_denoise_normalized(
    x: torch.Tensor,
    threshold: float = 0.90,
    k_candidates: list = None,
    k_min: int = 10,
    k_max: int = 50,
    norm_offset: float = 0.02,
    norm_scale: float = 0.04,
    apply_in_normalized: bool = True,
) -> tuple:
    """
    Adaptive Top-K FFT denoising with normalization, matching the
    fft_topk_denoise_normalized pattern from AWN_All.py.

    Args:
        x: Input tensor [N, 2, T] (I/Q), assumed unnormalized
        threshold: Energy fraction threshold for knee detection
        k_candidates: List of candidate K values
        k_min: Minimum allowed K
        k_max: Maximum allowed K
        norm_offset: Normalization offset (default 0.02)
        norm_scale: Normalization scale (default 0.04)
        apply_in_normalized: If True, normalize -> denoise -> denormalize

    Returns:
        (x_denoised, selected_k) where x_denoised is in original scale
    """
    if apply_in_normalized:
        x_norm = normalize_iq_data(x, norm_offset, norm_scale)
        x_denoised_norm, selected_k = fft_adaptive_topk_denoise(
            x_norm, threshold, k_candidates, k_min, k_max
        )
        return denormalize_iq_data(x_denoised_norm, norm_offset, norm_scale), selected_k
    else:
        return fft_adaptive_topk_denoise(x, threshold, k_candidates, k_min, k_max)


def highpass_diff(x: torch.Tensor, order: int = 1) -> torch.Tensor:
    """
    Simple time-domain high-pass via finite differences per I/Q channel.
    order=1 applies y[n]=x[n]-x[n-1]; order=2 applies second difference.
    """
    assert x.dim() == 3 and x.size(1) == 2, "Expected input of shape [N, 2, T] (I/Q)."
    if order not in (1, 2):
        raise ValueError("order must be 1 or 2")
    N, C, T = x.shape
    device = x.device
    dtype = x.dtype

    # Build kernels for depthwise conv1d: groups=2
    if order == 1:
        k = torch.tensor([1.0, -1.0], device=device, dtype=dtype).view(1, 1, 2)
    else:
        k = torch.tensor([1.0, -2.0, 1.0], device=device, dtype=dtype).view(1, 1, 3)
    kernel = torch.cat([k, k], dim=0)  # [2,1,K]
    pad = kernel.size(-1) - 1
    x_padded = torch.nn.functional.pad(x, (pad, 0), mode="replicate")
    y = torch.nn.functional.conv1d(x_padded, kernel, groups=2)
    return y


def auto_soft_notch_denoise(
    x: torch.Tensor,
    *,
    fmax: float = 0.08,
    ref_band: Tuple[float, float] = (0.15, 0.5),
    tau: float = 2.0,
    max_width_bins: int = 3,
    depth_max: float = 0.8,
    trans: int = 4,
) -> torch.Tensor:
    """
    Adaptive soft notch based on per-sample spectral peak in low band.

    - Detect the strongest bin in [0, fmax] using combined I/Q magnitude.
    - Estimate baseline from median magnitude in ref_band.
    - If peak/baseline > tau, apply a soft notch centered at that bin with width up to
      `max_width_bins` and depth scaled up to `depth_max`.
    - Apply same mask to I and Q, then inverse RFFT to recover.
    """
    assert x.dim() == 3 and x.size(1) == 2, "Expected input of shape [N, 2, T] (I/Q)."
    N, C, T = x.shape
    device = x.device
    dtype = x.dtype
    F = T // 2 + 1

    # RFFTs per channel
    X_i = torch.fft.rfft(x[:, 0, :], n=T, dim=1)
    X_q = torch.fft.rfft(x[:, 1, :], n=T, dim=1)
    mag = torch.sqrt((X_i.abs() ** 2) + (X_q.abs() ** 2))  # [N, F]

    # Band indices
    k_max = max(0, min(int(torch.floor(torch.tensor(fmax * T)).item()), F - 1))
    k_ref_low, k_ref_high = _band_to_bins(T, ref_band[0], ref_band[1])
    if k_ref_high <= k_ref_low:
        k_ref_low, k_ref_high = max(0, k_ref_low - 1), min(F - 1, k_ref_high + 1)

    # Baseline and peaks
    eps = 1e-12
    baseline = torch.median(mag[:, k_ref_low:k_ref_high + 1], dim=1).values.clamp_min(eps)  # [N]
    # Peak in low band (exclude DC only if available)
    search_slice = mag[:, 0:k_max + 1]
    peak_vals, peak_idx = torch.max(search_slice, dim=1)  # [N]
    ratio = (peak_vals / baseline).clamp_min(0.0)

    # Build per-sample masks
    masks = torch.ones(N, F, device=device, dtype=dtype)
    for n in range(N):
        if ratio[n] <= tau:
            continue  # no notch needed
        k0 = int(peak_idx[n].item())
        # Width scales mildly with ratio
        width = min(max_width_bins, max(1, int((ratio[n].item() - tau) * 1.0) + 1))
        k_low = max(0, k0 - width)
        k_high = min(F - 1, k0 + width)
        # Depth scales up to depth_max
        depth = max(0.0, min(depth_max, (ratio[n].item() - tau) / max(tau, 1e-6) * depth_max))
        m = _soft_notch_mask(F, k_low, k_high, depth=depth, trans=trans, device=device, dtype=dtype)
        masks[n, :] = masks[n, :] * m  # combine (could support multiple peaks later)

    # Apply masks
    X_i = X_i * masks
    X_q = X_q * masks
    yi = torch.fft.irfft(X_i, n=T, dim=1)
    yq = torch.fft.irfft(X_q, n=T, dim=1)
    return torch.stack([yi, yq], dim=1)


def dc_detrend(x: torch.Tensor) -> torch.Tensor:
    """
    Remove per-channel DC by subtracting the time mean. Preserves shape.
    """
    mean = x.mean(dim=2, keepdim=True)
    return x - mean


def normalize_iq_data(x: torch.Tensor,
                      offset: float = 0.02,
                      scale: float = 0.04) -> torch.Tensor:
    """
    Normalize I/Q data as in AWN_All.py: (x + offset) / scale

    Args:
        x: Tensor of shape [N, 2, T] (I/Q)
        offset: Normalization offset (default 0.02)
        scale: Normalization scale (default 0.04)

    Returns:
        Normalized tensor same shape as x
    """
    return (x + offset) / scale


def denormalize_iq_data(x: torch.Tensor,
                        offset: float = 0.02,
                        scale: float = 0.04) -> torch.Tensor:
    """
    Reverse normalization: x * scale - offset

    Args:
        x: Normalized tensor of shape [N, 2, T] (I/Q)
        offset: Normalization offset (default 0.02)
        scale: Normalization scale (default 0.04)

    Returns:
        Denormalized tensor same shape as x
    """
    return x * scale - offset


def fft_topk_denoise_normalized(
    x: torch.Tensor,
    topk: int,
    norm_offset: float = 0.02,
    norm_scale: float = 0.04,
    apply_in_normalized: bool = True
) -> torch.Tensor:
    """
    Apply FFT top-K denoising with optional normalization before/after,
    matching the AWN_All.py recovery pattern.

    Args:
        x: Input tensor [N, 2, T] (I/Q), assumed to be unnormalized
        topk: Number of FFT components to keep per channel
        norm_offset: Normalization offset
        norm_scale: Normalization scale
        apply_in_normalized: If True, normalize -> denoise -> denormalize

    Returns:
        Denoised tensor in original scale
    """
    if apply_in_normalized:
        x_norm = normalize_iq_data(x, norm_offset, norm_scale)
        x_denoised_norm = fft_topk_denoise(x_norm, topk)
        return denormalize_iq_data(x_denoised_norm, norm_offset, norm_scale)
    else:
        return fft_topk_denoise(x, topk)
