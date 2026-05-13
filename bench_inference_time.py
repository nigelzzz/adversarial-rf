"""
Standalone inference-time benchmark for adaptive-K and classical filter defenses.

Zero project dependencies: only torch, numpy, scipy.
All filter implementations are inlined so this script can run anywhere.

Filters benchmarked:
  Fixed-K FFT family
    fft_topk            -- keep top-K FFT bins per channel (K fixed)
    fft_notch           -- zero a frequency band
    fft_soft_notch      -- tapered notch

  Adaptive-K FFT family (per-sample K)
    energy_knee_topk    -- K = cumulative energy knee (e.g. 90% energy)
    magnitude_knee_topk -- K = first bin where |X[k]|/|X[0]| < 5%
    spectral_shape_topk -- K = count of bins above sig_pct * peak

  Classical baselines
    kalman              -- scalar constant-velocity Kalman smoother (CPU)
    wiener              -- scipy.signal.wiener (CPU)
    savgol              -- scipy.signal.savgol_filter (CPU)
    gaussian            -- depthwise F.conv1d Gaussian LPF (GPU-native)
    fir                 -- depthwise F.conv1d FIR LPF (GPU-native)

Usage:
    python bench_inference_time.py
    python bench_inference_time.py --batch_size 256 --time_len 128 --iters 200
    python bench_inference_time.py --device cpu
    python bench_inference_time.py --filters fft_topk,energy_knee_topk,gaussian
"""

import argparse
import time
from typing import Callable, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal import wiener as scipy_wiener
from scipy.signal import savgol_filter as scipy_savgol
from scipy.signal import firwin


# =============================================================================
# Fixed-K FFT family
# =============================================================================

def fft_topk(x: torch.Tensor, topk: int = 50) -> torch.Tensor:
    """Keep top-K FFT bins per (sample, channel)."""
    N, C, T = x.shape
    k = min(int(topk), T)
    X = torch.fft.fft(x, n=T, dim=2)
    mags = X.abs()
    _, idx = mags.topk(k=k, dim=2)
    mask = torch.zeros_like(mags, dtype=torch.bool)
    mask.scatter_(2, idx, True)
    X_filt = X * mask.to(X.dtype)
    return torch.fft.ifft(X_filt, n=T, dim=2).real


def fft_notch(x: torch.Tensor, f_low: float = 0.05, f_high: float = 0.25) -> torch.Tensor:
    """Zero a frequency band [f_low, f_high] (normalized 0..0.5)."""
    N, C, T = x.shape
    F_ = T // 2 + 1
    k_low = max(0, min(int(np.ceil(f_low * T)), F_ - 1))
    k_high = max(0, min(int(np.floor(f_high * T)), F_ - 1))
    if k_high < k_low:
        k_high = k_low
    X = torch.fft.rfft(x, n=T, dim=2)
    mask = torch.ones(F_, device=x.device, dtype=x.dtype)
    mask[k_low:k_high + 1] = 0.0
    X = X * mask[None, None, :]
    return torch.fft.irfft(X, n=T, dim=2)


def fft_soft_notch(x: torch.Tensor, f_low: float = 0.05, f_high: float = 0.25,
                   depth: float = 0.7, trans: int = 4) -> torch.Tensor:
    """Tapered (cosine) notch attenuator."""
    N, C, T = x.shape
    F_ = T // 2 + 1
    k_low = max(0, min(int(np.ceil(f_low * T)), F_ - 1))
    k_high = max(0, min(int(np.floor(f_high * T)), F_ - 1))
    if k_high < k_low:
        k_high = k_low
    mask = torch.ones(F_, device=x.device, dtype=x.dtype)
    mask[k_low:k_high + 1] = 1.0 - depth
    # cosine taper outside band
    for i in range(1, trans + 1):
        w = 0.5 * (1.0 - np.cos(np.pi * i / (trans + 1)))
        atten = 1.0 - depth * (1.0 - w)
        if k_low - i >= 0:
            mask[k_low - i] = min(mask[k_low - i].item(), atten)
        if k_high + i < F_:
            mask[k_high + i] = min(mask[k_high + i].item(), atten)
    X = torch.fft.rfft(x, n=T, dim=2)
    X = X * mask[None, None, :]
    return torch.fft.irfft(X, n=T, dim=2)


# =============================================================================
# Adaptive-K FFT family (per-sample K)
# =============================================================================

def _apply_grouped_topk(x: torch.Tensor, selected_k: torch.Tensor,
                       k_cands: List[int]) -> torch.Tensor:
    """Group samples by chosen K and apply fft_topk batched per group."""
    result = x.clone()
    for kc in k_cands:
        mask = (selected_k == kc)
        if mask.any():
            result[mask] = fft_topk(x[mask], topk=int(kc))
    return result


def energy_knee_topk(x: torch.Tensor, threshold: float = 0.90,
                     k_candidates: List[int] = None,
                     k_min: int = 10, k_max: int = 50) -> torch.Tensor:
    """Per-sample K = cumulative energy knee, snapped to candidate grid."""
    if k_candidates is None:
        k_candidates = [10, 15, 20, 30, 50]
    k_cands = sorted(k_candidates)
    N, C, T = x.shape

    X = torch.fft.fft(x, n=T, dim=2)
    energy = X.abs() ** 2
    sorted_e, _ = energy.sort(dim=2, descending=True)
    cumsum = sorted_e.cumsum(dim=2)
    total = energy.sum(dim=2, keepdim=True)
    reached = cumsum >= threshold * total
    knee_per_ch = reached.float().argmax(dim=2) + 1
    never = ~reached.any(dim=2)
    knee_per_ch[never] = T
    knee = knee_per_ch.max(dim=1).values.int()

    selected_k = torch.full_like(knee, k_cands[-1])
    for kc in reversed(k_cands):
        selected_k = torch.where(knee <= kc, kc, selected_k)
    selected_k = selected_k.clamp(k_min, k_max)
    return _apply_grouped_topk(x, selected_k, k_cands)


def magnitude_knee_topk(x: torch.Tensor, ratio_thresh: float = 0.05) -> torch.Tensor:
    """Per-sample K = first bin where |X[k]|/|X[0]| < ratio_thresh."""
    N, C, T = x.shape
    X = torch.fft.fft(x, dim=2)
    psd = (X.abs() ** 2).mean(dim=1)
    sorted_mag, _ = psd.sqrt().sort(dim=1, descending=True)
    peak = sorted_mag[:, 0:1].clamp(min=1e-12)
    ratio = sorted_mag / peak
    below = ratio < ratio_thresh
    knee = below.float().argmax(dim=1)
    never = ~below.any(dim=1)
    knee[never] = T
    knee = knee.clamp(min=1).int()

    # Group by unique K (potentially many distinct K values)
    result = x.clone()
    unique_ks = torch.unique(knee).tolist()
    for kc in unique_ks:
        mask = (knee == kc)
        if mask.any():
            result[mask] = fft_topk(x[mask], topk=int(kc))
    return result


def spectral_shape_topk(x: torch.Tensor, sig_pct: float = 0.10,
                        k_candidates: List[int] = None,
                        k_min: int = 10, k_max: int = 50) -> torch.Tensor:
    """Per-sample K = bin count above sig_pct * peak, snapped to candidate grid."""
    if k_candidates is None:
        k_candidates = [10, 15, 20, 30, 50]
    k_cands = sorted(k_candidates)
    N, C, T = x.shape

    X = torch.fft.fft(x, n=T, dim=2)
    mag = X.abs()
    peak = mag.max(dim=2, keepdim=True).values
    sig_mask = mag > (sig_pct * peak)
    count_per_ch = sig_mask.sum(dim=2)
    bin_count = count_per_ch.max(dim=1).values.int()

    selected_k = torch.full_like(bin_count, k_cands[-1])
    for kc in reversed(k_cands):
        selected_k = torch.where(bin_count <= kc, kc, selected_k)
    selected_k = selected_k.clamp(k_min, k_max)
    return _apply_grouped_topk(x, selected_k, k_cands)


# =============================================================================
# Adaptive-K v2 family (flatness-routed: quantize wideband, knee-Top-K narrowband)
# =============================================================================

def _per_sample_quantize(x: torch.Tensor, n_levels: int = 32) -> torch.Tensor:
    """Per-sample uniform quantization to n_levels discrete values."""
    N = x.shape[0]
    x_flat = x.flatten(1)
    x_min = x_flat.min(dim=1).values.view(N, 1, 1)
    x_max = x_flat.max(dim=1).values.view(N, 1, 1)
    rng = x_max - x_min + 1e-12
    x_norm = (x - x_min) / rng
    x_q = torch.round(x_norm * (n_levels - 1)) / (n_levels - 1)
    return x_q * rng + x_min


def _knee_from_fft_t(X_n: torch.Tensor, ratio_thresh: float = 0.05) -> torch.Tensor:
    """Sorted-magnitude knee from a precomputed complex FFT [M, 2, T]."""
    T = X_n.shape[2]
    psd = (X_n.abs() ** 2).mean(dim=1)
    sorted_mag, _ = psd.sqrt().sort(dim=1, descending=True)
    peak = sorted_mag[:, 0:1].clamp(min=1e-12)
    ratio = sorted_mag / peak
    below = ratio < ratio_thresh
    knee = below.float().argmax(dim=1)
    never = ~below.any(dim=1)
    knee[never] = T
    return knee.clamp(min=1).int()


def _topk_from_fft_t(X_n: torch.Tensor, knee: torch.Tensor) -> torch.Tensor:
    """Per-sample Top-K filter on precomputed FFT, grouped by unique K."""
    T = X_n.shape[2]
    out = X_n.clone()
    k_values = sorted(set(knee.cpu().tolist()))
    for k in k_values:
        m = (knee == k)
        if not m.any():
            continue
        Xs = X_n[m]
        mags = Xs.abs()
        _, idx = mags.topk(k=min(int(k), T), dim=2)
        fmask = torch.zeros_like(mags, dtype=torch.bool)
        fmask.scatter_(2, idx, True)
        out[m] = Xs * fmask.to(Xs.dtype)
    return torch.fft.ifft(out, n=T, dim=2).real


def adaptive_k_v2(x: torch.Tensor,
                  ratio_thresh: float = 0.05,
                  k_max: int = 20,
                  flatness_threshold: float = 0.4,
                  quant_levels: int = 32) -> torch.Tensor:
    """v2 fixed-cap: flatness routes wideband->quantize, narrowband->knee Top-K."""
    N, C, T = x.shape
    X = torch.fft.fft(x, n=T, dim=2)
    power = X.abs() ** 2 + 1e-20
    log_power = torch.log(power)
    geo = torch.exp(log_power.mean(dim=2))
    arith = power.mean(dim=2)
    flatness = (geo / (arith + 1e-12)).mean(dim=1)

    is_wide = flatness > flatness_threshold
    is_narrow = ~is_wide
    result = x.clone()
    if is_wide.any():
        result[is_wide] = _per_sample_quantize(x[is_wide], n_levels=quant_levels)
    if is_narrow.any():
        Xn = X[is_narrow]
        knee = _knee_from_fft_t(Xn, ratio_thresh).clamp(max=k_max)
        result[is_narrow] = _topk_from_fft_t(Xn, knee)
    return result


def adaptive_k_v2_snr(x: torch.Tensor,
                      ratio_thresh: float = 0.05,
                      flatness_threshold: float = 0.4,
                      quant_levels: int = 32,
                      pilot_k: int = 10,
                      snr_low: float = 3.0,
                      snr_high: float = 10.0,
                      k_max_low_snr: int = 35,
                      k_max_high_snr: int = 20) -> torch.Tensor:
    """v2 SNR-adaptive cap: same routing, but k_max scales with estimated SNR."""
    N, C, T = x.shape
    X = torch.fft.fft(x, n=T, dim=2)
    power = X.abs() ** 2 + 1e-20
    log_power = torch.log(power)
    geo = torch.exp(log_power.mean(dim=2))
    arith = power.mean(dim=2)
    flatness = (geo / (arith + 1e-12)).mean(dim=1)

    is_wide = flatness > flatness_threshold
    is_narrow = ~is_wide
    result = x.clone()
    if is_wide.any():
        result[is_wide] = _per_sample_quantize(x[is_wide], n_levels=quant_levels)
    if is_narrow.any():
        Xn = X[is_narrow]
        # SNR estimate from top-pilot_k vs rest
        psd = (Xn.abs() ** 2).mean(dim=1)
        sorted_psd, _ = psd.sort(dim=1, descending=True)
        sig_p = sorted_psd[:, :pilot_k].sum(dim=1)
        noi_p = sorted_psd[:, pilot_k:].sum(dim=1).clamp(min=1e-20)
        snr_est = sig_p / noi_p
        # Linear interpolate k_max (high SNR -> aggressive smaller k_max)
        t = ((snr_est - snr_low) / (snr_high - snr_low + 1e-12)).clamp(0.0, 1.0)
        k_max_per = (k_max_low_snr + t *
                     (k_max_high_snr - k_max_low_snr)).round().int()
        knee = _knee_from_fft_t(Xn, ratio_thresh)
        knee = torch.minimum(knee, k_max_per)
        result[is_narrow] = _topk_from_fft_t(Xn, knee)
    return result


# =============================================================================
# Classical baselines
# =============================================================================

def _kalman_1d(signal: np.ndarray, q: float = 1e-4, r: float = 0.01) -> np.ndarray:
    """Scalar Kalman smoother. O(T) per channel."""
    n = signal.shape[0]
    x_est = float(signal[0])
    p = 1.0
    out = np.empty(n, dtype=np.float32)
    for k in range(n):
        x_pred = x_est
        p_pred = p + q
        K = p_pred / (p_pred + r)
        x_est = x_pred + K * (signal[k] - x_pred)
        p = (1.0 - K) * p_pred
        out[k] = x_est
    return out


def kalman(x: torch.Tensor, process_noise: float = 1e-4,
           meas_noise: float = 0.01) -> torch.Tensor:
    arr = x.detach().cpu().numpy().astype(np.float32)
    N, C, T = arr.shape
    out = np.empty_like(arr)
    for n in range(N):
        for c in range(C):
            out[n, c] = _kalman_1d(arr[n, c], q=process_noise, r=meas_noise)
    return torch.from_numpy(out).to(device=x.device, dtype=x.dtype)


def wiener(x: torch.Tensor, noise: float = 0.01,
           filter_len: int = 5) -> torch.Tensor:
    arr = x.detach().cpu().numpy().astype(np.float32)
    N, C, T = arr.shape
    out = np.empty_like(arr)
    for n in range(N):
        for c in range(C):
            out[n, c] = scipy_wiener(arr[n, c], mysize=filter_len, noise=noise)
    return torch.from_numpy(out).to(device=x.device, dtype=x.dtype)


def savgol(x: torch.Tensor, window_length: int = 11,
           polyorder: int = 3) -> torch.Tensor:
    if window_length % 2 == 0:
        window_length += 1
    window_length = max(window_length, polyorder + 2)
    if window_length % 2 == 0:
        window_length += 1
    arr = x.detach().cpu().numpy().astype(np.float32)
    out = scipy_savgol(arr, window_length=window_length,
                       polyorder=polyorder, axis=-1)
    return torch.from_numpy(out.astype(np.float32)).to(device=x.device, dtype=x.dtype)


def gaussian(x: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """Depthwise gaussian conv1d (GPU-native)."""
    device, dtype = x.device, x.dtype
    ksize = int(6.0 * sigma + 1.0)
    if ksize % 2 == 0:
        ksize += 1
    half = ksize // 2
    t = torch.arange(-half, half + 1, dtype=torch.float32, device=device)
    g = torch.exp(-t ** 2 / (2.0 * sigma ** 2))
    g = g / g.sum()
    kernel = g.view(1, 1, -1).expand(2, 1, -1).to(dtype=dtype).contiguous()
    x_pad = F.pad(x, (half, half), mode='reflect')
    return F.conv1d(x_pad, kernel, groups=2)


def fir(x: torch.Tensor, cutoff: float = 0.1, numtaps: int = 31) -> torch.Tensor:
    """Depthwise FIR conv1d (GPU-native after kernel transfer)."""
    device, dtype = x.device, x.dtype
    if numtaps % 2 == 0:
        numtaps += 1
    coeffs = firwin(numtaps, cutoff, window='hamming')
    kernel = torch.tensor(coeffs, dtype=dtype, device=device)
    kernel = kernel.view(1, 1, -1).expand(2, 1, -1).contiguous()
    half = numtaps // 2
    x_pad = F.pad(x, (half, half), mode='reflect')
    return F.conv1d(x_pad, kernel, groups=2)


# =============================================================================
# Registry
# =============================================================================

FILTERS: Dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    # fixed-K
    'fft_topk_50':         lambda x: fft_topk(x, topk=50),
    'fft_topk_20':         lambda x: fft_topk(x, topk=20),
    'fft_notch':           lambda x: fft_notch(x, 0.05, 0.25),
    'fft_soft_notch':      lambda x: fft_soft_notch(x, 0.05, 0.25, 0.7, 4),
    # adaptive-K
    'energy_knee_topk':    lambda x: energy_knee_topk(x),
    'magnitude_knee_topk': lambda x: magnitude_knee_topk(x),
    'spectral_shape_topk': lambda x: spectral_shape_topk(x),
    # adaptive-K v2 (flatness-routed: wideband -> quantize, narrowband -> knee Top-K)
    'adaptive_k_v2':       lambda x: adaptive_k_v2(x),
    'adaptive_k_v2_snr':   lambda x: adaptive_k_v2_snr(x),
    # classical
    'kalman':              lambda x: kalman(x),
    'wiener':              lambda x: wiener(x),
    'savgol':              lambda x: savgol(x),
    'gaussian':            lambda x: gaussian(x),
    'fir':                 lambda x: fir(x),
}


# =============================================================================
# Benchmark harness
# =============================================================================

def make_signal(batch_size: int, time_len: int, device: str,
                seed: int = 0) -> torch.Tensor:
    """Synthetic IQ batch with realistic amplitude (~0.02)."""
    g = torch.Generator(device='cpu').manual_seed(seed)
    t = torch.linspace(0, 1, time_len, generator=g) if False else torch.arange(time_len, dtype=torch.float32)
    # Random per-sample carrier + noise, scaled to ~0.02 RMS like RML data
    rng = torch.Generator(device='cpu').manual_seed(seed)
    x = torch.randn(batch_size, 2, time_len, generator=rng) * 0.02
    return x.to(device)


def bench_one(name: str, fn: Callable, x: torch.Tensor,
              warmup: int, iters: int, device: str) -> Dict:
    """Time `fn(x)` over `iters` iterations after `warmup` warmups."""
    is_cuda = (device == 'cuda' and torch.cuda.is_available())

    # Warmup
    for _ in range(warmup):
        _ = fn(x)
    if is_cuda:
        torch.cuda.synchronize()

    # Per-iter timings
    times = np.empty(iters, dtype=np.float64)
    if is_cuda:
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        ends   = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        for i in range(iters):
            starts[i].record()
            _ = fn(x)
            ends[i].record()
        torch.cuda.synchronize()
        for i in range(iters):
            times[i] = starts[i].elapsed_time(ends[i]) / 1000.0  # ms -> s
    else:
        for i in range(iters):
            t0 = time.perf_counter()
            _ = fn(x)
            times[i] = time.perf_counter() - t0

    N = x.shape[0]
    return {
        'name':       name,
        'iters':      iters,
        'mean_ms':    times.mean() * 1e3,
        'median_ms':  np.median(times) * 1e3,
        'std_ms':     times.std() * 1e3,
        'p95_ms':     np.percentile(times, 95) * 1e3,
        'p99_ms':     np.percentile(times, 99) * 1e3,
        'min_ms':     times.min() * 1e3,
        'max_ms':     times.max() * 1e3,
        'per_sample_us': (times.mean() / N) * 1e6,
        'throughput_sps': N / times.mean(),
    }


def fmt_row(r: Dict) -> str:
    return (f"{r['name']:<22s} "
            f"{r['mean_ms']:>9.3f}  "
            f"{r['median_ms']:>9.3f}  "
            f"{r['p95_ms']:>9.3f}  "
            f"{r['p99_ms']:>9.3f}  "
            f"{r['per_sample_us']:>11.2f}  "
            f"{r['throughput_sps']:>14.1f}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--time_len',   type=int, default=128,
                   help='128 for RML2016, 1024 for RML2018')
    p.add_argument('--iters',      type=int, default=100)
    p.add_argument('--warmup',     type=int, default=10)
    p.add_argument('--device',     default='cuda',
                   choices=['cuda', 'cpu'])
    p.add_argument('--filters', default='all',
                   help='comma-separated subset, or "all"')
    p.add_argument('--seed',  type=int, default=0)
    p.add_argument('--csv',   default=None,
                   help='optional path to write CSV results')
    args = p.parse_args()

    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print('[warn] CUDA not available, falling back to CPU')
        device = 'cpu'

    # Select filters
    if args.filters == 'all':
        names = list(FILTERS.keys())
    else:
        names = [n.strip() for n in args.filters.split(',') if n.strip()]
        unknown = [n for n in names if n not in FILTERS]
        if unknown:
            raise SystemExit(f'Unknown filter(s): {unknown}\n'
                             f'Available: {list(FILTERS.keys())}')

    # Build input tensor once
    x = make_signal(args.batch_size, args.time_len, device, seed=args.seed)

    # Header
    print(f"\nDevice: {device}  |  batch={args.batch_size}  "
          f"time_len={args.time_len}  iters={args.iters}  warmup={args.warmup}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print('=' * 110)
    print(f"{'filter':<22s} "
          f"{'mean(ms)':>9s}  "
          f"{'median(ms)':>9s}  "
          f"{'p95(ms)':>9s}  "
          f"{'p99(ms)':>9s}  "
          f"{'us/sample':>11s}  "
          f"{'samples/sec':>14s}")
    print('-' * 110)

    results: List[Dict] = []
    for name in names:
        try:
            r = bench_one(name, FILTERS[name], x,
                          args.warmup, args.iters, device)
        except Exception as e:
            print(f"{name:<22s}  FAILED: {e}")
            continue
        results.append(r)
        print(fmt_row(r))

    print('=' * 110)

    if args.csv and results:
        import csv
        keys = ['name', 'iters', 'mean_ms', 'median_ms', 'std_ms',
                'p95_ms', 'p99_ms', 'min_ms', 'max_ms',
                'per_sample_us', 'throughput_sps']
        with open(args.csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in results:
                w.writerow({k: r[k] for k in keys})
        print(f"[saved] {args.csv}")


if __name__ == '__main__':
    main()
