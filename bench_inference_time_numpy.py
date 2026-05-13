"""
Standalone inference-time benchmark for adaptive-K and classical filter defenses.

Pure NumPy + SciPy implementation. No PyTorch, no CUDA — runs on any embedded
Linux (i.MX 95, Raspberry Pi, Jetson Nano, etc.) with `pip install numpy scipy`.

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
    kalman              -- scalar constant-velocity Kalman smoother
    wiener              -- scipy.signal.wiener
    savgol              -- scipy.signal.savgol_filter
    gaussian            -- scipy.ndimage.gaussian_filter1d
    fir                 -- scipy.signal.firwin + lfilter

Usage on i.MX 95 / aarch64:
    pip install numpy scipy
    python3 bench_inference_time_numpy.py
    python3 bench_inference_time_numpy.py --batch_size 1   # real-time scenario
    python3 bench_inference_time_numpy.py --batch_size 32 --iters 200
    python3 bench_inference_time_numpy.py --filters fft_topk_50,energy_knee_topk
    python3 bench_inference_time_numpy.py --csv imx95_results.csv

Real-time scenario for embedded boards:
    Set --batch_size 1 to measure single-sample latency. Per-sample microseconds
    will go up vs. batch=128 (kernel launch / FFT setup not amortized).
"""

import argparse
import sys
import time
from typing import Callable, Dict, List

import numpy as np
from scipy.signal import wiener as scipy_wiener
from scipy.signal import savgol_filter as scipy_savgol
from scipy.signal import firwin, lfilter
from scipy.ndimage import gaussian_filter1d


# =============================================================================
# Fixed-K FFT family
# =============================================================================

def fft_topk(x: np.ndarray, topk: int = 50) -> np.ndarray:
    """Keep top-K FFT bins per (sample, channel)."""
    N, C, T = x.shape
    k = min(int(topk), T)
    X = np.fft.fft(x, n=T, axis=2)
    mags = np.abs(X)
    # Indices of top-k along last axis
    idx = np.argpartition(mags, T - k, axis=2)[..., T - k:]
    mask = np.zeros_like(mags, dtype=bool)
    np.put_along_axis(mask, idx, True, axis=2)
    X_filt = X * mask
    return np.fft.ifft(X_filt, n=T, axis=2).real.astype(x.dtype)


def fft_notch(x: np.ndarray, f_low: float = 0.05, f_high: float = 0.25) -> np.ndarray:
    """Zero a frequency band [f_low, f_high] (normalized 0..0.5)."""
    N, C, T = x.shape
    F_ = T // 2 + 1
    k_low = max(0, min(int(np.ceil(f_low * T)), F_ - 1))
    k_high = max(0, min(int(np.floor(f_high * T)), F_ - 1))
    if k_high < k_low:
        k_high = k_low
    X = np.fft.rfft(x, n=T, axis=2)
    mask = np.ones(F_, dtype=x.dtype)
    mask[k_low:k_high + 1] = 0.0
    X = X * mask[None, None, :]
    return np.fft.irfft(X, n=T, axis=2).astype(x.dtype)


def fft_soft_notch(x: np.ndarray, f_low: float = 0.05, f_high: float = 0.25,
                   depth: float = 0.7, trans: int = 4) -> np.ndarray:
    """Tapered (cosine) notch attenuator."""
    N, C, T = x.shape
    F_ = T // 2 + 1
    k_low = max(0, min(int(np.ceil(f_low * T)), F_ - 1))
    k_high = max(0, min(int(np.floor(f_high * T)), F_ - 1))
    if k_high < k_low:
        k_high = k_low
    mask = np.ones(F_, dtype=x.dtype)
    mask[k_low:k_high + 1] = 1.0 - depth
    for i in range(1, trans + 1):
        w = 0.5 * (1.0 - np.cos(np.pi * i / (trans + 1)))
        atten = 1.0 - depth * (1.0 - w)
        if k_low - i >= 0:
            mask[k_low - i] = min(mask[k_low - i], atten)
        if k_high + i < F_:
            mask[k_high + i] = min(mask[k_high + i], atten)
    X = np.fft.rfft(x, n=T, axis=2)
    X = X * mask[None, None, :]
    return np.fft.irfft(X, n=T, axis=2).astype(x.dtype)


# =============================================================================
# Adaptive-K FFT family (per-sample K)
# =============================================================================

def _apply_grouped_topk(x: np.ndarray, selected_k: np.ndarray,
                        k_cands: List[int]) -> np.ndarray:
    """Group samples by chosen K and apply fft_topk batched per group."""
    result = x.copy()
    for kc in k_cands:
        mask = (selected_k == kc)
        if mask.any():
            result[mask] = fft_topk(x[mask], topk=int(kc))
    return result


def energy_knee_topk(x: np.ndarray, threshold: float = 0.90,
                     k_candidates: List[int] = None,
                     k_min: int = 10, k_max: int = 50) -> np.ndarray:
    """Per-sample K = cumulative energy knee, snapped to candidate grid."""
    if k_candidates is None:
        k_candidates = [10, 15, 20, 30, 50]
    k_cands = sorted(k_candidates)
    N, C, T = x.shape

    X = np.fft.fft(x, n=T, axis=2)
    energy = np.abs(X) ** 2
    sorted_e = np.sort(energy, axis=2)[..., ::-1]
    cumsum = np.cumsum(sorted_e, axis=2)
    total = energy.sum(axis=2, keepdims=True)
    reached = cumsum >= threshold * total
    # First True index along axis=2
    knee_per_ch = reached.argmax(axis=2) + 1
    never = ~reached.any(axis=2)
    knee_per_ch[never] = T
    knee = knee_per_ch.max(axis=1).astype(np.int32)

    selected_k = np.full_like(knee, k_cands[-1])
    for kc in reversed(k_cands):
        selected_k = np.where(knee <= kc, kc, selected_k)
    selected_k = np.clip(selected_k, k_min, k_max)
    return _apply_grouped_topk(x, selected_k, k_cands)


def magnitude_knee_topk(x: np.ndarray, ratio_thresh: float = 0.05) -> np.ndarray:
    """Per-sample K = first bin where |X[k]|/|X[0]| < ratio_thresh."""
    N, C, T = x.shape
    X = np.fft.fft(x, axis=2)
    psd = (np.abs(X) ** 2).mean(axis=1)  # [N, T]
    sorted_mag = np.sort(np.sqrt(psd), axis=1)[:, ::-1]
    peak = np.maximum(sorted_mag[:, 0:1], 1e-12)
    ratio = sorted_mag / peak
    below = ratio < ratio_thresh
    knee = below.argmax(axis=1)
    never = ~below.any(axis=1)
    knee[never] = T
    knee = np.clip(knee, 1, None).astype(np.int32)

    result = x.copy()
    unique_ks = np.unique(knee).tolist()
    for kc in unique_ks:
        mask = (knee == kc)
        if mask.any():
            result[mask] = fft_topk(x[mask], topk=int(kc))
    return result


def spectral_shape_topk(x: np.ndarray, sig_pct: float = 0.10,
                        k_candidates: List[int] = None,
                        k_min: int = 10, k_max: int = 50) -> np.ndarray:
    """Per-sample K = bin count above sig_pct * peak, snapped to candidate grid."""
    if k_candidates is None:
        k_candidates = [10, 15, 20, 30, 50]
    k_cands = sorted(k_candidates)
    N, C, T = x.shape

    X = np.fft.fft(x, n=T, axis=2)
    mag = np.abs(X)
    peak = mag.max(axis=2, keepdims=True)
    sig_mask = mag > (sig_pct * peak)
    count_per_ch = sig_mask.sum(axis=2)
    bin_count = count_per_ch.max(axis=1).astype(np.int32)

    selected_k = np.full_like(bin_count, k_cands[-1])
    for kc in reversed(k_cands):
        selected_k = np.where(bin_count <= kc, kc, selected_k)
    selected_k = np.clip(selected_k, k_min, k_max)
    return _apply_grouped_topk(x, selected_k, k_cands)


# =============================================================================
# Adaptive-K v2 family (flatness-routed: quantize wideband, knee Top-K narrowband)
# =============================================================================

def _per_sample_quantize(x: np.ndarray, n_levels: int = 32) -> np.ndarray:
    """Per-sample uniform quantization to n_levels discrete values."""
    N = x.shape[0]
    x_flat = x.reshape(N, -1)
    x_min = x_flat.min(axis=1).reshape(N, 1, 1)
    x_max = x_flat.max(axis=1).reshape(N, 1, 1)
    rng = (x_max - x_min) + 1e-12
    x_norm = (x - x_min) / rng
    x_q = np.round(x_norm * (n_levels - 1)) / (n_levels - 1)
    return (x_q * rng + x_min).astype(x.dtype)


def _knee_from_fft_np(X_n: np.ndarray, ratio_thresh: float = 0.05) -> np.ndarray:
    """Sorted-magnitude knee from precomputed complex FFT [M, 2, T]."""
    T = X_n.shape[2]
    psd = (np.abs(X_n) ** 2).mean(axis=1)  # [M, T]
    sorted_mag = np.sort(np.sqrt(psd), axis=1)[:, ::-1]
    peak = np.maximum(sorted_mag[:, 0:1], 1e-12)
    ratio = sorted_mag / peak
    below = ratio < ratio_thresh
    knee = below.argmax(axis=1)
    never = ~below.any(axis=1)
    knee[never] = T
    return np.clip(knee, 1, None).astype(np.int32)


def _topk_from_fft_np(X_n: np.ndarray, knee: np.ndarray) -> np.ndarray:
    """Per-sample Top-K filter on precomputed FFT, grouped by unique K."""
    T = X_n.shape[2]
    out = X_n.copy()
    for k in np.unique(knee).tolist():
        m = (knee == k)
        if not m.any():
            continue
        Xs = X_n[m]
        mags = np.abs(Xs)
        kk = min(int(k), T)
        idx = np.argpartition(mags, T - kk, axis=2)[..., T - kk:]
        fmask = np.zeros_like(mags, dtype=bool)
        np.put_along_axis(fmask, idx, True, axis=2)
        out[m] = Xs * fmask
    return np.fft.ifft(out, n=T, axis=2).real


def _flatness_route(x: np.ndarray, flatness_threshold: float):
    """Compute FFT, spectral flatness, and wideband/narrowband mask."""
    N, C, T = x.shape
    X = np.fft.fft(x, n=T, axis=2)
    power = np.abs(X) ** 2 + 1e-20
    log_power = np.log(power)
    geo = np.exp(log_power.mean(axis=2))
    arith = power.mean(axis=2)
    flatness = (geo / (arith + 1e-12)).mean(axis=1)
    is_wide = flatness > flatness_threshold
    return X, is_wide


def adaptive_k_v2(x: np.ndarray,
                  ratio_thresh: float = 0.05,
                  k_max: int = 20,
                  flatness_threshold: float = 0.4,
                  quant_levels: int = 32) -> np.ndarray:
    """v2 fixed-cap: flatness routes wideband->quantize, narrowband->knee Top-K."""
    X, is_wide = _flatness_route(x, flatness_threshold)
    is_narrow = ~is_wide
    result = x.copy()
    if is_wide.any():
        result[is_wide] = _per_sample_quantize(x[is_wide], n_levels=quant_levels)
    if is_narrow.any():
        Xn = X[is_narrow]
        knee = np.minimum(_knee_from_fft_np(Xn, ratio_thresh), k_max)
        result[is_narrow] = _topk_from_fft_np(Xn, knee).astype(x.dtype)
    return result


def adaptive_k_v2_snr(x: np.ndarray,
                      ratio_thresh: float = 0.05,
                      flatness_threshold: float = 0.4,
                      quant_levels: int = 32,
                      pilot_k: int = 10,
                      snr_low: float = 3.0,
                      snr_high: float = 10.0,
                      k_max_low_snr: int = 35,
                      k_max_high_snr: int = 20) -> np.ndarray:
    """v2 SNR-adaptive cap: same routing, but k_max scales with estimated SNR."""
    X, is_wide = _flatness_route(x, flatness_threshold)
    is_narrow = ~is_wide
    result = x.copy()
    if is_wide.any():
        result[is_wide] = _per_sample_quantize(x[is_wide], n_levels=quant_levels)
    if is_narrow.any():
        Xn = X[is_narrow]
        psd = (np.abs(Xn) ** 2).mean(axis=1)
        sorted_psd = np.sort(psd, axis=1)[:, ::-1]
        sig_p = sorted_psd[:, :pilot_k].sum(axis=1)
        noi_p = np.maximum(sorted_psd[:, pilot_k:].sum(axis=1), 1e-20)
        snr_est = sig_p / noi_p
        t = np.clip((snr_est - snr_low) / (snr_high - snr_low + 1e-12), 0.0, 1.0)
        k_max_per = np.round(
            k_max_low_snr + t * (k_max_high_snr - k_max_low_snr)
        ).astype(np.int32)
        knee = np.minimum(_knee_from_fft_np(Xn, ratio_thresh), k_max_per)
        result[is_narrow] = _topk_from_fft_np(Xn, knee).astype(x.dtype)
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


def kalman(x: np.ndarray, process_noise: float = 1e-4,
           meas_noise: float = 0.01) -> np.ndarray:
    N, C, T = x.shape
    out = np.empty_like(x)
    for n in range(N):
        for c in range(C):
            out[n, c] = _kalman_1d(x[n, c], q=process_noise, r=meas_noise)
    return out


def wiener(x: np.ndarray, noise: float = 0.01,
           filter_len: int = 5) -> np.ndarray:
    N, C, T = x.shape
    out = np.empty_like(x)
    for n in range(N):
        for c in range(C):
            out[n, c] = scipy_wiener(x[n, c], mysize=filter_len, noise=noise)
    return out


def savgol(x: np.ndarray, window_length: int = 11,
           polyorder: int = 3) -> np.ndarray:
    if window_length % 2 == 0:
        window_length += 1
    window_length = max(window_length, polyorder + 2)
    if window_length % 2 == 0:
        window_length += 1
    return scipy_savgol(x, window_length=window_length,
                        polyorder=polyorder, axis=-1).astype(x.dtype)


def gaussian(x: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    return gaussian_filter1d(x, sigma=sigma, axis=-1, mode='reflect').astype(x.dtype)


def fir(x: np.ndarray, cutoff: float = 0.1, numtaps: int = 31) -> np.ndarray:
    if numtaps % 2 == 0:
        numtaps += 1
    coeffs = firwin(numtaps, cutoff, window='hamming').astype(x.dtype)
    # lfilter is 1-D; apply along last axis with `axis` argument.
    return lfilter(coeffs, [1.0], x, axis=-1).astype(x.dtype)


# =============================================================================
# Registry
# =============================================================================

FILTERS: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
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

def make_signal(batch_size: int, time_len: int, seed: int = 0) -> np.ndarray:
    """Synthetic IQ batch with realistic amplitude (~0.02)."""
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((batch_size, 2, time_len)) * 0.02).astype(np.float32)


def bench_one(name: str, fn: Callable, x: np.ndarray,
              warmup: int, iters: int) -> Dict:
    """Time `fn(x)` over `iters` iterations after `warmup` warmups."""
    for _ in range(warmup):
        _ = fn(x)

    times = np.empty(iters, dtype=np.float64)
    for i in range(iters):
        t0 = time.perf_counter()
        _ = fn(x)
        times[i] = time.perf_counter() - t0

    N = x.shape[0]
    return {
        'name':           name,
        'iters':          iters,
        'mean_ms':        times.mean() * 1e3,
        'median_ms':      np.median(times) * 1e3,
        'std_ms':         times.std() * 1e3,
        'p95_ms':         np.percentile(times, 95) * 1e3,
        'p99_ms':         np.percentile(times, 99) * 1e3,
        'min_ms':         times.min() * 1e3,
        'max_ms':         times.max() * 1e3,
        'per_sample_us':  (times.mean() / N) * 1e6,
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


def get_cpu_info() -> str:
    """Best-effort one-line CPU description (works on Linux/aarch64)."""
    try:
        with open('/proc/cpuinfo') as f:
            text = f.read()
        # Try common keys in order
        for key in ('model name', 'Model', 'Hardware', 'Processor'):
            for line in text.split('\n'):
                if line.startswith(key):
                    return line.split(':', 1)[1].strip()
        # Fallback: count cores
        import os
        return f"{os.cpu_count()} cores (unknown model)"
    except Exception:
        return 'unknown CPU'


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--batch_size', type=int, default=128,
                   help='1 for single-sample real-time scenario')
    p.add_argument('--time_len',   type=int, default=128,
                   help='128 for RML2016, 1024 for RML2018')
    p.add_argument('--iters',      type=int, default=100)
    p.add_argument('--warmup',     type=int, default=10)
    p.add_argument('--filters',    default='all',
                   help='comma-separated subset, or "all"')
    p.add_argument('--seed',       type=int, default=0)
    p.add_argument('--csv',        default=None,
                   help='optional path to write CSV results')
    p.add_argument('--threads',    type=int, default=None,
                   help='OMP/MKL threads (default: leave env as-is)')
    args = p.parse_args()

    if args.threads is not None:
        import os
        os.environ['OMP_NUM_THREADS'] = str(args.threads)
        os.environ['OPENBLAS_NUM_THREADS'] = str(args.threads)
        os.environ['MKL_NUM_THREADS'] = str(args.threads)

    if args.filters == 'all':
        names = list(FILTERS.keys())
    else:
        names = [n.strip() for n in args.filters.split(',') if n.strip()]
        unknown = [n for n in names if n not in FILTERS]
        if unknown:
            print(f'Unknown filter(s): {unknown}', file=sys.stderr)
            print(f'Available: {list(FILTERS.keys())}', file=sys.stderr)
            sys.exit(1)

    x = make_signal(args.batch_size, args.time_len, seed=args.seed)

    print(f"\nCPU: {get_cpu_info()}")
    print(f"NumPy: {np.__version__}  |  Python: {sys.version.split()[0]}")
    print(f"batch={args.batch_size}  time_len={args.time_len}  "
          f"iters={args.iters}  warmup={args.warmup}")
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
            r = bench_one(name, FILTERS[name], x, args.warmup, args.iters)
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
