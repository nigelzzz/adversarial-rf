"""
Adaptive-K v2: Unified Spectral Defense (single-sample, self-contained).

Standalone port of the paper's proposed defense (Algorithm 1,
paper/latex/sections/proposed_method.tex), equivalent to
util/defense.py:adaptive_k_v2_snr_defense but for ONE sample of shape [2, 128].
Depends only on numpy, so it can serve as a golden reference for an HLS port.

Pipeline (one FFT + one IFFT):
    1. X = FFT(x) per I/Q channel
    2. Spectral flatness SF (mean over I/Q)
         SF > 0.4  -> wideband (AM-SSB): per-sample uniform quantization, L=32
         otherwise -> narrowband: steps 3-5
    3. SNR estimate: energy of top-10 PSD bins / energy of remaining bins
    4. K cap: 35 if SNR <= 3, 20 if SNR >= 10, linear interpolation between
    5. Knee: first index where sorted |X| / peak < 0.05;  K = min(knee, K_max)
       Keep the K largest |X| bins per channel, zero the rest, IFFT, take real.

Usage:
    from adaptive_k_v2 import adaptive_k_v2
    y = adaptive_k_v2(x)          # x: np.ndarray [2, 128] -> y: [2, 128]

    python awn_fpga/adaptive_k_v2.py            # self-test vs util/defense.py
"""

import numpy as np

# Paper defaults (Algorithm 1)
FLATNESS_THRESHOLD = 0.4
QUANT_LEVELS = 32
RATIO_THRESH = 0.05
PILOT_K = 10
SNR_LOW = 3.0
SNR_HIGH = 10.0
K_MAX_LOW_SNR = 35
K_MAX_HIGH_SNR = 20


def spectral_flatness(X: np.ndarray) -> float:
    """Geometric / arithmetic mean of |X|^2 per channel, averaged over I/Q."""
    power = np.abs(X) ** 2 + 1e-20
    geo_mean = np.exp(np.log(power).mean(axis=1))
    arith_mean = power.mean(axis=1)
    return float((geo_mean / (arith_mean + 1e-12)).mean())


def quantize(x: np.ndarray, n_levels: int = QUANT_LEVELS) -> np.ndarray:
    """Uniform quantization to n_levels over the sample's [min, max] range."""
    x_min, x_max = x.min(), x.max()
    rng = x_max - x_min + 1e-12
    x_q = np.round((x - x_min) / rng * (n_levels - 1)) / (n_levels - 1)
    return x_q * rng + x_min


def estimate_snr(psd: np.ndarray, pilot_k: int = PILOT_K) -> float:
    """Top-pilot_k PSD energy (signal proxy) over the rest (noise proxy)."""
    sorted_psd = np.sort(psd)[::-1]
    signal_power = sorted_psd[:pilot_k].sum()
    noise_power = max(sorted_psd[pilot_k:].sum(), 1e-20)
    return float(signal_power / noise_power)


def snr_to_k_max(snr_est: float) -> int:
    """High SNR -> aggressive cap (20), low SNR -> gentle cap (35)."""
    t = (snr_est - SNR_LOW) / (SNR_HIGH - SNR_LOW + 1e-12)
    t = min(max(t, 0.0), 1.0)
    return int(np.round(K_MAX_LOW_SNR + t * (K_MAX_HIGH_SNR - K_MAX_LOW_SNR)))


def magnitude_knee(psd: np.ndarray, ratio_thresh: float = RATIO_THRESH) -> int:
    """First index where sorted magnitude drops below ratio_thresh * peak."""
    T = psd.shape[0]
    sorted_mag = np.sort(np.sqrt(psd))[::-1]
    ratio = sorted_mag / max(sorted_mag[0], 1e-12)
    below = np.nonzero(ratio < ratio_thresh)[0]
    knee = int(below[0]) if below.size else T
    return max(knee, 1)


def topk_filter(X: np.ndarray, k: int) -> np.ndarray:
    """Keep the k largest-magnitude bins per channel, zero the rest, IFFT."""
    C, T = X.shape
    k = min(k, T)
    mask = np.zeros((C, T), dtype=bool)
    for c in range(C):
        idx = np.argsort(-np.abs(X[c]), kind='stable')[:k]
        mask[c, idx] = True
    return np.fft.ifft(X * mask, n=T, axis=1).real


def adaptive_k_v2(x: np.ndarray, return_info: bool = False):
    """
    Adaptive-K v2 defense for one IQ sample.

    Args:
        x: [2, 128] real array (row 0 = I, row 1 = Q), raw RML2016 scale
           (no normalization needed; Top-K is scale/shift invariant).
        return_info: also return a dict with the routing decision and K.

    Returns:
        y: [2, 128] defended signal, same dtype as x
        info (optional): {'flatness', 'wideband', 'snr_est', 'k_max', 'knee', 'k'}
    """
    x = np.asarray(x)
    assert x.ndim == 2 and x.shape[0] == 2, f'expected [2, T], got {x.shape}'
    T = x.shape[1]

    X = np.fft.fft(x.astype(np.float64), n=T, axis=1)   # ONE shared FFT
    sf = spectral_flatness(X)
    info = {'flatness': sf, 'wideband': sf > FLATNESS_THRESHOLD}

    if info['wideband']:
        y = quantize(x.astype(np.float64))
    else:
        psd = (np.abs(X) ** 2).mean(axis=0)              # [T], I/Q averaged
        info['snr_est'] = estimate_snr(psd)
        info['k_max'] = snr_to_k_max(info['snr_est'])
        info['knee'] = magnitude_knee(psd)
        info['k'] = min(info['knee'], info['k_max'])
        y = topk_filter(X, info['k'])

    y = y.astype(x.dtype if np.issubdtype(x.dtype, np.floating) else np.float32)
    return (y, info) if return_info else y


def _self_test(n: int = 2000, seed: int = 0):
    """Compare against util/defense.py on RML2016.10a samples (or random)."""
    import os
    import sys
    import pickle
    import torch
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    from util.defense import adaptive_k_v2_snr_defense

    rng = np.random.default_rng(seed)
    pkl = os.path.join(os.path.dirname(__file__), '..', 'data', 'RML2016.10a_dict.pkl')
    if os.path.exists(pkl):
        with open(pkl, 'rb') as f:
            data = pickle.load(f, encoding='bytes')
        xs = np.concatenate(list(data.values()), axis=0).astype(np.float32)
        xs = xs[rng.choice(len(xs), size=n, replace=False)]
        src = 'RML2016.10a'
    else:
        xs = (rng.standard_normal((n, 2, 128)) * 0.01).astype(np.float32)
        src = 'random'

    ref = adaptive_k_v2_snr_defense(torch.from_numpy(xs)).numpy()
    out = np.stack([adaptive_k_v2(s) for s in xs])
    err = np.abs(out - ref).max(axis=(1, 2))
    scale = np.abs(xs).max()
    n_bad = int((err > 1e-5 * scale).sum())
    n_wide = sum(adaptive_k_v2(s, return_info=True)[1]['wideband'] for s in xs)
    print(f'[{src}] {n} samples, wideband={n_wide}, max|err|={err.max():.3e} '
          f'(signal max {scale:.3e}), mismatched samples={n_bad}')
    return n_bad


if __name__ == '__main__':
    _self_test()
