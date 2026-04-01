"""
Classical signal-processing filter baselines for adversarial defense comparison.

Provides five filter functions that operate on IQ signal tensors of shape [N, 2, T].
These baselines are used for comparison against the unified detect->recover->classify
pipeline in the adversarial defense evaluation.

Defense functions:
    kalman_filter   -- Scalar constant-velocity Kalman smoother (CPU/numpy)
    wiener_filter   -- Wiener filter via scipy.signal.wiener (CPU/numpy)
    sg_filter       -- Savitzky-Golay filter via scipy.signal.savgol_filter (CPU/numpy)
    gaussian_filter -- Gaussian low-pass filter via depthwise F.conv1d (GPU-native)
    fir_filter      -- FIR low-pass filter via firwin + depthwise F.conv1d (GPU-native)

All functions accept x: torch.Tensor of shape [N, 2, T] and return the same shape
on the same device. GPU-native filters (gaussian, fir) never call .cpu() or .numpy().
CPU filters (kalman, wiener, sg) bridge tensor/numpy at the function boundary.
"""

import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal import wiener as scipy_wiener
from scipy.signal import savgol_filter as scipy_savgol
from scipy.signal import firwin

__all__ = ['kalman_filter', 'wiener_filter', 'sg_filter', 'gaussian_filter', 'fir_filter']


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _kalman_1d(signal: np.ndarray, q: float = 1e-4, r: float = 0.01) -> np.ndarray:
    """Scalar constant-velocity Kalman smoother. O(T) per channel.

    Args:
        signal: 1D float32 array of length T.
        q:      Process noise covariance (controls how much state can change).
        r:      Measurement noise covariance (controls trust in observations).

    Returns:
        Filtered 1D float32 array of length T.
    """
    n = len(signal)
    x_est = float(signal[0])
    p = 1.0
    out = np.empty(n, dtype=np.float32)
    for k in range(n):
        # Predict
        x_pred = x_est
        p_pred = p + q
        # Update
        K = p_pred / (p_pred + r)
        x_est = x_pred + K * (signal[k] - x_pred)
        p = (1.0 - K) * p_pred
        out[k] = x_est
    return out


# ---------------------------------------------------------------------------
# Public filter functions
# ---------------------------------------------------------------------------

def kalman_filter(x: torch.Tensor, *, process_noise: float = 1e-4,
                  meas_noise: float = 0.01) -> torch.Tensor:
    """Scalar 1D Kalman smoother applied independently to each I/Q channel.

    CPU-only path (no pykalman/filterpy required). Suitable for denoising
    adversarial perturbations by modelling IQ samples as a slowly-varying
    state with Gaussian measurement noise.

    Args:
        x:             Input tensor of shape [N, 2, T], any device.
        process_noise: Process noise covariance q (default 1e-4).
        meas_noise:    Measurement noise covariance r (default 0.01).

    Returns:
        Filtered tensor of shape [N, 2, T] on same device as x.
    """
    arr = x.detach().cpu().numpy().astype(np.float32)
    N, C, T = arr.shape
    out = np.empty_like(arr)
    for n in range(N):
        for c in range(C):
            out[n, c] = _kalman_1d(arr[n, c], q=process_noise, r=meas_noise)
    return torch.from_numpy(out).to(device=x.device, dtype=x.dtype)


def wiener_filter(x: torch.Tensor, *, noise: float = 0.01,
                  filter_len: int = 5) -> torch.Tensor:
    """Wiener filter (local statistics variant) via scipy.signal.wiener.

    Applies the filter independently to each I/Q channel per sample.
    Uses local statistics to estimate the noise and signal power, producing
    a Wiener-optimal least-squares estimate for each sample in a sliding window.

    Args:
        x:          Input tensor of shape [N, 2, T], any device.
        noise:      Assumed noise variance (default 0.01).
        filter_len: Size of the local estimation window in samples (default 5).

    Returns:
        Filtered tensor of shape [N, 2, T] on same device as x.
    """
    arr = x.detach().cpu().numpy().astype(np.float32)
    N, C, T = arr.shape
    out = np.empty_like(arr)
    for n in range(N):
        for c in range(C):
            out[n, c] = scipy_wiener(arr[n, c], mysize=filter_len, noise=noise)
    return torch.from_numpy(out).to(device=x.device, dtype=x.dtype)


def sg_filter(x: torch.Tensor, *, window_length: int = 11,
              polyorder: int = 3) -> torch.Tensor:
    """Savitzky-Golay smoothing filter via scipy.signal.savgol_filter.

    Operates on the full [N, 2, T] array at once (vectorized over batch and
    channel dimensions via axis=-1). Automatically enforces the constraints:
      - window_length must be odd
      - window_length must be > polyorder

    Args:
        x:             Input tensor of shape [N, 2, T], any device.
        window_length: Length of the filter window in samples (default 11).
                       Must be positive. Even values are incremented by 1.
        polyorder:     Polynomial order for fitting (default 3).
                       Must be less than window_length.

    Returns:
        Filtered tensor of shape [N, 2, T] on same device as x.
    """
    # Enforce odd window length
    if window_length % 2 == 0:
        window_length += 1
    # Enforce window_length > polyorder (must be at least polyorder+2 for odd constraint)
    window_length = max(window_length, polyorder + 2)
    # Re-enforce odd after max() in case polyorder+2 is even
    if window_length % 2 == 0:
        window_length += 1

    arr = x.detach().cpu().numpy().astype(np.float32)
    out = scipy_savgol(arr, window_length=window_length, polyorder=polyorder, axis=-1)
    return torch.from_numpy(out.astype(np.float32)).to(device=x.device, dtype=x.dtype)


def gaussian_filter(x: torch.Tensor, *, sigma: float = 1.0) -> torch.Tensor:
    """Gaussian low-pass filter via GPU-native depthwise F.conv1d.

    Builds a 1D Gaussian kernel on the input device and applies it to both
    I and Q channels simultaneously via groups=2 depthwise convolution.
    Never moves data to CPU. Reflect padding avoids edge artefacts.

    Args:
        x:     Input tensor of shape [N, 2, T], any device.
        sigma: Standard deviation of the Gaussian kernel in samples (default 1.0).

    Returns:
        Filtered tensor of shape [N, 2, T] on same device as x.
    """
    device = x.device
    dtype = x.dtype

    ksize = int(6.0 * sigma + 1.0)
    if ksize % 2 == 0:
        ksize += 1
    half = ksize // 2

    t = torch.arange(-half, half + 1, dtype=torch.float32, device=device)
    gauss = torch.exp(-t ** 2 / (2.0 * sigma ** 2))
    gauss = gauss / gauss.sum()

    # [2, 1, ksize] for depthwise conv1d with groups=2
    kernel = gauss.view(1, 1, -1).expand(2, 1, -1).to(dtype=dtype)
    kernel = kernel.contiguous()

    x_padded = F.pad(x, (half, half), mode='reflect')
    return F.conv1d(x_padded, kernel, groups=2)


def fir_filter(x: torch.Tensor, *, cutoff: float = 0.1,
               numtaps: int = 31) -> torch.Tensor:
    """FIR low-pass filter via GPU-native depthwise F.conv1d.

    Designs filter coefficients on CPU using scipy.signal.firwin (Hamming
    window method), then moves them to the input device for application via
    depthwise conv1d. Never moves input data to CPU. Reflect padding avoids
    edge artefacts. Coefficients are recomputed per call (not cached) to
    support calibration sweeps with varying parameters.

    Args:
        x:       Input tensor of shape [N, 2, T], any device.
        cutoff:  Normalised cutoff frequency in [0, 0.5] (default 0.1).
                 A value of 0.5 corresponds to the Nyquist rate.
        numtaps: Number of FIR filter taps (default 31). Even values are
                 incremented by 1 to produce a symmetric Type I filter.

    Returns:
        Filtered tensor of shape [N, 2, T] on same device as x.
    """
    device = x.device
    dtype = x.dtype

    if numtaps % 2 == 0:
        numtaps += 1  # firwin needs odd taps for Type I low-pass

    coeffs = firwin(numtaps, cutoff, window='hamming')

    kernel = torch.tensor(coeffs, dtype=dtype, device=device)
    # [2, 1, numtaps] for depthwise conv1d with groups=2
    kernel = kernel.view(1, 1, -1).expand(2, 1, -1).contiguous()

    half = numtaps // 2
    x_padded = F.pad(x, (half, half), mode='reflect')
    return F.conv1d(x_padded, kernel, groups=2)
