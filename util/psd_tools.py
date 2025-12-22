from typing import Tuple

import numpy as np
import torch


def compute_avg_psd_mask(x: torch.Tensor) -> np.ndarray:
    """
    Compute an average one-sided PSD-based mask from complex baseband IQ.

    Args:
        x: Tensor [N, 2, T] (I, Q)

    Returns:
        mask: numpy array of length F=T//2+1 normalized to max=1.0. This is a
              magnitude mask suitable for spectral_noise_attack(spec_type='psd_mask').
    """
    assert x.dim() == 3 and x.size(1) == 2
    N, C, T = x.shape

    z = x[:, 0, :] + 1j * x[:, 1, :]
    Z = torch.fft.fft(z, n=T, dim=1)
    Zp = Z[:, : T // 2 + 1]
    # Power spectrum and batch average
    psd = (Zp.abs() ** 2) / T
    psd_mean = psd.mean(dim=0)
    # Use amplitude mask proportional to sqrt(PSD), then normalize
    amp = psd_mean.sqrt().clamp_min(1e-12)
    amp = amp / amp.max()
    return amp.detach().cpu().numpy()

