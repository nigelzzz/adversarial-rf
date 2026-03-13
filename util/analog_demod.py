"""
Minimal analog demodulators for AM and FM labels in RadioML-style datasets.

These functions recover a baseband message waveform from complex IQ:
- AM-DSB: envelope detector
- AM-SSB: product detector (real part after optional phase rotation)
- FM/WBFM: phase-difference frequency discriminator
"""

from typing import Optional

import numpy as np


def _to_complex_1d(iq: np.ndarray) -> np.ndarray:
    """Accept [N], [2, N], or [N, 2] and return complex [N]."""
    x = np.asarray(iq)
    if np.iscomplexobj(x):
        return x.reshape(-1).astype(np.complex64)
    if x.ndim != 2:
        raise ValueError("Expected complex [N] or real [2,N]/[N,2] IQ array")
    if x.shape[0] == 2:
        return (x[0] + 1j * x[1]).astype(np.complex64)
    if x.shape[1] == 2:
        return (x[:, 0] + 1j * x[:, 1]).astype(np.complex64)
    raise ValueError("Expected shape [2,N] or [N,2] for real IQ array")


def _dc_block(x: np.ndarray) -> np.ndarray:
    """Remove DC offset."""
    return x - np.mean(x)


def _deemphasis_1pole(x: np.ndarray, fs: float, tau: float = 75e-6) -> np.ndarray:
    """
    Simple 1-pole de-emphasis IIR.

    tau=75us is standard in the US. Use 50e-6 in many other regions.
    """
    if fs <= 0:
        raise ValueError("fs must be > 0 for de-emphasis")
    a = np.exp(-1.0 / (fs * tau))
    y = np.empty_like(x, dtype=np.float32)
    y[0] = x[0]
    for n in range(1, len(x)):
        y[n] = a * y[n - 1] + (1.0 - a) * x[n]
    return y


def _normalize_peak(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    peak = float(np.max(np.abs(x)))
    if peak < eps:
        return x.astype(np.float32)
    return (x / peak).astype(np.float32)


def demod_am_dsb(iq: np.ndarray, remove_dc: bool = True) -> np.ndarray:
    """Envelope detector for AM-DSB."""
    z = _to_complex_1d(iq)
    audio = np.abs(z).astype(np.float32)
    if remove_dc:
        audio = _dc_block(audio)
    return _normalize_peak(audio)


def demod_am_ssb(iq: np.ndarray, carrier_phase: Optional[float] = None) -> np.ndarray:
    """
    Product detector for AM-SSB at complex baseband.

    If carrier phase is known/estimated, pass it. Otherwise 0 rad is used.
    """
    z = _to_complex_1d(iq)
    phi = 0.0 if carrier_phase is None else float(carrier_phase)
    audio = np.real(z * np.exp(-1j * phi)).astype(np.float32)
    audio = _dc_block(audio)
    return _normalize_peak(audio)


def demod_fm(iq: np.ndarray, fs: Optional[float] = None, deemph_tau: Optional[float] = 75e-6) -> np.ndarray:
    """
    Phase-difference FM discriminator.

    freq[n] = angle(z[n] * conj(z[n-1]))
    """
    z = _to_complex_1d(iq)
    if len(z) < 2:
        return np.zeros(len(z), dtype=np.float32)
    dphi = np.angle(z[1:] * np.conj(z[:-1])).astype(np.float32)
    audio = np.concatenate([dphi[:1], dphi], axis=0)
    audio = _dc_block(audio)
    if fs is not None and deemph_tau is not None:
        audio = _deemphasis_1pole(audio, fs=fs, tau=deemph_tau)
    return _normalize_peak(audio)


def demodulate_analog(
    iq: np.ndarray,
    mod_type: str,
    fs: Optional[float] = None,
    carrier_phase: Optional[float] = None,
) -> np.ndarray:
    """
    Dispatcher for dataset analog labels.

    Supported labels:
    - AM-DSB, AM-DSB-WC, AM-DSB-SC
    - AM-SSB, AM-SSB-WC, AM-SSB-SC
    - FM, WBFM
    """
    mod = mod_type.upper()
    if mod in {"AM-DSB", "AM-DSB-WC", "AM-DSB-SC"}:
        return demod_am_dsb(iq)
    if mod in {"AM-SSB", "AM-SSB-WC", "AM-SSB-SC"}:
        return demod_am_ssb(iq, carrier_phase=carrier_phase)
    if mod in {"FM", "WBFM"}:
        return demod_fm(iq, fs=fs, deemph_tau=75e-6)
    raise ValueError(f"Unsupported analog modulation: {mod_type}")

