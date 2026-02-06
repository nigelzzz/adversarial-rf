"""
Multi-attack evaluation with FFT recovery comparison by modulation and SNR.

Evaluates multiple attacks (fgsm, pgd, bim, cw, deepfool, etc.) and compares:
- Attack accuracy (after attack, before recovery)
- Top-10 FFT recovery accuracy
- Top-20 FFT recovery accuracy

Broken down by every modulation type and every SNR level.
"""

import os
from typing import List, Dict, Optional, Tuple, Any
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt

from util.utils import recover_constellation

try:
    import torchattacks
except ImportError:
    torchattacks = None

from util.adv_attack import (
    Model01Wrapper, iq_to_ta_input, ta_output_to_iq,
    iq_to_ta_input_minmax, ta_output_to_iq_minmax,
)
from util.defense import fft_topk_denoise_normalized


def plot_freq_comparison(
    x_clean: torch.Tensor,
    x_adv: torch.Tensor,
    attack_name: str,
    snr: int,
    mod: str,
    save_dir: str,
    n_samples: int = 5,
) -> None:
    """
    Plot frequency domain comparison between clean and adversarial signals.

    Args:
        x_clean: Clean signals [N, 2, T]
        x_adv: Adversarial signals [N, 2, T]
        attack_name: Name of the attack
        snr: SNR value
        mod: Modulation type
        save_dir: Directory to save plots
        n_samples: Number of samples to plot
    """
    os.makedirs(save_dir, exist_ok=True)

    x_clean = x_clean.cpu().numpy()
    x_adv = x_adv.cpu().numpy()
    n_samples = min(n_samples, x_clean.shape[0])

    T = x_clean.shape[2]
    freqs = np.fft.fftfreq(T)[:T // 2]  # Positive frequencies only

    for i in range(n_samples):
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        fig.suptitle(f'{attack_name.upper()} | {mod} | SNR={snr}dB | Sample {i+1}', fontsize=14)

        for ch, ch_name in enumerate(['I', 'Q']):
            # Compute FFT magnitudes
            fft_clean = np.fft.fft(x_clean[i, ch, :])
            fft_adv = np.fft.fft(x_adv[i, ch, :])
            fft_delta = fft_adv - fft_clean

            mag_clean = np.abs(fft_clean)[:T // 2]
            mag_adv = np.abs(fft_adv)[:T // 2]
            mag_delta = np.abs(fft_delta)[:T // 2]

            # Plot clean spectrum
            axes[ch, 0].plot(freqs, 20 * np.log10(mag_clean + 1e-10), 'b-', linewidth=0.8)
            axes[ch, 0].set_title(f'{ch_name} Channel - Clean')
            axes[ch, 0].set_xlabel('Normalized Frequency')
            axes[ch, 0].set_ylabel('Magnitude (dB)')
            axes[ch, 0].grid(True, alpha=0.3)

            # Plot adversarial spectrum
            axes[ch, 1].plot(freqs, 20 * np.log10(mag_adv + 1e-10), 'r-', linewidth=0.8)
            axes[ch, 1].set_title(f'{ch_name} Channel - Adversarial')
            axes[ch, 1].set_xlabel('Normalized Frequency')
            axes[ch, 1].set_ylabel('Magnitude (dB)')
            axes[ch, 1].grid(True, alpha=0.3)

            # Plot difference (perturbation spectrum)
            axes[ch, 2].plot(freqs, 20 * np.log10(mag_delta + 1e-10), 'g-', linewidth=0.8)
            axes[ch, 2].set_title(f'{ch_name} Channel - Perturbation')
            axes[ch, 2].set_xlabel('Normalized Frequency')
            axes[ch, 2].set_ylabel('Magnitude (dB)')
            axes[ch, 2].grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(save_dir, f'{attack_name}_{mod}_snr{snr}_sample{i+1}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


def plot_avg_freq_comparison(
    x_clean: torch.Tensor,
    x_adv: torch.Tensor,
    attack_name: str,
    snr: int,
    mod: str,
    save_dir: str,
) -> None:
    """
    Plot average frequency domain comparison across all samples.

    Args:
        x_clean: Clean signals [N, 2, T]
        x_adv: Adversarial signals [N, 2, T]
        attack_name: Name of the attack
        snr: SNR value
        mod: Modulation type
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)

    x_clean = x_clean.cpu().numpy()
    x_adv = x_adv.cpu().numpy()

    N, C, T = x_clean.shape
    freqs = np.fft.fftfreq(T)[:T // 2]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(f'{attack_name.upper()} | {mod} | SNR={snr}dB | Average over {N} samples', fontsize=14)

    for ch, ch_name in enumerate(['I', 'Q']):
        # Compute average FFT magnitudes
        mag_clean_all = []
        mag_adv_all = []
        mag_delta_all = []

        for i in range(N):
            fft_clean = np.fft.fft(x_clean[i, ch, :])
            fft_adv = np.fft.fft(x_adv[i, ch, :])
            fft_delta = fft_adv - fft_clean

            mag_clean_all.append(np.abs(fft_clean)[:T // 2])
            mag_adv_all.append(np.abs(fft_adv)[:T // 2])
            mag_delta_all.append(np.abs(fft_delta)[:T // 2])

        mag_clean_avg = np.mean(mag_clean_all, axis=0)
        mag_adv_avg = np.mean(mag_adv_all, axis=0)
        mag_delta_avg = np.mean(mag_delta_all, axis=0)

        # Plot clean spectrum
        axes[ch, 0].plot(freqs, 20 * np.log10(mag_clean_avg + 1e-10), 'b-', linewidth=1.0)
        axes[ch, 0].set_title(f'{ch_name} Channel - Clean (avg)')
        axes[ch, 0].set_xlabel('Normalized Frequency')
        axes[ch, 0].set_ylabel('Magnitude (dB)')
        axes[ch, 0].grid(True, alpha=0.3)

        # Plot adversarial spectrum
        axes[ch, 1].plot(freqs, 20 * np.log10(mag_adv_avg + 1e-10), 'r-', linewidth=1.0)
        axes[ch, 1].set_title(f'{ch_name} Channel - Adversarial (avg)')
        axes[ch, 1].set_xlabel('Normalized Frequency')
        axes[ch, 1].set_ylabel('Magnitude (dB)')
        axes[ch, 1].grid(True, alpha=0.3)

        # Plot difference (perturbation spectrum)
        axes[ch, 2].plot(freqs, 20 * np.log10(mag_delta_avg + 1e-10), 'g-', linewidth=1.0)
        axes[ch, 2].set_title(f'{ch_name} Channel - Perturbation (avg)')
        axes[ch, 2].set_xlabel('Normalized Frequency')
        axes[ch, 2].set_ylabel('Magnitude (dB)')
        axes[ch, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, f'{attack_name}_{mod}_snr{snr}_avg.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_overlay_comparison(
    x_clean: torch.Tensor,
    x_adv: torch.Tensor,
    attack_name: str,
    snr: int,
    mod: str,
    save_dir: str,
) -> None:
    """
    Plot overlaid clean vs adversarial spectra for easy comparison.

    Args:
        x_clean: Clean signals [N, 2, T]
        x_adv: Adversarial signals [N, 2, T]
        attack_name: Name of the attack
        snr: SNR value
        mod: Modulation type
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)

    x_clean = x_clean.cpu().numpy()
    x_adv = x_adv.cpu().numpy()

    N, C, T = x_clean.shape
    freqs = np.fft.fftfreq(T)[:T // 2]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'{attack_name.upper()} | {mod} | SNR={snr}dB | Overlay (avg of {N} samples)', fontsize=14)

    for ch, ch_name in enumerate(['I', 'Q']):
        mag_clean_all = []
        mag_adv_all = []

        for i in range(N):
            fft_clean = np.fft.fft(x_clean[i, ch, :])
            fft_adv = np.fft.fft(x_adv[i, ch, :])
            mag_clean_all.append(np.abs(fft_clean)[:T // 2])
            mag_adv_all.append(np.abs(fft_adv)[:T // 2])

        mag_clean_avg = np.mean(mag_clean_all, axis=0)
        mag_adv_avg = np.mean(mag_adv_all, axis=0)

        axes[ch].plot(freqs, 20 * np.log10(mag_clean_avg + 1e-10), 'b-',
                      linewidth=1.2, label='Clean', alpha=0.8)
        axes[ch].plot(freqs, 20 * np.log10(mag_adv_avg + 1e-10), 'r--',
                      linewidth=1.2, label='Adversarial', alpha=0.8)
        axes[ch].set_title(f'{ch_name} Channel')
        axes[ch].set_xlabel('Normalized Frequency')
        axes[ch].set_ylabel('Magnitude (dB)')
        axes[ch].legend()
        axes[ch].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, f'{attack_name}_{mod}_snr{snr}_overlay.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_iq_distribution(
    x_clean: torch.Tensor,
    x_adv: torch.Tensor,
    attack_name: str,
    snr: int,
    mod: str,
    save_dir: str,
    n_samples: int = 5,
    sps: int = 8,
) -> None:
    """
    Plot IQ scatter distribution comparing clean and adversarial signals.
    Shows both raw IQ trajectory (top row) and recovered constellation (bottom row).

    Args:
        x_clean: Clean signals [N, 2, T]
        x_adv: Adversarial signals [N, 2, T]
        attack_name: Name of the attack
        snr: SNR value
        mod: Modulation type
        save_dir: Directory to save plots
        n_samples: Number of samples to plot individually
        sps: Samples per symbol (8 for RML2016.10a)
    """
    os.makedirs(save_dir, exist_ok=True)

    x_clean = x_clean.detach().cpu().numpy()
    x_adv = x_adv.detach().cpu().numpy()
    n_samples = min(n_samples, x_clean.shape[0])

    for i in range(n_samples):
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'{attack_name.upper()} | {mod} | SNR={snr}dB | Sample {i+1}', fontsize=14)

        I_clean, Q_clean = x_clean[i, 0, :], x_clean[i, 1, :]
        I_adv, Q_adv = x_adv[i, 0, :], x_adv[i, 1, :]

        Ic_sym, Qc_sym = recover_constellation(I_clean, Q_clean, sps=sps)
        Ia_sym, Qa_sym = recover_constellation(I_adv, Q_adv, sps=sps)

        # Top row: Raw IQ
        axes[0, 0].scatter(I_clean, Q_clean, s=3, alpha=0.6, c='blue')
        axes[0, 0].set_title('Clean IQ (raw)')
        axes[0, 0].set_xlabel('I')
        axes[0, 0].set_ylabel('Q')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].axis('equal')

        axes[0, 1].scatter(I_adv, Q_adv, s=3, alpha=0.6, c='red')
        axes[0, 1].set_title('Adversarial IQ (raw)')
        axes[0, 1].set_xlabel('I')
        axes[0, 1].set_ylabel('Q')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].axis('equal')

        axes[0, 2].scatter(I_clean, Q_clean, s=3, alpha=0.5, c='blue', label='Clean')
        axes[0, 2].scatter(I_adv, Q_adv, s=3, alpha=0.5, c='red', label='Adversarial')
        axes[0, 2].set_title('Overlay (raw)')
        axes[0, 2].set_xlabel('I')
        axes[0, 2].set_ylabel('Q')
        axes[0, 2].legend(markerscale=3)
        axes[0, 2].grid(True, alpha=0.3)
        axes[0, 2].axis('equal')

        # Bottom row: Recovered constellation
        axes[1, 0].scatter(Ic_sym, Qc_sym, s=15, alpha=0.7, c='blue')
        axes[1, 0].set_title('Clean Constellation')
        axes[1, 0].set_xlabel('I')
        axes[1, 0].set_ylabel('Q')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].axis('equal')

        axes[1, 1].scatter(Ia_sym, Qa_sym, s=15, alpha=0.7, c='red')
        axes[1, 1].set_title('Adversarial Constellation')
        axes[1, 1].set_xlabel('I')
        axes[1, 1].set_ylabel('Q')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].axis('equal')

        axes[1, 2].scatter(Ic_sym, Qc_sym, s=15, alpha=0.5, c='blue', label='Clean')
        axes[1, 2].scatter(Ia_sym, Qa_sym, s=15, alpha=0.5, c='red', label='Adversarial')
        axes[1, 2].set_title('Overlay (constellation)')
        axes[1, 2].set_xlabel('I')
        axes[1, 2].set_ylabel('Q')
        axes[1, 2].legend(markerscale=2)
        axes[1, 2].grid(True, alpha=0.3)
        axes[1, 2].axis('equal')

        plt.tight_layout()
        save_path = os.path.join(save_dir, f'{attack_name}_{mod}_snr{snr}_iq_sample{i+1}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


def plot_iq_distribution_all(
    x_clean: torch.Tensor,
    x_adv: torch.Tensor,
    attack_name: str,
    snr: int,
    mod: str,
    save_dir: str,
    sps: int = 8,
) -> None:
    """
    Plot aggregated IQ scatter distribution across all samples.
    Shows both raw IQ (top) and recovered constellation (bottom).

    Args:
        x_clean: Clean signals [N, 2, T]
        x_adv: Adversarial signals [N, 2, T]
        attack_name: Name of the attack
        snr: SNR value
        mod: Modulation type
        save_dir: Directory to save plots
        sps: Samples per symbol (8 for RML2016.10a)
    """
    os.makedirs(save_dir, exist_ok=True)

    x_clean = x_clean.cpu().numpy()
    x_adv = x_adv.cpu().numpy()
    N = x_clean.shape[0]

    # Flatten raw
    I_clean = x_clean[:, 0, :].flatten()
    Q_clean = x_clean[:, 1, :].flatten()
    I_adv = x_adv[:, 0, :].flatten()
    Q_adv = x_adv[:, 1, :].flatten()

    # Recover constellation per-sample
    Ic_syms, Qc_syms, Ia_syms, Qa_syms = [], [], [], []
    for j in range(N):
        ic, qc = recover_constellation(x_clean[j, 0, :], x_clean[j, 1, :], sps=sps)
        ia, qa = recover_constellation(x_adv[j, 0, :], x_adv[j, 1, :], sps=sps)
        Ic_syms.append(ic)
        Qc_syms.append(qc)
        Ia_syms.append(ia)
        Qa_syms.append(qa)
    Ic_sym = np.concatenate(Ic_syms)
    Qc_sym = np.concatenate(Qc_syms)
    Ia_sym = np.concatenate(Ia_syms)
    Qa_sym = np.concatenate(Qa_syms)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'{attack_name.upper()} | {mod} | SNR={snr}dB | All {N} samples', fontsize=14)

    max_points = 10000

    def subsample(I, Q, mx=max_points):
        if len(I) > mx:
            idx = np.random.choice(len(I), mx, replace=False)
            return I[idx], Q[idx]
        return I, Q

    Icp, Qcp = subsample(I_clean, Q_clean)
    Iap, Qap = subsample(I_adv, Q_adv)
    Ics, Qcs = subsample(Ic_sym, Qc_sym)
    Ias, Qas = subsample(Ia_sym, Qa_sym)

    # Top row: Raw IQ
    axes[0, 0].scatter(Icp, Qcp, s=1, alpha=0.3, c='blue')
    axes[0, 0].set_title('Clean IQ (raw)')
    axes[0, 0].set_xlabel('I')
    axes[0, 0].set_ylabel('Q')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axis('equal')

    axes[0, 1].scatter(Iap, Qap, s=1, alpha=0.3, c='red')
    axes[0, 1].set_title('Adversarial IQ (raw)')
    axes[0, 1].set_xlabel('I')
    axes[0, 1].set_ylabel('Q')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axis('equal')

    axes[0, 2].scatter(Icp, Qcp, s=1, alpha=0.3, c='blue', label='Clean')
    axes[0, 2].scatter(Iap, Qap, s=1, alpha=0.3, c='red', label='Adversarial')
    axes[0, 2].set_title('Overlay (raw)')
    axes[0, 2].set_xlabel('I')
    axes[0, 2].set_ylabel('Q')
    axes[0, 2].legend(markerscale=5)
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].axis('equal')

    # Bottom row: Recovered constellation
    axes[1, 0].scatter(Ics, Qcs, s=5, alpha=0.4, c='blue')
    axes[1, 0].set_title('Clean Constellation')
    axes[1, 0].set_xlabel('I')
    axes[1, 0].set_ylabel('Q')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axis('equal')

    axes[1, 1].scatter(Ias, Qas, s=5, alpha=0.4, c='red')
    axes[1, 1].set_title('Adversarial Constellation')
    axes[1, 1].set_xlabel('I')
    axes[1, 1].set_ylabel('Q')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axis('equal')

    axes[1, 2].scatter(Ics, Qcs, s=5, alpha=0.3, c='blue', label='Clean')
    axes[1, 2].scatter(Ias, Qas, s=5, alpha=0.3, c='red', label='Adversarial')
    axes[1, 2].set_title('Overlay (constellation)')
    axes[1, 2].set_xlabel('I')
    axes[1, 2].set_ylabel('Q')
    axes[1, 2].legend(markerscale=5)
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].axis('equal')

    plt.tight_layout()
    save_path = os.path.join(save_dir, f'{attack_name}_{mod}_snr{snr}_iq_all.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_iq_density(
    x_clean: torch.Tensor,
    x_adv: torch.Tensor,
    attack_name: str,
    snr: int,
    mod: str,
    save_dir: str,
    bins: int = 100,
) -> None:
    """
    Plot 2D histogram (density) of IQ distribution.

    Args:
        x_clean: Clean signals [N, 2, T]
        x_adv: Adversarial signals [N, 2, T]
        attack_name: Name of the attack
        snr: SNR value
        mod: Modulation type
        save_dir: Directory to save plots
        bins: Number of bins for histogram
    """
    os.makedirs(save_dir, exist_ok=True)

    x_clean = x_clean.cpu().numpy()
    x_adv = x_adv.cpu().numpy()
    N = x_clean.shape[0]

    # Flatten all samples
    I_clean = x_clean[:, 0, :].flatten()
    Q_clean = x_clean[:, 1, :].flatten()
    I_adv = x_adv[:, 0, :].flatten()
    Q_adv = x_adv[:, 1, :].flatten()

    # Compute common range
    all_I = np.concatenate([I_clean, I_adv])
    all_Q = np.concatenate([Q_clean, Q_adv])
    range_I = [all_I.min(), all_I.max()]
    range_Q = [all_Q.min(), all_Q.max()]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'{attack_name.upper()} | {mod} | SNR={snr}dB | IQ Density ({N} samples)', fontsize=14)

    # Clean density
    h1 = axes[0].hist2d(I_clean, Q_clean, bins=bins, range=[range_I, range_Q], cmap='Blues')
    axes[0].set_title('Clean IQ Density')
    axes[0].set_xlabel('I (In-phase)')
    axes[0].set_ylabel('Q (Quadrature)')
    plt.colorbar(h1[3], ax=axes[0])

    # Adversarial density
    h2 = axes[1].hist2d(I_adv, Q_adv, bins=bins, range=[range_I, range_Q], cmap='Reds')
    axes[1].set_title('Adversarial IQ Density')
    axes[1].set_xlabel('I (In-phase)')
    axes[1].set_ylabel('Q (Quadrature)')
    plt.colorbar(h2[3], ax=axes[1])

    # Difference density
    H_clean, xedges, yedges = np.histogram2d(I_clean, Q_clean, bins=bins, range=[range_I, range_Q])
    H_adv, _, _ = np.histogram2d(I_adv, Q_adv, bins=bins, range=[range_I, range_Q])
    H_diff = H_adv - H_clean

    im = axes[2].imshow(H_diff.T, origin='lower', extent=[range_I[0], range_I[1], range_Q[0], range_Q[1]],
                        aspect='auto', cmap='RdBu_r')
    axes[2].set_title('Density Difference (Adv - Clean)')
    axes[2].set_xlabel('I (In-phase)')
    axes[2].set_ylabel('Q (Quadrature)')
    plt.colorbar(im, ax=axes[2])

    plt.tight_layout()
    save_path = os.path.join(save_dir, f'{attack_name}_{mod}_snr{snr}_iq_density.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# Default attack list (17 attacks including EADL1 and EADEN)
DEFAULT_ATTACKS = [
    'fgsm', 'pgd', 'bim', 'cw', 'deepfool', 'apgd', 'mifgsm',
    'rfgsm', 'upgd', 'eotpgd', 'vmifgsm', 'vnifgsm', 'jitter', 'ffgsm', 'pgdl2',
    'eadl1', 'eaden'
]

# Attacks that require fixed batch sizes (will be handled with padding)
FIXED_BATCH_ATTACKS = {'apgd'}


def build_snr_mod_index(
    SNRs: List[int],
    Labels: torch.Tensor,
    test_idx: np.ndarray,
    cfg,
) -> Dict[Tuple[int, str], List[int]]:
    """
    Build index mapping (snr, mod_name) -> [sample_indices].

    Args:
        SNRs: List of SNR values for all samples (full dataset)
        Labels: Label tensor for test samples (shape [N_test])
        test_idx: Indices of test samples in original dataset
        cfg: Config with class mappings

    Returns:
        Dictionary mapping (snr, mod_name) tuples to lists of test set indices
    """
    # Build reverse class mapping: idx -> mod_name (decoded string)
    idx_to_mod = {}
    for k, v in cfg.classes.items():
        if isinstance(k, bytes):
            idx_to_mod[v] = k.decode()
        else:
            idx_to_mod[v] = str(k)

    # Build the index
    snr_mod_idx = defaultdict(list)
    for i, orig_idx in enumerate(test_idx):
        snr = SNRs[orig_idx]
        label = int(Labels[i].item())
        mod_name = idx_to_mod.get(label, f"class_{label}")
        snr_mod_idx[(snr, mod_name)].append(i)

    return dict(snr_mod_idx)


def create_attack(
    attack_name: str,
    wrapped_model: nn.Module,
    cfg,
) -> Any:
    """
    Factory to create torchattacks attack objects.

    Args:
        attack_name: Name of attack (fgsm, pgd, cw, etc.)
        wrapped_model: Model wrapped with Model01Wrapper
        cfg: Config object

    Returns:
        torchattacks attack object
    """
    if torchattacks is None:
        raise ImportError("torchattacks not installed. Run: pip install torchattacks")

    name = attack_name.lower()

    # Get epsilon from config (default 0.03 for IQ data)
    # Note: IQ signals have typical amplitude ~0.02 in [-1,1] space, so eps=0.03 is appropriate
    # The old default of 0.3 was 15x larger than signal amplitude, overwhelming the attack
    eps = getattr(cfg, 'attack_eps', 0.03)

    # Warn if epsilon seems too large for IQ data
    if eps > 0.1:
        import logging
        logging.getLogger(__name__).warning(
            f"attack_eps={eps} may be too large for IQ data (typical amplitude ~0.02). "
            f"Consider --ta_box minmax or smaller eps (e.g., 0.03)."
        )
    alpha = eps / 4  # step size typically eps/4 for iterative attacks
    steps = 10

    if name == 'fgsm':
        return torchattacks.FGSM(wrapped_model, eps=eps)

    elif name == 'pgd':
        return torchattacks.PGD(wrapped_model, eps=eps, alpha=alpha, steps=steps)

    elif name == 'bim':
        return torchattacks.BIM(wrapped_model, eps=eps, alpha=alpha, steps=steps)

    elif name == 'cw':
        # Hardcoded CW parameters (or use cfg values as fallback)
        c = 10.0         # Confidence: higher = stronger attack
        cw_steps = 200   # Optimization steps
        cw_lr = 0.005    # Learning rate
        return torchattacks.CW(wrapped_model, c=c, steps=cw_steps, lr=cw_lr)

    elif name == 'deepfool':
        return torchattacks.DeepFool(wrapped_model, steps=50, overshoot=0.02)

    elif name == 'apgd':
        # APGD can have batch size issues - use n_restarts=1 for stability
        return torchattacks.APGD(wrapped_model, eps=eps, steps=steps, n_restarts=1)

    elif name == 'mifgsm':
        return torchattacks.MIFGSM(wrapped_model, eps=eps, steps=steps, decay=1.0)

    elif name == 'rfgsm':
        return torchattacks.RFGSM(wrapped_model, eps=eps, alpha=alpha, steps=steps)

    elif name == 'upgd':
        return torchattacks.UPGD(wrapped_model, eps=eps, steps=steps)

    elif name == 'eotpgd':
        return torchattacks.EOTPGD(wrapped_model, eps=eps, alpha=alpha, steps=steps, eot_iter=1)

    elif name == 'vmifgsm':
        return torchattacks.VMIFGSM(wrapped_model, eps=eps, steps=steps, N=5)

    elif name == 'vnifgsm':
        return torchattacks.VNIFGSM(wrapped_model, eps=eps, steps=steps, N=5)

    elif name == 'jitter':
        return torchattacks.Jitter(wrapped_model, eps=eps, alpha=alpha, steps=steps)

    elif name == 'ffgsm':
        return torchattacks.FFGSM(wrapped_model, eps=eps, alpha=eps * 1.25)

    elif name == 'pgdl2':
        # L2 norm bound scales differently; use eps * sqrt(signal_length) as rough heuristic
        eps_l2 = eps * 10  # L2 bound (larger than Linf)
        return torchattacks.PGDL2(wrapped_model, eps=eps_l2, alpha=eps_l2 / 5, steps=steps)

    elif name == 'eadl1':
        # EAD L1 attack - uses kappa (confidence), lr, max_iterations
        kappa = getattr(cfg, 'ead_kappa', 0)
        lr = getattr(cfg, 'ead_lr', 0.01)
        max_iterations = getattr(cfg, 'ead_max_iterations', 100)
        binary_search_steps = getattr(cfg, 'ead_binary_search_steps', 9)
        initial_const = getattr(cfg, 'ead_initial_const', 0.001)
        beta = getattr(cfg, 'ead_beta', 0.001)
        return torchattacks.EADL1(
            wrapped_model,
            kappa=kappa,
            lr=lr,
            max_iterations=max_iterations,
            binary_search_steps=binary_search_steps,
            initial_const=initial_const,
            beta=beta,
        )

    elif name == 'eaden':
        # EAD Elastic Net attack - uses kappa (confidence), lr, max_iterations
        kappa = getattr(cfg, 'ead_kappa', 0)
        lr = getattr(cfg, 'ead_lr', 0.01)
        max_iterations = getattr(cfg, 'ead_max_iterations', 100)
        binary_search_steps = getattr(cfg, 'ead_binary_search_steps', 9)
        initial_const = getattr(cfg, 'ead_initial_const', 0.001)
        beta = getattr(cfg, 'ead_beta', 0.001)
        return torchattacks.EADEN(
            wrapped_model,
            kappa=kappa,
            lr=lr,
            max_iterations=max_iterations,
            binary_search_steps=binary_search_steps,
            initial_const=initial_const,
            beta=beta,
        )

    else:
        raise ValueError(f"Unknown attack: {attack_name}")


def generate_adversarial(
    attack,
    x_iq: torch.Tensor,
    labels: torch.Tensor,
    wrapped_model: Optional[nn.Module] = None,
    ta_box: str = 'unit',
    pad_to_batch_size: Optional[int] = None,
    fallback_to_single: bool = True,
) -> torch.Tensor:
    """
    Generate adversarial examples using torchattacks.

    Handles IQ <-> torchattacks format conversion with support for
    different normalization modes.

    Args:
        attack: torchattacks attack object
        x_iq: Input IQ tensor [N, 2, T] in [-1, 1]
        labels: True labels [N]
        wrapped_model: Model01Wrapper instance (required for 'minmax' mode)
        ta_box: Normalization mode - 'unit' (default) or 'minmax'
            - 'unit': Maps [-1,1] to [0,1] using (x+1)/2. Simple but eps is in
              [0,1] space where signal only spans ~2% of range.
            - 'minmax': Per-sample min-max normalization to [0,1]. eps is
              relative to actual signal range, making attacks more effective.
        pad_to_batch_size: If set, pad batch to this size (for attacks like APGD
                          that need fixed batch sizes), then trim result
        fallback_to_single: If True and batch attack fails, process samples one by one

    Returns:
        Adversarial IQ tensor [N, 2, T] in [-1, 1]
    """
    original_size = x_iq.shape[0]
    box = str(ta_box).lower()

    # Pad if needed (for attacks that require fixed batch sizes)
    x_iq_padded = x_iq
    labels_padded = labels
    if pad_to_batch_size is not None and original_size < pad_to_batch_size:
        pad_size = pad_to_batch_size - original_size
        # Handle case where we need to repeat multiple times
        if pad_size > original_size:
            repeats = (pad_size // original_size) + 1
            x_pad = x_iq.repeat(repeats, 1, 1)[:pad_size]
            labels_pad = labels.repeat(repeats)[:pad_size]
        else:
            x_pad = x_iq[:pad_size]
            labels_pad = labels[:pad_size]
        x_iq_padded = torch.cat([x_iq, x_pad], dim=0)
        labels_padded = torch.cat([labels, labels_pad], dim=0)

    try:
        if box == 'minmax':
            # Per-sample min-max normalization - eps is relative to signal range
            if wrapped_model is None:
                raise ValueError("wrapped_model required for ta_box='minmax'")
            x_ta, a, b = iq_to_ta_input_minmax(x_iq_padded)  # [N, 2, T, 1] in [0, 1]
            wrapped_model.set_minmax(a, b)
            try:
                x_adv_ta = attack(x_ta, labels_padded)
                x_adv_iq = ta_output_to_iq_minmax(x_adv_ta, a, b)
            finally:
                wrapped_model.clear_minmax()
        else:
            # Unit normalization - simple [-1,1] -> [0,1] mapping
            x_ta = iq_to_ta_input(x_iq_padded)  # [N, 2, T, 1] in [0, 1]
            x_adv_ta = attack(x_ta, labels_padded)
            x_adv_iq = ta_output_to_iq(x_adv_ta)  # [N, 2, T] in [-1, 1]

        # Trim back to original size if we padded
        if pad_to_batch_size is not None and original_size < pad_to_batch_size:
            x_adv_iq = x_adv_iq[:original_size]

        return x_adv_iq

    except RuntimeError as e:
        if not fallback_to_single:
            raise
        # Fallback: process samples one by one
        # This is slower but handles problematic attacks like APGD
        adv_samples = []
        for i in range(original_size):
            x_single = x_iq[i:i+1]
            label_single = labels[i:i+1]
            try:
                if box == 'minmax':
                    x_ta_s, a_s, b_s = iq_to_ta_input_minmax(x_single)
                    wrapped_model.set_minmax(a_s, b_s)
                    try:
                        x_adv_ta_s = attack(x_ta_s, label_single)
                        x_adv_s = ta_output_to_iq_minmax(x_adv_ta_s, a_s, b_s)
                    finally:
                        wrapped_model.clear_minmax()
                else:
                    x_ta_s = iq_to_ta_input(x_single)
                    x_adv_ta_s = attack(x_ta_s, label_single)
                    x_adv_s = ta_output_to_iq(x_adv_ta_s)
                adv_samples.append(x_adv_s)
            except Exception:
                # If even single sample fails, use original
                adv_samples.append(x_single)
        return torch.cat(adv_samples, dim=0)


@torch.no_grad()
def compute_accuracy(
    model: nn.Module,
    x: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    """Compute classification accuracy."""
    logits, _ = model(x)
    preds = logits.argmax(dim=1)
    correct = (preds == labels).sum().item()
    return correct / len(labels) if len(labels) > 0 else 0.0


def run_multi_attack_snr_mod_eval(
    model: nn.Module,
    sig_test: torch.Tensor,
    lab_test: torch.Tensor,
    SNRs: List[int],
    test_idx: np.ndarray,
    cfg,
    logger,
    attacks: Optional[List[str]] = None,
    eval_limit_per_cell: Optional[int] = None,
    plot_freq: bool = False,
    plot_iq: bool = False,
    plot_n_samples: int = 3,
) -> pd.DataFrame:
    """
    Run multi-attack evaluation with FFT recovery comparison by modulation and SNR.

    Args:
        model: AWN classification model
        sig_test: Test signals tensor [N, 2, T]
        lab_test: Test labels tensor [N]
        SNRs: SNR values for all samples in original dataset
        test_idx: Indices of test samples in original dataset
        cfg: Configuration object
        logger: Logger instance
        attacks: List of attack names (defaults to all 15)
        eval_limit_per_cell: Max samples per (SNR, mod) cell
        plot_freq: If True, save frequency domain comparison plots
        plot_iq: If True, save IQ distribution comparison plots
        plot_n_samples: Number of individual samples to plot (if plot_freq/plot_iq=True)

    Returns:
        DataFrame with columns: attack, snr, modulation, n_samples, attack_acc, top10_acc, top20_acc
    """
    model.eval()
    device = cfg.device

    if attacks is None:
        attacks = DEFAULT_ATTACKS

    # Check if model has LSTM layers (cuDNN LSTM backward has issues even in training mode)
    has_lstm = any(isinstance(m, nn.LSTM) for m in model.modules())
    cudnn_was_enabled = torch.backends.cudnn.enabled
    if has_lstm:
        logger.info("Model contains LSTM layers - disabling cuDNN for adversarial generation")
        torch.backends.cudnn.enabled = False

    # Build index mapping
    logger.info("Building (SNR, modulation) index mapping...")
    snr_mod_idx = build_snr_mod_index(SNRs, lab_test, test_idx, cfg)
    logger.info(f"Found {len(snr_mod_idx)} (SNR, mod) cells")

    # Get unique SNRs and modulations for ordered iteration
    all_snrs = sorted(set(k[0] for k in snr_mod_idx.keys()))
    all_mods = sorted(set(k[1] for k in snr_mod_idx.keys()))
    logger.info(f"SNRs: {all_snrs}")
    logger.info(f"Modulations: {all_mods}")

    # Wrap model for torchattacks
    wrapped_model = Model01Wrapper(model)
    wrapped_model.to(device)
    # Note: wrapped_model will be set to train mode during attack generation if has_lstm

    # Get normalization mode from config
    ta_box = str(getattr(cfg, 'ta_box', 'unit')).lower()
    logger.info(f"Using ta_box={ta_box} normalization for torchattacks")
    if ta_box == 'minmax':
        logger.info("minmax mode: eps is relative to per-sample signal range")
    else:
        logger.info("unit mode: eps is in [0,1] space (signal uses ~2% of range)")

    results = []

    # Pre-compute clean accuracy for each (SNR, mod) cell
    # This ensures we use the exact same data for clean and attacked evaluation
    logger.info("\n=== Computing Clean Accuracy ===")
    clean_acc_cache = {}
    cell_data_cache = {}  # Cache cell data to ensure same data is used

    for snr in all_snrs:
        for mod in all_mods:
            key = (snr, mod)
            if key not in snr_mod_idx:
                continue

            indices = snr_mod_idx[key]

            # Apply limit if specified (same limit for clean and attacks)
            if eval_limit_per_cell is not None and len(indices) > eval_limit_per_cell:
                indices = indices[:eval_limit_per_cell]

            n_samples = len(indices)
            if n_samples == 0:
                continue

            # Get samples for this cell and cache them
            x_cell = sig_test[indices].to(device)
            y_cell = lab_test[indices].to(device)
            cell_data_cache[key] = (x_cell, y_cell, indices)

            # Compute clean accuracy
            clean_acc = compute_accuracy(model, x_cell, y_cell)
            clean_acc_cache[key] = clean_acc
            logger.info(f"  ({snr:3d}, {mod:8s}): clean_acc={clean_acc:.4f}, n={n_samples}")

    for attack_name in attacks:
        logger.info(f"\n=== Attack: {attack_name.upper()} ===")

        # Create attack object
        try:
            attack = create_attack(attack_name, wrapped_model, cfg)
        except Exception as e:
            logger.warning(f"Failed to create attack {attack_name}: {e}")
            continue

        # Check if this attack needs fixed batch sizes
        needs_padding = attack_name.lower() in FIXED_BATCH_ATTACKS
        # Use a reasonable padding size for cell-based evaluation
        pad_batch_size = 128 if needs_padding else None

        for snr in all_snrs:
            for mod in all_mods:
                key = (snr, mod)
                if key not in cell_data_cache:
                    continue

                # Use cached cell data (same data for all attacks)
                x_cell, y_cell, indices = cell_data_cache[key]
                n_samples = len(indices)
                clean_acc = clean_acc_cache[key]

                # Generate adversarial examples
                try:
                    x_adv = generate_adversarial(
                        attack, x_cell, y_cell,
                        wrapped_model=wrapped_model,
                        ta_box=ta_box,
                        pad_to_batch_size=pad_batch_size,
                    )
                except Exception as e:
                    logger.warning(f"Attack failed for ({snr}, {mod}): {e}")
                    continue

                # Compute attack accuracy (after attack, before recovery)
                model.eval()  # Ensure eval mode for inference
                attack_acc = compute_accuracy(model, x_adv, y_cell)

                # Plot frequency domain comparison if requested
                if plot_freq:
                    freq_plot_dir = os.path.join(cfg.result_dir, 'freq_plots')
                    logger.info(f"Generating freq plots for {attack_name}/{mod}/SNR={snr} -> {freq_plot_dir}")
                    # Plot individual samples
                    plot_freq_comparison(
                        x_cell, x_adv, attack_name, snr, mod,
                        freq_plot_dir, n_samples=plot_n_samples
                    )
                    # Plot average spectrum
                    plot_avg_freq_comparison(
                        x_cell, x_adv, attack_name, snr, mod, freq_plot_dir
                    )
                    # Plot overlay comparison
                    plot_overlay_comparison(
                        x_cell, x_adv, attack_name, snr, mod, freq_plot_dir
                    )
                    logger.info(f"Saved freq plots to {freq_plot_dir}")

                # Plot IQ distribution comparison if requested
                if plot_iq:
                    iq_plot_dir = os.path.join(cfg.result_dir, 'iq_plots')
                    logger.info(f"Generating IQ plots for {attack_name}/{mod}/SNR={snr} -> {iq_plot_dir}")
                    # Plot individual samples
                    plot_iq_distribution(
                        x_cell, x_adv, attack_name, snr, mod,
                        iq_plot_dir, n_samples=plot_n_samples
                    )
                    # Plot all samples aggregated
                    plot_iq_distribution_all(
                        x_cell, x_adv, attack_name, snr, mod, iq_plot_dir
                    )
                    # Plot density histogram
                    plot_iq_density(
                        x_cell, x_adv, attack_name, snr, mod, iq_plot_dir
                    )
                    logger.info(f"Saved IQ plots to {iq_plot_dir}")

                # Apply Top-10 FFT recovery
                x_top10 = fft_topk_denoise_normalized(
                    x_adv, topk=10,
                    norm_offset=0.02, norm_scale=0.04
                )
                top10_acc = compute_accuracy(model, x_top10, y_cell)

                # Apply Top-20 FFT recovery
                x_top20 = fft_topk_denoise_normalized(
                    x_adv, topk=20,
                    norm_offset=0.02, norm_scale=0.04
                )
                top20_acc = compute_accuracy(model, x_top20, y_cell)

                results.append({
                    'attack': attack_name,
                    'snr': snr,
                    'modulation': mod,
                    'n_samples': n_samples,
                    'clean_acc': clean_acc,
                    'attack_acc': attack_acc,
                    'top10_acc': top10_acc,
                    'top20_acc': top20_acc,
                })

        # Log summary for this attack
        attack_results = [r for r in results if r['attack'] == attack_name]
        if attack_results:
            avg_clean = np.mean([r['clean_acc'] for r in attack_results])
            avg_attack = np.mean([r['attack_acc'] for r in attack_results])
            avg_top10 = np.mean([r['top10_acc'] for r in attack_results])
            avg_top20 = np.mean([r['top20_acc'] for r in attack_results])
            logger.info(f"{attack_name}: clean={avg_clean:.4f}, attack={avg_attack:.4f}, "
                       f"top10={avg_top10:.4f}, top20={avg_top20:.4f}")

    # Create DataFrame
    df = pd.DataFrame(results)

    # Save to CSV
    os.makedirs(cfg.result_dir, exist_ok=True)
    csv_path = os.path.join(cfg.result_dir, 'multi_attack_snr_mod_eval.csv')
    df.to_csv(csv_path, index=False)
    logger.info(f"\nSaved results to: {csv_path}")

    # Log overall summary
    if len(df) > 0:
        logger.info("\n=== Overall Summary ===")
        summary = df.groupby('attack').agg({
            'clean_acc': 'mean',
            'attack_acc': 'mean',
            'top10_acc': 'mean',
            'top20_acc': 'mean',
            'n_samples': 'sum'
        }).round(4)
        logger.info(f"\n{summary.to_string()}")

    # Restore cuDNN state if it was modified
    if has_lstm:
        torch.backends.cudnn.enabled = cudnn_was_enabled

    return df
