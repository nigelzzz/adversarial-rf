"""
SigGuard-style evaluation: Compare attack accuracy with/without FFT Top-K defense.

Produces a table similar to:
| Sample Type | Disabled | Enabled |
|-------------|----------|---------|
| Intact      | 92.61%   | 92.20%  |
| CW          | 0.86%    | 80.43%  |
| FGSM        | 7.20%    | 9.32%   |
| ...         | ...      | ...     |

Also generates IQ distribution plots comparing clean vs adversarial signals.
"""

import os
from typing import List, Dict, Optional, Any
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from util.utils import recover_constellation

try:
    import torchattacks
except ImportError:
    torchattacks = None

from util.adv_attack import (
    Model01Wrapper, iq_to_ta_input, ta_output_to_iq,
    iq_to_ta_input_minmax, ta_output_to_iq_minmax,
    iq_to_ta_input_paper, ta_output_to_iq_paper,
)
from util.defense import fft_topk_denoise, fft_adaptive_topk_denoise
from util.adaptive_defense import (
    confidence_sweep_topk_denoise,
    classify_then_filter_topk_denoise,
    spectral_shape_topk_denoise,
)


# Phase order for M-th power phase de-rotation
PHASE_ORDER = {
    'BPSK': 2, 'QPSK': 4, '8PSK': 8, 'QAM16': 4, 'QAM64': 4,
    'PAM4': 2, 'WBFM': 1, 'AM-DSB': 1, 'AM-SSB': 1, 'CPFSK': 4, 'GFSK': 4,
}

# All 17 attacks (15 original + EADL1 + EADEN)
ALL_ATTACKS = [
    'fgsm', 'pgd', 'bim', 'cw', 'deepfool', 'apgd', 'mifgsm',
    'rfgsm', 'upgd', 'eotpgd', 'vmifgsm', 'vnifgsm', 'jitter', 'ffgsm', 'pgdl2',
    'eadl1', 'eaden'
]

# Attacks that require fixed batch sizes (will be handled specially)
FIXED_BATCH_ATTACKS = {'apgd'}


def create_attack(
    attack_name: str,
    wrapped_model: nn.Module,
    cfg,
) -> Any:
    """
    Create attack objects for evaluation.

    Supports 17 torchattacks: fgsm, pgd, bim, cw, deepfool, apgd, mifgsm,
    rfgsm, upgd, eotpgd, vmifgsm, vnifgsm, jitter, ffgsm, pgdl2, eadl1, eaden
    """
    if torchattacks is None:
        raise ImportError("torchattacks not installed. Run: pip install torchattacks")

    name = attack_name.lower()
    eps = getattr(cfg, 'attack_eps', 0.03)
    alpha = eps / 4
    steps = 10

    if name == 'fgsm':
        return torchattacks.FGSM(wrapped_model, eps=eps)

    elif name == 'pgd':
        return torchattacks.PGD(wrapped_model, eps=eps, alpha=alpha, steps=steps)

    elif name == 'bim':
        return torchattacks.BIM(wrapped_model, eps=eps, alpha=alpha, steps=steps)

    elif name == 'cw':
        c = getattr(cfg, 'cw_c', 10.0)
        cw_steps = getattr(cfg, 'cw_steps', 200)
        cw_lr = getattr(cfg, 'cw_lr', 0.005)
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
        eps_l2 = eps * 10
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

    elif name == 'fab':
        # FAB (Fast Adaptive Boundary) attack - minimum norm adversarial
        fab_steps = getattr(cfg, 'fab_steps', 100)
        fab_n_restarts = getattr(cfg, 'fab_n_restarts', 1)
        return torchattacks.FAB(
            wrapped_model,
            eps=eps,
            steps=fab_steps,
            n_restarts=fab_n_restarts,
            n_classes=getattr(cfg, 'num_classes', 11),
        )

    else:
        raise ValueError(f"Unknown attack: {attack_name}")


def _run_attack_single(attack, x_ta, labels, wrapped_model, box, x_iq):
    """Run attack on a single sample or batch."""
    if box == 'minmax':
        x_ta_single, a, b = iq_to_ta_input_minmax(x_iq)
        wrapped_model.set_minmax(a, b)
        try:
            x_adv_ta = attack(x_ta_single, labels)
            x_adv_iq = ta_output_to_iq_minmax(x_adv_ta, a, b)
        finally:
            wrapped_model.clear_minmax()
    else:
        x_adv_ta = attack(x_ta, labels)
        x_adv_iq = ta_output_to_iq(x_adv_ta)
    return x_adv_iq


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
    Generate adversarial examples with support for unit/minmax normalization.

    Args:
        attack: torchattacks attack object
        x_iq: Input IQ tensor [N, 2, T]
        labels: True labels [N]
        wrapped_model: Model01Wrapper instance (required for 'minmax' mode)
        ta_box: Normalization mode - 'unit' or 'minmax'
        pad_to_batch_size: If set, pad batch to this size (for attacks like APGD
                          that need fixed batch sizes), then trim result
        fallback_to_single: If True and batch attack fails, process samples one by one

    Returns:
        Adversarial IQ tensor [N, 2, T]
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
            if wrapped_model is None:
                raise ValueError("wrapped_model required for ta_box='minmax'")
            x_ta, a, b = iq_to_ta_input_minmax(x_iq_padded)
            wrapped_model.set_minmax(a, b)
            try:
                x_adv_ta = attack(x_ta, labels_padded)
                x_adv_iq = ta_output_to_iq_minmax(x_adv_ta, a, b)
            finally:
                wrapped_model.clear_minmax()
        elif box == 'paper':
            if wrapped_model is None:
                raise ValueError("wrapped_model required for ta_box='paper'")
            a = torch.tensor(-0.02, device=x_iq_padded.device).reshape(1, 1, 1)
            b = torch.tensor(0.04, device=x_iq_padded.device).reshape(1, 1, 1)
            wrapped_model.set_minmax(a, b)
            try:
                x_ta = iq_to_ta_input_paper(x_iq_padded)
                x_adv_ta = attack(x_ta, labels_padded)
                x_adv_iq = ta_output_to_iq_paper(x_adv_ta)
            finally:
                wrapped_model.clear_minmax()
        else:
            x_ta = iq_to_ta_input(x_iq_padded)
            x_adv_ta = attack(x_ta, labels_padded)
            x_adv_iq = ta_output_to_iq(x_adv_ta)

        # Trim back to original size if we padded
        if pad_to_batch_size is not None and original_size < pad_to_batch_size:
            x_adv_iq = x_adv_iq[:original_size]

        return x_adv_iq

    except RuntimeError as e:
        if not fallback_to_single:
            raise
        # Fallback: process samples one by one
        # This is slower but handles problematic attacks like APGD
        device = x_iq.device
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
                elif box == 'paper':
                    a_p = torch.tensor(-0.02, device=x_single.device).reshape(1, 1, 1)
                    b_p = torch.tensor(0.04, device=x_single.device).reshape(1, 1, 1)
                    wrapped_model.set_minmax(a_p, b_p)
                    try:
                        x_ta_s = iq_to_ta_input_paper(x_single)
                        x_adv_ta_s = attack(x_ta_s, label_single)
                        x_adv_s = ta_output_to_iq_paper(x_adv_ta_s)
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
    model.eval()
    logits, _ = model(x)
    preds = logits.argmax(dim=1)
    correct = (preds == labels).sum().item()
    return correct / len(labels) if len(labels) > 0 else 0.0


def plot_iq_distribution(
    x_clean: torch.Tensor,
    x_adv: torch.Tensor,
    attack_name: str,
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
        save_dir: Directory to save plots
        n_samples: Number of samples to plot
        sps: Samples per symbol (8 for RML2016.10a)
    """
    os.makedirs(save_dir, exist_ok=True)

    x_clean = x_clean.detach().cpu().numpy()
    x_adv = x_adv.detach().cpu().numpy()
    n_samples = min(n_samples, x_clean.shape[0])

    # Plot individual samples
    for i in range(n_samples):
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'{attack_name.upper()} | Sample {i+1}', fontsize=14)

        I_clean, Q_clean = x_clean[i, 0, :], x_clean[i, 1, :]
        I_adv, Q_adv = x_adv[i, 0, :], x_adv[i, 1, :]

        # Recover constellation points
        Ic_sym, Qc_sym = recover_constellation(I_clean, Q_clean, sps=sps)
        Ia_sym, Qa_sym = recover_constellation(I_adv, Q_adv, sps=sps)

        # --- Top row: Raw IQ trajectory ---
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

        # --- Bottom row: Recovered constellation ---
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
        save_path = os.path.join(save_dir, f'{attack_name}_iq_sample{i+1}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


def plot_iq_distribution_all(
    x_clean: torch.Tensor,
    x_adv: torch.Tensor,
    attack_name: str,
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
        save_dir: Directory to save plots
        sps: Samples per symbol (8 for RML2016.10a)
    """
    os.makedirs(save_dir, exist_ok=True)

    x_clean = x_clean.detach().cpu().numpy()
    x_adv = x_adv.detach().cpu().numpy()
    N = x_clean.shape[0]

    # Flatten all samples (raw)
    I_clean = x_clean[:, 0, :].flatten()
    Q_clean = x_clean[:, 1, :].flatten()
    I_adv = x_adv[:, 0, :].flatten()
    Q_adv = x_adv[:, 1, :].flatten()

    # Recover constellation per-sample and collect
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
    fig.suptitle(f'{attack_name.upper()} | All {N} samples', fontsize=14)

    # Subsample for plotting if too many points
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

    # --- Top row: Raw IQ ---
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

    # --- Bottom row: Recovered constellation ---
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
    save_path = os.path.join(save_dir, f'{attack_name}_iq_all.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_iq_density(
    x_clean: torch.Tensor,
    x_adv: torch.Tensor,
    attack_name: str,
    save_dir: str,
    bins: int = 100,
) -> None:
    """
    Plot 2D histogram (density) of IQ distribution.

    Args:
        x_clean: Clean signals [N, 2, T]
        x_adv: Adversarial signals [N, 2, T]
        attack_name: Name of the attack
        save_dir: Directory to save plots
        bins: Number of bins for histogram
    """
    os.makedirs(save_dir, exist_ok=True)

    x_clean = x_clean.detach().cpu().numpy()
    x_adv = x_adv.detach().cpu().numpy()
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
    fig.suptitle(f'{attack_name.upper()} | IQ Density ({N} samples)', fontsize=14)

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
    save_path = os.path.join(save_dir, f'{attack_name}_iq_density.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def _phase_align_clean(clean_np, mod_name):
    """Per-sample M-th power phase de-rotation (no RMS normalization)."""
    N = clean_np.shape[0]
    order = PHASE_ORDER.get(mod_name, 4)
    aligned = np.empty_like(clean_np)
    phases = []
    for k in range(N):
        iq = clean_np[k, 0, :] + 1j * clean_np[k, 1, :]
        if order > 1:
            phase_est = np.angle(np.mean(iq ** order)) / order
        else:
            phase_est = 0.0
        iq = iq * np.exp(-1j * phase_est)
        aligned[k, 0, :] = iq.real
        aligned[k, 1, :] = iq.imag
        phases.append(phase_est)
    return aligned, phases


def _apply_clean_phase(adv_np, phases):
    """Apply CLEAN signal's phase de-rotation to adversarial."""
    N = adv_np.shape[0]
    aligned = np.empty_like(adv_np)
    for k in range(N):
        iq = adv_np[k, 0, :] + 1j * adv_np[k, 1, :]
        iq = iq * np.exp(-1j * phases[k])
        aligned[k, 0, :] = iq.real
        aligned[k, 1, :] = iq.imag
    return aligned


def plot_iq_constellation_grid(
    x_clean: torch.Tensor,
    x_adv: torch.Tensor,
    labels: torch.Tensor,
    attack_name: str,
    save_dir: str,
    idx_to_mod: Dict[int, str],
    max_points: int = 15000,
) -> None:
    """
    Plot per-modulation IQ constellation: clean vs adversarial side-by-side.

    Creates a combined grid (N_mods x 2) and individual per-mod plots,
    using phase de-rotation from clean reference and raw IQ axis [-0.025, 0.025].

    Args:
        x_clean: Clean signals [N, 2, T]
        x_adv: Adversarial signals [N, 2, T]
        labels: Class labels [N]
        attack_name: Name of the attack
        save_dir: Directory to save plots
        idx_to_mod: Mapping from class index to modulation name
        max_points: Max scatter points per subplot
    """
    os.makedirs(save_dir, exist_ok=True)

    clean_np = x_clean.detach().cpu().numpy()
    adv_np = x_adv.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()

    # Find unique modulations present
    unique_labels = sorted(np.unique(labels_np).astype(int))
    mod_list = [(idx, idx_to_mod.get(idx, f'cls{idx}')) for idx in unique_labels]
    n_mods = len(mod_list)

    if n_mods == 0:
        return

    # Combined grid figure
    fig, axes = plt.subplots(n_mods, 2, figsize=(7, 3 * n_mods), squeeze=False)
    fig.suptitle(
        f'{attack_name.upper()} — Clean vs Adversarial IQ Distribution',
        fontsize=14, fontweight='bold', y=1.01)

    for row, (cls_idx, mod_name) in enumerate(mod_list):
        mask = labels_np == cls_idx
        clean_mod = clean_np[mask]
        adv_mod = adv_np[mask]

        # Phase-align using clean reference
        clean_derot, phases = _phase_align_clean(clean_mod, mod_name)
        adv_derot = _apply_clean_phase(adv_mod, phases)

        cI = clean_derot[:, 0, :].ravel()
        cQ = clean_derot[:, 1, :].ravel()
        aI = adv_derot[:, 0, :].ravel()
        aQ = adv_derot[:, 1, :].ravel()

        if len(cI) > max_points:
            sel = np.random.choice(len(cI), max_points, replace=False)
            cI, cQ = cI[sel], cQ[sel]
        if len(aI) > max_points:
            sel = np.random.choice(len(aI), max_points, replace=False)
            aI, aQ = aI[sel], aQ[sel]

        ax_c = axes[row, 0]
        ax_c.scatter(cI, cQ, s=2, alpha=0.4, c='#1f77b4',
                     edgecolors='none', rasterized=True)
        ax_c.set_title(f'{mod_name} — Clean (n={mask.sum()})', fontsize=11)
        ax_c.set_xlim(-0.025, 0.025)
        ax_c.set_ylim(-0.025, 0.025)
        ax_c.set_aspect('equal')
        ax_c.grid(True, alpha=0.2, linewidth=0.5)
        ax_c.tick_params(labelsize=8)

        ax_a = axes[row, 1]
        ax_a.scatter(aI, aQ, s=2, alpha=0.4, c='#d62728',
                     edgecolors='none', rasterized=True)
        ax_a.set_title(f'{mod_name} — {attack_name.upper()}', fontsize=11)
        ax_a.set_xlim(-0.025, 0.025)
        ax_a.set_ylim(-0.025, 0.025)
        ax_a.set_aspect('equal')
        ax_a.grid(True, alpha=0.2, linewidth=0.5)
        ax_a.tick_params(labelsize=8)

        if row == n_mods - 1:
            ax_c.set_xlabel('In-phase (I)', fontsize=9)
            ax_a.set_xlabel('In-phase (I)', fontsize=9)
        ax_c.set_ylabel('Quadrature (Q)', fontsize=9)

    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, f'{attack_name}_iq_constellation.png'),
                dpi=200, bbox_inches='tight')
    plt.close(fig)

    # Individual per-modulation plots
    for cls_idx, mod_name in mod_list:
        mask = labels_np == cls_idx
        clean_mod = clean_np[mask]
        adv_mod = adv_np[mask]

        clean_derot, phases = _phase_align_clean(clean_mod, mod_name)
        adv_derot = _apply_clean_phase(adv_mod, phases)

        cI, cQ = clean_derot[:, 0, :].ravel(), clean_derot[:, 1, :].ravel()
        aI, aQ = adv_derot[:, 0, :].ravel(), adv_derot[:, 1, :].ravel()
        if len(cI) > max_points:
            sel = np.random.choice(len(cI), max_points, replace=False)
            cI, cQ = cI[sel], cQ[sel]
        if len(aI) > max_points:
            sel = np.random.choice(len(aI), max_points, replace=False)
            aI, aQ = aI[sel], aQ[sel]

        fig2, (ax_c, ax_a) = plt.subplots(1, 2, figsize=(8, 4))
        fig2.suptitle(f'{mod_name} — {attack_name.upper()}',
                      fontsize=13, fontweight='bold')

        ax_c.scatter(cI, cQ, s=3, alpha=0.4, c='#1f77b4',
                     edgecolors='none', rasterized=True)
        ax_c.set_title('Clean', fontsize=11)
        ax_c.set_xlim(-0.025, 0.025)
        ax_c.set_ylim(-0.025, 0.025)
        ax_c.set_aspect('equal')
        ax_c.grid(True, alpha=0.2, linewidth=0.5)
        ax_c.set_xlabel('In-phase (I)')
        ax_c.set_ylabel('Quadrature (Q)')

        ax_a.scatter(aI, aQ, s=3, alpha=0.4, c='#d62728',
                     edgecolors='none', rasterized=True)
        ax_a.set_title(attack_name.upper(), fontsize=11)
        ax_a.set_xlim(-0.025, 0.025)
        ax_a.set_ylim(-0.025, 0.025)
        ax_a.set_aspect('equal')
        ax_a.grid(True, alpha=0.2, linewidth=0.5)
        ax_a.set_xlabel('In-phase (I)')
        ax_a.set_ylabel('Quadrature (Q)')

        plt.tight_layout()
        fig2.savefig(os.path.join(
            save_dir, f'{attack_name}_{mod_name}_iq_constellation.png'),
            dpi=200, bbox_inches='tight')
        plt.close(fig2)


ADAPTIVE_METHODS = [
    ('energy_acc', 'Energy'),
    ('confidence_acc', 'ConfSweep'),
    ('classify_acc', 'ClassifyK'),
    ('spectral_acc', 'SpectShape'),
]


def format_table(results: List[Dict], topk_list: List[int],
                  title: str = "SigGuard Evaluation",
                  show_adaptive: bool = False) -> str:
    """Format results as a pretty ASCII table with dynamic Top-K columns."""
    # Build header columns
    topk_headers = [f"Top-{k}" for k in topk_list]
    if show_adaptive:
        for _, label in ADAPTIVE_METHODS:
            topk_headers.append(label)
    col_width = 12
    header_cols = "".join(f"{h:>{col_width}}" for h in topk_headers)
    sep_width = 20 + col_width * (1 + len(topk_headers))

    lines = []
    lines.append("")
    lines.append(f"  {title}")
    lines.append("  " + "=" * sep_width)
    lines.append(f"  {'Sample Type':<20}{'Disabled':>{col_width}}{header_cols}")
    lines.append("  " + "-" * sep_width)

    for r in results:
        sample_type = r['sample_type']
        disabled = f"{r['disabled']*100:.2f}%"
        topk_vals = "".join(
            f"{r.get(f'top{k}_acc', 0.0)*100:.2f}%".rjust(col_width)
            for k in topk_list
        )
        if show_adaptive:
            for key, _ in ADAPTIVE_METHODS:
                adapt_val = f"{r.get(key, 0.0)*100:.2f}%".rjust(col_width)
                topk_vals += adapt_val
        lines.append(f"  {sample_type:<20}{disabled:>{col_width}}{topk_vals}")

    lines.append("  " + "=" * sep_width)
    lines.append("")

    return "\n".join(lines)


def run_sigguard_eval(
    model: nn.Module,
    sig_test: torch.Tensor,
    lab_test: torch.Tensor,
    cfg,
    logger,
    attacks: Optional[List[str]] = None,
    topk_list: Optional[List[int]] = None,
    eval_limit: Optional[int] = None,
    batch_size: int = 128,
    plot_iq: bool = True,
    plot_n_samples: int = 3,
    adaptive_topk: bool = False,
    adaptive_threshold: float = 0.90,
    adaptive_k_candidates: Optional[List[int]] = None,
    confidence_threshold: float = 0.8,
    spectral_sig_pct: float = 0.10,
) -> pd.DataFrame:
    """
    Run SigGuard-style evaluation comparing attack accuracy with/without defense.

    Args:
        model: AWN classification model
        sig_test: Test signals tensor [N, 2, T]
        lab_test: Test labels tensor [N]
        cfg: Configuration object
        logger: Logger instance
        attacks: List of attack names (default: all 15 attacks)
        topk_list: List of K values for FFT Top-K defense (default: [50])
        eval_limit: Limit number of test samples
        batch_size: Batch size for evaluation
        plot_iq: Whether to generate IQ distribution plots
        plot_n_samples: Number of individual samples to plot

    Returns:
        DataFrame with columns: sample_type, disabled, top{K}_acc...
    """
    if topk_list is None:
        topk_list = [50]
    model.eval()
    device = cfg.device

    if attacks is None:
        attacks = ALL_ATTACKS

    # Apply eval limit
    if eval_limit is not None and eval_limit < len(sig_test):
        indices = torch.randperm(len(sig_test))[:eval_limit]
        sig_test = sig_test[indices]
        lab_test = lab_test[indices]

    n_samples = len(sig_test)
    logger.info(f"Running SigGuard evaluation on {n_samples} samples")
    logger.info(f"Attacks: {attacks}")
    logger.info(f"FFT Top-K defense with K={topk_list}")

    # Get normalization mode
    ta_box = str(getattr(cfg, 'ta_box', 'unit')).lower()
    eps = getattr(cfg, 'attack_eps', 0.03)
    logger.info(f"Using ta_box={ta_box} normalization, eps={eps}")

    # Wrap model for torchattacks
    wrapped_model = Model01Wrapper(model)
    wrapped_model.to(device)
    wrapped_model.eval()

    # Build class index → modulation name mapping
    idx_to_mod = {}
    if hasattr(cfg, 'classes'):
        for mod_bytes, idx in cfg.classes.items():
            name = mod_bytes.decode() if isinstance(mod_bytes, bytes) else str(mod_bytes)
            idx_to_mod[idx] = name

    # Create plot directory
    if plot_iq:
        iq_plot_dir = os.path.join(cfg.result_dir, 'iq_plots')
        os.makedirs(iq_plot_dir, exist_ok=True)
        logger.info(f"IQ plots will be saved to: {iq_plot_dir}")

    results = []

    # 1. Evaluate clean accuracy (Intact)
    logger.info("\n=== Intact (Clean) ===")
    if adaptive_topk:
        logger.info(f"Adaptive Top-K enabled: threshold={adaptive_threshold}, "
                    f"candidates={adaptive_k_candidates or [10,15,20,30,50]}")
    clean_accs = []
    clean_defense_accs = {k: [] for k in topk_list}
    clean_adaptive_accs = {m: [] for m in ['energy', 'confidence', 'classify', 'spectral']}
    all_clean_samples = []

    for i in range(0, n_samples, batch_size):
        x_batch = sig_test[i:i+batch_size].to(device)
        y_batch = lab_test[i:i+batch_size].to(device)

        # Clean accuracy (disabled = no attack, no defense)
        acc = compute_accuracy(model, x_batch, y_batch)
        clean_accs.append(acc * len(y_batch))

        # Clean with defense for each Top-K
        for k in topk_list:
            x_defended = fft_topk_denoise(x_batch, topk=k)
            acc_def = compute_accuracy(model, x_defended, y_batch)
            clean_defense_accs[k].append(acc_def * len(y_batch))

        # Clean with all 4 adaptive methods
        if adaptive_topk:
            # 1. Energy knee
            x_e, _ = fft_adaptive_topk_denoise(
                x_batch, threshold=adaptive_threshold,
                k_candidates=adaptive_k_candidates,
            )
            clean_adaptive_accs['energy'].append(
                compute_accuracy(model, x_e, y_batch) * len(y_batch))

            # 2. Confidence sweep
            x_c, _ = confidence_sweep_topk_denoise(
                x_batch, model,
                k_candidates=adaptive_k_candidates,
                confidence_threshold=confidence_threshold,
            )
            clean_adaptive_accs['confidence'].append(
                compute_accuracy(model, x_c, y_batch) * len(y_batch))

            # 3. Classify-then-filter
            x_cl, _ = classify_then_filter_topk_denoise(
                x_batch, model, cfg,
            )
            clean_adaptive_accs['classify'].append(
                compute_accuracy(model, x_cl, y_batch) * len(y_batch))

            # 4. Spectral shape
            x_s, _ = spectral_shape_topk_denoise(
                x_batch, sig_pct=spectral_sig_pct,
                k_candidates=adaptive_k_candidates,
            )
            clean_adaptive_accs['spectral'].append(
                compute_accuracy(model, x_s, y_batch) * len(y_batch))

        # Store clean samples for plotting
        if plot_iq:
            all_clean_samples.append(x_batch.cpu())

    intact_disabled = sum(clean_accs) / n_samples
    intact_row = {'sample_type': 'Intact', 'disabled': intact_disabled}
    topk_strs = []
    for k in topk_list:
        acc_k = sum(clean_defense_accs[k]) / n_samples
        intact_row[f'top{k}_acc'] = acc_k
        topk_strs.append(f"Top-{k}={acc_k*100:.2f}%")
    if adaptive_topk:
        method_keys = [('energy', 'energy_acc'), ('confidence', 'confidence_acc'),
                       ('classify', 'classify_acc'), ('spectral', 'spectral_acc')]
        for mname, mkey in method_keys:
            if clean_adaptive_accs[mname]:
                acc_m = sum(clean_adaptive_accs[mname]) / n_samples
                intact_row[mkey] = acc_m
                topk_strs.append(f"{mname}={acc_m*100:.2f}%")
    logger.info(f"Intact: Disabled={intact_disabled*100:.2f}%, {', '.join(topk_strs)}")
    results.append(intact_row)

    # Concatenate clean samples for plotting
    if plot_iq and all_clean_samples:
        clean_for_plot = torch.cat(all_clean_samples, dim=0)
    else:
        clean_for_plot = None

    # 2. Evaluate each attack
    for attack_name in attacks:
        logger.info(f"\n=== {attack_name.upper()} ===")

        try:
            attack = create_attack(attack_name, wrapped_model, cfg)
        except Exception as e:
            logger.warning(f"Failed to create attack {attack_name}: {e}")
            continue

        # Check if this attack needs fixed batch sizes
        needs_padding = attack_name.lower() in FIXED_BATCH_ATTACKS

        attack_accs = []
        defense_accs = {k: [] for k in topk_list}
        adaptive_defense_accs = {m: [] for m in ['energy', 'confidence', 'classify', 'spectral']}
        all_adv_samples = []
        all_clean_for_attack = []
        all_labels_for_attack = []
        n_processed = 0

        for i in range(0, n_samples, batch_size):
            x_batch = sig_test[i:i+batch_size].to(device)
            y_batch = lab_test[i:i+batch_size].to(device)

            try:
                # Generate adversarial examples
                # For APGD and similar attacks, pad to batch_size if needed
                x_adv = generate_adversarial(
                    attack, x_batch, y_batch,
                    wrapped_model=wrapped_model,
                    ta_box=ta_box,
                    pad_to_batch_size=batch_size if needs_padding else None,
                )

                # Attack accuracy (disabled = attack, no defense)
                acc = compute_accuracy(model, x_adv, y_batch)
                attack_accs.append(acc * len(y_batch))

                # Defense accuracy for each Top-K
                for k in topk_list:
                    x_defended = fft_topk_denoise(x_adv, topk=k)
                    acc_def = compute_accuracy(model, x_defended, y_batch)
                    defense_accs[k].append(acc_def * len(y_batch))

                # All 4 adaptive methods
                if adaptive_topk:
                    # 1. Energy knee
                    x_e, _ = fft_adaptive_topk_denoise(
                        x_adv, threshold=adaptive_threshold,
                        k_candidates=adaptive_k_candidates,
                    )
                    adaptive_defense_accs['energy'].append(
                        compute_accuracy(model, x_e, y_batch) * len(y_batch))

                    # 2. Confidence sweep
                    x_c, _ = confidence_sweep_topk_denoise(
                        x_adv, model,
                        k_candidates=adaptive_k_candidates,
                        confidence_threshold=confidence_threshold,
                    )
                    adaptive_defense_accs['confidence'].append(
                        compute_accuracy(model, x_c, y_batch) * len(y_batch))

                    # 3. Classify-then-filter
                    x_cl, _ = classify_then_filter_topk_denoise(
                        x_adv, model, cfg,
                    )
                    adaptive_defense_accs['classify'].append(
                        compute_accuracy(model, x_cl, y_batch) * len(y_batch))

                    # 4. Spectral shape
                    x_s, _ = spectral_shape_topk_denoise(
                        x_adv, sig_pct=spectral_sig_pct,
                        k_candidates=adaptive_k_candidates,
                    )
                    adaptive_defense_accs['spectral'].append(
                        compute_accuracy(model, x_s, y_batch) * len(y_batch))

                n_processed += len(y_batch)

                # Store samples for plotting (keep all for per-mod coverage)
                if plot_iq:
                    all_adv_samples.append(x_adv.cpu())
                    all_clean_for_attack.append(x_batch.cpu())
                    all_labels_for_attack.append(y_batch.cpu())

            except Exception as e:
                logger.warning(f"Attack failed on batch {i}: {e}")
                continue

        if attack_accs and n_processed > 0:
            disabled = sum(attack_accs) / n_processed

            row = {'sample_type': attack_name.upper(), 'disabled': disabled}
            topk_strs = []
            for k in topk_list:
                acc_k = sum(defense_accs[k]) / n_processed
                row[f'top{k}_acc'] = acc_k
                topk_strs.append(f"Top-{k}={acc_k*100:.2f}%")
            if adaptive_topk:
                method_keys = [('energy', 'energy_acc'), ('confidence', 'confidence_acc'),
                               ('classify', 'classify_acc'), ('spectral', 'spectral_acc')]
                for mname, mkey in method_keys:
                    if adaptive_defense_accs[mname]:
                        acc_m = sum(adaptive_defense_accs[mname]) / n_processed
                        row[mkey] = acc_m
                        topk_strs.append(f"{mname}={acc_m*100:.2f}%")
            logger.info(f"{attack_name.upper()}: Disabled={disabled*100:.2f}%, {', '.join(topk_strs)}")
            results.append(row)

            # Generate IQ plots
            if plot_iq and all_adv_samples:
                adv_for_plot = torch.cat(all_adv_samples, dim=0)
                clean_plot = torch.cat(all_clean_for_attack, dim=0)
                labels_for_plot = torch.cat(all_labels_for_attack, dim=0)

                logger.info(f"Generating IQ plots for {attack_name}...")

                # Individual samples
                plot_iq_distribution(
                    clean_plot, adv_for_plot,
                    attack_name, iq_plot_dir,
                    n_samples=plot_n_samples
                )

                # Aggregated samples
                plot_iq_distribution_all(
                    clean_plot, adv_for_plot,
                    attack_name, iq_plot_dir
                )

                # Density plot
                plot_iq_density(
                    clean_plot, adv_for_plot,
                    attack_name, iq_plot_dir
                )

                # Per-modulation constellation plot
                if idx_to_mod:
                    plot_iq_constellation_grid(
                        clean_plot, adv_for_plot,
                        labels_for_plot,
                        attack_name, iq_plot_dir,
                        idx_to_mod=idx_to_mod,
                    )

    # Create DataFrame
    df = pd.DataFrame(results)

    # Print formatted table
    topk_label = ",".join(str(k) for k in topk_list)
    table_str = format_table(results, topk_list=topk_list,
                             title=f"AWN - SigGuard Evaluation (Top-{topk_label})",
                             show_adaptive=adaptive_topk)
    logger.info(table_str)
    print(table_str)

    # Save to CSV
    os.makedirs(cfg.result_dir, exist_ok=True)
    csv_path = os.path.join(cfg.result_dir, 'sigguard_eval.csv')
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved results to: {csv_path}")

    # Also save as formatted text
    txt_path = os.path.join(cfg.result_dir, 'sigguard_eval_table.txt')
    with open(txt_path, 'w') as f:
        f.write(table_str)
    logger.info(f"Saved table to: {txt_path}")

    if plot_iq:
        logger.info(f"IQ plots saved to: {iq_plot_dir}")

    return df
