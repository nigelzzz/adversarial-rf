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
from util.defense import fft_topk_denoise


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


def format_table(results: List[Dict], title: str = "SigGuard Evaluation") -> str:
    """Format results as a pretty ASCII table similar to the paper."""
    lines = []
    lines.append("")
    lines.append(f"  {title}")
    lines.append("  " + "=" * 50)
    lines.append(f"  {'Sample Type':<15} {'Disabled':>12} {'Enabled':>12}")
    lines.append("  " + "-" * 50)

    for r in results:
        sample_type = r['sample_type']
        disabled = f"{r['disabled']*100:.2f}%"
        enabled = f"{r['enabled']*100:.2f}%"
        lines.append(f"  {sample_type:<15} {disabled:>12} {enabled:>12}")

    lines.append("  " + "=" * 50)
    lines.append("")

    return "\n".join(lines)


def run_sigguard_eval(
    model: nn.Module,
    sig_test: torch.Tensor,
    lab_test: torch.Tensor,
    cfg,
    logger,
    attacks: Optional[List[str]] = None,
    topk: int = 50,
    eval_limit: Optional[int] = None,
    batch_size: int = 128,
    plot_iq: bool = True,
    plot_n_samples: int = 3,
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
        topk: K value for FFT Top-K defense
        eval_limit: Limit number of test samples
        batch_size: Batch size for evaluation
        plot_iq: Whether to generate IQ distribution plots
        plot_n_samples: Number of individual samples to plot

    Returns:
        DataFrame with columns: sample_type, disabled, enabled
    """
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
    logger.info(f"FFT Top-K defense with K={topk}")

    # Get normalization mode
    ta_box = str(getattr(cfg, 'ta_box', 'unit')).lower()
    eps = getattr(cfg, 'attack_eps', 0.03)
    logger.info(f"Using ta_box={ta_box} normalization, eps={eps}")

    # Wrap model for torchattacks
    wrapped_model = Model01Wrapper(model)
    wrapped_model.to(device)
    wrapped_model.eval()

    # Create plot directory
    if plot_iq:
        iq_plot_dir = os.path.join(cfg.result_dir, 'iq_plots')
        os.makedirs(iq_plot_dir, exist_ok=True)
        logger.info(f"IQ plots will be saved to: {iq_plot_dir}")

    results = []

    # 1. Evaluate clean accuracy (Intact)
    logger.info("\n=== Intact (Clean) ===")
    clean_accs = []
    clean_defense_accs = []
    all_clean_samples = []

    for i in range(0, n_samples, batch_size):
        x_batch = sig_test[i:i+batch_size].to(device)
        y_batch = lab_test[i:i+batch_size].to(device)

        # Clean accuracy (disabled = no attack, no defense)
        acc = compute_accuracy(model, x_batch, y_batch)
        clean_accs.append(acc * len(y_batch))

        # Clean with defense (enabled = no attack, with defense)
        x_defended = fft_topk_denoise(x_batch, topk=topk)
        acc_def = compute_accuracy(model, x_defended, y_batch)
        clean_defense_accs.append(acc_def * len(y_batch))

        # Store clean samples for plotting
        if plot_iq and len(all_clean_samples) * batch_size < 500:
            all_clean_samples.append(x_batch.cpu())

    intact_disabled = sum(clean_accs) / n_samples
    intact_enabled = sum(clean_defense_accs) / n_samples
    logger.info(f"Intact: Disabled={intact_disabled*100:.2f}%, Enabled={intact_enabled*100:.2f}%")

    results.append({
        'sample_type': 'Intact',
        'disabled': intact_disabled,
        'enabled': intact_enabled,
    })

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
        defense_accs = []
        all_adv_samples = []
        all_clean_for_attack = []
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

                # Defense accuracy (enabled = attack + defense)
                x_defended = fft_topk_denoise(x_adv, topk=topk)
                acc_def = compute_accuracy(model, x_defended, y_batch)
                defense_accs.append(acc_def * len(y_batch))

                n_processed += len(y_batch)

                # Store samples for plotting
                if plot_iq and len(all_adv_samples) * batch_size < 500:
                    all_adv_samples.append(x_adv.cpu())
                    all_clean_for_attack.append(x_batch.cpu())

            except Exception as e:
                logger.warning(f"Attack failed on batch {i}: {e}")
                continue

        if attack_accs and n_processed > 0:
            disabled = sum(attack_accs) / n_processed
            enabled = sum(defense_accs) / n_processed

            logger.info(f"{attack_name.upper()}: Disabled={disabled*100:.2f}%, Enabled={enabled*100:.2f}%")

            results.append({
                'sample_type': attack_name.upper(),
                'disabled': disabled,
                'enabled': enabled,
            })

            # Generate IQ plots
            if plot_iq and all_adv_samples:
                adv_for_plot = torch.cat(all_adv_samples, dim=0)
                clean_plot = torch.cat(all_clean_for_attack, dim=0)

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

    # Create DataFrame
    df = pd.DataFrame(results)

    # Print formatted table
    table_str = format_table(results, title=f"AWN - SigGuard Evaluation (Top-{topk})")
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
