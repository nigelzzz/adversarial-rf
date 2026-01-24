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

try:
    import torchattacks
except ImportError:
    torchattacks = None

from util.adv_attack import Model01Wrapper, iq_to_ta_input, ta_output_to_iq
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


# Default attack list
DEFAULT_ATTACKS = [
    'fgsm', 'pgd', 'bim', 'cw', 'deepfool', 'apgd', 'mifgsm',
    'rfgsm', 'upgd', 'eotpgd', 'vmifgsm', 'vnifgsm', 'jitter', 'ffgsm', 'pgdl2'
]


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

    # Get epsilon from config (default 0.3 for IQ data, much larger than image default 8/255)
    eps = getattr(cfg, 'attack_eps', 0.3)
    alpha = eps / 4  # step size typically eps/4 for iterative attacks
    steps = 10

    if name == 'fgsm':
        return torchattacks.FGSM(wrapped_model, eps=eps)

    elif name == 'pgd':
        return torchattacks.PGD(wrapped_model, eps=eps, alpha=alpha, steps=steps)

    elif name == 'bim':
        return torchattacks.BIM(wrapped_model, eps=eps, alpha=alpha, steps=steps)

    elif name == 'cw':
        c = getattr(cfg, 'cw_c', 1.0)
        cw_steps = getattr(cfg, 'cw_steps', 100)
        cw_lr = getattr(cfg, 'cw_lr', 0.01)
        return torchattacks.CW(wrapped_model, c=c, steps=cw_steps, lr=cw_lr)

    elif name == 'deepfool':
        return torchattacks.DeepFool(wrapped_model, steps=50, overshoot=0.02)

    elif name == 'apgd':
        return torchattacks.APGD(wrapped_model, eps=eps, steps=steps)

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

    else:
        raise ValueError(f"Unknown attack: {attack_name}")


def generate_adversarial(
    attack,
    x_iq: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """
    Generate adversarial examples using torchattacks.

    Handles IQ <-> torchattacks format conversion.

    Args:
        attack: torchattacks attack object
        x_iq: Input IQ tensor [N, 2, T] in [-1, 1]
        labels: True labels [N]

    Returns:
        Adversarial IQ tensor [N, 2, T] in [-1, 1]
    """
    # Convert IQ to torchattacks format
    x_ta = iq_to_ta_input(x_iq)  # [N, 2, T, 1] in [0, 1]

    # Run attack
    x_adv_ta = attack(x_ta, labels)

    # Convert back to IQ format
    x_adv_iq = ta_output_to_iq(x_adv_ta)  # [N, 2, T] in [-1, 1]

    return x_adv_iq


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
        plot_n_samples: Number of individual samples to plot (if plot_freq=True)

    Returns:
        DataFrame with columns: attack, snr, modulation, n_samples, attack_acc, top10_acc, top20_acc
    """
    model.eval()
    device = cfg.device

    if attacks is None:
        attacks = DEFAULT_ATTACKS

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
    wrapped_model.eval()

    results = []

    for attack_name in attacks:
        logger.info(f"\n=== Attack: {attack_name.upper()} ===")

        # Create attack object
        try:
            attack = create_attack(attack_name, wrapped_model, cfg)
        except Exception as e:
            logger.warning(f"Failed to create attack {attack_name}: {e}")
            continue

        for snr in all_snrs:
            for mod in all_mods:
                key = (snr, mod)
                if key not in snr_mod_idx:
                    continue

                indices = snr_mod_idx[key]

                # Apply limit if specified
                if eval_limit_per_cell is not None and len(indices) > eval_limit_per_cell:
                    indices = indices[:eval_limit_per_cell]

                n_samples = len(indices)
                if n_samples == 0:
                    continue

                # Get samples for this cell
                x_cell = sig_test[indices].to(device)
                y_cell = lab_test[indices].to(device)

                # Generate adversarial examples
                try:
                    x_adv = generate_adversarial(attack, x_cell, y_cell)
                except Exception as e:
                    logger.warning(f"Attack failed for ({snr}, {mod}): {e}")
                    continue

                # Compute attack accuracy (after attack, before recovery)
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
                    'attack_acc': attack_acc,
                    'top10_acc': top10_acc,
                    'top20_acc': top20_acc,
                })

        # Log summary for this attack
        attack_results = [r for r in results if r['attack'] == attack_name]
        if attack_results:
            avg_attack = np.mean([r['attack_acc'] for r in attack_results])
            avg_top10 = np.mean([r['top10_acc'] for r in attack_results])
            avg_top20 = np.mean([r['top20_acc'] for r in attack_results])
            logger.info(f"{attack_name}: avg_attack_acc={avg_attack:.4f}, "
                       f"avg_top10={avg_top10:.4f}, avg_top20={avg_top20:.4f}")

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
            'attack_acc': 'mean',
            'top10_acc': 'mean',
            'top20_acc': 'mean',
            'n_samples': 'sum'
        }).round(4)
        logger.info(f"\n{summary.to_string()}")

    return df
