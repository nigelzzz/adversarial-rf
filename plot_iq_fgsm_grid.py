"""
Plot IQ distribution before and after FGSM attack for all modulations at SNR 0.
Creates a grid visualization showing clean vs adversarial IQ scatter plots.
"""

import os
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
import torchattacks

# Configuration
DATASET = '2016.10a'
SNR_FILTER = 0
CKPT_PATH = './checkpoint'
ATTACK_EPS = 0.03
TA_BOX = 'minmax'  # 'minmax' or 'unit'
SAMPLES_PER_MOD = 200  # Number of samples to plot per modulation
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Class mapping for 2016.10a
CLASSES = {b'QAM16': 0, b'QAM64': 1, b'8PSK': 2, b'WBFM': 3, b'BPSK': 4,
           b'CPFSK': 5, b'AM-DSB': 6, b'GFSK': 7, b'PAM4': 8, b'QPSK': 9, b'AM-SSB': 10}
CLASS_NAMES = ['QAM16', 'QAM64', '8PSK', 'WBFM', 'BPSK', 'CPFSK', 'AM-DSB', 'GFSK', 'PAM4', 'QPSK', 'AM-SSB']

def load_data_snr(snr_filter=0):
    """Load dataset filtered by SNR."""
    file_pointer = f'./data/RML2016.10a_dict.pkl'
    data = pickle.load(open(file_pointer, 'rb'), encoding='bytes')

    signals_by_mod = {}
    labels_by_mod = {}

    for mod_bytes, class_idx in CLASSES.items():
        key = (mod_bytes, snr_filter)
        if key in data:
            arr = data[key]
            signals_by_mod[class_idx] = torch.from_numpy(arr.astype(np.float32))
            labels_by_mod[class_idx] = torch.full((arr.shape[0],), class_idx, dtype=torch.long)

    return signals_by_mod, labels_by_mod

def create_model():
    """Create and load the AWN model."""
    from models.model import AWN
    import yaml

    cfg = yaml.safe_load(open(f'./config/{DATASET}.yml', 'r'))
    model = AWN(
        num_classes=cfg['num_classes'],
        num_levels=cfg.get('num_levels', cfg.get('num_level', 1)),
        kernel_size=cfg['kernel_size'],
        in_channels=cfg['in_channels'],
        latent_dim=cfg['latent_dim']
    )

    ckpt_file = os.path.join(CKPT_PATH, f'{DATASET}_AWN.pkl')
    model.load_state_dict(torch.load(ckpt_file, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

def run_fgsm_attack(model_wrapper, signals, labels, eps=0.03, batch_size=32, ta_box='minmax'):
    """Apply FGSM attack to signals in batches using proper IQ conversion."""
    from util.adv_attack import (iq_to_ta_input, ta_output_to_iq,
                                  iq_to_ta_input_minmax, ta_output_to_iq_minmax)

    attack = torchattacks.FGSM(model_wrapper, eps=eps)

    all_adv = []
    n_samples = signals.shape[0]

    for i in range(0, n_samples, batch_size):
        batch_signals = signals[i:i+batch_size]
        batch_labels = labels[i:i+batch_size]

        if ta_box == 'minmax':
            # Per-sample min-max normalization to [0,1]
            x01_4d, a, b = iq_to_ta_input_minmax(batch_signals)
            model_wrapper.set_minmax(a, b)
            adv01_4d = attack(x01_4d, batch_labels)
            adv_iq = ta_output_to_iq_minmax(adv01_4d, a, b)
            model_wrapper.clear_minmax()
        else:
            # Unit normalization: (x+1)/2
            x01_4d = iq_to_ta_input(batch_signals)
            adv01_4d = attack(x01_4d, batch_labels)
            adv_iq = ta_output_to_iq(adv01_4d)

        all_adv.append(adv_iq)

    return torch.cat(all_adv, dim=0)

def plot_iq_grid(signals_by_mod, labels_by_mod, model_wrapper, output_path='iq_fgsm_grid_snr0.png'):
    """Create grid visualization of IQ distributions before/after FGSM."""

    n_mods = len(CLASS_NAMES)
    fig, axes = plt.subplots(n_mods, 2, figsize=(10, 4 * n_mods))
    fig.suptitle(f'IQ Distribution: Clean vs FGSM (SNR={SNR_FILTER}dB, eps={ATTACK_EPS}, ta_box={TA_BOX})',
                 fontsize=14, fontweight='bold')

    for idx, (class_idx, mod_name) in enumerate(zip(range(n_mods), CLASS_NAMES)):
        if class_idx not in signals_by_mod:
            continue

        signals = signals_by_mod[class_idx][:SAMPLES_PER_MOD]
        labels = labels_by_mod[class_idx][:SAMPLES_PER_MOD]

        # Move to device
        signals = signals.to(DEVICE)
        labels = labels.to(DEVICE)

        # Generate adversarial examples
        with torch.enable_grad():
            adv_signals = run_fgsm_attack(model_wrapper, signals, labels, eps=ATTACK_EPS, ta_box=TA_BOX)

        # Extract I and Q channels (flatten all samples)
        clean_i = signals[:, 0, :].cpu().numpy().flatten()
        clean_q = signals[:, 1, :].cpu().numpy().flatten()
        adv_i = adv_signals[:, 0, :].cpu().detach().numpy().flatten()
        adv_q = adv_signals[:, 1, :].cpu().detach().numpy().flatten()

        # Plot clean IQ
        ax_clean = axes[idx, 0]
        ax_clean.scatter(clean_i, clean_q, s=0.5, alpha=0.3, c='blue')
        ax_clean.set_xlabel('I (In-phase)')
        ax_clean.set_ylabel('Q (Quadrature)')
        ax_clean.set_title(f'{mod_name} - Clean')
        ax_clean.set_xlim(-0.08, 0.08)
        ax_clean.set_ylim(-0.08, 0.08)
        ax_clean.set_aspect('equal')
        ax_clean.grid(True, alpha=0.3)

        # Plot adversarial IQ
        ax_adv = axes[idx, 1]
        ax_adv.scatter(adv_i, adv_q, s=0.5, alpha=0.3, c='red')
        ax_adv.set_xlabel('I (In-phase)')
        ax_adv.set_ylabel('Q (Quadrature)')
        ax_adv.set_title(f'{mod_name} - FGSM')
        ax_adv.set_xlim(-0.08, 0.08)
        ax_adv.set_ylim(-0.08, 0.08)
        ax_adv.set_aspect('equal')
        ax_adv.grid(True, alpha=0.3)

        print(f'Processed {mod_name}')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {output_path}')

def plot_iq_overlay(signals_by_mod, labels_by_mod, model_wrapper, output_path='iq_fgsm_overlay_snr0.png'):
    """Create overlay visualization showing clean vs adversarial on same plot."""

    n_mods = len(CLASS_NAMES)
    n_cols = 4
    n_rows = (n_mods + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = axes.flatten()
    fig.suptitle(f'IQ Distribution Overlay: Clean (blue) vs FGSM (red)\nSNR={SNR_FILTER}dB, eps={ATTACK_EPS}, ta_box={TA_BOX}',
                 fontsize=14, fontweight='bold')

    for idx, (class_idx, mod_name) in enumerate(zip(range(n_mods), CLASS_NAMES)):
        if class_idx not in signals_by_mod:
            axes[idx].set_visible(False)
            continue

        signals = signals_by_mod[class_idx][:SAMPLES_PER_MOD]
        labels = labels_by_mod[class_idx][:SAMPLES_PER_MOD]

        # Move to device
        signals = signals.to(DEVICE)
        labels = labels.to(DEVICE)

        # Generate adversarial examples
        with torch.enable_grad():
            adv_signals = run_fgsm_attack(model_wrapper, signals, labels, eps=ATTACK_EPS, ta_box=TA_BOX)

        # Extract I and Q channels (flatten all samples)
        clean_i = signals[:, 0, :].cpu().numpy().flatten()
        clean_q = signals[:, 1, :].cpu().numpy().flatten()
        adv_i = adv_signals[:, 0, :].cpu().detach().numpy().flatten()
        adv_q = adv_signals[:, 1, :].cpu().detach().numpy().flatten()

        ax = axes[idx]
        ax.scatter(clean_i, clean_q, s=0.5, alpha=0.3, c='blue', label='Clean')
        ax.scatter(adv_i, adv_q, s=0.5, alpha=0.3, c='red', label='FGSM')
        ax.set_xlabel('I')
        ax.set_ylabel('Q')
        ax.set_title(f'{mod_name}')
        ax.set_xlim(-0.08, 0.08)
        ax.set_ylim(-0.08, 0.08)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        print(f'Processed {mod_name}')

    # Hide unused subplots
    for idx in range(n_mods, len(axes)):
        axes[idx].set_visible(False)

    # Add legend
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label='Clean'),
               plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='FGSM')]
    fig.legend(handles=handles, loc='upper right', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {output_path}')

def main():
    from util.adv_attack import Model01Wrapper

    print(f"Loading data for SNR={SNR_FILTER}...")
    signals_by_mod, labels_by_mod = load_data_snr(SNR_FILTER)
    print(f"Loaded {len(signals_by_mod)} modulations")

    print("Loading model...")
    model = create_model()
    model_wrapper = Model01Wrapper(model).eval()

    print("Generating IQ distribution plots...")

    # Side-by-side grid (clean | adversarial for each modulation)
    plot_iq_grid(signals_by_mod, labels_by_mod, model_wrapper,
                 output_path='iq_fgsm_grid_snr0.png')

    # Overlay plot (all modulations in compact grid)
    plot_iq_overlay(signals_by_mod, labels_by_mod, model_wrapper,
                    output_path='iq_fgsm_overlay_snr0.png')

    print("Done!")

if __name__ == '__main__':
    main()
