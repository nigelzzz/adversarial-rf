"""
Plot IQ distribution for all 17 attacks across all modulations at SNR 0.
Generates one figure per attack showing clean vs adversarial for each modulation.
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
TA_BOX = 'minmax'
SAMPLES_PER_MOD = 100  # Reduced for faster processing
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUTPUT_DIR = './iq_all_attacks'

# All 17 attacks
ATTACKS = ['fgsm', 'pgd', 'bim', 'cw', 'deepfool', 'apgd', 'mifgsm', 'rfgsm',
           'upgd', 'eotpgd', 'vmifgsm', 'vnifgsm', 'jitter', 'ffgsm', 'pgdl2',
           'eadl1', 'eaden']

# Class mapping for 2016.10a (11 classes)
CLASSES = {b'QAM16': 0, b'QAM64': 1, b'8PSK': 2, b'WBFM': 3, b'BPSK': 4,
           b'CPFSK': 5, b'AM-DSB': 6, b'GFSK': 7, b'PAM4': 8, b'QPSK': 9, b'AM-SSB': 10}
CLASS_NAMES = ['QAM16', 'QAM64', '8PSK', 'WBFM', 'BPSK', 'CPFSK', 'AM-DSB', 'GFSK', 'PAM4', 'QPSK', 'AM-SSB']


def load_data_snr(snr_filter=0):
    """Load dataset filtered by SNR."""
    file_pointer = './data/RML2016.10a_dict.pkl'
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


def get_attack(attack_name, model_wrapper, eps=0.03):
    """Create torchattacks attack object."""
    attack_name = attack_name.lower()

    if attack_name == 'fgsm':
        return torchattacks.FGSM(model_wrapper, eps=eps)
    elif attack_name == 'pgd':
        return torchattacks.PGD(model_wrapper, eps=eps, alpha=eps/4, steps=10)
    elif attack_name == 'bim':
        return torchattacks.BIM(model_wrapper, eps=eps, alpha=eps/4, steps=10)
    elif attack_name == 'cw':
        return torchattacks.CW(model_wrapper, c=1.0, kappa=0, steps=50, lr=0.01)
    elif attack_name == 'deepfool':
        return torchattacks.DeepFool(model_wrapper, steps=50)
    elif attack_name == 'apgd':
        return torchattacks.APGD(model_wrapper, eps=eps, steps=10)
    elif attack_name == 'mifgsm':
        return torchattacks.MIFGSM(model_wrapper, eps=eps, alpha=eps/4, steps=10)
    elif attack_name == 'rfgsm':
        return torchattacks.RFGSM(model_wrapper, eps=eps, alpha=eps/2, steps=10)
    elif attack_name == 'upgd':
        return torchattacks.UPGD(model_wrapper, eps=eps, alpha=eps/4, steps=10)
    elif attack_name == 'eotpgd':
        return torchattacks.EOTPGD(model_wrapper, eps=eps, alpha=eps/4, steps=10)
    elif attack_name == 'vmifgsm':
        return torchattacks.VMIFGSM(model_wrapper, eps=eps, alpha=eps/4, steps=10)
    elif attack_name == 'vnifgsm':
        return torchattacks.VNIFGSM(model_wrapper, eps=eps, alpha=eps/4, steps=10)
    elif attack_name == 'jitter':
        return torchattacks.Jitter(model_wrapper, eps=eps, alpha=eps/4, steps=10)
    elif attack_name == 'ffgsm':
        return torchattacks.FFGSM(model_wrapper, eps=eps, alpha=eps)
    elif attack_name == 'pgdl2':
        return torchattacks.PGDL2(model_wrapper, eps=eps*10, alpha=eps*2, steps=10)
    elif attack_name == 'eadl1':
        return torchattacks.EADL1(model_wrapper, kappa=0, lr=0.01, max_iterations=50)
    elif attack_name == 'eaden':
        return torchattacks.EADEN(model_wrapper, kappa=0, lr=0.01, max_iterations=50)
    else:
        raise ValueError(f"Unknown attack: {attack_name}")


def run_attack(model_wrapper, signals, labels, attack_name, eps=0.03, ta_box='minmax'):
    """Apply attack to signals."""
    from util.adv_attack import (iq_to_ta_input, ta_output_to_iq,
                                  iq_to_ta_input_minmax, ta_output_to_iq_minmax)

    attack = get_attack(attack_name, model_wrapper, eps=eps)

    if ta_box == 'minmax':
        x01_4d, a, b = iq_to_ta_input_minmax(signals)
        model_wrapper.set_minmax(a, b)
        adv01_4d = attack(x01_4d, labels)
        adv_iq = ta_output_to_iq_minmax(adv01_4d, a, b)
        model_wrapper.clear_minmax()
    else:
        x01_4d = iq_to_ta_input(signals)
        adv01_4d = attack(x01_4d, labels)
        adv_iq = ta_output_to_iq(adv01_4d)

    return adv_iq


def plot_attack_all_mods(attack_name, signals_by_mod, labels_by_mod, model_wrapper, output_dir):
    """Create a figure for one attack showing all modulations."""

    n_mods = len(CLASS_NAMES)
    n_cols = 4
    n_rows = (n_mods + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    axes = axes.flatten()
    fig.suptitle(f'{attack_name.upper()} Attack - IQ Distribution (SNR={SNR_FILTER}dB, eps={ATTACK_EPS}, ta_box={TA_BOX})',
                 fontsize=14, fontweight='bold')

    for idx, (class_idx, mod_name) in enumerate(zip(range(n_mods), CLASS_NAMES)):
        if class_idx not in signals_by_mod:
            axes[idx].set_visible(False)
            continue

        signals = signals_by_mod[class_idx][:SAMPLES_PER_MOD].to(DEVICE)
        labels = labels_by_mod[class_idx][:SAMPLES_PER_MOD].to(DEVICE)

        # Generate adversarial examples
        try:
            with torch.enable_grad():
                adv_signals = run_attack(model_wrapper, signals, labels, attack_name, eps=ATTACK_EPS, ta_box=TA_BOX)
        except Exception as e:
            print(f"  Error with {attack_name} on {mod_name}: {e}")
            axes[idx].set_title(f'{mod_name} (error)')
            continue

        # Extract I and Q channels
        clean_i = signals[:, 0, :].cpu().numpy().flatten()
        clean_q = signals[:, 1, :].cpu().numpy().flatten()
        adv_i = adv_signals[:, 0, :].cpu().detach().numpy().flatten()
        adv_q = adv_signals[:, 1, :].cpu().detach().numpy().flatten()

        ax = axes[idx]
        ax.scatter(clean_i, clean_q, s=0.5, alpha=0.4, c='blue', label='Clean')
        ax.scatter(adv_i, adv_q, s=0.5, alpha=0.4, c='red', label='Adv')
        ax.set_xlabel('I')
        ax.set_ylabel('Q')
        ax.set_title(f'{mod_name}')
        ax.set_xlim(-0.08, 0.08)
        ax.set_ylim(-0.08, 0.08)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(n_mods, len(axes)):
        axes[idx].set_visible(False)

    # Add legend
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label='Clean'),
               plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='Adversarial')]
    fig.legend(handles=handles, loc='upper right', fontsize=10)

    plt.tight_layout()
    output_path = os.path.join(output_dir, f'{attack_name}_all_mods.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {output_path}')


def plot_mod_all_attacks(mod_name, class_idx, signals, labels, model_wrapper, output_dir):
    """Create a figure for one modulation showing all 17 attacks."""

    n_attacks = len(ATTACKS)
    n_cols = 6
    n_rows = (n_attacks + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 3.5 * n_rows))
    axes = axes.flatten()
    fig.suptitle(f'{mod_name} - All Attacks IQ Distribution (SNR={SNR_FILTER}dB, ta_box={TA_BOX})',
                 fontsize=14, fontweight='bold')

    signals = signals[:SAMPLES_PER_MOD].to(DEVICE)
    labels_t = labels[:SAMPLES_PER_MOD].to(DEVICE)

    clean_i = signals[:, 0, :].cpu().numpy().flatten()
    clean_q = signals[:, 1, :].cpu().numpy().flatten()

    for idx, attack_name in enumerate(ATTACKS):
        ax = axes[idx]

        try:
            with torch.enable_grad():
                adv_signals = run_attack(model_wrapper, signals, labels_t, attack_name, eps=ATTACK_EPS, ta_box=TA_BOX)

            adv_i = adv_signals[:, 0, :].cpu().detach().numpy().flatten()
            adv_q = adv_signals[:, 1, :].cpu().detach().numpy().flatten()

            ax.scatter(clean_i, clean_q, s=0.3, alpha=0.4, c='blue')
            ax.scatter(adv_i, adv_q, s=0.3, alpha=0.4, c='red')
            ax.set_title(f'{attack_name.upper()}')
        except Exception as e:
            ax.set_title(f'{attack_name.upper()} (error)')
            print(f"  Error with {attack_name}: {e}")

        ax.set_xlim(-0.08, 0.08)
        ax.set_ylim(-0.08, 0.08)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)

    # Hide unused subplots
    for idx in range(n_attacks, len(axes)):
        axes[idx].set_visible(False)

    # Add legend
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label='Clean'),
               plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='Adversarial')]
    fig.legend(handles=handles, loc='upper right', fontsize=10)

    plt.tight_layout()
    output_path = os.path.join(output_dir, f'{mod_name}_all_attacks.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {output_path}')


def main():
    from util.adv_attack import Model01Wrapper

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Loading data for SNR={SNR_FILTER}...")
    signals_by_mod, labels_by_mod = load_data_snr(SNR_FILTER)
    print(f"Loaded {len(signals_by_mod)} modulations")

    print("Loading model...")
    model = create_model()
    model_wrapper = Model01Wrapper(model).eval()

    # Generate per-modulation plots (all 17 attacks per modulation)
    print("\n=== Generating per-modulation plots (17 attacks each) ===")
    for class_idx, mod_name in enumerate(CLASS_NAMES):
        if class_idx in signals_by_mod:
            print(f"Processing {mod_name}...")
            plot_mod_all_attacks(
                mod_name, class_idx,
                signals_by_mod[class_idx],
                labels_by_mod[class_idx],
                model_wrapper, OUTPUT_DIR
            )

    # Generate per-attack plots (all modulations per attack)
    print("\n=== Generating per-attack plots (all modulations each) ===")
    for attack_name in ATTACKS:
        print(f"Processing {attack_name.upper()}...")
        plot_attack_all_mods(attack_name, signals_by_mod, labels_by_mod, model_wrapper, OUTPUT_DIR)

    print("\nDone! All plots saved to:", OUTPUT_DIR)


if __name__ == '__main__':
    main()
