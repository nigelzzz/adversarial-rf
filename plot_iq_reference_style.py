"""
Plot IQ constellation matching reference paper style:
- Raw IQ values (no RMS normalization)
- Axis range [-0.02, 0.02]
- Multiple SNR levels for comparison
- No phase rotation
"""

import os
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
import torchattacks

# Configuration
DATASET = '2016.10a'
CKPT_PATH = './checkpoint'
ATTACK_EPS = 0.03
TA_BOX = 'minmax'
SAMPLES_PER_MOD = 300
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUTPUT_DIR = './iq_reference_style'

ATTACKS = ['fgsm', 'pgd', 'bim', 'cw', 'deepfool', 'mifgsm', 'rfgsm',
           'upgd', 'eotpgd', 'vmifgsm', 'vnifgsm', 'jitter', 'ffgsm', 'pgdl2',
           'eadl1', 'eaden']

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


def get_raw_iq(signals):
    """Get raw IQ values without normalization - matching reference style."""
    if isinstance(signals, torch.Tensor):
        signals = signals.cpu().numpy()

    I = signals[:, 0, :].flatten()
    Q = signals[:, 1, :].flatten()
    return I, Q


def get_attack(attack_name, model_wrapper, eps=0.03):
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


def plot_intact_reference_style(snr=10):
    """Plot intact data matching reference image style."""
    signals_by_mod, _ = load_data_snr(snr)

    key_mods = ['BPSK', 'QPSK', 'QAM16', 'QAM64']

    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    fig.suptitle(f'(a) Intact Data (SNR={snr}dB)', fontsize=14, fontweight='bold')

    for idx, mod_name in enumerate(key_mods):
        row, col = divmod(idx, 2)
        ax = axes[row, col]

        class_idx = CLASS_NAMES.index(mod_name)
        if class_idx not in signals_by_mod:
            continue

        signals = signals_by_mod[class_idx][:SAMPLES_PER_MOD]
        I, Q = get_raw_iq(signals)

        ax.scatter(I, Q, s=2, alpha=0.35, c='#1f77b4', edgecolors='none', rasterized=True)
        ax.set_title(mod_name, fontsize=12)
        ax.set_xlim(-0.025, 0.025)
        ax.set_ylim(-0.025, 0.025)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.2, linewidth=0.5)

        if row == 1:
            ax.set_xlabel('In-phase (I)', fontsize=10)
        if col == 0:
            ax.set_ylabel('Quadrature (Q)', fontsize=10)

    plt.tight_layout()
    return fig


def plot_mod_attacks_reference_style(mod_name, signals_by_mod, labels_by_mod, model_wrapper, snr, output_dir):
    """Plot 1 clean + 17 attacks for a modulation in reference style."""

    class_idx = CLASS_NAMES.index(mod_name)
    if class_idx not in signals_by_mod:
        return

    signals_t = signals_by_mod[class_idx][:SAMPLES_PER_MOD].to(DEVICE)
    labels_t = labels_by_mod[class_idx][:SAMPLES_PER_MOD].to(DEVICE)

    # Get clean IQ
    clean_I, clean_Q = get_raw_iq(signals_t)

    # Create figure: 3 rows x 6 cols = 18 subplots (1 clean + 17 attacks)
    fig, axes = plt.subplots(3, 6, figsize=(20, 10))
    axes = axes.flatten()
    fig.suptitle(f'{mod_name} - IQ Distribution (SNR={snr}dB, ta_box={TA_BOX})', fontsize=14, fontweight='bold')

    # Plot clean
    axes[0].scatter(clean_I, clean_Q, s=2, alpha=0.35, c='blue', edgecolors='none', rasterized=True)
    axes[0].set_title('Clean', fontsize=10, fontweight='bold')
    axes[0].set_xlim(-0.025, 0.025)
    axes[0].set_ylim(-0.025, 0.025)
    axes[0].set_aspect('equal')
    axes[0].grid(True, alpha=0.2)
    axes[0].set_xlabel('I')
    axes[0].set_ylabel('Q')

    # Plot 16 attacks (skip apgd due to errors)
    for idx, attack_name in enumerate(ATTACKS):
        ax = axes[idx + 1]
        try:
            with torch.enable_grad():
                adv_signals = run_attack(model_wrapper, signals_t, labels_t, attack_name, eps=ATTACK_EPS, ta_box=TA_BOX)
            adv_I, adv_Q = get_raw_iq(adv_signals.detach())
            ax.scatter(adv_I, adv_Q, s=2, alpha=0.35, c='red', edgecolors='none', rasterized=True)
            ax.set_title(f'{attack_name.upper()}', fontsize=10)
        except Exception as e:
            ax.set_title(f'{attack_name.upper()} (err)', fontsize=10)

        ax.set_xlim(-0.025, 0.025)
        ax.set_ylim(-0.025, 0.025)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.2)
        ax.set_xlabel('I')
        ax.set_ylabel('Q')

    # Hide unused
    for idx in range(len(ATTACKS) + 1, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f'{mod_name}_snr{snr}_iq_all.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {mod_name} SNR={snr}')


def main():
    from util.adv_attack import Model01Wrapper

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading model...")
    model = create_model()
    model_wrapper = Model01Wrapper(model).eval()

    # Plot intact data at multiple SNRs
    print("\n=== Generating intact constellation (reference style) ===")
    for snr in [0, 10, 18]:
        fig = plot_intact_reference_style(snr)
        fig.savefig(os.path.join(OUTPUT_DIR, f'intact_snr{snr}.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'  Saved intact SNR={snr}')

    # Plot all modulations at SNR=10 (cleaner than SNR=0)
    print("\n=== Generating attack plots at SNR=10 ===")
    signals_by_mod, labels_by_mod = load_data_snr(10)

    for mod_name in CLASS_NAMES:
        print(f"Processing {mod_name}...")
        plot_mod_attacks_reference_style(mod_name, signals_by_mod, labels_by_mod, model_wrapper, 10, OUTPUT_DIR)

    # Also at SNR=0 for comparison
    print("\n=== Generating attack plots at SNR=0 ===")
    signals_by_mod, labels_by_mod = load_data_snr(0)

    for mod_name in CLASS_NAMES:
        print(f"Processing {mod_name}...")
        plot_mod_attacks_reference_style(mod_name, signals_by_mod, labels_by_mod, model_wrapper, 0, OUTPUT_DIR)

    print(f"\nDone! All plots saved to: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
