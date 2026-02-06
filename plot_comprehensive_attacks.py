"""
Comprehensive attack visualization for each modulation:
- 1 clean IQ distribution
- 17 adversarial IQ distributions
- 17 frequency domain plots (clean vs adversarial)
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
SAMPLES_PER_MOD = 200
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUTPUT_DIR = './iq_freq_comprehensive'

# All 17 attacks
ATTACKS = ['fgsm', 'pgd', 'bim', 'cw', 'deepfool', 'apgd', 'mifgsm', 'rfgsm',
           'upgd', 'eotpgd', 'vmifgsm', 'vnifgsm', 'jitter', 'ffgsm', 'pgdl2',
           'eadl1', 'eaden']

# Class mapping for 2016.10a
CLASSES = {b'QAM16': 0, b'QAM64': 1, b'8PSK': 2, b'WBFM': 3, b'BPSK': 4,
           b'CPFSK': 5, b'AM-DSB': 6, b'GFSK': 7, b'PAM4': 8, b'QPSK': 9, b'AM-SSB': 10}
CLASS_NAMES = ['QAM16', 'QAM64', '8PSK', 'WBFM', 'BPSK', 'CPFSK', 'AM-DSB', 'GFSK', 'PAM4', 'QPSK', 'AM-SSB']

# Phase alignment order
PHASE_ORDER = {
    'BPSK': 2, 'QPSK': 4, '8PSK': 8, 'QAM16': 4, 'QAM64': 4,
    'PAM4': 2, 'WBFM': 1, 'AM-DSB': 1, 'AM-SSB': 1, 'CPFSK': 4, 'GFSK': 4,
}


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


def normalize_and_align(signals, mod_name):
    """Apply RMS normalization and phase alignment."""
    if signals.ndim == 2:
        signals = signals[np.newaxis, :, :]

    N = signals.shape[0]
    order = PHASE_ORDER.get(mod_name, 4)

    all_I, all_Q = [], []

    for k in range(N):
        I_k = signals[k, 0, :]
        Q_k = signals[k, 1, :]
        iq = I_k + 1j * Q_k

        rms = np.sqrt(np.mean(np.abs(iq)**2))
        if rms > 1e-10:
            iq = iq / rms

        if order > 1:
            phase_est = np.angle(np.mean(iq**order)) / order
            iq = iq * np.exp(-1j * phase_est)

        all_I.append(iq.real)
        all_Q.append(iq.imag)

    return np.concatenate(all_I), np.concatenate(all_Q)


def compute_psd(signals):
    """Compute average PSD across samples."""
    # signals: [N, 2, L]
    N, _, L = signals.shape

    # Combine I and Q as complex signal
    iq_complex = signals[:, 0, :] + 1j * signals[:, 1, :]

    # Compute FFT for each sample
    fft_result = np.fft.fft(iq_complex, axis=1)
    fft_result = np.fft.fftshift(fft_result, axes=1)

    # Compute PSD (magnitude squared)
    psd = np.abs(fft_result) ** 2

    # Average across samples
    avg_psd = np.mean(psd, axis=0)

    # Convert to dB
    avg_psd_db = 10 * np.log10(avg_psd + 1e-10)

    # Frequency axis (normalized)
    freq = np.fft.fftshift(np.fft.fftfreq(L))

    return freq, avg_psd_db


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


def plot_modulation_comprehensive(mod_name, class_idx, signals, labels, model_wrapper, output_dir):
    """
    Create comprehensive visualization for one modulation:
    - Figure 1: IQ distributions (1 clean + 17 attacks) - 6x3 grid
    - Figure 2: Frequency domain (17 attacks, each showing clean vs adv) - 6x3 grid
    """

    mod_dir = os.path.join(output_dir, mod_name)
    os.makedirs(mod_dir, exist_ok=True)

    signals_t = signals[:SAMPLES_PER_MOD].to(DEVICE)
    labels_t = labels[:SAMPLES_PER_MOD].to(DEVICE)

    # Get clean data
    clean_np = signals_t.cpu().numpy()
    clean_I, clean_Q = normalize_and_align(clean_np, mod_name)
    clean_freq, clean_psd = compute_psd(clean_np)

    # Store adversarial results
    adv_results = {}

    for attack_name in ATTACKS:
        try:
            with torch.enable_grad():
                adv_signals = run_attack(model_wrapper, signals_t, labels_t, attack_name, eps=ATTACK_EPS, ta_box=TA_BOX)
            adv_np = adv_signals.cpu().detach().numpy()
            adv_I, adv_Q = normalize_and_align(adv_np, mod_name)
            adv_freq, adv_psd = compute_psd(adv_np)
            adv_results[attack_name] = {
                'I': adv_I, 'Q': adv_Q,
                'freq': adv_freq, 'psd': adv_psd
            }
        except Exception as e:
            print(f"    Error with {attack_name}: {e}")
            adv_results[attack_name] = None

    # ========== Figure 1: IQ Distributions (1 clean + 17 attacks) ==========
    fig1, axes1 = plt.subplots(3, 6, figsize=(24, 12))
    axes1 = axes1.flatten()
    fig1.suptitle(f'{mod_name} - IQ Constellation (SNR={SNR_FILTER}dB, ta_box={TA_BOX})',
                  fontsize=16, fontweight='bold')

    # Plot clean first
    axes1[0].scatter(clean_I, clean_Q, s=1, alpha=0.4, c='blue')
    axes1[0].set_title('Clean', fontsize=10, fontweight='bold')
    axes1[0].set_xlim(-2.5, 2.5)
    axes1[0].set_ylim(-2.5, 2.5)
    axes1[0].set_aspect('equal')
    axes1[0].grid(True, alpha=0.3)
    axes1[0].set_xlabel('I')
    axes1[0].set_ylabel('Q')

    # Plot 17 attacks
    for idx, attack_name in enumerate(ATTACKS):
        ax = axes1[idx + 1]
        if adv_results[attack_name] is not None:
            ax.scatter(adv_results[attack_name]['I'], adv_results[attack_name]['Q'],
                      s=1, alpha=0.4, c='red')
            ax.set_title(f'{attack_name.upper()}', fontsize=10)
        else:
            ax.set_title(f'{attack_name.upper()} (error)', fontsize=10)
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-2.5, 2.5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('I')
        ax.set_ylabel('Q')

    plt.tight_layout()
    fig1.savefig(os.path.join(mod_dir, f'{mod_name}_iq_all.png'), dpi=150, bbox_inches='tight')
    plt.close(fig1)

    # ========== Figure 2: Frequency Domain (17 attacks) ==========
    fig2, axes2 = plt.subplots(3, 6, figsize=(24, 12))
    axes2 = axes2.flatten()
    fig2.suptitle(f'{mod_name} - Frequency Domain: Clean (blue) vs Adversarial (red) (SNR={SNR_FILTER}dB)',
                  fontsize=16, fontweight='bold')

    for idx, attack_name in enumerate(ATTACKS):
        ax = axes2[idx]
        ax.plot(clean_freq, clean_psd, 'b-', alpha=0.7, linewidth=1, label='Clean')

        if adv_results[attack_name] is not None:
            ax.plot(adv_results[attack_name]['freq'], adv_results[attack_name]['psd'],
                   'r-', alpha=0.7, linewidth=1, label='Adv')

        ax.set_title(f'{attack_name.upper()}', fontsize=10)
        ax.set_xlabel('Normalized Freq')
        ax.set_ylabel('PSD (dB)')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.5, 0.5)

    # Hide last subplot, add legend there
    axes2[17].axis('off')
    axes2[17].legend(['Clean', 'Adversarial'], loc='center', fontsize=12)

    plt.tight_layout()
    fig2.savefig(os.path.join(mod_dir, f'{mod_name}_freq_all.png'), dpi=150, bbox_inches='tight')
    plt.close(fig2)

    # ========== Figure 3: Combined IQ + Freq for each attack (individual files) ==========
    for attack_name in ATTACKS:
        fig3, (ax_iq, ax_freq) = plt.subplots(1, 2, figsize=(12, 5))
        fig3.suptitle(f'{mod_name} - {attack_name.upper()} Attack (SNR={SNR_FILTER}dB)',
                      fontsize=14, fontweight='bold')

        # IQ plot
        ax_iq.scatter(clean_I, clean_Q, s=1, alpha=0.4, c='blue', label='Clean')
        if adv_results[attack_name] is not None:
            ax_iq.scatter(adv_results[attack_name]['I'], adv_results[attack_name]['Q'],
                         s=1, alpha=0.4, c='red', label='Adversarial')
        ax_iq.set_title('IQ Constellation')
        ax_iq.set_xlabel('In-phase (I)')
        ax_iq.set_ylabel('Quadrature (Q)')
        ax_iq.set_xlim(-2.5, 2.5)
        ax_iq.set_ylim(-2.5, 2.5)
        ax_iq.set_aspect('equal')
        ax_iq.grid(True, alpha=0.3)
        ax_iq.legend(markerscale=5)

        # Frequency plot
        ax_freq.plot(clean_freq, clean_psd, 'b-', alpha=0.8, linewidth=1.5, label='Clean')
        if adv_results[attack_name] is not None:
            ax_freq.plot(adv_results[attack_name]['freq'], adv_results[attack_name]['psd'],
                        'r-', alpha=0.8, linewidth=1.5, label='Adversarial')
        ax_freq.set_title('Power Spectral Density')
        ax_freq.set_xlabel('Normalized Frequency')
        ax_freq.set_ylabel('PSD (dB)')
        ax_freq.grid(True, alpha=0.3)
        ax_freq.set_xlim(-0.5, 0.5)
        ax_freq.legend()

        plt.tight_layout()
        fig3.savefig(os.path.join(mod_dir, f'{mod_name}_{attack_name}_iq_freq.png'), dpi=150, bbox_inches='tight')
        plt.close(fig3)

    print(f'  Saved all plots for {mod_name}')


def main():
    from util.adv_attack import Model01Wrapper

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Loading data for SNR={SNR_FILTER}...")
    signals_by_mod, labels_by_mod = load_data_snr(SNR_FILTER)
    print(f"Loaded {len(signals_by_mod)} modulations")

    print("Loading model...")
    model = create_model()
    model_wrapper = Model01Wrapper(model).eval()

    print("\n=== Generating comprehensive plots for each modulation ===")
    for class_idx, mod_name in enumerate(CLASS_NAMES):
        if class_idx in signals_by_mod:
            print(f"Processing {mod_name}...")
            plot_modulation_comprehensive(
                mod_name, class_idx,
                signals_by_mod[class_idx],
                labels_by_mod[class_idx],
                model_wrapper, OUTPUT_DIR
            )

    print(f"\nDone! All plots saved to: {OUTPUT_DIR}")
    print("\nFor each modulation folder:")
    print("  - {MOD}_iq_all.png: 1 clean + 17 attack IQ distributions")
    print("  - {MOD}_freq_all.png: 17 frequency domain comparisons")
    print("  - {MOD}_{attack}_iq_freq.png: Individual IQ + freq for each attack")


if __name__ == '__main__':
    main()
