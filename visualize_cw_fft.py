#!/usr/bin/env python
"""
Visualize CW attack effects in frequency domain and I/Q distribution.

Compares:
1. Clean signals
2. CW adversarial examples
3. FFT Top-K recovered signals

Shows:
- Frequency domain (FFT magnitude spectrum)
- I/Q scatter plots
- Time domain waveforms
"""

import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from util.config import Config
try:
    import torchattacks  # Optional; falls back to internal CW if unavailable
    TA_AVAILABLE = True
except Exception:
    torchattacks = None
    TA_AVAILABLE = False
from util.adv_attack import (
    Model01Wrapper,
    iq_to_ta_input,
    iq_to_ta_input_minmax,
    ta_output_to_iq,
    ta_output_to_iq_minmax,
    cw_l2_attack,
)

print('='*70)
print('CW Attack Frequency Domain & I/Q Analysis')
print('='*70)

# Configuration
cfg = Config('2016.10a', train=False)
DATASET_PATH = './data/RML2016.10a_dict.pkl'
MODEL_PATH = './2016.10a_AWN.pkl'
SNR_FILTER = 18
NUM_SAMPLES = 2  # Keep small for quick runs
CW_STEPS = 10    # Fewer steps for speed
TOPK_LIST = [10, 20, 30]  # Evaluate multiple K values
TA_BOX = 'minmax'  # Use per-sample [0,1] box (CW expects 0..1)
DEVICE = 'cpu'

# Load dataset
print('\n[1/5] Loading dataset...')
with open(DATASET_PATH, 'rb') as f:
    Xd = pickle.load(f, encoding='bytes')

snrs, mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1, 0])
print(f'  Modulations: {[m.decode() if isinstance(m, bytes) else m for m in mods]}')
print(f'  SNR range: {min(snrs)} to {max(snrs)} dB')

# Extract samples at specific SNR
X = []
lbl = []
mod_names = []
label_map = cfg.classes
for mod in mods:
    if (mod, SNR_FILTER) in Xd:
        data = Xd[(mod, SNR_FILTER)]
        X.append(data[:NUM_SAMPLES])  # Take first NUM_SAMPLES of each modulation
        mod_idx = label_map[mod]
        for _ in range(NUM_SAMPLES):
            lbl.append(mod_idx)
            mod_names.append(mod.decode() if isinstance(mod, bytes) else mod)

X = np.vstack(X)
lbl = np.array(lbl)
print(f'  Selected {len(X)} samples at SNR={SNR_FILTER} dB from {len(set(mod_names))} modulations')

X_tensor = torch.from_numpy(X).float()
y_tensor = torch.from_numpy(lbl).long()

# Load model
print('\n[2/5] Loading AWN model...')
from util.utils import create_AWN_model

cfg.device = DEVICE
model = create_AWN_model(cfg)

ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
model.load_state_dict(ckpt)
model.eval()
print(f'  ✓ Loaded {MODEL_PATH}')

# Generate CW adversarial examples (torchattacks if available; otherwise internal CW)
print(f'\n[3/5] Generating CW adversarial examples (steps={CW_STEPS})...')
if TA_AVAILABLE:
    print('  Using torchattacks backend')
    wrapped = Model01Wrapper(model).eval()
    atk = torchattacks.CW(wrapped, c=1.0, kappa=0.0, steps=CW_STEPS, lr=1e-2)
    box = TA_BOX.lower()
    if box == 'minmax':
        X01_4d, a, b = iq_to_ta_input_minmax(X_tensor)
        wrapped.set_minmax(a, b)
        X_adv01_4d = atk(X01_4d, y_tensor)
        X_adv = ta_output_to_iq_minmax(X_adv01_4d, a, b)
        wrapped.clear_minmax()
    else:
        X01_4d = iq_to_ta_input(X_tensor)
        X_adv01_4d = atk(X01_4d, y_tensor)
        X_adv = ta_output_to_iq(X_adv01_4d)
else:
    print('  Using internal CW backend (util.adv_attack.cw_l2_attack)')
    # Internal CW operates directly on IQ in [-1,1]
    with torch.no_grad():
        pass
    X_adv = cw_l2_attack(
        model,
        X_tensor,
        y=y_tensor,
        targeted=False,
        c=1.0,
        kappa=0.0,
        steps=CW_STEPS,
        lr=1e-2,
        lowpass=True,
        lowpass_kernel=17,
        device=DEVICE,
    )
print(f'  ✓ Generated {len(X_adv)} adversarial examples')

# Apply FFT Top-K recovery for each K and track metrics
print(f'\n[4/5] Applying FFT Top-K recovery for K in {TOPK_LIST}...')
from util.defense import normalize_iq_data, denormalize_iq_data, fft_topk_denoise

X_adv_norm = normalize_iq_data(X_adv, 0.02, 0.04)
rec_by_k = {}
for k in TOPK_LIST:
    rec_norm = fft_topk_denoise(X_adv_norm, topk=k)
    rec = denormalize_iq_data(rec_norm, 0.02, 0.04)
    rec_by_k[k] = rec
print(f'  ✓ Applied FFT Top-K recovery for all K')

# Get predictions
print(f'\n[5/5] Computing predictions...')
with torch.no_grad():
    logits_clean, _ = model(X_tensor)
    preds_clean = torch.argmax(logits_clean, dim=1)

    logits_adv, _ = model(X_adv)
    preds_adv = torch.argmax(logits_adv, dim=1)

    preds_rec = {}
    for k, rec in rec_by_k.items():
        logits_rec, _ = model(rec)
        preds_rec[k] = torch.argmax(logits_rec, dim=1)

clean_acc = (preds_clean == y_tensor).float().mean().item() * 100
adv_acc = (preds_adv == y_tensor).float().mean().item() * 100
rec_acc = {k: (pred == y_tensor).float().mean().item() * 100 for k, pred in preds_rec.items()}

print(f'  Clean Acc: {clean_acc:.1f}% | After CW: {adv_acc:.1f}% | After Recovery: ' +
      ", ".join([f'K={k}: {v:.1f}%' for k, v in rec_acc.items()]))

# Visualization
print('\n[6/6] Creating visualizations...')

def compute_fft_magnitude(x):
    """Compute FFT magnitude for I/Q signal [2, T]"""
    fft_i = np.fft.fft(x[0])
    fft_q = np.fft.fft(x[1])
    mag_i = np.abs(fft_i)
    mag_q = np.abs(fft_q)
    return mag_i, mag_q

def plot_sample(idx, save_path=None):
    """Plot comparison for one sample"""
    clean = X_tensor[idx].numpy()
    adv = X_adv[idx].numpy()
    rec = X_rec[idx].numpy()

    true_label = y_tensor[idx].item()
    pred_clean = preds_clean[idx].item()
    pred_adv = preds_adv[idx].item()
    pred_rec = preds_rec[idx].item()

    mod_name = mod_names[idx]

    # Compute FFT magnitudes
    mag_i_clean, mag_q_clean = compute_fft_magnitude(clean)
    mag_i_adv, mag_q_adv = compute_fft_magnitude(adv)
    mag_i_rec, mag_q_rec = compute_fft_magnitude(rec)

    # Compute perturbation
    pert = adv - clean
    mag_i_pert, mag_q_pert = compute_fft_magnitude(pert)

    # Create figure with 4 rows x 3 columns
    fig = plt.figure(figsize=(18, 16))
    gs = GridSpec(4, 3, figure=fig, hspace=0.35, wspace=0.3)

    # Title
    fig.suptitle(f'Sample {idx}: {mod_name} (True label={true_label}, SNR={SNR_FILTER} dB)\n'
                 f'Predictions - Clean: {pred_clean} | CW: {pred_adv} | Recovered: {pred_rec}',
                 fontsize=14, fontweight='bold')

    # Row 1: Time domain I channel
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(clean[0], 'b-', linewidth=0.8, label='Clean')
    ax.set_title('Clean - I Channel (Time)', fontsize=11)
    ax.set_xlabel('Sample')
    ax.set_ylabel('Amplitude')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    ax = fig.add_subplot(gs[0, 1])
    ax.plot(adv[0], 'r-', linewidth=0.8, label='CW Adversarial')
    ax.set_title('CW Adversarial - I Channel (Time)', fontsize=11)
    ax.set_xlabel('Sample')
    ax.set_ylabel('Amplitude')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    ax = fig.add_subplot(gs[0, 2])
    ax.plot(rec[0], 'g-', linewidth=0.8, label='FFT Top-K Recovered')
    ax.set_title(f'FFT Top-K Recovered (K={TOPK}) - I Channel (Time)', fontsize=11)
    ax.set_xlabel('Sample')
    ax.set_ylabel('Amplitude')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    # Row 2: Frequency domain I channel
    freq_bins = np.fft.fftfreq(len(clean[0]), d=1.0)

    ax = fig.add_subplot(gs[1, 0])
    ax.plot(freq_bins[:len(freq_bins)//2], mag_i_clean[:len(mag_i_clean)//2], 'b-', linewidth=0.8)
    ax.set_title('Clean - I Channel (Frequency)', fontsize=11)
    ax.set_xlabel('Normalized Frequency')
    ax.set_ylabel('Magnitude')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    ax = fig.add_subplot(gs[1, 1])
    ax.plot(freq_bins[:len(freq_bins)//2], mag_i_adv[:len(mag_i_adv)//2], 'r-', linewidth=0.8)
    ax.set_title('CW Adversarial - I Channel (Frequency)', fontsize=11)
    ax.set_xlabel('Normalized Frequency')
    ax.set_ylabel('Magnitude')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    ax = fig.add_subplot(gs[1, 2])
    ax.plot(freq_bins[:len(freq_bins)//2], mag_i_rec[:len(mag_i_rec)//2], 'g-', linewidth=0.8)
    ax.axvline(x=TOPK/(2*len(clean[0])), color='orange', linestyle='--', linewidth=1.5,
               label=f'Top-{TOPK} cutoff', alpha=0.7)
    ax.set_title(f'FFT Top-K Recovered - I Channel (Frequency)', fontsize=11)
    ax.set_xlabel('Normalized Frequency')
    ax.set_ylabel('Magnitude')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    ax.legend(fontsize=9)

    # Row 3: I/Q scatter plots
    ax = fig.add_subplot(gs[2, 0])
    ax.scatter(clean[0], clean[1], c='blue', alpha=0.6, s=20, edgecolors='none')
    ax.set_title('Clean - I/Q Constellation', fontsize=11)
    ax.set_xlabel('I')
    ax.set_ylabel('Q')
    ax.grid(True, alpha=0.3)
    ax.axis('equal')

    ax = fig.add_subplot(gs[2, 1])
    ax.scatter(adv[0], adv[1], c='red', alpha=0.6, s=20, edgecolors='none')
    ax.set_title('CW Adversarial - I/Q Constellation', fontsize=11)
    ax.set_xlabel('I')
    ax.set_ylabel('Q')
    ax.grid(True, alpha=0.3)
    ax.axis('equal')

    ax = fig.add_subplot(gs[2, 2])
    ax.scatter(rec[0], rec[1], c='green', alpha=0.6, s=20, edgecolors='none')
    ax.set_title('FFT Top-K Recovered - I/Q Constellation', fontsize=11)
    ax.set_xlabel('I')
    ax.set_ylabel('Q')
    ax.grid(True, alpha=0.3)
    ax.axis('equal')

    # Row 4: Perturbation analysis
    ax = fig.add_subplot(gs[3, 0])
    ax.plot(pert[0], 'purple', linewidth=0.8, label='I channel')
    ax.plot(pert[1], 'orange', linewidth=0.8, label='Q channel', alpha=0.7)
    ax.set_title('CW Perturbation (Time Domain)', fontsize=11)
    ax.set_xlabel('Sample')
    ax.set_ylabel('Amplitude')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    ax = fig.add_subplot(gs[3, 1])
    ax.plot(freq_bins[:len(freq_bins)//2], mag_i_pert[:len(mag_i_pert)//2],
            'purple', linewidth=0.8, label='I channel')
    ax.plot(freq_bins[:len(freq_bins)//2], mag_q_pert[:len(mag_q_pert)//2],
            'orange', linewidth=0.8, label='Q channel', alpha=0.7)
    ax.set_title('CW Perturbation (Frequency Domain)', fontsize=11)
    ax.set_xlabel('Normalized Frequency')
    ax.set_ylabel('Magnitude')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    ax.legend(fontsize=9)

    # Statistics
    ax = fig.add_subplot(gs[3, 2])
    ax.axis('off')

    l2_pert = np.linalg.norm(pert)
    l2_clean = np.linalg.norm(clean)
    rel_pert = l2_pert / l2_clean * 100

    stats_text = f"""
    Statistics:

    L2 Norm (Clean):        {l2_clean:.4f}
    L2 Norm (Perturbation): {l2_pert:.4f}
    Relative Perturbation:  {rel_pert:.2f}%

    Frequency Bins Kept:    {TOPK} / {len(clean[0])} ({TOPK/len(clean[0])*100:.1f}%)

    Prediction Changes:
    Clean → CW:     {pred_clean} → {pred_adv} {'✓' if pred_clean == pred_adv else '✗ MISCLASSIFIED'}
    CW → Recovered: {pred_adv} → {pred_rec} {'✓ RECOVERED' if pred_rec == true_label else '✗'}
    """

    ax.text(0.1, 0.5, stats_text, fontsize=10, verticalalignment='center',
            family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'  ✓ Saved: {save_path}')
    else:
        plt.show()

    plt.close()

# Generate plots for first few samples
output_dir = './cw_analysis'
import os
os.makedirs(output_dir, exist_ok=True)

print(f'\nGenerating visualizations (saving to {output_dir}/)...')
for i in range(min(NUM_SAMPLES * len(set(mod_names)), len(X_tensor))):
    plot_sample(i, save_path=f'{output_dir}/sample_{i:03d}_{mod_names[i]}.png')

print(f'\n{"="*70}')
print(f'✓ Analysis complete!')
print(f'  Output directory: {output_dir}/')
print(f'  Generated {min(NUM_SAMPLES * len(set(mod_names)), len(X_tensor))} visualizations')
print(f'\nKey Insights:')
print(f'  - Clean Accuracy:    {clean_acc:.1f}%')
print(f'  - After CW Attack:   {adv_acc:.1f}%')
print(f'  - After Recovery:    {rec_acc:.1f}%')
print(f'  - Recovery Gain:     {rec_acc - adv_acc:+.1f}%')
print(f'{"="*70}')
