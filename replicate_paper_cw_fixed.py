#!/usr/bin/env python
"""
Replicate the CW attack experiment from the paper - FIXED VERSION

Following gpt.md guidelines:
1. Proper [0,1] ↔ [-1,1] conversion for torchattacks
2. Proper 3D ↔ 4D dimension handling
3. Return logits (not softmax probabilities)
4. Generate I/Q constellation plots matching the paper
"""

import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import torchattacks
import torch.nn as nn

print('='*70)
print('Replicating Paper CW Attack - FIXED (Following gpt.md)')
print('='*70)

# Configuration
DATASET_PATH = './data/RML2016.10a_dict.pkl'
MODEL_PATH = './2016.10a_AWN.pkl'
DETECTOR_PATH = './checkpoint/detector_ae.pth'
SNR = 18
MODULATIONS = [b'BPSK', b'QPSK', b'QAM16', b'QAM64']
SAMPLES_PER_MOD = 500
CW_STEPS = 200  # Increased as per gpt.md recommendation
CW_C = 1.0
TOPK = 50
DETECTOR_THRESHOLD = 0.004468
DEVICE = 'cpu'

print(f'\nKey Fix: Proper [0,1] ↔ [-1,1] conversion for torchattacks')
print(f'\nConfiguration:')
print(f'  Modulations: {[m.decode() for m in MODULATIONS]}')
print(f'  SNR: {SNR} dB')
print(f'  Samples per modulation: {SAMPLES_PER_MOD}')
print(f'  CW attack: steps={CW_STEPS}, c={CW_C}')

# Load dataset
print(f'\n[1/7] Loading dataset...')
with open(DATASET_PATH, 'rb') as f:
    Xd = pickle.load(f, encoding='bytes')

snrs, mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1, 0])

# Extract samples for selected modulations at SNR=18
X_dict = {}
y_dict = {}

for mod in MODULATIONS:
    if (mod, SNR) in Xd:
        data = Xd[(mod, SNR)][:SAMPLES_PER_MOD]
        X_dict[mod] = torch.from_numpy(data).float()
        y_dict[mod] = torch.full((len(data),), mods.index(mod), dtype=torch.long)

        # Check data range
        print(f'  {mod.decode()}: {len(data)} samples, range [{X_dict[mod].min():.3f}, {X_dict[mod].max():.3f}]')

# Load AWN model
print(f'\n[2/7] Loading AWN model...')
from util.utils import create_AWN_model
from util.config import Config

cfg = Config('2016.10a', train=False)
cfg.device = DEVICE
base_model = create_AWN_model(cfg)
ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
base_model.load_state_dict(ckpt)
base_model.eval()
print(f'  ✓ Loaded base model')

# Create wrapper model following gpt.md
print(f'\n[3/7] Creating Model01Wrapper for torchattacks compatibility...')

class Model01Wrapper(nn.Module):
    """
    Wrapper to convert between:
    - torchattacks expects: [0,1] range, 4D shape (N,C,H,W)
    - AWN model expects: [-1,1] range, 3D shape (N,2,L)

    Also ensures logits output (not softmax).
    """
    def __init__(self, base_model):
        super().__init__()
        self.base = base_model

    def forward(self, x01):
        # x01: [N,2,L,1] or [N,2,1,L] from torchattacks
        # Convert to [N,2,L]
        if x01.dim() == 4 and x01.shape[-1] == 1:
            x01 = x01.squeeze(-1)  # [N,2,L]
        elif x01.dim() == 4 and x01.shape[-2] == 1:
            x01 = x01.squeeze(-2)  # [N,2,L]
        elif x01.dim() == 3:
            pass  # Already [N,2,L]
        else:
            raise ValueError(f"Unexpected input shape: {x01.shape}")

        # Convert [0,1] → [-1,1]
        x_iq = 2.0 * x01 - 1.0

        # Forward through AWN model
        logits, _ = self.base(x_iq)  # AWN returns (logits, regu_sum)

        return logits  # Return only logits for torchattacks

wrapped_model = Model01Wrapper(base_model).eval()
print(f'  ✓ Created wrapper (converts [0,1]↔[-1,1] and 4D↔3D)')

# Generate CW attacks for each modulation
print(f'\n[4/7] Generating CW adversarial examples...')
print(f'  (Using proper data range conversion)')

X_adv_dict = {}
X_clean_01_dict = {}  # Store [0,1] versions for reference

for mod in MODULATIONS:
    print(f'  Attacking {mod.decode()}...')
    X_clean_iq = X_dict[mod]  # [-1,1] range, [N,2,L]
    y_clean = y_dict[mod]

    # Convert [-1,1] → [0,1] and add dummy dimension for torchattacks
    X_clean_01 = (X_clean_iq + 1.0) / 2.0  # [N,2,L] in [0,1]
    X_clean_01_4d = X_clean_01.unsqueeze(-1)  # [N,2,L,1]

    X_clean_01_dict[mod] = X_clean_01

    print(f'    Input range after conversion: [{X_clean_01_4d.min():.3f}, {X_clean_01_4d.max():.3f}]')

    # Create CW attack
    atk = torchattacks.CW(wrapped_model, c=CW_C, kappa=0, steps=CW_STEPS, lr=1e-2)

    # Generate adversarial examples in batches
    batch_size = 50
    adv_01_batches = []

    for i in range(0, len(X_clean_01_4d), batch_size):
        batch_x = X_clean_01_4d[i:i+batch_size]
        batch_y = y_clean[i:i+batch_size]
        adv_01 = atk(batch_x, batch_y)  # Still in [0,1]
        adv_01_batches.append(adv_01)

    X_adv_01_4d = torch.cat(adv_01_batches, dim=0)

    # Convert back: [0,1] → [-1,1] and remove dummy dimension
    X_adv_iq = 2.0 * X_adv_01_4d.squeeze(-1) - 1.0  # [N,2,L] in [-1,1]
    X_adv_dict[mod] = X_adv_iq

    print(f'    ✓ Generated {len(X_adv_iq)} adversarial samples')
    print(f'    Output range: [{X_adv_iq.min():.3f}, {X_adv_iq.max():.3f}]')

# Apply detector-gated FFT Top-K recovery
print(f'\n[5/7] Applying detector-gated FFT Top-K recovery...')
from util.defense import normalize_iq_data, denormalize_iq_data, fft_topk_denoise
from util.detector import RFSignalAutoEncoder, detector_gate_fft_topk

# Load detector
detector = RFSignalAutoEncoder().to(DEVICE)
det_ckpt = torch.load(DETECTOR_PATH, map_location=DEVICE, weights_only=False)
detector.load_state_dict(det_ckpt)
detector.eval()
print(f'  ✓ Loaded detector')

X_rec_dict = {}
detection_stats = {}

for mod in MODULATIONS:
    print(f'  Recovering {mod.decode()}...')
    X_adv = X_adv_dict[mod]

    with torch.no_grad():
        X_rec, kl_vals = detector_gate_fft_topk(
            X_adv,
            detector,
            threshold=DETECTOR_THRESHOLD,
            topk=TOPK,
            norm_offset=0.02,
            norm_scale=0.04,
            apply_in_normalized=True
        )

    X_rec_dict[mod] = X_rec

    defended = (kl_vals <= DETECTOR_THRESHOLD).sum().item()
    total = len(kl_vals)
    detection_stats[mod] = {
        'defended': defended,
        'total': total,
        'ratio': defended / total,
        'mean_kl': kl_vals.mean().item(),
    }
    print(f'    Defended: {defended}/{total} ({defended/total*100:.1f}%)')

# Compute accuracies
print(f'\n[6/7] Computing accuracies...')
print(f'\n{"Modulation":<10} {"Clean %":<10} {"CW %":<10} {"Recovered %":<12} {"Recovery Gain"}')
print('-'*70)

accuracy_results = {}

for mod in MODULATIONS:
    X_clean = X_dict[mod]
    X_adv = X_adv_dict[mod]
    X_rec = X_rec_dict[mod]
    y_true = y_dict[mod]

    with torch.no_grad():
        # Clean predictions
        logits_clean, _ = base_model(X_clean)
        preds_clean = torch.argmax(logits_clean, dim=1)
        clean_acc = (preds_clean == y_true).float().mean().item() * 100

        # Adversarial predictions
        logits_adv, _ = base_model(X_adv)
        preds_adv = torch.argmax(logits_adv, dim=1)
        adv_acc = (preds_adv == y_true).float().mean().item() * 100

        # Recovered predictions
        logits_rec, _ = base_model(X_rec)
        preds_rec = torch.argmax(logits_rec, dim=1)
        rec_acc = (preds_rec == y_true).float().mean().item() * 100

    recovery_gain = rec_acc - adv_acc

    accuracy_results[mod] = {
        'clean': clean_acc,
        'adversarial': adv_acc,
        'recovered': rec_acc,
        'gain': recovery_gain
    }

    print(f'{mod.decode():<10} {clean_acc:>6.2f}%    {adv_acc:>6.2f}%    {rec_acc:>6.2f}%      {recovery_gain:>+6.2f}%')

# Generate paper-style I/Q constellation plots
print(f'\n[7/7] Generating constellation plots (paper style)...')

fig = plt.figure(figsize=(16, 12))
gs = GridSpec(4, 3, figure=fig, hspace=0.35, wspace=0.25)

fig.suptitle('CW Attack on RF Modulation Classification: I/Q Constellation Analysis\n'
             f'(SNR={SNR} dB, CW steps={CW_STEPS}, c={CW_C}, Fixed data range conversion)',
             fontsize=14, fontweight='bold')

for idx, mod in enumerate(MODULATIONS):
    mod_name = mod.decode()

    # Get data (already in [-1,1] range)
    X_clean = X_dict[mod].numpy()
    X_adv = X_adv_dict[mod].numpy()
    X_rec = X_rec_dict[mod].numpy()

    # Extract I and Q channels and flatten
    n_points = 10000  # Use more points for better constellation
    I_clean = X_clean[:, 0, :].flatten()[:n_points]
    Q_clean = X_clean[:, 1, :].flatten()[:n_points]

    I_adv = X_adv[:, 0, :].flatten()[:n_points]
    Q_adv = X_adv[:, 1, :].flatten()[:n_points]

    I_rec = X_rec[:, 0, :].flatten()[:n_points]
    Q_rec = X_rec[:, 1, :].flatten()[:n_points]

    # (a) Intact Data
    ax = fig.add_subplot(gs[idx, 0])
    ax.scatter(I_clean, Q_clean, c='blue', alpha=0.3, s=1, edgecolors='none')
    ax.set_title(f'{mod_name}\nIntact Data', fontsize=11, fontweight='bold')
    ax.set_xlabel('In-phase (I)', fontsize=9)
    ax.set_ylabel('Quadrature (Q)', fontsize=9)
    ax.grid(True, alpha=0.2)
    ax.set_xlim(-0.02, 0.02)
    ax.set_ylim(-0.02, 0.02)
    ax.set_aspect('equal', adjustable='box')

    # Add accuracy text
    acc_text = f'Acc: {accuracy_results[mod]["clean"]:.1f}%'
    ax.text(0.05, 0.95, acc_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # (b) CW Attack
    ax = fig.add_subplot(gs[idx, 1])
    ax.scatter(I_adv, Q_adv, c='red', alpha=0.3, s=1, edgecolors='none')
    ax.set_title(f'{mod_name}\nCW Attack', fontsize=11, fontweight='bold')
    ax.set_xlabel('In-phase (I)', fontsize=9)
    ax.set_ylabel('Quadrature (Q)', fontsize=9)
    ax.grid(True, alpha=0.2)
    ax.set_xlim(-0.02, 0.02)
    ax.set_ylim(-0.02, 0.02)
    ax.set_aspect('equal', adjustable='box')

    # Add accuracy and attack success
    acc_text = f'Acc: {accuracy_results[mod]["adversarial"]:.1f}%\nASR: {accuracy_results[mod]["clean"] - accuracy_results[mod]["adversarial"]:.1f}%'
    ax.text(0.05, 0.95, acc_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # (c) After Recovery
    ax = fig.add_subplot(gs[idx, 2])
    ax.scatter(I_rec, Q_rec, c='green', alpha=0.3, s=1, edgecolors='none')
    ax.set_title(f'{mod_name}\nAfter Recovery', fontsize=11, fontweight='bold')
    ax.set_xlabel('In-phase (I)', fontsize=9)
    ax.set_ylabel('Quadrature (Q)', fontsize=9)
    ax.grid(True, alpha=0.2)
    ax.set_xlim(-0.02, 0.02)
    ax.set_ylim(-0.02, 0.02)
    ax.set_aspect('equal', adjustable='box')

    # Add accuracy and recovery info
    acc_text = f'Acc: {accuracy_results[mod]["recovered"]:.1f}%\nGain: {accuracy_results[mod]["gain"]:+.1f}%\nDef: {detection_stats[mod]["ratio"]*100:.0f}%'
    ax.text(0.05, 0.95, acc_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.savefig('paper_cw_replication_fixed.png', dpi=300, bbox_inches='tight')
print(f'  ✓ Saved: paper_cw_replication_fixed.png')

plt.savefig('paper_cw_replication_fixed.pdf', bbox_inches='tight')
print(f'  ✓ Saved: paper_cw_replication_fixed.pdf')

# Print summary
print(f'\n{"="*70}')
print('SUMMARY')
print(f'{"="*70}')

overall_clean = np.mean([accuracy_results[m]['clean'] for m in MODULATIONS])
overall_adv = np.mean([accuracy_results[m]['adversarial'] for m in MODULATIONS])
overall_rec = np.mean([accuracy_results[m]['recovered'] for m in MODULATIONS])
overall_gain = overall_rec - overall_adv

print(f'\nOverall Performance:')
print(f'  Clean Accuracy:        {overall_clean:6.2f}%')
print(f'  After CW Attack:       {overall_adv:6.2f}%  (ASR: {overall_clean - overall_adv:.2f}%)')
print(f'  After Recovery:        {overall_rec:6.2f}%')
print(f'  Recovery Gain:         {overall_gain:+6.2f}%')

overall_defended = sum([detection_stats[m]['defended'] for m in MODULATIONS])
overall_total = sum([detection_stats[m]['total'] for m in MODULATIONS])
print(f'\nDetector Statistics:')
print(f'  Samples defended:      {overall_defended}/{overall_total} ({overall_defended/overall_total*100:.1f}%)')

print(f'\n{"="*70}')
print('KEY IMPROVEMENTS (following gpt.md):')
print(f'{"="*70}')
print('✓ Proper [0,1] ↔ [-1,1] conversion')
print('✓ Proper 3D ↔ 4D dimension handling')
print('✓ Model wrapper returns logits (not softmax)')
print('✓ Data range preserved throughout pipeline')
print(f'{"="*70}')

# Save results
results = {
    'configuration': {
        'dataset': DATASET_PATH,
        'snr': SNR,
        'modulations': [m.decode() for m in MODULATIONS],
        'samples_per_mod': SAMPLES_PER_MOD,
        'cw_steps': CW_STEPS,
        'cw_c': CW_C,
        'topk': TOPK,
        'detector_threshold': DETECTOR_THRESHOLD,
        'fixed': 'Following gpt.md guidelines for proper data range conversion',
    },
    'accuracy': {m.decode(): accuracy_results[m] for m in MODULATIONS},
    'detection': {m.decode(): detection_stats[m] for m in MODULATIONS},
    'overall': {
        'clean_acc': overall_clean,
        'adversarial_acc': overall_adv,
        'recovered_acc': overall_rec,
        'recovery_gain': overall_gain,
        'defended_ratio': overall_defended / overall_total,
    }
}

import json
with open('paper_cw_replication_fixed_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f'\n✓ Results saved to: paper_cw_replication_fixed_results.json')
print(f'\n✓ Experiment complete!')
