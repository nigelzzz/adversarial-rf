#!/usr/bin/env python
"""
Replicate the CW attack experiment from the paper.

Shows I/Q constellation plots for:
- (a) Intact Data (clean)
- (b) CW Attack
- (c) After Recovery (detector-gated FFT Top-K)

For modulation types: BPSK, QPSK, QAM16, QAM64
"""

import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import torchattacks
from util.adv_attack import Model01Wrapper, iq_to_ta_input, ta_output_to_iq
from util.utils import recover_constellation

print('='*70)
print('Replicating Paper CW Attack Experiment')
print('='*70)

# Configuration matching paper
DATASET_PATH = './data/RML2016.10a_dict.pkl'
MODEL_PATH = './2016.10a_AWN.pkl'
DETECTOR_PATH = './checkpoint/detector_ae.pth'
SNR = 18  # High SNR for clear constellations
MODULATIONS = [b'BPSK', b'QPSK', b'QAM16', b'QAM64']  # Paper modulations
SAMPLES_PER_MOD = 500  # Enough for good constellation
CW_STEPS = 100  # Strong attack
CW_C = 1.0
TOPK = 50
DETECTOR_THRESHOLD = 0.004468
DEVICE = 'cpu'

print(f'\nConfiguration:')
print(f'  Modulations: {[m.decode() for m in MODULATIONS]}')
print(f'  SNR: {SNR} dB')
print(f'  Samples per modulation: {SAMPLES_PER_MOD}')
print(f'  CW attack: steps={CW_STEPS}, c={CW_C}')

# Load dataset
print(f'\n[1/6] Loading dataset...')
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
        print(f'  Loaded {len(data)} samples for {mod.decode()}')

# Load AWN model
print(f'\n[2/6] Loading AWN model...')
from util.utils import create_AWN_model
from util.config import Config

cfg = Config('2016.10a', train=False)
cfg.device = DEVICE
model = create_AWN_model(cfg)
ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
model.load_state_dict(ckpt)
model.eval()
print(f'  ✓ Loaded model')

# Load detector
print(f'\n[3/6] Loading detector...')
from util.detector import RFSignalAutoEncoder

detector = RFSignalAutoEncoder().to(DEVICE)
det_ckpt = torch.load(DETECTOR_PATH, map_location=DEVICE, weights_only=False)
detector.load_state_dict(det_ckpt)
detector.eval()
print(f'  ✓ Loaded detector')

# Model wrapper for torchattacks (handles [0,1] and 4D per gpt.md)
wrapped = Model01Wrapper(model).eval()

# Generate CW attacks for each modulation
print(f'\n[4/6] Generating CW adversarial examples with proper [0,1]↔[-1,1] handling...')
atk = torchattacks.CW(wrapped, c=CW_C, kappa=0.0, steps=CW_STEPS, lr=1e-2)

X_adv_dict = {}
for mod in MODULATIONS:
    print(f'  Attacking {mod.decode()}...')
    X_clean = X_dict[mod]
    y_clean = y_dict[mod]

    # Convert inputs to [0,1] and add dummy dim for torchattacks
    X01_4d = iq_to_ta_input(X_clean)

    # Generate adversarial examples in batches
    batch_size = 50
    adv_batches = []
    for i in range(0, len(X_clean), batch_size):
        batch_x = X01_4d[i:i+batch_size]
        batch_y = y_clean[i:i+batch_size]
        adv01 = atk(batch_x, batch_y)
        adv_batches.append(adv01)

    # Convert back to [-1,1] IQ and drop dummy dim
    X_adv01_4d = torch.cat(adv_batches, dim=0)
    X_adv_dict[mod] = ta_output_to_iq(X_adv01_4d)
    print(f'    ✓ Generated {len(X_adv_dict[mod])} adversarial samples')

# Apply detector-gated FFT Top-K recovery
print(f'\n[5/6] Applying detector-gated FFT Top-K recovery...')
from util.defense import normalize_iq_data, denormalize_iq_data, fft_topk_denoise
from util.detector import detector_gate_fft_topk

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

    # Statistics
    defended = (kl_vals <= DETECTOR_THRESHOLD).sum().item()
    total = len(kl_vals)
    detection_stats[mod] = {
        'defended': defended,
        'total': total,
        'ratio': defended / total,
        'mean_kl': kl_vals.mean().item(),
        'max_kl': kl_vals.max().item()
    }
    print(f'    Defended: {defended}/{total} ({defended/total*100:.1f}%)')
    print(f'    Mean KL: {kl_vals.mean().item():.6f}, Max KL: {kl_vals.max().item():.6f}')

# Compute accuracies
print(f'\n[6/6] Computing accuracies...')
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
        logits_clean, _ = model(X_clean)
        preds_clean = torch.argmax(logits_clean, dim=1)
        clean_acc = (preds_clean == y_true).float().mean().item() * 100

        # Adversarial predictions
        logits_adv, _ = model(X_adv)
        preds_adv = torch.argmax(logits_adv, dim=1)
        adv_acc = (preds_adv == y_true).float().mean().item() * 100

        # Recovered predictions
        logits_rec, _ = model(X_rec)
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
print(f'\n[7/7] Generating constellation plots...')

fig = plt.figure(figsize=(16, 12))
gs = GridSpec(4, 3, figure=fig, hspace=0.35, wspace=0.25)

fig.suptitle('CW Attack on RF Modulation Classification: I/Q Constellation Analysis\n'
             f'(SNR={SNR} dB, CW steps={CW_STEPS}, c={CW_C}, Detector-gated FFT Top-K recovery with K={TOPK})',
             fontsize=14, fontweight='bold')

for idx, mod in enumerate(MODULATIONS):
    mod_name = mod.decode()

    # Get data
    X_clean = X_dict[mod].numpy()
    X_adv = X_adv_dict[mod].numpy()
    X_rec = X_rec_dict[mod].numpy()

    # Extract I and Q channels
    I_clean = X_clean[:, 0, :].flatten()
    Q_clean = X_clean[:, 1, :].flatten()

    I_adv = X_adv[:, 0, :].flatten()
    Q_adv = X_adv[:, 1, :].flatten()

    I_rec = X_rec[:, 0, :].flatten()
    Q_rec = X_rec[:, 1, :].flatten()

    # (a) Intact Data
    ax = fig.add_subplot(gs[idx, 0])
    ax.scatter(I_clean, Q_clean, c='blue', alpha=0.3, s=1, edgecolors='none')
    ax.set_title(f'{mod_name}\nIntact Data', fontsize=11, fontweight='bold')
    ax.set_xlabel('In-phase (I)', fontsize=9)
    ax.set_ylabel('Quadrature (Q)', fontsize=9)
    ax.grid(True, alpha=0.2)
    ax.set_xlim(-0.02, 0.02)
    ax.set_ylim(-0.02, 0.02)

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

    # Add accuracy and attack success
    acc_text = f'Acc: {accuracy_results[mod]["adversarial"]:.1f}%\nASR: {accuracy_results[mod]["clean"] - accuracy_results[mod]["adversarial"]:.1f}%'
    ax.text(0.05, 0.95, acc_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # (c) After Recovery (Detector-gated FFT Top-K)
    ax = fig.add_subplot(gs[idx, 2])
    ax.scatter(I_rec, Q_rec, c='green', alpha=0.3, s=1, edgecolors='none')
    ax.set_title(f'{mod_name}\nAfter Recovery', fontsize=11, fontweight='bold')
    ax.set_xlabel('In-phase (I)', fontsize=9)
    ax.set_ylabel('Quadrature (Q)', fontsize=9)
    ax.grid(True, alpha=0.2)
    ax.set_xlim(-0.02, 0.02)
    ax.set_ylim(-0.02, 0.02)

    # Add accuracy and recovery info
    acc_text = f'Acc: {accuracy_results[mod]["recovered"]:.1f}%\nGain: {accuracy_results[mod]["gain"]:+.1f}%\nDef: {detection_stats[mod]["ratio"]*100:.0f}%'
    ax.text(0.05, 0.95, acc_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.savefig('paper_cw_replication.png', dpi=300, bbox_inches='tight')
print(f'  ✓ Saved: paper_cw_replication.png')

plt.savefig('paper_cw_replication.pdf', bbox_inches='tight')
print(f'  ✓ Saved: paper_cw_replication.pdf')

# Generate recovered constellation plots
print(f'\n  Generating recovered constellation plots...')

fig2 = plt.figure(figsize=(16, 12))
gs2 = GridSpec(4, 3, figure=fig2, hspace=0.35, wspace=0.25)

fig2.suptitle('Recovered Constellation (MF + Symbol Timing + Phase Recovery)\n'
             f'(SNR={SNR} dB, sps=8, RRC beta=0.35)',
             fontsize=14, fontweight='bold')

for idx, mod in enumerate(MODULATIONS):
    mod_name = mod.decode()
    X_clean_np = X_dict[mod].numpy()
    X_adv_np = X_adv_dict[mod].numpy()
    X_rec_np = X_rec_dict[mod].numpy()

    for col, (data, color, label) in enumerate([
        (X_clean_np, 'blue', 'Intact'),
        (X_adv_np, 'red', 'CW Attack'),
        (X_rec_np, 'green', 'After Recovery'),
    ]):
        Is, Qs = [], []
        for j in range(data.shape[0]):
            ic, qc = recover_constellation(data[j, 0, :], data[j, 1, :], sps=8)
            Is.append(ic)
            Qs.append(qc)
        I_all = np.concatenate(Is)
        Q_all = np.concatenate(Qs)
        if len(I_all) > 12000:
            sel = np.random.choice(len(I_all), 12000, replace=False)
            I_all, Q_all = I_all[sel], Q_all[sel]

        ax = fig2.add_subplot(gs2[idx, col])
        ax.scatter(I_all, Q_all, c=color, alpha=0.4, s=5, edgecolors='none')
        ax.set_title(f'{mod_name}\n{label}', fontsize=11, fontweight='bold')
        ax.set_xlabel('I', fontsize=9)
        ax.set_ylabel('Q', fontsize=9)
        ax.grid(True, alpha=0.2)
        ax.set_aspect('equal', adjustable='box')

plt.savefig('paper_cw_constellation.png', dpi=300, bbox_inches='tight')
print(f'  ✓ Saved: paper_cw_constellation.png')
plt.savefig('paper_cw_constellation.pdf', bbox_inches='tight')
print(f'  ✓ Saved: paper_cw_constellation.pdf')
plt.close(fig2)

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
print(f'  Threshold:             {DETECTOR_THRESHOLD}')

print(f'\n{"="*70}')
print('✓ Experiment complete!')
print(f'  Output: paper_cw_replication.png')
print(f'{"="*70}')

# Save numerical results
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
with open('paper_cw_replication_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f'\n✓ Results saved to: paper_cw_replication_results.json')
