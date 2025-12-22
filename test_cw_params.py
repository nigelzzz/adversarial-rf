#!/usr/bin/env python
"""
Test different CW attack parameters to find settings where:
1. Attack is effective (reduces accuracy)
2. Perturbation is small (signal similar to original)
3. Recovery has potential to work

This helps identify the sweet spot for evaluating FFT Top-K recovery.
"""

import torch
import numpy as np
import pickle
from tqdm import tqdm
import torchattacks
from util.adv_attack import Model01Wrapper, iq_to_ta_input, ta_output_to_iq

print('='*70)
print('CW Parameter Sweep (SNR >= 0)')
print('='*70)

# Configuration
DATASET_PATH = './data/RML2016.10a_dict.pkl'
MODEL_PATH = './2016.10a_AWN.pkl'
SNR_THRESHOLD = 0
NUM_SAMPLES = 1000  # Test on reasonable sample size
DEVICE = 'cpu'
TOPK = 50

# CW parameter grid to test
CW_CONFIGS = [
    # (steps, c, kappa, description)
    (10, 0.1, 0.0, 'Very weak attack (c=0.1, steps=10)'),
    (30, 0.1, 0.0, 'Weak attack (c=0.1, steps=30)'),
    (30, 0.5, 0.0, 'Medium-weak attack (c=0.5, steps=30)'),
    (30, 1.0, 0.0, 'Medium attack (c=1.0, steps=30)'),
    (50, 1.0, 0.0, 'Strong attack (c=1.0, steps=50)'),
    (100, 1.0, 0.0, 'Very strong attack (c=1.0, steps=100)'),
    (30, 2.0, 0.0, 'High confidence attack (c=2.0, steps=30)'),
]

print(f'\nConfiguration:')
print(f'  Dataset: {DATASET_PATH}')
print(f'  Model: {MODEL_PATH}')
print(f'  SNR threshold: >= {SNR_THRESHOLD} dB')
print(f'  Test samples: {NUM_SAMPLES}')
print(f'  FFT Top-K: {TOPK}')
print(f'  Attack configs: {len(CW_CONFIGS)}')

# Load dataset
print(f'\n[1/4] Loading dataset...')
with open(DATASET_PATH, 'rb') as f:
    Xd = pickle.load(f, encoding='bytes')

snrs, mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1, 0])
X, lbl, snr_vals = [], [], []

for mod in mods:
    for snr in snrs:
        X.append(Xd[(mod, snr)])
        for i in range(Xd[(mod, snr)].shape[0]):
            lbl.append(mods.index(mod))
            snr_vals.append(snr)

X = np.vstack(X)
lbl = np.array(lbl)
snr_vals = np.array(snr_vals)

# Filter SNR >= threshold
mask = snr_vals >= SNR_THRESHOLD
X_filt = X[mask]
lbl_filt = lbl[mask]
snr_filt = snr_vals[mask]

print(f'  Total samples: {len(X)}')
print(f'  After SNR >= {SNR_THRESHOLD} filter: {len(X_filt)}')

# Random sample for testing
np.random.seed(42)
indices = np.random.choice(len(X_filt), min(NUM_SAMPLES, len(X_filt)), replace=False)
X_test = X_filt[indices]
lbl_test = lbl_filt[indices]
snr_test = snr_filt[indices]

X_test = torch.from_numpy(X_test).float()
y_test = torch.from_numpy(lbl_test).long()

print(f'  Selected {len(X_test)} samples (SNR {snr_test.min():.0f} to {snr_test.max():.0f} dB)')

# Load model
print(f'\n[2/4] Loading AWN model...')
from util.utils import create_AWN_model
from util.config import Config

cfg = Config('2016.10a', train=False)
cfg.device = DEVICE
model = create_AWN_model(cfg)

ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
model.load_state_dict(ckpt)
model.eval()
print(f'  ✓ Loaded {MODEL_PATH}')

wrapped = Model01Wrapper(model).eval()

# Clean accuracy
print(f'\n[3/4] Computing clean accuracy...')
with torch.no_grad():
    logits_clean, _ = model(X_test)
    preds_clean = torch.argmax(logits_clean, dim=1)
clean_acc = (preds_clean == y_test).float().mean().item() * 100
print(f'  Clean Accuracy: {clean_acc:.2f}%')

# FFT Top-K recovery functions
from util.defense import normalize_iq_data, denormalize_iq_data, fft_topk_denoise

# Test each CW configuration
print(f'\n[4/4] Testing CW attack configurations...')
print(f'\n{"="*120}')
print(f'{"Config":<35} {"Steps":>6} {"c":>6} {"Adv%":>8} {"Drop%":>8} {"L2 Norm":>10} {"Rec%":>8} {"Gain%":>8} {"Sim%":>8}')
print(f'{"="*120}')

results = []

for steps, c, kappa, desc in CW_CONFIGS:
    # Generate adversarial examples
    atk = torchattacks.CW(wrapped, c=c, kappa=kappa, steps=steps, lr=1e-2)

    batch_size = 50
    adv_all = []
    for i in range(0, len(X_test), batch_size):
        batch_x = X_test[i:i+batch_size]
        batch_y = y_test[i:i+batch_size]
        batch_x01_4d = iq_to_ta_input(batch_x)
        adv01_4d = atk(batch_x01_4d, batch_y)
        adv = ta_output_to_iq(adv01_4d)
        adv_all.append(adv)

    adv_all = torch.cat(adv_all, dim=0)

    # Compute perturbation statistics
    pert = adv_all - X_test
    l2_pert = torch.norm(pert.view(len(pert), -1), p=2, dim=1).mean().item()
    l2_clean = torch.norm(X_test.view(len(X_test), -1), p=2, dim=1).mean().item()
    rel_pert = l2_pert / l2_clean

    # Adversarial accuracy
    with torch.no_grad():
        logits_adv, _ = model(adv_all)
        preds_adv = torch.argmax(logits_adv, dim=1)
    adv_acc = (preds_adv == y_test).float().mean().item() * 100

    # Apply FFT Top-K recovery
    rec_all = []
    for i in range(0, len(adv_all), batch_size):
        batch = adv_all[i:i+batch_size]
        norm = normalize_iq_data(batch, 0.02, 0.04)
        filt = fft_topk_denoise(norm, topk=TOPK)
        rec = denormalize_iq_data(filt, 0.02, 0.04)
        rec_all.append(rec)

    rec_all = torch.cat(rec_all, dim=0)

    with torch.no_grad():
        logits_rec, _ = model(rec_all)
        preds_rec = torch.argmax(logits_rec, dim=1)
    rec_acc = (preds_rec == y_test).float().mean().item() * 100

    # Compute similarity (1 - relative perturbation)
    similarity = (1 - rel_pert) * 100

    # Recovery gain
    rec_gain = rec_acc - adv_acc

    # Accuracy drop
    acc_drop = clean_acc - adv_acc

    # Store results
    results.append({
        'desc': desc,
        'steps': steps,
        'c': c,
        'adv_acc': adv_acc,
        'acc_drop': acc_drop,
        'l2_pert': l2_pert,
        'rel_pert': rel_pert,
        'rec_acc': rec_acc,
        'rec_gain': rec_gain,
        'similarity': similarity,
    })

    print(f'{desc:<35} {steps:>6} {c:>6.1f} {adv_acc:>7.2f}% {acc_drop:>7.2f}% {l2_pert:>10.4f} {rec_acc:>7.2f}% {rec_gain:>+7.2f}% {similarity:>7.1f}%')

print(f'{"="*120}')

# Summary and recommendations
print(f'\n{"="*70}')
print(f'SUMMARY & RECOMMENDATIONS')
print(f'{"="*70}')
print(f'\nClean Accuracy: {clean_acc:.2f}%\n')

# Find best configurations
print('Best configurations for recovery evaluation:\n')

# 1. Best recovery gain
best_gain = max(results, key=lambda x: x['rec_gain'])
print(f'1. Best Recovery Gain:')
print(f'   {best_gain["desc"]}')
print(f'   Adv: {best_gain["adv_acc"]:.2f}% → Rec: {best_gain["rec_acc"]:.2f}% (gain: {best_gain["rec_gain"]:+.2f}%)')
print(f'   Similarity: {best_gain["similarity"]:.1f}%\n')

# 2. Best trade-off (good attack, high similarity)
scored = [(r, r['acc_drop'] * r['similarity']) for r in results]
best_tradeoff = max(scored, key=lambda x: x[1])[0]
print(f'2. Best Attack/Similarity Trade-off:')
print(f'   {best_tradeoff["desc"]}')
print(f'   Adv: {best_tradeoff["adv_acc"]:.2f}% (drop: {best_tradeoff["acc_drop"]:.2f}%)')
print(f'   Similarity: {best_tradeoff["similarity"]:.1f}%')
print(f'   Recovery: {best_tradeoff["rec_acc"]:.2f}% (gain: {best_tradeoff["rec_gain"]:+.2f}%)\n')

# 3. Moderate attack (40-60% accuracy after attack)
moderate = [r for r in results if 40 <= r['adv_acc'] <= 60]
if moderate:
    best_moderate = max(moderate, key=lambda x: x['similarity'])
    print(f'3. Best Moderate Attack (40-60% accuracy):')
    print(f'   {best_moderate["desc"]}')
    print(f'   Adv: {best_moderate["adv_acc"]:.2f}% → Rec: {best_moderate["rec_acc"]:.2f}% (gain: {best_moderate["rec_gain"]:+.2f}%)')
    print(f'   Similarity: {best_moderate["similarity"]:.1f}%\n')

print('Observations:')
if all(r['rec_gain'] <= 0.5 for r in results):
    print('  ⚠ FFT Top-K recovery shows minimal improvement across all CW configurations')
    print('  ⚠ This suggests CW perturbations are primarily low-frequency (within Top-K bins)')
    print('  → Recommendation: Try detector-gated recovery (ae_fft_topk) instead')
else:
    gainful = [r for r in results if r['rec_gain'] > 1.0]
    print(f'  ✓ Found {len(gainful)} configuration(s) with recovery gain > 1%')
    print('  → Recommendation: Use these configs for further evaluation')

print(f'\n{"="*70}')

# Save results
import json
with open('cw_param_sweep_results.json', 'w') as f:
    json.dump({
        'clean_acc': clean_acc,
        'num_samples': len(X_test),
        'snr_range': [int(snr_test.min()), int(snr_test.max())],
        'topk': TOPK,
        'results': results
    }, f, indent=2)

print(f'\n✓ Results saved to cw_param_sweep_results.json')
