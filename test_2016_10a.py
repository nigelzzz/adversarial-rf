#!/usr/bin/env python
"""Test 2016.10a model with CW attack and recovery on SNR >= 0"""
import torch
import numpy as np
import pickle
from tqdm import tqdm

print('='*70)
print('CW Attack + Recovery Test (SNR >= 0)')
print('Model: 2016.10a_AWN.pkl | Dataset: RML2016.10a_dict.pkl')
print('='*70)

# Config
SNR_THRESHOLD = 0
NUM_SAMPLES = 500
CW_STEPS = 30
TOPK = 50
DEVICE = 'cpu'

# Load dataset
print('\n[1/6] Loading dataset...')
with open('./data/RML2016.10a_dict.pkl', 'rb') as f:
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

# Filter SNR >= 0
mask = snr_vals >= SNR_THRESHOLD
X_filt = X[mask]
lbl_filt = lbl[mask]
snr_filt = snr_vals[mask]

# Use random sampling across all SNRs
np.random.seed(42)
indices = np.random.choice(len(X_filt), min(NUM_SAMPLES, len(X_filt)), replace=False)
X_test = X_filt[indices]
lbl_test = lbl_filt[indices]
snr_test = snr_filt[indices]

X_test = torch.from_numpy(X_test).float()
y_test = torch.from_numpy(lbl_test).long()

print(f'  Samples: {len(X_test)} (SNR {snr_test.min():.0f} to {snr_test.max():.0f} dB)')

# Load model
print('\n[2/6] Loading pretrained model...')
from util.utils import create_AWN_model
from util.config import Config

cfg = Config('2016.10a', train=False)
cfg.device = DEVICE
model = create_AWN_model(cfg)

ckpt = torch.load('./2016.10a_AWN.pkl', map_location=DEVICE, weights_only=True)
model.load_state_dict(ckpt)
model.eval()
print('  ✓ Loaded 2016.10a_AWN.pkl')

# Clean accuracy
print('\n[3/6] Measuring clean accuracy...')
with torch.no_grad():
    logits, _ = model(X_test)
    clean_preds = torch.argmax(logits, dim=1)
clean_acc = (clean_preds == y_test).float().mean().item() * 100
print(f'  Clean Accuracy: {clean_acc:.2f}%')

# Setup CW attack
print(f'\n[4/6] Generating CW adversarial examples (steps={CW_STEPS})...')
import torchattacks
from util.adv_attack import Model01Wrapper, iq_to_ta_input, ta_output_to_iq

wrapped = Model01Wrapper(model).eval()
atk = torchattacks.CW(wrapped, c=1.0, kappa=0.0, steps=CW_STEPS, lr=1e-2)

# Generate adversarial examples
bs = 50
adv_all = []
for i in tqdm(range(0, len(X_test), bs), total=(len(X_test)+bs-1)//bs):
    batch_x = X_test[i:i+bs]
    batch_y = y_test[i:i+bs]
    batch_x01_4d = iq_to_ta_input(batch_x)
    adv01_4d = atk(batch_x01_4d, batch_y)
    adv = ta_output_to_iq(adv01_4d)
    adv_all.append(adv)

adv_all = torch.cat(adv_all, dim=0)

# Adversarial accuracy
print('\n[5/6] Measuring adversarial accuracy...')
with torch.no_grad():
    logits_adv, _ = model(adv_all)
    adv_preds = torch.argmax(logits_adv, dim=1)
adv_acc = (adv_preds == y_test).float().mean().item() * 100
print(f'  After CW: {adv_acc:.2f}%')

# Recovery
print(f'\n[6/6] Applying FFT Top-K recovery (K={TOPK})...')
from util.defense import normalize_iq_data, denormalize_iq_data, fft_topk_denoise

rec_all = []
for i in tqdm(range(0, len(adv_all), bs), total=(len(adv_all)+bs-1)//bs):
    batch = adv_all[i:i+bs]
    norm = normalize_iq_data(batch, 0.02, 0.04)
    filt = fft_topk_denoise(norm, topk=TOPK)
    rec = denormalize_iq_data(filt, 0.02, 0.04)
    rec_all.append(rec)

rec_all = torch.cat(rec_all, dim=0)

with torch.no_grad():
    logits_rec, _ = model(rec_all)
    rec_preds = torch.argmax(logits_rec, dim=1)
rec_acc = (rec_preds == y_test).float().mean().item() * 100

# Results
print()
print('='*70)
print(f'RESULTS ({NUM_SAMPLES} samples, SNR >= {SNR_THRESHOLD} dB)')
print('='*70)
print(f'  1. Clean Accuracy:      {clean_acc:6.2f}%')
print(f'  2. After CW Attack:     {adv_acc:6.2f}%  (↓ {clean_acc - adv_acc:.2f}%)')
print(f'  3. After Recovery:      {rec_acc:6.2f}%  (↑ {rec_acc - adv_acc:+.2f}%)')
print()
print(f'  Recovery Gain: {rec_acc - adv_acc:+.2f}%')
if clean_acc > 0:
    print(f'  Overall: {rec_acc/clean_acc*100:.1f}% of clean accuracy recovered')
print('='*70)

# Per-SNR breakdown
unique_snrs = sorted(np.unique(snr_test))
print('\nPer-SNR Breakdown:')
print(f'{"SNR":<6} {"N":<6} {"Clean%":<10} {"After CW%":<12} {"Recovered%":<12} {"Gain"}')
print('-'*70)

for snr_val in unique_snrs:
    mask = snr_test == snr_val
    mask_t = torch.from_numpy(mask)
    n = mask.sum()

    c = (clean_preds[mask_t] == y_test[mask_t]).float().mean().item() * 100
    a = (adv_preds[mask_t] == y_test[mask_t]).float().mean().item() * 100
    r = (rec_preds[mask_t] == y_test[mask_t]).float().mean().item() * 100

    print(f'{snr_val:>4.0f}   {n:<6} {c:>6.2f}%    {a:>6.2f}%      {r:>6.2f}%       {r-a:+6.2f}%')
