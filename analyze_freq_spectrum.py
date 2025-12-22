#!/usr/bin/env python
"""
Analyze the frequency spectrum characteristics of clean vs CW adversarial signals
to understand why FFT Top-K recovery isn't working.
"""

import torch
import numpy as np
import pickle

print('='*70)
print('Frequency Spectrum Analysis: Clean vs CW vs FFT Top-K')
print('='*70)

# Load a few samples at SNR=18 for analysis
DATASET_PATH = './data/RML2016.10a_dict.pkl'
MODEL_PATH = './2016.10a_AWN.pkl'
SNR = 18
NUM_SAMPLES = 10
CW_STEPS = 50
TOPK = 50
DEVICE = 'cpu'

print(f'\n[1/5] Loading {NUM_SAMPLES} samples at SNR={SNR} dB...')
with open(DATASET_PATH, 'rb') as f:
    Xd = pickle.load(f, encoding='bytes')

snrs, mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1, 0])

# Take samples from one modulation type
mod = b'QPSK'  # Use QPSK as example
X_samples = Xd[(mod, SNR)][:NUM_SAMPLES]
y_samples = np.array([mods.index(mod)] * NUM_SAMPLES)

X_tensor = torch.from_numpy(X_samples).float()
y_tensor = torch.from_numpy(y_samples).long()

print(f'  Loaded {NUM_SAMPLES} {mod.decode()} samples')

# Load model
print(f'\n[2/5] Loading model...')
from util.utils import create_AWN_model
from util.config import Config

cfg = Config('2016.10a', train=False)
cfg.device = DEVICE
model = create_AWN_model(cfg)
ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
model.load_state_dict(ckpt)
model.eval()

# Generate CW adversarial
print(f'\n[3/5] Generating CW adversarial examples...')
import torchattacks
from util.adv_attack import Model01Wrapper, iq_to_ta_input, ta_output_to_iq

wrapped = Model01Wrapper(model).eval()
atk = torchattacks.CW(wrapped, c=1.0, kappa=0.0, steps=CW_STEPS, lr=1e-2)
X01_4d = iq_to_ta_input(X_tensor)
X_adv01_4d = atk(X01_4d, y_tensor)
X_adv = ta_output_to_iq(X_adv01_4d)

# Apply FFT Top-K
print(f'\n[4/5] Applying FFT Top-K recovery...')
from util.defense import normalize_iq_data, denormalize_iq_data, fft_topk_denoise

X_adv_norm = normalize_iq_data(X_adv, 0.02, 0.04)
X_rec_norm = fft_topk_denoise(X_adv_norm, topk=TOPK)
X_rec = denormalize_iq_data(X_rec_norm, 0.02, 0.04)

# Analyze frequency characteristics
print(f'\n[5/5] Analyzing frequency characteristics...')
print(f'\n{"="*90}')
print(f'{"Sample":<8} {"Channel":<8} {"Top-10 Bins (Clean)":<30} {"Top-10 Bins (CW)":<30}')
print(f'{"="*90}')

for i in range(min(3, NUM_SAMPLES)):  # Analyze first 3 samples
    clean = X_tensor[i].numpy()
    adv = X_adv[i].numpy()
    pert = adv - clean

    for ch, ch_name in enumerate(['I', 'Q']):
        # FFT
        fft_clean = np.fft.fft(clean[ch])
        fft_adv = np.fft.fft(adv[ch])
        fft_pert = np.fft.fft(pert[ch])

        # Magnitudes
        mag_clean = np.abs(fft_clean)
        mag_adv = np.abs(fft_adv)
        mag_pert = np.abs(fft_pert)

        # Top-10 bins by magnitude
        top10_clean = np.argsort(mag_clean)[-10:][::-1]
        top10_adv = np.argsort(mag_adv)[-10:][::-1]
        top10_pert = np.argsort(mag_pert)[-10:][::-1]

        print(f'{i:<8} {ch_name:<8} {str(top10_clean.tolist()):<30} {str(top10_adv.tolist()):<30}')

print(f'{"="*90}')

# Detailed analysis for first sample
print(f'\n{"="*70}')
print(f'DETAILED FREQUENCY ANALYSIS (Sample 0, I channel)')
print(f'{"="*70}')

clean_0 = X_tensor[0, 0].numpy()  # I channel
adv_0 = X_adv[0, 0].numpy()
pert_0 = adv_0 - clean_0

fft_clean_0 = np.fft.fft(clean_0)
fft_adv_0 = np.fft.fft(adv_0)
fft_pert_0 = np.fft.fft(pert_0)

mag_clean_0 = np.abs(fft_clean_0)
mag_adv_0 = np.abs(fft_adv_0)
mag_pert_0 = np.abs(fft_pert_0)

# Energy distribution
total_energy_clean = np.sum(mag_clean_0**2)
total_energy_adv = np.sum(mag_adv_0**2)
total_energy_pert = np.sum(mag_pert_0**2)

# Energy in Top-K bins
top_k_indices_clean = np.argsort(mag_clean_0)[-TOPK:]
top_k_indices_adv = np.argsort(mag_adv_0)[-TOPK:]
top_k_indices_pert = np.argsort(mag_pert_0)[-TOPK:]

energy_topk_clean = np.sum(mag_clean_0[top_k_indices_clean]**2)
energy_topk_adv = np.sum(mag_adv_0[top_k_indices_adv]**2)
energy_topk_pert = np.sum(mag_pert_0[top_k_indices_pert]**2)

pct_topk_clean = energy_topk_clean / total_energy_clean * 100
pct_topk_adv = energy_topk_adv / total_energy_adv * 100
pct_topk_pert = energy_topk_pert / total_energy_pert * 100

print(f'\nEnergy Distribution:')
print(f'  Total bins: {len(clean_0)} (keeping Top-{TOPK} = {TOPK/len(clean_0)*100:.1f}%)')
print(f'\n  Clean signal:')
print(f'    Total energy: {total_energy_clean:.2f}')
print(f'    Energy in Top-{TOPK}: {energy_topk_clean:.2f} ({pct_topk_clean:.1f}%)')
print(f'\n  CW adversarial signal:')
print(f'    Total energy: {total_energy_adv:.2f}')
print(f'    Energy in Top-{TOPK}: {energy_topk_adv:.2f} ({pct_topk_adv:.1f}%)')
print(f'\n  CW perturbation:')
print(f'    Total energy: {total_energy_pert:.2f}')
print(f'    Energy in Top-{TOPK}: {energy_topk_pert:.2f} ({pct_topk_pert:.1f}%)')

# Bin overlap analysis
overlap_clean_adv = len(set(top_k_indices_clean) & set(top_k_indices_adv))
print(f'\nBin Overlap Analysis:')
print(f'  Top-{TOPK} bins overlap (Clean vs CW): {overlap_clean_adv}/{TOPK} ({overlap_clean_adv/TOPK*100:.1f}%)')

# Where is perturbation energy concentrated?
# Check if perturbation is in low/mid/high frequency
low_freq_bins = list(range(0, len(clean_0)//4))
mid_freq_bins = list(range(len(clean_0)//4, len(clean_0)//2))
high_freq_bins = list(range(len(clean_0)//2, len(clean_0)))

energy_pert_low = np.sum(mag_pert_0[low_freq_bins]**2)
energy_pert_mid = np.sum(mag_pert_0[mid_freq_bins]**2)
energy_pert_high = np.sum(mag_pert_0[high_freq_bins]**2)

print(f'\nPerturbation Energy by Frequency Band:')
print(f'  Low freq  (bins 0-{len(clean_0)//4}):   {energy_pert_low:.2f} ({energy_pert_low/total_energy_pert*100:.1f}%)')
print(f'  Mid freq  (bins {len(clean_0)//4}-{len(clean_0)//2}):  {energy_pert_mid:.2f} ({energy_pert_mid/total_energy_pert*100:.1f}%)')
print(f'  High freq (bins {len(clean_0)//2}-{len(clean_0)}): {energy_pert_high:.2f} ({energy_pert_high/total_energy_pert*100:.1f}%)')

# Check if perturbation bins overlap with clean signal's top bins
pert_top10 = np.argsort(mag_pert_0)[-10:][::-1]
clean_top50 = np.argsort(mag_clean_0)[-50:][::-1]
pert_in_clean_top50 = len(set(pert_top10) & set(clean_top50))

print(f'\nPerturbation vs Clean Signal:')
print(f'  Top-10 perturbation bins inside Clean Top-50: {pert_in_clean_top50}/10')
print(f'  Top-10 perturbation bins: {pert_top10.tolist()}')

# Predictions
with torch.no_grad():
    logits_clean, _ = model(X_tensor[[0]])
    logits_adv, _ = model(X_adv[[0]])
    logits_rec, _ = model(X_rec[[0]])

    pred_clean = torch.argmax(logits_clean, dim=1).item()
    pred_adv = torch.argmax(logits_adv, dim=1).item()
    pred_rec = torch.argmax(logits_rec, dim=1).item()

print(f'\nPredictions (Sample 0):')
print(f'  True label: {y_tensor[0].item()} ({mod.decode()})')
print(f'  Clean:      {pred_clean} {"✓" if pred_clean == y_tensor[0].item() else "✗"}')
print(f'  CW Attack:  {pred_adv} {"✓" if pred_adv == y_tensor[0].item() else "✗ FOOLED"}')
print(f'  Recovered:  {pred_rec} {"✓ RECOVERED" if pred_rec == y_tensor[0].item() else "✗"}')

print(f'\n{"="*70}')
print('INTERPRETATION:')
print('='*70)

if pct_topk_pert > 80:
    print('✗ Most perturbation energy is in Top-K bins')
    print('  → FFT Top-K keeps the attack (explains why recovery fails)')
    print('  → CW attack is "frequency-sparse" and overlaps with clean signal')
elif pct_topk_pert < 40:
    print('✓ Most perturbation energy is outside Top-K bins')
    print('  → FFT Top-K should remove the attack')
    print('  → If recovery still fails, check if clean signal energy is also removed')
else:
    print('~ Perturbation energy is split between Top-K and non-Top-K bins')
    print('  → FFT Top-K provides partial attack removal')

if pert_in_clean_top50 >= 7:
    print('\n✗ Perturbation overlaps heavily with clean signal frequencies')
    print('  → Cannot filter without damaging clean signal')
    print('  → This is why FFT Top-K doesn\'t help')
else:
    print('\n✓ Perturbation uses different frequencies than clean signal')
    print('  → FFT Top-K has potential to work')

print(f'{"="*70}')
