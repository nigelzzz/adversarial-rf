#!/usr/bin/env python
"""
Simple demonstration of CW attack with recovery on SNR >= 0 samples.

This demo shows the complete pipeline:
1. Load data and filter for SNR >= 0
2. Generate CW adversarial examples using torchattacks
3. Apply FFT Top-K recovery
4. Compare predictions before/after recovery

Usage:
    python demo_cw_recovery.py
"""

import torch
import numpy as np
import pickle
from tqdm import tqdm

print("="*70)
print("CW Attack + FFT Top-K Recovery Demo (SNR >= 0)")
print("="*70)

# Configuration
DATASET_PATH = './data/RML2016.10a_dict.pkl'
SNR_THRESHOLD = 0
NUM_SAMPLES = 500  # Limit for quick demo
CW_STEPS = 30  # Reduced for speed
DEF_TOPK = 50
DEVICE = 'cpu'

print(f"\nConfiguration:")
print(f"  Dataset: {DATASET_PATH}")
print(f"  SNR threshold: >= {SNR_THRESHOLD} dB")
print(f"  Test samples: {NUM_SAMPLES}")
print(f"  CW steps: {CW_STEPS}")
print(f"  FFT Top-K: {DEF_TOPK}")
print(f"  Device: {DEVICE}")

# Step 1: Load data
print(f"\n[1/5] Loading dataset...")
try:
    with open(DATASET_PATH, 'rb') as f:
        Xd = pickle.load(f, encoding='bytes')
except Exception as e:
    print(f"ERROR: Could not load dataset: {e}")
    print(f"Please ensure {DATASET_PATH} exists")
    exit(1)

# Extract modulations and SNRs
snrs, mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1, 0])
X = []
lbl = []
snr_vals = []

for mod in mods:
    for snr in snrs:
        X.append(Xd[(mod, snr)])
        for i in range(Xd[(mod, snr)].shape[0]):
            lbl.append(mods.index(mod))
            snr_vals.append(snr)

X = np.vstack(X)
lbl = np.array(lbl)
snr_vals = np.array(snr_vals)

print(f"  Total samples: {len(X)}")
print(f"  Modulations: {len(mods)}")
print(f"  SNR range: {min(snrs)} to {max(snrs)} dB")

# Filter for SNR >= threshold
mask = snr_vals >= SNR_THRESHOLD
X_filt = X[mask]
lbl_filt = lbl[mask]
snr_filt = snr_vals[mask]

print(f"  After SNR >= {SNR_THRESHOLD} filter: {len(X_filt)} samples")

# Limit samples for demo
if len(X_filt) > NUM_SAMPLES:
    indices = np.random.choice(len(X_filt), NUM_SAMPLES, replace=False)
    X_filt = X_filt[indices]
    lbl_filt = lbl_filt[indices]
    snr_filt = snr_filt[indices]
    print(f"  Using {NUM_SAMPLES} random samples for demo")

# Convert to torch tensors
X_test = torch.from_numpy(X_filt).float().to(DEVICE)
y_test = torch.from_numpy(lbl_filt).long().to(DEVICE)

print(f"  Final test set: {X_test.shape}")

# Step 2: Create model
print(f"\n[2/5] Creating AWN model...")
from util.utils import create_AWN_model
from util.config import Config

cfg = Config('2016.10a', train=False)
cfg.device = DEVICE
model = create_AWN_model(cfg)
model.to(DEVICE)
model.eval()

# Try to load pretrained weights if available
try:
    ckpt = torch.load('./2016.10a_AWN.pkl', map_location=DEVICE, weights_only=True)
    model.load_state_dict(ckpt)
    print(f"  ✓ Loaded pretrained weights")
    PRETRAINED = True
except:
    print(f"  ⚠ No pretrained weights found, using random initialization")
    print(f"    (Results will not be meaningful, but demo will run)")
    PRETRAINED = False

# Step 3: Check torchattacks and wrap model
print(f"\n[3/5] Setting up torchattacks...")
try:
    import torchattacks
    print(f"  ✓ torchattacks {torchattacks.__version__} available")
except ImportError:
    print(f"  ERROR: torchattacks not installed")
    print(f"  Run: pip install torchattacks")
    exit(1)

# Wrap AWN model to return only logits (torchattacks expects single output)
from util.adv_attack import Model01Wrapper, iq_to_ta_input, ta_output_to_iq
wrapped_model = Model01Wrapper(model).eval()
print(f"  ✓ Wrapped model for torchattacks compatibility ([0,1] 4D input)")

# Step 4: Generate CW adversarial examples
print(f"\n[4/5] Generating CW adversarial examples...")
atk = torchattacks.CW(wrapped_model, c=1.0, kappa=0.0, steps=CW_STEPS, lr=1e-2)

batch_size = 50
adv_samples = []
clean_preds_all = []
adv_preds_all = []

num_batches = (len(X_test) + batch_size - 1) // batch_size

for i in tqdm(range(0, len(X_test), batch_size), total=num_batches):
    batch_x = X_test[i:i+batch_size]
    batch_y = y_test[i:i+batch_size]

    # Clean predictions
    with torch.no_grad():
        logits_clean, _ = model(batch_x)
        preds_clean = torch.argmax(logits_clean, dim=1)
        clean_preds_all.append(preds_clean)

    # Generate adversarial examples via [0,1] 4D mapping
    batch_x01_4d = iq_to_ta_input(batch_x)
    adv01_4d = atk(batch_x01_4d, batch_y)
    adv = ta_output_to_iq(adv01_4d)
    adv_samples.append(adv)

    # Adversarial predictions
    with torch.no_grad():
        logits_adv, _ = model(adv)
        preds_adv = torch.argmax(logits_adv, dim=1)
        adv_preds_all.append(preds_adv)

adv_samples = torch.cat(adv_samples, dim=0)
clean_preds_all = torch.cat(clean_preds_all)
adv_preds_all = torch.cat(adv_preds_all)

print(f"  ✓ Generated {len(adv_samples)} adversarial examples")

# Step 5: Apply FFT Top-K recovery
print(f"\n[5/5] Applying FFT Top-K recovery (AWN_All.py pattern)...")
from util.defense import normalize_iq_data, denormalize_iq_data, fft_topk_denoise

recovered_preds_all = []

for i in tqdm(range(0, len(adv_samples), batch_size), total=num_batches):
    batch_adv = adv_samples[i:i+batch_size]

    # Normalize
    batch_norm = normalize_iq_data(batch_adv, offset=0.02, scale=0.04)

    # FFT Top-K filtering
    batch_filt = fft_topk_denoise(batch_norm, topk=DEF_TOPK)

    # Denormalize
    batch_recovered = denormalize_iq_data(batch_filt, offset=0.02, scale=0.04)

    # Predictions after recovery
    with torch.no_grad():
        logits_rec, _ = model(batch_recovered)
        preds_rec = torch.argmax(logits_rec, dim=1)
        recovered_preds_all.append(preds_rec)

recovered_preds_all = torch.cat(recovered_preds_all)

# Results
clean_acc = (clean_preds_all == y_test).float().mean().item() * 100
adv_acc = (adv_preds_all == y_test).float().mean().item() * 100
rec_acc = (recovered_preds_all == y_test).float().mean().item() * 100
recovery_gain = rec_acc - adv_acc

print(f"\n{'='*70}")
print(f"RESULTS ({NUM_SAMPLES} samples, SNR >= {SNR_THRESHOLD} dB)")
print(f"{'='*70}")

if PRETRAINED:
    print(f"  Clean Accuracy:        {clean_acc:6.2f}%")
    print(f"  Adversarial Accuracy:  {adv_acc:6.2f}%  (CW attack degradation: {clean_acc - adv_acc:-.2f}%)")
    print(f"  Recovered Accuracy:    {rec_acc:6.2f}%  (FFT Top-K={DEF_TOPK})")
    print(f"  Recovery Gain:         {recovery_gain:+6.2f}%")
else:
    print(f"  ⚠ Using random weights (not trained model)")
    print(f"  Clean Accuracy:        {clean_acc:6.2f}%")
    print(f"  Adversarial Accuracy:  {adv_acc:6.2f}%")
    print(f"  Recovered Accuracy:    {rec_acc:6.2f}%")
    print(f"  Recovery Gain:         {recovery_gain:+6.2f}%")
    print(f"\n  Note: Train a model first to see meaningful results:")
    print(f"    python main.py --mode train --dataset 2016.10a")

print(f"{'='*70}")

# Per-SNR breakdown
unique_snrs = sorted(np.unique(snr_filt))
if len(unique_snrs) <= 10:  # Only show if reasonable number of SNRs
    print(f"\nPer-SNR Breakdown:")
    print(f"{'SNR':<6} {'Samples':<10} {'Clean%':<10} {'Adv%':<10} {'Recovered%':<12} {'Gain'}")
    print(f"{'-'*70}")

    for snr_val in unique_snrs:
        mask = snr_filt == snr_val
        mask_t = torch.from_numpy(mask)

        n_samples = mask.sum()
        clean_acc_snr = (clean_preds_all[mask_t] == y_test[mask_t]).float().mean().item() * 100
        adv_acc_snr = (adv_preds_all[mask_t] == y_test[mask_t]).float().mean().item() * 100
        rec_acc_snr = (recovered_preds_all[mask_t] == y_test[mask_t]).float().mean().item() * 100
        gain = rec_acc_snr - adv_acc_snr

        print(f"{snr_val:>4.0f}   {n_samples:<10} {clean_acc_snr:>6.2f}%    {adv_acc_snr:>6.2f}%    {rec_acc_snr:>6.2f}%       {gain:+6.2f}%")

print(f"\n✓ Demo complete!")
print(f"\nKey findings:")
print(f"  1. torchattacks CW attack successfully integrated")
print(f"  2. FFT Top-K recovery (AWN_All.py pattern) applied")
print(f"  3. Recovery gain of {recovery_gain:+.2f}% demonstrated")
