# CW Attack with Recovery - Implementation Guide

This guide documents the integration of the torchattacks library for CW attacks and the FFT Top-K recovery mechanism inspired by AWN_All.py.

## Overview

The implementation provides:

1. **CW Attack using torchattacks**: Standard Carlini-Wagner L2 attack from the widely-used [torchattacks library](https://github.com/Harry24k/adversarial-attacks-pytorch)
2. **FFT Top-K Recovery**: Keep only top-K frequency components per I/Q channel (AWN_All.py pattern)
3. **Detector-Gated Recovery**: Optional autoencoder-based anomaly detection to selectively apply recovery

## Quick Start

### Basic CW Attack

```bash
# Standard CW attack with torchattacks (default backend)
python main.py --mode adv_eval --dataset 2016.10a --attack cw \
    --cw_steps 100 --cw_c 1.0 --ckpt_path ./2016.10a_AWN.pkl
```

### CW with FFT Top-K Recovery

```bash
# Apply FFT Top-K to all adversarial samples
python main.py --mode adv_eval --dataset 2016.10a --attack cw \
    --defense fft_topk --def_topk 50 --ckpt_path ./2016.10a_AWN.pkl
```

### CW with Detector-Gated Recovery (Recommended)

```bash
# Step 1: Train the autoencoder detector
python main.py --mode train_detector --dataset 2016.10a --det_epochs 10

# Step 2: Calibrate the KL divergence threshold
python main.py --mode calibrate_detector --dataset 2016.10a \
    --detector_ckpt ./checkpoint/detector_ae.pth

# Step 3: Run CW attack with detector-gated recovery
python main.py --mode adv_eval --dataset 2016.10a --attack cw \
    --defense ae_fft_topk --def_topk 50 \
    --detector_ckpt ./checkpoint/detector_ae.pth \
    --detector_threshold 0.004468 \
    --ckpt_path ./2016.10a_AWN.pkl
```

### Standalone Test Script

```bash
# Test CW attack with recovery on a subset
python test_cw_recovery.py --dataset 2016.10a \
    --ckpt_path ./2016.10a_AWN.pkl \
    --defense fft_topk --def_topk 50 \
    --test_samples 500
```

## Key Parameters

### CW Attack Parameters
- `--attack cw`: Enable CW attack mode
- `--attack_backend torchattacks`: Use torchattacks library (default)
- `--attack_backend internal`: Use custom CW implementation
- `--cw_steps 100`: Number of optimization steps
- `--cw_c 1.0`: Confidence parameter (balance between perturbation and misclassification)
- `--cw_kappa 0.0`: Margin parameter
- `--cw_lr 1e-2`: Learning rate for optimization
- `--lowpass True`: Apply low-pass filter to perturbation (IQ-specific)

### Defense Parameters
- `--defense fft_topk`: Apply FFT Top-K to all samples
- `--defense ae_fft_topk`: Apply FFT Top-K only to samples flagged by detector
- `--def_topk 50`: Number of frequency components to keep (default 50)
- `--cmp_defense True`: Compare accuracy before/after defense

### Detector Parameters
- `--detector_ckpt <path>`: Path to pretrained autoencoder checkpoint
- `--detector_threshold 0.004468`: KL divergence threshold for anomaly detection
- `--detector_norm_offset 0.02`: Normalization offset for detector (default 0.02)
- `--detector_norm_scale 0.04`: Normalization scale for detector (default 0.04)

## Implementation Details

### Recovery Pipeline (AWN_All.py Pattern)

The recovery mechanism follows these steps:

1. **Normalization**: `x_norm = (x + 0.02) / 0.04`
2. **FFT Transform**: Apply full complex FFT to both I and Q channels
3. **Top-K Selection**: Keep only K bins with largest magnitude per channel
4. **Inverse FFT**: Reconstruct time-domain signal
5. **Denormalization**: `x_recovered = x_norm * 0.04 - 0.02`

### Detector-Gated Recovery

When using `ae_fft_topk` defense:

1. **Detection**: Pass normalized input through autoencoder
2. **KL Divergence**: Compute KL(input || reconstruction) per sample
3. **Thresholding**: If KL > threshold, apply FFT Top-K; otherwise pass through
4. **Classification**: Feed recovered signals to classifier

### Code Organization

- **`util/adv_attack.py`**: Custom CW implementation (fallback)
- **`util/adv_eval.py`**: Adversarial evaluation loop with torchattacks integration
- **`util/defense.py`**: FFT-based defenses including Top-K and normalization utilities
- **`util/detector.py`**: Autoencoder detector and KL divergence computation
- **`util/detector_train.py`**: Training loop for detector
- **`test_cw_recovery.py`**: Standalone test script

## Files Modified

1. **`util/defense.py`**: Added normalization/denormalization utilities and `fft_topk_denoise_normalized`
2. **`util/adv_eval.py`**: Changed default backend from `'internal'` to `'torchattacks'`
3. **`main.py`**: Updated `--attack_backend` default to `'torchattacks'`
4. **`CLAUDE.md`**: Updated documentation with CW+recovery examples
5. **`test_cw_recovery.py`**: New standalone test script

## Performance Tips

1. **Limit test samples**: Use `--eval_limit 1000` for faster iteration during development
2. **Batch size**: Adjust based on GPU memory (default 50 samples per batch)
3. **CW steps**: 100 steps is usually sufficient; more steps = stronger attack but slower
4. **Top-K value**: Start with 50; higher values retain more frequency content
5. **Detector threshold**: Calibrate on validation set using `--mode calibrate_detector`

## Example Workflow

```bash
# 1. Train AWN classifier (if not already done)
python main.py --mode train --dataset 2016.10a

# 2. Train detector on clean signals
python main.py --mode train_detector --dataset 2016.10a --det_epochs 10

# 3. Calibrate detector threshold (90th percentile KL on val set)
python main.py --mode calibrate_detector --dataset 2016.10a \
    --detector_ckpt ./checkpoint/detector_ae.pth \
    --det_calib_quantile 0.90

# 4. Run CW attack with detector-gated recovery
python main.py --mode adv_eval --dataset 2016.10a --attack cw \
    --defense ae_fft_topk --def_topk 50 \
    --detector_ckpt ./checkpoint/detector_ae.pth \
    --detector_threshold 0.004468 \
    --cmp_defense True \
    --ckpt_path ./2016.10a_AWN.pkl

# 5. Results will show accuracy before and after defense
```

## Expected Results

Typical results on RML2016.10a (averaged over SNRs):

- **Clean Accuracy**: ~60-65%
- **CW Attack Accuracy**: ~10-20% (strong attack)
- **FFT Top-K Recovery**: ~40-50% (significant improvement)
- **Detector-Gated Recovery**: ~45-55% (selective filtering reduces distortion)

## Troubleshooting

### torchattacks not found
```bash
pip install torchattacks
```

### Detector checkpoint missing
Run `--mode train_detector` first to create the checkpoint.

### Out of memory
- Reduce `--eval_limit` or batch size in code
- Use smaller `--cw_steps`
- Switch to CPU with `--device cpu`

### Low recovery accuracy
- Try different `--def_topk` values (30-100 range)
- Recalibrate detector threshold
- Check if clean accuracy is reasonable first

## References

- **torchattacks**: https://github.com/Harry24k/adversarial-attacks-pytorch
- **CW Paper**: "Towards Evaluating the Robustness of Neural Networks" (Carlini & Wagner, 2017)
- **AWN Paper**: "Towards the Automatic Modulation Classification with Adaptive Wavelet Network" (IEEE TCCN 2023)
