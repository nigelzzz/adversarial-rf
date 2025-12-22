# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AWN (Adaptive Wavelet Network) is a PyTorch implementation for automatic modulation classification (AMC) of radio frequency signals using an adaptive lifting scheme-based wavelet decomposition. The model classifies modulation formats (BPSK, QPSK, QAM, PSK, etc.) from I/Q signal samples at various SNR levels.

**Paper**: "Towards the Automatic Modulation Classification with Adaptive Wavelet Network" (IEEE TCCN 2023)

## Core Architecture

### Signal Processing Pipeline

1. **Input**: I/Q signal tensors `[batch, 2, time_len]` where time_len is 128 (RML2016) or 1024 (RML2018)
2. **Feature Extraction** (`models/model.py:AWN`):
   - `conv1`: 2D conv integrates I/Q channels → `[batch, 64, 1, time_len]`
   - `conv2`: 1D conv for temporal features
3. **Adaptive Wavelet Decomposition** (`models/lifting.py:LiftingScheme`):
   - Splits signals into even/odd samples
   - Uses learnable Predictor (P) and Updator (U) operators
   - Produces approximation (c) and detail (d) coefficients
   - Applied recursively for `num_levels` decompositions
4. **Attention & Classification**:
   - Squeeze-Excitation attention scores multi-scale features
   - FC layers map to `num_classes` modulation types
5. **Loss**: CrossEntropyLoss + regularization on details (`regu_details`) and approximation mean (`regu_approx`)

### Key Components

- **`models/model.py:AWN`**: Main network combining conv layers, wavelet decomposition levels, SE attention, and classifier
- **`models/lifting.py:LiftingScheme`**: Implements the adaptive lifting scheme with learnable P/U operators
- **`data_loader/data_loader.py`**: Loads RML datasets (pickle/hdf5), splits into train/val/test with stratification across SNR×modulation combinations
- **`util/config.py:Config`**: YAML-based config loader that creates run directories (`training/<dataset>_*/` or `inference/<dataset>_*/`)
- **`util/training.py:Trainer`**: Training loop with early stopping, learning rate decay, epoch-level train/val logic
- **`util/evaluation.py:Run_Eval`**: Computes per-SNR accuracy, confusion matrix, macro F1, Kappa coefficient

## Dataset Configuration

Each dataset has a YAML config in `config/<dataset>.yml`:
- `2016.10a`: 11 classes, 220K samples (2×128)
- `2016.10b`: 10 classes, 1.2M samples (2×128)
- `2018.01a`: 24 classes, 2.5M samples (2×1024)

**Class mappings** are defined in both `util/config.py:Config.__init__` and `data_loader/data_loader.py:Load_Dataset`. When adding datasets, update both locations.

**Dataset files** must be placed in `./data/`:
- `RML2016.10a_dict.pkl`
- `RML2016.10b.dat`
- `GOLD_XYZ_OSC.0001_1024.hdf5`

## Common Commands

### Training
```bash
python main.py --mode train --dataset 2016.10a
```
Creates `training/2016.10a_*/` with subdirs: `models/`, `log/`, `result/`. Model saved as `<dataset>_AWN.pkl`.

### Evaluation
```bash
python main.py --mode eval --dataset 2016.10a --ckpt_path ./checkpoint
```
Loads pretrained model, computes accuracy/F1/Kappa, saves confusion matrix and SNR-accuracy curves to `inference/<dataset>_*/result/`.

### Visualization
```bash
python main.py --mode visualize --dataset 2016.10a
```
Plots lifting scheme decomposition (approx/details coefficients) as SVG files.

### Adversarial Evaluation

**CW attack** (uses torchattacks library by default):
```bash
# Standard CW attack with torchattacks (default)
python main.py --mode adv_eval --dataset 2016.10a --attack cw --cw_steps 100 --cw_c 1.0

# Use internal CW implementation instead
python main.py --mode adv_eval --dataset 2016.10a --attack cw --attack_backend internal --cw_steps 100

# CW with FFT Top-K recovery (AWN_All.py pattern)
python main.py --mode adv_eval --dataset 2016.10a --attack cw --defense fft_topk --def_topk 50

# CW with detector-gated recovery (recommended)
python main.py --mode adv_eval --dataset 2016.10a --attack cw --defense ae_fft_topk \
  --def_topk 50 --detector_ckpt ./checkpoint/detector_ae.pth --detector_threshold 0.004468
```

**Spectral perturbations** (no optimization):
```bash
# CW tone (single frequency jammer)
python main.py --mode adv_eval --dataset 2016.10a --attack spectral --spec_type cw_tone --spec_eps 0.1

# Band-limited noise
python main.py --mode adv_eval --dataset 2016.10a --attack spectral --spec_type psd_band \
  --spec_band_low 0.05 --spec_band_high 0.25 --spec_eps 0.1
```

**Defenses** (FFT-domain recovery):
```bash
# Hard notch (zero specific band)
python main.py --mode adv_eval --dataset 2016.10a --attack spectral --spec_type psd_band \
  --defense fft_notch --def_band_low 0.05 --def_band_high 0.25 --cmp_defense True

# Soft notch (tapered suppression)
python main.py --mode adv_eval --dataset 2016.10a --attack spectral --spec_type psd_band \
  --defense fft_soft_notch --def_notch_depth 0.7 --def_notch_trans 4 --cmp_defense True

# Top-K FFT (keep K largest bins per channel)
python main.py --mode adv_eval --dataset 2016.10a --attack spectral --defense fft_topk --def_topk 50

# AE detector-gated Top-K (denoise only if KL > threshold)
python main.py --mode adv_eval --dataset 2016.10a --attack spectral --defense ae_fft_topk \
  --def_topk 50 --detector_ckpt ./checkpoint/detector_ae.pth --detector_threshold 0.004468
```

**Detector training/calibration**:
```bash
# Train 1D conv autoencoder on clean signals
python main.py --mode train_detector --dataset 2016.10a --det_epochs 10

# Calibrate threshold on validation set (e.g., 90th percentile KL)
python main.py --mode calibrate_detector --dataset 2016.10a --detector_ckpt ./checkpoint/detector_ae.pth
```

### Other Modes
```bash
# Compare spectral profiles
python main.py --mode freq_compare --dataset 2016.10a --spec_type cw_tone --spec_eps 0.1

# Build average PSD mask from filtered subset
python main.py --mode build_psd_mask --dataset 2016.10a --mod_filter QAM16 --snr_filter 18

# Run attack benchmark
python main.py --mode adv_bench --dataset 2016.10a
```

## Attack and Defense Pipeline

### CW Attack with Recovery (AWN_All.py Pattern)

The codebase now uses the **torchattacks** library (https://github.com/Harry24k/adversarial-attacks-pytorch) by default for CW attacks. The recovery mechanism follows AWN_All.py:

1. **CW Attack**: Generate adversarial examples using `torchattacks.CW`
2. **Detection** (optional): Use autoencoder to compute KL divergence between input and reconstruction
3. **Recovery**: Apply FFT Top-K filtering to suspected adversarial samples
   - Normalize signals: `(x + 0.02) / 0.04`
   - Keep top-K FFT components per I/Q channel
   - Denormalize: `x * 0.04 - 0.02`

**Defense modes**:
- `fft_topk`: Apply Top-K to all samples (K set by `--def_topk`)
- `ae_fft_topk`: Gate Top-K with detector (only denoise if KL > threshold)

**Key parameters**:
- `--attack_backend torchattacks`: Use torchattacks.CW (default)
- `--attack_backend internal`: Use custom CW implementation
- `--defense ae_fft_topk`: Enable detector-gated recovery
- `--detector_ckpt <path>`: Path to pretrained autoencoder (train with `--mode train_detector`)
- `--detector_threshold <float>`: KL divergence threshold (calibrate with `--mode calibrate_detector`)
- `--def_topk <int>`: Number of FFT components to keep (default 50)

## Development Notes

### Adding New Modes
1. Add argparse flags in `main.py:__main__` (lines 17-87)
2. Implement handler as `elif args.mode == 'new_mode':` (lines 109-285)
3. Update `AGENTS.md` and this file

### Modifying Defense Strategies
- FFT-domain defenses live in `util/defense.py`
- Spectral attacks in `util/adv_attack.py` (spectral noise) and `util/adv_attack.py:cw_l2_attack` (CW)
- Detector architectures in `util/detector.py`, training loop in `util/detector_train.py`

### Config Handling
- Runtime args are merged into Config via `util/config.py:merge_args2cfg`
- Access via `cfg.<field>` (e.g., `cfg.cw_c`, `cfg.defense`)
- New fields should have sensible defaults in argparse

### Output Directories
- **Training**: `training/<dataset>_<index>/models/`, `/log/`, `/result/`
- **Inference**: `inference/<dataset>_<index>/` (same structure)
- Index auto-increments to avoid overwrites

### Filtering Data
Use `--mod_filter <MOD>` and/or `--snr_filter <SNR>` to subset data:
```bash
python main.py --mode eval --dataset 2016.10a --mod_filter QAM16 --snr_filter 18
```
Useful for fast debugging or building PSD masks for specific conditions.

## Code Style

- Python 3.6+, PyTorch 1.7+ (tested on 1.8.1)
- 4-space indentation
- Existing public APIs use PascalCase (e.g., `Create_Data_Loader`, `Run_Eval`); preserve for compatibility
- New code should prefer snake_case
- Line length ≤ 100 characters where practical

## Testing Strategy

No formal unit tests exist. Validate changes via:
1. Run `--mode train` on a small dataset/subset
2. Check `training/<dataset>_*/log/log.txt` for errors
3. Run `--mode eval` and verify metrics match expected values
4. Review plots in `result/` (acc curves, confusion matrix)
5. For adversarial code, use `--eval_limit 1000` to speed up iteration

## Important Implementation Details

- **Model returns**: `(logit, regu_sum)` where `regu_sum` is list of regularization terms per level
- **Dataset split**: Stratified by (modulation, SNR) to ensure balanced train/val/test distributions
- **Early stopping**: Monitors validation loss, triggers LR decay every `milestone_step` patience increments
- **Spectral attacks**: Normalized frequencies in [0, 0.5] (Nyquist); `spec_eps` is L2 norm per sample
- **Defense naming**: `fft_*` defenses apply real FFT, manipulate spectrum, then inverse FFT; `ae_fft_topk` gates Top-K denoising with an autoencoder anomaly detector
