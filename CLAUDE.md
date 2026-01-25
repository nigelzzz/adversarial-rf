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

### Multi-Attack Evaluation with FFT Recovery

Evaluates multiple attacks and compares attack accuracy vs FFT Top-K recovery accuracy, broken down by modulation and SNR.

```bash
# Full evaluation with all 15 attacks
python main.py --mode multi_attack_eval --dataset 2016.10a --ckpt_path ./checkpoint

# Subset of attacks
python main.py --mode multi_attack_eval --dataset 2016.10a --ckpt_path ./checkpoint \
  --attack_list "fgsm,pgd,cw,deepfool"

# Filter by modulation and SNR
python main.py --mode multi_attack_eval --dataset 2016.10a --ckpt_path ./checkpoint \
  --mod_filter QAM64 --snr_filter 18 --attack_list fgsm

# Speed up with sample limit per (SNR, mod) cell
python main.py --mode multi_attack_eval --dataset 2016.10a --ckpt_path ./checkpoint \
  --eval_limit_per_cell 50

# With frequency domain comparison plots
python main.py --mode multi_attack_eval --dataset 2016.10a --ckpt_path ./checkpoint \
  --mod_filter QAM64 --snr_filter 18 --attack_list fgsm --plot_freq --plot_n_samples 5
```

**Available attacks** (17 total):
`fgsm`, `pgd`, `bim`, `cw`, `deepfool`, `apgd`, `mifgsm`, `rfgsm`, `upgd`, `eotpgd`, `vmifgsm`, `vnifgsm`, `jitter`, `ffgsm`, `pgdl2`, `eadl1`, `eaden`

**Key parameters**:
- `--attack_list <attacks>`: Comma-separated attack names (default: all 17)
- `--attack_eps <float>`: Epsilon for Linf attacks (default: 0.03 for IQ data)
- `--ta_box <unit|minmax>`: Normalization mode for torchattacks (default: unit)
- `--eval_limit_per_cell <int>`: Max samples per (SNR, mod) cell
- `--plot_freq`: Generate frequency domain comparison plots
- `--plot_n_samples <int>`: Number of individual samples to plot (default: 3)

**Output**:
- CSV: `inference/<dataset>_*/result/multi_attack_snr_mod_eval.csv`
  - Columns: `attack, snr, modulation, n_samples, attack_acc, top10_acc, top20_acc`
- Plots (if `--plot_freq`): `inference/<dataset>_*/result/freq_plots/`
  - `<attack>_<mod>_snr<snr>_sample<n>.png`: Individual sample spectra
  - `<attack>_<mod>_snr<snr>_avg.png`: Average spectra across samples
  - `<attack>_<mod>_snr<snr>_overlay.png`: Clean vs adversarial overlay

### SigGuard-Style Evaluation

Produces a table comparing attack accuracy with/without FFT Top-K defense, similar to academic paper format. Also generates IQ distribution plots comparing clean vs adversarial signals by default.

```bash
# Default: All 15 attacks with IQ plots
python main.py --mode sigguard_eval --dataset 2016.10a --ckpt_path ./checkpoint

# With minmax normalization for better attack effectiveness
python main.py --mode sigguard_eval --dataset 2016.10a --ckpt_path ./checkpoint \
  --ta_box minmax --attack_eps 0.1

# Custom attack list
python main.py --mode sigguard_eval --dataset 2016.10a --ckpt_path ./checkpoint \
  --attack_list "cw,fgsm,pgd,deepfool"

# Faster evaluation with sample limit
python main.py --mode sigguard_eval --dataset 2016.10a --ckpt_path ./checkpoint \
  --sigguard_topk 30 --eval_limit 1000

# Disable IQ plots for faster runs
python main.py --mode sigguard_eval --dataset 2016.10a --ckpt_path ./checkpoint \
  --no_plot_iq
```

**Available attacks** (17 total, all run by default):
`fgsm`, `pgd`, `bim`, `cw`, `deepfool`, `apgd`, `mifgsm`, `rfgsm`, `upgd`, `eotpgd`, `vmifgsm`, `vnifgsm`, `jitter`, `ffgsm`, `pgdl2`, `eadl1`, `eaden`

**Output format:**
```
  AWN - SigGuard Evaluation (Top-50)
  ==================================================
  Sample Type         Disabled      Enabled
  --------------------------------------------------
  Intact              92.61%        92.20%
  FGSM                 7.20%         9.32%
  PGD                  5.10%        12.45%
  CW                   0.86%        80.43%
  EADL1                0.00%        78.34%
  EADEN                0.00%        74.01%
  ...                   ...           ...
  ==================================================
```

**Key parameters:**
- `--attack_list <attacks>`: Comma-separated attacks (default: all 17)
- `--sigguard_topk <int>`: Top-K value for FFT defense (default: 50)
- `--ta_box <unit|minmax>`: Normalization mode (default: unit)
- `--eval_limit <int>`: Limit test samples for faster evaluation
- `--no_plot_iq`: Disable IQ distribution plots
- `--plot_n_samples <int>`: Number of individual samples to plot (default: 3)

**Output files:**
- `inference/<dataset>_*/result/sigguard_eval.csv`: Raw CSV data
- `inference/<dataset>_*/result/sigguard_eval_table.txt`: Formatted table
- `inference/<dataset>_*/result/iq_plots/`: IQ distribution plots
  - `<attack>_iq_sample1.png`, ...: Individual sample scatter plots
  - `<attack>_iq_all.png`: Aggregated scatter plot
  - `<attack>_iq_density.png`: 2D histogram density comparison

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

### Epsilon Configuration for RF IQ Data

RF IQ signals require different epsilon values than images. Key differences:

**The Problem with Default Image Epsilon:**
- torchattacks is designed for images in [0, 1] range
- IQ signals are in [-1, 1] but have typical amplitude ~±0.02 (very small)
- After unit conversion `(x+1)/2`: values are ~[0.49, 0.51] (only 2% of range)
- Using `eps=0.3` (old default, common for images) is 15x larger than signal amplitude
- Result: perturbation overwhelms signal → effectively random noise → no accuracy drop

**Normalization Modes (`--ta_box`):**

| Mode | Mapping | Epsilon Interpretation | Best For |
|------|---------|----------------------|----------|
| `unit` | `(x+1)/2` | Absolute in [0,1] space | Simple, needs small eps (~0.03) |
| `minmax` | Per-sample min-max to [0,1] | Relative to signal range | More intuitive eps values |

**Recommended Epsilon Values:**

| Mode | Epsilon | Effect |
|------|---------|--------|
| `unit` | 0.01-0.03 | Subtle perturbation |
| `unit` | 0.05-0.1 | Moderate attack |
| `minmax` | 0.05-0.1 | Subtle perturbation |
| `minmax` | 0.2-0.3 | Moderate attack |

**Example Commands:**

```bash
# Recommended: minmax mode with moderate epsilon
python main.py --mode multi_attack_eval --dataset 2016.10a --ckpt_path ./checkpoint \
  --attack_list fgsm --ta_box minmax --attack_eps 0.1

# Alternative: unit mode with small absolute epsilon
python main.py --mode multi_attack_eval --dataset 2016.10a --ckpt_path ./checkpoint \
  --attack_list fgsm --attack_eps 0.03
```

**Verification:** A working attack should show `attack_acc` significantly lower than clean accuracy (e.g., 60-90% → 20-40%).

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
