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

## Synthetic Data Generation & Finetuning

### Overview

The pipeline generates synthetic IQ bursts that mimic RML2016.10a characteristics, then finetunes the pretrained AWN model on a mix of synthetic + real data. This improves robustness to channel impairments and adversarial attacks while preventing catastrophic forgetting of analog modulations (WBFM, AM-DSB, AM-SSB) which cannot be synthesized.

**Key files:**
- `synth_finetune.py` — Main finetuning script (data gen + training + eval)
- `util/synth_txrx.py` — Burst generator, channel models, modulation/demodulation

### Step 1: Synthetic Burst Generation

`make_rml_like_burst()` in `util/synth_txrx.py` generates IQ bursts with configurable channel impairments controlled by `RmlChanCfg` presets:

| Preset | Phase | Gain Jitter | CFO | Multipath | Use Case |
|--------|-------|-------------|-----|-----------|----------|
| `clean` | No | No | No | No | Baseline (no impairments) |
| `phase_gain` | Yes | Yes | No | No | Curriculum stage 1 |
| `phase_gain_cfo` | Yes | Yes | Yes (std=0.015) | No | Curriculum stage 2 |
| `full` | Yes | Yes | Yes (std=0.015) | Yes (3 taps, decay=1.5) | Single-stage finetune |
| `rml_like` | Yes | Yes | Yes (std=0.007) | Yes (2 taps, decay=0.5) | Matches RML2016 parameters |

**Channel impairment details:**
- **Random phase**: Uniform initial carrier phase U(0, 2π)
- **Gain jitter**: Log-normal RMS rescaling per burst (CV varies by mod: BPSK=0.09, QAM64=0.06)
- **CFO**: Carrier frequency offset as phase ramp, `cfo ~ N(0, cfo_std²)`
- **Multipath**: Rayleigh-faded FIR taps with exponential power-delay profile

**Supported modulations (8 digital):**
- Constellation: BPSK, QPSK, 8PSK, QAM16, QAM64, PAM4
- FSK: CPFSK, GFSK

**Burst structure:** 16 symbols × 8 sps = 128 samples (matches RML2016 signal length), with 2 pilot symbols, RRC pulse shaping (β=0.35), target RMS=0.006.

```bash
# Generate synthetic dataset only (no training)
python synth_finetune.py --mode gen --n_per_cell 2000 --channel_preset full

# Custom SNR range and modulations
python synth_finetune.py --mode gen --n_per_cell 1000 \
  --snr_list "-4,-2,0,2,4,6,8,10,12,14,16,18" \
  --mod_list "BPSK,QPSK,8PSK,QAM16,QAM64,PAM4"
```

### Step 2: Finetuning

Two strategies available:

#### Curriculum Finetuning (Recommended)

Three progressive stages with decaying learning rate. Each stage mixes synthetic data (current preset) with full real RML2016 train split to prevent catastrophic forgetting.

```
Stage 1: phase_gain      (LR = 1e-4)    — learn phase/gain invariance
Stage 2: phase_gain_cfo  (LR = 5e-5)    — add CFO robustness
Stage 3: rml_like        (LR = 2.5e-5)  — add mild multipath
```

```bash
# Curriculum finetune (recommended)
python synth_finetune.py --mode finetune --curriculum \
  --n_per_cell 2000 --ft_epochs 50 --ft_lr 1e-4 --ft_patience 8

# Resume from existing finetuned checkpoint
python synth_finetune.py --mode finetune --curriculum --resume \
  --n_per_cell 2000 --ft_epochs 50
```

#### Single-Stage Finetuning

Uses `full` preset synthetic + real RML data in one training run.

```bash
python synth_finetune.py --mode finetune \
  --channel_preset full --n_per_cell 2000 --ft_epochs 50
```

**Training details:**
- **Optimizer:** Adam (LR=1e-4 default)
- **Loss:** CrossEntropyLoss + AWN internal regularization
- **Scheduler:** ReduceLROnPlateau (factor=0.5, patience=4)
- **Early stopping:** patience=8 epochs
- **Data mix:** `N_synth_digital + N_real_all_11_mods` (real data anchors analog classes)
- **Train/val split:** 85/15
- **Batch size:** 256
- **Output:** `checkpoint/2016.10a_AWN_ft.pkl`

### Step 3: Evaluation

```bash
# Evaluate finetuned model on real RML2016 test set
python synth_finetune.py --mode eval --ckpt_path ./checkpoint/2016.10a_AWN_ft.pkl

# Evaluate on real + synthetic test sets
python synth_finetune.py --mode eval --ckpt_path ./checkpoint/2016.10a_AWN_ft.pkl \
  --synth_path ./data/synth_2016.10a.pkl
```

**What to check:**
- Real RML test accuracy should stay within ~1-2% of base model (no forgetting)
- Synthetic test accuracy should be significantly higher than base model
- Per-SNR breakdown shows improvements primarily at SNR ≥ 0 dB

### Key Parameters Reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--n_per_cell` | 2000 | Synthetic samples per (mod, snr) cell |
| `--channel_preset` | `full` | Channel impairment preset |
| `--curriculum` | off | Enable 3-stage curriculum training |
| `--ft_epochs` | 50 | Max epochs per stage |
| `--ft_lr` | 1e-4 | Initial learning rate |
| `--ft_patience` | 8 | Early stopping patience |
| `--batch_size` | 256 | Training batch size |
| `--target_rms` | 0.006 | Burst RMS normalization target |
| `--resume` | off | Resume from `2016.10a_AWN_ft.pkl` |
| `--seed` | 42 | Random seed |

### Typical Workflow

```bash
# 1. Activate environment
source venv/bin/activate

# 2. Curriculum finetune (generates data + trains + evaluates)
python synth_finetune.py --mode finetune --curriculum --n_per_cell 2000

# 3. Verify finetuned model on adversarial attacks
python crc_defense_fec_multi_attack.py --use_ft \
  --attacks "deepfool,eadl1,cw,fgsm,pgd" \
  --snr "0,18" --mods "BPSK,QPSK,8PSK,QAM16,QAM64,PAM4"

# 4. Compare base vs finetuned on SigGuard evaluation
python main.py --mode sigguard_eval --dataset 2016.10a --ckpt_path ./checkpoint \
  --ta_box minmax --attack_eps 0.1
```

### Making Results Similar to RML2016a

The `rml_like` preset is calibrated to match RML2016.10a signal statistics:
- **CFO std=0.007**: Matches RML2016 uniform[-0.1, 0.1] × symbol_rate at sps=8
- **Multipath 2 taps, decay=0.5**: Mild frequency-selective fading (RML uses simple channels)
- **Gain jitter CV by mod**: Calibrated from RML2016 inter-burst RMS variance
- **Random phase**: Always present (RML has arbitrary carrier phase)

If synthetic signals still differ from RML, tune these knobs:
1. **Increase `n_per_cell`** (more diversity per cell)
2. **Adjust `cfo_std`** (higher = more frequency offset spread)
3. **Adjust `mp_decay`** (lower = stronger multipath effect)
4. **Use curriculum** (progressive exposure prevents overfitting to one channel condition)

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

<!-- GSD:project-start source:PROJECT.md -->
## Project

**Real-Time Defense Pipeline for Adversarial Attacks on AMC**

A unified real-time defense framework for automatic modulation classification (AMC) that combines adversarial detection, frequency-domain recovery (FFT Top-K), and robust classification into a single pipeline. Targets IEEE TCCN/TWC submission using RML2016.10a data, comparing against classical signal processing baselines (Kalman, Wiener, Savitzky-Golay, Gaussian, FIR filters) and randomized smoothing.

**Core Value:** Demonstrate that a unified detect→recover→classify pipeline outperforms individual classical filtering defenses against optimization-based adversarial attacks (CW, EAD) on RF signals, while maintaining real-time feasibility.

### Constraints

- **Data**: RML2016.10a only (11 classes, SNR range -20 to +18 dB)
- **Timeline**: ~1 month to submission-ready paper
- **Compute**: Single GPU (existing setup)
- **Format**: IEEE transaction paper format (double-column LaTeX)
- **Attacks**: Must cover CW (L2), EAD (L1, EN), FGSM, PGD at minimum
- **Epsilon**: Must use RF-appropriate epsilon values (not image defaults)
<!-- GSD:project-end -->

<!-- GSD:stack-start source:codebase/STACK.md -->
## Technology Stack

## Languages
- Python 3.6+ - All core source code, scripts, and utilities
## Runtime
- CPython 3.6+ (tested on 3.8+)
- CUDA-capable GPU recommended (auto-detected, falls back to CPU)
- pip
- Lockfile: requirements.txt present
## Frameworks
- PyTorch 1.7+ (tested on 1.8.1) - Neural network training, inference, tensor operations
- torchattacks - Adversarial attack library (FGSM, PGD, CW, DeepFool, and 12+ more)
- No formal unit test framework detected
- Validation via manual runs: `--mode train`, `--mode eval`, `--mode adv_eval`
- argparse - Command-line argument parsing
- tqdm - Progress bar visualization
## Key Dependencies
- numpy - Numerical computing, signal processing, array operations
- torch - Deep learning framework (model, optim, data)
- torchvision - Pretrained models (if extended), transforms
- torchaudio - Audio processing utilities
- torchattacks - Adversarial attack library (17 attack methods available)
- scikit-learn 0.24+ - Metrics (accuracy_score, confusion_matrix, f1_score, cohen_kappa_score)
- h5py - HDF5 dataset I/O for RML2018.01a format
- pyyaml - YAML configuration parsing
- scipy - Scientific computing (signal processing in defense/attack modules)
- matplotlib - Plotting and visualization for analysis scripts
- pandas - Data analysis and CSV/table output
## Configuration
- Runtime device selection: Auto-detects CUDA, falls back to CPU
- Environment variable used: `PYHONHASHSEED` set in `util/utils.py:fix_seed()` for reproducibility
- Model checkpoint path: Configurable via `--ckpt_path` (default: `./checkpoint`)
- No build system (pure Python)
- Entry point: `main.py` with mode-based dispatching (train, eval, adv_eval, visualize, etc.)
- Dataset configs: YAML files in `config/` directory
## Platform Requirements
- Python 3.6+ with pip
- PyTorch installation (CPU or GPU)
- Disk space for dataset files (RML2016: ~500MB, RML2018: ~3GB)
- Recommended: NVIDIA GPU with CUDA support for faster training
- Same as development (pure Python + PyTorch)
- Model checkpoint files: `<dataset>_AWN.pkl` (500MB-1.5GB per model)
- Evaluation outputs: `inference/<dataset>_*/result/` directory structure
## Data Storage
- Pickle (`.pkl`) - RML2016.10a and RML2016.10b signal data
- HDF5 (`.hdf5`) - RML2018.01a signal data
- PyTorch `.pkl` format (state_dict)
- Path: `./checkpoint/<dataset>_AWN.pkl` (main model)
- Detector: `./checkpoint/detector_ae.pth` (autoencoder for adversarial detection)
- Loading: `torch.load(ckpt_path, map_location=device, weights_only=True)`
- Directory: `training/<dataset>_<index>/log/log.txt`
- CSV output: `training/<dataset>_<index>/log/` (epoch stats)
- Results: `training/<dataset>_<index>/result/` and `inference/<dataset>_*/result/`
## Utility Libraries
- scipy.signal - Filtering, FFT operations
- numpy.fft - Fast Fourier Transform for frequency-domain analysis and defense
- pandas - DataFrame operations for metric aggregation and CSV export
- sklearn.metrics - Confusion matrix, F1, Kappa, accuracy computation
- pickle - Python object serialization (dataset loading)
- h5py - HDF5 file I/O (RML2018 dataset)
## Code Structure
- `main.py` - Primary command dispatcher for all modes (train, eval, adv_eval, visualize, etc.)
- `synth_finetune.py` - Standalone script for synthetic data generation and finetuning
- Multiple plot/test scripts: `plot_*.py`, `test_*.py`, `*_experiment.py`
- `models/` - Neural network architectures
- `util/` - Utilities and algorithms
- `data_loader/` - Dataset loading and preprocessing
<!-- GSD:stack-end -->

<!-- GSD:conventions-start source:CONVENTIONS.md -->
## Conventions

## Naming Patterns
- Snake_case with underscores: `data_loader.py`, `synth_txrx.py`, `adv_attack.py`
- Main entry point: `main.py`
- Experiment/standalone scripts: descriptive names in snake_case (e.g., `test_2016_10a.py`, `plot_iq_distribution.py`, `crc_experiment.py`)
- **Core library functions:** PascalCase (preserve for backward compatibility)
- **New utility functions:** snake_case preferred (e.g., `rrc_filter()`, `recover_constellation()`, `fix_seed()` in `util/utils.py`)
- **Helper/internal functions:** snake_case with leading underscore if private (e.g., `_lowpass_filter()`, `_batch_clip()` in `util/adv_attack.py`)
- All lowercase with underscores: `signal_len`, `num_classes`, `batch_size`, `test_idx`
- Loop counters: Single letters acceptable (e.g., `i`, `snr_i`)
- Tensor batch variables: Descriptive snake_case (e.g., `sig_batch`, `lab_batch`, `adv_norm`)
- Constants: UPPER_CASE (e.g., `NUM_SAMPLES`, `SNR_THRESHOLD`, `DEVICE`)
- All PascalCase: `AWN`, `Config`, `Trainer`, `EarlyStopping`, `LiftingScheme`, `AverageMeter`
- No suffix for module classes (use `AWN` not `AWNModel`)
- PascalCase in YAML files (config/*.yml) → snake_case when accessed as Python object attributes
## Code Style
- No automated formatter configured (no .eslintrc, .prettierrc, biome.json detected)
- Follow Python style informally; consistency within files takes priority
- 4-space indentation used throughout
- Line length: Aim for ≤100 characters where practical (seen in docstrings and comments)
- No linting rules enforced (no .eslintrc* files found)
- Code relies on developer discipline; bugs discovered through runtime validation
- No path aliases configured (no jsconfig/tsconfig-style aliases)
- Relative imports from project root: `from models.model import AWN`, `from util.training import Trainer`
## Error Handling
- Use `try/except ImportError` for optional dependencies that will be selectively used
- Use `try/except Exception` to skip non-critical features (plotting, visualization) and log the skip
- Use `NotImplementedError` for unsupported dataset types or config options
- Use `ValueError` with descriptive messages for invalid arguments (e.g., unknown model name)
- Prefer explicit error messages with context (dataset name, config key, etc.)
## Logging
- Config/setup info: `log_exp_settings(logger, cfg)` in `util/utils.py` logs all config attributes once at startup
- Per-epoch metrics: Loss, accuracy, learning rate, elapsed time
- Important state changes: Model checkpoint saved, early stopping triggered, validation improved
- Warnings: Skipped features, invalid filters applied, dataset splits
- Individual batch processing (use tqdm progress bar instead)
- Per-sample predictions (would flood log; compute aggregate metrics instead)
## Comments
- **Module-level docstrings:** Explain overall purpose and usage (especially test/experiment scripts)
- **Class docstrings:** What the class does, key methods
- **Complex algorithms:** Inline comments for mathematical operations (e.g., M-th power method in `recover_constellation()`)
- **Non-obvious design choices:** Why a particular approach was selected
- **Attribution:** Link to external sources (e.g., "adopted from pytorchtools: https://...") in `util/early_stop.py`
- Not used (Python project)
- Docstrings follow NumPy style informally where present:
## Function Design
- Typical range: 10-50 lines
- Complex algorithms (signal processing): 50-100 lines acceptable (e.g., `Dataset_Split()` in `data_loader/data_loader.py` is 80 lines with per-SNR stratification logic)
- Avoid exceeding 150 lines; split into smaller functions if logic becomes hard to follow
- Order: Data → Configuration → Optional settings
- Use keyword arguments for optional parameters (lambda functions use positional for brevity)
- Default values preferred: `kernel_size=17`, `beta=0.35`, `mod_order=4`
- Single return: scalar or array
- Multiple returns: tuple (unpacking at call site)
## Module Design
- Public functions and classes: No `__all__` lists; anything not prefixed with `_` is public
- Core library modules (`data_loader`, `util`, `models`): Export factory functions and main classes
- Not used (no index.ts style)
- Each module imported explicitly: `from util.training import Trainer`
- Used for heavy/optional dependencies:
## Special Patterns
- Singleton Config per run: `cfg = Config(dataset, train=True/False)`
- Args merged into Config: `cfg = merge_args2cfg(cfg, vars(args))`
- Access via dot notation: `cfg.dataset`, `cfg.epochs`, `cfg.device`
- Initialize directories after args merged: `cfg.init_dir()`
- Used sparingly for cached state or module-level setup
- Example: `_MCLDNN_PyTorch` in `util/utils.py` (lazy import cache)
- Example: `global regu_d, regu_c` in `models/model.py` (in conditional block; not ideal but preserved)
- Dataset class mappings use byte-string keys: `{b'BPSK': 4, b'QPSK': 9}`
- Necessary because pickle dataset files store modulation names as bytes
- Converted to int indices when building labels: `Labels = [classes[i] for i in Labels]`
<!-- GSD:conventions-end -->

<!-- GSD:architecture-start source:ARCHITECTURE.md -->
## Architecture

## Pattern Overview
- Modular neural network layers for signal processing (convolution → wavelet decomposition → attention → classification)
- Attack-defense evaluation pipeline with multiple threat models
- Pluggable defense mechanisms in frequency and time domains
- Dataset-agnostic modulation classification supporting RML2016 and RML2018 datasets
- Synthetic data generation for robustness training
## Layers
- Purpose: Load and normalize RF IQ signal tensors
- Location: `data_loader/data_loader.py:Load_Dataset()`
- Contains: Dataset loaders for RML2016.10a/b and RML2018.01a (pickle/HDF5 formats)
- Depends on: Config for class mappings and signal length parameters
- Used by: Training and evaluation pipelines
- Purpose: Integrate I/Q channels and extract temporal features
- Location: `models/model.py:AWN.conv1`, `AWN.conv2`
- Contains: 2D conv to merge I/Q channels, 1D temporal convolution with batch norm
- Pattern: Zero-padded 2D conv `[2, 7]` kernel → squeeze to 1D → 1D conv with LeakyReLU
- Output: `[batch, 64, time_len]` feature maps
- Purpose: Multi-level learnable lifting scheme to capture multi-scale signal structure
- Location: `models/lifting.py:LiftingScheme`, `models/model.py:LevelTWaveNet`
- Contains: Signal splitting into even/odd samples, learnable Predictor (P) and Updator (U) operators
- Pattern: Forward lifting scheme (split → update → predict → split again for next level)
- Applied recursively: `num_levels` decompositions produce approximation and detail coefficients
- Output: Multi-scale coefficient tensors fed into attention layer
- Purpose: Weight and combine multi-scale features
- Location: `models/model.py:AWN.SE_attention_score`, `AWN.avgpool`
- Contains: Squeeze-excitation attention (linear → ReLU → sigmoid) and adaptive pooling
- Pattern: Average pool detail/approximation coefficients per level → concatenate → pass through SE attention
- Output: Attention-weighted feature vector ready for classification
- Purpose: Map weighted features to modulation class logits
- Location: `models/model.py:AWN.fc`
- Contains: Two fully-connected layers with LeakyReLU, dropout on SE attention
- Returns: `logit` (class logits) and `regu_sum` (list of regularization terms per level)
- Purpose: Optimize model on clean and noisy signals
- Location: `util/training.py:Trainer`
- Pattern: Epoch-based training with per-batch forward pass, CrossEntropyLoss + regularization sum, Adam optimizer
- Includes: Early stopping with learning rate decay on plateau, per-SNR stratified train/val/test split
- Output: Model checkpoint saved to `<cfg.model_dir>/<dataset>_<model>.pkl`
- Purpose: Generate adversarial examples for robustness evaluation
- Location: `util/adv_attack.py`
- Supports: CW L2 attack (internal implementation), torchattacks library attacks (FGSM, PGD, CW, APGD, DeepFool, EAD, etc.)
- Normalization modes:
- Features: Optional low-pass smoothing post-perturbation, spectral noise attacks (CW tone, band-limited noise)
- Purpose: Recover from adversarial perturbations in frequency domain
- Location: `util/defense.py`
- Types:
- Pattern: Normalize → rFFT → apply mask/filter → iRFFT → denormalize
- Purpose: Identify adversarial samples without retraining main model
- Location: `util/detector.py:RFSignalAutoEncoder`, `util/detector_train.py`
- Architecture: 1D conv encoder (→ 128 latent dims) with channel attention + decoder with skip connections
- Gating: Compute KL divergence between input and reconstruction; threshold determines if denoising is applied
- Calibration: Quantile-based threshold selection on clean validation set (default 90th percentile)
- Purpose: Compute per-SNR accuracy, confusion matrices, and robustness metrics
- Location: `util/evaluation.py:Run_Eval()`, `util/adv_eval.py:Run_Adv_Eval()`
- Metrics: Per-SNR accuracy, macro F1, Cohen's Kappa, confusion matrix heatmaps
- Outputs: Plots to `<cfg.result_dir>/` (confusion matrix, SNR-accuracy curves)
## Data Flow
- Config object (`cfg`) holds dataset-specific parameters: class mappings, signal length, network hyperparameters
- Model state dict saved/loaded via `torch.save/load()` to `checkpoint/<dataset>_<model>.pkl`
- Detector state saved separately to `checkpoint/detector_ae.pth`
- Per-run output directories auto-increment: `training/<dataset>_0/`, `inference/<dataset>_1/`, etc.
## Key Abstractions
- Representation: `[batch, 2, time_len]` where dimension 1 is I/Q channels
- Normalization: IQ samples typically in [-0.02, 0.02] range; models expect [-1, 1] or [0, 1] depending on mode
- Dataset variants: RML2016 has `time_len=128`, RML2018 has `time_len=1024`
- Purpose: Adapt AWN (returns logits + regularization) for torchattacks (expects 4D image-style inputs)
- Implementation: `util/adv_attack.py:Model01Wrapper`
- Maps torchattacks [0,1] inputs back to [-1,1] IQ, forwards base model, returns logits only
- Supports dual normalization: unit (linear) and minmax (per-sample)
- Purpose: Centralized hyperparameter and path management
- Location: `util/config.py:Config`
- Init: Loads YAML from `config/<dataset>.yml` (epochs, lr, regularization weights, network depth)
- Dir management: Auto-creates `training/` or `inference/` subdirs with auto-incrementing indices
- Class mapping: Stores byte-keyed modulation↔int label dictionaries per dataset
- Abstraction: All defenses follow same interface: `def defense_fn(x: Tensor, **params) -> Tensor`
- Allows composability: can chain defenses or A/B test alternatives
- Normalization-aware: normalize/denormalize calls managed separately from FFT operation
## Entry Points
- Location: `main.py:__main__`, args.mode == 'train'
- Triggers: Command-line argument `--mode train`
- Responsibilities:
- Location: `main.py:__main__`, args.mode == 'eval'
- Triggers: `--mode eval --ckpt_path <checkpoint>`
- Responsibilities:
- Location: `main.py:__main__`, args.mode == 'adv_eval'
- Triggers: `--mode adv_eval --attack <type> --defense <type>`
- Responsibilities:
- Location: `main.py:__main__`, args.mode == 'multi_attack_eval'
- Triggers: `--mode multi_attack_eval --attack_list "fgsm,pgd,cw,..." --plot_freq`
- Responsibilities:
- Location: `main.py:__main__`, args.mode == 'sigguard_eval'
- Triggers: `--mode sigguard_eval --ckpt_path <checkpoint>`
- Responsibilities:
- Location: `main.py:__main__`, args.mode == 'train_detector'
- Triggers: `--mode train_detector --det_epochs 10 --det_batch_size 256`
- Responsibilities:
- Location: `main.py:__main__`, args.mode == 'calibrate_detector'
- Triggers: `--mode calibrate_detector --detector_ckpt ./checkpoint/detector_ae.pth`
- Responsibilities:
## Error Handling
- `torch.cuda.is_available()`: Auto-detect GPU, fallback to CPU
- Dataset not found: Raise `NotImplementedError` with path hint
- Config YAML missing: Raise `NotImplementedError` with expected location
- Visualization imports optional: Try-except on `util.visualize`, log skip message
- Defense mismatch: Check defense type against available functions, raise on unknown
- Attack backend selection: Try `torchattacks` first, fallback to internal implementation if requested
- Detector missing: Gracefully skip detector gating if `--detector_ckpt` not provided
## Cross-Cutting Concerns
- Framework: Python `logging` module via `util/logger.py:create_logger()`
- Pattern: All major functions log progress, warnings, and results to both file and console
- Location: `<cfg.log_dir>/log.txt` and stdout
- Tensor shape assertions in most functions (e.g., `assert x.dim() == 3 and x.size(1) == 2`)
- SNR/modulation filtering allows subset evaluation for fast iteration
- Signal normalization checks in defense and attack modules
- Not applicable (academic research tool)
- Seed control: `util/utils.py:fix_seed()` sets numpy, torch, and random seeds
- Config-driven: All hyperparameters in YAML files
- Deterministic data split: Stratified by (modulation, SNR) with fixed random seed
- Checkpoint versioning: Auto-increment directories prevent overwrites
<!-- GSD:architecture-end -->

<!-- GSD:workflow-start source:GSD defaults -->
## GSD Workflow Enforcement

Before using Edit, Write, or other file-changing tools, start work through a GSD command so planning artifacts and execution context stay in sync.

Use these entry points:
- `/gsd:quick` for small fixes, doc updates, and ad-hoc tasks
- `/gsd:debug` for investigation and bug fixing
- `/gsd:execute-phase` for planned phase work

Do not make direct repo edits outside a GSD workflow unless the user explicitly asks to bypass it.
<!-- GSD:workflow-end -->

<!-- GSD:profile-start -->
## Developer Profile

> Profile not yet configured. Run `/gsd:profile-user` to generate your developer profile.
> This section is managed by `generate-claude-profile` -- do not edit manually.
<!-- GSD:profile-end -->
