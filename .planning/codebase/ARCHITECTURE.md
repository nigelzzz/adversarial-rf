# Architecture

**Analysis Date:** 2026-03-31

## Pattern Overview

**Overall:** Signal classification with adversarial robustness framework

**Key Characteristics:**
- Modular neural network layers for signal processing (convolution → wavelet decomposition → attention → classification)
- Attack-defense evaluation pipeline with multiple threat models
- Pluggable defense mechanisms in frequency and time domains
- Dataset-agnostic modulation classification supporting RML2016 and RML2018 datasets
- Synthetic data generation for robustness training

## Layers

**Signal Input Layer:**
- Purpose: Load and normalize RF IQ signal tensors
- Location: `data_loader/data_loader.py:Load_Dataset()`
- Contains: Dataset loaders for RML2016.10a/b and RML2018.01a (pickle/HDF5 formats)
- Depends on: Config for class mappings and signal length parameters
- Used by: Training and evaluation pipelines

**Feature Extraction (Convolution):**
- Purpose: Integrate I/Q channels and extract temporal features
- Location: `models/model.py:AWN.conv1`, `AWN.conv2`
- Contains: 2D conv to merge I/Q channels, 1D temporal convolution with batch norm
- Pattern: Zero-padded 2D conv `[2, 7]` kernel → squeeze to 1D → 1D conv with LeakyReLU
- Output: `[batch, 64, time_len]` feature maps

**Adaptive Wavelet Decomposition:**
- Purpose: Multi-level learnable lifting scheme to capture multi-scale signal structure
- Location: `models/lifting.py:LiftingScheme`, `models/model.py:LevelTWaveNet`
- Contains: Signal splitting into even/odd samples, learnable Predictor (P) and Updator (U) operators
- Pattern: Forward lifting scheme (split → update → predict → split again for next level)
- Applied recursively: `num_levels` decompositions produce approximation and detail coefficients
- Output: Multi-scale coefficient tensors fed into attention layer

**Attention & Aggregation:**
- Purpose: Weight and combine multi-scale features
- Location: `models/model.py:AWN.SE_attention_score`, `AWN.avgpool`
- Contains: Squeeze-excitation attention (linear → ReLU → sigmoid) and adaptive pooling
- Pattern: Average pool detail/approximation coefficients per level → concatenate → pass through SE attention
- Output: Attention-weighted feature vector ready for classification

**Classification Layer:**
- Purpose: Map weighted features to modulation class logits
- Location: `models/model.py:AWN.fc`
- Contains: Two fully-connected layers with LeakyReLU, dropout on SE attention
- Returns: `logit` (class logits) and `regu_sum` (list of regularization terms per level)

**Training Loop:**
- Purpose: Optimize model on clean and noisy signals
- Location: `util/training.py:Trainer`
- Pattern: Epoch-based training with per-batch forward pass, CrossEntropyLoss + regularization sum, Adam optimizer
- Includes: Early stopping with learning rate decay on plateau, per-SNR stratified train/val/test split
- Output: Model checkpoint saved to `<cfg.model_dir>/<dataset>_<model>.pkl`

**Adversarial Attack Module:**
- Purpose: Generate adversarial examples for robustness evaluation
- Location: `util/adv_attack.py`
- Supports: CW L2 attack (internal implementation), torchattacks library attacks (FGSM, PGD, CW, APGD, DeepFool, EAD, etc.)
- Normalization modes:
  - `unit`: Linear mapping `(x + 1) / 2` to [0,1] (default, needs small eps ~0.01-0.03)
  - `minmax`: Per-sample min-max normalization (more intuitive, eps ~0.05-0.1)
- Features: Optional low-pass smoothing post-perturbation, spectral noise attacks (CW tone, band-limited noise)

**Defense Module:**
- Purpose: Recover from adversarial perturbations in frequency domain
- Location: `util/defense.py`
- Types:
  - Binary notch: Zero frequency band
  - Soft notch: Tapered spectral suppression with raised-cosine transitions
  - FFT Top-K: Keep K largest magnitude bins per channel
  - FFT Top-K percent: Adaptive K based on energy threshold
  - AE detector-gated: Use autoencoder KL divergence to selectively denoise
- Pattern: Normalize → rFFT → apply mask/filter → iRFFT → denormalize

**Anomaly Detection (Detector):**
- Purpose: Identify adversarial samples without retraining main model
- Location: `util/detector.py:RFSignalAutoEncoder`, `util/detector_train.py`
- Architecture: 1D conv encoder (→ 128 latent dims) with channel attention + decoder with skip connections
- Gating: Compute KL divergence between input and reconstruction; threshold determines if denoising is applied
- Calibration: Quantile-based threshold selection on clean validation set (default 90th percentile)

**Evaluation Module:**
- Purpose: Compute per-SNR accuracy, confusion matrices, and robustness metrics
- Location: `util/evaluation.py:Run_Eval()`, `util/adv_eval.py:Run_Adv_Eval()`
- Metrics: Per-SNR accuracy, macro F1, Cohen's Kappa, confusion matrix heatmaps
- Outputs: Plots to `<cfg.result_dir>/` (confusion matrix, SNR-accuracy curves)

## Data Flow

**Training Flow:**

1. Load dataset (pickle/HDF5) → normalize to float32 tensors `[N, 2, T]`
2. Stratified split by (modulation, SNR) → train 60%, val 20%, test 20%
3. Create data loaders with batch stratification
4. For each epoch:
   - Forward pass: `model(x_batch)` → `(logit, regu_sum)`
   - Loss: `CrossEntropyLoss(logit, label) + sum(regu_sum)`
   - Backward + optimizer step
   - Validation on holdout set, save best checkpoint
5. Early stopping if validation loss plateaus for `patience` epochs

**Adversarial Attack Flow:**

1. Load clean test set and pretrained model
2. For each SNR level and batch:
   - Normalize IQ to [0,1] using wrapper (unit or minmax mode)
   - Create torchattacks wrapper: `Model01Wrapper(model)`
   - Run attack (FGSM, PGD, CW, etc.) → adversarial batch `x_adv`
   - Optionally apply low-pass smoothing to delta
   - Denormalize back to [-1,1] IQ
3. Evaluate clean and adversarial accuracies per SNR

**Defense Flow:**

1. Given adversarial batch `x_adv` and defense type
2. If detector-gated:
   - Normalize to detector scale
   - Pass through AE, compute KL divergence
   - Identify high-KL samples (suspected adversarial)
3. For flagged samples, apply defense:
   - rFFT each IQ channel
   - Apply mask (notch, soft notch, top-K) in frequency domain
   - iRFFT to recover time domain
   - Return modified signal
4. Forward through clean model, measure recovered accuracy

**State Management:**

- Config object (`cfg`) holds dataset-specific parameters: class mappings, signal length, network hyperparameters
- Model state dict saved/loaded via `torch.save/load()` to `checkpoint/<dataset>_<model>.pkl`
- Detector state saved separately to `checkpoint/detector_ae.pth`
- Per-run output directories auto-increment: `training/<dataset>_0/`, `inference/<dataset>_1/`, etc.

## Key Abstractions

**Signal Tensor:**
- Representation: `[batch, 2, time_len]` where dimension 1 is I/Q channels
- Normalization: IQ samples typically in [-0.02, 0.02] range; models expect [-1, 1] or [0, 1] depending on mode
- Dataset variants: RML2016 has `time_len=128`, RML2018 has `time_len=1024`

**Model Wrapper:**
- Purpose: Adapt AWN (returns logits + regularization) for torchattacks (expects 4D image-style inputs)
- Implementation: `util/adv_attack.py:Model01Wrapper`
- Maps torchattacks [0,1] inputs back to [-1,1] IQ, forwards base model, returns logits only
- Supports dual normalization: unit (linear) and minmax (per-sample)

**Config Class:**
- Purpose: Centralized hyperparameter and path management
- Location: `util/config.py:Config`
- Init: Loads YAML from `config/<dataset>.yml` (epochs, lr, regularization weights, network depth)
- Dir management: Auto-creates `training/` or `inference/` subdirs with auto-incrementing indices
- Class mapping: Stores byte-keyed modulation↔int label dictionaries per dataset

**Defense Wrapper:**
- Abstraction: All defenses follow same interface: `def defense_fn(x: Tensor, **params) -> Tensor`
- Allows composability: can chain defenses or A/B test alternatives
- Normalization-aware: normalize/denormalize calls managed separately from FFT operation

## Entry Points

**Training:**
- Location: `main.py:__main__`, args.mode == 'train'
- Triggers: Command-line argument `--mode train`
- Responsibilities:
  1. Create model and load dataset
  2. Instantiate Trainer with train/val loaders
  3. Run training loop with early stopping
  4. Final evaluation on test set

**Evaluation (Clean):**
- Location: `main.py:__main__`, args.mode == 'eval'
- Triggers: `--mode eval --ckpt_path <checkpoint>`
- Responsibilities:
  1. Load pretrained model from checkpoint
  2. Run clean-signal inference on test set
  3. Compute per-SNR accuracies and confusion matrices
  4. Plot curves and heatmaps

**Adversarial Evaluation:**
- Location: `main.py:__main__`, args.mode == 'adv_eval'
- Triggers: `--mode adv_eval --attack <type> --defense <type>`
- Responsibilities:
  1. Load model and test set
  2. Generate adversarial examples with specified attack
  3. Optionally apply defense
  4. Evaluate accuracy before/after defense
  5. Save SNR-accuracy curves and defense effectiveness

**Multi-Attack Evaluation:**
- Location: `main.py:__main__`, args.mode == 'multi_attack_eval'
- Triggers: `--mode multi_attack_eval --attack_list "fgsm,pgd,cw,..." --plot_freq`
- Responsibilities:
  1. Load model and test set
  2. Run 15 different attacks (FGSM, PGD, CW, DeepFool, APGD, etc.)
  3. Test FFT Top-K recovery with multiple K values
  4. Compare attack accuracy vs defense accuracy per (modulation, SNR)
  5. Generate frequency domain comparison plots (clean vs adversarial spectra)

**SigGuard Evaluation:**
- Location: `main.py:__main__`, args.mode == 'sigguard_eval'
- Triggers: `--mode sigguard_eval --ckpt_path <checkpoint>`
- Responsibilities:
  1. Run 15 attacks and FFT Top-K defense
  2. Generate academic-style table: attack type vs accuracy (disabled/enabled)
  3. Generate IQ constellation scatter plots showing attack effect
  4. Output CSV and formatted text table

**Detector Training:**
- Location: `main.py:__main__`, args.mode == 'train_detector'
- Triggers: `--mode train_detector --det_epochs 10 --det_batch_size 256`
- Responsibilities:
  1. Load clean train/val splits
  2. Train RFSignalAutoEncoder with MSE loss
  3. Early stopping on validation reconstruction error
  4. Save detector to `checkpoint/detector_ae.pth`

**Detector Calibration:**
- Location: `main.py:__main__`, args.mode == 'calibrate_detector'
- Triggers: `--mode calibrate_detector --detector_ckpt ./checkpoint/detector_ae.pth`
- Responsibilities:
  1. Load detector and clean validation set
  2. Compute KL divergence for all clean samples
  3. Estimate threshold at specified quantile (default 90%)
  4. Print recommended `--detector_threshold` value

## Error Handling

**Strategy:** Graceful degradation with logging, silent fallback for optional components

**Patterns:**
- `torch.cuda.is_available()`: Auto-detect GPU, fallback to CPU
- Dataset not found: Raise `NotImplementedError` with path hint
- Config YAML missing: Raise `NotImplementedError` with expected location
- Visualization imports optional: Try-except on `util.visualize`, log skip message
- Defense mismatch: Check defense type against available functions, raise on unknown
- Attack backend selection: Try `torchattacks` first, fallback to internal implementation if requested
- Detector missing: Gracefully skip detector gating if `--detector_ckpt` not provided

## Cross-Cutting Concerns

**Logging:**
- Framework: Python `logging` module via `util/logger.py:create_logger()`
- Pattern: All major functions log progress, warnings, and results to both file and console
- Location: `<cfg.log_dir>/log.txt` and stdout

**Validation:**
- Tensor shape assertions in most functions (e.g., `assert x.dim() == 3 and x.size(1) == 2`)
- SNR/modulation filtering allows subset evaluation for fast iteration
- Signal normalization checks in defense and attack modules

**Authentication/Authorization:**
- Not applicable (academic research tool)

**Reproducibility:**
- Seed control: `util/utils.py:fix_seed()` sets numpy, torch, and random seeds
- Config-driven: All hyperparameters in YAML files
- Deterministic data split: Stratified by (modulation, SNR) with fixed random seed
- Checkpoint versioning: Auto-increment directories prevent overwrites
