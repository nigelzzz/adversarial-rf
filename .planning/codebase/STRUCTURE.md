# Codebase Structure

**Analysis Date:** 2026-03-31

## Directory Layout

```
adversarial-rf/
├── main.py                          # Entry point: arg parsing, mode routing
├── synth_finetune.py               # Synthetic data generation and finetuning
├── crc_experiment.py               # CRC/FEC experiment harness
├── crc_defense_*.py                # CRC defense pipeline variants
├── adaptive_k_*.py                 # Adaptive Top-K selection experiments
├── plot_*.py                       # Visualization/analysis scripts (standalone)
│
├── models/                          # Neural network architectures
│   ├── model.py                    # AWN: main classifier (conv+lifting+attention+fc)
│   ├── lifting.py                  # LiftingScheme: adaptive wavelet decomposition
│   ├── vtcnn2.py                   # VTCNN2: alternative architecture
│   ├── resnet1d.py                 # ResNet1D: alternative architecture
│   └── lstm_amc.py                 # LSTM-based AMC model
│
├── data_loader/                     # Dataset management
│   └── data_loader.py              # Load_Dataset, Dataset_Split, Create_Data_Loader
│
├── util/                            # Utilities and core algorithms
│   ├── config.py                   # Config class: hyperparameter management
│   ├── training.py                 # Trainer: epoch-based training loop
│   ├── evaluation.py               # Run_Eval: clean signal evaluation
│   ├── adv_eval.py                 # Run_Adv_Eval: adversarial evaluation
│   ├── adv_attack.py               # CW attack, spectral attacks, attack wrappers
│   ├── defense.py                  # FFT notch, soft notch, top-K defenses
│   ├── detector.py                 # RFSignalAutoEncoder: anomaly detector
│   ├── detector_train.py           # Train and calibrate detector
│   ├── multi_attack_eval.py        # Multi-attack evaluation framework
│   ├── sigguard_eval.py            # SigGuard-style paper table generation
│   ├── synth_txrx.py               # Synthetic TX/RX chain (modulation, CRC, FEC)
│   ├── utils.py                    # Utilities: seed fixing, model creation, RRC filter
│   ├── logger.py                   # Logging and AverageMeter
│   ├── visualize.py                # Plot confusion matrix, SNR curves, lifting scheme
│   ├── early_stop.py               # EarlyStopping: validation-based stopping
│   ├── freq_compare.py             # Frequency domain comparison
│   ├── freq_topk_eval.py           # FFT Top-K percentage evaluation
│   ├── freq_topk_adv_eval.py       # Adversarial Top-K evaluation
│   ├── bench.py                    # Attack benchmarking
│   ├── psd_tools.py                # PSD mask computation
│   ├── transfer_eval.py            # Transfer attack evaluation across models
│   ├── power_budget_eval.py        # Power constraint evaluation
│   ├── adaptive_defense.py         # Adaptive defense selection
│   ├── adaptive_eval.py            # Adaptive evaluation
│   ├── adaptive_k_calibration.py   # Adaptive K threshold learning
│   ├── adaptive_attack.py          # Adaptive attack strategies
│   └── adv_training.py             # Adversarial training loop
│
├── config/                          # Dataset configurations
│   ├── 2016.10a.yml                # RML2016.10a: 11 classes, 128 samples
│   ├── 2016.10b.yml                # RML2016.10b: 10 classes, 128 samples
│   └── 2018.01a.yml                # RML2018.01a: 24 classes, 1024 samples
│
├── data/                            # Dataset files (git-ignored, user-downloaded)
│   ├── RML2016.10a_dict.pkl        # RML2016.10a pickle
│   ├── RML2016.10b.dat             # RML2016.10b pickle
│   └── GOLD_XYZ_OSC.0001_1024.hdf5 # RML2018.01a HDF5
│
├── checkpoint/                      # Model checkpoints (auto-created)
│   ├── 2016.10a_AWN.pkl            # Main classifier checkpoint
│   ├── 2016.10a_AWN_ft.pkl         # Finetuned variant
│   └── detector_ae.pth             # Anomaly detector checkpoint
│
├── training/                        # Training outputs (auto-created, indexed)
│   └── 2016.10a_0/                 # Run directory (index auto-increments)
│       ├── models/                 # Saved checkpoints per epoch
│       ├── log/                    # Training logs
│       │   └── log.txt             # Detailed epoch-level metrics
│       └── result/                 # Plots and results
│           ├── acc_snr_curve.png
│           └── confusion_matrix.png
│
├── inference/                       # Inference outputs (auto-created, indexed)
│   └── 2016.10a_0/                 # Run directory (index auto-increments)
│       ├── models/                 # [symlink/reference to checkpoint]
│       ├── log/                    # Evaluation logs
│       └── result/                 # Analysis results
│           ├── confusion_matrix.png
│           ├── sigguard_eval.csv
│           ├── multi_attack_snr_mod_eval.csv
│           ├── freq_plots/         # Frequency domain comparisons (if --plot_freq)
│           └── iq_plots/           # IQ constellation plots (if --plot_iq)
│
├── results/                         # Experiment results (user-created)
│   ├── crc_defense_direct/
│   ├── crc_defense_fec/
│   ├── crc_defense_pipeline/
│   └── runs/
│
├── reports/                         # Analysis reports (markdown)
│   ├── ead_adaptive_k_report.md
│   └── ead_adaptive_k_report_CN.md
│
├── .planning/codebase/              # GSD codebase documentation
│   ├── ARCHITECTURE.md
│   ├── STRUCTURE.md
│   ├── STACK.md
│   ├── INTEGRATIONS.md
│   ├── CONVENTIONS.md
│   ├── TESTING.md
│   └── CONCERNS.md
│
├── CLAUDE.md                        # Project instructions (this file)
├── AGENTS.md                        # Agent instructions
└── README.md                        # General documentation

```

## Directory Purposes

**models/:**
- Purpose: Neural network architectures for automatic modulation classification
- Contains: AWN (main), VTCNN2, ResNet1D, LSTM-based variants
- Key files: `model.py` (AWN forward pass), `lifting.py` (wavelet decomposition operators)

**data_loader/:**
- Purpose: Load and prepare datasets for training/evaluation
- Contains: RML2016.10a/b pickle loaders, RML2018.01a HDF5 loader, stratified train/val/test splitting
- Key files: `data_loader.py` (all dataset operations)

**util/:**
- Purpose: Core algorithms and utilities (training, attacks, defenses, evaluation, synthetic data)
- Contains: 25+ utility modules organized by function (training, adversarial, defense, etc.)
- Key files:
  - Training: `training.py`, `utils.py`
  - Attacks: `adv_attack.py`, `adaptive_attack.py`
  - Defenses: `defense.py`, `adaptive_defense.py`
  - Evaluation: `evaluation.py`, `adv_eval.py`, `multi_attack_eval.py`, `sigguard_eval.py`
  - Synthetic: `synth_txrx.py` (TX/RX chain, CRC, FEC)
  - Detection: `detector.py`, `detector_train.py`

**config/:**
- Purpose: Dataset-specific hyperparameters and network configuration
- Contains: YAML files for RML2016.10a, 2016.10b, 2018.01a
- Format: epochs, batch_size, learning_rate, model depth, regularization weights, class counts

**checkpoint/:**
- Purpose: Saved model state dicts (PyTorch pickles)
- Contains: `<dataset>_<model>.pkl` and `detector_ae.pth`
- Naming: `2016.10a_AWN.pkl`, `2016.10a_VTCNN2.pkl`, etc.
- User-downloaded: Yes (too large for git)

**training/:**
- Purpose: Auto-generated output from `--mode train`
- Contains: Model checkpoints, training logs, plots
- Directory structure: `2016.10a_0/`, `2016.10a_1/`, etc. (auto-incremented per run)
- Subdirs: `models/`, `log/`, `result/`

**inference/:**
- Purpose: Auto-generated output from evaluation modes (`--mode eval`, `adv_eval`, `multi_attack_eval`, etc.)
- Contains: Results, plots, CSV tables
- Directory structure: `2016.10a_0/`, `2016.10a_1/`, etc. (auto-incremented)
- Subdirs: `log/`, `result/` (contains CSV, plots, frequency comparisons)

**results/:**
- Purpose: User-created experiment result aggregations
- Contains: CRC experiment outputs, ablation study results, comparison tables
- Subdirs: `crc_defense_direct/`, `crc_defense_fec/`, etc.

**reports/:**
- Purpose: Human-written analysis and findings
- Contains: Markdown reports on specific experiments (adaptive K selection, EAD attacks, etc.)

## Key File Locations

**Entry Points:**
- `main.py`: Command-line interface, mode routing (train/eval/adv_eval/multi_attack_eval/sigguard_eval/etc.)
- `synth_finetune.py`: Synthetic data generation and finetuning workflow
- `crc_experiment.py`: CRC/FEC communication experiment

**Configuration:**
- `util/config.py:Config`: Runtime configuration class with dataset-specific mappings
- `config/2016.10a.yml`: Hyperparameters (epochs=100, batch_size=128, lr=0.001, etc.)
- `CLAUDE.md`: Project instructions with all command examples

**Core Logic (Models):**
- `models/model.py:AWN`: Main classifier combining conv, lifting, attention, FC layers
- `models/lifting.py:LiftingScheme`: Adaptive wavelet decomposition with learnable P/U operators
- `util/training.py:Trainer`: Training loop with early stopping and LR decay

**Attacks & Defenses:**
- `util/adv_attack.py`: CW L2 attack, spectral noise attacks, torchattacks wrapper
- `util/defense.py`: FFT notch, soft notch, Top-K, AE detector gating
- `util/detector.py:RFSignalAutoEncoder`: 1D conv autoencoder for anomaly detection

**Evaluation:**
- `util/evaluation.py:Run_Eval()`: Clean signal evaluation (per-SNR accuracy, confusion matrix)
- `util/adv_eval.py:Run_Adv_Eval()`: Single attack evaluation with optional defense
- `util/multi_attack_eval.py:run_multi_attack_snr_mod_eval()`: 15 attacks, FFT Top-K comparison, frequency plots
- `util/sigguard_eval.py:run_sigguard_eval()`: Academic-style table + IQ constellation plots

**Synthetic Data:**
- `util/synth_txrx.py`: TX/RX chain (modulation mapping, RRC shaping, channel noise, matched filter, CRC, FEC)
- `synth_finetune.py`: Curriculum finetuning on synthetic data (3 progressive stages)

**Testing:**
- No formal unit tests (validation via `--eval_limit` for fast iteration)
- Test scripts: `test_*.py` in root directory (quick hypothesis checks)

## Naming Conventions

**Files:**
- Python modules: `snake_case.py` (e.g., `data_loader.py`, `adv_attack.py`)
- Classes: `PascalCase` (e.g., `AWN`, `Trainer`, `Config`)
- Functions: Mixed (legacy `PascalCase` for public APIs like `Load_Dataset()`, new code uses `snake_case`)
- Config files: `<dataset>.yml` (e.g., `2016.10a.yml`)
- Model checkpoints: `<dataset>_<model>.pkl` (e.g., `2016.10a_AWN.pkl`)
- Directories: `snake_case` (e.g., `data_loader/`, `util/`)

**Directories:**
- Auto-generated run directories: `<base>/<dataset>_<index>/` where base is `training/` or `inference/`
- Output subdirs: `models/`, `log/`, `result/`
- Result files: `<experiment>_<descriptor>.csv` (e.g., `multi_attack_snr_mod_eval.csv`)
- Plot files: `<attack>_<modulation>_snr<N>_<type>.png` or similar

## Where to Add New Code

**New Attack:**
1. Implement attack function in `util/adv_attack.py` with signature: `def my_attack(model, x, y, **kwargs) -> x_adv`
2. Add case in `util/adv_eval.py:Run_Adv_Eval()` to detect attack type and call function
3. Register in `util/multi_attack_eval.py:run_multi_attack_snr_mod_eval()` attack list if for batch evaluation
4. Add command-line args in `main.py` if needs new parameters (use `parser.add_argument()`)

**New Defense:**
1. Implement function in `util/defense.py` with signature: `def my_defense(x: Tensor, **params) -> Tensor`
2. Ensure it handles `[N, 2, T]` IQ tensors and returns same shape
3. Add case in `util/adv_eval.py:Run_Adv_Eval()` or `util/defense.py:apply_defense()` dispatcher
4. Register in command-line args: `--defense my_defense_name`

**New Model Architecture:**
1. Create `models/my_model.py` with class `MyModel(nn.Module)`
2. Return tuple `(logits, regu_list)` from `forward()` to match AWN interface
3. Register in `util/utils.py:create_model()` factory function
4. Add command-line arg `--model my_model`

**New Evaluation Mode:**
1. Create analysis function (e.g., `util/my_analysis.py:run_my_analysis()`)
2. Call from `main.py` in appropriate `elif args.mode == 'my_mode':` block
3. Add command-line `parser.add_argument('--mode')` if new mode not yet recognized
4. Output results to `cfg.result_dir/` with descriptive filename

**New Utility Function:**
1. Place in appropriate `util/` module by function (if training → `util/training.py`, if attack → `util/adv_attack.py`, etc.)
2. Use type hints: `def fn(x: torch.Tensor, param: float) -> torch.Tensor:`
3. Add docstring explaining input/output shapes and semantics
4. Prefer snake_case for new functions

## Special Directories

**checkpoint/:**
- Purpose: Persistent model storage
- Generated: User downloads pretrained `.pkl` files from source
- Committed: No (too large, typically 500MB+ per model)
- Management: Manually place files or save via `torch.save(model.state_dict(), path)`

**training/ and inference/:**
- Purpose: Experiment outputs
- Generated: Yes, auto-created by `Config.init_dir()` on first run
- Committed: No (.gitignore)
- Management: Auto-increment directories prevent overwrites; clean old runs manually if space needed

**data/:**
- Purpose: Dataset files
- Generated: No (user downloads from external source)
- Committed: No (.gitignore)
- Management: User places `RML2016.10a_dict.pkl`, `RML2016.10b.dat`, `GOLD_XYZ_OSC.0001_1024.hdf5`

**.planning/codebase/:**
- Purpose: GSD orchestrator documentation (internal)
- Generated: No (manually written by code mapping agents)
- Committed: Yes
- Files: ARCHITECTURE.md, STRUCTURE.md, STACK.md, INTEGRATIONS.md, CONVENTIONS.md, TESTING.md, CONCERNS.md

**results/:**
- Purpose: Experiment aggregations
- Generated: Yes, but manually organized
- Committed: No (.gitignore typically)
- Subdirs: Created per experiment (e.g., `crc_defense_fec/` for FEC-based CRC defense results)

---

## Summary: File Organization Strategy

- **Top-level** (`main.py`, `*.py` scripts): Entry points and standalone experiments
- **models/**: Network architectures (AWN, alternatives)
- **data_loader/**: Dataset I/O and splitting logic
- **util/**: Algorithms (training, attack, defense, evaluation, synthetic data)
- **config/**: YAML hyperparameter files (dataset-specific)
- **checkpoint/, training/, inference/**: Runtime auto-generated output
- **data/**: User-provided datasets
- **.planning/codebase/**: GSD documentation

This structure balances modularity (separate concerns in util/), clarity (descriptive names), and scalability (new modes added without modifying core).
