# Technology Stack

**Analysis Date:** 2026-03-31

## Languages

**Primary:**
- Python 3.6+ - All core source code, scripts, and utilities

## Runtime

**Environment:**
- CPython 3.6+ (tested on 3.8+)
- CUDA-capable GPU recommended (auto-detected, falls back to CPU)

**Package Manager:**
- pip
- Lockfile: requirements.txt present

## Frameworks

**Core ML:**
- PyTorch 1.7+ (tested on 1.8.1) - Neural network training, inference, tensor operations
  - `torch.nn` - Model definitions (Conv1d, Conv2d, Linear, ModuleList)
  - `torch.optim` - Optimizers (Adam, SGD via optim.Adam, optim.SGD)
  - `torch.utils.data` - DataLoader for batch processing

**Attack/Defense:**
- torchattacks - Adversarial attack library (FGSM, PGD, CW, DeepFool, and 12+ more)
  - Used in `util/adv_eval.py` and `util/multi_attack_eval.py`
  - Supports both internal and library-based CW attacks
  - Default epsilon handling via `--ta_box` normalization (unit or minmax)

**Testing:**
- No formal unit test framework detected
- Validation via manual runs: `--mode train`, `--mode eval`, `--mode adv_eval`

**Build/Dev:**
- argparse - Command-line argument parsing
- tqdm - Progress bar visualization

## Key Dependencies

**Critical:**
- numpy - Numerical computing, signal processing, array operations
- torch - Deep learning framework (model, optim, data)
- torchvision - Pretrained models (if extended), transforms
- torchaudio - Audio processing utilities
- torchattacks - Adversarial attack library (17 attack methods available)

**Infrastructure:**
- scikit-learn 0.24+ - Metrics (accuracy_score, confusion_matrix, f1_score, cohen_kappa_score)
  - Used in `util/adv_eval.py:Run_Adv_Eval` and `util/evaluation.py:Run_Eval`
- h5py - HDF5 dataset I/O for RML2018.01a format
  - Used in `data_loader/data_loader.py` for GOLD_XYZ_OSC.0001_1024.hdf5
- pyyaml - YAML configuration parsing
  - Used in `util/config.py:Config.__init__` to load dataset configs from `config/2016.10a.yml` etc.
- scipy - Scientific computing (signal processing in defense/attack modules)
- matplotlib - Plotting and visualization for analysis scripts
- pandas - Data analysis and CSV/table output

## Configuration

**Environment:**
- Runtime device selection: Auto-detects CUDA, falls back to CPU
- Environment variable used: `PYHONHASHSEED` set in `util/utils.py:fix_seed()` for reproducibility
- Model checkpoint path: Configurable via `--ckpt_path` (default: `./checkpoint`)

**Build:**
- No build system (pure Python)
- Entry point: `main.py` with mode-based dispatching (train, eval, adv_eval, visualize, etc.)
- Dataset configs: YAML files in `config/` directory
  - `config/2016.10a.yml` - 11 modulation classes, 128-sample signals
  - `config/2016.10b.yml` - 10 modulation classes, 128-sample signals
  - `config/2018.01a.yml` - 24 modulation classes, 1024-sample signals

## Platform Requirements

**Development:**
- Python 3.6+ with pip
- PyTorch installation (CPU or GPU)
- Disk space for dataset files (RML2016: ~500MB, RML2018: ~3GB)
- Recommended: NVIDIA GPU with CUDA support for faster training

**Production/Inference:**
- Same as development (pure Python + PyTorch)
- Model checkpoint files: `<dataset>_AWN.pkl` (500MB-1.5GB per model)
- Evaluation outputs: `inference/<dataset>_*/result/` directory structure

## Data Storage

**Dataset Formats:**
- Pickle (`.pkl`) - RML2016.10a and RML2016.10b signal data
  - Keys: `(modulation_bytes, snr_int)` tuples
  - Values: numpy arrays of I/Q samples
  - Located: `./data/RML2016.10a_dict.pkl`, `./data/RML2016.10b.dat`
- HDF5 (`.hdf5`) - RML2018.01a signal data
  - Datasets: `X` (signals), `Y` (labels), `Z` (SNR values)
  - Located: `./data/GOLD_XYZ_OSC.0001_1024.hdf5`

**Model Checkpoints:**
- PyTorch `.pkl` format (state_dict)
- Path: `./checkpoint/<dataset>_AWN.pkl` (main model)
- Detector: `./checkpoint/detector_ae.pth` (autoencoder for adversarial detection)
- Loading: `torch.load(ckpt_path, map_location=device, weights_only=True)`

**Logging:**
- Directory: `training/<dataset>_<index>/log/log.txt`
- CSV output: `training/<dataset>_<index>/log/` (epoch stats)
- Results: `training/<dataset>_<index>/result/` and `inference/<dataset>_*/result/`

## Utility Libraries

**Signal Processing:**
- scipy.signal - Filtering, FFT operations
- numpy.fft - Fast Fourier Transform for frequency-domain analysis and defense

**Evaluation:**
- pandas - DataFrame operations for metric aggregation and CSV export
- sklearn.metrics - Confusion matrix, F1, Kappa, accuracy computation

**Data Loading:**
- pickle - Python object serialization (dataset loading)
- h5py - HDF5 file I/O (RML2018 dataset)

## Code Structure

**Entry Points:**
- `main.py` - Primary command dispatcher for all modes (train, eval, adv_eval, visualize, etc.)
- `synth_finetune.py` - Standalone script for synthetic data generation and finetuning
- Multiple plot/test scripts: `plot_*.py`, `test_*.py`, `*_experiment.py`

**Core Modules:**
- `models/` - Neural network architectures
  - `model.py:AWN` - Adaptive Wavelet Network
  - `lifting.py:LiftingScheme` - Lifting scheme wavelet decomposition
  - `vtcnn2.py`, `resnet1d.py`, `lstm_amc.py` - Alternative architectures
- `util/` - Utilities and algorithms
  - `config.py` - YAML config loader
  - `training.py:Trainer` - Training loop with early stopping
  - `evaluation.py:Run_Eval` - Evaluation metrics
  - `adv_eval.py:Run_Adv_Eval` - Adversarial evaluation
  - `adv_attack.py` - Attack implementations (CW, spectral noise)
  - `defense.py` - Defense mechanisms (FFT-domain recovery)
  - `synth_txrx.py` - Synthetic signal generation
  - `detector.py` - Autoencoder for anomaly detection
- `data_loader/` - Dataset loading and preprocessing

---

*Stack analysis: 2026-03-31*
