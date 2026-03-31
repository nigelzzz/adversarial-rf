# Coding Conventions

**Analysis Date:** 2026-03-31

## Naming Patterns

**Files:**
- Snake_case with underscores: `data_loader.py`, `synth_txrx.py`, `adv_attack.py`
- Main entry point: `main.py`
- Experiment/standalone scripts: descriptive names in snake_case (e.g., `test_2016_10a.py`, `plot_iq_distribution.py`, `crc_experiment.py`)

**Functions:**
- **Core library functions:** PascalCase (preserve for backward compatibility)
  - Examples: `Load_Dataset()`, `Create_Data_Loader()`, `Run_Eval()`, `Run_Adv_Eval()` in `data_loader/data_loader.py`, `util/evaluation.py`, `util/adv_eval.py`
  - These form the public API and changing them breaks existing scripts
- **New utility functions:** snake_case preferred (e.g., `rrc_filter()`, `recover_constellation()`, `fix_seed()` in `util/utils.py`)
- **Helper/internal functions:** snake_case with leading underscore if private (e.g., `_lowpass_filter()`, `_batch_clip()` in `util/adv_attack.py`)

**Variables:**
- All lowercase with underscores: `signal_len`, `num_classes`, `batch_size`, `test_idx`
- Loop counters: Single letters acceptable (e.g., `i`, `snr_i`)
- Tensor batch variables: Descriptive snake_case (e.g., `sig_batch`, `lab_batch`, `adv_norm`)
- Constants: UPPER_CASE (e.g., `NUM_SAMPLES`, `SNR_THRESHOLD`, `DEVICE`)

**Types/Classes:**
- All PascalCase: `AWN`, `Config`, `Trainer`, `EarlyStopping`, `LiftingScheme`, `AverageMeter`
- No suffix for module classes (use `AWN` not `AWNModel`)

**Configuration Attributes:**
- PascalCase in YAML files (config/*.yml) → snake_case when accessed as Python object attributes
  - YAML: `epochs: 100` → Python: `cfg.epochs`
  - Class mappings: `cfg.classes` stores modulation name→index mappings as dicts with byte-string keys (e.g., `b'BPSK': 4`)

## Code Style

**Formatting:**
- No automated formatter configured (no .eslintrc, .prettierrc, biome.json detected)
- Follow Python style informally; consistency within files takes priority
- 4-space indentation used throughout
- Line length: Aim for ≤100 characters where practical (seen in docstrings and comments)

**Linting:**
- No linting rules enforced (no .eslintrc* files found)
- Code relies on developer discipline; bugs discovered through runtime validation

**Import Organization:**
```python
# Order in files:
1. Standard library (os, sys, time, random, pickle, logging)
2. Third-party (numpy, torch, torchvision, sklearn, matplotlib, yaml, h5py, scipy)
3. Local (data_loader, util, models)

# Example from main.py:
import argparse
import os.path
import sys

import numpy as np
import torch

from data_loader.data_loader import Create_Data_Loader, Load_Dataset, Dataset_Split
from util.config import Config, merge_args2cfg
```

**Path Aliases:**
- No path aliases configured (no jsconfig/tsconfig-style aliases)
- Relative imports from project root: `from models.model import AWN`, `from util.training import Trainer`

## Error Handling

**Pattern:**
```python
# Optional dependency handling (graceful degradation)
try:
    import torchattacks
    print("Using torchattacks library for CW attack")
except ImportError:
    print("ERROR: torchattacks not installed. Run: pip install torchattacks")
    return

# Conditional visualization (skip if matplotlib unavailable)
if cfg.Draw_Confmat is True:
    try:
        from util.visualize import Draw_Confmat
        Draw_Confmat(Confmat_Set, snrs, cfg)
    except Exception as e:
        logger.info(f'Skip confmat plotting: {e}')

# Config validation with NotImplementedError
if dataset == '2016.10a':
    self.signal_len = 128
    # ...
else:
    raise NotImplementedError(f"can not find cfg file: {yaml_name}")
```

**Conventions:**
- Use `try/except ImportError` for optional dependencies that will be selectively used
- Use `try/except Exception` to skip non-critical features (plotting, visualization) and log the skip
- Use `NotImplementedError` for unsupported dataset types or config options
- Use `ValueError` with descriptive messages for invalid arguments (e.g., unknown model name)
- Prefer explicit error messages with context (dataset name, config key, etc.)

## Logging

**Framework:** Python's built-in `logging` module via `util/logger.py:create_logger()`

**Setup Pattern:**
```python
from util.logger import create_logger

# Create once per run
logger = create_logger(f'{cfg.log_dir}/log.txt', file_handle=True)
logger.info('Starting training epoch 0')
logger.info(f'Setting learning rate to {lr:.5f}')
```

**When to Log:**
- Config/setup info: `log_exp_settings(logger, cfg)` in `util/utils.py` logs all config attributes once at startup
- Per-epoch metrics: Loss, accuracy, learning rate, elapsed time
- Important state changes: Model checkpoint saved, early stopping triggered, validation improved
- Warnings: Skipped features, invalid filters applied, dataset splits

**No logging for:**
- Individual batch processing (use tqdm progress bar instead)
- Per-sample predictions (would flood log; compute aggregate metrics instead)

## Comments

**When to Comment:**
- **Module-level docstrings:** Explain overall purpose and usage (especially test/experiment scripts)
- **Class docstrings:** What the class does, key methods
- **Complex algorithms:** Inline comments for mathematical operations (e.g., M-th power method in `recover_constellation()`)
- **Non-obvious design choices:** Why a particular approach was selected
- **Attribution:** Link to external sources (e.g., "adopted from pytorchtools: https://...") in `util/early_stop.py`

**JSDoc/TSDoc:**
- Not used (Python project)
- Docstrings follow NumPy style informally where present:
  ```python
  def recover_constellation(I, Q, sps=8, beta=0.35, mod_order=4):
      """
      Recover constellation points from raw oversampled IQ data.

      Applies matched filter (RRC), symbol-rate downsampling, and
      blind phase recovery (Viterbi & Viterbi for M-th power).

      Args:
          I: In-phase samples, 1-D numpy array
          Q: Quadrature samples, 1-D numpy array
          sps: Samples per symbol (8 for RML2016.10a)
          ...
      Returns:
          I_sym, Q_sym: Symbol-rate constellation points
      """
  ```

## Function Design

**Size:**
- Typical range: 10-50 lines
- Complex algorithms (signal processing): 50-100 lines acceptable (e.g., `Dataset_Split()` in `data_loader/data_loader.py` is 80 lines with per-SNR stratification logic)
- Avoid exceeding 150 lines; split into smaller functions if logic becomes hard to follow

**Parameters:**
- Order: Data → Configuration → Optional settings
  ```python
  def Run_Eval(model, sig_test, lab_test, SNRs, test_idx, cfg, logger):
      # model & data first, config/logger last
  ```
- Use keyword arguments for optional parameters (lambda functions use positional for brevity)
- Default values preferred: `kernel_size=17`, `beta=0.35`, `mod_order=4`

**Return Values:**
- Single return: scalar or array
- Multiple returns: tuple (unpacking at call site)
  ```python
  Signals, Labels, SNRs, snrs, mods = Load_Dataset(...)
  (Signals_train, Labels_train), (Signals_test, Labels_test), (Signals_val, Labels_val), test_idx = Dataset_Split(...)
  logit, regu_sum = model(sig_batch)  # Model returns both logit and regularization terms
  ```

## Module Design

**Exports:**
- Public functions and classes: No `__all__` lists; anything not prefixed with `_` is public
- Core library modules (`data_loader`, `util`, `models`): Export factory functions and main classes
  - `data_loader/data_loader.py`: Exports `Load_Dataset`, `Dataset_Split`, `Create_Data_Loader`
  - `util/utils.py`: Exports `create_AWN_model`, `create_VTCNN2_model`, `fix_seed`, etc.

**Barrel Files:**
- Not used (no index.ts style)
- Each module imported explicitly: `from util.training import Trainer`

**Lazy Imports:**
- Used for heavy/optional dependencies:
  ```python
  # util/utils.py: Lazy load MCLDNN (git submodule)
  _MCLDNN_PyTorch = None

  def _get_mcldnn():
      global _MCLDNN_PyTorch
      if _MCLDNN_PyTorch is None:
          sys.path.insert(0, os.path.join(..., 'MCLDNN'))
          from mcldnn_pytorch import MCLDNN_PyTorch
          _MCLDNN_PyTorch = MCLDNN_PyTorch
      return _MCLDNN_PyTorch

  # util/training.py: Lazy import visualize only when needed
  # from util.visualize import ... (avoided in main code path)
  ```

## Special Patterns

**Config Management:**
- Singleton Config per run: `cfg = Config(dataset, train=True/False)`
- Args merged into Config: `cfg = merge_args2cfg(cfg, vars(args))`
- Access via dot notation: `cfg.dataset`, `cfg.epochs`, `cfg.device`
- Initialize directories after args merged: `cfg.init_dir()`

**Global Variables:**
- Used sparingly for cached state or module-level setup
- Example: `_MCLDNN_PyTorch` in `util/utils.py` (lazy import cache)
- Example: `global regu_d, regu_c` in `models/model.py` (in conditional block; not ideal but preserved)

**Byte String Keys:**
- Dataset class mappings use byte-string keys: `{b'BPSK': 4, b'QPSK': 9}`
- Necessary because pickle dataset files store modulation names as bytes
- Converted to int indices when building labels: `Labels = [classes[i] for i in Labels]`

---

*Convention analysis: 2026-03-31*
