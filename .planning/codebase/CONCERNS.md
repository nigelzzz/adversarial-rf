# Codebase Concerns

**Analysis Date:** 2026-03-31

## Resource Management Issues

### Unclosed File Handles

**Issue:** Multiple locations load files without using context managers, risking file descriptor leaks.

**Files:**
- `data_loader/data_loader.py:37` — `pickle.load(open(file_pointer, 'rb'), encoding='bytes')`
- `util/config.py:27` — `cfg = yaml.safe_load(open(yaml_name, 'r'))`
- `plot_all_attacks_iq.py:37, 57` — `pickle.load(open(...))`, `yaml.safe_load(open(...))`
- Multiple plot/test scripts with same pattern

**Impact:** Under long-running processes or repeated dataset loads, file handle exhaustion could cause `OSError: Too many open files` and crash evaluation pipelines.

**Fix approach:** Replace `open()` with context managers:
```python
# BAD (current)
Set = pickle.load(open(file_pointer, 'rb'), encoding='bytes')

# GOOD
with open(file_pointer, 'rb') as f:
    Set = pickle.load(f, encoding='bytes')
```

---

## Global Variable Anti-Patterns

### Mutable State in Forward Pass

**Issue:** Global variables used to return values from model forward operations.

**Files:**
- `models/model.py:25` — `global regu_d, regu_c` inside `LevelTWaveNet.forward()`
- `paper/AWN_All.py:182` — Same pattern

**Impact:** Not thread-safe. Running parallel evaluations or data loading with `num_workers > 0` can cause race conditions where regularization terms get corrupted.

**Safe modification:**
```python
# CURRENT: unsafe
def forward(self, x):
    global regu_d, regu_c
    (L, H) = self.wavelet(x)
    if self.regu_approx + self.regu_details != 0.0:
        if self.regu_details:
            regu_d = self.regu_details * H.abs().mean()
        ...
    return approx, details, regu

# BETTER: local variables
def forward(self, x):
    (L, H) = self.wavelet(x)
    regu = None
    if self.regu_approx + self.regu_details != 0.0:
        regu_d = self.regu_details * H.abs().mean() if self.regu_details else 0
        regu_c = self.regu_approx * torch.dist(...) if self.regu_approx else 0
        regu = regu_d + regu_c
    return approx, details, regu
```

---

## Data Loading Fragility

### Hardcoded Class Mappings in Two Locations

**Issue:** Class label mappings (modulation format to index) are duplicated and must be kept in sync.

**Files:**
- `util/config.py:50-66` — Classes defined in `Config.__init__` for 2016.10a, 2016.10b, 2018.01a
- `data_loader/data_loader.py:11-22` — Identical mappings defined again

**Impact:** Adding a dataset or reordering classes in one place breaks consistency. Tests silently misclassify if mappings drift.

**Fix approach:** Define once in shared module (`util/dataset_config.py`), import both places:
```python
# util/dataset_config.py
DATASET_CLASSES = {
    '2016.10a': {b'QAM16': 0, b'QAM64': 1, ...},
    '2016.10b': {...},
    ...
}

# In both files
from util.dataset_config import DATASET_CLASSES
classes = DATASET_CLASSES[dataset]
```

---

## Typo in Fix Seed

**Issue:** Environment variable typo prevents reproducibility guarantee.

**Files:**
- `util/utils.py:96` — `os.environ['PYHONHASHSEED'] = str(seed)` (should be `PYTHONHASHSEED`)

**Impact:** Python hash randomization is NOT disabled because the env var name is wrong. Floating-point operations and set iteration may vary across runs despite `fix_seed()` being called.

**Fix approach:** Correct the env var name:
```python
os.environ['PYTHONHASHSEED'] = str(seed)  # was: PYHONHASHSEED
```

---

## Test/Evaluation Data Leakage

### Global Variable in Dataset Split

**Issue:** `test_idx` is defined as global but shadowed locally.

**Files:**
- `data_loader/data_loader.py:99` — `global test_idx` declared but never used outside function
- Line 146 returns `test_idx` from local scope

**Impact:** Code is confusing. If another caller tries to access `test_idx` globally, they get stale or undefined data. The global is unnecessary.

**Fix approach:** Remove the unused global declaration:
```python
def Dataset_Split(...):
    # Remove: global test_idx
    test_idx = []
    # ... rest of function
    return (...), test_idx
```

---

## Regularization Term Edge Case

**Issue:** In `models/model.py:29-45`, if both `regu_approx` and `regu_details` are 0.0, the function returns `None` for regularization instead of a list.

**Files:**
- `models/model.py:19-45` — `LevelTWaveNet.forward()` has conditional return

**Impact:** Calling code expects `regu` to always be a tensor/float for `loss += sum(regu_sum)` (line 100 in `util/training.py`). If regu is None and added to list, `sum(regu_sum)` will fail or behave unexpectedly.

**Verification:** Look at line 100 of `util/training.py`:
```python
loss += sum(regu_sum)
```
If `regu_sum` contains None values, this will cause `TypeError: unsupported operand type(s) for +: 'int' and 'NoneType'`.

**Fix approach:** Always return a tensor (even if zero):
```python
def forward(self, x):
    (L, H) = self.wavelet(x)
    regu = torch.tensor(0.0, device=x.device, dtype=x.dtype)
    if self.regu_approx + self.regu_details != 0.0:
        # ... compute regu_d, regu_c
        regu = regu_d + regu_c
    return approx, details, regu
```

---

## Potential NumPy/Torch Tensor Conversion Issues

**Issue:** Multiple places mix NumPy arrays and PyTorch tensors without explicit conversion checks.

**Files:**
- `util/synth_txrx.py:254-298` — `compute_llr()` uses NumPy, returns NumPy arrays
- `util/synth_txrx.py:300-316` — `estimate_noise_var()` uses NumPy
- `synth_finetune.py:100-120` — Synthetic data generation uses NumPy, later converts to torch

**Impact:** Callers might pass torch tensors to NumPy functions expecting arrays, causing silent failures or degraded performance. Type confusion can cause subtle bugs in batch processing.

**Fix approach:** Add explicit type checks and conversions:
```python
def compute_llr(symbols, mod_type, noise_var):
    """Compute per-bit LLRs."""
    if isinstance(symbols, torch.Tensor):
        symbols = symbols.cpu().numpy()
    # ... rest of function
```

---

## Missing Input Validation

### Unvalidated File Paths

**Issue:** Config expects files at hardcoded paths (e.g., `./data/RML2016.10a_dict.pkl`) without existence checks before loading.

**Files:**
- `util/config.py:25-26` — Checks if YAML exists but doesn't validate dataset files until runtime
- `data_loader/data_loader.py:30-37` — Constructs path, loads without checking existence

**Impact:** If dataset file is missing, user gets confusing pickle/h5py error instead of clear message. Multi-hour evaluations can fail partway through.

**Fix approach:** Add early validation in `Load_Dataset`:
```python
if not os.path.exists(file_pointer):
    raise FileNotFoundError(f"Dataset not found: {file_pointer}")
```

### Unvalidated SNR/Modulation Filters

**Issue:** `--mod_filter` and `--snr_filter` arguments are not validated before data loading.

**Files:**
- `main.py:30-31` — Arguments accepted as strings/ints
- `data_loader/data_loader.py:40-48` — Filters silently produce empty dataset if invalid

**Impact:** Typo in `--mod_filter QAM16X` (X at end) silently returns zero samples instead of error. Evaluation runs but reports 0% accuracy.

**Fix approach:** Validate against known classes:
```python
if mod_filter is not None:
    if isinstance(mod_filter, str):
        mod_filter = mod_filter.encode()
    if mod_filter not in classes:
        raise ValueError(f"Unknown modulation: {mod_filter}")
```

---

## Determinism Issues Under Parallelism

**Issue:** `fix_seed()` sets torch seeds but doesn't prevent non-determinism if `num_workers > 0` in DataLoader.

**Files:**
- `util/utils.py:93-102` — Sets random seeds
- `util/training.py:154-166` — Creates DataLoader with `num_workers=cfg.num_workers`
- `data_loader/data_loader.py:154-166` — Passes `num_workers` through

**Impact:** With multi-worker data loading, different workers may use different random states, producing non-reproducible batches. Model training produces different results on different runs despite fixed seed.

**Fix approach:** Set worker initialization function:
```python
def _worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

train_loader = Data.DataLoader(
    dataset=train_data,
    batch_size=cfg.batch_size,
    shuffle=True,
    num_workers=cfg.num_workers,
    worker_init_fn=_worker_init_fn,
)
```

---

## Inconsistent Naming Conventions

**Issue:** Mixed naming style makes code harder to navigate. Existing classes use PascalCase (`Create_Data_Loader`, `Run_Eval`, `Load_Dataset`), but newer code uses snake_case.

**Files:**
- Old API: `data_loader/data_loader.py` — `Create_Data_Loader`, `Load_Dataset`, `Dataset_Split`
- New code: `util/synth_txrx.py` — `make_rml_like_burst`, `estimate_noise_var`, `get_constellation`
- New code: `util/defense.py` — `fft_notch_denoise`, `fft_soft_notch_denoise`, `_band_to_bins`

**Impact:** Developers must remember two conventions. Scripts mixing old and new APIs are inconsistent and harder to review.

**Fix approach:** Document decision (e.g., "All new public APIs use snake_case; existing PascalCase functions preserved for backward compatibility") and enforce with linter.

---

## Missing Configuration Defaults

**Issue:** Many optional parameters have hardcoded defaults in function signatures rather than config files.

**Files:**
- `util/synth_txrx.py:74` — `channel_preset='full'` hardcoded
- `util/synth_txrx.py:74` — `target_rms=0.006` hardcoded
- `main.py` line 81 — `detector_threshold` hardcoded to `4.468164592981338e-03`

**Impact:** Users cannot easily override defaults without modifying source. Magic numbers in function calls are hard to track.

**Fix approach:** Move to config files or argparse defaults:
```python
parser.add_argument('--synth_channel_preset', type=str, default='full')
parser.add_argument('--synth_target_rms', type=float, default=0.006)
```

---

## Unreachable Code Path

### Incomplete Regularization Logic

**Issue:** In `models/model.py:35-43`, the logic for selecting regularization terms can be clearer.

**Files:**
- `models/model.py:35-43` — Multiple if/elif branches for `regu_approx` and `regu_details`

**Code:**
```python
if self.regu_approx == 0.0:
    regu = regu_d  # Only details
elif self.regu_details == 0.0:
    regu = regu_c  # Only approx
else:
    regu = regu_d + regu_c  # Both
```

**Impact:** If both are 0.0, the outer if at line 29 is False, so this code never runs. But then line 45 returns without a return statement (implicit None). This is caught by the training loop but creates unnecessary complexity.

---

## Critical: Potential NaN/Inf Propagation

### Unprotected Division in Defense

**Issue:** Soft notch computation uses torch operations that could produce NaN/Inf.

**Files:**
- `util/defense.py:123` — `t = 0.5 * (1 + torch.cos(...))`
- `util/defense.py:130` — Same computation in loop

**Edge case:** If transition width is very large or frequency range is zero, cos evaluation could produce unexpected values.

**Impact:** Silent NaN propagation through to accuracy calculations. Defense output becomes garbage, metrics unreliable.

**Fix approach:** Add bounds checking:
```python
def _soft_notch_mask(...):
    # ... validate inputs
    if trans <= 0 or k_high < k_low:
        # Return binary mask instead
        m = torch.ones(F, device=device, dtype=dtype)
        m[k_low:k_high + 1] = 1.0 - float(depth)
        return m
```

---

## Performance Bottlenecks

### Inefficient Slicing in Dataset_Split

**Issue:** Dataset splitting uses nested NumPy operations that could be vectorized.

**Files:**
- `data_loader/data_loader.py:107-124` — Loop over cells, uses `np.linspace`, `np.random.choice`, set operations

**Code complexity:** O(n_mods * n_snrs * log n) due to repeated choice operations.

**Impact:** For 2018.01a with 24 modulations and 20+ SNRs, dataset split can take seconds. Not a blocker but noticeable on repeated runs.

**Fix approach:** Use stratified split from sklearn:
```python
from sklearn.model_selection import train_test_split
# Stratify by (modulation, SNR) tuple for balanced splits
```

---

## Test Coverage Gaps

**Issue:** No formal unit tests. Validation is manual via CLAUDE.md commands.

**Files:**
- `test/` directory exists but is empty or contains only demo scripts
- Main validation approach: run `python main.py --mode train` and check logs

**Impact:** Refactoring risks breaking undocumented behaviors. Regressions only caught during full training runs (hours).

**Recommendation:** Add minimal smoke tests:
```python
# test/test_model.py
def test_awn_forward_shape():
    cfg = Config('2016.10a')
    model = AWN(num_classes=11, num_levels=2)
    x = torch.randn(4, 2, 128)
    logit, regu = model(x)
    assert logit.shape == (4, 11)
    assert isinstance(regu, list)
```

---

## Documentation Debt

### Missing Docstrings

**Issue:** Many public functions lack docstrings or have incomplete ones.

**Files:**
- `util/defense.py:22-59` — `fft_notch_denoise` has docstring, but most smaller helpers don't
- `util/synth_txrx.py` — Good docstrings, but some edge cases not documented
- `data_loader/data_loader.py:92-146` — `Dataset_Split` lacks return type documentation

**Impact:** New developers can't determine function intent without reading implementation. Easy to misuse APIs.

**Fix approach:** Enforce docstrings with linter:
```python
def Dataset_Split(Signals, Labels, snrs, mods, logger, val_size=0.2, test_size=0.2):
    """Split signals/labels into train/val/test preserving SNR/modulation balance.

    Args:
        Signals: [N, 2, T] tensor
        Labels: [N] tensor of class indices
        snrs: List of SNR values in dataset
        mods: List of modulation types
        logger: Logger object
        val_size: Fraction for validation (default 0.2)
        test_size: Fraction for test (default 0.2)

    Returns:
        (train_set, test_set, val_set, test_idx) where each set is (Signals, Labels) tuple
    """
```

---

## Security Note: Model Deserialization

**Issue:** Models are saved/loaded with torch.save/torch.load without version pinning.

**Files:**
- `main.py:198` — `torch.load(..., map_location=cfg.device)`
- `synth_finetune.py` — Similar pattern

**Risk:** Arbitrary code execution if untrusted model weights are loaded (pickle-based format).

**Mitigation already in place:** Code loads from trusted checkpoints directory. But no validation.

**Recommendation:** Use `weights_only=True` in future PyTorch versions:
```python
# PyTorch 1.13+
torch.load(..., weights_only=True)
```

---

## Summary of Priority Issues

| Priority | Issue | Impact | Effort |
|----------|-------|--------|--------|
| **HIGH** | Typo in PYTHONHASHSEED (utils.py:96) | Breaks reproducibility | 1 min |
| **HIGH** | Global variables in model.forward (models/model.py:25) | Thread-unsafe, breaks parallel eval | 15 min |
| **HIGH** | Unclosed file handles (data_loader.py:37, config.py:27) | File descriptor leak under load | 30 min |
| **MEDIUM** | Duplicate class mappings (config.py, data_loader.py) | Sync burden, error-prone | 45 min |
| **MEDIUM** | Missing input validation (file paths, mod filters) | Silent failures, hard to debug | 1 hr |
| **MEDIUM** | Regularization edge case (models/model.py:29) | Potential TypeError in training | 20 min |
| **LOW** | Inefficient dataset split (data_loader.py:107-124) | Slow but not blocking | 1 hr |
| **LOW** | No unit tests (test/) | Regression risk | 3+ hrs |

---

*Concerns audit: 2026-03-31*
