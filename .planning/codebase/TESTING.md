# Testing Patterns

**Analysis Date:** 2026-03-31

## Test Framework

**Status:** No formal unit test framework (pytest, unittest, vitest) configured.

**Testing Approach:** Manual validation scripts. Tests are executable Python scripts that:
- Load a model and dataset
- Run an operation (training, adversarial attack, defense)
- Print results to stdout
- Optionally save results to `.pt` files or CSVs

**Run Tests:**
```bash
# Individual test script
python test_2016_10a.py

# With custom parameters
python test_cw_recovery.py --dataset 2016.10a --ckpt_path ./checkpoint/2016.10a_AWN.pkl \
  --defense fft_topk --def_topk 50

# Experiment script (more complex validation)
python crc_experiment.py
python synth_finetune.py --mode finetune --curriculum
```

**No automated test runners:** Tests must be executed manually and results inspected by human.

## Test File Organization

**Location:** Root directory (co-located with main.py)

**Naming Pattern:**
- `test_*.py`: Unit-level validation of specific features
  - `test_2016_10a.py` — CW attack + recovery on RML2016.10a
  - `test_cw_recovery.py` — CW with optional detector-gated defense
  - `test_cw_params.py` — Parameter sweep to find effective CW settings
  - `test_cw_snr0.py` — Adversarial test at SNR=0 dB only
  - `test_spectral_gate.py` — FFT notch-based spectral gating
  - `test_unified_defense.py` — Multiple defense modes compared

- `*_experiment.py`: Multi-step workflows combining training, attack, and defense
  - `crc_experiment.py` — Synthetic data generation + finetuning + CRC defense evaluation
  - `adaptive_k_experiment.py` — Adaptive K computation across SNR/modulation grid
  - `burst_length_experiment.py` — Test robustness to variable signal lengths

- `plot_*.py`: Visualization scripts (analysis, not validation)
  - `plot_iq_constellation.py` — IQ scatter plots of clean vs adversarial
  - `plot_all_attacks_iq.py` — Grid of IQ distributions for multiple attacks

**Standalone Scripts:**
- No shared test fixtures or test utils imported across scripts
- Each script: self-contained imports, dataset loading, model creation

## Test Structure

**Pattern: Linear validation flow**

```python
#!/usr/bin/env python
"""Clear docstring explaining what test demonstrates."""

import torch
import numpy as np
from tqdm import tqdm

# 1. Configuration (hardcoded or via argparse)
SEED = 42
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_SAMPLES = 500

# 2. Setup
print('='*70)
print('Test: CW Attack + Recovery')
print('='*70)

torch.manual_seed(SEED)
np.random.seed(SEED)

# 3. Load dependencies
from util.config import Config
from util.utils import create_AWN_model
from data_loader.data_loader import Load_Dataset, Dataset_Split

# 4. Load model
cfg = Config('2016.10a', train=False)
model = create_AWN_model(cfg)
ckpt = torch.load('./checkpoint/2016.10a_AWN.pkl', map_location=DEVICE)
model.load_state_dict(ckpt)
model.eval()

# 5. Load data
Signals, Labels, SNRs, snrs, mods = Load_Dataset('2016.10a', None)
# Apply filtering if needed
mask = SNRs >= 0  # SNR >= 0 dB threshold
Signals_filt = Signals[mask]
Labels_filt = Labels[mask]

# Sample subset for faster iteration
indices = np.random.choice(len(Signals_filt), min(NUM_SAMPLES, len(Signals_filt)), replace=False)
X_test = Signals_filt[indices]
y_test = Labels_filt[indices]

# 6. Validate baseline (clean accuracy)
print('[1/N] Measuring clean accuracy...')
with torch.no_grad():
    logits, _ = model(X_test)
    clean_preds = torch.argmax(logits, dim=1)
clean_acc = (clean_preds == y_test).float().mean().item() * 100
print(f'  Clean: {clean_acc:.2f}%')

# 7. Run test operation (e.g., adversarial attack)
print('[2/N] Generating adversarial examples...')
from util.adv_attack import Model01Wrapper
import torchattacks

wrapped = Model01Wrapper(model).eval()
atk = torchattacks.CW(wrapped, c=1.0, steps=100)

adv_all = []
batch_size = 50
for i in tqdm(range(0, len(X_test), batch_size)):
    batch_x = X_test[i:i+batch_size]
    batch_y = y_test[i:i+batch_size]
    batch_x01_4d = iq_to_ta_input(batch_x)
    adv01_4d = atk(batch_x01_4d, batch_y)
    adv = ta_output_to_iq(adv01_4d)
    adv_all.append(adv)

adv_all = torch.cat(adv_all, dim=0)

# 8. Validate attack effectiveness
print('[3/N] Measuring adversarial accuracy...')
with torch.no_grad():
    logits_adv, _ = model(adv_all)
    adv_preds = torch.argmax(logits_adv, dim=1)
adv_acc = (adv_preds == y_test).float().mean().item() * 100
print(f'  After attack: {adv_acc:.2f}%')

# 9. Run recovery/defense
print('[4/N] Applying FFT Top-K recovery...')
from util.defense import fft_topk_denoise, normalize_iq_data, denormalize_iq_data

rec_all = []
for i in tqdm(range(0, len(adv_all), batch_size)):
    batch = adv_all[i:i+batch_size]
    norm = normalize_iq_data(batch, 0.02, 0.04)
    filt = fft_topk_denoise(norm, topk=50)
    rec = denormalize_iq_data(filt, 0.02, 0.04)
    rec_all.append(rec)

rec_all = torch.cat(rec_all, dim=0)

# 10. Validate recovery
print('[5/N] Measuring recovered accuracy...')
with torch.no_grad():
    logits_rec, _ = model(rec_all)
    rec_preds = torch.argmax(logits_rec, dim=1)
rec_acc = (rec_preds == y_test).float().mean().item() * 100
print(f'  After recovery: {rec_acc:.2f}%')

# 11. Report results
print()
print('='*70)
print(f'RESULTS ({len(y_test)} samples)')
print('='*70)
print(f'  Clean:       {clean_acc:6.2f}%')
print(f'  After CW:    {adv_acc:6.2f}%  (↓ {clean_acc - adv_acc:.2f}%)')
print(f'  Recovered:   {rec_acc:6.2f}%  (↑ {rec_acc - adv_acc:+.2f}%)')
print('='*70)

# 12. Optional: Save results for batch validation
results = {
    'clean_acc': clean_acc,
    'adv_acc': adv_acc,
    'recovered_acc': rec_acc,
    'num_samples': len(y_test),
}
torch.save(results, 'test_results.pt')
```

**Key Patterns:**
1. Banner (printouts with `=` dividers) to mark test sections
2. Numbered progress markers: `[1/N]`, `[2/N]` etc.
3. Per-section result reporting
4. Final summary table with key metrics
5. Optional result saving for later comparison

## Mocking

**Strategy:** No mocking framework used (no pytest-mock, unittest.mock)

**When tests interact with real dependencies:**
- **Model checkpoint:** Loads actual pretrained `.pkl` file (not mocked)
- **Dataset:** Loads actual pickle/HDF5 files from `./data/`
- **Attacks/defenses:** Uses real torchattacks library or internal implementations

**Test isolation:**
- Achieved via filesystem isolation (separate config dirs per run)
- Each test loads its own model checkpoint and dataset subset
- No shared state between test scripts (each runs independently)

**What NOT to test:**
- Individual PyTorch layer operations (rely on PyTorch's own testing)
- File I/O edge cases (rely on Python's pickle/h5py)
- Logging output (only check terminal prints or log files manually)

## Fixtures and Factories

**Test Data:**
- No dedicated fixture files
- Use dataset splits from `data_loader/data_loader.py:Dataset_Split()`
- Apply SNR/modulation filters at load time:
  ```python
  # Example from test_2016_10a.py
  mask = snr_vals >= SNR_THRESHOLD  # SNR >= 0 dB
  X_filt = X[mask]
  lbl_filt = lbl[mask]

  # Random sample for reproducibility
  np.random.seed(42)
  indices = np.random.choice(len(X_filt), min(NUM_SAMPLES, len(X_filt)), replace=False)
  X_test = X_filt[indices]
  ```

**Factory Functions:**
- Use `util/utils.py` factories: `create_AWN_model()`, `create_VTCNN2_model()`, etc.
- Config via `util/config.py:Config(dataset, train=False)`

## Coverage

**Requirements:** None enforced

**View Coverage:** Not applicable (no test runner)

**Implicit coverage areas:**
- **Training pipeline:** Tested via `main.py --mode train`
- **Evaluation:** Tested via `main.py --mode eval` and `test_*.py` scripts
- **Adversarial attacks:** Tested via `test_cw_*.py` and `*_experiment.py`
- **Defenses:** Tested via `test_spectral_gate.py`, `test_unified_defense.py`
- **Synthetic data:** Tested via `synth_finetune.py --mode gen`

**Known gaps:**
- No edge case testing (what if SNR filter matches zero samples?)
- No error recovery testing (missing checkpoint files, corrupted data)
- No numeric stability testing (NaN/Inf checks on losses)

## Test Types

**Validation Scripts (standalone executables):**
- Purpose: Verify a feature works end-to-end
- Run: `python test_name.py [--optional-args]`
- Output: Console printouts + optional `.pt`/`.csv` files
- Example: `test_2016_10a.py` validates CW attack + recovery

**Experiment Scripts (workflows):**
- Purpose: Generate results for paper/reports; combine multiple steps
- Run: `python experiment_name.py [--optional-args]`
- Output: Results CSVs, plots (in `inference/` or `training/` dirs)
- Example: `crc_experiment.py` does data gen + finetune + eval

**Quick Sanity Checks:**
- Often run at top-level in `main.py` via mode flags
- Example: `python main.py --mode train --dataset 2016.10a --eval_limit 100` (limit to 100 samples per SNR for fast iteration)

## Common Patterns

**Async/Batching:**
```python
# Test processes data in batches to fit in memory
batch_size = 50
for i in tqdm(range(0, len(X_test), batch_size)):
    batch_x = X_test[i:i+batch_size].to(device)
    batch_y = y_test[i:i+batch_size].to(device)
    # Process batch
    with torch.no_grad():
        logits, _ = model(batch_x)
```

**Error Testing:**
- Not formalized; rely on try/except to catch runtime errors
- Example: If dataset file missing, pickle.load() fails with FileNotFoundError
- Tests validate happy path only; errors cause immediate failure

**Seeding:**
- Fixed seed for reproducibility: `torch.manual_seed(42)`, `np.random.seed(42)`
- Called at start of each test script
- Ensures consistent sampling and attack generation across runs

**Per-SNR Breakdown:**
```python
# Common in tests to validate robustness across conditions
unique_snrs = sorted(np.unique(snr_test))
for snr_val in unique_snrs:
    mask = snr_test == snr_val
    mask_t = torch.from_numpy(mask)

    c = (clean_preds[mask_t] == y_test[mask_t]).float().mean().item() * 100
    a = (adv_preds[mask_t] == y_test[mask_t]).float().mean().item() * 100
    r = (rec_preds[mask_t] == y_test[mask_t]).float().mean().item() * 100

    print(f'{snr_val:>4.0f}   {c:>6.2f}%    {a:>6.2f}%      {r:>6.2f}%')
```

## Test Execution Examples

**Quick validation (2-3 min):**
```bash
python test_2016_10a.py
```
Output: Clean/adversarial/recovered accuracy on 500 samples (SNR ≥ 0).

**Parameter sweep (30-60 min):**
```bash
python test_cw_params.py
```
Output: Parameter sweep results in JSON, showing which CW settings are most effective.

**Full pipeline (2-4 hours):**
```bash
python synth_finetune.py --mode finetune --curriculum --n_per_cell 2000
```
Output: Pretrained + synthetic data finetuning, final model checkpoint, per-SNR accuracy table.

**Experiment with custom attacks (30 min):**
```bash
python main.py --mode multi_attack_eval --dataset 2016.10a --ckpt_path ./checkpoint \
  --attack_list "fgsm,pgd,cw,deepfool" --ta_box minmax --attack_eps 0.1
```
Output: CSV with per-attack/per-SNR/per-modulation accuracy breakdown.

---

*Testing analysis: 2026-03-31*
