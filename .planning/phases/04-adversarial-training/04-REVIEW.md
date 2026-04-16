---
phase: 04-adversarial-training
reviewed: 2026-04-16T11:00:00Z
depth: standard
files_reviewed: 1
files_reviewed_list:
  - adv_train.py
findings:
  critical: 0
  warning: 4
  info: 1
  total: 5
status: issues_found
---

# Phase 4: Code Review Report

**Reviewed:** 2026-04-16T11:00:00Z
**Depth:** standard
**Files Reviewed:** 1
**Status:** issues_found

## Summary

Reviewed `adv_train.py`, a new adversarial training script implementing dual-batch AT with FGSM/PGD/EADL1/EADEN attacks on the AWN model. The code is well-structured with clear separation of concerns (data loading, attack factory, training loop, validation, config persistence). Docstrings are thorough. Four warnings were identified: a dead `--mode eval` code path that silently runs training instead, a potential data leak between the training split and the sanity evaluation test split, a non-reproducible test split in the sanity evaluation, and unused large tensors held in memory during the entire training run. One informational item regarding an unused import.

## Warnings

### WR-01: `--mode eval` accepted but never dispatched -- silently runs training

**File:** `adv_train.py:549`
**Issue:** The argparse definition accepts `choices=['train', 'eval']` at line 549, but `main()` never checks `args.mode`. Regardless of the mode argument, `main()` always proceeds to build data loaders, construct attacks, and execute the full `adv_train()` training loop (lines 592-633). A user running `python adv_train.py --mode eval` would unknowingly start a new training run, potentially overwriting the saved checkpoint.
**Fix:** Add a mode dispatch in `main()`. For example:
```python
if args.mode == 'train':
    # existing training code (lines 620-633)
    ...
elif args.mode == 'eval':
    run_sanity_eval(model, device, args)
```
Alternatively, if `--mode eval` is not yet needed, remove `'eval'` from the choices to avoid user confusion:
```python
parser.add_argument('--mode', choices=['train'], default='train', ...)
```

### WR-02: Data leak between training split and sanity evaluation test split

**File:** `adv_train.py:92-115` and `adv_train.py:502-505`
**Issue:** `build_loaders()` (line 97-104) creates a train/val split using `np.random.default_rng(seed).permutation()` -- a simple random shuffle with no stratification. Later, `run_sanity_eval()` (line 504) calls `Dataset_Split()` which uses a completely different splitting algorithm (stratified by mod/SNR slices using `np.random.choice`). These two splitting strategies are not coordinated, so the "test set" produced by `Dataset_Split` in the sanity eval will likely overlap with the training set used by `build_loaders`. This means the sanity eval accuracy numbers are inflated and unreliable -- they include samples the model was trained on.
**Fix:** Either (a) use `Dataset_Split` consistently for both training and sanity eval, or (b) save the train/val indices from `build_loaders` and derive the test set as the complement. Option (b):
```python
def build_loaders(dataset='2016.10a', batch_size=256, val_ratio=0.15,
                  seed=42, snr_min=0):
    ...
    # Save indices for later use
    return train_loader, val_loader, Signals, Labels, SNRs, train_idx, val_idx
```
Then in `run_sanity_eval`, load the full dataset (all SNRs) and exclude training indices rather than calling `Dataset_Split`.

### WR-03: Non-reproducible test split in sanity evaluation

**File:** `adv_train.py:504`
**Issue:** `Dataset_Split()` uses `np.random.choice` which depends on the numpy global random state. By the time `run_sanity_eval()` is called (after the entire training loop), the numpy global random state has been mutated by many operations (data shuffling, attack generation, etc.). The `fix_seed(args.seed)` call at line 589 sets the initial state, but it is not re-seeded before `run_sanity_eval`. This means the test split produced is non-deterministic and will differ between runs even with the same seed, making sanity eval results non-reproducible.
**Fix:** Re-seed before calling `Dataset_Split` in `run_sanity_eval`:
```python
def run_sanity_eval(model, device, args):
    ...
    np.random.seed(args.seed)  # Reset to canonical state
    _, test_set, _, test_idx = Dataset_Split(
        Signals, Labels, snrs, mods, logger)
```

### WR-04: Large tensors held in memory unnecessarily during training

**File:** `adv_train.py:592`
**Issue:** Line 592 captures `Signals`, `Labels`, `SNRs` from `build_loaders()`, but these variables are never used again in `main()`. Since the DataLoader already holds references to the train/val subsets, the full pre-split tensors (`Signals` is the entire SNR-filtered dataset) remain in memory for the entire multi-epoch training loop. For RML2016.10a with SNR >= 0, this is roughly 110K samples * 2 * 128 * 4 bytes = ~113 MB of wasted GPU/CPU memory.
**Fix:** Discard the unused return values:
```python
train_loader, val_loader, _, _, _ = build_loaders(
    dataset=args.dataset,
    batch_size=args.batch_size,
    val_ratio=0.15,
    seed=args.seed,
    snr_min=0,
)
```
Or better, refactor `build_loaders` to not return the full tensors if they are only needed for external use.

## Info

### IN-01: Unused `subprocess` import can be scoped locally

**File:** `adv_train.py:34`
**Issue:** The `subprocess` module is imported at the top level but only used in `save_config()` (line 453) for a single `git rev-parse` call. While not incorrect, top-level import of `subprocess` is unusual for a training script and slightly obscures the module's dependencies.
**Fix:** Move the import inside `save_config()`:
```python
def save_config(path, args, best_epoch, best_weighted, epochs_trained):
    try:
        import subprocess
        sha = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            capture_output=True, text=True, timeout=5
        ).stdout.strip()
    except Exception:
        sha = 'unknown'
```

---

_Reviewed: 2026-04-16T11:00:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
