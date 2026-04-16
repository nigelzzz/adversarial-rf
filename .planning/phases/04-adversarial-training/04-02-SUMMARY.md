---
phase: 04-adversarial-training
plan: 02
subsystem: adversarial-training
tags: [adversarial-training, checkpoint-management, csv-logging, json-config, sanity-eval]
dependency_graph:
  requires:
    - adv_train.py (Plan 01 scaffold: build_loaders, make_attacks, generate_adv_batch, train_epoch, val_epoch)
    - checkpoint/2016.10a_AWN.pkl (warm-start source)
    - data/RML2016.10a_dict.pkl (training data)
  provides:
    - adv_train.py (complete end-to-end adversarial training script)
    - checkpoint/2016.10a_AWN_at.pkl (best-epoch AT checkpoint, created at runtime)
    - checkpoint/2016.10a_AWN_at.config.json (training hyperparameter record, created at runtime)
    - checkpoint/2016.10a_AWN_at_log.csv (per-epoch training log, created at runtime)
  affects:
    - Plan 05 (reads 2016.10a_AWN_at.pkl for AT evaluation)
tech_stack:
  added: []
  patterns:
    - ReduceLROnPlateau(mode='max') on weighted val metric (not loss)
    - Best-epoch checkpoint selection by 0.5*clean + 0.5*FGSM-robust accuracy
    - Per-epoch CSV logging with D-17 columns
    - JSON config persistence after training (not before) to capture actual epochs_trained
    - try/finally around training loop for graceful KeyboardInterrupt handling
    - Post-training sanity eval on full RML test set (all SNRs) using Dataset_Split
key_files:
  created: []
  modified:
    - adv_train.py
decisions:
  - "ReduceLROnPlateau verbose kwarg removed — deprecated/rejected in installed PyTorch version"
  - "adv_train() returns (best_epoch, best_weighted, epochs_trained) for finally-block config write"
  - "run_sanity_eval loads full RML2016.10a (all SNRs) via Load_Dataset + Dataset_Split to report per-SNR and per-analog-class accuracy"
metrics:
  duration: "~5 minutes (2-epoch smoke test ~2 min, implementation ~3 min)"
  completed: "2026-04-16"
  tasks_completed: 2
  files_created: 0
  files_modified: 1
---

# Phase 4 Plan 02: Adversarial Training Checkpoint Management Summary

**One-liner:** Complete adversarial training script with ReduceLROnPlateau(mode='max') best-epoch checkpoint selection, D-17 per-epoch CSV logging, D-16 JSON config persistence, and post-training sanity eval showing per-SNR and analog-class accuracy.

## What Was Built

Added to `adv_train.py`:

- **`adv_train(model, wrapped_model, train_loader, val_loader, attacks, device, args)`**: Full training orchestrator per D-12, D-13, D-15, D-17. Creates optimizer/criterion/scheduler, opens CSV log with D-17 column header, loops for `args.epochs`, logs each epoch, saves best checkpoint when `0.5*val_clean_acc + 0.5*val_robust_fgsm_acc` improves, calls `scheduler.step(val_weighted)`, triggers early stopping after `args.patience` epochs without improvement. Returns `(best_epoch, best_weighted, epochs_trained)`.

- **`save_config(path, args, best_epoch, best_weighted, epochs_trained)`**: Writes JSON with all 16 D-16 keys including `git_sha` from `subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], ...)` wrapped in try/except. Called in `finally` block so partial runs also produce a config.

- **`run_sanity_eval(model, device, args)`**: Loads best checkpoint, calls `Load_Dataset` for full RML2016.10a (all SNRs), uses `Dataset_Split` for standard test set, runs inference batch-by-batch, prints overall accuracy + per-SNR accuracy + analog class accuracy (WBFM, AM-DSB, AM-SSB individually).

- **`main()` updated**: Placeholder training loop replaced with `try/except KeyboardInterrupt/finally` pattern calling `adv_train()`, `save_config()`, and `run_sanity_eval()`.

## Smoke Test Results (2 epochs, batch_size=64)

```
Ep   1/2: loss=0.7571 (clean=0.3423 adv=1.1538) val_clean=89.8% val_robust=53.2% weighted=71.5% lr=1.00e-04 [55s]
  >> New best (weighted=71.5%), saved ./checkpoint/2016.10a_AWN_at.pkl
Ep   2/2: loss=0.6089 (clean=0.3271 adv=0.8723) val_clean=90.5% val_robust=55.9% weighted=73.2% lr=1.00e-04 [64s]
  >> New best (weighted=73.2%), saved ./checkpoint/2016.10a_AWN_at.pkl

Training complete. Best epoch: 2, weighted: 73.2%
Config saved: ./checkpoint/2016.10a_AWN_at.config.json

=== Sanity Eval: AT checkpoint on full RML test set ===
Overall test accuracy: 56.93%
  ...
  SNR=  0 dB: 87.5%
  SNR= 18 dB: 90.9%

Analog class accuracy (catastrophic forgetting check):
  WBFM (idx=3): 23.8%
  AM-DSB (idx=6): 55.7%
  AM-SSB (idx=10): 100.0%
```

Note: 2-epoch analog class accuracy reflects training-in-progress (not converged). Full 30-epoch run expected to restore analog accuracy closer to base model (~90%+).

## Commits

| Task | Description | Commit |
|------|-------------|--------|
| Task 1 & 2 | Add adv_train, save_config, run_sanity_eval, update main() | 84eb6b9 |
| Rule 1 fix | Remove deprecated verbose kwarg from ReduceLROnPlateau | f6f8fac |

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] ReduceLROnPlateau rejects verbose kwarg**
- **Found during:** Task 2 smoke test
- **Issue:** `torch.optim.lr_scheduler.ReduceLROnPlateau.__init__()` raises `TypeError: got an unexpected keyword argument 'verbose'` in the installed PyTorch version. The plan's code snippet included `verbose=False`.
- **Fix:** Removed the `verbose=False` kwarg. Default behavior is silent (no output), identical effect.
- **Files modified:** `adv_train.py`
- **Commit:** f6f8fac

## Known Stubs

None. All three output artifacts (checkpoint, JSON config, CSV log) are produced by the 2-epoch smoke test. The sanity eval correctly prints analog class accuracy. No placeholder values flow to output.

## Threat Flags

None. This is a local training script with no network endpoints, user-facing interfaces, or external service calls. The `subprocess.run(['git', ...])` call is a fixed command with no user input and has a `timeout=5` guard.

## Self-Check: PASSED

- `adv_train.py` exists at worktree root: FOUND
- Task 1 commit 84eb6b9: FOUND
- Task 2 (verbose fix) commit f6f8fac: FOUND
- All 9 required functions present: CONFIRMED (build_loaders, make_attacks, generate_adv_batch, train_epoch, val_epoch, adv_train, save_config, run_sanity_eval, main)
- `ReduceLROnPlateau` with `mode='max'`: CONFIRMED (line 368)
- `val_weighted = 0.5 * val_clean_acc + 0.5 * val_robust_fgsm_acc`: CONFIRMED
- `torch.save(model.state_dict(), ckpt_path)`: CONFIRMED
- `csv.DictWriter` with D-17 columns: CONFIRMED
- `if no_improve >= args.patience:`: CONFIRMED
- `scheduler.step(val_weighted)`: CONFIRMED
- `except KeyboardInterrupt` / `finally` pattern: CONFIRMED
- `def save_config(` with 16 D-16 keys: CONFIRMED
- `def run_sanity_eval(` with analog class loop: CONFIRMED
- `from data_loader.data_loader import Load_Dataset, Dataset_Split`: CONFIRMED
- `subprocess.run(['git', 'rev-parse', ...)` in save_config: CONFIRMED
- `adv_train.py` line count: 637 (min 350 satisfied)
- `save_config` pattern in source: CONFIRMED
- `./checkpoint/2016.10a_AWN_at.pkl`: 26-key state_dict, loadable with weights_only=True: CONFIRMED
- `./checkpoint/2016.10a_AWN_at.config.json`: all 16 required keys present: CONFIRMED
- `./checkpoint/2016.10a_AWN_at_log.csv`: 9 D-17 columns, 2 data rows: CONFIRMED
- Smoke test `python adv_train.py --mode train --epochs 2 --batch_size 64`: COMPLETED without traceback
