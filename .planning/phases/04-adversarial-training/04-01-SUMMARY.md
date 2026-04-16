---
phase: 04-adversarial-training
plan: 01
subsystem: adversarial-training
tags: [adversarial-training, dual-batch-loss, analog-substitution, torchattacks]
dependency_graph:
  requires:
    - checkpoint/2016.10a_AWN.pkl
    - data/RML2016.10a_dict.pkl
    - util/adv_attack.py (Model01Wrapper, iq_to_ta_input_minmax, ta_output_to_iq_minmax)
    - data_loader/data_loader.py (Load_Dataset with snr_min)
    - util/utils.py (create_model, fix_seed)
    - util/config.py (Config)
  provides:
    - adv_train.py (adversarial training script with full pipeline)
  affects:
    - Plan 04-02 (adds checkpoint saving, CSV logging, scheduler, eval hook to adv_train.py)
tech_stack:
  added: []
  patterns:
    - dual-batch adversarial training (clean + adv forward passes per batch)
    - per-batch random attack selection (uniform over FGSM/PGD/EADL1/EADEN)
    - analog substitution (WBFM/AM-DSB/AM-SSB kept clean in adversarial stream)
    - minmax normalisation for torchattacks compatibility (iq_to_ta_input_minmax)
key_files:
  created:
    - adv_train.py
  modified: []
decisions:
  - "ead_iters clamped to >= 10 to avoid ZeroDivisionError in torchattacks EADL1/EADEN (iteration % (max_iterations // 10))"
  - "EAD default max_iterations set to 10 (matches D-03 budget with torchattacks minimum)"
  - "val_epoch uses FGSM as proxy robust accuracy metric (fast, representative)"
  - "Analog substitution applied after attack generation (not before) to avoid shape issues"
metrics:
  duration: "~7 minutes (including 1-epoch smoke test)"
  completed: "2026-04-16"
  tasks_completed: 2
  files_created: 1
  files_modified: 0
---

# Phase 4 Plan 01: Adversarial Training Script Scaffold Summary

**One-liner:** Standalone `adv_train.py` with dual-batch AT loop (alpha-weighted clean+adv loss), per-batch random attack selection from FGSM/PGD/EADL1/EADEN, analog modulation substitution, and minmax-normalised torchattacks integration — warm-starts from pretrained AWN checkpoint.

## What Was Built

`adv_train.py` at repo root implements the core adversarial training pipeline:

- **`build_loaders`**: Loads RML2016.10a filtered to SNR >= 0 dB (110K samples), splits 85/15 train/val using `np.random.default_rng(seed).permutation`
- **`make_attacks`**: Instantiates FGSM, PGD (7 steps, alpha=eps/4), EADL1, EADEN with configurable eps=0.1, bss=1 for training speed
- **`generate_adv_batch`**: Applies per-sample minmax normalisation, runs attack via torchattacks, inverts normalisation, substitutes clean signals for analog mods (WBFM=3, AM-DSB=6, AM-SSB=10)
- **`train_epoch`**: Dual-batch forward (clean + adversarial), loss = alpha*L_adv + (1-alpha)*L_clean + sum(regu_adv), per-batch random attack selection via `random.choice(ATTACK_NAMES)`
- **`val_epoch`**: Clean accuracy + FGSM robust accuracy (model switched to train mode for gradient generation, back to eval for inference)
- **`main`**: Argparse CLI, device auto-detection, warm-start from pretrained checkpoint, temporary per-epoch training loop (Plan 02 adds checkpoint saving, CSV, scheduler, early stopping)

## Smoke Test Results

```
Using device: cuda
AT dataset: 93500 train, 16500 val (SNR >= 0 dB)
Warm-started from: ./checkpoint/2016.10a_AWN.pkl
Attacks: ['fgsm', 'pgd', 'eadl1', 'eaden'] | eps=0.1
Ep   1: loss=0.7571 (clean=0.3423, adv=1.1538) | val_clean=89.8% robust=53.2%
```

- Clean accuracy 89.8% (close to base model ~92%) confirms warm-start is working
- FGSM robust accuracy 53.2% (down from ~89.8%) confirms attacks are effective
- Dual-batch loss decomposition is correct: 0.5*1.15 + 0.5*0.34 + regu ≈ 0.76

## Commits

| Task | Description | Commit |
|------|-------------|--------|
| Task 1 | Script scaffold with argparse, data loading, attack factory, training loop | 6b23432 |
| Task 2 | Fix EADL1/EADEN ZeroDivisionError when ead_iters < 10 | 58507cb |

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed ZeroDivisionError in EADL1/EADEN when ead_iters=7**
- **Found during:** Task 2 smoke test
- **Issue:** `torchattacks.EADL1` and `torchattacks.EADEN` compute `iteration % (max_iterations // 10)` internally. When `max_iterations < 10`, this raises `ZeroDivisionError: integer division or modulo by zero`.
- **Fix:** Added `ead_iters_safe = max(ead_iters, 10)` inside `make_attacks()`. Updated argparse default for `--ead_iters` from 7 to 10. The D-03 research recommendation of 7 was written before verifying the torchattacks minimum.
- **Files modified:** `adv_train.py`
- **Commit:** 58507cb

## Known Stubs

None. The temporary training loop (plain epoch iteration without checkpoint saving, CSV logging, or early stopping) is intentional and documented in the source. Plan 02 completes these features.

## Threat Flags

None. This is a local training script with no network endpoints, user-facing interfaces, or external service calls.

## Self-Check: PASSED

- `adv_train.py` exists: FOUND
- Task 1 commit 6b23432: FOUND
- Task 2 commit 58507cb: FOUND
- All 6 required functions present: CONFIRMED (build_loaders, make_attacks, generate_adv_batch, train_epoch, val_epoch, main)
- All 21 acceptance criteria checks: PASS
- 1-epoch smoke test: COMPLETED without error
