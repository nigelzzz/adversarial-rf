---
status: passed
phase: 04-adversarial-training
source: [04-VERIFICATION.md]
started: 2026-04-16
updated: 2026-04-16
---

## Current Test

[all tests complete]

## Tests

### 1. Analog class retention after full 30-epoch adversarial training
expected: WBFM, AM-DSB, AM-SSB each retain non-trivial accuracy (>10%) in sanity eval output after running `python adv_train.py --mode train --epochs 30 --batch_size 256` to completion. 2-epoch smoke test showed WBFM=23.8%, AM-DSB=55.7%, AM-SSB=100% (not converged). A full run confirms the analog substitution mechanism prevents catastrophic forgetting.
result: passed
actual:
  - Training completed 30/30 epochs. Best epoch: 28, best weighted metric: 76.6%.
  - Final val clean acc: 91.5%, val FGSM robust acc: 60.6%.
  - Sanity eval on full RML2016.10a test set (all SNRs):
    - WBFM (idx=3): 25.4% — above threshold
    - AM-DSB (idx=6): 51.7% — above threshold
    - AM-SSB (idx=10): 100.0% — above threshold
  - Overall test accuracy: 57.03% (depressed by very low SNRs where any classifier is ~9% random; at SNR >= 0 dB, accuracy is 89.6%–93.3%).
  - Artifacts produced:
    - ./checkpoint/2016.10a_AWN_at.pkl (496K, pure state_dict)
    - ./checkpoint/2016.10a_AWN_at.config.json (all 16 D-16 keys)
    - ./checkpoint/2016.10a_AWN_at_log.csv (31 lines = header + 30 epochs)
  - Training log: training_logs/adv_train_full_20260416_214010.log
  - No catastrophic forgetting. Analog substitution mechanism confirmed working.

## Summary

total: 1
passed: 1
issues: 0
pending: 0
skipped: 0
blocked: 0

## Gaps
