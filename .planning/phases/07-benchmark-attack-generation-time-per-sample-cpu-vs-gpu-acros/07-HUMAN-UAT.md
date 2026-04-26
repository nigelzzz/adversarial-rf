---
status: partial
phase: 07-benchmark-attack-generation-time-per-sample-cpu-vs-gpu-acros
source: [07-VERIFICATION.md]
started: 2026-04-26T18:01:00Z
updated: 2026-04-26T18:01:00Z
---

## Current Test

[awaiting human testing]

## Tests

### 1. Visual inspection of paper/latex/figures/attack_bench_latency.pdf
expected: Grouped bar chart with 5 attack groups (FGSM/PGD/CW/EAD-L1/EAD-EN), 2 bars per group (gray=CPU, blue=GPU), log-scale y-axis labelled 'Latency (ms/sample)', visible std error bars, IEEE-style typography matching v1.0 paper figures.
result: [pending]

### 2. Full-budget regeneration before camera-ready submission
expected: Run `python main.py --mode attack_bench --dataset 2016.10a --ckpt_path ./checkpoint` (defaults N=512, R=5, batch=128, warmup=3), then `cd paper/scripts && python plot_attack_bench_latency.py`. CSV in newest inference dir should show n_samples=512 and n_reps=5; PDF numbers should reflect those values.
result: [pending]

## Summary

total: 2
passed: 0
issues: 0
pending: 2
skipped: 0
blocked: 0

## Gaps
