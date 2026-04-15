---
phase: 01-defense-implementations
plan: 03
subsystem: defense
tags: [calibration, latency-benchmark, parameter-sweep, pytorch, scipy, gpu-timing, torch-cuda-event]

# Dependency graph
requires:
  - phase: 01-defense-implementations
    plan: 01
    provides: "Five classical filter functions in util/defense_baselines.py"
  - phase: 01-defense-implementations
    plan: 02
    provides: "DEFENSE_REGISTRY dict and defend() pipeline in util/defense_registry.py"
provides:
  - "PARAM_GRIDS dict with 84 calibration combinations across 5 filters (D-04)"
  - "calibrate_filter() composite-score grid search (alpha * clean_acc + (1-alpha) * defended_acc) (D-02)"
  - "run_calibration_sweep() per-SNR calibration loop saving results to calibration_params.json (D-03)"
  - "run_latency_benchmark() torch.cuda.Event GPU / time.perf_counter CPU timing with warmup (D-11/D-13)"
  - "validate_clean_accuracy() PIPE-03 clean accuracy degradation check with 2% threshold"
affects:
  - evaluation scripts that run comparative defense benchmarks
  - paper results section (latency table, calibrated filter performance)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "GPU timing via _time_gpu_op(): 10 warmup + 50 measure iterations with torch.cuda.Event"
    - "CPU timing via _time_cpu_op(): 3 warmup + 20 measure iterations with time.perf_counter"
    - "Calibration sample cap: max_samples_per_snr=200 to keep CW sweep tractable (Pitfall 5)"
    - "Per-sample latency: divide batch_time_ms by batch_size after loop (D-12)"
    - "cfg.defense attribute temporarily overridden in validate_clean_accuracy(), restored on exit"

key-files:
  created:
    - util/defense_calibrate.py
  modified: []

key-decisions:
  - "Calibration sample cap of 200 per SNR cell to keep CW attack tractable during grid search (Pitfall 5)"
  - "Both batch_size=32 and batch_size=1 benchmarked in run_latency_benchmark for deployment comparison"
  - "validate_clean_accuracy temporarily sets cfg.defense then restores original value after loop"
  - "Detector benchmark skipped gracefully when detector=None (optional dependency)"

patterns-established:
  - "Pattern: _time_gpu_op(fn, n_warmup=10, n_measure=50) / _time_cpu_op(fn, n_warmup=3, n_measure=20) as reusable timing primitives"
  - "Pattern: PARAM_GRIDS construction enforces SG constraint (p < w) at grid build time, not at call time"

requirements-completed: [BASE-07, PIPE-02, PIPE-03]

# Metrics
duration: 3min
completed: 2026-04-01
---

# Phase 01 Plan 03: Defense Calibration Summary

**Parameter calibration sweeps (PARAM_GRIDS with 84 combos), per-SNR composite-score grid search, torch.cuda.Event GPU / time.perf_counter CPU latency benchmark with 10-warmup, and PIPE-03 clean accuracy validation with 2% threshold in util/defense_calibrate.py**

## Performance

- **Duration:** 3 min
- **Started:** 2026-04-01T03:14:18Z
- **Completed:** 2026-04-01T03:17:00Z
- **Tasks:** 1/1
- **Files modified:** 1

## Accomplishments

- `PARAM_GRIDS` has 84 total calibration combinations: kalman(20), wiener(25), savitzky_golay(18), gaussian(6), fir(15)
- `calibrate_filter()` implements composite score `alpha * clean_acc + (1-alpha) * defended_acc` (D-02) with per-SNR granularity (D-03)
- `run_latency_benchmark()` uses `torch.cuda.Event` for GPU-native filters and `time.perf_counter` for CPU filters, with 10 warmup iterations before measurement, at both batch_size=32 and batch_size=1
- `validate_clean_accuracy()` flags any defense where clean accuracy drops >2% (PIPE-03), covers all DEFENSE_REGISTRY entries plus rand_smooth

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement calibration sweep and parameter grids** - `3df1730` (feat)

**Plan metadata:** _(docs commit follows)_

## Files Created/Modified

- `util/defense_calibrate.py` - PARAM_GRIDS, calibrate_filter, run_calibration_sweep, run_latency_benchmark, validate_clean_accuracy with `__all__` exports

## Decisions Made

- Capped calibration val samples at 200 per SNR (`max_samples_per_snr=200`) to prevent CW attack sweep from taking hours; documented as Pitfall 5 from RESEARCH.md
- Both `batch_size=32` (realistic batch) and `batch_size=1` (single-sample latency) are benchmarked automatically in `run_latency_benchmark` to provide both deployment-relevant and single-inference latency figures for the paper
- `validate_clean_accuracy` temporarily overrides `cfg.defense` and restores it on exit to avoid side effects on caller's cfg state
- `_time_gpu_op` uses `n_warmup=10` (vs n_warmup=3 for CPU) because CUDA kernel compilation adds first-call overhead per Pitfall 1

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Known Stubs

None - all five exported functions are fully implemented. `run_calibration_sweep` and `run_latency_benchmark` require a loaded AWN model and dataset to produce results, but the functions themselves are complete.

## Next Phase Readiness

- `util/defense_calibrate.py` is ready for import by evaluation scripts and paper result generation
- `run_calibration_sweep` accepts any `attack_fn` callable — plug in a `torchattacks.CW` instance to run the sweep
- `run_latency_benchmark` produces a dict that can be formatted directly into a paper latency table
- Phase 01 is now complete: defense_baselines.py + defense_registry.py + defense_calibrate.py form the full defense framework

---
*Phase: 01-defense-implementations*
*Completed: 2026-04-01*

## Self-Check: PASSED

- [FOUND] util/defense_calibrate.py
- [FOUND] commit 3df1730 in git log
