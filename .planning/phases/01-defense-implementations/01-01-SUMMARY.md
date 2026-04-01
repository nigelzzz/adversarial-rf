---
phase: 01-defense-implementations
plan: 01
subsystem: defense
tags: [kalman, wiener, savitzky-golay, gaussian, fir, signal-processing, pytorch, scipy, numpy]

# Dependency graph
requires: []
provides:
  - "Five classical filter defense functions in util/defense_baselines.py"
  - "GPU-native Gaussian and FIR filters via depthwise F.conv1d"
  - "CPU-path Kalman, Wiener, Savitzky-Golay filters via numpy/scipy"
  - "Unified [N,2,T] tensor interface with automatic device placement"
affects:
  - 01-02-defense-registry
  - 01-03-defense-calibration

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "GPU-native depthwise conv1d with groups=2 for channel-wise filtering"
    - "CPU/numpy bridge pattern for scipy-based filters with device restoration"
    - "SG constraint enforcement: odd window_length > polyorder + auto-correction"
    - "Reflect padding for all conv1d filters to avoid edge artefacts"

key-files:
  created:
    - util/defense_baselines.py
  modified: []

key-decisions:
  - "GPU-native (F.conv1d) for Gaussian and FIR to avoid CPU roundtrip invalidating latency claims"
  - "Manual scalar NumPy Kalman loop because pykalman/filterpy are not installed"
  - "sg_filter uses vectorized scipy_savgol(axis=-1) operating on full [N,2,T] array"
  - "FIR coefficients computed per call (not cached) to support calibration parameter sweeps"

patterns-established:
  - "Filter signature: def <name>_filter(x: torch.Tensor, *, param1=val, ...) -> torch.Tensor"
  - "All filters preserve shape [N, 2, T] and device placement"
  - "GPU filters: contiguous kernel before F.conv1d; reflect padding for half=kernel//2"
  - "CPU filters: detach().cpu().numpy().astype(float32) at entry, torch.from_numpy().to(device) at exit"

requirements-completed: [BASE-01, BASE-02, BASE-03, BASE-04, BASE-05]

# Metrics
duration: 2min
completed: 2026-04-01
---

# Phase 01 Plan 01: Classical Filter Defense Baselines Summary

**Five classical signal-processing filter baselines (Kalman, Wiener, Savitzky-Golay, Gaussian, FIR) implemented in util/defense_baselines.py with GPU-native depthwise conv1d for Gaussian/FIR and scipy/numpy CPU paths for the others**

## Performance

- **Duration:** 2 min
- **Started:** 2026-04-01T03:08:46Z
- **Completed:** 2026-04-01T03:10:27Z
- **Tasks:** 1/1
- **Files modified:** 1

## Accomplishments

- All five classical filter functions importable and callable with `(x: Tensor, **kwargs) -> Tensor` signature
- GPU-native Gaussian and FIR filters use depthwise `F.conv1d` with `groups=2` and never call `.cpu()` or `.numpy()`
- Savitzky-Golay filter auto-corrects invalid window/polyorder combinations before calling scipy
- All filters verified on both CPU and CUDA tensors returning the same shape and device as input

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement five classical filter defense functions** - `839fe8c` (feat)

**Plan metadata:** _(docs commit follows)_

## Files Created/Modified

- `util/defense_baselines.py` - Five classical filter defense functions with `__all__` exports, docstrings, and `_kalman_1d` helper

## Decisions Made

- FIR coefficients are recomputed per call (not cached) to allow calibration sweeps to vary `cutoff` and `numtaps` without stale state
- Savitzky-Golay uses `max(window_length, polyorder + 2)` then re-enforces odd to guarantee both odd and > polyorder in one pass
- `kernel.contiguous()` called before `F.conv1d` for Gaussian and FIR to avoid PyTorch expand/stride issues on some CUDA versions

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Known Stubs

None - all five filter functions are fully wired with real implementations.

## Next Phase Readiness

- `util/defense_baselines.py` is ready for Plan 02 (defense registry) to import and register all five functions
- Plan 03 (calibration sweep) can call any filter with keyword parameters that vary per sweep iteration
- GPU-native filters validated on CUDA; CPU filters validated on CPU and CUDA (via device restore)

---
*Phase: 01-defense-implementations*
*Completed: 2026-04-01*

## Self-Check: PASSED

- [FOUND] util/defense_baselines.py
- [FOUND] commit 839fe8c in git log
