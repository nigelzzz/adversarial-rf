---
phase: 01-defense-implementations
plan: 02
subsystem: defense
tags: [pytorch, defense-registry, randomized-smoothing, pipeline, latency, gpu-timing]

# Dependency graph
requires: []
provides:
  - DEFENSE_REGISTRY dict mapping 8 defense names to callables in util/defense_registry.py
  - defend() unified detect->recover->classify pipeline with per-component latency breakdown
  - randomized_smoothing_predict() majority-vote classifier wrapper (k=20, sigma=0.01)
affects:
  - 01-03 (defense calibration will import DEFENSE_REGISTRY and defend())
  - evaluation scripts that run comparative defense benchmarks
  - any future mode that dispatches to defenses by name

# Tech tracking
tech-stack:
  added: []
  patterns:
    - DEFENSE_REGISTRY dict maps string defense names to callable (x: Tensor, **kwargs) -> Tensor
    - rand_smooth dispatches to RS wrapper (not signal filter path)
    - GPU ops timed with torch.cuda.Event; CPU ops with time.perf_counter
    - Per-sample latency via /N division; 3 warmup calls before measurement
    - try/except ImportError allows registry to load even if defense_baselines.py absent

key-files:
  created:
    - util/defense_registry.py
  modified: []

key-decisions:
  - "DEFENSE_REGISTRY uses try/except ImportError for baseline imports so Plan 02 can run before Plan 01 completes (parallel execution)"
  - "rand_smooth sentinel None in registry + explicit dispatch path separates RS from signal filter path (D-10)"
  - "_apply_filter helper extracts filter kwargs from cfg by defense name, keeping defend() clean"
  - "Detector-gated path folds filter latency into detector_ms report; filter_ms=0 in that branch (accurately reflects combined operation)"

patterns-established:
  - "Pattern: DEFENSE_REGISTRY stores callables for filters, None for special-dispatch defenses"
  - "Pattern: defend(x, model, detector, cfg) -> (predictions, latency_breakdown) is the single inference entry point"
  - "Pattern: _warmup(fn, *args, n=3) called before all latency measurements"

requirements-completed: [PIPE-01, PIPE-02, BASE-06]

# Metrics
duration: 2min
completed: 2026-04-01
---

# Phase 01 Plan 02: Defense Registry and Unified Pipeline Summary

**DEFENSE_REGISTRY dict with 8 entries, defend() detect->recover->classify pipeline with GPU/CPU latency breakdown, and randomized smoothing majority-vote classifier wrapper (k=20, sigma=0.01)**

## Performance

- **Duration:** 2 min
- **Started:** 2026-04-01T03:08:53Z
- **Completed:** 2026-04-01T03:10:52Z
- **Tasks:** 1
- **Files modified:** 1

## Accomplishments
- Created `util/defense_registry.py` (460 lines) satisfying all acceptance criteria
- `DEFENSE_REGISTRY` maps 8 defense names: kalman, wiener, savitzky_golay, gaussian, fir, fft_topk, spectral_gated, rand_smooth
- `defend()` composes detector gate -> filter -> classifier with separate `torch.cuda.Event` (GPU) and `time.perf_counter` (CPU) timing per component, per-sample division, and 3 warmup runs
- `randomized_smoothing_predict()` implements k=20 majority vote at sigma=0.01 with OOM fallback to micro-batch

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement randomized smoothing classifier wrapper** - `0178d6d` (feat)

**Plan metadata:** _(TBD — final metadata commit)_

## Files Created/Modified
- `util/defense_registry.py` - DEFENSE_REGISTRY dict, defend() unified pipeline, randomized_smoothing_predict()

## Decisions Made
- Used `try/except ImportError` for `util.defense_baselines` imports so Plan 02 can run in parallel with Plan 01; classical filter entries show `None` until Plan 01 completes and the module becomes importable
- Separated `rand_smooth` into its own dispatch branch inside `defend()` (not the filter branch), matching D-10
- `_apply_filter()` helper extracts filter-specific kwargs from cfg by `defense_name`, avoiding a large if-elif in `defend()` itself
- Detector-gated path times detection + KL computation together (single CUDA event pair), sets `filter_ms=0` since filtering is folded into that path — this accurately reflects the combined operation latency

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
- `util/defense_baselines.py` absent (Plan 01 running in parallel) — handled gracefully via try/except ImportError as specified in the plan. Classical filter entries in registry are `None` until Plan 01 completes.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- `DEFENSE_REGISTRY`, `defend()`, and `randomized_smoothing_predict()` are ready for import by Plan 03 (calibration sweep)
- Once Plan 01 (`util/defense_baselines.py`) is present, all 8 registry entries will be populated
- The `defend()` function's cfg-parameter extraction (`cfg.def_topk`, `cfg.kalman_process_noise`, etc.) is ready for calibrated parameters to be injected by Plan 03

---
*Phase: 01-defense-implementations*
*Completed: 2026-04-01*

## Self-Check: PASSED
- util/defense_registry.py: FOUND
- Commit 0178d6d: FOUND
