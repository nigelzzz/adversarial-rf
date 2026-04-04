---
phase: 02-experimental-results
plan: 04
subsystem: defense-evaluation
tags: [calibration, defense_compare, classical-filters, cfg-params, json-loading]

# Dependency graph
requires:
  - phase: 01-defense-implementations
    provides: run_calibration_sweep in util/defense_calibrate.py, PARAM_GRIDS for 5 classical filters
  - phase: 02-experimental-results
    plan: 01
    provides: run_defense_compare, generate_confusion_matrices, generate_budget_curves in util/defense_compare.py
provides:
  - Per-SNR calibrated filter parameter loading from calibration_params.json before defense evaluations
  - --mode calibrate_defenses entry point in main.py producing calibration_params.json
  - Auto-detection of most recent calibration_params.json in defense_compare mode
affects: [02-05-experiments, paper-results-generation]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "cfg save/restore pattern: _set_calibrated_params saves originals, _restore_cfg_params restores in finally block"
    - "Lazy JSON loading: _load_calib_params returns empty dict on missing file (no crash, falls back to defaults)"
    - "Auto-detection glob pattern for calibration JSON: inference/*/result/calibration_params.json"

key-files:
  created: []
  modified:
    - util/defense_compare.py
    - main.py

key-decisions:
  - "calibration_path=None falls through to cfg defaults — no regressions when calibration JSON absent"
  - "calib_params loaded once per function call, not per (defense, SNR) — avoids repeated JSON file I/O"
  - "_CALIB_TO_CFG is module-level constant (not per-call) for easy inspection and testing"
  - "Auto-detection picks the last (alphabetically latest) inference/*/result/calibration_params.json when multiple exist"

patterns-established:
  - "Gap 4 fix pattern: _set_calibrated_params / _restore_cfg_params around _apply_defense calls"

requirements-completed: [EVAL-01, EVAL-02, EVAL-05]

# Metrics
duration: 8min
completed: 2026-04-04
---

# Phase 02 Plan 04: Calibration Fix Summary

**Per-SNR calibrated filter params loaded from calibration_params.json and injected into cfg before each classical filter defense evaluation, with --mode calibrate_defenses entry point producing the JSON**

## Performance

- **Duration:** ~8 min
- **Started:** 2026-04-04T15:36:00Z
- **Completed:** 2026-04-04T15:44:13Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments

- Added `_CALIB_TO_CFG` mapping and `_set_calibrated_params`/`_restore_cfg_params`/`_load_calib_params` helpers to `util/defense_compare.py` — covers all 5 classical filters (kalman, wiener, savitzky_golay, gaussian, fir) with correct cfg attribute names
- Added `calibration_path` parameter to `run_defense_compare`, `generate_confusion_matrices`, and `generate_budget_curves`; classical filter calls now wrapped with set/restore
- Added `--mode calibrate_defenses` to `main.py` that runs `run_calibration_sweep` with CW attack using minmax normalization (consistent with D-05), producing `calibration_params.json`
- Added auto-detection of most recent `calibration_params.json` in defense_compare dispatch

## Task Commits

1. **Task 1: Add calibration param loading to defense_compare.py** - `aceb2bb` (feat)
2. **Task 2: Wire --mode calibrate_defenses into main.py and pass calibration_path** - `a52c799` (feat)

**Plan metadata:** (docs commit below)

## Files Created/Modified

- `util/defense_compare.py` — Added `_CALIB_TO_CFG`, `_set_calibrated_params`, `_restore_cfg_params`, `_load_calib_params` helpers; `calibration_path` param on all 3 public functions; calibration load + set/restore wrapping around `_apply_defense` calls in `run_defense_compare` and `generate_budget_curves`
- `main.py` — Added `--calibration_path` argparse arg; added `--mode calibrate_defenses` dispatch block; updated `--mode defense_compare` to auto-detect and pass `calibration_path` to all three evaluation functions

## Decisions Made

- `calibration_path=None` falls through to cfg defaults without error — backward-compatible, no regressions
- `calib_params` loaded once per function call (not per defense call) to avoid redundant file I/O
- `_CALIB_TO_CFG` is module-level for easy inspection and unit testing
- Auto-detection uses `sorted(glob(...))[-1]` — picks alphabetically latest run if multiple calibration JSONs exist

## Deviations from Plan

None - plan executed exactly as written. The worktree branch required a merge from main to pick up previously committed files (util/defense_compare.py, util/defense_calibrate.py, etc.) before work could begin.

## Issues Encountered

- Worktree branch `worktree-agent-aba163cd` was not up to date with main branch — performed `git merge main --no-edit` to bring in all previous plan commits (Plans 01-03) before executing this plan. Fast-forward merge succeeded cleanly.

## Next Phase Readiness

- `util/defense_compare.py` is now calibration-aware — ready for Plan 05 experiment execution
- To use calibrated params: first run `--mode calibrate_defenses` then `--mode defense_compare` (auto-detection picks up JSON automatically)
- No blockers for Plan 05

---
*Phase: 02-experimental-results*
*Completed: 2026-04-04*
