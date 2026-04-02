---
phase: 02-experimental-results
plan: 01
subsystem: evaluation
tags: [torchattacks, defense, evaluation, csv, adversarial, fft, kalman, wiener, savitzky-golay, gaussian, fir, randomized-smoothing]

# Dependency graph
requires:
  - phase: 01-defense-implementations
    provides: DEFENSE_REGISTRY, defend(), randomized_smoothing_predict(), classical filter baselines

provides:
  - util/defense_compare.py with run_defense_compare() — iterates 9 defenses x 5 attacks x 10 SNR points
  - --mode defense_compare CLI entry point in main.py
  - Per-attack pivot CSV tables (defense_compare_{attack}.csv)
  - Full results CSV (defense_compare.csv)

affects: [03-paper-writing, confusion-matrix-plan, budget-curve-plan]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - defense_compare pattern: attack generation with minmax normalization -> per-defense application -> per-SNR accuracy CSV
    - _apply_defense dispatcher: no_defense / ae_fft_topk / rand_smooth / classical filter branches
    - per-modulation sub-sampling at each SNR (max_per_cell=200 cap)

key-files:
  created:
    - util/defense_compare.py
  modified:
    - main.py

key-decisions:
  - "minmax normalization (ta_box=minmax) for all torchattacks calls — consistent with D-05 and Phase 1 calibration"
  - "ae_fft_topk branch calls defend() pipeline, temporarily overriding cfg.defense to 'fft_topk'"
  - "max_per_cell=200 per modulation (not total) at each SNR — matches Phase 1 calibration cap (D-04)"
  - "create_attack() for defense_compare duplicated from sigguard_eval.py to limit to the 5 paper attacks (D-01)"

patterns-established:
  - "Defense application: _apply_defense(defense_name, x_adv, model, detector, cfg) -> numpy predictions"
  - "Per-SNR evaluation: filter test_idx by SNRs array, sub-sample per-class, batch-generate adversarial, apply defenses"
  - "CSV output: full flat CSV + per-attack pivot saved to result_dir/defense_compare/"

requirements-completed: [EVAL-01, EVAL-02, EVAL-05]

# Metrics
duration: 8min
completed: 2026-04-02
---

# Phase 02 Plan 01: Defense Compare Summary

**Defense comparison evaluation framework: 9 defenses x 5 attacks x 10 SNR points with minmax torchattacks and per-modulation 200-sample cap, outputting per-attack pivot CSVs**

## Performance

- **Duration:** ~8 min
- **Started:** 2026-04-02T10:48:25Z
- **Completed:** 2026-04-02T10:52:30Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments

- Created `util/defense_compare.py` with `run_defense_compare()` covering all 9 defenses (no_defense, ae_fft_topk, spectral_gated, kalman, wiener, savitzky_golay, gaussian, fir, rand_smooth) against 5 paper attacks at 10 SNR points
- Implemented `_apply_defense()` with correct dispatch branches for each defense type (unified pipeline / classical filters / rand_smooth / no-defense)
- Wired `--mode defense_compare` into main.py with `--max_per_cell`, `--attack_list`, and optional `--detector_ckpt` parameters

## Task Commits

1. **Task 1: Implement util/defense_compare.py** - `c6dd2e9` (feat)
2. **Task 2: Wire --mode defense_compare into main.py** - `a7d3eac` (feat)

**Plan metadata:** (in this commit)

## Files Created/Modified

- `/home/nigel/opensource/adversarial-rf/util/defense_compare.py` — Defense comparison core: run_defense_compare(), ATTACKS, SNR_POINTS, DEFENSE_CONFIGS, create_attack(), _apply_defense(), _get_filter_kwargs()
- `/home/nigel/opensource/adversarial-rf/main.py` — Added --max_per_cell argparse arg and elif args.mode == 'defense_compare' dispatch block

## Decisions Made

- `ae_fft_topk` dispatch temporarily overrides `cfg.defense = 'fft_topk'` before calling `defend()`, restores original value after — cleanest way to reuse the existing unified pipeline
- `max_per_cell=200` applies per modulation class (not total) at each SNR, consistent with Phase 1 calibration sample cap (D-04)
- `create_attack()` in defense_compare.py is a focused duplicate of the one in sigguard_eval.py, restricted to the 5 paper attacks; avoids importing the larger function with 17 attacks

## Deviations from Plan

None — plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- `--mode defense_compare` is ready to run once `checkpoint/2016.10a_AWN.pkl` is available
- Optional: `--detector_ckpt checkpoint/detector_ae.pth` for ae_fft_topk defense
- Plan 02 (confusion matrices) and Plan 03 (budget curves) can now reuse `create_attack()` and `_apply_defense()` from `util/defense_compare.py`
- Concern: CW/EAD attacks are slow at 200 samples/cell x 10 SNRs x 5 attacks — full run may take 1-2 hours on GPU

## Self-Check

Files exist:
- util/defense_compare.py: FOUND
- main.py (modified): FOUND

Commits exist:
- c6dd2e9: FOUND
- a7d3eac: FOUND

## Self-Check: PASSED

---
*Phase: 02-experimental-results*
*Completed: 2026-04-02*
