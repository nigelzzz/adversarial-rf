---
phase: 02-experimental-results
plan: 02
subsystem: evaluation
tags: [confusion-matrix, sklearn, numpy, pandas, defense-compare, adversarial-evaluation]

# Dependency graph
requires:
  - phase: 02-experimental-results
    plan: 01
    provides: run_defense_compare(), create_attack(), ATTACKS, SNR_POINTS, DEFENSE_CONFIGS in util/defense_compare.py
  - phase: 01-defense-implementations
    provides: defend() unified pipeline in util/defense_registry.py, DEFENSE_REGISTRY
provides:
  - generate_confusion_matrices() function in util/defense_compare.py
  - CONFMAT_ATTACKS=['cw','eadl1','eaden'] and CONFMAT_SNRS=[0,10,18] constants
  - 18 confusion matrix .npy files (3 attacks x 3 SNRs x before/after) saved to defense_compare/confmat/
  - Row-normalized percentage CSVs and confmat_summary.csv for Phase 3 rendering
  - --skip_confmat flag for fast iteration without regenerating matrices
affects: [03-paper-figures, phase-3-latex-rendering]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "cfg override pattern: save orig_defense, set cfg.defense='fft_topk', call defend(), restore"
    - "Row normalization: cm / row_sums.clip(min=1) * 100.0 for per-row percentages"
    - "Byte-string key decode: k.decode() if isinstance(k, bytes) else str(k) for cfg.classes"

key-files:
  created: []
  modified:
    - util/defense_compare.py
    - main.py

key-decisions:
  - "generate_confusion_matrices() saves raw .npy for Phase 3 rendering and row-normalized CSVs for inspection"
  - "After-defense condition uses defend() with temporary cfg.defense='fft_topk' override, consistent with ae_fft_topk dispatch in _apply_defense()"
  - "--skip_confmat flag allows skipping confmat generation when only defense_compare table is needed"

patterns-established:
  - "Cfg override pattern: save/set/restore cfg attributes around defense calls to reuse unified pipeline"

requirements-completed: [EVAL-03]

# Metrics
duration: 2min
completed: 2026-04-02
---

# Phase 2 Plan 02: Confusion Matrix Generation Summary

**generate_confusion_matrices() added to util/defense_compare.py producing 18 raw .npy and row-normalized CSV confusion matrices (3 optimization attacks x 3 SNRs x before/after FFT Top-K defense) for Phase 3 figure rendering**

## Performance

- **Duration:** 2 min
- **Started:** 2026-04-02T10:55:12Z
- **Completed:** 2026-04-02T10:57:30Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments
- Added CONFMAT_ATTACKS=['cw','eadl1','eaden'] and CONFMAT_SNRS=[0,10,18] constants (D-12, D-13)
- Implemented generate_confusion_matrices() generating 18 full 11x11 matrices with row-normalized percentages (D-14, D-15, D-16)
- Each matrix saved as raw .npy for Phase 3 rendering plus row-normalized _pct.csv for inspection
- confmat_summary.csv captures diagonal accuracy for all 18 matrices as quick reference
- Wired into --mode defense_compare in main.py with --skip_confmat for fast iteration

## Task Commits

Each task was committed atomically:

1. **Task 1: Add generate_confusion_matrices() to util/defense_compare.py** - `675abc0` (feat)
2. **Task 2: Wire confusion matrix generation into --mode defense_compare in main.py** - `c7892b2` (feat)

## Files Created/Modified
- `util/defense_compare.py` - Added CONFMAT_ATTACKS, CONFMAT_SNRS constants and generate_confusion_matrices() function; updated __all__
- `main.py` - Added --skip_confmat argparse flag; added generate_confusion_matrices() call after run_defense_compare() in defense_compare mode

## Decisions Made
- Raw .npy files preserved alongside _pct.csv files so Phase 3 can apply its own color scaling and styling to heatmaps
- cfg.defense override pattern (save/set/restore) used to reuse defend() pipeline for "after" condition, consistent with _apply_defense() ae_fft_topk branch
- --skip_confmat flag allows CI-style runs that only need the defense comparison table without re-running slow optimization attacks for confusion matrices

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- All 18 confusion matrix .npy files will be available at defense_compare/confmat/ after --mode defense_compare runs
- Phase 3 (paper figures) can load these directly with np.load() and render as seaborn/matplotlib heatmaps
- confmat_summary.csv provides quick accuracy overview without loading all .npy files

---
*Phase: 02-experimental-results*
*Completed: 2026-04-02*
