---
phase: 02-experimental-results
verified: 2026-04-02T11:06:28Z
status: gaps_found
score: 2/4 success criteria verified
gaps:
  - truth: "CSV files with defense comparison results exist (9 defenses x 5 attacks x 10 SNRs)"
    status: failed
    reason: "Experiments have not been run. No defense_compare.csv or defense_compare_*.csv files exist anywhere under inference/. Infrastructure is fully implemented but --mode defense_compare has never been executed."
    artifacts:
      - path: "inference/<dataset>_*/result/defense_compare/defense_compare.csv"
        issue: "File does not exist — command must be run to produce it"
      - path: "inference/<dataset>_*/result/defense_compare/defense_compare_cw.csv"
        issue: "Per-attack pivot tables do not exist"
    missing:
      - "Run: python main.py --mode defense_compare --dataset 2016.10a --ckpt_path ./checkpoint"
      - "Per-attack pivot CSVs: defense_compare_cw.csv, defense_compare_eadl1.csv, defense_compare_eaden.csv, defense_compare_fgsm.csv, defense_compare_pgd.csv"

  - truth: "18 confusion matrix .npy files exist (3 attacks x 3 SNRs x before/after)"
    status: failed
    reason: "No .npy or _pct.csv confusion matrix files exist. The generate_confusion_matrices() function is implemented but the comparison run that triggers it has never been executed."
    artifacts:
      - path: "inference/<dataset>_*/result/defense_compare/confmat/cw_snr0_before.npy"
        issue: "Does not exist"
      - path: "inference/<dataset>_*/result/defense_compare/confmat/confmat_summary.csv"
        issue: "Does not exist"
    missing:
      - "Run --mode defense_compare (without --skip_confmat) to produce 18 .npy + 18 _pct.csv files"

  - truth: "Perturbation budget curve CSVs exist (8 Linf eps points, 4 c values)"
    status: failed
    reason: "budget_curves_detail.csv, budget_curves_agg.csv, and per-attack pivot CSVs do not exist. generate_budget_curves() is implemented but not yet executed."
    artifacts:
      - path: "inference/<dataset>_*/result/defense_compare/budget_curves/budget_curves_detail.csv"
        issue: "Does not exist"
      - path: "inference/<dataset>_*/result/defense_compare/budget_curves/budget_curves_agg.csv"
        issue: "Does not exist"
    missing:
      - "Run --mode defense_compare (without --skip_budget) to produce budget_curves_detail.csv, budget_curves_agg.csv, and 5 per-attack pivot CSVs"

  - truth: "Calibrated best-parameters from Phase 1 are used in defense comparison evaluations (BASE-07 fulfillment)"
    status: failed
    reason: "defense_compare.py does not import PARAM_GRIDS or load calibration_params.json from Phase 1. The plan required 'from util.defense_calibrate import PARAM_GRIDS' and using calibrated parameters. Instead, _get_filter_kwargs() uses hardcoded defaults (e.g., gaussian_sigma=1.0, sg_window_length=11). The comparison table will use uncalibrated parameters, potentially understating classical filter performance."
    artifacts:
      - path: "util/defense_compare.py"
        issue: "No import of PARAM_GRIDS or calibration_params.json loading. _get_filter_kwargs() uses getattr(cfg, 'gaussian_sigma', 1.0) style defaults, not Phase 1 calibrated best-params."
    missing:
      - "Load calibration_params.json (produced by --mode calibrate in Phase 1) into cfg before running comparison"
      - "Or: import PARAM_GRIDS from util.defense_calibrate and use best params per filter per Phase 1 calibration results"
      - "This is a fairness concern: paper claims to compare calibrated baselines but uses default params"

human_verification:
  - test: "Run --mode defense_compare on actual data and verify output"
    expected: "CSV files produced in inference/<dataset>_*/result/defense_compare/ with 9 defense rows, 5 attack columns, 10 SNR + weighted_avg columns per pivot table"
    why_human: "Requires pretrained model checkpoint and RML2016.10a dataset, plus 1-2 hours GPU time for full CW/EAD attacks"
  - test: "Verify unified pipeline (ae_fft_topk) outperforms classical filters on CW and EAD"
    expected: "ae_fft_topk row has highest accuracy on CW and EAD attacks in comparison tables"
    why_human: "Requires actual experimental results to exist — cannot be verified without running the experiments"
---

# Phase 2: Experimental Results Verification Report

**Phase Goal:** All numerical results needed for paper tables and figures exist as validated CSV files
**Verified:** 2026-04-02T11:06:28Z
**Status:** gaps_found
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|---------|
| 1 | `--mode defense_compare` produces CSV with 9 defenses x 5 attacks x 10 SNRs | ✗ FAILED | Infrastructure exists (util/defense_compare.py:run_defense_compare verified), but command has never been run — no CSV files exist in any inference/ directory |
| 2 | 18 confusion matrix .npy files exist (3 attacks x 3 SNRs x before/after) | ✗ FAILED | generate_confusion_matrices() is implemented and wired, but no .npy or _pct.csv files exist anywhere |
| 3 | Perturbation budget curve CSVs exist (8 Linf eps, 4 c values per attack) | ✗ FAILED | generate_budget_curves() is implemented and wired, but budget_curves_detail.csv and budget_curves_agg.csv do not exist |
| 4 | Unified pipeline outperforms classical filters on CW and EAD (SC-4) | ? UNCERTAIN | Cannot verify without actual result data — requires human to run experiments |

**Score:** 0/4 success criteria satisfied (experimental data does not exist)

**Infrastructure Score:** 4/4 evaluation functions implemented, all constants correct, all wiring verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `util/defense_compare.py` | Core evaluation framework | ✓ VERIFIED | 1081 lines, all 3 functions implemented, all constants match plan exactly |
| `main.py` | `--mode defense_compare` dispatch | ✓ VERIFIED | Lines 555-608: full dispatch with model load, detector load, attack_list parse, all 3 function calls |
| `inference/*/result/defense_compare/defense_compare.csv` | Full results matrix | ✗ MISSING | File does not exist — experiments not run |
| `inference/*/result/defense_compare/defense_compare_cw.csv` | Per-attack pivot (CW) | ✗ MISSING | File does not exist |
| `inference/*/result/defense_compare/defense_compare_eadl1.csv` | Per-attack pivot (EAD-L1) | ✗ MISSING | File does not exist |
| `inference/*/result/defense_compare/defense_compare_eaden.csv` | Per-attack pivot (EAD-EN) | ✗ MISSING | File does not exist |
| `inference/*/result/defense_compare/defense_compare_fgsm.csv` | Per-attack pivot (FGSM) | ✗ MISSING | File does not exist |
| `inference/*/result/defense_compare/defense_compare_pgd.csv` | Per-attack pivot (PGD) | ✗ MISSING | File does not exist |
| `inference/*/result/defense_compare/confmat/cw_snr0_before.npy` | Confusion matrix .npy | ✗ MISSING | File does not exist |
| `inference/*/result/defense_compare/confmat/confmat_summary.csv` | Confmat summary | ✗ MISSING | File does not exist |
| `inference/*/result/defense_compare/budget_curves/budget_curves_detail.csv` | Budget curve detail | ✗ MISSING | File does not exist |
| `inference/*/result/defense_compare/budget_curves/budget_curves_agg.csv` | Budget curve aggregated | ✗ MISSING | File does not exist |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `util/defense_compare.py` | `util/defense_registry.py` | `from util.defense_registry import DEFENSE_REGISTRY, defend, randomized_smoothing_predict` | ✓ WIRED | Lines 44-49; defend() called at line 262, 715; DEFENSE_REGISTRY used at line 278 |
| `util/defense_compare.py` | `util/sigguard_eval.py` | create_attack reuse pattern | ✓ WIRED | Plan allowed duplication; create_attack() is self-contained in defense_compare.py with pattern matched |
| `util/defense_compare.py` | `util/defense_calibrate.py` | `from util.defense_calibrate import PARAM_GRIDS` | ✗ NOT_WIRED | Link is absent. `_get_filter_kwargs()` uses hardcoded cfg defaults, not Phase 1 calibrated parameters. calibration_params.json is never loaded. |
| `main.py` | `util/defense_compare.py` | `from util.defense_compare import run_defense_compare` | ✓ WIRED | Line 556; also generates_confusion_matrices line 583, generate_budget_curves line 597 |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|--------------|--------|--------------------|--------|
| `run_defense_compare()` | `preds` (per defense), `acc` (accuracy_score) | Real model inference via `_apply_defense()` → `model(x_adv).argmax` | Yes — real model calls, real attack generation | ✓ FLOWING (infrastructure) |
| `generate_confusion_matrices()` | `cm_before`, `cm_after` | `confusion_matrix(labs_np, preds)` | Yes — sklearn confusion_matrix on real predictions | ✓ FLOWING (infrastructure) |
| `generate_budget_curves()` | `results` rows with accuracy | `_run_attack_snr()` inner function with real attack generation | Yes — same data path as run_defense_compare | ✓ FLOWING (infrastructure) |
| `defense_compare.csv` | All rows | `df_full.to_csv()` after real eval loop | N/A — file never created | ✗ DISCONNECTED — never produced |
| `budget_curves_detail.csv` | All rows | `df.to_csv()` after real eval loop | N/A — file never created | ✗ DISCONNECTED — never produced |

Note: The code paths that produce CSV data are correct and complete. The data flow is disconnected only because the experiments have not been run, not because the implementation is hollow.

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| defense_compare.py importable | `python3 -c "from util.defense_compare import run_defense_compare, ATTACKS, SNR_POINTS, DEFENSE_CONFIGS"` | All imported successfully | ✓ PASS |
| ATTACKS constant exact match | `assert ATTACKS == ['cw','eadl1','eaden','fgsm','pgd']` | Passes | ✓ PASS |
| SNR_POINTS exact match | `assert SNR_POINTS == [0,2,4,6,8,10,12,14,16,18]` | Passes | ✓ PASS |
| DEFENSE_CONFIGS has 9 keys | `assert len(DEFENSE_CONFIGS) == 9` | Passes | ✓ PASS |
| All plan-specified keys present | no_defense, ae_fft_topk, spectral_gated, kalman, wiener, savitzky_golay, gaussian, fir, rand_smooth | All 9 present | ✓ PASS |
| CONFMAT_ATTACKS and CONFMAT_SNRS | `assert CONFMAT_ATTACKS==['cw','eadl1','eaden']; assert CONFMAT_SNRS==[0,10,18]` | Passes | ✓ PASS |
| LINF_EPSILONS 8 points | `assert len(LINF_EPSILONS)==8` | Passes | ✓ PASS |
| OPT_C_VALUES 4 points | `assert len(OPT_C_VALUES)==4` | Passes | ✓ PASS |
| main.py defense_compare mode | `grep -c defense_compare main.py` | 8 matches | ✓ PASS |
| All 6 commits verified | `git log --oneline \| grep -E "c6dd2e9\|a7d3eac\|675abc0\|c7892b2\|32bb551\|5ff9f7b"` | All 6 found | ✓ PASS |
| Actual CSV files exist | `find inference/ -name defense_compare.csv` | No output | ✗ FAIL — experiments not run |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|---------|
| EVAL-01 | 02-01-PLAN.md | Multi-attack comparison table: all defenses vs CW, EAD L1, EAD EN, FGSM, PGD | ✗ BLOCKED | Infrastructure complete (run_defense_compare(), 5 attacks x 9 defenses), but no CSV file exists |
| EVAL-02 | 02-01-PLAN.md | Per-SNR accuracy breakdown for each defense at representative SNR points | ✗ BLOCKED | SNR columns 0,2,4,...,18 are in the CSV schema but no file has been produced |
| EVAL-03 | 02-02-PLAN.md | Confusion matrices before/after defense for CW and EAD attacks | ✗ BLOCKED | generate_confusion_matrices() implemented for 18 matrices (3 attacks x 3 SNRs x 2), no .npy files exist |
| EVAL-04 | 02-03-PLAN.md | Perturbation budget curves (accuracy vs epsilon) for each attack type | ✗ BLOCKED | generate_budget_curves() implemented for 8 Linf eps + 4 c values, no CSV files exist |
| EVAL-05 | 02-01-PLAN.md | Defense comparison table matching paper Table format (all defenses x all attacks) | ✗ BLOCKED | defense_compare_{attack}.csv pivot tables planned, not produced |

All 5 evaluation requirements are BLOCKED because experiments have not been run. REQUIREMENTS.md marks them as complete — this is a false positive in the requirements file.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `util/defense_compare.py` | 196-218 | `_get_filter_kwargs()` uses hardcoded defaults instead of Phase 1 calibrated parameters | ⚠️ Warning | Classical filter comparisons may be unfair — Phase 1 calibration (BASE-07) produced optimal parameters but they are not applied here. Gaussian uses sigma=1.0 (default), SG uses window_length=11, etc. Paper claim of comparing "calibrated baselines" may be weakened. |
| `util/defense_compare.py` | 218 | `return {}` in `_get_filter_kwargs()` for unknown defense names | ℹ️ Info | Appropriate early-exit guard, not a stub |
| `util/defense_compare.py` | 872, 887 | `return []` in `_run_attack_snr()` | ℹ️ Info | Appropriate guard for empty SNR bins, not a stub |

No TODO/FIXME/PLACEHOLDER comments found. No hollow return-null patterns found in data paths. No hardcoded empty data arrays in rendering paths.

### Human Verification Required

#### 1. Full Defense Comparison Experiment Run

**Test:** `python main.py --mode defense_compare --dataset 2016.10a --ckpt_path ./checkpoint --detector_ckpt ./checkpoint/detector_ae.pth`
**Expected:** Creates `inference/2016.10a_*/result/defense_compare/defense_compare.csv` with rows for all 9 defenses x 5 attacks x 10 SNR points plus weighted_avg rows. Five per-attack pivot tables also created.
**Why human:** Requires RML2016.10a dataset (~500MB pickle) and pretrained model checkpoint. Runtime: ~1-2 hours for CW/EAD attacks at 200 samples/cell x 10 SNRs. Cannot be run in verification context.

#### 2. Verify Unified Pipeline Outperforms Classical Filters (Success Criterion 4)

**Test:** After running experiments, inspect defense_compare_cw.csv and defense_compare_eadl1.csv — check that ae_fft_topk row has higher accuracy than all classical filter rows (kalman, wiener, savitzky_golay, gaussian, fir) at the same SNR points.
**Expected:** ae_fft_topk beats all 5 classical filters on at least CW and EAD-L1 attacks.
**Why human:** Requires actual experimental data to exist. Cannot be verified programmatically without running the evaluation.

#### 3. Calibrated Parameters Check

**Test:** Before running experiments, load calibration_params.json from Phase 1 and verify filter parameters are set in cfg, OR add loading logic to defense_compare.py. The current implementation uses hardcoded defaults that may differ from Phase 1 best-params.
**Expected:** Filter parameters used in comparison match the best parameters found during Phase 1 calibration sweep.
**Why human:** Requires checking calibration_params.json output path and deciding whether to patch defense_compare.py before running the full experiment.

## Gaps Summary

### Critical: Experiments Not Run (4 gaps)

The phase goal states results must "exist as validated CSV files." No such files exist. The entire evaluation infrastructure was built — `util/defense_compare.py` (1081 lines, 3 functions), all constants verified correct, wiring into main.py confirmed — but the single command to produce results was never executed.

**Root cause:** The plans built infrastructure (tasks 1-2 in each plan) and marked themselves complete after the code was committed, without requiring the experiments to actually run and produce output files. The SUMMARY files and REQUIREMENTS.md marks are premature completions.

**To close these gaps:** A single run of `--mode defense_compare` will produce all required CSVs (comparison table, confusion matrices, budget curves) in approximately 1-2 GPU hours.

### Warning: Calibrated Parameters Not Applied (1 gap)

The plan's key link from `defense_compare.py` to `defense_calibrate.py` (via `PARAM_GRIDS`) was not implemented. The evaluation will use hardcoded default parameters for classical filters rather than the Phase 1 calibrated best-parameters. This affects the fairness of the classical filter comparison in the paper.

**Impact:** Paper claim "all baselines use calibrated parameters" is weakened. May need to either (a) add calibration_params.json loading before the comparison run, or (b) acknowledge default parameters in the paper.

---

_Verified: 2026-04-02T11:06:28Z_
_Verifier: Claude (gsd-verifier)_
