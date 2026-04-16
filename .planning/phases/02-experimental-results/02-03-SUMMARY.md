---
phase: 02-experimental-results
plan: "03"
subsystem: evaluation
tags: [budget-curves, perturbation-sweep, defense-compare, eval]
dependency_graph:
  requires: [02-01]
  provides: [generate_budget_curves, LINF_EPSILONS, OPT_C_VALUES, LINF_ATTACKS, OPT_ATTACKS]
  affects: [main.py defense_compare mode, Phase 3 figure rendering]
tech_stack:
  added: []
  patterns: [cfg-save-restore for parameter sweeps, nested attack loop per perturbation strength]
key_files:
  created: []
  modified:
    - util/defense_compare.py
    - main.py
decisions:
  - "Use cfg save/restore pattern (not copy.copy) for attack_eps/cw_c/ead_initial_const overrides to avoid Config object complexity"
  - "Inline _run_attack_snr() helper inside generate_budget_curves() to avoid parameter threading complexity"
  - "Per-attack pivot CSVs use param_value as row index and DEFENSE_CONFIGS key order for columns"
metrics:
  duration: "5m"
  completed: "2026-04-02T11:01:48Z"
  tasks_completed: 2
  files_modified: 2
---

# Phase 2 Plan 3: Budget Curve Generation Summary

**One-liner:** Perturbation budget curve sweep — 8 Linf epsilon points (FGSM/PGD) and 4 c values (CW/EAD-L1/EAD-EN) measuring all 9 defenses per strength, wired into --mode defense_compare.

## What Was Built

Added `generate_budget_curves()` to `util/defense_compare.py` (EVAL-04) and wired it into `main.py --mode defense_compare`.

### New Constants (D-06, D-07, D-08)

```python
LINF_EPSILONS = [0.01, 0.03, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]   # 8 points
OPT_C_VALUES  = [0.01, 0.1, 1.0, 10.0]                            # 4 points
LINF_ATTACKS  = ['fgsm', 'pgd']
OPT_ATTACKS   = ['cw', 'eadl1', 'eaden']
```

### Function: generate_budget_curves()

Signature:
```python
def generate_budget_curves(
    model, sig_test, lab_test, SNRs, test_idx, cfg, logger,
    detector=None, linf_epsilons=None, opt_c_values=None,
    target_snrs=None, max_per_cell=200, batch_size=64,
) -> pd.DataFrame
```

**Part A (Linf):** For each eps in LINF_EPSILONS, temporarily sets `cfg.attack_eps = eps`, creates attack object, generates adversarial examples per SNR in target_snrs, evaluates all 9 DEFENSE_CONFIGS, records row with `param_name='eps'`.

**Part B (Optimization):** For each c in OPT_C_VALUES, temporarily sets `cfg.cw_c = c_val` and `cfg.ead_initial_const = c_val`, runs CW/EAD-L1/EAD-EN per SNR, records row with `param_name='c'`.

**Outputs to `defense_compare/budget_curves/`:**
- `budget_curves_detail.csv` — full per-SNR data (all rows)
- `budget_curves_agg.csv` — weighted average per (attack, param_name, param_value, defense)
- `budget_fgsm.csv`, `budget_pgd.csv`, `budget_cw.csv`, `budget_eadl1.csv`, `budget_eaden.csv` — pivot tables

### main.py Changes

- Added `--skip_budget` argparse flag
- After confusion matrices: calls `generate_budget_curves()` unless `--skip_budget`
- Single `python main.py --mode defense_compare` now produces: comparison table + confusion matrices + budget curves

## Commits

| Task | Commit | Files |
|------|--------|-------|
| 1: generate_budget_curves() | 32bb551 | util/defense_compare.py |
| 2: Wire into main.py | 5ff9f7b | main.py |

## Deviations from Plan

None — plan executed exactly as written.

## Known Stubs

None. The generate_budget_curves() function is fully implemented with real attack generation, defense application, and CSV output. No placeholders or hardcoded empty values in the data flow.

## Self-Check: PASSED

- util/defense_compare.py: FOUND (modified with generate_budget_curves)
- main.py: FOUND (modified with generate_budget_curves call and --skip_budget)
- Commits 32bb551 and 5ff9f7b: verified in git log
