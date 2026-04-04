---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: executing
stopped_at: Completed 02-experimental-results/02-04-PLAN.md
last_updated: "2026-04-04T15:45:49.405Z"
last_activity: 2026-04-04
progress:
  total_phases: 3
  completed_phases: 1
  total_plans: 8
  completed_plans: 7
  percent: 0
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-31)

**Core value:** Demonstrate that a unified detect→recover→classify pipeline outperforms classical filtering defenses against optimization-based adversarial attacks on RF signals, while maintaining real-time feasibility
**Current focus:** Phase 02 — experimental-results

## Current Position

Phase: 02 (experimental-results) — EXECUTING
Plan: 2 of 5
Status: Ready to execute
Last activity: 2026-04-04

Progress: [░░░░░░░░░░] 0%

## Performance Metrics

**Velocity:**

- Total plans completed: 0
- Average duration: —
- Total execution time: 0 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| - | - | - | - |

**Recent Trend:**

- Last 5 plans: —
- Trend: —

*Updated after each plan completion*
| Phase 01-defense-implementations P01 | 2 | 1 tasks | 1 files |
| Phase 01-defense-implementations P02 | 2 | 1 tasks | 1 files |
| Phase 01-defense-implementations P03 | 3 | 1 tasks | 1 files |
| Phase 02-experimental-results P01 | 8 | 2 tasks | 2 files |
| Phase 02-experimental-results P02 | 2 | 2 tasks | 2 files |
| Phase 02-experimental-results P03 | 5 | 2 tasks | 2 files |
| Phase 02-experimental-results P04 | 8 | 2 tasks | 2 files |

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- Init: Unified pipeline (detect+recover+classify) as main contribution — novel for RF domain
- Init: Classical filters as baselines (not other ML defenses) — shows improvement over signal processing
- Init: GPU-native ops (depthwise conv1d) for Gaussian and FIR to avoid CPU roundtrip invalidating latency claims
- [Phase 01-defense-implementations]: GPU-native F.conv1d for Gaussian and FIR filters to avoid CPU roundtrip invalidating latency claims
- [Phase 01-defense-implementations]: Manual scalar NumPy Kalman loop used because pykalman/filterpy are not installed
- [Phase 01-defense-implementations]: FIR coefficients computed per call (not cached) to support calibration parameter sweeps
- [Phase 01-defense-implementations]: DEFENSE_REGISTRY uses try/except ImportError for baseline imports so Plan 02 can run before Plan 01 completes (parallel execution)
- [Phase 01-defense-implementations]: defend() unified pipeline has separate rand_smooth dispatch path; GPU ops use torch.cuda.Event; CPU ops use time.perf_counter
- [Phase 01-defense-implementations]: Calibration sample cap of 200 per SNR cell to keep CW sweep tractable; PARAM_GRIDS with 84 combinations for 5 filters; run_latency_benchmark uses torch.cuda.Event for GPU, time.perf_counter for CPU with 10 warmup iterations
- [Phase 02-experimental-results]: minmax normalization for defense_compare torchattacks calls — consistent with D-05 and Phase 1 calibration
- [Phase 02-experimental-results]: ae_fft_topk branch calls defend() pipeline with temporary cfg.defense override to reuse unified pipeline
- [Phase 02-experimental-results]: generate_confusion_matrices() saves raw .npy for Phase 3 rendering and row-normalized CSVs for inspection
- [Phase 02-experimental-results]: cfg override pattern (save/set/restore cfg.defense) reuses defend() pipeline for confusion matrix after-defense condition
- [Phase 02-experimental-results]: Use cfg save/restore pattern (not copy.copy) for attack_eps/cw_c/ead_initial_const overrides to avoid Config object complexity
- [Phase 02-experimental-results]: calibration_path=None falls through to cfg defaults — backward-compatible, no regressions when calibration JSON absent
- [Phase 02-experimental-results]: _CALIB_TO_CFG module-level constant for 5 classical filters with cfg attribute names — covers kalman, wiener, savitzky_golay, gaussian, fir

### Pending Todos

None yet.

### Blockers/Concerns

- Kalman and Wiener require CPU fallback (pykalman); honest latency reporting required in paper
- Baseline parameter sweeps (filter order, cutoff, window) must be run before results are meaningful — risk of reviewer rejection for under-tuned baselines
- CW/EAD attack effectiveness must be verified on undefended model before defense comparison proceeds

## Session Continuity

Last session: 2026-04-04T15:45:49.399Z
Stopped at: Completed 02-experimental-results/02-04-PLAN.md
Resume file: None
