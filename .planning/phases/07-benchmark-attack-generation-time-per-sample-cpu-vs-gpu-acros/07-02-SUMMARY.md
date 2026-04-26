---
phase: 07-benchmark-attack-generation-time-per-sample-cpu-vs-gpu-acros
plan: 02
subsystem: testing
tags: [pytorch, argparse, cli, dispatcher, benchmarking, latency]

# Dependency graph
requires:
  - phase: 07-benchmark-attack-generation-time-per-sample-cpu-vs-gpu-acros
    provides: util/attack_bench.run_attack_bench_5x2 engine (Plan 01) — Plan 02 wires it into main.py
provides:
  - "--mode attack_bench" CLI dispatcher branch in main.py
  - Four new argparse flags (--bench_n_samples, --bench_n_reps, --bench_batch_size, --bench_warmup) merged into cfg via merge_args2cfg
  - Single-command entry point that runs the full 5-attack x 2-device latency table and emits CSV+JSON artifacts
  - D-05 paper-default hyperparameter stamping at dispatch time (cfg.ta_box='unit', cfg.attack_eps=0.03, cfg.cw_c=1.0, cfg.cw_steps=100, cfg.cw_lr=0.01, cfg.ead_max_iterations=100)
affects: [07-03-paper-figure, 06-paper-update]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Dispatcher branch pattern: load checkpoint with weights_only=True, pin paper-default cfg fields in-place, forward Signals_test/Labels_test/SNRs/test_idx to bench engine"
    - "CLI bench-knob convention: --bench_<name> flags grouped together with sensible defaults (n_samples=512, n_reps=5, batch_size=128, warmup=3)"

key-files:
  created: []
  modified:
    - main.py

key-decisions:
  - "Inserted attack_bench dispatcher between adv_bench and multi_attack_eval to keep the benchmarking modes adjacent in the elif chain"
  - "Pin paper-default attack hyperparameters on cfg inside the dispatcher BEFORE create_attack() reads them, even though Plan 01's bench also stamps defaults — belt-and-suspenders so the CLI behaves correctly when invoked outside the bench"
  - "Use weights_only=True on the new torch.load call (CLAUDE.md project convention; mitigates Threat T-07-05)"
  - "Forward both SNRs and test_idx as kwargs so run_attack_bench_5x2 can stratify across (snr, label) buckets without the user passing extra flags (D-15)"

patterns-established:
  - "Phase 7 user-visible entry: `python main.py --mode attack_bench --dataset 2016.10a --ckpt_path ./checkpoint` runs the full 5x2 bench with no extra flags required"
  - "Bench CLI flags use --bench_ prefix and live in their own short section after --ta_box in the argparse block"

requirements-completed: []

# Metrics
duration: ~10min
completed: 2026-04-26
---

# Phase 7 Plan 02: main.py wiring for --mode attack_bench Summary

**Wired Plan 01's run_attack_bench_5x2 engine into main.py — added four --bench_* argparse flags and an attack_bench dispatcher branch that loads the checkpoint with weights_only=True, pins D-05 paper-default attack hyperparameters on cfg, and forwards stratification metadata so a single command produces the canonical 10-row CSV plus env JSON sidecar.**

## Performance

- **Duration:** ~10 min
- **Started:** 2026-04-26T08:15:00Z (approx)
- **Completed:** 2026-04-26T08:25:00Z (approx)
- **Tasks:** 3 / 3 (Task 3 was a human-verify checkpoint, approved by user)
- **Files modified:** 1

## Accomplishments
- Registered four new argparse flags in main.py (`--bench_n_samples`, `--bench_n_reps`, `--bench_batch_size`, `--bench_warmup`) with sensible paper-aligned defaults (512 / 5 / 128 / 3)
- Added `elif args.mode == 'attack_bench':` dispatcher branch positioned between the existing `adv_bench` and `multi_attack_eval` branches
- Dispatcher pins D-05 paper hyperparameters on cfg in-place (ta_box='unit', attack_eps=0.03, cw_c=1.0, cw_steps=100, cw_lr=0.01, ead_max_iterations=100) BEFORE the engine's create_attack() reads them
- Checkpoint loaded via `torch.load(..., weights_only=True)` per CLAUDE.md project convention (Threat T-07-05 mitigation)
- Smoke run with `--bench_n_samples 64 --bench_n_reps 2 --bench_warmup 1` produced canonical 12-line CSV (1 env-comment + 1 header + 10 data rows) plus non-empty `attack_bench_env.json` — verified by user

## Task Commits

Each task was committed atomically:

1. **Task 1: Add --bench_* argparse flags to main.py** — `ac6733d` (feat)
2. **Task 2: Add `elif args.mode == 'attack_bench':` dispatcher branch** — `6c77172` (feat)
3. **Task 3: Smoke-run and verify CSV+JSON outputs** — checkpoint:human-verify, approved by user (no commit; verification only)

**Plan metadata:** _pending_ (this SUMMARY commit)

## Files Created/Modified
- `main.py` — Added four `--bench_*` argparse flags (Task 1) and the `attack_bench` dispatcher branch (Task 2). Net additions: 5-line argparse block immediately after `--ta_box`, and a ~20-line dispatcher block immediately after `adv_bench`. No existing line modified or removed.

## Decisions Made
- **Stamp D-05 paper hyperparameters on cfg in the dispatcher.** Even though Plan 01's `_stamp_paper_defaults` does the same inside the engine, doing it again at dispatch time is intentional: it makes the CLI behaviour explicit and self-documenting at the entry point, and any future caller reading main.py sees the contract without having to dig into util/attack_bench.py.
- **weights_only=True on the new torch.load call.** The existing `adv_bench` branch uses the older signature without `weights_only=True`; the new branch uses it because the project standard (CLAUDE.md) requires it for fresh code, and Threat T-07-05 in the plan's STRIDE register names `weights_only=True` as the mitigation.
- **Forward both `snrs_test=SNRs` and `test_idx=test_idx` as kwargs.** Optional in the engine signature but the dispatcher always passes both so the stratified sampler picks (snr, label) buckets, not label-only buckets. This honours D-15.
- **Insertion point chosen between adv_bench and multi_attack_eval.** Keeps the two benchmarking modes (`adv_bench`, `attack_bench`) adjacent in the elif chain for readability.

## Deviations from Plan

None — plan executed exactly as written. Tasks 1 and 2 each landed cleanly with all acceptance grep counts hitting their target values. Task 3 was a human-verify checkpoint; the user ran the smoke command, observed the 12-line CSV and non-empty JSON, and typed "approved".

## Issues Encountered

- **None during this plan.** Tasks 1-2 were executed via a parallel worktree with `--no-verify` allowed in autonomous mode and merged back to main as commit `5537c7b`. Task 3's smoke run completed successfully; the user's approval skipped the orchestrator's re-verification step.

## User Setup Required

None — pure local CLI wiring. No external service configuration required.

## Next Phase Readiness

- **Plan 03 ready:** Plan 03's figure generator can now invoke the canonical command (`python main.py --mode attack_bench --dataset 2016.10a --ckpt_path ./checkpoint`) to produce a fresh `attack_bench.csv` with full defaults (N=512, R=5), then read the CSV from `inference/2016.10a_*/result/attack_bench.csv` to render the bar chart.
- **No blockers.** End-to-end pipeline confirmed via the Task 3 smoke run.

## Self-Check: PASSED

- Tasks 1 and 2 commits present in git log:
  - `ac6733d` feat(07-02): add Phase 7 --bench_* argparse flags to main.py — FOUND
  - `6c77172` feat(07-02): add attack_bench dispatcher branch to main.py — FOUND
- main.py contains the new dispatcher: `grep -n "args.mode == 'attack_bench'" main.py` → line 433 — FOUND
- main.py contains the new argparse section: `grep -n "Phase 7 attack-bench knobs" main.py` → line 55 — FOUND
- Task 3 smoke run produced CSV+JSON artifacts (verified by user, who typed "approved")

---
*Phase: 07-benchmark-attack-generation-time-per-sample-cpu-vs-gpu-acros*
*Completed: 2026-04-26*
