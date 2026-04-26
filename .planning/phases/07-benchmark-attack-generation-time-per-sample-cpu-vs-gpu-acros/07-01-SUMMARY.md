---
phase: 07-benchmark-attack-generation-time-per-sample-cpu-vs-gpu-acros
plan: 01
subsystem: testing
tags: [pytorch, torchattacks, benchmarking, latency, cuda, perf_counter]

# Dependency graph
requires:
  - phase: 05-at-evaluation
    provides: 5-attack list (cw, eadl1, eaden, fgsm, pgd) — Phase 7 reuses verbatim
  - phase: 04-adversarial-training
    provides: util/sigguard_eval.create_attack factory + Model01Wrapper IQ adapter
provides:
  - run_attack_bench_5x2 public function for 5-attack x 2-device latency benchmark
  - Stratified (snr, mod) sampler + env-metadata helper
  - Sync-bracketed perf_counter timing pattern with warmup + R=5 reps
  - CSV (10 rows) + JSON env sidecar output convention
affects: [07-02-main-py-wiring, 07-03-paper-figure, 06-paper-update]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Attack-factory reuse: bench imports util.sigguard_eval.create_attack rather than duplicating attack registry"
    - "Belt-and-suspenders state_dict reload between device passes (Threat T-07-01 mitigation)"
    - "Cuda sync brackets perf_counter on GPU; CPU side skips sync"

key-files:
  created:
    - util/attack_bench.py
  modified: []

key-decisions:
  - "Halt early with RuntimeError when CUDA is unavailable (D-16) — bench compares CPU vs GPU, a CPU-only report would be misleading"
  - "Stamp paper-default hyperparameters onto cfg in-place inside the bench (eps=0.03, cw_c=1.0, cw_steps=100, cw_lr=0.01, ead_max_iterations=100) so create_attack returns D-05-locked attack objects"
  - "Accept PGD alpha = eps/4 = 0.0075 (sigguard_eval default) instead of paper alpha=0.01; documented in module docstring; latency delta is negligible"
  - "Round-robin stratification across (snr, label) buckets when SNRs are provided, else label-only stratification; deterministic seed=2022"
  - "Snapshot original state_dict on CPU once and reload on the target device before each pass (mitigates in-place mutation by attacks)"

patterns-established:
  - "bench module pattern: public run_*_bench function + private _collect_env / _stratified_indices / _time_one_cell helpers"
  - "CSV convention: env-comment line `# env: torch=... cuda=... gpu=... cpu=... cores=...` + header + data rows; JSON sidecar with the raw row dicts"

requirements-completed: []

# Metrics
duration: 2min
completed: 2026-04-26
---

# Phase 7 Plan 01: 5-attack x 2-device latency bench engine Summary

**New util/attack_bench.py module that times fgsm/pgd/cw/eadl1/eaden adversarial generation per sample on CPU and GPU in one invocation, reusing sigguard_eval.create_attack and Model01Wrapper, and emits a 10-row CSV + JSON env sidecar.**

## Performance

- **Duration:** ~2 min
- **Started:** 2026-04-26T08:11:00Z
- **Completed:** 2026-04-26T08:13:00Z
- **Tasks:** 1 / 1
- **Files modified:** 1 (1 created, 0 modified)

## Accomplishments
- Implemented `run_attack_bench_5x2(model, sig_test, lab_test, cfg, logger, snrs_test=None, test_idx=None)` public function (471 lines including docstrings)
- Implemented private helpers `_collect_env`, `_stratified_indices`, `_iter_batches`, `_time_one_cell`, `_stamp_paper_defaults`
- Encoded all D-01..D-16 invariants in code: CUDA-required halt, two-device loop, state_dict reload, sync-bracketed timing, R=5 reps, paper-fixed `ATTACKS_5` list, CSV + JSON outputs at `cfg.result_dir/attack_bench{,_env}.{csv,json}`
- Module docstring documents the PGD-alpha trade-off (0.0075 vs paper 0.01) and stratification choice for reviewer-visible transparency

## Task Commits

Each task was committed atomically:

1. **Task 1: Create util/attack_bench.py with stratified sampler and env helper** — `a0c6ad7` (feat)

## Files Created/Modified
- `util/attack_bench.py` — New module exposing `run_attack_bench_5x2` plus three private helpers; encodes the D-01..D-16 contract (CUDA-required halt, sync-bracketed timing, state_dict reload between device passes, stratified N=512 sampler, paper-default hyperparameter stamping, CSV + JSON sidecar)

## Decisions Made
- **Reuse, don't duplicate, the attack factory.** Imported `create_attack` and `generate_adversarial` directly from `util.sigguard_eval`. Plan 02 will own CLI defaults; Plan 01 stamps paper defaults onto `cfg` in-place at the top of the bench so the factory returns D-05-locked objects without needing a parallel registry.
- **Accept PGD alpha=0.0075 instead of paper-quoted 0.01.** Documented as a divergence in the module docstring. Latency cost is dominated by the 10 inner forward/backward passes, not the step size, so the bench number is faithful to the paper's runtime.
- **Stratify by (snr, label) when both are provided, else by label only.** Caller's choice — `snrs_test`/`test_idx` are optional kwargs. This keeps Plan 01 callable without Plan 02's CLI wiring being in place.
- **Snapshot state_dict on CPU and reload per-device.** Belt-and-suspenders against in-place mutation by attacks (Threat T-07-01).
- **Halt with RuntimeError if CUDA is unavailable (D-16).** Project explicitly targets a single-GPU host; emitting only a CPU column would mislead reviewers.

## Deviations from Plan

None — plan executed exactly as written. The verification environment-prep step in `<worktree_branch_check>` required a `git reset --hard` to align the worktree branch with the planning base commit (`8d99abe`), which is expected behaviour and not a code deviation.

## Issues Encountered

- **Worktree base mismatch.** Worktree HEAD started at `862a94f` (parent of the planning base `8d99abe`); the phase-07 plan files weren't in the working tree. Resolved per the `<worktree_branch_check>` protocol with a hard reset to `8d99abe`, then copied the three Plan markdown files from the main worktree's `.planning/phases/07-...` directory so the executor could read 07-01-PLAN.md and 07-CONTEXT.md.

## User Setup Required

None — no external service configuration required. Bench is pure local measurement (Threat surface is intentionally minimal per the plan's STRIDE register).

## Next Phase Readiness

- **Plan 02 ready:** `run_attack_bench_5x2(model, sig_test, lab_test, cfg, logger, snrs_test, test_idx)` is importable and ready for `main.py --mode attack_bench` wiring. CLI defaults (`--bench_n_samples`, `--bench_n_reps`, `--bench_batch_size`, `--bench_warmup`) are picked up via `getattr(cfg, ...)` with sensible fallbacks.
- **Plan 03 ready (post-Plan 02):** `attack_bench.csv` schema is locked at `attack,device,batch_size,n_samples,n_reps,mean_ms_per_sample,std_ms_per_sample,total_seconds`. The figure generator can consume the CSV without ambiguity.
- **No blockers.** Module imports cleanly with no top-level side effects (no `print`, no `torch.load`, no I/O at import time).

## Self-Check: PASSED

- File `util/attack_bench.py` — FOUND
- Commit `a0c6ad7` — FOUND in `git log`
- All 10 acceptance grep checks pass:
  - `def run_attack_bench_5x2(` count = 1
  - `ATTACKS_5 = ['fgsm', 'pgd', 'cw', 'eadl1', 'eaden']` count = 1
  - `torch.cuda.synchronize()` count = 2 (≥ 2)
  - `torch.cuda.is_available()` count = 3 (≥ 1)
  - `from util.sigguard_eval import create_attack, generate_adversarial` count = 1
  - `from util.adv_attack import Model01Wrapper` count = 1
  - `load_state_dict` count = 1 (≥ 1)
  - `python -c "import ast; ast.parse(open('util/attack_bench.py').read())"` exits 0
  - `python -c "from util.attack_bench import run_attack_bench_5x2, ATTACKS_5; assert ATTACKS_5 == ['fgsm','pgd','cw','eadl1','eaden']"` exits 0
  - `awk 'length>110' util/attack_bench.py | wc -l` = 0

---
*Phase: 07-benchmark-attack-generation-time-per-sample-cpu-vs-gpu-acros*
*Completed: 2026-04-26*
