---
phase: 07-benchmark-attack-generation-time-per-sample-cpu-vs-gpu-acros
plan: 03
subsystem: paper
tags: [matplotlib, pandas, ieee_style, paper-figure, latency-plot]

# Dependency graph
requires:
  - phase: 07-benchmark-attack-generation-time-per-sample-cpu-vs-gpu-acros
    provides: util/attack_bench.run_attack_bench_5x2 (Plan 01) + main.py --mode attack_bench (Plan 02) — Plan 03 reads the CSV they produce
provides:
  - paper/scripts/plot_attack_bench_latency.py — standalone CSV→PDF plotter
  - paper/latex/figures/attack_bench_latency.pdf — drop-in figure for Phase 6 camera-ready paper integration
affects: [06-paper-update]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Local sibling import: sys.path.insert(0, dirname(__file__)) + 'from ieee_style import ...' so the script runs cleanly from anywhere"
    - "Repo-root anchoring: paper/scripts/<file> -> paper/scripts -> paper -> repo via three dirname() hops; lets --csv/--out accept relative paths without surprises"
    - "Defensive CSV validation: reject sets that drift from {fgsm,pgd,cw,eadl1,eaden} x {cpu,cuda} or row counts != 10 before plotting (T-07-08 mitigation)"

key-files:
  created:
    - paper/scripts/plot_attack_bench_latency.py
    - paper/latex/figures/attack_bench_latency.pdf
  modified:
    - main.py

key-decisions:
  - "Use single_col_fig (3.487 in wide) by default — Phase 6 will decide column placement; single-col is the safer default and matches v1.0 figure conventions"
  - "Pair gray (#7f7f7f, no_defense) for CPU with blue (#1f77b4, adaptive_k) for GPU — both come from the existing DEFENSE_COLORS palette and stay distinguishable in greyscale printing"
  - "Log-scale y-axis (D-10): CW/EAD measure ~100x FGSM, so a linear axis would compress FGSM/PGD bars to invisibility"
  - "Validate CSV before plotting: reject mismatched attack/device sets and row counts != 10. Halts loud and clear if a future caller hands the script the wrong CSV"
  - "Default --csv resolves the newest inference/2016.10a_*/result/attack_bench.csv via glob + sorted()[-1]; lexicographic sort is also chronological because the index auto-increments"

patterns-established:
  - "paper/scripts/plot_*.py convention: matplotlib.use('Agg') for headless rendering, sibling sys.path import of ieee_style, argparse with sane defaults, repo-root anchoring for relative paths"
  - "Phase 7 figure regen recipe: 1) python main.py --mode attack_bench ..., 2) cd paper/scripts && python plot_attack_bench_latency.py — two commands and the PDF is up to date"

requirements-completed: []

# Metrics
duration: ~12min
completed: 2026-04-26
---

# Phase 7 Plan 03: Attack-bench latency figure for paper Summary

**Standalone paper/scripts/plot_attack_bench_latency.py renders the Phase 7 attack_bench.csv as an IEEE-styled grouped CPU/GPU bar chart with log-y axis and std error bars, and writes paper/latex/figures/attack_bench_latency.pdf for Phase 6 camera-ready integration. Also fixed a wiring bug in Plan 02's main.py dispatcher that passed full-dataset SNRs into the test-split bench engine.**

## Performance

- **Duration:** ~12 min
- **Started:** 2026-04-26T16:30:00Z
- **Completed:** 2026-04-26T16:55:00Z
- **Tasks:** 2 / 2
- **Files modified:** 3 (2 created, 1 modified)

## Accomplishments

- Implemented `paper/scripts/plot_attack_bench_latency.py` (~166 lines) with `_resolve_default_csv`, `_load_and_validate`, `_plot`, and `main` helpers. Standalone, headless-safe (`matplotlib.use('Agg')`), reuses `apply_ieee_style` + `single_col_fig` from `paper/scripts/ieee_style.py` for v1.0 figure consistency.
- Generated `paper/latex/figures/attack_bench_latency.pdf` (13.9 KB, single-page PDF) from the smoke-run CSV at `inference/2016.10a_165/result/attack_bench.csv` (n=64 samples, 2 reps; numbers will be regenerated from a full-budget CSV during Phase 6).
- Fixed Plan 02 wiring bug: `main.py`'s attack_bench dispatcher passed the full 220 000-row `SNRs` array into `run_attack_bench_5x2` while the bench was given the 44 000-row test split. Sliced `snrs_test = [SNRs[i] for i in test_idx]` to match `Labels_test`, mirroring the existing pattern in `util/multi_attack_eval.py:build_snr_mod_index`.

## Task Commits

Each task was committed atomically:

1. **Task 1: Write paper/scripts/plot_attack_bench_latency.py** — `f8fc57a` (feat)
2. **Task 2: Render attack_bench_latency.pdf** — `f202463` (feat)
3. **Deviation fix (Rule 1 bug in Plan 02 main.py wiring)** — `6db6362` (fix)

## Files Created/Modified

- `paper/scripts/plot_attack_bench_latency.py` — New standalone plotter. Resolves CSV via `glob('inference/2016.10a_*/result/attack_bench.csv')`, validates the 5x2=10-row shape, then renders a grouped bar chart with `single_col_fig` + log-y axis + std error bars. Uses `DEFENSE_COLORS['no_defense']` (gray) for CPU and `DEFENSE_COLORS['adaptive_k']` (blue) for GPU.
- `paper/latex/figures/attack_bench_latency.pdf` — 13.9 KB single-page PDF rendered from the smoke-run CSV. Drop-in for Phase 6 camera-ready paper integration.
- `main.py` — Sliced `snrs_test` to the test split before forwarding to `run_attack_bench_5x2`. Net: 4 lines added, 1 changed; no other edits.

## Decisions Made

- **Default to `single_col_fig` (3.487 in wide).** Half-page-friendly figure that fits a single IEEEtran column. Phase 6 decides exactly where the figure lands (Table I context vs computational-cost appendix); single-col is the safer default and avoids assuming a layout.
- **Sibling-style local import for `ieee_style`.** `sys.path.insert(0, dirname(abspath(__file__)))` plus `from ieee_style import ...` mirrors the convention used by `paper/scripts/generate_figures.py`. Lets the script run from any cwd, not just `paper/scripts`.
- **Validate CSV before plotting.** `_load_and_validate` rejects CSVs whose attack/device sets drift from the paper-fixed `{fgsm,pgd,cw,eadl1,eaden} x {cpu,cuda}` or whose row count != 10. Maps to threat T-07-08 in the plan's STRIDE register.
- **Resolved CSV via newest-glob.** `sorted(glob('inference/2016.10a_*/result/attack_bench.csv'))[-1]` — lexicographic sort is chronological because the inference index auto-increments. Lets reviewers re-run the bench and re-render without thinking about paths.
- **Fixed Plan 02 wiring bug inline (Rule 1).** Could not produce the CSV needed for Task 2 verification without slicing `SNRs` to the test split first. Fixed in main.py and committed as a separate `fix(07-03)` commit so the bug history is captured. Smoke run then produced the expected 10-row CSV.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Plan 02 main.py wiring passed full-dataset SNRs to test-split bench**
- **Found during:** Task 2 (running the canonical smoke command).
- **Issue:** `main.py:451` called `run_attack_bench_5x2(..., snrs_test=SNRs, test_idx=test_idx)` with `SNRs` of length 220000 while `Signals_test/Labels_test` are the 44000-row test split. The engine's `_stratified_indices` raised `ValueError: snrs_test length 220000 != lab_test length 44000` and aborted before any timing happened.
- **Fix:** Built `snrs_test = [SNRs[i] for i in test_idx]` (mirrors `util/multi_attack_eval.py:build_snr_mod_index`'s `SNRs[orig_idx]` access pattern) and forwarded the sliced list. Added a 2-line comment explaining why.
- **Files modified:** `main.py` (4 lines added, 1 changed).
- **Commit:** `6db6362`

### Pre-existing context drift (not deviations from this plan)

- **Worktree base mismatch.** This worktree's HEAD started at `862a94f` (parent of the planning base `5c016b4`); the Phase 7 plan files and Plan 01/02 deliverables (`util/attack_bench.py`, `main.py` dispatcher) were not yet in the worktree's tree. Resolved per the `<worktree_branch_check>` protocol with a soft reset to `5c016b4` plus `git checkout HEAD -- .` to materialize the working tree, then copied the three Plan markdown files from the main repo's `.planning/phases/07-...` directory.
- **CSV not pre-existing.** The orchestrator prompt stated `inference/2016.10a_*/result/attack_bench.csv` already existed from a user smoke run. It did not exist on disk anywhere under `inference/` (newest dirs are dated April 2). Produced it inline via the same canonical smoke command (`--bench_n_samples 64 --bench_n_reps 2 --bench_warmup 1`) using the tile-lang virtualenv (`/home/nigel/opensource/gpu_env/tile-lang/bin/python`) which has `torch=2.9.0+cu130` + `torchattacks=3.5.1`. The numbers in the resulting PDF reflect that smoke-budget run; Phase 6 will regenerate against a full-budget (N=512, R=5) CSV.

## Issues Encountered

- **Plan 02 main.py wiring bug** — see Auto-fixed Issues above. Surfaced immediately on the first smoke invocation.
- **Worktree had no Python venv of its own.** Used the user's tile-lang environment which already has `torch+cuda+torchattacks`. The bench imports cleanly without `h5py` because `data_loader.py` only imports `h5py` for non-RML2016 datasets (`pickle` path is taken for `2016.10a`).

## User Setup Required

None — pure local artifact generation. No external service configuration required.

## Next Phase Readiness

- **Plan 03 deliverables landed:**
  - `paper/scripts/plot_attack_bench_latency.py` is committed and importable. Re-running it (`cd paper/scripts && python plot_attack_bench_latency.py`) regenerates the figure idempotently against whichever CSV is newest.
  - `paper/latex/figures/attack_bench_latency.pdf` is committed (smoke-budget numbers; ready to be regenerated against a full-budget CSV during Phase 6 if desired).
- **Phase 6 ready:** The PDF can be `\includegraphics{figures/attack_bench_latency}` directly in `paper/latex/main.tex`. Numbers reflect a 64-sample, 2-rep smoke run on an RTX 5060 Ti — Phase 6 should re-run with `python main.py --mode attack_bench --dataset 2016.10a --ckpt_path ./checkpoint` (full defaults: N=512, R=5) and re-render before submission.
- **No blockers.** End-to-end pipeline (engine → CLI → CSV → plotter → PDF) is now confirmed working end to end.

## Self-Check: PASSED

- File `paper/scripts/plot_attack_bench_latency.py` — FOUND
- File `paper/latex/figures/attack_bench_latency.pdf` — FOUND, `file` reports `PDF document, version 1.4, 1 pages`, size 13916 bytes (>1024)
- Commits `f8fc57a`, `f202463`, `6db6362` — FOUND in `git log --oneline`
- All Task 1 acceptance grep checks pass:
  - `from ieee_style import` count = 1
  - `apply_ieee_style()` count = 1
  - `set_yscale('log')` count = 1
  - `yerr=stds` count = 1
  - `ATTACK_ORDER = ['fgsm', 'pgd', 'cw', 'eadl1', 'eaden']` count = 1
  - `DEVICE_ORDER = ['cpu', 'cuda']` count = 1
  - `len(df) == 10` count = 1
  - `comment='#'` count = 1
  - `awk 'length>110'` returns 0
  - `python -c "import ast; ast.parse(...)"` exits 0
- Task 2 acceptance:
  - `test -f paper/latex/figures/attack_bench_latency.pdf` exits 0
  - `file ... | grep PDF document` matches
  - `stat -c%s ... > 1024` true (13916 bytes)
  - Plotter stdout contained `Wrote /home/nigel/opensource/adversarial-rf/paper/latex/figures/attack_bench_latency.pdf`

---
*Phase: 07-benchmark-attack-generation-time-per-sample-cpu-vs-gpu-acros*
*Completed: 2026-04-26*
