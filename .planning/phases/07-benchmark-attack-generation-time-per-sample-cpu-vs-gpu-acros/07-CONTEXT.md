# Phase 7: Benchmark attack generation time per sample (CPU vs GPU) across 5 attacks - Context

**Gathered:** 2026-04-26
**Status:** Ready for planning

<domain>
## Phase Boundary

Measure adversarial-attack generation latency per sample, on CPU and GPU, for
the 5 paper attacks (FGSM, PGD, EAD-L1, EAD-EN, CW), on RML2016.10a IQ inputs,
using the v1.0 paper's hyperparameter defaults. Deliverables: a standalone
attack_bench.csv under inference/ plus a paper-ready bar chart figure for
Phase 6 camera-ready integration.

This phase is pure measurement. It does NOT introduce new attacks, new
defenses, or new training. ASR/accuracy of generated adversarial examples is
already covered by Phase 5 — Phase 7 reports latency only.

</domain>

<decisions>
## Implementation Decisions

### Hardware & Timing Methodology
- **D-01:** CPU side runs the same script with `cfg.device='cpu'`; GPU side
  runs with `cfg.device='cuda'`. Single codebase, two device passes inside
  one invocation (see D-13). The bench fully respects whatever device the
  Config object holds — no separate CPU-only environment, no
  `CUDA_VISIBLE_DEVICES` gating required.
- **D-02:** Per-attack warmup before each timed cell: 3–5 generation
  iterations on a small batch, results discarded. Catches lazy CUDA init,
  cuDNN autotune, and torchattacks hook setup. Standard PyTorch
  benchmarking practice. CPU side does the same warmup count for symmetry.
- **D-03:** GPU timing uses `torch.cuda.synchronize()` immediately before
  `t0 = time.perf_counter()` and immediately before `t1 = time.perf_counter()`.
  CPU side skips sync (no-op anyway). Wraps each timed block, not each
  batch within the block. Standard correct pattern for PyTorch latency
  benchmarks.
- **D-04:** Statistics per (attack, device) cell: mean ± std of per-sample
  latency over R=5 repetitions of the full timing block. Two columns in
  CSV (`mean_ms_per_sample`, `std_ms_per_sample`). Mean drives the paper
  bar chart; std drives the error bars.

### Attack Hyperparameters & Sample Scope
- **D-05:** Use the v1.0 paper's per-attack hyperparameter defaults
  verbatim — same eps, steps, c, lr, kappa, ta_box mode that the paper
  reports. This means each attack's intrinsic cost is what gets measured
  (CW/EAD will be ~10× PGD; FGSM is single-step). Latency numbers reflect
  what a reproducer running the paper's settings actually sees. Concrete
  values come from `util/sigguard_eval.py:create_attack()` defaults
  (FGSM eps=0.03, PGD eps=0.03/steps=10/alpha=0.01, CW c=1.0/steps=100/lr=0.01,
  EAD-L1 steps=100, EAD-EN steps=100, ta_box=unit). Planner verifies these
  match the paper before committing.
- **D-06:** Time N=512 IQ samples per (attack, device) cell, drawn from the
  RML2016.10a test split. Sample stratified across modulations × SNRs to
  avoid degenerate-difficulty cells. Enough for stable per-sample mean,
  feasible on CPU within minutes for CW/EAD.
- **D-07:** Fixed `batch_size=128` matching the project's standard
  `test_batch_size`. Per-sample latency = total time for 128 / 128. One
  number per (attack, device). Streaming (batch=1) and throughput sweeps
  are deferred — see Deferred Ideas.
- **D-08:** Latency only — no ASR/accuracy reported in this phase. ASR is
  already in Phase 5 outputs. Phase 7 adds zero new accuracy claims.
  CSV columns: `attack, device, batch_size, n_samples, n_reps,
  mean_ms_per_sample, std_ms_per_sample, total_seconds`.

### Output Deliverables & Paper Integration
- **D-09:** Primary artifact: standalone `attack_bench.csv` under
  `inference/<dataset>_*/result/`. Schema matches D-08. Exactly 10 rows
  (5 attacks × 2 devices). Lives next to other inference/ artifacts;
  follows `util/bench.py:run_attack_bench` output convention.
- **D-10:** Phase 7 also produces a paper figure:
  `paper/latex/figures/attack_bench_latency.pdf`. Grouped bar chart, 5
  attacks on x-axis, 2 bars per group (CPU, GPU), error bars from std,
  log-scale y-axis (CW/EAD will be ~100× FGSM). Uses the existing
  `paper/figures/ieee_style.py` styling so the figure is consistent
  with v1.0 paper figures. Phase 6 decides exactly which section/appendix
  it lands in — Phase 7 just produces the PDF and a placeholder caption.
- **D-11:** Logging follows the project's existing pattern:
  `util/logger.py:create_logger` for structured log file +
  `tqdm` for per-rep progress. Final log writes a formatted summary
  table (attacks × devices). Matches `util/multi_attack_eval.py` and
  `util/sigguard_eval.py` conventions.
- **D-12:** Environment metadata is recorded for reproducibility:
  `torch.cuda.get_device_name(0)`, `platform.processor()` (best effort),
  `torch.__version__`, `torch.version.cuda`, `cpu_count`. Embedded as a
  CSV header comment line (`# env: ...`) AND written to a companion
  `attack_bench_env.json`. Reviewers can verify which GPU/CPU produced
  which column.

### Code Structure & Invocation
- **D-13:** New module: `util/attack_bench.py`. Public function:
  `run_attack_bench_5x2(model, sig_test, lab_test, cfg, logger)`. Loops
  `for device in [cpu, cuda]:` outermost, calls `model.to(device)`,
  iterates 5 attacks inside. Existing `util/bench.py:run_attack_bench`
  (CW vs spectral) is left untouched — different scope, different
  output. Caveat: between device passes, ensure model state is identical
  (re-load state_dict if any attack mutates parameters in place).
- **D-14:** New main.py mode: `--mode attack_bench`. Dispatcher block
  follows the existing `elif args.mode == 'adv_bench':` pattern at
  main.py:417. Accepts the standard `--ckpt_path`, `--dataset`, plus new
  flags for the bench: `--bench_n_samples` (default 512),
  `--bench_n_reps` (default 5), `--bench_batch_size` (default 128),
  `--bench_warmup` (default 3). All have sensible defaults; running
  `python main.py --mode attack_bench --dataset 2016.10a --ckpt_path
  ./checkpoint` produces the full bench out of the box.
- **D-15:** Attack construction reuses
  `util/sigguard_eval.py:create_attack()` plus the existing
  `util/adv_attack.py:Model01Wrapper` for IQ↔[0,1] normalization. Same
  factory `util/adv_training.py:AdversarialTrainer` already uses. No
  duplicated attack registry, no parallel codepath.
- **D-16:** Device loop pattern: single invocation runs both CPU and GPU
  passes in sequence and emits a single CSV with both. The bench
  function takes the host-side tensors once, moves them per device pass,
  and tears down. Halt early with a clear error if CUDA is unavailable
  on the host (the project's PROJECT.md constraints say single-GPU host).

### Claude's Discretion
- Which AWN checkpoint to load by default (base `2016.10a_AWN.pkl` is
  most reproducible; AT/ft variants are also valid). Planner picks the
  base checkpoint unless a strong reason emerges.
- Exact tqdm bar formatting and log layout.
- Sample stratification implementation detail (round-robin across
  (mod, snr) cells vs fixed seed shuffle of the indexed test split).
- Bar chart color palette (must match ieee_style; otherwise
  Claude's choice).
- Whether to also dump a JSON metrics file alongside the CSV (mirrors
  `util/bench.py:run_attack_bench` precedent — likely yes).

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Existing benchmark scaffold (the closest precedent)
- `util/bench.py` — Existing `run_attack_bench()` for CW vs spectral.
  Pattern to mirror: chunked iteration, perf_counter timing, tqdm,
  JSON metrics dump. Different scope (this phase is 5 attacks × 2
  devices), so a new module is appropriate (D-13).

### Attack factory & wrapper (REUSE — do not reimplement)
- `util/sigguard_eval.py` — Defines `create_attack(name, model_wrapper,
  cfg)` and `generate_adversarial(attack, x, y)`. Source of paper-default
  hyperparameters for each attack.
- `util/adv_attack.py` — Defines `Model01Wrapper`,
  `iq_to_ta_input_minmax`, `ta_output_to_iq_minmax`. Required for
  every torchattacks call on IQ data.
- `util/adv_training.py` — Reference implementation of how to use the
  factory + wrapper together inside a tight loop.

### Paper integration & figure styling
- `paper/figures/ieee_style.py` — Project-wide matplotlib style for
  paper figures. The new bar chart MUST use this.
- `paper/latex/figures/` — Where the new `attack_bench_latency.pdf`
  belongs.
- `paper/reproduce.sh` — End-to-end reproducibility script. Phase 6
  may add the `--mode attack_bench` invocation here.

### Project conventions
- `main.py` — Mode dispatcher. New `--mode attack_bench` handler goes
  in the same `elif args.mode == ...` chain (around line 417 next to
  the existing `adv_bench` mode).
- `util/logger.py` — `create_logger()` and `AverageMeter`. Use these
  rather than print().
- `util/config.py` — Config flow. Any new CLI flags must merge cleanly
  via `merge_args2cfg`.
- `util/multi_attack_eval.py` — Reference for iterating multiple
  attacks while reusing the factory; pattern for sample stratification
  across (snr, mod) cells.
- `CLAUDE.md` — Project guidelines, especially the "Epsilon
  Configuration for RF IQ Data" section that explains why ta_box and
  attack_eps matter.
- `.planning/PROJECT.md` — Constraints (single-GPU host, RML2016.10a
  only, paper attacks list).
- `.planning/codebase/STACK.md`, `.planning/codebase/CONVENTIONS.md` —
  Stack and naming conventions.

### Prior phase context (consistency)
- `.planning/phases/05-at-evaluation/05-CONTEXT.md` — D-05 of Phase 5
  locks the 5-attack list (cw, eadl1, eaden, fgsm, pgd). Phase 7 uses
  the same list.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `util/sigguard_eval.py:create_attack` — Already handles all 5 paper
  attacks with paper-default hyperparameters. Plug-and-play for the
  bench.
- `util/adv_attack.py:Model01Wrapper`, `iq_to_ta_input_minmax`,
  `ta_output_to_iq_minmax` — IQ↔[0,1] adapters required for every
  torchattacks invocation. Already exercised by Phase 4 AT training
  and the v1.0 sigguard_eval pipeline.
- `util/bench.py:run_attack_bench` — Pre-existing benchmark scaffold.
  Reference for tqdm + perf_counter + JSON metrics conventions, but
  too narrow in scope to extend cleanly.
- `util/logger.py:create_logger`, `util/logger.py:AverageMeter` —
  Standard logging plumbing.
- `data_loader/data_loader.py:Create_Data_Loader` — Test split loading
  with `mod_filter`/`snr_filter`. Used to draw the stratified N=512
  sample.

### Established Patterns
- Mode-based dispatch in main.py with `argparse` flags and
  `merge_args2cfg(cfg, vars(args))`. Each mode is one `elif` block.
- Output goes to `inference/<dataset>_<index>/result/`, auto-created
  by Config.
- All adversarial-attack callers go through `Model01Wrapper`, never
  directly through the AWN model. AWN's `(logit, regu_sum)` return
  shape is incompatible with torchattacks otherwise.
- Paper figures use `paper/figures/ieee_style.py` for consistent
  styling; rendered to `paper/latex/figures/*.pdf`.

### Integration Points
- `main.py` — New `--mode attack_bench` handler.
- `inference/<dataset>_*/result/` — New `attack_bench.csv` and
  companion `attack_bench_env.json`.
- `paper/latex/figures/` — New `attack_bench_latency.pdf` (Phase 6
  decides where it lands in the manuscript).
- Phase 6 paper update — Phase 7's CSV/PDF feed Phase 6 narrative
  as a "Computational Cost" appendix or discussion paragraph.

</code_context>

<specifics>
## Specific Ideas

- Mirror the existing `util/bench.py` JSON-metrics dump alongside the
  CSV — reviewers and CI scripts can consume either format.
- The bar chart should use a log-scale y-axis (CW/EAD will be ~100×
  FGSM, so a linear y compresses the lower bars to invisibility).
- Stratify the N=512 sample across all (mod, snr) cells so the
  benchmark doesn't accidentally measure an artifact of one easy SNR
  band.
- Re-load model state_dict between the CPU and GPU passes as a
  belt-and-suspenders against any in-place mutation by an attack.

</specifics>

<deferred>
## Deferred Ideas

- **Batch-size sweep (batch=1 streaming, 32, 128)** — D-07 fixes batch=128.
  A streaming/per-arrival latency story is a separate, larger phase
  (would also need queue-management discussion and is more about
  deployment than benchmarking). Note for backlog.
- **Steps sweep per attack** — Sweep CW/EAD/PGD across step counts to
  build a steps-vs-latency curve. Deferred; v1.1 paper integration only
  needs the paper-default operating point.
- **ASR alongside latency in the same CSV** — Could be added cheaply
  but Phase 5 already covers ASR. Out of scope.
- **CUDA Graph / torch.compile speedups** — Could materially change CW
  numbers. Out of scope; the bench measures the codebase as it is.
- **Multi-GPU or per-GPU comparison** — Project constraint is
  single-GPU.
- **Over-the-air timing** — Out of scope for the project entirely.

</deferred>

---

*Phase: 07-benchmark-attack-generation-time-per-sample-cpu-vs-gpu-acros*
*Context gathered: 2026-04-26*
