# Phase 7: Benchmark attack generation time per sample (CPU vs GPU) across 5 attacks - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-26
**Phase:** 07-benchmark-attack-generation-time-per-sample-cpu-vs-gpu-acros
**Areas discussed:** Hardware & timing methodology, Attack hyperparameters & sample scope, Output deliverables & paper integration, Code structure & invocation

---

## Hardware & Timing Methodology

### Q1: How should the CPU side of the benchmark be run on your single-GPU host?

| Option | Description | Selected |
|--------|-------------|----------|
| Same script, --device cpu toggle | Run the same benchmark twice with cfg.device='cpu' and cfg.device='cuda'. Simplest, fully reproducible. | ✓ |
| Two invocations, CUDA_VISIBLE_DEVICES gating | Belt-and-suspenders against accidental CUDA fallbacks. | |
| Single process, sequential CPU then GPU | One run, sequential. Avoids two-invocation orchestration. | |

**User's choice:** Same script, --device cpu toggle (Recommended)

### Q2: What warmup strategy before measuring each attack?

| Option | Description | Selected |
|--------|-------------|----------|
| Per-attack warmup, ~3-5 iters discarded | Catches lazy CUDA init, cuDNN autotune, attack hook setup. | ✓ |
| Single global warmup at script start | One forward + one short PGD, then time all attacks. | |
| No warmup, report cold-start | Time as-is. | |

**User's choice:** Per-attack warmup, ~3-5 iters discarded (Recommended)

### Q3: How to handle GPU async / CPU timing alignment?

| Option | Description | Selected |
|--------|-------------|----------|
| torch.cuda.synchronize() before & after each timed block | Standard correct pattern. | ✓ |
| torch.cuda.Event-based timing | Lower overhead, GPU-only. | |
| Wall time only, no sync | Risks under-counting. | |

**User's choice:** torch.cuda.synchronize() before & after each timed block (Recommended)

### Q4: What statistics to report per (attack, device) cell?

| Option | Description | Selected |
|--------|-------------|----------|
| Mean ms/sample + std over N reps | Single representative number per cell, easy table integration. | ✓ |
| Full distribution: mean, median, p95, p99 | Richer view, surfaces tail latency. | |
| Min over N reps | Best-case throughput. | |

**User's choice:** Mean ms/sample + std over N reps (Recommended) — N=5 reps

---

## Attack Hyperparameters & Sample Scope

### Q1: Per-attack hyperparameters — what values to use when timing each attack?

| Option | Description | Selected |
|--------|-------------|----------|
| Paper defaults across all attacks | Match v1.0 paper config exactly. | ✓ |
| Normalized step count (steps=100 for all) | Cleaner cross-attack comparison. | |
| Sweep steps per attack | Steps-vs-latency curve. | |

**User's choice:** Paper defaults across all attacks (Recommended)

### Q2: How many samples to time per (attack, device) cell?

| Option | Description | Selected |
|--------|-------------|----------|
| Fixed N=512 from RML2016.10a test split | Stratified across (mod, snr). | ✓ |
| Single SNR slice (SNR=18 dB), 512 samples | Removes signal-difficulty confounder. | |
| Full test split (~22k samples) | Maximum confidence, hours on CPU for CW/EAD. | |

**User's choice:** Fixed N=512 from RML2016.10a test split (Recommended)

### Q3: Batch size dimension — sweep or fixed?

| Option | Description | Selected |
|--------|-------------|----------|
| Fixed batch_size=128, single column | Match paper test_batch_size. | ✓ |
| Sweep {1, 32, 128} — three columns | Captures streaming + throughput. | |
| Sweep {1, 128} — streaming vs batched only | Two operating points. | |

**User's choice:** Fixed batch_size=128, single column (Recommended)

### Q4: Should the benchmark also report attack success rate (ASR) alongside latency?

| Option | Description | Selected |
|--------|-------------|----------|
| Latency only | ASR already covered by Phase 5. | ✓ |
| Latency + ASR (cheap to add) | Validates attacks worked. | |
| Latency only, with sanity asserts | Internal sanity check, no reporting. | |

**User's choice:** Latency only (Recommended)

---

## Output Deliverables & Paper Integration

### Q1: Primary CSV deliverable — schema and location?

| Option | Description | Selected |
|--------|-------------|----------|
| Standalone attack_bench.csv under inference/<dataset>_*/result/ | One CSV, 10 rows. | ✓ |
| Append to defense_compare.csv as latency columns | Tighter Phase 6 integration. | |
| Both — standalone + summary row | Full coverage. | |

**User's choice:** Standalone attack_bench.csv under inference/<dataset>_*/result/ (Recommended)

### Q2: Should this phase produce a paper figure for Phase 6 camera-ready?

| Option | Description | Selected |
|--------|-------------|----------|
| Bar chart: attack-generation latency CPU vs GPU | Grouped bar chart, 5 attacks × 2 devices, error bars. | ✓ |
| LaTeX table only, no figure | CSV → LaTeX. | |
| Both table and figure | Maximum coverage. | |
| CSV only, paper integration deferred to Phase 6 | Minimum scope. | |

**User's choice:** Bar chart: attack-generation latency CPU vs GPU (Recommended)

### Q3: Logging — what runtime output should the benchmark emit?

| Option | Description | Selected |
|--------|-------------|----------|
| Structured logger + tqdm progress per attack | Match existing util/bench.py and util/multi_attack_eval.py. | ✓ |
| Print-only, no log file | Simpler. | |
| JSON metrics file alongside CSV | Programmatic consumption. | |

**User's choice:** Structured logger + tqdm progress per attack (Recommended)

### Q4: Should the benchmark record machine/environment metadata for reproducibility?

| Option | Description | Selected |
|--------|-------------|----------|
| Yes — record GPU name, CPU model, torch version, CUDA version | Cheap, paper-grade rigor. | ✓ |
| Yes, but only in JSON metadata file | CSV stays clean. | |
| No — user records env manually | Out of scope. | |

**User's choice:** Yes — record GPU name, CPU model, torch version, CUDA version (Recommended). Embedded as both CSV header comment and companion JSON.

---

## Code Structure & Invocation

### Q1: Where should the new benchmarking code live?

| Option | Description | Selected |
|--------|-------------|----------|
| New util/attack_bench.py module | Clean separation. | ✓ |
| Extend util/bench.py with a new function | Same file, two functions. | |
| Refactor util/bench.py to handle both flows | Generalize existing function. | |

**User's choice:** New util/attack_bench.py module (Recommended)

### Q2: How should the benchmark be invoked from main.py?

| Option | Description | Selected |
|--------|-------------|----------|
| New main.py mode: --mode attack_bench | Distinct from existing --mode adv_bench. | ✓ |
| Extend --mode adv_bench with a flag | Couples two distinct benchmarks. | |
| Standalone script: scripts/attack_bench.py | Outside main.py. | |

**User's choice:** New main.py mode: --mode attack_bench (Recommended)

### Q3: How should attacks be constructed inside the benchmark?

| Option | Description | Selected |
|--------|-------------|----------|
| Reuse util/sigguard_eval.py:create_attack + Model01Wrapper | Maximum reuse. | ✓ |
| Mirror util/multi_attack_eval.py pattern | Slight divergence. | |
| Hand-roll a thin attack registry inside attack_bench.py | Self-contained but duplicative. | |

**User's choice:** Reuse util/sigguard_eval.py:create_attack + Model01Wrapper (Recommended)

### Q4: How should device switching work inside one process?

| Option | Description | Selected |
|--------|-------------|----------|
| Loop over [cpu, cuda] inside the bench function | Single invocation, single CSV. | ✓ |
| Two separate invocations, --bench_device flag | Manual orchestration. | |
| Auto-loop both devices only when CUDA available | Graceful degradation for portability. | |

**User's choice:** Loop over [cpu, cuda] inside the bench function, model.to(device) per iter (Recommended)

---

## Claude's Discretion

- Which AWN checkpoint to load by default (planner picks base 2016.10a_AWN.pkl unless a strong reason emerges)
- Exact tqdm bar formatting and log layout
- Sample stratification implementation detail
- Bar chart color palette (must match ieee_style)
- Whether to also dump a JSON metrics file alongside the CSV (likely yes, mirrors util/bench.py)

## Deferred Ideas

- Batch-size sweep (batch=1 streaming, 32, 128) — separate phase
- Steps sweep per attack — separate phase
- ASR alongside latency in the same CSV — Phase 5 already covers ASR
- CUDA Graph / torch.compile speedups — out of scope
- Multi-GPU or per-GPU comparison — single-GPU constraint
- Over-the-air timing — out of scope for project
