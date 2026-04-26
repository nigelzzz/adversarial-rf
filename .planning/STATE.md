---
gsd_state_version: 1.0
milestone: v1.1
milestone_name: Robustness Baselines
status: executing
stopped_at: Phase 7 context gathered
last_updated: "2026-04-26T12:04:33.118Z"
last_activity: 2026-04-26
progress:
  total_phases: 3
  completed_phases: 1
  total_plans: 5
  completed_plans: 4
  percent: 80
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-04-15)

**Core value:** A unified detect→recover→classify pipeline that outperforms
classical filtering defenses against optimization-based adversarial attacks
on RF signals, while maintaining real-time feasibility.
**Current focus:** Phase 04 — adversarial-training
**Previous:** v1.0 Paper Submission Package shipped 2026-04-15 (see milestones/v1.0-ROADMAP.md)

## Current Position

Milestone: v1.1 Robustness Baselines
Phase: 07
Plan: Not started
Status: Executing Phase 04
Last activity: 2026-04-26

Progress: [░░░░░░░░░░] 0%  (0/3 v1.1 phases)

## Accumulated Context

### Decisions

Decisions from v1.0 are archived in `milestones/v1.0-ROADMAP.md` and
`.planning/PROJECT.md` Key Decisions table. Top decisions still in force:

- Unified detect→recover→classify pipeline is the main contribution
- Adaptive-K v2 (shared FFT + spectral flatness routing + SNR-adaptive cap) is the recovery core
- RML2016.10a is the working dataset; RML2018.01a deferred
- IEEE TCCN/TWC is the target venue
- GPU-native filters (depthwise conv1d) used to preserve honest latency claims

**v1.1 decisions:**

- Adversarial training is a baseline, not the main contribution; Adaptive-K remains primary
- AT warm-starts from `./checkpoint/2016.10a_AWN.pkl` (not scratch) to save ~10x compute
- CW held out from AT training; used only in evaluation to test generalization
- Mixed clean+adversarial loss (alpha=0.5 default) prevents forgetting of analog mods
- AT training pattern parallels `synth_finetune.py` finetuning structure
- CRTD requirements (camera-ready debt) bundled into Phase 6 with paper update (coarse granularity; both produce paper artifacts; CRTD-01 enables clean figure fonts for PAPRU figures)

### Pending Todos

- Plan Phase 4: Adversarial Training (AT-01..05)
- Implement adversarial training script (pattern: synth_finetune.py with attack loop replacing synthetic generation)
- Confirm alpha=0.5 default preserves WBFM/AM-DSB/AM-SSB accuracy before full AT run

### Roadmap Evolution

- Phase 7 added: Benchmark attack generation time per sample (CPU vs GPU) across 5 attacks

### Blockers/Concerns

v1.0 tech debt (all addressed in Phase 6):

- Stale `status: gaps_found` in phases 02 and 03 VERIFICATION.md frontmatter
  (gaps closed by commits 69b2595, de3da10 and plan 02-05)

- `freq_spectra_cw.pdf` uses placeholder synthetic spectra in environments
  without venv/dataset (documented limitation)

- `text.usetex=False` in `ieee_style.py` → matplotlib fallback fonts; enable
  for camera-ready submission

- NYCU-thesis-template submodule left dirty (unrelated to paper artifacts)

## Session Continuity

Last session: 2026-04-26T07:39:18.338Z
Stopped at: Phase 7 context gathered
Resume file: .planning/phases/07-benchmark-attack-generation-time-per-sample-cpu-vs-gpu-acros/07-CONTEXT.md
