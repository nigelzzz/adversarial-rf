# Real-Time Defense Pipeline for Adversarial Attacks on AMC

## Current State

**Shipped:** v1.0 Paper Submission Package (2026-04-15) — IEEE TCCN/TWC
manuscript (13 pages, 41 citations, 11 figures) with full reproducible
defense-vs-attack evaluation pipeline. UAT 10/10.

**Active:** v1.1 Robustness Baselines — see Current Milestone below.

## Current Milestone: v1.1 Robustness Baselines

**Goal:** Strengthen v1.0 paper's defense narrative by adding an
adversarial-training baseline and closing camera-ready tech debt.

**Target features:**

- Adversarial-training baseline: finetune AWN on mixed FGSM/PGD/EAD-L1/EAD-EN
  with CW held out for evaluation, mixed clean loss to preserve analog mods
- Adaptive-K + AT composition study: layered defense comparison, add rows
  to the paper's defense table
- v1.0 camera-ready tech debt: `text.usetex=True`, regenerate real
  `freq_spectra_cw.pdf`, clean stale VERIFICATION frontmatter

## What This Is

A unified real-time defense framework for automatic modulation classification
(AMC) that combines adversarial detection, frequency-domain recovery (FFT
Top-K via Adaptive-K v2), and robust classification into a single pipeline.
Shipped an IEEE TCCN/TWC submission on RML2016.10a comparing against five
classical signal-processing baselines (Kalman, Wiener, Savitzky-Golay,
Gaussian, FIR) and randomized smoothing.

## Core Value

A unified detect→recover→classify pipeline that outperforms individual
classical filtering defenses against optimization-based adversarial attacks
(CW, EAD) on RF signals, while maintaining real-time feasibility.

**Validated in v1.0:** Adaptive-K v2 beats every classical-filter baseline
on CW and EAD attacks at SNR ≥ 0 dB with a ~1 ms/sample GPU cost.

## Requirements

### Validated

- ✓ AWN model training and evaluation on RML2016.10a — existing
- ✓ CW (L2) and EAD (L1, EN) adversarial attack generation via torchattacks — existing
- ✓ FGSM/PGD/BIM gradient-based attack generation — existing
- ✓ FFT Top-K frequency-domain recovery — existing
- ✓ Autoencoder-based adversarial detection (KL divergence) — existing
- ✓ SigGuard-style evaluation tables (attack acc vs defense acc) — existing
- ✓ Multi-attack evaluation with per-SNR/per-modulation breakdown — existing
- ✓ Synthetic data generation and curriculum finetuning for robust classification — existing
- ✓ Unified detect→recover→classify pipeline as single inference path — v1.0
- ✓ 5 classical filter baselines (Kalman, Wiener, SG, Gaussian, FIR) with calibration — v1.0
- ✓ Randomized smoothing baseline (σ=0.01, k=20) — v1.0
- ✓ Comparative evaluation tables (9 defenses × 5 attacks × 10 SNRs) — v1.0
- ✓ Per-SNR and per-modulation accuracy breakdown — v1.0
- ✓ Latency/throughput benchmarks — v1.0
- ✓ IEEE TCCN/TWC LaTeX manuscript with 7 sections — v1.0
- ✓ Publication-quality figures (11 PDFs, 300 dpi) — v1.0
- ✓ Reproducibility scripts (`paper/reproduce.sh`) — v1.0

### Active

Defined in the next milestone (v1.1) via `/gsd-new-milestone`. Likely
candidates based on project discussion:

- [ ] Adversarial-training baseline (mixed FGSM/PGD/EAD-L1/EAD-EN training, CW held-out)
- [ ] Adaptive-K + AT composition study (does the unified pipeline stack?)
- [ ] Close non-blocking v1.0 tech debt (usetex, stale VERIFICATION frontmatter, freq_spectra_cw real-data regen)

### Out of Scope

| Feature | Reason |
|---------|--------|
| RML2018.01a experiments | Deferred; v1.0 focused on RML2016.10a. Revisit for journal revision if reviewers request. |
| Novel attack development | Paper is defense-focused |
| Over-the-air validation | Requires hardware; listed as future work in paper |
| GUI or web interface | Research code only |
| Multi-model evaluation | AWN-only; other classifiers are future work |
| Adversarial training as **primary** defense | Adaptive-K remains main contribution; AT is a baseline for v1.1 |

## Context

- **Existing codebase**: AWN (Adaptive Wavelet Network) for AMC with extensive adversarial attack/defense infrastructure
- **Model**: Pretrained AWN checkpoint at `./checkpoint/2016.10a_AWN.pkl`, finetuned variant at `./checkpoint/2016.10a_AWN_ft.pkl`
- **Dataset**: RML2016.10a (11 modulations, 220K samples, IQ format [2, 128])
- **Attack library**: torchattacks with 17 attack methods, custom wrappers for IQ normalization (`Model01Wrapper`, `iq_to_ta_input_minmax`)
- **Defense registry**: 9 defenses in `DEFENSE_REGISTRY` (unified, 5 classical, rand_smooth, fft_topk, ae_fft_topk)
- **Experimental results**: CSVs under `inference/2016.10a_165/result/defense_compare/`
- **Paper artifacts**: `paper/latex/main.pdf` (13 pages), `paper/reproduce.sh`
- **v1.0 outcome**: Shipped IEEE TCCN/TWC-ready manuscript with passing UAT

## Constraints

- **Data**: RML2016.10a only (11 classes, SNR range -20 to +18 dB)
- **Compute**: Single GPU (existing setup)
- **Format**: IEEE transaction paper format (double-column LaTeX)
- **Attacks**: Must cover CW (L2), EAD (L1, EN), FGSM, PGD at minimum
- **Epsilon**: Must use RF-appropriate epsilon values (not image defaults)

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Unified pipeline (detect+recover+classify) as main contribution | Novel for RF domain, practical for deployment | ✓ Good — shipped in v1.0, validated on CW/EAD |
| Classical filters as baselines (not other ML defenses) | Shows improvement over signal-processing approach | ✓ Good — 5 baselines provide fair comparison in Table I |
| RML2016.10a only | Standard benchmark, sufficient for initial submission | ✓ Good — kept scope manageable; revisit for journal rev. |
| IEEE TCCN/TWC as target venue | Matches RF + adversarial ML topic | ✓ Good — paper drafted in IEEEtran format |
| Adaptive-K v2 with SNR-adaptive cap over fixed K | Handles both low/high SNR regimes without labels | ✓ Good — core to proposed method |
| Shared-FFT design (1 FFT + 1 IFFT) | Real-time feasibility requirement | ✓ Good — O(T log T) per sample |
| GPU-native depthwise conv1d for Gaussian/FIR | Avoid CPU roundtrip invalidating latency claims | ✓ Good — clean latency story |
| Spectral flatness > 0.4 → quantization (wideband routing) | Top-K fails on AM-SSB; quantization preserves classification | ✓ Good — isolated 18× margin between narrowband and AM-SSB |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition:**
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone:**
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-04-15 after v1.0 Paper Submission Package milestone*
