---
phase: 03-paper
plan: "02"
subsystem: paper-sections
tags: [latex, ieee-tccn, adversarial-ml, amc, paper-writing]
dependency_graph:
  requires: [03-01]
  provides: [introduction, related-work, system-model, proposed-method]
  affects: [03-03]
tech_stack:
  added: []
  patterns:
    - IEEE TCCN double-column LaTeX section structure
    - IEEEtran algorithmic environments (algpseudocode)
    - LaTeX equation environments with aligned attack objectives
key_files:
  created: []
  modified:
    - paper/latex/sections/introduction.tex
    - paper/latex/sections/related_work.tex
    - paper/latex/sections/system_model.tex
    - paper/latex/sections/proposed_method.tex
decisions:
  - "Removed \\nocite{*} from main.tex not needed — kept to ensure all bib entries compile cleanly"
  - "proposed_method.tex includes CRC table inline (adapted from USENIX draft) to motivate control-plane insight without a separate section file"
  - "Classical filter descriptions kept concise (3-4 sentences each) since implementation details are in Phase 1 code"
  - "Adaptive-K Algorithm 2 uses cumulative energy threshold eta=0.95 as primary selector, not elbow detection, matching actual implementation"
metrics:
  duration_minutes: 4
  completed_date: "2026-04-06"
  tasks_completed: 2
  tasks_total: 2
  files_created: 0
  files_modified: 4
---

# Phase 03 Plan 02: Paper Sections I-IV Summary

Four complete IEEE TCCN paper sections replacing stubs: introduction with 4 numbered contributions, related work with 3 subsections and 22+ citations, system model with attack equations and threat model, proposed method with 2 algorithm environments covering spectral-gated and adaptive-K defenses.

## What Was Built

### Task 1: introduction.tex and related_work.tex

**`paper/latex/sections/introduction.tex`** (97 lines, 11 citations):
- Paragraph 1: AMC importance in cognitive radio, spectrum monitoring; DNN classifiers achieve 91%+ on RML2016.10a; adversarial threat with citations to li2023awn, otoole2016rml, silvaco2023adversarial_rf, szegedy2014intriguing, lin2020tactics
- Paragraph 2: Problem statement — adversarial perturbations collapse accuracy to 0%; existing defenses (adversarial training, input transformation) and the gap (no systematic classical filter comparison); cites carlini2017towards, chen2020ead, madry2018towards, guo2018input, xu2018feature, haykin2002adaptive, proakis2006digital
- Paragraph 3: Our approach — adaptive spectral defense, key insight about high-frequency perturbation energy, benchmark against 5 classical filters; cites cohen2019certified
- Paragraph 4: 4 numbered contributions: (1) control-plane attack characterization with CRC data, (2) adaptive-K FFT defense with knee-point detection, (3) systematic classical filter comparison (outperforms by 4-8% on CW/EAD), (4) spectral-gated routing for AM-SSB recovery (0% → 20-65%)
- Paragraph 5: Paper organization (7 sections)

**`paper/latex/sections/related_work.tex`** (124 lines, 22 citations):
- Subsection A (Adversarial Attacks on AMC): Lin 2020 INFOCOM pioneering work, Flowers/Buehrer evaluation, Yang 2019 geometry, Bahramali 2021 over-the-air attacks, Silvaco 2023 survey, Zhang 2024 stealthy attacks, Kim 2024 frequency-domain attacks; contrasts gradient-based vs. optimization-based effectiveness
- Subsection B (Defense Methods for AMC): adversarial training (Madry, Tramer), defensive distillation (Papernot), input transformations (Guo, Xu, Dziugaite), randomized smoothing (Cohen), RF-specific frequency defenses (Kim 2024)
- Subsection C (Classical Signal Processing Filters): Wiener MMSE estimation, Kalman state-space, Savitzky-Golay polynomial smoothing, Gaussian/FIR lowpass; hypothesis vs. adversarial reality discussion
- Closing paragraph positioning our contribution relative to literature

### Task 2: system_model.tex and proposed_method.tex

**`paper/latex/sections/system_model.tex`** (148 lines, 14 citations):
- Subsection A (Signal Representation and Dataset): IQ tensor format [N, 2, T], T=128, RML2016.10a 11 classes SNR -20 to +18 dB, AWN 91%+ clean accuracy
- Subsection B (AWN Classifier Architecture): 3-stage pipeline with equations — convolutional feature extraction, adaptive wavelet decomposition (Predictor P, Updator U) with regularization loss L = L_CE + R_d + R_c, SE attention + FC classification; cites li2023awn, ioffe2015batch, hu2018squeeze, kingma2015adam
- Subsection C (Attack Models): Formal objectives for all 4 attacks with equations — FGSM (sign gradient), PGD (projected iteration), CW L2 (min ||delta||_2 + c*f), EAD (beta*||delta||_1 + ||delta||_2 + c*f); normalization mode description; cites goodfellow2015fgsm, madry2018pgd, carlini2017towards, chen2020ead, croce2020reliable, kim2020torchattacks
- Subsection D (Threat Model): White-box attacker capabilities, untargeted goal, norm constraints, defense-unaware model, defender capabilities with real-time latency requirement and 3 deployment contexts (ITU/FCC, CBRS ESC/SAS, military ESM)

**`paper/latex/sections/proposed_method.tex`** (279 lines, 5 citations):
- Subsection A (Control-Plane Attack Insight): Two-track experiment, Table 1 with CRC pass rates for 8 modulations (Adv+Oracle stays 71-100%, Adv+AMC collapses to 0-39%), motivates frequency-domain defense
- Subsection B (FFT Top-K Filtering): 5-step procedure with equations, normalized rFFT → TopK mask → irFFT → denormalize; compact equation hat_x = irFFT(TopK(rFFT(x_tilde), K))
- Subsection C (Spectral-Gated Defense, Algorithm 1): Spectral flatness definition (geometric/arithmetic mean ratio), 18x separation between AM-SSB (~0.55) and narrowband (<0.04), shared-FFT efficiency, Algorithm 1 pseudocode with if/else routing
- Subsection D (Adaptive-K Defense, Algorithm 2): Cumulative energy threshold eta=0.95 for per-sample K selection, K_min/K_max clipping, Algorithm 2 pseudocode; explains narrowband (BPSK: K=10-15) vs. wideband (QAM64: K=20-35) adaptation; cites 1-3% improvement over fixed-K on CW, 3-5% on EAD
- Subsection E (Classical Filter Baselines): 5 filter descriptions with calibration procedure (grid search per-SNR on validation accuracy) — Kalman (Q/R parameters), Wiener (H_k frequency weighting), Savitzky-Golay (degree d, window w), Gaussian (sigma, GPU F.conv1d), FIR (cutoff f_c, order M, GPU depthwise); randomized smoothing (sigma=0.1, k=100 copies) as certified baseline
- Subsection F (Complexity Analysis): O(T log T) for FFT methods, O(T*M) for FIR/Gaussian, O(T) sequential for Kalman (CPU fallback), Wiener frequency domain O(T log T + T), randomized smoothing 100x more expensive

## Verification Results

1. `latexmk -pdf main.tex` — compiles without errors (7-page PDF, 255KB)
2. No undefined citation warnings in main.log
3. `grep -c '\\cite' sections/introduction.tex` — 11 (>= 8 required)
4. `grep -c '\\cite' sections/related_work.tex` — 22 (>= 20 required)
5. `grep -c '\\begin{algorithm}' sections/proposed_method.tex` — 2 (>= 2 required)
6. Line counts: introduction 97, related_work 124, system_model 148, proposed_method 279 (all exceed minimums)

## Deviations from Plan

None — plan executed exactly as written.

## Known Stubs

None in these four sections — all content is complete with real experimental numbers, formal equations, and full algorithmic pseudocode. The sections reference figures (e.g., Figure 1 for defense overview) that are already generated as PDFs from Plan 01.

Note: `experimental_setup.tex`, `results.tex`, and `conclusion.tex` remain as stubs — those are Plan 03 scope.

## Self-Check: PASSED

Files verified:
- `paper/latex/sections/introduction.tex` exists (97 lines)
- `paper/latex/sections/related_work.tex` exists (124 lines)
- `paper/latex/sections/system_model.tex` exists (148 lines)
- `paper/latex/sections/proposed_method.tex` exists (279 lines)
- `paper/latex/main.pdf` generated (7 pages)

Commits verified:
- `4b80207` — Task 1: introduction.tex and related_work.tex
- `608d8c7` — Task 2: system_model.tex and proposed_method.tex
