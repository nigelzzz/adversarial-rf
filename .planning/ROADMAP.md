# Roadmap: Real-Time Defense Pipeline for Adversarial AMC

## Overview

Starting from an existing AWN classifier with attack infrastructure, this roadmap builds the classical filter baselines and unified pipeline (Phase 1), runs the full defense-vs-attack experimental matrix to produce paper-quality result tables (Phase 2), then writes and submits the complete IEEE TCCN/TWC manuscript (Phase 3). Each phase delivers a coherent, verifiable capability that the next phase depends on.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [ ] **Phase 1: Defense Implementations** - Unified pipeline and all classical/RS baselines implemented and validated
- [ ] **Phase 2: Experimental Results** - Full defense-vs-attack evaluation matrix producing all paper tables and figures data
- [ ] **Phase 3: Paper** - Complete IEEE TCCN/TWC manuscript with publication-quality figures and reproducibility scripts

## Phase Details

### Phase 1: Defense Implementations
**Goal**: All defenses exist, are validated, and can be dispatched through a common interface
**Depends on**: Nothing (brownfield — existing AWN, torchattacks, FFT Top-K infrastructure)
**Requirements**: PIPE-01, PIPE-02, PIPE-03, BASE-01, BASE-02, BASE-03, BASE-04, BASE-05, BASE-06, BASE-07
**Success Criteria** (what must be TRUE):
  1. A single function call runs the full detect→recover→classify pipeline on a batch of IQ signals and returns predictions with latency breakdown
  2. Each of the five classical filters (Kalman, Wiener, Savitzky-Golay, Gaussian, FIR) and randomized smoothing can be invoked by name through DEFENSE_REGISTRY with no additional setup
  3. Running the unified pipeline on clean RML2016.10a test signals produces accuracy within 2% of baseline AWN (PIPE-03 verified)
  4. Parameter calibration sweep has been run for each baseline and best parameters are recorded in a config or docstring
  5. GPU-native filters (Gaussian, FIR) show measurably lower latency than CPU-fallback filters (Kalman, Wiener) in the latency benchmark output
**Plans:** 2/3 plans executed

Plans:
- [x] 01-01-PLAN.md — Classical filter baselines (Kalman, Wiener, SG, Gaussian, FIR)
- [x] 01-02-PLAN.md — Defense registry, unified pipeline, randomized smoothing
- [x] 01-03-PLAN.md — Parameter calibration sweep, latency benchmark, clean accuracy validation

### Phase 2: Experimental Results
**Goal**: All numerical results needed for paper tables and figures exist as validated CSV files
**Depends on**: Phase 1
**Requirements**: EVAL-01, EVAL-02, EVAL-03, EVAL-04, EVAL-05
**Success Criteria** (what must be TRUE):
  1. A single command (`--mode defense_compare`) produces a CSV with rows for every defense (unified pipeline, 5 classical filters, randomized smoothing, no-defense baseline) crossed with every required attack (CW, EAD L1, EAD EN, FGSM, PGD) and key SNR points
  2. Confusion matrices exist for CW and EAD attacks both before and after the unified pipeline defense
  3. Perturbation budget curves (accuracy vs epsilon) exist for each attack type showing that attack effectiveness is real (undefended accuracy drops substantially at chosen epsilon values)
  4. The unified pipeline row in the comparison table outperforms every classical filter baseline on at least the two strongest attacks (CW, EAD)
**Plans**: TBD

### Phase 3: Paper
**Goal**: A submission-ready IEEE TCCN/TWC manuscript with all required sections, figures, and reproducibility support
**Depends on**: Phase 2
**Requirements**: PAPER-01, PAPER-02, PAPER-03, PAPER-04, PAPER-05, PAPER-06, PAPER-07, PAPER-08, PAPER-09, PAPER-10
**Success Criteria** (what must be TRUE):
  1. A complete IEEEtran journal LaTeX manuscript compiles without errors and covers all required sections (introduction, related work, system model and threat model, proposed method, experimental setup, results and analysis, conclusion)
  2. All figures (accuracy-vs-SNR curves, confusion matrices, defense comparison bar charts, frequency-domain spectra) render as PDF at 300 dpi with IEEE-matching fonts and are referenced from the manuscript
  3. A single shell script re-runs all experiments from raw data and regenerates all result CSVs and figures referenced in the paper
**Plans**: TBD
**UI hint**: yes

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Defense Implementations | 2/3 | In Progress|  |
| 2. Experimental Results | 0/TBD | Not started | - |
| 3. Paper | 0/TBD | Not started | - |
