# Real-Time Defense Pipeline for Adversarial Attacks on AMC

## What This Is

A unified real-time defense framework for automatic modulation classification (AMC) that combines adversarial detection, frequency-domain recovery (FFT Top-K), and robust classification into a single pipeline. Targets IEEE TCCN/TWC submission using RML2016.10a data, comparing against classical signal processing baselines (Kalman, Wiener, Savitzky-Golay, Gaussian, FIR filters) and randomized smoothing.

## Core Value

Demonstrate that a unified detect→recover→classify pipeline outperforms individual classical filtering defenses against optimization-based adversarial attacks (CW, EAD) on RF signals, while maintaining real-time feasibility.

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

### Active

- [ ] Unified detect→recover→classify pipeline as single inference path
- [ ] Baseline implementations: Kalman filter, Wiener filter, Savitzky-Golay filter, Gaussian filter, FIR filter
- [ ] Randomized smoothing baseline (σ=0.01, k=1)
- [ ] Comparative evaluation tables: unified pipeline vs each baseline vs no defense
- [ ] Per-SNR and per-modulation accuracy breakdown for all defenses
- [ ] Latency/throughput benchmarks proving real-time feasibility
- [ ] LaTeX paper draft for IEEE TCCN/TWC (intro, related work, method, experiments, conclusion)
- [ ] Publication-quality figures: accuracy curves, confusion matrices, defense comparison charts
- [ ] Reproducibility scripts to regenerate all results from scratch

### Out of Scope

- RML2018.01a evaluation — focus on RML2016.10a for this submission
- Novel attack development — paper is defense-focused
- Adversarial training as primary defense — already explored via finetuning, not the paper's contribution
- Over-the-air validation — simulation-only for this submission
- GUI or web interface — research code only

## Context

- **Existing codebase**: AWN (Adaptive Wavelet Network) for AMC with extensive adversarial attack/defense infrastructure
- **Model**: Pretrained AWN checkpoint at `./checkpoint/2016.10a_AWN.pkl`, finetuned variant at `./checkpoint/2016.10a_AWN_ft.pkl`
- **Dataset**: RML2016.10a (11 modulations, 220K samples, IQ format [2, 128])
- **Attack library**: torchattacks with 17 attack methods, custom wrappers for IQ normalization (`Model01Wrapper`, `iq_to_ta_input_minmax`)
- **Detection**: Conv autoencoder detector in `util/detector.py`, trained/calibrated via `main.py --mode train_detector/calibrate_detector`
- **Recovery**: FFT Top-K in `util/defense.py`, gated by detector (`ae_fft_topk` mode)
- **Prior experiments**: Extensive results in `inference/` and `reports/` directories
- **Baselines needed**: Classical signal processing filters not yet implemented as defense baselines
- **Timeline**: ~1 month to submission

## Constraints

- **Data**: RML2016.10a only (11 classes, SNR range -20 to +18 dB)
- **Timeline**: ~1 month to submission-ready paper
- **Compute**: Single GPU (existing setup)
- **Format**: IEEE transaction paper format (double-column LaTeX)
- **Attacks**: Must cover CW (L2), EAD (L1, EN), FGSM, PGD at minimum
- **Epsilon**: Must use RF-appropriate epsilon values (not image defaults)

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Unified pipeline (detect+recover+classify) as main contribution | Novel for RF domain, practical for deployment | — Pending |
| Classical filters as baselines (not other ML defenses) | Shows improvement over signal processing approach | — Pending |
| RML2016.10a only | Standard benchmark, sufficient for initial submission | — Pending |
| IEEE TCCN/TWC as target venue | Matches RF + adversarial ML topic | — Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd:transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd:complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-03-31 after initialization*
