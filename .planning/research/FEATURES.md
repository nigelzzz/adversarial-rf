# Features Research: Adversarial Defense for AMC

**Research Date:** 2026-03-31
**Domain:** Real-time adversarial defense for automatic modulation classification
**Target Venue:** IEEE TCCN/TWC

## Table Stakes (Must-have for IEEE TCCN/TWC acceptance)

### Evaluation Against Multiple Attack Types
- **Complexity:** Low (existing infrastructure)
- **Rationale:** Reviewers expect defense evaluated against at least 3-4 attack families
- **Required attacks:** FGSM, PGD (Linf gradient), CW (L2 optimization), EAD (L1/EN)
- **Dependencies:** torchattacks integration already exists

### Per-SNR Accuracy Breakdown
- **Complexity:** Low (existing infrastructure)
- **Rationale:** SNR is the primary axis of variation in AMC papers; defense must work across SNR range
- **Expected:** Tables showing accuracy at key SNR points (-10, 0, 6, 10, 18 dB)
- **Dependencies:** Data loader already supports SNR filtering

### Confusion Matrix Analysis
- **Complexity:** Low (existing infrastructure)
- **Rationale:** Shows which modulations are most vulnerable/best defended
- **Expected:** Before/after defense confusion matrices for strongest attacks (CW, EAD)

### Baseline Comparisons
- **Complexity:** Medium (new implementations needed)
- **Rationale:** Reviewers require comparison against existing defense approaches
- **Required baselines:** Classical filters (Kalman, Wiener, Savitzky-Golay, Gaussian, FIR), randomized smoothing
- **Dependencies:** Must implement all baseline filters from scratch

### Clean Accuracy Preservation
- **Complexity:** Low
- **Rationale:** Defense must not significantly degrade clean signal classification
- **Expected:** Show <2% accuracy drop on unperturbed signals

### Perturbation Budget Analysis
- **Complexity:** Medium
- **Rationale:** Evaluate defense across different epsilon values
- **Expected:** Accuracy vs epsilon curves for each attack type

## Differentiators (Competitive advantages)

### Unified Pipeline Architecture
- **Complexity:** Medium
- **Rationale:** Most prior work evaluates individual defenses in isolation; a unified detect→recover→classify pipeline is novel for RF/AMC
- **Advantage:** Single inference path, no manual defense selection
- **Dependencies:** Autoencoder detector + FFT Top-K recovery + AWN classifier

### Real-Time Latency Analysis
- **Complexity:** Medium
- **Rationale:** Most adversarial defense papers ignore computational cost; proving real-time feasibility is a strong differentiator
- **Expected:** Per-component latency breakdown, comparison of defense overhead vs classification time
- **Dependencies:** All defenses must be implemented before benchmarking

### Detector-Gated Recovery
- **Complexity:** Low (partially exists as ae_fft_topk)
- **Rationale:** Adaptive defense that only applies recovery when adversarial perturbation detected, preserving clean signal quality
- **Advantage:** Better clean accuracy than always-on defenses

### Frequency-Domain Visualization
- **Complexity:** Low (plotting infrastructure exists)
- **Rationale:** Visual evidence of how attacks manifest in frequency domain and how defense removes them
- **Expected:** Spectral plots showing clean → attacked → recovered signals

## Anti-Features (Deliberately exclude)

### Adaptive Attack Evaluation (Carlini-style)
- **Why exclude:** Full adaptive attack evaluation (attacker knows defense) requires significant extra work and is typically a follow-up paper concern. Mention as future work.
- **Risk if included:** Scope creep, potential negative results that undermine the paper

### Multi-Dataset Evaluation
- **Why exclude:** RML2016.10a is the standard benchmark; adding RML2018 doubles experiment time without proportional value for first submission
- **Risk if included:** Timeline blow-up

### End-to-End Adversarial Training
- **Why exclude:** Already explored via curriculum finetuning; not the paper's contribution. The contribution is the defense pipeline, not a training procedure.

### Over-the-Air Validation
- **Why exclude:** Requires hardware setup, adds months. Mention as future work.

## Feature Dependencies

```
Baseline filters ──┐
                    ├── Comparison evaluation ── Paper tables/figures
Unified pipeline ──┘
                    │
Detector training ──┤── Detector-gated recovery
                    │
FFT Top-K ─────────┤── Unified pipeline
                    │
Latency benchmark ──── All defenses implemented first
```

## Complexity Summary

| Feature | Complexity | Existing Code | Priority |
|---------|-----------|---------------|----------|
| Multi-attack evaluation | Low | Yes | P0 |
| Per-SNR breakdown | Low | Yes | P0 |
| Confusion matrices | Low | Yes | P0 |
| Clean accuracy preservation | Low | Yes | P0 |
| Baseline filter implementations | Medium | No | P0 |
| Unified pipeline integration | Medium | Partial | P0 |
| Perturbation budget analysis | Medium | Partial | P1 |
| Real-time latency analysis | Medium | No | P1 |
| Detector-gated recovery | Low | Yes | P0 |
| Frequency-domain visualization | Low | Yes | P1 |
| Paper draft (LaTeX) | High | No | P0 |
