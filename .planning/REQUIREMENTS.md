# Requirements: Real-Time Defense Pipeline for Adversarial AMC

**Defined:** 2026-03-31
**Core Value:** Demonstrate that a unified detect→recover→classify pipeline outperforms classical filtering defenses against optimization-based adversarial attacks on RF signals

## v1 Requirements

### Defense Pipeline

- [x] **PIPE-01**: Unified detect→recover→classify inference path as single callable function
- [x] **PIPE-02**: Latency benchmark per pipeline component (detector, recovery, classifier) in milliseconds
- [x] **PIPE-03**: Clean accuracy preservation — defense degrades unperturbed accuracy by <2%

### Baseline Implementations

- [x] **BASE-01**: Kalman filter defense baseline with parameter sweep
- [x] **BASE-02**: Wiener filter defense baseline with parameter sweep
- [x] **BASE-03**: Savitzky-Golay filter defense baseline with parameter sweep
- [x] **BASE-04**: Gaussian filter defense baseline with parameter sweep
- [x] **BASE-05**: FIR low-pass filter defense baseline with parameter sweep
- [x] **BASE-06**: Randomized smoothing baseline (σ=0.01, majority vote over k copies)
- [x] **BASE-07**: Parameter calibration sweep for each filter baseline (fair comparison)

### Evaluation

- [ ] **EVAL-01**: Multi-attack comparison table: all defenses vs CW, EAD L1, EAD EN, FGSM, PGD
- [ ] **EVAL-02**: Per-SNR accuracy breakdown for each defense at representative SNR points
- [ ] **EVAL-03**: Confusion matrices before/after defense for CW and EAD attacks
- [ ] **EVAL-04**: Perturbation budget curves (accuracy vs epsilon) for each attack type
- [ ] **EVAL-05**: Defense comparison table matching paper Table format (all defenses × all attacks)

### Paper

- [ ] **PAPER-01**: IEEE TCCN/TWC LaTeX manuscript — Introduction section
- [ ] **PAPER-02**: IEEE TCCN/TWC LaTeX manuscript — Related Work section
- [ ] **PAPER-03**: IEEE TCCN/TWC LaTeX manuscript — System Model & Threat Model section
- [ ] **PAPER-04**: IEEE TCCN/TWC LaTeX manuscript — Proposed Defense Method section
- [ ] **PAPER-05**: IEEE TCCN/TWC LaTeX manuscript — Experimental Setup section
- [ ] **PAPER-06**: IEEE TCCN/TWC LaTeX manuscript — Results & Analysis section
- [ ] **PAPER-07**: IEEE TCCN/TWC LaTeX manuscript — Conclusion section
- [ ] **PAPER-08**: Publication-quality figures (accuracy curves, confusion matrices, spectral plots)
- [ ] **PAPER-09**: Frequency-domain visualization plots (clean→attacked→recovered spectra)
- [ ] **PAPER-10**: Reproducibility scripts to regenerate all experimental results

## v2 Requirements

### Extended Evaluation

- **EXTEVAL-01**: Adaptive attack evaluation (attacker knows defense exists)
- **EXTEVAL-02**: RML2018.01a dataset evaluation
- **EXTEVAL-03**: Ablation study of pipeline components (detector only, recovery only, etc.)

### Extended Paper

- **EXTPAPER-01**: Over-the-air validation discussion with hardware setup
- **EXTPAPER-02**: Computational complexity analysis (FLOPs, memory)

## Out of Scope

| Feature | Reason |
|---------|--------|
| RML2018.01a experiments | Focus on RML2016.10a for this submission; doubles experiment time |
| Novel attack development | Paper is defense-focused |
| Over-the-air validation | Requires hardware; mention as future work |
| GUI or web interface | Research code only |
| Adversarial training as primary defense | Already explored via finetuning; not the paper's contribution |
| Multi-model evaluation | Focus on AWN; other models are future work |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| PIPE-01 | Phase 1 | Complete |
| PIPE-02 | Phase 1 | Complete |
| PIPE-03 | Phase 1 | Complete |
| BASE-01 | Phase 1 | Complete |
| BASE-02 | Phase 1 | Complete |
| BASE-03 | Phase 1 | Complete |
| BASE-04 | Phase 1 | Complete |
| BASE-05 | Phase 1 | Complete |
| BASE-06 | Phase 1 | Complete |
| BASE-07 | Phase 1 | Complete |
| EVAL-01 | Phase 2 | Pending |
| EVAL-02 | Phase 2 | Pending |
| EVAL-03 | Phase 2 | Pending |
| EVAL-04 | Phase 2 | Pending |
| EVAL-05 | Phase 2 | Pending |
| PAPER-01 | Phase 3 | Pending |
| PAPER-02 | Phase 3 | Pending |
| PAPER-03 | Phase 3 | Pending |
| PAPER-04 | Phase 3 | Pending |
| PAPER-05 | Phase 3 | Pending |
| PAPER-06 | Phase 3 | Pending |
| PAPER-07 | Phase 3 | Pending |
| PAPER-08 | Phase 3 | Pending |
| PAPER-09 | Phase 3 | Pending |
| PAPER-10 | Phase 3 | Pending |

**Coverage:**
- v1 requirements: 25 total
- Mapped to phases: 25
- Unmapped: 0

---
*Requirements defined: 2026-03-31*
*Last updated: 2026-03-31 after roadmap creation*
