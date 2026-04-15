# Requirements: Real-Time Defense Pipeline for Adversarial AMC — v1.1 Robustness Baselines

**Defined:** 2026-04-15
**Milestone Goal:** Strengthen v1.0 paper's defense narrative by adding an
adversarial-training baseline and closing camera-ready tech debt.

## v1.1 Requirements

### Adversarial Training

- [ ] **AT-01**: Training script finetunes AWN using mixed FGSM/PGD/EAD-L1/EAD-EN adversarial training with per-batch random attack selection
- [ ] **AT-02**: Training saves checkpoint `./checkpoint/2016.10a_AWN_at.pkl` and per-epoch log with train/val loss and clean/robust accuracy
- [ ] **AT-03**: Mixed clean+adversarial loss with configurable alpha (default α=0.5) to prevent catastrophic forgetting of analog modulations (AM-DSB, AM-SSB, WBFM)
- [ ] **AT-04**: Warm-start from pretrained AWN checkpoint (`./checkpoint/2016.10a_AWN.pkl`), not from scratch
- [ ] **AT-05**: Training hyperparameters (epochs, LR, attack iters, eps, ta_box mode) persisted to a JSON config saved alongside the checkpoint

### Evaluation

- [ ] **ATEVAL-01**: AT model evaluated against held-out CW attack across paper SNR points (0, 6, 12, 18 dB)
- [ ] **ATEVAL-02**: AT model evaluated against all 5 attacks (FGSM, PGD, EAD-L1, EAD-EN, CW) at SNR=18 dB for full sanity matrix
- [ ] **ATEVAL-03**: Layered defense "AT + Adaptive-K v2" evaluated on the same 5-attack matrix to measure composition effect
- [ ] **ATEVAL-04**: Two new rows (`at`, `at_adaptive_k`) added to `defense_compare.csv` with per-attack, per-SNR accuracies
- [ ] **ATEVAL-05**: Per-SNR accuracy curves generated for AT and AT+Adaptive-K defenses

### Paper Update

- [ ] **PAPRU-01**: New "Adversarial Training" row added to Table I (defense comparison matrix) in `paper/latex/sections/results.tex`
- [ ] **PAPRU-02**: New "AT + Adaptive-K v2" row added to Table I showing layered composition
- [ ] **PAPRU-03**: Results narrative updated in `results.tex` to report AT baseline findings (expected: improves over undefended but below Adaptive-K on CW)
- [ ] **PAPRU-04**: Discussion paragraph covering trade-offs: training-free recovery vs robust training, compute cost, deployment implications

### Camera-Ready Tech Debt

- [ ] **CRTD-01**: `text.usetex=True` enabled in `paper/figures/ieee_style.py` so figures use Computer Modern Roman matching IEEE body text
- [ ] **CRTD-02**: `paper/latex/figures/freq_spectra_cw.pdf` regenerated with real CW attack data (via active venv) replacing placeholder synthetic spectra
- [ ] **CRTD-03**: Verify `\nocite{*}` removed from `paper/latex/main.tex` and all 41 `refs.bib` entries are explicitly cited (or unused entries removed)
- [ ] **CRTD-04**: Clean stale `status: gaps_found` in archived `VERIFICATION.md` files (02 and 03) — mark as `passed` with closure note referencing commits that closed the gaps

## Future Requirements

Deferred from v1.0 and v1.1 discussion — revisit in a later milestone:

- **EXTEVAL-01**: Adaptive attack evaluation (attacker knows defense exists) — BPDA, transfer attacks
- **EXTEVAL-02**: RML2018.01a dataset evaluation
- **EXTEVAL-03**: Ablation study of pipeline components (detector only, recovery only, etc.)
- **EXTPAPER-01**: Over-the-air validation with hardware setup
- **EXTPAPER-02**: Computational complexity analysis (FLOPs, memory)

## Out of Scope

| Feature | Reason |
|---------|--------|
| RML2018.01a experiments | Deferred to future milestone; v1.1 stays on RML2016.10a |
| Novel attack development | Paper remains defense-focused |
| Over-the-air validation | Requires hardware; listed as future work |
| GUI or web interface | Research code only |
| Multi-model evaluation | AWN-only; other classifiers are future work |
| Adversarial training as **primary** defense | Adaptive-K remains main contribution; AT is a baseline |
| Training AT from scratch (no warm-start) | Saves ~10× compute; warm-start is standard practice |
| Adaptive attacks on AT | Deferred — v1.1 uses standard attacks only; BPDA goes to EXTEVAL-01 |

## Traceability

| Requirement  | Phase   | Status     |
|--------------|---------|------------|
| AT-01        | Phase 4 | Pending    |
| AT-02        | Phase 4 | Pending    |
| AT-03        | Phase 4 | Pending    |
| AT-04        | Phase 4 | Pending    |
| AT-05        | Phase 4 | Pending    |
| ATEVAL-01    | Phase 5 | Pending    |
| ATEVAL-02    | Phase 5 | Pending    |
| ATEVAL-03    | Phase 5 | Pending    |
| ATEVAL-04    | Phase 5 | Pending    |
| ATEVAL-05    | Phase 5 | Pending    |
| PAPRU-01     | Phase 6 | Pending    |
| PAPRU-02     | Phase 6 | Pending    |
| PAPRU-03     | Phase 6 | Pending    |
| PAPRU-04     | Phase 6 | Pending    |
| CRTD-01      | Phase 6 | Pending    |
| CRTD-02      | Phase 6 | Pending    |
| CRTD-03      | Phase 6 | Pending    |
| CRTD-04      | Phase 6 | Pending    |

**Coverage:**
- v1.1 requirements: 18 total
- Mapped to phases: 18
- Unmapped: 0

---
*Requirements defined: 2026-04-15*
