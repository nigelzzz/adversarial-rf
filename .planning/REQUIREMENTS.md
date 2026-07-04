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

### Certified Randomized Smoothing (Phase 05.1)

- [ ] **RS-01**: `util/randomized_smoothing.py` implements Cohen et al. 2019 `SmoothedClassifier` with `certify(x, n0, n, alpha, batch_size) -> (class, L2_radius)` (ABSTAIN = (-1, 0.0)) and `predict(...)`, using a scipy-only Clopper-Pearson lower bound (`beta.ppf(alpha, NA, N-NA+1)`) and `binomtest` — no `statsmodels`, no removed `binom_test` — with a unit test pinning the CP bound to the verified value 0.3744139230995136; unwraps AWN's `(logit, regu_sum)`
- [ ] **RS-02**: `synth_finetune.py --rs_sigma` runs a SINGLE-STAGE Gaussian-noise-augmented finetune (warm-started from `checkpoint/2016.10a_AWN.pkl`, `x += randn_like(x)*sigma` per train and val batch), saving `checkpoint/2016.10a_AWN_rs{sigma}.pkl` for each sigma in {0.002, 0.005, 0.01, 0.02}; does NOT compose with `--curriculum`
- [ ] **RS-03**: `main.py --mode certify_rs` runs a leading timing smoke test then a per-(SNR,mod) certified-accuracy-vs-L2-radius sweep (n0=100, n=10000, alpha=0.001; SNR points {-6,0,6,12,18}; ~50/cell) across the sigma grid, plus a matched CW comparison bucketing CW's achieved raw-IQ L2 norm against RS certified accuracy — emitting `certify_rs_certified_acc.csv`, `certify_rs_curve.png`, and `certify_rs_cw_compare.csv` consumable by Phase 6; L∞/L1 attacks stay empirical-only
- [ ] **RS-04**: Thesis writeup — Cohen 2019 `@inproceedings` entry in `thesis/ref.bib`, a self-contained randomized-smoothing paragraph in `thesis/Sections/2.Relatedwork.tex`, and a new self-contained `\section{Certified Robustness via Randomized Smoothing}` (setup/results/CW-comparison) appended to `thesis/Sections/5.Evaluation.tex`; additive-only, `paper/latex/` untouched

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
| Certified L∞/L1 radii | Cohen 2019 certifies L2 only; FGSM/PGD/EAD stay empirical-only comparisons |
| `paper/latex/` edits in Phase 05.1 | Camera-ready Table I integration is Phase 6's responsibility (PAPRU-01..04) |
| Noise-augmented finetune composed with 3-stage curriculum | Cohen protocol is single-stage; keeps checkpoint naming `2016.10a_AWN_rs{sigma}.pkl` clean |

## Traceability

| Requirement  | Phase     | Status     |
|--------------|-----------|------------|
| AT-01        | Phase 4   | Pending    |
| AT-02        | Phase 4   | Pending    |
| AT-03        | Phase 4   | Pending    |
| AT-04        | Phase 4   | Pending    |
| AT-05        | Phase 4   | Pending    |
| ATEVAL-01    | Phase 5   | Pending    |
| ATEVAL-02    | Phase 5   | Pending    |
| ATEVAL-03    | Phase 5   | Pending    |
| ATEVAL-04    | Phase 5   | Pending    |
| ATEVAL-05    | Phase 5   | Pending    |
| RS-01        | Phase 05.1 | Pending   |
| RS-02        | Phase 05.1 | Pending   |
| RS-03        | Phase 05.1 | Pending   |
| RS-04        | Phase 05.1 | Pending   |
| PAPRU-01     | Phase 6   | Pending    |
| PAPRU-02     | Phase 6   | Pending    |
| PAPRU-03     | Phase 6   | Pending    |
| PAPRU-04     | Phase 6   | Pending    |
| CRTD-01      | Phase 6   | Pending    |
| CRTD-02      | Phase 6   | Pending    |
| CRTD-03      | Phase 6   | Pending    |
| CRTD-04      | Phase 6   | Pending    |

**Coverage:**
- v1.1 requirements: 22 total
- Mapped to phases: 22
- Unmapped: 0

---
*Requirements defined: 2026-04-15*
*Phase 05.1 (RS-01..RS-04) inserted: 2026-07-04*
