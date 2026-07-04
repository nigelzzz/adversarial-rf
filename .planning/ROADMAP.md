# Roadmap: Real-Time Defense Pipeline for Adversarial AMC

## Milestones

- ✅ **v1.0 Paper Submission Package** — Phases 1-3 (shipped 2026-04-15) — see [milestones/v1.0-ROADMAP.md](milestones/v1.0-ROADMAP.md)
- 🚧 **v1.1 Robustness Baselines** — Phases 4-6 (active)

## Phases

<details>
<summary>✅ v1.0 Paper Submission Package (Phases 1-3) — SHIPPED 2026-04-15</summary>

- [x] Phase 1: Defense Implementations (3/3 plans) — completed 2026-04-01
- [x] Phase 2: Experimental Results (5/5 plans) — completed 2026-04-06
- [x] Phase 3: Paper (3/3 plans) — completed 2026-04-06

Delivered: unified detect→recover→classify pipeline with 5 classical-filter
baselines, randomized smoothing, full 9×5×10 defense-vs-attack comparison
matrix, and complete IEEE TCCN/TWC 13-page manuscript with reproduce.sh
end-to-end pipeline. UAT 10/10. See [archive](milestones/v1.0-ROADMAP.md).

</details>

### 🚧 v1.1 Robustness Baselines (Phases 4-6)

- [ ] **Phase 4: Adversarial Training** — Finetune AWN on mixed FGSM/PGD/EAD attacks to produce AT checkpoint
- [ ] **Phase 5: AT Evaluation** — Benchmark AT and AT+Adaptive-K on held-out CW and full 5-attack matrix
- [ ] **Phase 6: Paper Update + Camera-Ready** — Integrate AT results into Table I, update narrative, close v1.0 tech debt

## Phase Details

### Phase 4: Adversarial Training
**Goal**: A trained AT checkpoint exists that robustifies AWN against gradient-based attacks without catastrophic forgetting of analog modulations
**Depends on**: Nothing (warm-starts from existing v1.0 checkpoint)
**Requirements**: AT-01, AT-02, AT-03, AT-04, AT-05
**Success Criteria** (what must be TRUE):
  1. `./checkpoint/2016.10a_AWN_at.pkl` exists and loads without error into AWN
  2. Training log shows both clean accuracy and robust accuracy per epoch, confirming analog classes (AM-DSB, AM-SSB, WBFM) retain non-trivial accuracy
  3. A JSON config file saved alongside the checkpoint records epochs, LR, attack list, eps, ta_box mode, and alpha used
  4. Per-batch random attack selection draws from FGSM, PGD, EAD-L1, EAD-EN; CW is absent from training
**Plans:** 2 plans
Plans:
- [x] 04-01-PLAN.md — Script scaffold with data loading, attack factory, and dual-batch training loop
- [x] 04-02-PLAN.md — Checkpoint management, CSV logging, JSON config, and sanity eval

### Phase 5: AT Evaluation
**Goal**: Quantitative evidence exists comparing AT and AT+Adaptive-K to the v1.0 defense table across all paper attack/SNR conditions
**Depends on**: Phase 4 (requires `2016.10a_AWN_at.pkl`)
**Requirements**: ATEVAL-01, ATEVAL-02, ATEVAL-03, ATEVAL-04, ATEVAL-05
**Success Criteria** (what must be TRUE):
  1. `defense_compare.csv` contains new rows `at` and `at_adaptive_k` with per-attack, per-SNR accuracy values for all 5 attacks (FGSM, PGD, EAD-L1, EAD-EN, CW)
  2. AT vs held-out CW accuracy is reportable at SNR points 0, 6, 12, 18 dB
  3. AT+Adaptive-K composition results show whether layering improves, matches, or degrades Adaptive-K alone
  4. Per-SNR accuracy curve plots exist for both `at` and `at_adaptive_k` defenses alongside the existing v1.0 curves
**Plans**: TBD

### Phase 05.1: Certified Randomized Smoothing Baseline (Cohen et al. 2019) — L2-certified baseline defense: certify()/predict() module, noise-augmented finetune, certify_rs eval mode, thesis baseline writeup (INSERTED)

**Goal:** A certified L2-robustness randomized-smoothing baseline exists for AWN — noise-augmented checkpoints, a Cohen 2019 certify()/predict() module, a certify_rs sweep producing certified-accuracy-vs-L2-radius CSVs + curve and a CW L2-bucketed empirical comparison, and a self-contained thesis section citing Cohen 2019 — feeding Phase 6 camera-ready Table I
**Requirements**: RS-01, RS-02, RS-03, RS-04
**Depends on:** Phase 5
**Success Criteria** (what must be TRUE):
  1. `util/randomized_smoothing.py` exposes `certify()`/`predict()` (scipy-only Clopper-Pearson) and a passing unit test pins the CP bound to 0.3744139230995136
  2. Four noise-augmented checkpoints `checkpoint/2016.10a_AWN_rs{0.002,0.005,0.01,0.02}.pkl` exist and load with `weights_only=True`
  3. `main.py --mode certify_rs` writes `certify_rs_certified_acc.csv` (snr,modulation,sigma,radius,certified_acc,n_samples) + `certify_rs_curve.png` + `certify_rs_cw_compare.csv`, gated by a leading timing smoke test
  4. The thesis has a Cohen 2019 bib entry, a Related Work paragraph, and a self-contained `\section{Certified Robustness via Randomized Smoothing}`; `paper/latex/` is untouched
**Plans:** 4 plans
Plans:
- [ ] 05.1-01-PLAN.md — SmoothedClassifier (certify/predict) module + Clopper-Pearson unit test [RS-01]
- [ ] 05.1-02-PLAN.md — `--rs_sigma` single-stage noise-augmented finetune + 4 sigma checkpoints [RS-02]
- [ ] 05.1-03-PLAN.md — `--mode certify_rs`: timing smoke test, certify sweep, CW L2-bucketed comparison, CSV+curve [RS-03]
- [ ] 05.1-04-PLAN.md — Thesis: Cohen bib entry + Related Work + self-contained Evaluation section [RS-04]

### Phase 6: Paper Update + Camera-Ready
**Goal**: The manuscript reflects v1.1 findings with AT baseline rows in Table I and all v1.0 camera-ready debt resolved
**Depends on**: Phase 5 (requires evaluation CSVs for Table I values); CRTD requirements are independent
**Requirements**: PAPRU-01, PAPRU-02, PAPRU-03, PAPRU-04, CRTD-01, CRTD-02, CRTD-03, CRTD-04
**Success Criteria** (what must be TRUE):
  1. Table I in `results.tex` has two new rows — "Adversarial Training" and "AT + Adaptive-K v2" — with numeric accuracy values drawn from Phase 5 CSVs
  2. Results narrative names the AT baseline finding explicitly (e.g., "AT improves over undefended at high SNR but remains below Adaptive-K on CW")
  3. Discussion section contains a paragraph covering training-free recovery vs robust training trade-offs and deployment implications
  4. `paper/latex/figures/freq_spectra_cw.pdf` is regenerated from real CW attack data (not placeholder synthetic spectra)
  5. `paper/figures/ieee_style.py` has `text.usetex=True` and the compiled PDF uses Computer Modern Roman fonts; citation audit confirms `\nocite{*}` is removed and all 41 entries are explicitly cited or removed
**Plans**: TBD
**UI hint**: no

## Progress

| Phase                          | Milestone | Plans Complete | Status      | Completed |
|--------------------------------|-----------|----------------|-------------|-----------|
| 1. Defense Implementations     | v1.0      | 3/3            | Complete    | 2026-04-01 |
| 2. Experimental Results        | v1.0      | 5/5            | Complete    | 2026-04-06 |
| 3. Paper                       | v1.0      | 3/3            | Complete    | 2026-04-06 |
| 4. Adversarial Training        | v1.1      | 0/2            | Planning    | -          |
| 5. AT Evaluation               | v1.1      | 0/TBD          | Not started | -          |
| 05.1 Certified Rand. Smoothing | v1.1      | 0/4            | Planned     | -          |
| 6. Paper Update + Camera-Ready | v1.1      | 0/TBD          | Not started | -          |
