# Phase 2: Experimental Results - Context

**Gathered:** 2026-04-02
**Status:** Ready for planning

<domain>
## Phase Boundary

Run the full defense-vs-attack evaluation matrix and produce all CSV data files needed for paper tables and figures. Output validated CSVs covering: multi-defense comparison tables (one per attack), perturbation budget curves (accuracy vs epsilon), and confusion matrices (before/after defense). No paper writing or figure rendering in this phase — just the numerical data.

</domain>

<decisions>
## Implementation Decisions

### Evaluation Matrix Scope
- **D-01:** 5 attacks only: CW (L2), EAD-L1, EAD-EN, FGSM (Linf), PGD (Linf)
- **D-02:** SNR >= 0 dB only — 10 points: 0, 2, 4, 6, 8, 10, 12, 14, 16, 18
- **D-03:** 9 defense rows: no-defense baseline, unified pipeline (ae_fft_topk), spectral gated, Kalman, Wiener, Savitzky-Golay, Gaussian, FIR, randomized smoothing
- **D-04:** 200 samples per (SNR, modulation) cell — matches Phase 1 calibration cap

### Epsilon Configuration
- **D-05:** Main comparison tables use eps=0.03 (minmax) for Linf attacks (FGSM, PGD) and c=1.0 for optimization attacks (CW, EAD-L1, EAD-EN)
- **D-06:** Linf budget curve: 8 epsilon points [0.01, 0.03, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3] with minmax normalization
- **D-07:** L2 budget curve (CW): vary confidence c = [0.01, 0.1, 1.0, 10.0]
- **D-08:** L1/EN budget curves (EAD-L1, EAD-EN): vary confidence c = [0.01, 0.1, 1.0, 10.0]

### Paper Table Format
- **D-09:** One comparison table per attack (5 tables total). Rows = 9 defenses, columns = 10 SNR points + weighted average column
- **D-10:** Single run, report accuracy percentages. Bold best-performing defense per column
- **D-11:** Weighted average column (weighted by samples per SNR — roughly equal for RML2016.10a)

### Confusion Matrices
- **D-12:** 3 optimization attacks: CW, EAD-L1, EAD-EN
- **D-13:** 3 SNR points: 0, 10, 18 dB
- **D-14:** Before and after unified pipeline defense = 18 confusion matrices total (3 attacks x 3 SNRs x 2)
- **D-15:** Full 11x11 modulation confusion matrices (all classes)
- **D-16:** Heatmap format with row-normalized percentages (standard AMC paper format)

### Claude's Discretion
- `--mode defense_compare` implementation details (function structure, batching, checkpoint/resume)
- Budget curve script organization (inline in defense_compare or separate mode)
- CSV column naming conventions
- Confusion matrix save format (PNG, PDF, or raw numpy for Phase 3 to render)
- Whether to generate intermediate progress logs during long runs

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Phase 1 Infrastructure (inputs to this phase)
- `util/defense_registry.py` — DEFENSE_REGISTRY dict, defend() pipeline function, randomized_smoothing_predict()
- `util/defense_baselines.py` — Classical filter implementations (Kalman, Wiener, SG, Gaussian, FIR)
- `util/defense_calibrate.py` — Parameter calibration sweep and best-parameter storage
- `util/defense.py` — FFT Top-K, spectral gated defense, normalize/denormalize helpers

### Attack Infrastructure
- `util/adv_attack.py` — Model01Wrapper, iq_to_ta_input_minmax/ta_output_to_iq_minmax, attack generation
- `util/adv_eval.py` — Run_Adv_Eval, existing adversarial evaluation pipeline

### Existing Evaluation Patterns
- `util/multi_attack_eval.py` — Multi-attack evaluation framework (reference for new defense_compare mode)
- `util/sigguard_eval.py` — SigGuard-style comparison tables (reference for table output format)
- `util/evaluation.py` — Run_Eval, confusion matrix generation, per-SNR accuracy

### Model and Data
- `models/model.py` — AWN model (returns logit, regu_sum)
- `data_loader/data_loader.py` — Load_Dataset with SNR/modulation filtering
- `checkpoint/2016.10a_AWN.pkl` — Pretrained AWN checkpoint
- `checkpoint/detector_ae.pth` — Autoencoder detector checkpoint (for unified pipeline)

### Configuration
- `util/config.py` — Config class, merge_args2cfg
- `config/2016.10a.yml` — Dataset config
- `main.py` — Mode dispatch, argparse definitions (add new modes here)

### Phase 1 Context
- `.planning/phases/01-defense-implementations/01-CONTEXT.md` — Phase 1 decisions (D-01 through D-14)

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `util/multi_attack_eval.py:run_multi_attack_snr_mod_eval()` — Already iterates attacks x SNR x modulation; can be extended for defense dimension
- `util/sigguard_eval.py:run_sigguard_eval()` — Produces formatted comparison tables; reference for table generation
- `util/evaluation.py:Run_Eval()` — Confusion matrix + per-SNR accuracy computation
- `DEFENSE_REGISTRY` — All 8 defenses dispatchable by string name
- `defend()` — Unified pipeline with latency breakdown

### Established Patterns
- Attack generation via torchattacks with Model01Wrapper + minmax normalization
- Defense applied to signal tensor before classification: `defended = defense_fn(adversarial)`
- Results output as CSV to `inference/<dataset>_*/result/`
- Confusion matrices via sklearn.metrics.confusion_matrix + seaborn/matplotlib heatmap

### Integration Points
- New `--mode defense_compare` in `main.py` (primary entry point)
- New evaluation utility (e.g., `util/defense_compare.py`) following multi_attack_eval pattern
- Results to `inference/<dataset>_*/result/defense_compare/` (CSV + confusion matrix data)
- Budget curve data as separate CSV (eps/c as column, accuracy per defense as rows)

</code_context>

<specifics>
## Specific Ideas

- SNR >= 0 only — user explicitly excluded negative SNR points as not meaningful for adversarial evaluation
- Per-SNR calibrated filter parameters from Phase 1 must be loaded and applied per-SNR during evaluation
- Budget curves use consistent c-parameter sweep for all optimization attacks (CW, EAD-L1, EAD-EN) — symmetric presentation
- 18 confusion matrices is thorough — may end up in supplementary material but data should exist

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 02-experimental-results*
*Context gathered: 2026-04-02*
