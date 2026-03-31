# Project Research Summary

**Project:** Real-Time Defense Pipeline for Adversarial Attacks on AMC
**Domain:** Adversarial robustness for RF automatic modulation classification (IEEE TCCN submission)
**Researched:** 2026-03-31
**Confidence:** HIGH

## Executive Summary

This project extends an existing, working AWN classifier with a unified adversarial defense pipeline targeting IEEE TCCN publication. The core architecture is well-understood because the codebase already contains the main components: the AE-based detector (`util/detector.py`), FFT Top-K recovery (`util/defense.py`), and torchattacks integration. The primary remaining work is (1) implementing classical filter baselines for fair comparison, (2) building a defense comparison evaluator that runs all defenses against all attacks in one pass, (3) latency benchmarking, and (4) producing publication-quality figures and a complete LaTeX manuscript.

The recommended approach is a linear signal chain: normalize → detect → recover (or pass-through) → denormalize → classify. Classical baselines (Kalman, Wiener, Savitzky-Golay, Gaussian, FIR low-pass) and randomized smoothing must be implemented in a new `util/defense_baselines.py` and wired into a `DEFENSE_REGISTRY` dispatch pattern. Randomized smoothing is architecturally different from filters — it wraps the classifier with majority voting and must not be placed in the signal-recovery slot. All baselines should be implemented with GPU-native operations (depthwise conv1d) to avoid CPU roundtrip latency that would invalidate real-time claims.

The highest-risk pitfalls are normalization boundary bugs (AWN expects raw IQ scale, detector and FFT Top-K use a shifted scale) and unfair baseline comparisons (under-tuned baselines will attract reviewer rejection). Both risks are mitigated by architectural decisions already present in the codebase — the normalization wrapper pattern is established and the dispatch registry approach isolates each defense cleanly. The paper has a strong, citable narrative: detect first to avoid unnecessary processing, recover in the frequency domain to remove structured perturbations, classify as final safety net.

---

## Key Findings

### Key Stack Decisions

The stack is almost entirely locked in. Only one new dependency is needed.

**Core technologies:**
- **PyTorch 2.9.0 + torch.fft:** All FFT-domain operations — already installed and in use.
- **SciPy 1.15.3 (`scipy.signal`, `scipy.ndimage`):** Provides all five classical filter baselines (`savgol_filter`, `wiener`, `firwin`/`lfilter`, `gaussian_filter1d`). Do not upgrade during submission window.
- **pykalman 0.11.2 (NEW):** Only new package needed. Use over filterpy (abandoned 2018). Its `KalmanFilter.smooth()` uses the Rauch-Tung-Striebel smoother. Processes IQ channels sequentially — acceptable for offline baseline, must not be called in the GPU hot path.
- **torch.utils.benchmark.Timer:** GPU-correct latency measurement (handles `cuda.synchronize()`). Already in PyTorch. Do not use `time.perf_counter` for GPU timing.
- **IEEEtran 1.8b + BibTeX + matplotlib PDF export:** Full LaTeX toolchain already present. Use `\documentclass[journal]{IEEEtran}`, IEEEtran.bst, and matplotlib with Times/9pt/300dpi/PDF output. Do not use BibLaTeX or tikzplotlib.
- **Inline randomized smoothing (~80 lines):** Do not pull in ART (14+ packages). Implement Cohen et al. (2019) directly; adapt sigma to IQ amplitude range (~0.006–0.02 RMS), report sigma in {0.005, 0.01, 0.02}.

### Table Stakes Features

These are required for IEEE TCCN acceptance and are either fully or partially implemented.

**Must have (table stakes):**
- Multi-attack evaluation (FGSM, PGD, CW, EAD minimum; 17 available) — infrastructure exists.
- Per-SNR accuracy breakdown at key points (-10, 0, 6, 10, 18 dB) — infrastructure exists.
- Confusion matrix before/after defense for strongest attacks (CW, EAD) — infrastructure exists.
- Clean accuracy preservation row in every table ("Intact" baseline, target <2% drop) — must be enforced in table design.
- Classical filter baseline comparisons (Kalman, Wiener, SG, Gaussian, FIR) — must be implemented.
- Perturbation budget curves (accuracy vs epsilon per attack) — partially exists; needs unified output.

**Should have (differentiators):**
- Unified detect → recover → classify pipeline as single inference path — partially exists as `ae_fft_topk`, needs comparison evaluator.
- Real-time latency breakdown (per-component ms, GPU batch=1 and batch=256) — not yet measured for defenses.
- Detector-gated recovery showing better clean accuracy than always-on defenses — exists as `ae_fft_topk`.
- Frequency-domain visualization (clean → attacked → recovered spectra) — plotting infrastructure exists.

**Defer (v2+):**
- Adaptive attack evaluation (attacker knows defense) — scope creep risk.
- Multi-dataset evaluation (RML2018) — doubles experiment time without proportional value.
- Over-the-air hardware validation — requires hardware setup, months of additional work.

### Architecture Highlights

The pipeline is a linear signal transformation chain where every stage consumes and produces `[N, 2, 128]` IQ tensors. The critical invariant is that the AWN classifier always receives raw-scale IQ (amplitude ~±0.02); normalization wraps only the detector and FFT recovery stages. Classical filter baselines bypass the normalization layer entirely — they operate on raw IQ directly, which is a clean separation point.

**Major components:**
1. **Signal Normalization/Denormalization** (`util/defense.py`) — `(x+0.02)/0.04` affine shift; always invert before passing to AWN. Never pass normalized tensors to the model.
2. **AE Detector** (`util/detector.py`, `RFSignalAutoEncoder`) — 1D conv autoencoder, KL divergence gate, threshold=0.004468 (90th percentile clean validation). Cached on `cfg` to avoid reloading.
3. **FFT Top-K Recovery** (`util/defense.py`, `fft_topk_denoise`) — complex FFT, keep top-K magnitude bins, IFFT. Symmetric over positive/negative frequencies per AWN_All.py reference.
4. **Classical Filter Baselines** (`util/defense_baselines.py`, TO BUILD) — uniform interface `def filter(x, **kwargs) -> Tensor`. Gaussian and FIR must use `torch.nn.functional.conv1d` (depthwise) to stay GPU-native. Kalman and Wiener require CPU fallback; report their latency honestly.
5. **Randomized Smoothing** (`util/defense_baselines.py`, TO BUILD) — classifier wrapper (majority vote over n_samples noisy copies); not a signal filter; handled in a separate evaluation branch.
6. **Defense Comparison Evaluator** (`util/defense_comparison_eval.py`, TO BUILD) — accepts list of defense names and attack names, runs full matrix, outputs `defense × attack × snr → accuracy` CSV. Uses `DEFENSE_REGISTRY` dict for dispatch.
7. **Latency Benchmark** (`util/defense_latency_bench.py`, TO BUILD) — GPU warmup, timed loop with `torch.utils.benchmark.Timer`, JSON output per defense.
8. **Paper Figures** (`util/paper_figures.py`, TO BUILD) — reads result CSVs, renders matplotlib PDF figures with IEEE-matching rcParams.

### Critical Pitfalls

1. **Normalization boundary bugs** — AWN expects raw IQ; FFT Top-K and AE detector use normalized scale. Mixing scales produces nonsense predictions or NaN. Enforce: always denormalize before `model(x)`; test with a sanity check that raw-scale clean signals produce expected clean accuracy.
2. **Unfair baseline comparisons** — Baselines must have parameter sweeps (filter order, cutoff, window size), not ad-hoc defaults. Reviewers will reject if all baselines appear uniformly bad. Run grid over key parameters and report best.
3. **Epsilon mismatch for IQ signals** — Do not use image-domain epsilon (0.3). IQ RMS is ~0.006–0.02. Use unit mode eps=0.01–0.03 or minmax mode eps=0.05–0.1. Validate by confirming attack drops undefended accuracy substantially (e.g., 90% → 20–40%).
4. **Real-time claims without latency evidence** — TCCN reviewers will reject "real-time" claims unsupported by numbers. Measure every defense with `torch.utils.benchmark.Timer`. Note that Kalman/Wiener require CPU fallback and will have higher latency; report this honestly.
5. **Randomized smoothing misimplementation** — It is a classifier wrapper (majority vote over k noisy passes), not a signal filter. Inserting it into the standard defense slot produces worse-than-random accuracy.

---

## Implications for Roadmap

Based on combined research, the build order has clear dependencies. Classical baselines have no model dependency and can be built and validated in isolation first. The comparison evaluator then reuses existing eval loops rather than reimplementing metric logic. Randomized smoothing is isolated because it requires a different code path. Latency benchmarking requires all defenses to exist first. Paper figures depend on result CSVs from completed experiments.

### Phase 1: Classical Filter Baselines
**Rationale:** No model dependency; can be validated independently with shape/dtype checks; unblocks all downstream comparison work. Single engineer can complete before evaluation is wired.
**Delivers:** `util/defense_baselines.py` with Kalman, Wiener, Savitzky-Golay, Gaussian (GPU-native depthwise conv1d), FIR low-pass (GPU-native depthwise conv1d). Uniform `def <filter>(x: Tensor, **kwargs) -> Tensor` interface. `apply_baseline(name, x, **kwargs)` dispatch function.
**Addresses:** Baseline comparisons (P0 table-stakes feature).
**Avoids:** Anti-pattern of scipy per-sample CPU loop in GPU batch; implements GPU-native convolution for Gaussian and FIR.
**Install:** `pip install pykalman==0.11.2`.

### Phase 2: Defense Comparison Evaluator
**Rationale:** Reuses validated `Run_Eval` and `Run_Adv_Eval` loops to avoid diverging from established metric computation. Produces the core experimental result for the paper.
**Delivers:** `util/defense_comparison_eval.py` with `DEFENSE_REGISTRY` (FFT Top-K, AE-gated, all 5 classical filters, RS), multi-defense × multi-attack × SNR CSV output. `main.py --mode defense_compare` entry point.
**Addresses:** Multi-attack evaluation, per-SNR breakdown, confusion matrices, clean accuracy preservation (all P0 features).
**Avoids:** Adding more elif branches directly to `main.py`; cherry-picking SNR ranges (full -20 to +18 dB required); omitting "Intact" row from tables.
**Research flag:** Standard patterns; no additional research phase needed.

### Phase 3: Randomized Smoothing
**Rationale:** Isolated because it does not produce a recovered signal; requires separate branch in the evaluation loop; must not be wired into the standard defense slot.
**Delivers:** `RandomizedSmoother` class in `util/defense_baselines.py` (inline ~80 lines, Cohen et al. 2019 adapted for IQ). n_samples=16–100, sigma in {0.005, 0.01, 0.02}. Separate evaluation path in comparison evaluator.
**Addresses:** Baseline comparisons (reviewers expect this baseline for adversarial robustness papers).
**Avoids:** Treating randomized smoothing as a filter (critical misimplementation).

### Phase 4: Latency Benchmarking
**Rationale:** Requires all defenses to exist before comparing. Supports the "real-time feasibility" differentiator claim.
**Delivers:** `util/defense_latency_bench.py` using `torch.utils.benchmark.Timer`. Per-component breakdown (detector, FFT recovery, each baseline filter). JSON output. Target: total pipeline <1ms per sample on GPU.
**Addresses:** Real-time latency analysis (P1 differentiator). Honestly reports Kalman/Wiener CPU fallback cost.
**Avoids:** Claiming "real-time" without evidence; using `time.perf_counter` without GPU sync.

### Phase 5: Perturbation Budget Curves and Final Experiment Runs
**Rationale:** Epsilon sweep requires comparison evaluator from Phase 2 to be complete. Final experiment runs produce all data for paper tables.
**Delivers:** Accuracy vs epsilon curves per attack type. Full experiment results with sufficient attack steps (CW ≥ 100, EAD ≥ 100, PGD ≥ 40). Results over full test set for statistical reliability.
**Addresses:** Perturbation budget analysis (P1 feature); weak attack pitfall; statistical significance pitfall.
**Avoids:** Using too few optimization steps (CW accuracy barely dropping); single-run results without variance.

### Phase 6: Paper Figures and LaTeX Manuscript
**Rationale:** Depends on all result CSVs being present. Figures and manuscript are the final deliverable.
**Delivers:** `util/paper_figures.py` with matplotlib IEEE-matching rcParams (Times 9pt, 300dpi, PDF). Accuracy-vs-SNR lines, clean accuracy degradation bar chart, latency bar chart. Full IEEEtran journal manuscript in `paper/latex/`.
**Addresses:** Paper draft (P0 feature); frequency-domain visualization (P1 feature).
**Avoids:** tikzplotlib (brittle), BibLaTeX (IEEE submission risk), "list of techniques" paper structure. Frame narrative as three-stage principled defense.
**Research flag:** Paper structure is well-established for IEEE TCCN; no additional research phase needed.

### Phase Ordering Rationale

- Phase 1 before Phase 2 because baselines must be registered before the comparison evaluator can dispatch them.
- Phase 3 is isolated between comparison evaluator and latency because randomized smoothing needs to be in the registry before latency can benchmark it.
- Phase 4 requires all defenses to exist before meaningful latency comparison is possible.
- Phase 5 before Phase 6 because figures depend on experiment CSVs.
- Building `util/defense_comparison_eval.py` as a new module rather than extending `main.py` avoids breaking the 20+ existing modes and matches the anti-pattern warning from ARCHITECTURE.md.

### Research Flags

Phases with standard patterns (skip additional research):
- **Phase 1 (Classical filters):** SciPy and PyTorch APIs are stable and well-documented. Confidence HIGH.
- **Phase 2 (Comparison evaluator):** Pattern is fully established in `adv_eval.py` and `sigguard_eval.py`; this is mechanical extension.
- **Phase 4 (Latency):** `torch.utils.benchmark` API is documented and stable. Pattern exists in `util/bench.py`.
- **Phase 6 (Paper):** IEEEtran format is fixed; matplotlib rcParams are verified.

Phases that may need targeted investigation:
- **Phase 3 (Randomized smoothing):** Sigma values need empirical calibration against actual IQ signal amplitudes in the test set. STACK.md suggests sigma in {0.005, 0.01, 0.02} based on known RMS range, but the best value for this classifier/dataset combination needs a small sweep.
- **Phase 5 (Epsilon sweep):** The appropriate epsilon range for each attack type needs confirmation that attacks are actually effective before defense comparison is meaningful. Run undefended baseline first.

---

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | Almost entirely locked-in existing dependencies. Only pykalman is new; verified on PyPI with active maintenance. |
| Features | HIGH | Standard IEEE TCCN expectations for adversarial defense papers; existing infrastructure covers most P0 items. |
| Architecture | HIGH | Derived from direct codebase inspection of live code, not training data. Boundary rules are explicit in existing code. |
| Pitfalls | HIGH | Most pitfalls reflect concrete bugs that have already manifested (epsilon mismatch documented in CLAUDE.md, normalization boundary documented in code comments). |

**Overall confidence:** HIGH

### Gaps to Address

- **Randomized smoothing sigma calibration:** Best sigma value for IQ signals in this dataset is not known a priori. Run a small sweep (sigma 0.005, 0.01, 0.02) and report the tradeoff curve.
- **Baseline filter hyperparameter tuning:** Optimal window lengths, cutoff frequencies, and filter orders for RML2016.10a IQ signals are not known. Must run parameter sweeps during Phase 1 implementation and report best parameters alongside results — required for fair comparison.
- **CW/EAD attack effectiveness verification:** Before defense comparison, verify that undefended accuracy drops substantially under CW (≥100 steps) and EAD attacks. If not, attack configuration needs adjustment before results are meaningful.
- **Statistical reporting plan:** Decide before running final experiments whether to report variance over seeds or rely on test-set size for confidence. RML2016.10a test set is large (~22K samples); single-run variance should be small but should be computed.

---

## Sources

### Primary (HIGH confidence)
- SciPy 1.17.1 docs: https://docs.scipy.org/doc/scipy/ — filter API verification
- pykalman 0.11.2 on PyPI: https://pypi.org/project/pykalman/ — version and maintenance status
- torch.utils.benchmark docs: https://docs.pytorch.org/docs/stable/benchmark_utils.html — GPU timing
- IEEE TCCN submission guidelines (Jan 2026): https://www.comsoc.org/publications/journals/ieee-tccn/ieee-transactions-cognitive-communications-and-networking-submit — page limits, format
- Direct codebase inspection: `util/defense.py`, `util/detector.py`, `util/adv_eval.py`, `util/sigguard_eval.py`, `util/adaptive_defense.py`, `paper/AWN_All.py`, `main.py` — all architectural claims

### Secondary (MEDIUM confidence)
- Cohen et al. (2019) randomized smoothing: https://github.com/locuslab/smoothing — sigma adaptation to IQ domain is inference, not validated
- IEEEtran on CTAN: https://ctan.org/pkg/ieeetran — confirmed current version 1.8b

---
*Research completed: 2026-03-31*
*Ready for roadmap: yes*
