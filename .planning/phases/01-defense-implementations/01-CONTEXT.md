# Phase 1: Defense Implementations - Context

**Gathered:** 2026-04-01
**Status:** Ready for planning

<domain>
## Phase Boundary

Implement all defense mechanisms (unified detect→recover→classify pipeline, 5 classical filter baselines, randomized smoothing) as dispatchable functions through a common DEFENSE_REGISTRY interface. Include latency benchmarking and parameter auto-calibration.

</domain>

<decisions>
## Implementation Decisions

### Filter Parameter Calibration
- **D-01:** Auto-calibrate filter parameters on validation set (not fixed literature values)
- **D-02:** Optimization metric is composite score: weighted average of clean accuracy + defended accuracy (against CW attack)
- **D-03:** Calibrate per-SNR — separate optimal parameters at each SNR point for each filter
- **D-04:** Parameters to sweep per filter: Kalman (process/measurement noise), Wiener (noise variance, filter length), Savitzky-Golay (window size, polynomial order), Gaussian (sigma), FIR (cutoff frequency, filter order)

### Pipeline Interface
- **D-05:** DEFENSE_REGISTRY is a dict mapping string names to callable functions: `{'kalman': kalman_filter, 'wiener': wiener_filter, 'fft_topk': fft_topk_defense, ...}`
- **D-06:** Normalization approach: use minmax normalization before attack, denormalize after attack. Each defense receives signals in the appropriate scale.
- **D-07:** Unified pipeline function signature: `defend(x, model, detector, cfg) -> (predictions, latency_breakdown)`

### Randomized Smoothing
- **D-08:** k=20 noisy copies for majority vote
- **D-09:** σ=0.01 fixed (matches IQ signal scale)
- **D-10:** Implemented as classifier wrapper (NOT a signal filter) — separate code path from filter baselines

### Latency Benchmarking
- **D-11:** Use torch.cuda.Event for GPU timing, time.perf_counter for CPU operations
- **D-12:** Report per-sample latency (divide batch time by batch size)
- **D-13:** Include warmup runs before measurement (standard ML benchmarking practice)
- **D-14:** Measure each component separately: detector inference, FFT recovery, each filter, classifier inference

### Claude's Discretion
- File organization: where to put new code (`util/defense_baselines.py`, `util/defense_registry.py`, etc.)
- Batch size for latency benchmarking
- Number of calibration sweep iterations per filter
- Whether to use scipy or pure torch for filter implementations

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Existing Defense Infrastructure
- `util/defense.py` — FFT Top-K recovery implementation (reference for normalization pattern)
- `util/detector.py` — Conv autoencoder detector architecture
- `util/detector_train.py` — Detector training loop
- `util/adv_eval.py` — Attack/defense evaluation pipeline (Model01Wrapper, IQ normalization helpers)
- `util/adv_attack.py` — Attack wrappers and spectral attacks

### Model and Data
- `models/model.py` — AWN model definition (returns logit, regu_sum)
- `data_loader/data_loader.py` — RML2016.10a dataset loader with SNR/modulation filtering
- `checkpoint/2016.10a_AWN.pkl` — Pretrained model checkpoint

### Evaluation Framework
- `util/evaluation.py` — Clean evaluation (Run_Eval)
- `util/multi_attack_eval.py` — Multi-attack evaluation framework
- `util/sigguard_eval.py` — SigGuard-style comparison tables

### Configuration
- `util/config.py` — Config class and merge_args2cfg
- `config/2016.10a.yml` — Dataset configuration

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `util/defense.py:fft_topk_defense()` — FFT Top-K implementation, reference for new defenses
- `util/defense.py:detector_gate_fft_topk()` — Gated defense pattern (detect → conditionally recover)
- `util/adv_eval.py:Model01Wrapper` — Wraps AWN for torchattacks compatibility
- `util/adv_eval.py:iq_to_ta_input_minmax()` / `ta_output_to_iq_minmax()` — IQ normalization helpers

### Established Patterns
- Defense functions take `(x_tensor, **kwargs)` and return recovered tensor
- Normalization boundary: `(x+0.02)/0.04` for detector/FFT, raw scale for AWN classifier
- Config-driven: parameters passed via `cfg` object, CLI args merged in `main.py`

### Integration Points
- New defenses integrate via `main.py` modes (add new `--defense` options)
- Evaluation framework expects defense as callable applied before classification
- Results output to `inference/<dataset>_*/result/` as CSV

</code_context>

<specifics>
## Specific Ideas

- User wants per-SNR calibration for fair baseline comparison — this is the key differentiator vs naive fixed-parameter baselines
- Randomized smoothing at k=20, σ=0.01 specifically — not a sweep
- CUDA event timing for latency — paper needs precise GPU measurements

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 01-defense-implementations*
*Context gathered: 2026-04-01*
