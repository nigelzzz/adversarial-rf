# Architecture Patterns

**Domain:** Adversarial defense pipeline for automatic modulation classification (AMC)
**Researched:** 2026-03-31
**Confidence:** HIGH — based on direct codebase inspection of all relevant modules

---

## Recommended Architecture

The pipeline is a **linear signal transformation chain** with a branch for comparison outputs.
Each stage consumes `[N, 2, 128]` IQ tensors and produces `[N, 2, 128]` tensors or scalar scores.

```
Raw IQ signal [N, 2, 128]
        |
        v
  [Normalization]          (x + 0.02) / 0.04 — matches AWN_All.py convention
        |
        v
  [Detector]               RFSignalAutoEncoder → KL divergence score per sample [N]
        |          |
   KL > thr    KL <= thr
        |          |
  [Recovery]  [Pass-through]
        |          |
        +----------+
        |
        v
  [Denormalization]        x * 0.04 - 0.02
        |
        v
  [Classifier]             AWN model → logits [N, 11]
        |
        v
  [Metrics]                accuracy / F1 / kappa per SNR and modulation
```

The key invariant: **the classifier always receives IQ in original scale**. The
normalization wraps only the detector and recovery stages, not the AWN model itself.

---

## Component Boundaries

### 1. Signal Normalization / Denormalization

| Component | Responsibility | File | Communicates With |
|-----------|---------------|------|-------------------|
| `normalize_iq_data` | `(x+0.02)/0.04` affine shift | `util/defense.py` | Detector, Recovery |
| `denormalize_iq_data` | `x*0.04-0.02` inverse | `util/defense.py` | Classifier |
| `normalize_for_detector` | Same transform, aliased | `util/detector.py` | Detector only |

**Boundary rule:** The AWN classifier was trained on raw-scale IQ (amplitude ~±0.02).
Never pass normalized tensors to AWN. Always denormalize before inference.

---

### 2. Detection Stage

| Component | Responsibility | File | Communicates With |
|-----------|---------------|------|-------------------|
| `RFSignalAutoEncoder` | 1D conv AE, 4 enc + 4 dec layers, skip connection | `util/detector.py` | Recovery gate |
| `kl_divergence_timewise` | Softmax-normalized KL per sample → `[N]` | `util/detector.py` | Gate decision |
| `detector_gate_fft_topk` | Loads AE, computes KL, routes to recovery | `util/detector.py` | Recovery, returns `(x_proc, kl_vals)` |

**Architecture detail:** The detector is trained on clean normalized IQ with MSE loss
(`util/detector_train.py`). The threshold (default 0.004468) is calibrated at the 90th
percentile KL on clean validation signals. During inference, detection runs at eval mode
with `@torch.no_grad()`.

**Training interface:** `main.py --mode train_detector` calls `util/detector_train.py`.
Checkpoint saved to `./checkpoint/detector_ae.pth`.

---

### 3. Recovery Stage — FFT-Domain (Unified Pipeline)

| Component | Responsibility | File | Communicates With |
|-----------|---------------|------|-------------------|
| `fft_topk_denoise` | Full complex FFT, keep top-K by magnitude, IFFT | `util/defense.py` | Classifier |
| `fft_topk_denoise_normalized` | Normalize → top-K → denormalize wrapper | `util/defense.py` | Evaluation loops |
| `fft_adaptive_topk_denoise` | Per-sample K via energy knee | `util/defense.py` | Multi-attack eval |
| `spectral_gated_defense` | Unified: spectral flatness → route to top-K or quantization | `util/defense.py` | `test_unified_defense.py` |

**Key design:** `fft_topk_denoise` operates in the complex FFT domain (not rFFT), so the
top-K selection is symmetric across positive and negative frequencies. This matches the
AWN_All.py reference implementation exactly.

**`spectral_gated_defense`** is the single-inference unified defense: one FFT is shared
for spectral flatness computation and Top-K filtering. Signals with flatness > 0.4 (only
AM-SSB in RML2016.10a) are routed to per-sample quantization instead.

---

### 4. Recovery Stage — Classical Filter Baselines (To Build)

These do not exist yet. Each baseline must conform to the same interface:

```python
def <filter>_defense(x: torch.Tensor, **kwargs) -> torch.Tensor:
    # x: [N, 2, T]  IQ in original scale
    # returns: [N, 2, T]  filtered IQ in original scale
```

| Baseline | Implementation Path | Key Parameters | Notes |
|----------|-------------------|----------------|-------|
| Kalman filter | `util/defense_baselines.py` | process_noise, obs_noise | Per-channel 1D scalar Kalman; no state across samples |
| Wiener filter | `util/defense_baselines.py` | noise_power or auto-estimated | `scipy.signal.wiener` or manual FFT-based; batch via torch |
| Savitzky-Golay | `util/defense_baselines.py` | window_length, polyorder | `scipy.signal.savgol_filter`; must handle [N,2,T] shape |
| Gaussian filter | `util/defense_baselines.py` | sigma | Gaussian kernel conv1d; depthwise over I and Q |
| FIR lowpass | `util/defense_baselines.py` | num_taps, cutoff_freq | `scipy.signal.firwin` + `torch.nn.functional.conv1d`; depthwise |
| Randomized smoothing | `util/defense_baselines.py` | sigma, n_samples | Add Gaussian noise, majority-vote classification (no signal recovery) |

**Implementation recommendation:** Build all baselines in a single file `util/defense_baselines.py`,
with a dispatch function `apply_baseline(name, x, **kwargs) -> torch.Tensor`. This
mirrors the pattern in `util/adv_eval.py` where defense dispatch is a single if/elif block.

**Randomized smoothing diverges from the others:** it does not produce a recovered signal;
instead it runs `n_samples` noisy copies through the classifier and takes majority vote.
It should be handled separately in the evaluation loop, not via the recovery slot.

---

### 5. Adaptive Defense Variants

| Component | Strategy | File | Notes |
|-----------|----------|------|-------|
| `confidence_sweep_topk_denoise` | Try K = 10,20,30,50; stop when softmax conf ≥ threshold | `util/adaptive_defense.py` | Requires model; 4 forward passes |
| `classify_then_filter_topk_denoise` | Classify raw, look up mod→K table, apply | `util/adaptive_defense.py` | Requires model; 2 forward passes |
| `spectral_shape_topk_denoise` | Count significant bins above pct*peak; map to K | `util/adaptive_defense.py` | Pure signal processing |
| `concentration_distortion_topk_denoise` | Accept smallest K with C ≥ thresh AND D ≤ thresh | `util/adaptive_defense.py` | Pure signal processing |

These are **not baseline competitors** — they are variants of the unified pipeline used
to ablate the K-selection strategy. They belong in the ablation section of the paper, not
the main comparison table.

---

### 6. Evaluation Framework

| Component | Responsibility | File | Produces |
|-----------|---------------|------|---------|
| `Run_Eval` | Clean accuracy per SNR, confusion matrix, F1, kappa | `util/evaluation.py` | `Accuracy_list`, `Confmat_Set` |
| `Run_Adv_Eval` | Single attack, optional defense, optional compare | `util/adv_eval.py` | Same metrics + optional defense comparison |
| `Run_Multi_Attack_Eval` | All attacks × all (snr, mod) cells with FFT recovery | `util/multi_attack_eval.py` | CSV: `multi_attack_snr_mod_eval.csv` |
| `Run_SigGuard_Eval` | All attacks × defense table (disabled / enabled) | `util/sigguard_eval.py` | `sigguard_eval.csv`, `sigguard_eval_table.txt` |
| `run_attack_bench` | Latency/throughput for attack and defense calls | `util/bench.py` | `adv_bench.json` |

**The experiment runner for the paper** needs a new module (to build), which calls the
evaluation framework for each defense variant and collects results into a single
comparison table. Call it `util/defense_comparison_eval.py` (see Data Flow below).

---

### 7. AWN Classifier

| Component | Responsibility | File | Notes |
|-----------|---------------|------|-------|
| `AWN` | Adaptive wavelet network; returns `(logits, regu_sum)` | `models/model.py` | Never receives normalized tensors |
| `Model01Wrapper` | Adapts AWN for torchattacks (4D [0,1] input) | `util/adv_attack.py` | Used only during attack generation |

**Critical:** `Model01Wrapper` is an attack wrapper, not a defense wrapper. The
classifier always receives raw-scale IQ `[N, 2, 128]` with dtype float32.

---

## Data Flow

### Inference Path (Paper's Unified Pipeline)

```
disk: RML2016.10a_dict.pkl
    → Load_Dataset()                  → Signals [N, 2, 128], Labels [N], SNRs [N]
    → Dataset_Split()                 → test split stratified by (mod, SNR)
    → attack generation               → adv [N, 2, 128]  (torchattacks via Model01Wrapper)
    → normalize_iq_data(adv)         → adv_norm [N, 2, 128]  scale to ~[-0.5, 0.5]
    → RFSignalAutoEncoder(adv_norm)  → recon [N, 2, 128]
    → kl_divergence_timewise()       → kl [N]
    → gate: kl > threshold?
        YES → fft_topk_denoise(adv_norm, K=50) → recovered_norm [N, 2, 128]
        NO  → adv_norm passes through
    → denormalize_iq_data()          → recovered [N, 2, 128]
    → AWN(recovered)                  → logits [N, 11]
    → argmax → predictions           → metrics
```

### Baseline Comparison Path (To Build)

```
adv [N, 2, 128]
    → apply_baseline(name, adv)      → recovered [N, 2, 128]  (no normalization needed)
    → AWN(recovered)                  → logits [N, 11]
    → argmax → predictions           → metrics
```

**Note:** Classical filters operate directly on raw IQ without normalization. Only the
AE-based pipeline requires the normalization layer. This is a clean separation point.

### Latency Measurement Path

```
x [N, 2, 128] on GPU
    → time defense function call     → elapsed ms
    → divide by N                    → ms per sample
    → compare against real-time threshold (128 samples at RF bandwidth)
```

The existing `util/bench.py` measures attack latency. A latency module for defenses
should follow the same pattern: GPU warmup → timed loop → JSON output.

---

## Patterns to Follow

### Pattern 1: Defense as Pure Function

Every defense (FFT variants, baselines) is a standalone function, not a class. Stateless
for signal-processing defenses; the detector is the only stateful component.

```python
def my_defense(x: torch.Tensor, **kwargs) -> torch.Tensor:
    assert x.dim() == 3 and x.size(1) == 2
    N, C, T = x.shape
    # ... operate on x ...
    return y  # same shape as x
```

This pattern is established across `util/defense.py` and should be followed for baselines.

### Pattern 2: Defense Dispatch in Evaluation

Both `adv_eval.py` and `sigguard_eval.py` use an if/elif block keyed on a string name to
select defense. The new comparison evaluator should follow the same pattern, adding
baseline names to the dispatch table.

```python
DEFENSE_REGISTRY = {
    'fft_topk':     lambda x, cfg: fft_topk_denoise(x, cfg.def_topk),
    'ae_fft_topk':  lambda x, cfg: detector_gate_fft_topk(x, det, threshold=...),
    'kalman':       lambda x, cfg: kalman_defense(x, ...),
    'wiener':       lambda x, cfg: wiener_defense(x, ...),
    'savgol':       lambda x, cfg: savgol_defense(x, window_length=..., polyorder=2),
    'gaussian':     lambda x, cfg: gaussian_defense(x, sigma=...),
    'fir':          lambda x, cfg: fir_defense(x, cutoff=..., num_taps=...),
}
```

### Pattern 3: Per-SNR Accuracy Collection

All existing evaluators collect per-SNR accuracy into `Accuracy_list[snr_i]`. The paper
requires this breakdown for all defenses simultaneously. The comparison evaluator should
accumulate a dict `{defense_name: Accuracy_list}` and write a single CSV.

### Pattern 4: Checkpoint Caching on cfg

The detector model is cached via `setattr(cfg, '_detector_model', det)` to avoid
reloading across batches. Follow this pattern for any stateful component loaded once.

---

## Anti-Patterns to Avoid

### Anti-Pattern 1: Normalizing Before AWN

**What goes wrong:** Passing `(x+0.02)/0.04` directly into AWN instead of denormalizing first.
**Why bad:** AWN was trained on raw-scale IQ; normalized tensors cause nonsense predictions.
**Instead:** Always denormalize before `model(x)`. The existing `adv_eval.py` handles this
correctly: normalization is applied for recovery then denormalization before inference.

### Anti-Pattern 2: Scipy Filters as Numpy Operations in GPU Batch

**What goes wrong:** Calling `scipy.signal.savgol_filter(x.cpu().numpy(), ...)` inside a
per-sample loop, then moving back to GPU.
**Why bad:** GPU→CPU→GPU roundtrip dominates latency; breaks the real-time claim.
**Instead:** Implement Gaussian and FIR filters as `torch.nn.functional.conv1d` with
depthwise groups=2 (I and Q independently). Savitzky-Golay can be precomputed as a
fixed polynomial kernel and applied the same way. Kalman and Wiener are the only filters
that may require CPU fallback, and their latency must be reported honestly.

### Anti-Pattern 3: Treating Randomized Smoothing as a Signal Recovery Defense

**What goes wrong:** Piping the noisy average through the classifier once like other defenses.
**Why bad:** Randomized smoothing requires majority voting over `n_samples` noisy copies.
Running it once produces worse accuracy than no defense at all.
**Instead:** Implement randomized smoothing as a separate function that calls the classifier
`n_samples` times and returns the majority predicted class. It does not produce a recovered
signal and cannot be inserted into the standard defense slot.

### Anti-Pattern 4: A Single Experiment Script for Everything

**What goes wrong:** Adding baseline evaluation as more elif branches in `main.py`.
**Why bad:** `main.py` is already 300+ lines with 20+ modes. Baseline comparison requires
iterating over multiple defenses × multiple attacks, which needs nested loops.
**Instead:** Build `util/defense_comparison_eval.py` as a self-contained evaluator that
accepts a list of defense names and attack names, runs the full matrix, and outputs a
comparison CSV. Register it in `main.py` as a single new mode (`--mode defense_compare`).

---

## Scalability Considerations

| Concern | Current State | At Paper Scale | Mitigation |
|---------|--------------|---------------|------------|
| Attack generation | Per-batch, stateless | 17 attacks × N_test samples | Use `eval_limit_per_cell` to bound CW/EAD cost |
| Defense latency | Measured in `bench.py` for attacks only | Need per-defense GPU timing | Add `util/defense_latency_bench.py` |
| Filter parameter sweep | Not implemented | Grid over K, sigma, cutoff | Parameter-agnostic dispatch function; loop outside evaluator |
| Figures | Per-mode output to `inference/*/result/` | 8-10 paper figures | `util/paper_figures.py` as a separate rendering pass |

---

## Suggested Build Order

Component dependencies flow in this direction. Build in order; each stage can be tested
before the next is started.

```
Stage 1: Classical filter baselines (no dependencies)
    → util/defense_baselines.py
    → unit test: apply each filter to a random [16, 2, 128] tensor, check shape and dtype
    → verify Gaussian and FIR are GPU-native (no scipy in hot path)

Stage 2: Defense comparison evaluator (depends on Stage 1 + existing eval framework)
    → util/defense_comparison_eval.py
    → DEFENSE_REGISTRY maps string names to callables
    → accepts model, dataset, attack list, defense list, cfg
    → produces CSV: defense × attack × snr → accuracy
    → integrates with existing Run_Eval / Run_Adv_Eval conventions

Stage 3: Randomized smoothing (separate branch, depends on classifier only)
    → util/defense_baselines.py (same file, but classified separately)
    → wrap AWN with n_samples=16 noise draws, majority vote
    → verify correct vote aggregation on clean signals first

Stage 4: Latency benchmark for defenses (depends on Stages 1–3)
    → util/defense_latency_bench.py
    → GPU warmup → timed loop → per-sample ms → JSON output
    → compare all defenses on same N=512 batch

Stage 5: Main mode wiring (depends on Stages 1–4)
    → main.py: add --mode defense_compare
    → add argparse flags: --defense_list, --use_ft
    → output: inference/<dataset>_*/result/defense_comparison.csv

Stage 6: Paper figures (depends on Stages 2–5)
    → util/paper_figures.py (new)
    → reads defense_comparison.csv and produces matplotlib figures
    → accuracy vs SNR lines (one per defense + no-defense baseline)
    → bar chart: clean accuracy degradation under each defense
    → latency bar chart from JSON
```

**Why this order:**
- Stage 1 is pure signal processing with no model dependency; it can be developed and
  validated by a single engineer before any evaluation is wired up
- Stage 2 reuses the existing evaluation loops (Run_Eval, Run_Adv_Eval) rather than
  reimplementing them; this avoids diverging from validated metric computation
- Stage 3 is isolated because randomized smoothing does not produce a recovered signal
  and must be implemented as a classifier wrapper, not a filter
- Stage 4 requires all defenses to be implemented before comparing latency
- Stage 5 is thin wiring; doing it last avoids breaking existing modes during development
- Stage 6 is last because it depends on all result CSVs being present

---

## Component Summary

```
util/
  defense.py              EXISTING — FFT-domain defenses (topk, notch, adaptive)
  defense_baselines.py    TO BUILD — Kalman, Wiener, SG, Gaussian, FIR, RS
  adaptive_defense.py     EXISTING — adaptive K strategies (ablation use)
  detector.py             EXISTING — AE detector + gate function
  detector_train.py       EXISTING — AE training loop
  defense_comparison_eval.py  TO BUILD — multi-defense × multi-attack matrix
  defense_latency_bench.py    TO BUILD — per-defense latency timing
  paper_figures.py        TO BUILD — publication-quality figure rendering
  adv_eval.py             EXISTING — single-attack evaluation with defense option
  sigguard_eval.py        EXISTING — SigGuard table (disabled vs enabled)
  multi_attack_eval.py    EXISTING — multi-attack with Top-K comparison
  evaluation.py           EXISTING — clean accuracy baseline
  bench.py                EXISTING — attack latency benchmark
  adv_attack.py           EXISTING — torchattacks wrappers + IQ normalization
  config.py               EXISTING — YAML config + arg merge

models/
  model.py                EXISTING — AWN classifier

main.py                   EXISTING — mode dispatcher; add --mode defense_compare
```

---

## Sources

- Direct codebase inspection: `util/defense.py`, `util/detector.py`, `util/adv_eval.py`,
  `util/sigguard_eval.py`, `util/multi_attack_eval.py`, `util/adaptive_defense.py`,
  `util/bench.py`, `paper/AWN_All.py`, `main.py`, `models/model.py`
- Reference implementation: `paper/AWN_All.py` — the original detect+recover+classify
  pipeline that this codebase replicates and extends
- Confidence: HIGH for all architectural claims — derived from reading live code, not
  training data
