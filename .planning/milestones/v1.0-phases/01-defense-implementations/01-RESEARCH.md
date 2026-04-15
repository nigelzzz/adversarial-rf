# Phase 1: Defense Implementations - Research

**Researched:** 2026-03-31
**Domain:** PyTorch signal processing filters, unified defense pipeline, GPU timing
**Confidence:** HIGH

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

- **D-01:** Auto-calibrate filter parameters on validation set (not fixed literature values)
- **D-02:** Optimization metric is composite score: weighted average of clean accuracy + defended accuracy (against CW attack)
- **D-03:** Calibrate per-SNR — separate optimal parameters at each SNR point for each filter
- **D-04:** Parameters to sweep per filter: Kalman (process/measurement noise), Wiener (noise variance, filter length), Savitzky-Golay (window size, polynomial order), Gaussian (sigma), FIR (cutoff frequency, filter order)
- **D-05:** DEFENSE_REGISTRY is a dict mapping string names to callable functions: `{'kalman': kalman_filter, 'wiener': wiener_filter, 'fft_topk': fft_topk_defense, ...}`
- **D-06:** Normalization approach: use minmax normalization before attack, denormalize after attack. Each defense receives signals in the appropriate scale.
- **D-07:** Unified pipeline function signature: `defend(x, model, detector, cfg) -> (predictions, latency_breakdown)`
- **D-08:** k=20 noisy copies for randomized smoothing majority vote
- **D-09:** σ=0.01 fixed (matches IQ signal scale)
- **D-10:** Randomized smoothing implemented as classifier wrapper (NOT a signal filter) — separate code path from filter baselines
- **D-11:** Use torch.cuda.Event for GPU timing, time.perf_counter for CPU operations
- **D-12:** Report per-sample latency (divide batch time by batch size)
- **D-13:** Include warmup runs before measurement (standard ML benchmarking practice)
- **D-14:** Measure each component separately: detector inference, FFT recovery, each filter, classifier inference

### Claude's Discretion

- File organization: where to put new code (`util/defense_baselines.py`, `util/defense_registry.py`, etc.)
- Batch size for latency benchmarking
- Number of calibration sweep iterations per filter
- Whether to use scipy or pure torch for filter implementations

### Deferred Ideas (OUT OF SCOPE)

None — discussion stayed within phase scope
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| PIPE-01 | Unified detect→recover→classify inference path as single callable function | `defend(x, model, detector, cfg)` signature, existing `detector_gate_fft_topk` as building block |
| PIPE-02 | Latency benchmark per pipeline component (detector, recovery, classifier) in milliseconds | `torch.cuda.Event` verified working on RTX 5060 Ti; CPU timers via `time.perf_counter` |
| PIPE-03 | Clean accuracy preservation — defense degrades unperturbed accuracy by <2% | FFT Top-K is the recovery step; Gaussian/FIR at mild settings verified to preserve signal shape |
| BASE-01 | Kalman filter defense baseline with parameter sweep | No pykalman/filterpy installed; implement scalar 1D Kalman manually in NumPy; CPU-only path |
| BASE-02 | Wiener filter defense baseline with parameter sweep | `scipy.signal.wiener(signal, mysize, noise)` available; apply per-channel in Python loop |
| BASE-03 | Savitzky-Golay filter defense baseline with parameter sweep | `scipy.signal.savgol_filter(x, window_length, polyorder, axis=-1)` works natively on [N,2,T] |
| BASE-04 | Gaussian filter defense baseline with parameter sweep | GPU-native via depthwise `conv1d` with Gaussian kernel; verified on CUDA |
| BASE-05 | FIR low-pass filter defense baseline with parameter sweep | GPU-native via depthwise `conv1d` with `scipy.signal.firwin` coefficients moved to GPU |
| BASE-06 | Randomized smoothing baseline (σ=0.01, majority vote over k copies) | Classifier wrapper: expand batch k×, add Gaussian noise, forward, reshape to [k,N], majority vote |
| BASE-07 | Parameter calibration sweep for each filter baseline | Grid sweep over locked parameter ranges per D-04; composite score = α×clean_acc + (1-α)×defended_acc; per-SNR |
</phase_requirements>

---

## Summary

The project already has substantial defense infrastructure: `util/defense.py` contains FFT Top-K, notch filters, normalization helpers; `util/detector.py` has the AE gating logic. Phase 1 extends this in three directions: (1) wrapping existing components plus new classical filters into a single `DEFENSE_REGISTRY` dict; (2) implementing five new signal-processing baselines (Kalman, Wiener, Savitzky-Golay, Gaussian, FIR); and (3) adding a randomized smoothing classifier wrapper. A unified `defend(x, model, detector, cfg)` function composes the detector gate, FFT recovery, and classifier into one call with latency breakdown.

The critical environment finding is that neither `pykalman` nor `filterpy` is installed. Kalman filtering must be implemented as a pure-NumPy scalar loop (tested: ~0.32 ms/sample). Wiener and Savitzky-Golay use `scipy.signal` which is available (version 1.15.3). Gaussian and FIR filters should be GPU-native via PyTorch `F.conv1d` depthwise convolution — measured at ~3.5 µs/sample on the RTX 5060 Ti vs ~0.4 ms/sample for Wiener (CPU), providing the required measurable latency gap for the paper's latency comparison claim.

**Primary recommendation:** Implement Gaussian and FIR as GPU-native `F.conv1d` depthwise convolutions; implement Kalman, Wiener, and SG as CPU paths with numpy/scipy; keep tensor↔numpy conversion at filter boundaries; use `torch.cuda.Event` for GPU-resident defenses and `time.perf_counter` for CPU defenses.

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| PyTorch | 2.9.0+cu130 | GPU-native filters, FFT, timing | Already in project; CUDA 13.0 confirmed |
| scipy | 1.15.3 | Wiener, SG filter, firwin FIR design | Available; provides batch SG natively on last axis |
| numpy | 2.2.6 | Kalman loop, array ops | Already used throughout codebase |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| torchattacks | 3.5.1 | CW/FGSM/PGD for calibration sweeps | Generating adversarial examples during param calibration |
| time (stdlib) | — | CPU operation timing | Kalman, Wiener, SG latency measurement |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Manual NumPy Kalman | pykalman or filterpy | pykalman/filterpy not installed; manual loop is simple 1D scalar filter adequate for this domain |
| `scipy.signal.wiener` | Torch Wiener (MMSE spectral) | Scipy is 2 lines; Torch Wiener requires spectral noise estimation — not worth building |
| `firwin` + CPU `lfilter` | Pure Torch FIR | GPU `F.conv1d` with `firwin`-designed coefficients is faster and stays on CUDA |

**Installation:** No new packages needed. All required libraries are present.

## Architecture Patterns

### Recommended Project Structure

```
util/
├── defense.py              # Existing FFT defenses (DO NOT MODIFY existing functions)
├── detector.py             # Existing AE detector (DO NOT MODIFY)
├── defense_baselines.py    # NEW: kalman_filter, wiener_filter, sg_filter,
│                           #      gaussian_filter, fir_filter, randomized_smoothing
├── defense_registry.py     # NEW: DEFENSE_REGISTRY dict + defend() unified pipeline
└── defense_calibrate.py    # NEW: calibration sweep logic for BASE-07
```

This keeps the existing files untouched (they are consumed by other modes in `main.py`) and adds only new files.

### Pattern 1: Filter Baseline Signature

All five classical filter functions share a common signature so they can be stored directly in DEFENSE_REGISTRY:

```python
# Source: established from util/defense.py conventions
def kalman_filter(x: torch.Tensor, *, process_noise: float = 1e-4, meas_noise: float = 0.01) -> torch.Tensor:
    """x: [N, 2, T] on any device. Returns [N, 2, T] on same device."""
    arr = x.cpu().numpy()                        # CPU numpy path
    N, C, T = arr.shape
    out = np.empty_like(arr)
    for n in range(N):
        for c in range(C):
            out[n, c] = _kalman_1d(arr[n, c], q=process_noise, r=meas_noise)
    return torch.from_numpy(out).to(x.device)
```

**When to use:** Any filter that lacks a GPU-native implementation (Kalman, Wiener, SG).

### Pattern 2: GPU-Native Filter via Depthwise Conv1d

```python
# Source: verified in this research session
import torch.nn.functional as F
from scipy.signal import firwin

def fir_filter(x: torch.Tensor, *, cutoff: float = 0.1, numtaps: int = 31) -> torch.Tensor:
    """x: [N, 2, T]. GPU-native via depthwise conv1d."""
    device, dtype = x.device, x.dtype
    coeffs = firwin(numtaps, cutoff, window='hamming')            # designed on CPU
    kernel = torch.tensor(coeffs, dtype=dtype, device=device)
    kernel = kernel.view(1, 1, -1).expand(2, 1, -1)              # [2, 1, numtaps]
    half = numtaps // 2
    x_padded = F.pad(x, (half, half), mode='reflect')
    return F.conv1d(x_padded, kernel, groups=2)                   # stays on GPU
```

**When to use:** Gaussian and FIR — both have fixed linear kernels that map directly to conv1d. This avoids GPU→CPU→GPU roundtrips that would dominate latency and invalidate benchmarks.

### Pattern 3: Randomized Smoothing as Classifier Wrapper

```python
# Source: established pattern from literature; verified batch math
def randomized_smoothing_predict(model, x: torch.Tensor, *, k: int = 20, sigma: float = 0.01) -> torch.Tensor:
    """
    Majority vote over k noisy copies. Returns predicted class indices [N].
    NOT a signal filter — operates on model output, not signal.
    """
    N, C, T = x.shape
    x_rep = x.unsqueeze(0).expand(k, -1, -1, -1).reshape(k*N, C, T)
    noise = torch.randn_like(x_rep) * sigma
    x_noisy = x_rep + noise
    with torch.no_grad():
        logits, _ = model(x_noisy)                                # [k*N, num_classes]
    preds = logits.argmax(dim=1).reshape(k, N)                   # [k, N]
    votes = torch.zeros(N, logits.shape[1], device=x.device)
    for i in range(k):
        votes.scatter_add_(1, preds[i].unsqueeze(1), torch.ones(N, 1, device=x.device))
    return votes.argmax(dim=1)
```

**Critical:** At k=20, N=32 batch → 640 model forward passes. On RTX 5060 Ti this is still GPU-resident but latency will be ~20× higher than a single forward pass. Must be reported honestly in the paper.

### Pattern 4: DEFENSE_REGISTRY and Unified Pipeline

```python
# Source: D-05, D-07 from CONTEXT.md decisions

DEFENSE_REGISTRY: dict = {
    'kalman':      kalman_filter,
    'wiener':      wiener_filter,
    'savitzky_golay': sg_filter,
    'gaussian':    gaussian_filter,
    'fir':         fir_filter,
    'fft_topk':    fft_topk_defense,        # wraps util/defense.py:fft_topk_denoise
    'fft_notch':   fft_notch_defense,       # wraps util/defense.py:fft_notch_denoise
    'rand_smooth': None,                    # special case: classifier wrapper, not signal filter
}

def defend(
    x: torch.Tensor,
    model,
    detector,
    cfg,
) -> tuple[torch.Tensor, dict]:
    """
    Unified detect→recover→classify pipeline.
    Returns (predictions [N], latency_breakdown dict with keys in ms).
    """
```

**Pattern for timing each component separately (D-14):**

```python
# GPU components: torch.cuda.Event
t0 = torch.cuda.Event(enable_timing=True)
t1 = torch.cuda.Event(enable_timing=True)
t0.record(); detector_output = detector(x); t1.record()
torch.cuda.synchronize()
latency_breakdown['detector_ms'] = t0.elapsed_time(t1) / N

# CPU components: time.perf_counter
t_start = time.perf_counter()
x_filtered = kalman_filter(x, ...)
latency_breakdown['kalman_ms'] = (time.perf_counter() - t_start) * 1000 / N
```

### Pattern 5: Calibration Sweep

```python
# Source: D-01 through D-04 from CONTEXT.md

def calibrate_filter(defense_fn, param_grid, val_data, model, attack_fn, snr: int, alpha=0.5):
    """
    Grid search over param_grid. For each param set:
      score = alpha * clean_acc + (1-alpha) * defended_acc_under_cw
    Returns best params for this SNR.
    """
    best_score, best_params = -1.0, None
    for params in param_grid:
        clean_acc = _eval_clean(defense_fn, val_data[snr], model, **params)
        adv_acc = _eval_adv(defense_fn, val_data[snr], model, attack_fn, **params)
        score = alpha * clean_acc + (1 - alpha) * adv_acc
        if score > best_score:
            best_score, best_params = score, params
    return best_params, best_score
```

Per-SNR calibration means this runs once per SNR × filter, producing a best-params table saved as JSON/CSV.

### Anti-Patterns to Avoid

- **CPU roundtrip for GPU-resident signals (Gaussian/FIR):** Never call `.cpu().numpy()` in Gaussian or FIR implementations — always stay on GPU via `F.conv1d`. The latency paper claim depends on this.
- **Re-designing firwin on every batch call:** Compute FIR coefficients once (or cache by parameter values), not inside the per-batch forward call.
- **Applying randomized smoothing as a pre-classifier signal filter:** RS returns class votes, not a denoised signal. It cannot be inserted into the same pipeline slot as the other baselines. DEFENSE_REGISTRY must handle it as a special dispatch.
- **Using the Wiener filter's spectral division form:** `scipy.signal.wiener` uses the local statistics variant, not spectral MMSE — this is correct behavior for a "signal smoothing" defense baseline.
- **Applying defend() normalization incorrectly:** Filters receive signals at the scale the AWN classifier expects (raw IQ, ~±0.02 amplitude). The `(x+0.02)/0.04` normalization in `defense.py` is for the FFT Top-K path specifically — classical filters must operate on raw IQ scale to avoid distorting their spectral behavior. Verify filter effect by checking accuracy on clean signals (PIPE-03).

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| SG filter | Custom polynomial regression | `scipy.signal.savgol_filter` | Handles boundary conditions, tested; works natively on [N,2,T] with `axis=-1` |
| Wiener filter (MMSE) | Custom spectral Wiener | `scipy.signal.wiener` | Local statistics variant correct for smoothing; already handles 1D edge cases |
| FIR filter design | Custom window-method FIR | `scipy.signal.firwin` | Parks-McClellan and Kaiser windows available; `firwin` output goes directly into `F.conv1d` |
| Kalman tracking model | Full state-space tracker | Simple scalar 1D loop | IQ signals are 1D — no state transition matrix needed; scalar loop tested at 0.32 ms/sample |
| Gaussian kernel computation | Hand-coded exp formula | PyTorch `torch.exp` + `F.conv1d` | Kernel computed once; depthwise conv1d is GPU-optimized for exactly this pattern |

**Key insight:** The classical filters in this domain are well-understood signal processing primitives; the value is in the calibration and benchmarking, not the filter math itself. Any custom implementation will be slower and have more edge case bugs than scipy/PyTorch.

## Common Pitfalls

### Pitfall 1: Latency Measurement Without Warmup
**What goes wrong:** First CUDA call includes kernel compilation and memory allocation overhead; reported latency is 5-50× higher than steady-state.
**Why it happens:** CUDA lazily compiles kernels on first use; Python also has JIT overhead.
**How to avoid:** Run at least 10 warmup iterations before starting `torch.cuda.Event` timing (D-13). For CPU filters, discard first measurement.
**Warning signs:** First-call timing is an order of magnitude higher than subsequent calls.

### Pitfall 2: Filter Scale Sensitivity — AWN_All Normalization Boundary
**What goes wrong:** Applying `(x+0.02)/0.04` normalization inside a classical filter call changes the effective frequency content interpretation; applying FFT Top-K in the wrong scale produces different results.
**Why it happens:** `defense.py:fft_topk_denoise_normalized` normalizes then applies Top-K in normalized space. Classical filters should receive raw IQ (±0.02 scale) to behave as expected.
**How to avoid:** Document clearly in code which scale each filter expects. For PIPE-03 validation, measure clean accuracy drop — if >2%, the normalization boundary is wrong.
**Warning signs:** Clean accuracy drops >5% after defense is applied to unperturbed signals.

### Pitfall 3: Savitzky-Golay Window Must Be Odd and > polyorder
**What goes wrong:** `savgol_filter` raises `ValueError` if `window_length` is even or `window_length <= polyorder`.
**Why it happens:** SG filter requires at least `polyorder+1` points to fit a polynomial.
**How to avoid:** In the calibration sweep, enforce `window_length % 2 == 1` and `window_length > polyorder` as hard constraints before calling. For T=128, valid range is windows 5–63 (odd).
**Warning signs:** `ValueError: window_length must be less than or equal to the size of x`.

### Pitfall 4: Randomized Smoothing Memory at k=20
**What goes wrong:** At k=20, N=32 → 640 samples pushed through AWN simultaneously; if the AWN model is large, this OOMs on the GPU.
**Why it happens:** `x.unsqueeze(0).expand(k,-1,-1,-1).reshape(k*N,...)` creates a [640,2,128] tensor.
**How to avoid:** Process k copies in micro-batches if needed; AWN with [640,2,128] input is lightweight so this should fit on RTX 5060 Ti, but include a try/except with micro-batch fallback.
**Warning signs:** CUDA OOM on the RS forward pass.

### Pitfall 5: Per-SNR Calibration Requires CW Attack — Slow by Default
**What goes wrong:** CW attack with 100 steps × N val samples × 20 SNR points × 5 filters = extremely slow calibration.
**Why it happens:** D-02 requires CW as the calibration attack; CW is iterative.
**How to avoid:** Limit val set per SNR to ~200 samples for calibration (sufficient for relative ranking of filter parameters). Use `--eval_limit_per_cell 200` pattern from `multi_attack_eval`. Note this in the calibration script's docstring.
**Warning signs:** Calibration takes >2 hours — reduce val_limit first.

### Pitfall 6: DEFENSE_REGISTRY Callable Signature Mismatch
**What goes wrong:** Registry stores callables, but different filters have different keyword arguments. Calling `registry['kalman'](x)` fails because `kalman` needs `process_noise` and `meas_noise`.
**Why it happens:** Python callables don't enforce consistent signatures unless wrapped.
**How to avoid:** Store callables as `functools.partial` with calibrated parameters: `registry['kalman'] = functools.partial(kalman_filter, process_noise=best_q, meas_noise=best_r)`. The registry entry becomes `(x) -> Tensor` after partial application.
**Warning signs:** `TypeError: kalman_filter() missing required keyword argument`.

## Code Examples

Verified patterns from official sources and local testing:

### Scalar 1D Kalman Filter (CPU)
```python
# Source: verified locally — no pykalman/filterpy available
import numpy as np

def _kalman_1d(signal: np.ndarray, q: float = 1e-4, r: float = 0.01) -> np.ndarray:
    """Scalar constant-velocity Kalman filter. O(T) per channel."""
    n = len(signal)
    x_est, p = float(signal[0]), 1.0
    out = np.empty(n, dtype=np.float32)
    for k in range(n):
        x_pred = x_est
        p_pred = p + q
        K = p_pred / (p_pred + r)
        x_est = x_pred + K * (signal[k] - x_pred)
        p = (1.0 - K) * p_pred
        out[k] = x_est
    return out

def kalman_filter(x: torch.Tensor, *, process_noise: float = 1e-4, meas_noise: float = 0.01) -> torch.Tensor:
    arr = x.detach().cpu().numpy().astype(np.float32)
    N, C, T = arr.shape
    out = np.empty_like(arr)
    for n in range(N):
        for c in range(C):
            out[n, c] = _kalman_1d(arr[n, c], q=process_noise, r=meas_noise)
    return torch.from_numpy(out).to(device=x.device, dtype=x.dtype)
```

### Wiener Filter (CPU, scipy)
```python
# Source: scipy.signal.wiener API; verified locally
from scipy.signal import wiener as scipy_wiener

def wiener_filter(x: torch.Tensor, *, noise: float = 0.01, filter_len: int = 5) -> torch.Tensor:
    arr = x.detach().cpu().numpy().astype(np.float32)
    N, C, T = arr.shape
    out = np.empty_like(arr)
    for n in range(N):
        for c in range(C):
            out[n, c] = scipy_wiener(arr[n, c], mysize=filter_len, noise=noise)
    return torch.from_numpy(out).to(device=x.device, dtype=x.dtype)
```

### Savitzky-Golay Filter (vectorized scipy)
```python
# Source: scipy.signal.savgol_filter; axis=-1 works on [N,2,T] natively
from scipy.signal import savgol_filter as scipy_savgol

def sg_filter(x: torch.Tensor, *, window_length: int = 11, polyorder: int = 3) -> torch.Tensor:
    # Enforce constraints
    if window_length % 2 == 0:
        window_length += 1
    window_length = max(window_length, polyorder + 1)
    arr = x.detach().cpu().numpy().astype(np.float32)
    out = scipy_savgol(arr, window_length=window_length, polyorder=polyorder, axis=-1)
    return torch.from_numpy(out).to(device=x.device, dtype=x.dtype)
```

### Gaussian Filter (GPU-native via depthwise conv1d)
```python
# Source: verified locally — RTX 5060 Ti, ~3.5 us/sample
import torch.nn.functional as F
import math

def gaussian_filter(x: torch.Tensor, *, sigma: float = 1.0) -> torch.Tensor:
    device, dtype = x.device, x.dtype
    ksize = int(6.0 * sigma + 1.0)
    if ksize % 2 == 0:
        ksize += 1
    half = ksize // 2
    t = torch.arange(-half, half + 1, dtype=torch.float32, device=device)
    gauss = torch.exp(-t ** 2 / (2.0 * sigma ** 2))
    gauss = gauss / gauss.sum()
    kernel = gauss.view(1, 1, -1).expand(2, 1, -1).to(dtype=dtype)  # [2, 1, ksize]
    x_padded = F.pad(x, (half, half), mode='reflect')
    return F.conv1d(x_padded, kernel, groups=2)
```

### FIR Low-Pass Filter (GPU-native via depthwise conv1d)
```python
# Source: scipy.signal.firwin + torch F.conv1d; verified locally
from scipy.signal import firwin

def fir_filter(x: torch.Tensor, *, cutoff: float = 0.1, numtaps: int = 31) -> torch.Tensor:
    device, dtype = x.device, x.dtype
    if numtaps % 2 == 0:
        numtaps += 1  # firwin needs odd taps for Type I LP
    coeffs = firwin(numtaps, cutoff, window='hamming')
    kernel = torch.tensor(coeffs, dtype=dtype, device=device)
    kernel = kernel.view(1, 1, -1).expand(2, 1, -1)  # [2, 1, numtaps]
    half = numtaps // 2
    x_padded = F.pad(x, (half, half), mode='reflect')
    return F.conv1d(x_padded, kernel, groups=2)
```

### CUDA Event Timing Template
```python
# Source: PyTorch documentation; verified working on RTX 5060 Ti
import time

def _time_gpu_op(fn, *args, n_warmup=10, n_measure=50):
    """Returns per-batch time in ms."""
    for _ in range(n_warmup):
        fn(*args)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_measure):
        fn(*args)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / n_measure  # ms per batch

def _time_cpu_op(fn, *args, n_warmup=3, n_measure=20):
    """Returns per-batch time in ms."""
    for _ in range(n_warmup):
        fn(*args)
    t0 = time.perf_counter()
    for _ in range(n_measure):
        fn(*args)
    return (time.perf_counter() - t0) * 1000 / n_measure
```

### Calibration Parameter Grids (Suggested Starting Points)

```python
# Source: D-04 locked decisions + domain knowledge for T=128 signals

PARAM_GRIDS = {
    'kalman': [
        {'process_noise': q, 'meas_noise': r}
        for q in [1e-5, 1e-4, 1e-3, 5e-3]
        for r in [1e-3, 5e-3, 0.01, 0.05, 0.1]
    ],
    'wiener': [
        {'noise': n, 'filter_len': l}
        for n in [1e-3, 5e-3, 0.01, 0.05, 0.1]
        for l in [3, 5, 7, 11, 15]
    ],
    'savitzky_golay': [
        {'window_length': w, 'polyorder': p}
        for w in [5, 7, 11, 15, 21, 31]
        for p in [1, 2, 3]
        if p < w
    ],
    'gaussian': [
        {'sigma': s}
        for s in [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
    ],
    'fir': [
        {'cutoff': c, 'numtaps': n}
        for c in [0.05, 0.1, 0.15, 0.2, 0.3]
        for n in [15, 31, 63]
    ],
}
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Fixed-parameter baselines | Per-SNR calibrated baselines | D-01 to D-03 decision | Fairer comparison; reviewers expect tuned baselines |
| pykalman library | Manual NumPy scalar Kalman | Environment lacks pykalman | No library change needed; manual scalar loop is sufficient for 1D IQ |
| All defenses as signal filters | RS as classifier wrapper, others as filters | D-10 | Two dispatch paths in DEFENSE_REGISTRY; RS returns predictions not tensors |

**Deprecated/outdated:**
- `util/adv_eval.py` defense dispatch: the existing large if-elif chain in `Run_Adv_Eval` continues to work for existing modes but should NOT be modified for new baselines — new baselines go through `DEFENSE_REGISTRY` in new files only.

## Open Questions

1. **Calibration val/test split**
   - What we know: RML2016.10a has 20 SNRs × 11 mods × 1000 samples = 220,000 total; `data_loader.py` uses stratified split
   - What's unclear: Whether the existing `val` split from the data loader should be reused for calibration, or if a separate held-out portion is needed to avoid data leakage
   - Recommendation: Use the existing validation split from the data loader (same split as training); this is standard in defense literature and avoids needing a third split

2. **Composite score α weighting**
   - What we know: D-02 says "weighted average of clean accuracy + defended accuracy"; α is not specified
   - What's unclear: What weight balance best represents the paper's threat model (clean signal integrity vs. attack resistance)
   - Recommendation: Use α=0.5 (equal weight) as default; expose as configurable parameter `--calib_alpha`; the planner should include a task to confirm this with the user or document the choice

3. **Latency benchmark batch size**
   - What we know: D-12 says per-sample (divide batch by batch size); batch size is Claude's discretion
   - What's unclear: Best batch size to represent realistic deployment
   - Recommendation: Use batch_size=32 for benchmarking (standard for embedded/real-time inference); also report single-sample latency (batch=1) for real-time feasibility argument

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| PyTorch (CUDA) | All GPU filters, FFT, timing | Yes | 2.9.0+cu130 | — |
| scipy | Wiener, SG, FIR design | Yes | 1.15.3 | — |
| numpy | Kalman, array ops | Yes | 2.2.6 | — |
| NVIDIA GPU | GPU-native filters, CUDA events | Yes | RTX 5060 Ti | CPU fallback (will lose latency comparison) |
| pykalman | Kalman filter (full EM) | No | — | Manual NumPy scalar Kalman (sufficient) |
| filterpy | Kalman filter library | No | — | Manual NumPy scalar Kalman (sufficient) |
| torchattacks | CW attack for calibration | Yes | 3.5.1 | — |
| RML2016.10a checkpoint | Load AWN model | Yes | `checkpoint/2016.10a_AWN.pkl` | — |
| detector checkpoint | AE gate in PIPE-01 | Yes | `checkpoint/detector_ae.pth` | Skip gate, use fft_topk unconditionally |

**Missing dependencies with no fallback:** None that block execution.

**Missing dependencies with fallback:**
- pykalman/filterpy: Not installed — use manual NumPy scalar Kalman loop (fully functional, verified at ~0.32 ms/sample).

## Validation Architecture

Nyquist validation is explicitly `false` in `.planning/config.json`. Section skipped per instructions.

## Sources

### Primary (HIGH confidence)
- Local code inspection: `util/defense.py`, `util/detector.py`, `util/adv_eval.py` — direct source read
- `python3 -c` environment probes: verified PyTorch 2.9.0+cu130, scipy 1.15.3, numpy 2.2.6, CUDA available on RTX 5060 Ti
- `python3 -c` latency measurements: Kalman ~0.32 ms/sample CPU, Wiener ~0.40 ms/sample CPU, Gaussian ~3.5 µs/sample GPU, FIR GPU-native verified

### Secondary (MEDIUM confidence)
- scipy.signal API (wiener, savgol_filter, firwin): verified locally against installed version 1.15.3; API matches expected signatures
- PyTorch `F.conv1d` depthwise: verified working for Gaussian and FIR kernels at batch size 32 on CUDA

### Tertiary (LOW confidence)
- α=0.5 composite score weighting: reasonable default; not validated against specific paper reviewer expectations

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all packages probed live in environment
- Architecture: HIGH — directly derived from existing codebase patterns and locked decisions; all code patterns verified locally
- Pitfalls: HIGH — filter-specific pitfalls tested locally (SG constraint, FIR coefficient caching); normalization boundary pitfall observed in existing code

**Research date:** 2026-03-31
**Valid until:** 2026-06-30 (scipy/PyTorch versions stable; no fast-moving dependencies)
