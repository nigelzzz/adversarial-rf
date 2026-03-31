# Technology Stack

**Project:** Real-Time Defense Pipeline for Adversarial Attacks on AMC
**Researched:** 2026-03-31
**Milestone:** Unified defense pipeline + IEEE TCCN paper

---

## Context: What Already Exists

Before recommendations, what the codebase already has locked in:

| Component | Existing | Version Confirmed |
|-----------|----------|-------------------|
| PyTorch | Yes | 2.9.0+cu130 (installed) |
| NumPy | Yes | 2.2.6 (installed) |
| SciPy | Yes | 1.15.3 (installed) |
| matplotlib | Yes | 3.10.7 (installed) |
| torchattacks | Yes | in requirements.txt |
| scikit-learn | Yes | in requirements.txt |
| FFT Top-K defense | Yes | `util/defense.py` (torch.fft.rfft based) |
| Autoencoder detector | Yes | `util/detector.py` |
| IEEEtran .tex files | Yes | `paper/latex/crc_experiment_ieee.tex` |

**Do not re-research or replace these.** The milestone adds on top.

---

## Recommended Stack: New Components Only

### Signal Processing Filter Baselines

#### Primary: SciPy 1.15.3 (already installed)

Use the already-installed SciPy for all five classical filter baselines. No new dependency needed. The current installation (1.15.3) is recent enough — current release is 1.17.1 (Feb 2026) but 1.15.3 has all required functions.

| Filter | SciPy Function | Module | Notes |
|--------|---------------|--------|-------|
| Savitzky-Golay | `scipy.signal.savgol_filter` | `scipy.signal` | window_length=11, polyorder=3 for IQ; must be odd length |
| Wiener | `scipy.signal.wiener` | `scipy.signal` | mysize=5 start point; estimates noise automatically if noise=None |
| Gaussian | `scipy.ndimage.gaussian_filter1d` | `scipy.ndimage` | sigma=1.0; applies per-channel along time axis |
| FIR low-pass | `scipy.signal.firwin` + `scipy.signal.lfilter` | `scipy.signal` | Design with firwin, apply with lfilter; numtaps=31, cutoff=0.25 |
| Kalman smoother | `pykalman.KalmanFilter` | external (new) | See below |

**Why SciPy over custom implementations:** These are the standard scipy functions that reviewers will recognize and accept without justification. Using anything else requires explanation. SciPy's `wiener` is documented as analogous to MATLAB's `wiener2`, which strengthens reproducibility claims.

**Confidence: HIGH** — verified against scipy.org/doc/scipy v1.17.0 docs.

#### Kalman Filter Baseline: pykalman 0.11.2

pykalman is the right choice over filterpy because:
- filterpy's last release was October 2018 (1.4.5) — abandoned, single maintainer
- pykalman 0.11.2 was released January 31, 2026; actively maintained with 0.11.0/0.11.1/0.11.2 all in Q4-2025/Q1-2026
- pykalman's `KalmanFilter.smooth()` returns the Rauch-Tung-Striebel smoother (better than forward-only filter for offline baseline)

**Usage pattern for IQ signals:**
```python
from pykalman import KalmanFilter
# Treat each channel (I, Q) as independent 1D observation
# x shape: [N, 2, 128] → reshape to [128, 1] per sample per channel
kf = KalmanFilter(
    transition_matrices=[[1]],
    observation_matrices=[[1]],
    transition_covariance=1e-4 * np.eye(1),
    observation_covariance=noise_var * np.eye(1),
)
smoothed_means, _ = kf.smooth(obs)  # obs shape [T, 1]
```

**Limitation:** pykalman processes each sample sequentially (no batching). For 22K test samples × 2 channels, expect ~10-30s total. Acceptable for offline baseline evaluation but must not be claimed as "real-time."

**Install:** `pip install pykalman==0.11.2`

**Confidence: HIGH** — verified on PyPI.

---

### Randomized Smoothing Baseline

**Do NOT use ART (adversarial-robustness-toolbox).** ART 1.20.1 is a large framework dependency (14+ sub-packages, complex install, MIT licensed but heavy). For a single smoothing baseline, it introduces more risk than value.

**Use: Inline implementation based on Cohen et al. (2019).**

The locuslab/smoothing reference implementation is ~100 lines. The relevant `Smooth` class in `core.py` takes any PyTorch classifier. For IQ signals the only change from the image-domain original is the noise distribution applies to the [2, 128] tensor directly.

```python
# Minimal randomized smoothing wrapper (no external dependency)
class RandomizedSmoother:
    def __init__(self, base_classifier, num_classes: int, sigma: float, n_samples: int = 100):
        self.classifier = base_classifier
        self.sigma = sigma
        self.n_samples = n_samples
        self.num_classes = num_classes

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, 2, 128]
        votes = torch.zeros(x.shape[0], self.num_classes, device=x.device)
        for _ in range(self.n_samples):
            noise = torch.randn_like(x) * self.sigma
            logits, _ = self.classifier(x + noise)
            votes += torch.nn.functional.one_hot(
                logits.argmax(dim=1), self.num_classes
            ).float()
        return votes.argmax(dim=1)
```

**Why inline over locuslab repo:** The locuslab repo is image-domain specific (normalizes for ImageNet). Adapting it to IQ signals is non-trivial. Writing ~80 lines avoids misuse and is reviewable in the paper supplement.

**Parameter choice: sigma=0.01.** IQ signals have RMS ~0.006-0.02 (see CLAUDE.md memory). sigma=0.01 is approximately 0.5-1.5x signal RMS — meaningful perturbation without destroying signal. The paper should report results for sigma in {0.005, 0.01, 0.02} to show the robustness/accuracy tradeoff.

**Confidence: HIGH** for implementation approach; MEDIUM for sigma values (based on existing codebase signal amplitude knowledge, not formal derivation).

---

### Latency Benchmarking

**Use: `torch.utils.benchmark.Timer`** — part of PyTorch, already installed.

Do not use `time.perf_counter` or `timeit` alone for GPU timing. They measure CPU dispatch time, not GPU completion time. `torch.utils.benchmark.Timer` automatically calls `torch.cuda.synchronize()`, performs warmup iterations, and computes statistics.

```python
import torch.utils.benchmark as benchmark

stmt = """
pipeline.forward(x_batch)
"""
t = benchmark.Timer(
    stmt=stmt,
    globals={"pipeline": pipeline, "x_batch": x_batch},
    num_threads=1,
)
result = t.timeit(1000)
print(f"Median: {result.median * 1e3:.2f} ms")
print(f"IQR:    {result.iqr * 1e3:.2f} ms")
```

**What to measure and report:**
- End-to-end pipeline: detect + recover + classify (single forward pass, batch=1 for latency, batch=256 for throughput)
- Per-component breakdown: detector alone, FFT Top-K alone, AWN classifier alone
- Target claim: total pipeline < 1ms per sample at batch=1 on GPU (FFT and conv ops are fast; autoencoder forward is the bottleneck)

**Confidence: HIGH** — torch.utils.benchmark is the official recommendation per PyTorch docs.

---

### LaTeX Paper Toolchain

#### Document Class: IEEEtran 1.8b (already present)

IEEEtran 1.8b is the current CTAN version (last updated August 2015 — IEEE has not updated it since). The project already has `paper/latex/crc_experiment_ieee.tex` using IEEEtran. The TCCN submission uses the same IEEEtran class with `\documentclass[journal]{IEEEtran}` — no new template needed.

**TCCN-specific constraints (confirmed Jan 2026):**
- Max 13 pages in double-column format for regular papers
- Submission via IEEE Author Portal (ScholarOne Manuscripts)
- LaTeX + PDF submission accepted

**Confidence: HIGH** — verified against comsoc.org TCCN submission page.

#### Bibliography: BibTeX + IEEEtran.bst

Use `IEEEtran.bst` (included with IEEEtran distribution). It produces the correct citation style for IEEE transactions. Do not use BibLaTeX — TCCN uses traditional BibTeX and the submission system may not support BibLaTeX-generated PDFs.

#### Figure Generation: matplotlib with IEEE-matching parameters

matplotlib 3.10.7 is already installed. Use these settings for publication-quality figures:

```python
import matplotlib
matplotlib.rcParams.update({
    "text.usetex": True,           # Requires texlive-full or texlive-latex-extra
    "font.family": "serif",
    "font.serif": ["Times"],       # IEEE uses Times New Roman
    "font.size": 9,                # IEEE caption size
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.format": "pdf",       # Vector PDF for IEEE submission
    "figure.figsize": (3.45, 2.5), # Single column: 3.45in wide
})
# Double-column figure: figsize=(7.16, 2.8)
```

**Do not use tikzplotlib.** It adds a LaTeX dependency chain that breaks often and produces output that requires manual editing. Export as PDF directly from matplotlib — IEEE accepts embedded PDFs in figures.

**Confidence: HIGH** — matplotlib rcParams approach is standard practice per publications guide.

#### Revision Tracking: latexdiff + git

For reviewer response, use `latexdiff` (system tool, not Python) combined with `git-latexdiff`:

```bash
# Install: apt install latexdiff
# Compare current version against submission tag
git-latexdiff HEAD~1 HEAD -- paper/main.tex
```

No Python library needed. `latexdiff` is a Perl script distributed with most TeX distributions (included in texlive).

**Confidence: MEDIUM** — standard academic practice, no version concerns.

---

## Alternatives Considered

| Category | Recommended | Alternative | Why Not |
|----------|-------------|-------------|---------|
| Kalman filter | pykalman 0.11.2 | filterpy 1.4.5 | filterpy abandoned since 2018 |
| Randomized smoothing | Inline implementation | ART 1.20.1 | ART is 14+ packages for one feature; overkill |
| Randomized smoothing | Inline implementation | locuslab/smoothing repo | Image-domain assumptions; IQ adaptation unclear |
| Latency timing | torch.utils.benchmark | time.perf_counter | No GPU sync; measures CPU dispatch, not GPU completion |
| Figures | matplotlib PDF export | tikzplotlib | Brittle dependency chain; requires LaTeX editing |
| Bibliography | IEEEtran.bst + BibTeX | BibLaTeX | IEEE submission systems prefer classic BibTeX |
| FIR filter | scipy.signal.firwin | custom convolution | scipy version is auditable, reviewer-recognizable |
| Gaussian filter | scipy.ndimage.gaussian_filter1d | torch.nn.functional.conv1d with Gaussian kernel | scipy is cleaner for offline baselines |

---

## Installation

All new dependencies (only one truly new):

```bash
# Only new package needed:
pip install pykalman==0.11.2

# System LaTeX (if not already present):
sudo apt install texlive-full latexdiff git-latexdiff
# or minimal: texlive-latex-base texlive-fonts-recommended texlive-latex-extra
```

Everything else is already in `requirements.txt` or part of PyTorch.

---

## Upgrade Note: SciPy

Current installed SciPy is 1.15.3; current release is 1.17.1. All five filter functions (`wiener`, `savgol_filter`, `firwin`, `lfilter`, `gaussian_filter1d`) exist in 1.15.3 and have stable APIs going back to 1.6. **Do not upgrade SciPy** unless a specific bug is encountered — avoiding dependency drift during a 1-month submission window matters more than having the latest version.

---

## Sources

- SciPy 1.17.1 release: https://pypi.org/project/scipy/
- SciPy signal.wiener docs: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.wiener.html
- SciPy signal.savgol_filter docs: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html
- SciPy ndimage.gaussian_filter1d: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter1d.html
- pykalman 0.11.2: https://pypi.org/project/pykalman/
- filterpy (abandoned): https://pypi.org/project/filterpy/
- Cohen et al. randomized smoothing (2019): https://github.com/locuslab/smoothing
- ART 1.20.1: https://pypi.org/project/adversarial-robustness-toolbox/
- torch.utils.benchmark: https://docs.pytorch.org/docs/stable/benchmark_utils.html
- PyTorch benchmark tutorial: https://docs.pytorch.org/tutorials/recipes/recipes/benchmark.html
- IEEEtran on CTAN: https://ctan.org/pkg/ieeetran
- IEEE TCCN submission guidelines (updated Jan 2026): https://www.comsoc.org/publications/journals/ieee-tccn/ieee-transactions-cognitive-communications-and-networking-submit
- Overleaf IEEE official templates: https://www.overleaf.com/gallery/tagged/ieee-official
