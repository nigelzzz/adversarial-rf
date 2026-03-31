# Pitfalls Research: Adversarial Defense for AMC

**Research Date:** 2026-03-31
**Domain:** Real-time adversarial defense for automatic modulation classification
**Target Venue:** IEEE TCCN/TWC

## Critical Pitfalls

### 1. Unfair Baseline Comparisons
- **Description:** Using suboptimal hyperparameters for baseline filters while tuning the proposed method heavily
- **Warning signs:** Baselines all perform badly with no parameter sweep; reviewers will call this out
- **Prevention:** Run parameter sweeps for each baseline filter (filter order, cutoff, window size). Report best results for each baseline.
- **Phase relevance:** Baseline implementation phase — must include parameter calibration

### 2. Epsilon Mismatch for IQ Signals
- **Description:** Using image-domain epsilon values (0.3) on IQ signals where amplitude is ~0.02. Makes attacks appear stronger than they are.
- **Warning signs:** All attacks show 0% accuracy (perturbation overwhelms signal)
- **Prevention:** Use RF-appropriate epsilons: unit mode eps=0.01-0.03, minmax mode eps=0.05-0.1. Already documented in CLAUDE.md.
- **Phase relevance:** All experiment phases — critical invariant

### 3. Ignoring Clean Accuracy Drop
- **Description:** Reporting only defended-under-attack accuracy without showing the cost to clean signal classification
- **Warning signs:** Paper only has "attack accuracy" columns, no "clean accuracy" row
- **Prevention:** Always include "Intact" (no attack) row in every comparison table. Flag any defense with >2% clean accuracy drop.
- **Phase relevance:** Evaluation phase — table design

### 4. Cherry-Picking SNR Ranges
- **Description:** Only showing results at high SNR where defense works well, hiding poor low-SNR performance
- **Warning signs:** Tables only show SNR ≥ 10 dB
- **Prevention:** Report full SNR range (-20 to +18 dB) or at minimum include representative low/medium/high SNR points. Acknowledge where defense degrades.
- **Phase relevance:** Evaluation phase — result reporting

### 5. Normalization Boundary Bugs
- **Description:** The AWN model expects raw IQ scale, but FFT Top-K and detector use normalized scale `(x+0.02)/0.04`. Mixing these up produces garbage results.
- **Warning signs:** Defense makes accuracy worse than no defense; NaN values in evaluation
- **Prevention:** Centralize normalization in a single wrapper function. Baseline filters must also respect this boundary — they operate on raw IQ or normalized IQ, never a mix.
- **Phase relevance:** Pipeline integration phase — architecture decision

### 6. Randomized Smoothing Misimplementation
- **Description:** Randomized smoothing for signals is architecturally different from filters. It wraps the classifier with multiple noisy copies, not the signal. Treating it as a filter baseline will give wrong results.
- **Warning signs:** Randomized smoothing shows same accuracy as Gaussian filter
- **Prevention:** Implement as classifier wrapper (majority vote over k noisy copies), not as signal preprocessing. Keep separate from filter baseline code path.
- **Phase relevance:** Baseline implementation phase

### 7. Real-Time Claims Without Evidence
- **Description:** Claiming "real-time" defense without latency measurements. IEEE reviewers in TCCN/TWC are domain experts who will question this.
- **Warning signs:** Paper says "real-time" but has no latency table or throughput numbers
- **Prevention:** Measure per-component latency (detector inference, FFT recovery, classifier inference). Report in milliseconds. Compare against signal duration (128 samples at typical sample rates). Note: Kalman/Wiener may require CPU fallback and will have higher latency.
- **Phase relevance:** Latency benchmarking phase

### 8. Weak Attack Configuration
- **Description:** Using too few CW/EAD optimization steps, making attacks easy to defend against
- **Warning signs:** CW attack accuracy is barely lower than clean accuracy
- **Prevention:** Use sufficient attack steps: CW ≥ 100 steps, EAD ≥ 100 steps, PGD ≥ 40 steps. Report attack success rate on undefended model to prove attacks are effective.
- **Phase relevance:** All experiment phases

### 9. Missing Statistical Significance
- **Description:** Reporting single-run results without variance. IEEE TCCN expects reliable numbers.
- **Warning signs:** No standard deviation or confidence intervals in tables
- **Prevention:** Run experiments with at least 3 random seeds or report accuracy over full test set with confidence intervals. The RML2016.10a test set is large enough that variance should be small, but report it.
- **Phase relevance:** Final evaluation phase

### 10. Paper Structure Misalignment
- **Description:** Writing the paper as "we tried many things" instead of a coherent narrative. Defense pipeline should be motivated top-down, not presented as an ad-hoc combination.
- **Warning signs:** Method section reads like a list of techniques with no unifying framework
- **Prevention:** Frame the pipeline as a principled three-stage defense: (1) detection to avoid unnecessary processing, (2) frequency-domain recovery to remove structured perturbations, (3) robust classification as final safety net. Each stage has a clear role.
- **Phase relevance:** Paper writing phase

## Pitfall Priority Matrix

| Pitfall | Severity | Likelihood | Phase |
|---------|----------|------------|-------|
| Epsilon mismatch | Critical | Medium | All experiments |
| Normalization bugs | Critical | High | Pipeline integration |
| Unfair baselines | High | High | Baseline implementation |
| Weak attacks | High | Medium | All experiments |
| Cherry-picking SNR | High | Medium | Evaluation |
| Clean accuracy drop | High | Medium | Evaluation |
| Real-time claims | Medium | High | Latency benchmarking |
| Randomized smoothing | Medium | Medium | Baseline implementation |
| Statistical significance | Medium | Low | Final evaluation |
| Paper structure | Medium | Medium | Paper writing |
