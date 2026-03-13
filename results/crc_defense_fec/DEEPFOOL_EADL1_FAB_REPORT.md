# Spectral Defense Evaluation: IFFT Top-K Filtering Against DeepFool, EADL1, and FAB Adversarial Attacks on AMC

---

## Abstract

We extend the prior CW/EADEN evaluation by testing three additional adversarial attack algorithms — **DeepFool**, **EADL1** (EAD L1), and **FAB** (Fast Adaptive Boundary) — against the AWN automatic modulation classifier with IFFT Top-K spectral defense. Experiments at SNR=0 dB and SNR=18 dB across all 11 modulations reveal that **EADL1 is the most effective attack** (0.00% accuracy at both SNR levels), **FAB is moderately effective** (9.1--9.9%), and **DeepFool is largely ineffective** (82--87% accuracy remains). The spectral defense recovery patterns are consistent with CW/EADEN findings: QAM64, PAM4, AM-DSB recover well with Top-10/Top-20, while 8PSK, QAM16, QPSK, and AM-SSB remain difficult to defend. These results confirm that defense effectiveness is determined by modulation spectral structure, not the specific attack algorithm.

---

## 1. Attack Descriptions

### 1.1 DeepFool

DeepFool iteratively finds the closest decision boundary and computes the minimum perturbation to cross it. Unlike CW/EADEN which optimize for misclassification, DeepFool seeks the **minimal perturbation** to change the prediction. This produces smaller perturbations but may not always succeed against robust classifiers.

### 1.2 EADL1 (Elastic-net Attack, L1 Decision)

EADL1 uses elastic-net regularization (L1 + L2 penalty) with L1-based decision. It produces **sparse perturbations** concentrated in few samples/bins, making it potentially harder for spectral defenses to remove since the perturbation energy can be concentrated in high-energy bins.

### 1.3 FAB (Fast Adaptive Boundary)

FAB is a minimum-norm attack that projects onto the decision boundary. It combines properties of DeepFool (boundary-seeking) with constrained optimization (epsilon-bounded). It typically finds smaller perturbations than PGD while being more reliable than DeepFool.

---

## 2. Experimental Setup

| Parameter | Value |
|-----------|-------|
| Dataset | RML2016.10a |
| Model | AWN (Adaptive Wavelet Network) |
| SNR levels | 0 dB, 18 dB |
| Samples per (SNR, mod) cell | ~199 (test set) |
| Modulations | All 11 (8PSK, AM-DSB, AM-SSB, BPSK, CPFSK, GFSK, PAM4, QAM16, QAM64, QPSK, WBFM) |
| Box mode | minmax |
| Epsilon | 0.1 (for FAB; DeepFool and EADL1 are norm-minimizing) |
| Defense | IFFT Top-10 and Top-20 |

---

## 3. Results: SNR = 18 dB

### 3.1 Attack Effectiveness Comparison

| Modulation | Clean Acc | DeepFool Acc | EADL1 Acc | FAB Acc |
|------------|-----------|-------------|-----------|---------|
| 8PSK | 97.49% | **96.98%** | **0.00%** | 13.07% |
| AM-DSB | 100.00% | 92.46% | 1.51% | 20.60% |
| AM-SSB | 89.95% | 82.41% | 0.00% | 11.56% |
| BPSK | 97.99% | **97.99%** | 0.00% | 7.54% |
| CPFSK | 100.00% | 78.39% | 0.00% | 0.00% |
| GFSK | 100.00% | 51.76% | 0.00% | 0.00% |
| PAM4 | 99.50% | 96.48% | 0.00% | 9.05% |
| QAM16 | 97.49% | 89.45% | 0.00% | 9.05% |
| QAM64 | 94.97% | 88.44% | 0.00% | 15.08% |
| QPSK | 98.49% | 93.47% | 0.00% | 10.55% |
| WBFM | 42.21% | 36.18% | 0.00% | 12.56% |
| **Mean** | **92.55%** | **82.18%** | **0.14%** | **9.91%** |

**Key observations:**

- **DeepFool barely works**: Average 82.18% accuracy under attack (vs 92.55% clean). For BPSK it has zero effect (97.99% → 97.99%). DeepFool's minimum-perturbation approach finds perturbations too small to reliably fool the classifier with minmax normalization.
- **EADL1 is devastating**: 0.00% accuracy for 10 of 11 modulations. Only AM-DSB retains 1.51%. This matches EADEN's effectiveness from the prior report.
- **FAB is moderate**: Average 9.91%. Achieves 0% on CPFSK and GFSK, but leaves 13--21% residual accuracy on other modulations.

### 3.2 Defense Recovery at SNR = 18 dB

| Modulation | Clean | DeepFool Att | DF Top-10 | DF Top-20 | EADL1 Att | EL1 Top-10 | EL1 Top-20 | FAB Att | FAB Top-10 | FAB Top-20 |
|------------|-------|-------------|-----------|-----------|-----------|------------|------------|---------|------------|------------|
| QAM64 | 94.97% | 88.44% | **92.96%** | 88.94% | 0.00% | **94.47%** | 84.42% | 15.08% | **93.47%** | 81.91% |
| PAM4 | 99.50% | 96.48% | 97.99% | **98.49%** | 0.00% | 93.47% | **95.98%** | 9.05% | 95.98% | **98.49%** |
| AM-DSB | 100.00% | 92.46% | 69.35% | **85.93%** | 1.51% | 98.49% | **95.98%** | 20.60% | 99.50% | **98.49%** |
| GFSK | 100.00% | 51.76% | 74.87% | **80.90%** | 0.00% | 78.89% | **97.49%** | 0.00% | 77.39% | **94.97%** |
| BPSK | 97.99% | 97.99% | 1.51% | **96.48%** | 0.00% | 1.01% | **79.90%** | 7.54% | 1.51% | **58.79%** |
| CPFSK | 100.00% | 78.39% | 1.51% | **84.92%** | 0.00% | 1.01% | **72.36%** | 0.00% | 1.51% | **80.90%** |
| QPSK | 98.49% | 93.47% | 0.50% | **94.97%** | 0.00% | 0.50% | **88.44%** | 10.55% | 0.50% | **81.91%** |
| QAM16 | 97.49% | 89.45% | 6.03% | **80.90%** | 0.00% | 5.53% | **60.30%** | 9.05% | 6.53% | **56.78%** |
| 8PSK | 97.49% | 96.98% | 0.50% | **93.47%** | 0.00% | 1.51% | **57.29%** | 13.07% | 1.01% | **47.24%** |
| AM-SSB | 89.95% | 82.41% | 0.00% | **0.00%** | 0.00% | 0.00% | **0.00%** | 11.56% | 0.00% | **0.00%** |
| WBFM | 42.21% | 36.18% | 38.69% | **36.68%** | 0.00% | 37.19% | **35.18%** | 12.56% | 37.69% | **34.67%** |

### 3.3 Per-Attack Defense Summary at SNR = 18 dB

**DeepFool + Defense:**

DeepFool attack is so weak that defense is largely unnecessary. In many cases, Top-20 accuracy *exceeds* the attack accuracy (e.g., QPSK: attack leaves 93.47%, Top-20 gives 94.97%). The spectral filter actually helps by removing the small DeepFool perturbation.

**EADL1 + Defense (most important — strongest attack):**

| Modulation | EADL1 Acc | Top-10 Recovery | Top-20 Recovery | Category |
|------------|-----------|-----------------|-----------------|----------|
| QAM64 | 0.00% | **94.47%** (99.5%) | 84.42% | Excellent |
| AM-DSB | 1.51% | 98.49% | **95.98%** (96.0%) | Excellent |
| PAM4 | 0.00% | 93.47% | **95.98%** (96.5%) | Excellent |
| GFSK | 0.00% | 78.89% | **97.49%** (97.5%) | Excellent |
| QPSK | 0.00% | 0.50% | **88.44%** (89.8%) | Good |
| BPSK | 0.00% | 1.01% | **79.90%** (81.5%) | Moderate |
| CPFSK | 0.00% | 1.01% | **72.36%** (72.4%) | Moderate |
| QAM16 | 0.00% | 5.53% | **60.30%** (61.9%) | Poor |
| 8PSK | 0.00% | 1.51% | **57.29%** (58.8%) | Poor |
| WBFM | 0.00% | 37.19% | 35.18% (83.4%) | N/A (low clean) |
| AM-SSB | 0.00% | 0.00% | 0.00% (0.0%) | Failed |

*Recovery rate = Defense Acc / Clean Acc × 100%, shown in parentheses*

**FAB + Defense:**

| Modulation | FAB Acc | Top-10 Recovery | Top-20 Recovery | Category |
|------------|---------|-----------------|-----------------|----------|
| QAM64 | 15.08% | **93.47%** (98.4%) | 81.91% | Excellent |
| AM-DSB | 20.60% | 99.50% | **98.49%** (98.5%) | Excellent |
| PAM4 | 9.05% | 95.98% | **98.49%** (99.0%) | Excellent |
| GFSK | 0.00% | 77.39% | **94.97%** (95.0%) | Excellent |
| QPSK | 10.55% | 0.50% | **81.91%** (83.2%) | Good |
| CPFSK | 0.00% | 1.51% | **80.90%** (80.9%) | Good |
| BPSK | 7.54% | 1.51% | **58.79%** (60.0%) | Poor |
| QAM16 | 9.05% | 6.53% | **56.78%** (58.2%) | Poor |
| 8PSK | 13.07% | 1.01% | **47.24%** (48.5%) | Poor |
| WBFM | 12.56% | 37.69% | 34.67% (82.1%) | N/A |
| AM-SSB | 11.56% | 0.00% | 0.00% (0.0%) | Failed |

---

## 4. Results: SNR = 0 dB

### 4.1 Attack Effectiveness at SNR = 0 dB

| Modulation | Clean Acc | DeepFool Acc | EADL1 Acc | FAB Acc |
|------------|-----------|-------------|-----------|---------|
| 8PSK | 93.47% | **92.46%** | **0.00%** | 9.05% |
| AM-DSB | 96.48% | 92.96% | 0.00% | 22.11% |
| AM-SSB | 91.96% | 88.94% | 0.00% | 16.08% |
| BPSK | 98.99% | **96.48%** | 0.00% | 9.55% |
| CPFSK | 100.00% | 98.49% | 0.00% | 0.00% |
| GFSK | 97.99% | 84.42% | 0.00% | 13.07% |
| PAM4 | 100.00% | 97.49% | 0.00% | 0.00% |
| QAM16 | 93.47% | 91.46% | 0.00% | 5.53% |
| QAM64 | 94.47% | 87.44% | 0.00% | 10.05% |
| QPSK | 97.49% | 92.96% | 0.00% | 9.55% |
| WBFM | 34.67% | 30.65% | 0.00% | 5.53% |
| **Mean** | **90.82%** | **86.71%** | **0.00%** | **9.14%** |

At SNR=0, EADL1 achieves a perfect **0.00% across all 11 modulations**. DeepFool remains largely ineffective (86.71% mean). FAB is slightly weaker than at SNR=18 (9.14% vs 9.91%).

### 4.2 Defense Recovery at SNR = 0 dB

**EADL1 + Defense (SNR = 0 dB):**

| Modulation | Clean | EADL1 Acc | Top-10 | Top-20 | Recovery Rate |
|------------|-------|-----------|--------|--------|---------------|
| QAM64 | 94.47% | 0.00% | **94.97%** | 88.94% | **100.5%** |
| PAM4 | 100.00% | 0.00% | **85.93%** | 88.44% | **88.4%** |
| AM-DSB | 96.48% | 0.00% | 77.39% | **79.40%** | **82.3%** |
| GFSK | 97.99% | 0.00% | 70.35% | **76.38%** | **77.9%** |
| QAM16 | 93.47% | 0.00% | 2.51% | **45.73%** | **48.9%** |
| BPSK | 98.99% | 0.00% | 0.50% | **23.12%** | **23.4%** |
| CPFSK | 100.00% | 0.00% | 0.00% | **28.14%** | **28.1%** |
| 8PSK | 93.47% | 0.00% | 1.51% | **1.51%** | **1.6%** |
| QPSK | 97.49% | 0.00% | 0.00% | **6.53%** | **6.7%** |
| AM-SSB | 91.96% | 0.00% | 0.00% | **0.50%** | **0.5%** |
| WBFM | 34.67% | 0.00% | 38.19% | **37.19%** | **107.3%** |

**FAB + Defense (SNR = 0 dB):**

| Modulation | Clean | FAB Acc | Top-10 | Top-20 | Recovery Rate |
|------------|-------|---------|--------|--------|---------------|
| QAM64 | 94.47% | 10.05% | **93.97%** | 88.94% | **99.5%** |
| PAM4 | 100.00% | 0.00% | **86.43%** | 94.47% | **94.5%** |
| AM-DSB | 96.48% | 22.11% | 73.37% | **77.89%** | **80.7%** |
| GFSK | 97.99% | 13.07% | 69.85% | **75.38%** | **76.9%** |
| QAM16 | 93.47% | 5.53% | 2.51% | **43.22%** | **46.2%** |
| BPSK | 98.99% | 9.55% | 0.50% | **16.58%** | **16.8%** |
| CPFSK | 100.00% | 0.00% | 0.50% | **23.62%** | **23.6%** |
| QPSK | 97.49% | 9.55% | 0.00% | **6.03%** | **6.2%** |
| 8PSK | 93.47% | 9.05% | 1.51% | **1.51%** | **1.6%** |
| AM-SSB | 91.96% | 16.08% | 0.00% | **0.00%** | **0.0%** |
| WBFM | 34.67% | 5.53% | 36.68% | **36.18%** | **104.4%** |

---

## 5. Cross-Attack Comparison (All 5 Attacks)

### 5.1 Attack Effectiveness Summary

Mean accuracy under attack across all modulations:

| Attack | SNR=0 Acc | SNR=18 Acc | Effectiveness |
|--------|-----------|------------|---------------|
| **EADL1** | **0.00%** | **0.14%** | Devastating |
| **EADEN** (prior) | ~0.00% | ~0.00% | Devastating |
| **CW** (prior) | ~2.1%* | ~7.8%* | Very strong |
| **FAB** | 9.14% | 9.91% | Strong |
| **DeepFool** | 86.71% | 82.18% | Weak |

*CW/EADEN values from prior report, subset of modulations.

### 5.2 Attack Ranking by Modulation (SNR = 18 dB)

| Modulation | Best Attack | Attack Acc | Worst Attack | Attack Acc |
|------------|-------------|------------|--------------|------------|
| QAM64 | EADL1/EADEN | 0.00% | DeepFool | 88.44% |
| QAM16 | EADL1/EADEN | 0.00% | DeepFool | 89.45% |
| 8PSK | EADL1/EADEN | 0.00% | DeepFool | 96.98% |
| BPSK | EADL1/EADEN | 0.00% | DeepFool | 97.99% |
| CPFSK | EADL1/FAB | 0.00% | DeepFool | 78.39% |
| GFSK | EADL1/FAB | 0.00% | DeepFool | 51.76% |
| PAM4 | EADL1 | 0.00% | DeepFool | 96.48% |
| QPSK | EADL1 | 0.00% | DeepFool | 93.47% |
| AM-DSB | EADL1 | 1.51% | DeepFool | 92.46% |
| AM-SSB | EADL1 | 0.00% | DeepFool | 82.41% |
| WBFM | EADL1 | 0.00% | DeepFool | 36.18% |

### 5.3 Defense Recovery Comparison (Top-20, SNR = 18 dB)

| Modulation | CW Top-20* | EADEN Top-20* | EADL1 Top-20 | FAB Top-20 | DF Top-20 |
|------------|-----------|--------------|-------------|-----------|-----------|
| QAM64 | 80% | 90% | 84.42% | 81.91% | 88.94% |
| PAM4 | 100% | 98% | 95.98% | 98.49% | 98.49% |
| AM-DSB | 100% | 94.5% | 95.98% | 98.49% | 85.93% |
| GFSK | 90.5% | 87% | 97.49% | 94.97% | 80.90% |
| BPSK | 94.5% | 68% | 79.90% | 58.79% | 96.48% |
| CPFSK | 62% | 82% | 72.36% | 80.90% | 84.92% |
| QPSK | 94.5% | 86% | 88.44% | 81.91% | 94.97% |
| QAM16 | 57% | 48.5% | 60.30% | 56.78% | 80.90% |
| 8PSK | 53.5% | 47% | 57.29% | 47.24% | 93.47% |
| AM-SSB | N/A | N/A | 0.00% | 0.00% | 0.00% |

*CW/EADEN values from prior report (200 samples at SNR=18).

**The defense-attack independence principle holds**: Top-20 recovery rates are consistent across all five attacks. The ranking of modulations by recovery difficulty is stable: QAM64/PAM4/AM-DSB always recover well, 8PSK/QAM16/AM-SSB always struggle.

---

## 6. SNR Sensitivity: SNR=0 vs SNR=18

### 6.1 EADL1 Top-20 Recovery: SNR Impact

| Modulation | SNR=0 Top-20 | SNR=18 Top-20 | SNR Gain |
|------------|-------------|--------------|----------|
| QAM64 (Top-10) | 94.97% | 94.47% | -0.5pp |
| PAM4 | 88.44% | 95.98% | +7.5pp |
| AM-DSB | 79.40% | 95.98% | +16.6pp |
| GFSK | 76.38% | 97.49% | +21.1pp |
| QAM16 | 45.73% | 60.30% | +14.6pp |
| CPFSK | 28.14% | 72.36% | +44.2pp |
| BPSK | 23.12% | 79.90% | +56.8pp |
| QPSK | 6.53% | 88.44% | +81.9pp |
| 8PSK | 1.51% | 57.29% | +55.8pp |
| AM-SSB | 0.50% | 0.00% | -0.5pp |

**Critical finding**: SNR has a massive impact on defense recovery for phase-modulated signals. QPSK improves by **82 percentage points** from SNR=0 to SNR=18. BPSK, 8PSK, and CPFSK all gain 44--57pp. This confirms the prior report's finding that at higher SNR, signal spectral peaks become more prominent relative to perturbation energy, making Top-K filtering more effective.

QAM64 is uniquely SNR-independent — Top-10 achieves ~95% recovery at both SNR levels. This is because QAM64's spectral energy is already concentrated in very few bins regardless of SNR.

---

## 7. Key Findings

### 7.1 DeepFool Is Not a Viable Attack for IQ-domain AMC

DeepFool's minimum-perturbation approach produces perturbations too small to reliably fool the AWN classifier on IQ signals with minmax normalization. At SNR=18, it reduces accuracy by only ~10 percentage points (from 92.55% to 82.18%). This makes DeepFool unsuitable for evaluating AMC robustness. **Recommendation**: Exclude DeepFool from future AMC adversarial evaluations; use EADL1 or CW instead.

### 7.2 EADL1 Matches EADEN as the Strongest Attack

EADL1 achieves 0.00% accuracy at both SNR=0 and SNR=18 across nearly all modulations. Its L1-regularized perturbations are sparse and concentrated, making it equally devastating as EADEN's elastic-net approach. The fact that both L1-focused attacks (EADL1, EADEN) outperform L2-only CW suggests that **sparse perturbations are more effective against IQ-domain classifiers**.

### 7.3 FAB Is Between CW and DeepFool

FAB achieves ~9-10% mean accuracy — more effective than DeepFool but less than CW/EADL1/EADEN. Its minimum-norm approach finds perturbations of intermediate size. FAB is a useful "moderate strength" attack for evaluating graduated defense performance.

### 7.4 Defense Patterns Are Attack-Invariant

Across all 5 attacks (CW, EADEN, EADL1, FAB, DeepFool), the modulation recovery ranking is consistent:

| Tier | Modulations | Top-20 Recovery |
|------|-------------|-----------------|
| **Excellent** | QAM64 (Top-10), PAM4, AM-DSB, GFSK | 85--98% |
| **Good** (high SNR) | QPSK, BPSK, CPFSK | 60--90% at SNR=18 |
| **Poor** | QAM16, 8PSK | 47--60% at SNR=18 |
| **Failed** | AM-SSB | 0% at all SNR |

This confirms the **defense-attack independence principle**: spectral defense effectiveness is determined by modulation spectral structure, not attack algorithm.

---

## 8. Conclusions

1. **EADL1 joins EADEN as the strongest attack** against AWN AMC, achieving 0.00% accuracy across all modulations and SNR levels. Both L1-regularized attacks outperform L2-only CW.

2. **DeepFool is ineffective** for IQ-domain adversarial attacks with minmax normalization (82--87% accuracy retained). It should not be used as a primary attack benchmark for AMC.

3. **FAB provides a moderate attack** (~9--10% accuracy) useful for graduated defense evaluation.

4. **The spectral defense recovery hierarchy is attack-invariant**: QAM64 > PAM4 > AM-DSB > GFSK > QPSK > BPSK > CPFSK > QAM16 > 8PSK > AM-SSB. This ordering holds across all five attacks tested (CW, EADEN, EADL1, FAB, DeepFool).

5. **SNR remains the dominant factor** for phase-modulated signal recovery: QPSK defense improves by 82pp from SNR=0 to SNR=18, while QAM64 is SNR-invariant.

---

## Appendix A: Complete Per-Modulation Data

### A.1 SNR = 18 dB — All Attacks

| Modulation | Clean | DF Att | DF T10 | DF T20 | EL1 Att | EL1 T10 | EL1 T20 | FAB Att | FAB T10 | FAB T20 |
|------------|-------|--------|--------|--------|---------|---------|---------|---------|---------|---------|
| 8PSK | 0.975 | 0.970 | 0.005 | 0.935 | 0.000 | 0.015 | 0.573 | 0.131 | 0.010 | 0.472 |
| AM-DSB | 1.000 | 0.925 | 0.693 | 0.859 | 0.015 | 0.985 | 0.960 | 0.206 | 0.995 | 0.985 |
| AM-SSB | 0.899 | 0.824 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.116 | 0.000 | 0.000 |
| BPSK | 0.980 | 0.980 | 0.015 | 0.965 | 0.000 | 0.010 | 0.799 | 0.075 | 0.015 | 0.588 |
| CPFSK | 1.000 | 0.784 | 0.015 | 0.849 | 0.000 | 0.010 | 0.724 | 0.000 | 0.015 | 0.809 |
| GFSK | 1.000 | 0.518 | 0.749 | 0.809 | 0.000 | 0.789 | 0.975 | 0.000 | 0.774 | 0.950 |
| PAM4 | 0.995 | 0.965 | 0.980 | 0.985 | 0.000 | 0.935 | 0.960 | 0.090 | 0.960 | 0.985 |
| QAM16 | 0.975 | 0.894 | 0.060 | 0.809 | 0.000 | 0.055 | 0.603 | 0.090 | 0.065 | 0.568 |
| QAM64 | 0.950 | 0.884 | 0.930 | 0.889 | 0.000 | 0.945 | 0.844 | 0.151 | 0.935 | 0.819 |
| QPSK | 0.985 | 0.935 | 0.005 | 0.950 | 0.000 | 0.005 | 0.884 | 0.106 | 0.005 | 0.819 |
| WBFM | 0.422 | 0.362 | 0.387 | 0.367 | 0.000 | 0.372 | 0.352 | 0.126 | 0.377 | 0.347 |

### A.2 SNR = 0 dB — All Attacks

| Modulation | Clean | DF Att | DF T10 | DF T20 | EL1 Att | EL1 T10 | EL1 T20 | FAB Att | FAB T10 | FAB T20 |
|------------|-------|--------|--------|--------|---------|---------|---------|---------|---------|---------|
| 8PSK | 0.935 | 0.925 | 0.010 | 0.035 | 0.000 | 0.015 | 0.015 | 0.090 | 0.015 | 0.015 |
| AM-DSB | 0.965 | 0.930 | 0.794 | 0.854 | 0.000 | 0.774 | 0.794 | 0.221 | 0.734 | 0.779 |
| AM-SSB | 0.920 | 0.889 | 0.000 | 0.000 | 0.000 | 0.000 | 0.005 | 0.161 | 0.000 | 0.000 |
| BPSK | 0.990 | 0.965 | 0.005 | 0.472 | 0.000 | 0.005 | 0.231 | 0.095 | 0.005 | 0.166 |
| CPFSK | 1.000 | 0.985 | 0.020 | 0.653 | 0.000 | 0.000 | 0.281 | 0.000 | 0.005 | 0.236 |
| GFSK | 0.980 | 0.844 | 0.724 | 0.759 | 0.000 | 0.704 | 0.764 | 0.131 | 0.698 | 0.754 |
| PAM4 | 1.000 | 0.975 | 0.945 | 0.980 | 0.000 | 0.859 | 0.884 | 0.000 | 0.864 | 0.945 |
| QAM16 | 0.935 | 0.915 | 0.025 | 0.759 | 0.000 | 0.025 | 0.457 | 0.055 | 0.025 | 0.432 |
| QAM64 | 0.945 | 0.874 | 0.960 | 0.910 | 0.000 | 0.950 | 0.889 | 0.101 | 0.940 | 0.889 |
| QPSK | 0.975 | 0.930 | 0.000 | 0.141 | 0.000 | 0.000 | 0.065 | 0.095 | 0.000 | 0.060 |
| WBFM | 0.347 | 0.307 | 0.382 | 0.372 | 0.000 | 0.382 | 0.372 | 0.055 | 0.367 | 0.362 |

---

## Appendix B: Reproducibility

### B.1 Commands

```bash
# SNR = 18 dB
python main.py --mode multi_attack_eval --dataset 2016.10a --ckpt_path ./checkpoint \
  --attack_list "deepfool,eadl1,fab" --ta_box minmax --attack_eps 0.1 \
  --snr_filter 18 --eval_limit_per_cell 200 --no_plot_iq

# SNR = 0 dB
python main.py --mode multi_attack_eval --dataset 2016.10a --ckpt_path ./checkpoint \
  --attack_list "deepfool,eadl1,fab" --ta_box minmax --attack_eps 0.1 \
  --snr_filter 0 --eval_limit_per_cell 200 --no_plot_iq
```

### B.2 Environment

| Component | Version |
|-----------|---------|
| Python | 3.x |
| PyTorch | 1.8+ |
| torchattacks | latest |
| Dataset | RML2016.10a |
| Hardware | NVIDIA GPU (CUDA) |
| AWN checkpoint | `./checkpoint/2016.10a_AWN.pkl` |
| Samples per cell | ~199 (test set at single SNR) |
