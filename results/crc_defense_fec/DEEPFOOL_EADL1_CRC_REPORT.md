# CRC Defense Evaluation: DeepFool and EADL1 Adversarial Attacks on AMC and Data-Link Integrity

---

## Abstract

We evaluate the impact of **DeepFool** and **EADL1** (Elastic-net Attack, L1 Decision) adversarial attacks on both the AMC control plane and the data-link integrity (CRC pass rate) using a synthetic TX/RX chain with FEC. Experiments across eight digital modulations at SNR=0 dB and SNR=18 dB reveal a striking contrast: **EADL1 is devastating on the control plane** (AMC accuracy drops to 0.4--7.6%) while **DeepFool is largely ineffective** (AMC accuracy remains 70--96%). However, both attacks cause comparable **data-plane corruption** on high-order modulations (QAM64, PAM4), with Adv+Oracle CRC dropping 18--32 percentage points at SNR=18. The end-to-end metric (Adv+AMC CRC) reveals that EADL1 collapses the full communication link to near-zero CRC pass rate across all modulations, while DeepFool's weak control-plane attack preserves partial link functionality. These findings extend the prior CW result that adversarial attacks on AMC are primarily control-plane, with the caveat that **data-plane corruption is attack-independent and modulation-dependent** -- all three attacks (CW, EADL1, DeepFool) cause similar data corruption on QAM64 and PAM4 regardless of their control-plane effectiveness.

---

## 1. Introduction

Our prior CRC defense experiment (CW attack, CRC_EXPERIMENT_REPORT) established that CW adversarial attacks on AMC are primarily **control-plane attacks**: the perturbation fools the classifier but does not corrupt the underlying data when the correct demodulator is used (Oracle scenario). The key exception was QAM64 and PAM4, where even Oracle demodulation showed CRC degradation.

This report extends that analysis to two additional attacks:

- **EADL1**: An elastic-net (L1+L2) attack that produces sparse perturbations. It represents the strongest class of attacks against IQ-domain AMC classifiers.
- **DeepFool**: A minimum-perturbation attack that seeks the smallest change to cross the decision boundary. It represents the weakest practical attack.

By comparing a devastating attack (EADL1) with a weak one (DeepFool) against the same CRC defense pipeline, we can disentangle **control-plane damage** (AMC misclassification) from **data-plane damage** (waveform corruption).

---

## 2. Experimental Setup

### 2.1 Pipeline

```
Random bits --> CRC-8 --> Modulate --> RRC pulse-shape --> AWGN channel
    --> [Attack] --> AWN Classifier --> Demodulate --> CRC Check
```

The synthetic TX/RX chain generates bursts of 16 symbols (2 pilot + 14 data) with 8 samples/symbol, RRC rolloff=0.35, target_rms=0.0082 (matching RML2016.10a amplitude distribution). Guard symbols are used at TX to avoid RRC filter transients. The adversarial perturbation is computed on real RML2016.10a data (Track A) and transferred to the synthetic burst (Track B).

### 2.2 Parameters

| Parameter | Value |
|-----------|-------|
| Dataset | RML2016.10a |
| Model | AWN (Adaptive Wavelet Network) |
| SNR levels | 0 dB, 18 dB |
| Bursts per (mod, SNR) cell | 500 |
| Modulations | BPSK, QPSK, 8PSK, QAM16, QAM64, PAM4, GFSK, CPFSK |
| Attack (EADL1) | torchattacks EADL1, minmax box |
| Attack (DeepFool) | torchattacks DeepFool, minmax box |
| Perturbation transfer | Track A real delta applied to Track B synthetic burst |

### 2.3 Scenario Definitions

| Scenario | Signal | Demod Selection | What It Measures |
|----------|--------|-----------------|------------------|
| **Clean+Oracle** | Original | True modulation | Baseline CRC (channel quality) |
| **Clean+AMC** | Original | AWN prediction | AMC reliability on clean signals |
| **Adv+Oracle** | Attacked | True modulation | Does the attack corrupt data? |
| **Adv+AMC** | Attacked | AWN prediction | Full end-to-end attack impact |

---

## 3. Results: EADL1 Attack

### 3.1 Control-Plane Damage: AMC Accuracy

EADL1 is devastating -- AMC accuracy drops to near-zero for all modulations at both SNR levels.

| Modulation | Clean AMC Acc | EADL1 AMC (SNR=0) | EADL1 AMC (SNR=18) |
|------------|:------------:|:------------------:|:-------------------:|
| BPSK | 98.2% | **2.0%** | **1.2%** |
| QPSK | 96.2% | **3.4%** | **1.0%** |
| 8PSK | 94.4% | **4.2%** | **3.0%** |
| QAM16 | 91.2% | **7.6%** | **6.6%** |
| QAM64 | 93.4% | **4.0%** | **6.4%** |
| PAM4 | 98.8% | **0.4%** | **0.4%** |
| GFSK | 98.0% | **0.8%** | **1.6%** |
| CPFSK | 100.0% | **0.8%** | **0.4%** |
| **Mean** | **96.3%** | **2.9%** | **2.6%** |

EADL1 reduces AMC accuracy to an average of 2.6--2.9% across all modulations. No modulation retains more than 7.6% accuracy. This confirms EADL1 as one of the most effective attacks against IQ-domain AMC classifiers, consistent with the IFFT spectral defense report findings.

### 3.2 Data-Plane Damage: CRC Under Oracle Demodulation

The critical question: does EADL1 corrupt the underlying data, or is the damage purely control-plane?

**SNR = 18 dB (high quality channel):**

| Modulation | Clean+Oracle CRC | Adv+Oracle CRC | CRC Delta | Data Corrupted? |
|------------|:----------------:|:--------------:|:---------:|:---------------:|
| BPSK | 100.0% | **100.0%** | 0.0pp | No |
| QPSK | 100.0% | **100.0%** | 0.0pp | No |
| 8PSK | 100.0% | **100.0%** | 0.0pp | No |
| QAM16 | 100.0% | **99.6%** | -0.4pp | Negligible |
| QAM64 | 95.6% | **77.6%** | **-18.0pp** | **Yes** |
| PAM4 | 100.0% | **67.8%** | **-32.2pp** | **Yes** |
| GFSK | 99.4% | **99.4%** | 0.0pp | No |
| CPFSK | 100.0% | **100.0%** | 0.0pp | No |

**SNR = 0 dB (noisy channel):**

| Modulation | Clean+Oracle CRC | Adv+Oracle CRC | CRC Delta | Data Corrupted? |
|------------|:----------------:|:--------------:|:---------:|:---------------:|
| BPSK | 99.8% | **98.8%** | -1.0pp | Negligible |
| QPSK | 79.2% | **77.6%** | -1.6pp | Negligible |
| 8PSK | 6.8% | **5.8%** | -1.0pp | Negligible |
| QAM16 | 0.6% | **0.6%** | 0.0pp | No (floor) |
| QAM64 | 0.2% | **0.2%** | 0.0pp | No (floor) |
| PAM4 | 22.6% | **14.8%** | **-7.8pp** | Moderate |
| GFSK | 5.8% | **5.8%** | 0.0pp | No |
| CPFSK | 6.8% | **6.8%** | 0.0pp | No |

**Key findings:**

1. **EADL1 corrupts data for PAM4 and QAM64 at high SNR.** PAM4 drops 32.2 percentage points (100% to 67.8%) and QAM64 drops 18.0pp (95.6% to 77.6%) even with Oracle demodulation.
2. **Phase-modulated signals (BPSK, QPSK, 8PSK) are immune to data-plane corruption.** The perturbation fools the classifier but does not degrade demodulation.
3. **At SNR=0, the channel noise floor masks data-plane corruption.** QAM16/QAM64 baseline CRC is already near 0%, so perturbation effects are invisible.

### 3.3 End-to-End Impact: CRC Under AMC-Guided Demodulation

When the receiver relies on the compromised AMC to select the demodulator, the link collapses completely.

**SNR = 18 dB:**

| Modulation | Clean+AMC CRC | Adv+AMC CRC | CRC Collapse |
|------------|:------------:|:-----------:|:------------:|
| BPSK | 99.4% | **2.2%** | 97.2pp |
| QPSK | 98.8% | **1.8%** | 97.0pp |
| 8PSK | 98.0% | **3.4%** | 94.6pp |
| QAM16 | 94.2% | **7.2%** | 87.0pp |
| QAM64 | 90.6% | **5.4%** | 85.2pp |
| PAM4 | 98.8% | **1.4%** | 97.4pp |
| GFSK | 99.4% | **1.6%** | 97.8pp |
| CPFSK | 100.0% | **0.6%** | 99.4pp |

**SNR = 0 dB:**

| Modulation | Clean+AMC CRC | Adv+AMC CRC | CRC Collapse |
|------------|:------------:|:-----------:|:------------:|
| BPSK | 98.8% | **2.6%** | 96.2pp |
| QPSK | 76.6% | **2.6%** | 74.0pp |
| 8PSK | 6.4% | **0.8%** | 5.6pp |
| QAM16 | 0.8% | **0.2%** | 0.6pp |
| QAM64 | 0.2% | **0.2%** | 0.0pp |
| PAM4 | 22.6% | **1.0%** | 21.6pp |
| GFSK | 5.6% | **0.0%** | 5.6pp |
| CPFSK | 6.8% | **0.4%** | 6.4pp |

EADL1 destroys the communication link: at SNR=18, Adv+AMC CRC ranges from 0.6% to 7.2%. The attack mechanism is clear -- AMC misclassification causes the receiver to apply the wrong demodulator, producing random bits and near-zero CRC.

---

## 4. Results: DeepFool Attack

### 4.1 Control-Plane Damage: AMC Accuracy

DeepFool is a weak attack on IQ-domain AMC. Most modulations retain 70--96% accuracy.

| Modulation | Clean AMC Acc | DeepFool AMC (SNR=0) | DeepFool AMC (SNR=18) |
|------------|:------------:|:--------------------:|:---------------------:|
| BPSK | 98.2% | 93.0% | 87.4% |
| QPSK | 96.2% | 88.0% | 96.4% |
| 8PSK | 94.4% | 89.6% | 84.6% |
| QAM16 | 91.2% | 85.0% | 90.4% |
| QAM64 | 93.4% | 82.2% | 85.6% |
| PAM4 | 98.8% | 89.4% | 91.6% |
| GFSK | 98.0% | 84.0% | 70.0% |
| CPFSK | 100.0% | 95.0% | 86.6% |
| **Mean** | **96.3%** | **88.3%** | **86.6%** |

DeepFool reduces AMC accuracy by only 8--10 percentage points on average. GFSK shows the largest drop at SNR=18 (100% to 70%), but most modulations retain >85% accuracy. This confirms DeepFool's minimum-perturbation approach produces changes too small to reliably fool the AWN classifier.

### 4.2 Data-Plane Damage: CRC Under Oracle Demodulation

Despite being a weak control-plane attack, DeepFool still causes data-plane corruption on the same modulations as EADL1.

**SNR = 18 dB:**

| Modulation | Clean+Oracle CRC | Adv+Oracle CRC | CRC Delta | Data Corrupted? |
|------------|:----------------:|:--------------:|:---------:|:---------------:|
| BPSK | 100.0% | **100.0%** | 0.0pp | No |
| QPSK | 100.0% | **100.0%** | 0.0pp | No |
| 8PSK | 100.0% | **99.6%** | -0.4pp | Negligible |
| QAM16 | 100.0% | **100.0%** | 0.0pp | No |
| QAM64 | 95.6% | **71.6%** | **-24.0pp** | **Yes** |
| PAM4 | 100.0% | **77.4%** | **-22.6pp** | **Yes** |
| GFSK | 99.4% | **99.4%** | 0.0pp | No |
| CPFSK | 100.0% | **100.0%** | 0.0pp | No |

**SNR = 0 dB:**

| Modulation | Clean+Oracle CRC | Adv+Oracle CRC | CRC Delta | Data Corrupted? |
|------------|:----------------:|:--------------:|:---------:|:---------------:|
| BPSK | 99.8% | **100.0%** | +0.2pp | No |
| QPSK | 79.2% | **79.4%** | +0.2pp | No |
| 8PSK | 6.8% | **5.6%** | -1.2pp | Negligible |
| QAM16 | 0.6% | **0.6%** | 0.0pp | No (floor) |
| QAM64 | 0.2% | **0.2%** | 0.0pp | No (floor) |
| PAM4 | 22.6% | **14.0%** | **-8.6pp** | Moderate |
| GFSK | 5.8% | **5.8%** | 0.0pp | No |
| CPFSK | 6.8% | **6.8%** | 0.0pp | No |

**Key findings:**

1. **DeepFool corrupts QAM64 and PAM4 data at high SNR**, with QAM64 dropping 24.0pp and PAM4 dropping 22.6pp. This is **comparable to EADL1** despite DeepFool being a far weaker control-plane attack.
2. **The data-plane corruption pattern is identical to EADL1**: only QAM64 and PAM4 are affected; phase-modulated signals are immune.
3. At SNR=0, PAM4 shows moderate corruption (-8.6pp), also consistent with EADL1.

### 4.3 End-to-End Impact: CRC Under AMC-Guided Demodulation

Since DeepFool only partially degrades AMC, the end-to-end CRC impact is proportional to residual AMC accuracy.

**SNR = 18 dB:**

| Modulation | Clean+AMC CRC | Adv+AMC CRC | CRC Retained |
|------------|:------------:|:-----------:|:------------:|
| BPSK | 99.4% | **87.8%** | 88.3% |
| QPSK | 98.8% | **96.6%** | 97.8% |
| 8PSK | 98.0% | **84.4%** | 86.1% |
| QAM16 | 94.2% | **90.4%** | 95.9% |
| QAM64 | 90.6% | **62.0%** | 68.4% |
| PAM4 | 98.8% | **70.8%** | 71.7% |
| GFSK | 99.4% | **69.4%** | 69.8% |
| CPFSK | 100.0% | **86.8%** | 86.8% |

**SNR = 0 dB:**

| Modulation | Clean+AMC CRC | Adv+AMC CRC | CRC Retained |
|------------|:------------:|:-----------:|:------------:|
| BPSK | 98.8% | **93.0%** | 94.1% |
| QPSK | 76.6% | **69.8%** | 91.1% |
| 8PSK | 6.4% | **4.8%** | 75.0% |
| QAM16 | 0.8% | **0.4%** | 50.0% |
| QAM64 | 0.2% | **0.0%** | 0.0% |
| PAM4 | 22.6% | **12.0%** | 53.1% |
| GFSK | 5.6% | **5.2%** | 92.9% |
| CPFSK | 6.8% | **6.4%** | 94.1% |

Unlike EADL1 which collapses the link entirely, DeepFool preserves partial functionality. At SNR=18, QPSK retains 96.6% CRC, QAM16 retains 90.4%, and BPSK retains 87.8%. The most degraded modulations (QAM64: 62.0%, PAM4: 70.8%) suffer from a combination of weak control-plane attack and data-plane corruption.

---

## 5. Cross-Attack Comparison: CW vs EADL1 vs DeepFool

### 5.1 Control-Plane Effectiveness (AMC Accuracy Under Attack, SNR=18)

| Modulation | CW AMC* | EADL1 AMC | DeepFool AMC | Most Effective |
|------------|:-------:|:---------:|:------------:|:--------------:|
| BPSK | ~1% | 1.2% | 87.4% | CW/EADL1 |
| QPSK | ~1% | 1.0% | 96.4% | CW/EADL1 |
| 8PSK | ~3% | 3.0% | 84.6% | CW/EADL1 |
| QAM16 | ~0% | 6.6% | 90.4% | CW |
| QAM64 | ~0.5% | 6.4% | 85.6% | CW |
| PAM4 | ~85.5% | 0.4% | 91.6% | EADL1 |
| GFSK | ~25.5% | 1.6% | 70.0% | EADL1 |
| CPFSK | ~0.4% | 0.4% | 86.6% | CW/EADL1 |

*CW values from prior CRC experiment report (n=200, SNR=18).

EADL1 and CW are both devastating on the control plane, with EADL1 being particularly effective against PAM4 and GFSK where CW partially fails. DeepFool is an order of magnitude weaker across all modulations.

### 5.2 Data-Plane Corruption (Adv+Oracle CRC, SNR=18)

This is the most important comparison -- it reveals whether different attack algorithms cause different levels of data corruption.

| Modulation | CW Adv+Oracle CRC* | EADL1 Adv+Oracle CRC | DeepFool Adv+Oracle CRC |
|------------|:------------------:|:--------------------:|:----------------------:|
| BPSK | 100.0% | 100.0% | 100.0% |
| QPSK | 100.0% | 100.0% | 100.0% |
| 8PSK | 100.0% | 100.0% | 99.6% |
| QAM16 | 99.5% | 99.6% | 100.0% |
| **QAM64** | **76.0%** | **77.6%** | **71.6%** |
| **PAM4** | **71.0%** | **67.8%** | **77.4%** |
| GFSK | 99.5% | 99.4% | 99.4% |
| CPFSK | 100.0% | 100.0% | 100.0% |

*CW values from prior CRC experiment report (n=200, SNR=18).

**This is the central finding of the report:** all three attacks -- CW (L2-optimized), EADL1 (L1+L2 sparse), and DeepFool (minimum perturbation) -- cause **virtually identical data-plane corruption patterns**:

- **QAM64**: 71.6--77.6% Oracle CRC across all three attacks (19--24pp drop)
- **PAM4**: 67.8--77.4% Oracle CRC across all three attacks (23--32pp drop)
- **All other modulations**: >99% Oracle CRC (no data corruption)

The data-plane corruption is **independent of attack algorithm and attack strength**. A minimum-perturbation attack (DeepFool) that barely affects AMC accuracy causes the same data corruption as a devastating attack (EADL1) that reduces AMC to near-zero.

### 5.3 End-to-End CRC (Adv+AMC, SNR=18)

| Modulation | CW Adv+AMC* | EADL1 Adv+AMC | DeepFool Adv+AMC |
|------------|:-----------:|:-------------:|:----------------:|
| BPSK | 39.0% | 2.2% | 87.8% |
| QPSK | 1.5% | 1.8% | 96.6% |
| 8PSK | 0.5% | 3.4% | 84.4% |
| QAM16 | 0.5% | 7.2% | 90.4% |
| QAM64 | 0.5% | 5.4% | 62.0% |
| PAM4 | 2.0% | 1.4% | 70.8% |
| GFSK | 0.0% | 1.6% | 69.4% |
| CPFSK | 18.5% | 0.6% | 86.8% |

*CW values from prior CRC experiment report (n=200, SNR=18).

The end-to-end impact is dominated by control-plane effectiveness:
- **EADL1**: 0.6--7.2% CRC across all modulations (complete link failure)
- **CW**: 0.0--39.0% CRC (near-complete failure, with BPSK/CPFSK as partial exceptions)
- **DeepFool**: 62.0--96.6% CRC (partial degradation, link remains usable for most modulations)

### 5.4 The Data-Plane Corruption Independence Principle

The near-identical Oracle CRC values across three fundamentally different attacks (Table 5.2) establishes a key principle: **data-plane corruption is determined by the perturbation's interaction with the modulation's symbol geometry, not by the perturbation's optimization objective.**

Why QAM64 and PAM4 are uniquely vulnerable:

- **QAM64** has 64 densely-packed constellation points (6 bits/symbol). Even small perturbations can push symbols across decision boundaries, corrupting bits. The minimum distance between constellation points is small relative to the perturbation magnitude.
- **PAM4** has 4 amplitude levels with relatively small spacing in the IQ plane. While simpler than QAM64, the purely amplitude-based modulation is sensitive to additive perturbations that shift symbol amplitudes.
- **Phase modulations** (BPSK, QPSK, 8PSK) have constant-envelope constellations where additive perturbation must overcome larger decision regions. The perturbation energy is insufficient to move symbols across phase boundaries.

---

## 6. SNR Sensitivity Analysis

### 6.1 Control-Plane Damage: SNR=0 vs SNR=18

| Modulation | EADL1 (SNR=0) | EADL1 (SNR=18) | DeepFool (SNR=0) | DeepFool (SNR=18) |
|------------|:-------------:|:--------------:|:----------------:|:-----------------:|
| BPSK | 2.0% | 1.2% | 93.0% | 87.4% |
| QPSK | 3.4% | 1.0% | 88.0% | 96.4% |
| 8PSK | 4.2% | 3.0% | 89.6% | 84.6% |
| QAM16 | 7.6% | 6.6% | 85.0% | 90.4% |
| QAM64 | 4.0% | 6.4% | 82.2% | 85.6% |
| PAM4 | 0.4% | 0.4% | 89.4% | 91.6% |
| GFSK | 0.8% | 1.6% | 84.0% | 70.0% |
| CPFSK | 0.8% | 0.4% | 95.0% | 86.6% |

EADL1 is SNR-independent -- it is equally devastating at both SNR levels. DeepFool shows irregular SNR dependence: some modulations are attacked more effectively at high SNR (GFSK: 84%->70%, CPFSK: 95%->87%), while others are attacked less effectively (QPSK: 88%->96%).

### 6.2 Data-Plane Damage: Oracle CRC at SNR=0 vs SNR=18

| Modulation | EADL1 Oracle (SNR=0) | EADL1 Oracle (SNR=18) | DF Oracle (SNR=0) | DF Oracle (SNR=18) |
|------------|:--------------------:|:---------------------:|:-----------------:|:------------------:|
| BPSK | 98.8% | 100.0% | 100.0% | 100.0% |
| QPSK | 77.6% | 100.0% | 79.4% | 100.0% |
| 8PSK | 5.8% | 100.0% | 5.6% | 99.6% |
| QAM16 | 0.6% | 99.6% | 0.6% | 100.0% |
| QAM64 | 0.2% | 77.6% | 0.2% | 71.6% |
| PAM4 | 14.8% | 67.8% | 14.0% | 77.4% |
| GFSK | 5.8% | 99.4% | 5.8% | 99.4% |
| CPFSK | 6.8% | 100.0% | 6.8% | 100.0% |

At SNR=0, the Oracle CRC values are dominated by channel noise, not attack effects. The attack-vs-clean Oracle CRC differences are negligible at SNR=0 for all modulations except PAM4 (-7.8pp for EADL1, -8.6pp for DeepFool). This means data-plane corruption is primarily a high-SNR phenomenon where the perturbation exceeds the noise floor.

---

## 7. Detailed Per-Modulation Analysis

### 7.1 QAM64: Dual-Plane Vulnerability

QAM64 is unique in being vulnerable on both the control plane and data plane.

| Scenario | SNR=0 | SNR=18 |
|----------|:-----:|:------:|
| Clean+Oracle CRC | 0.2% | 95.6% |
| Clean+AMC CRC | 0.2% | 90.6% |
| EADL1 Adv+Oracle CRC | 0.2% | **77.6%** |
| EADL1 Adv+AMC CRC | 0.2% | **5.4%** |
| DeepFool Adv+Oracle CRC | 0.2% | **71.6%** |
| DeepFool Adv+AMC CRC | 0.0% | **62.0%** |

At SNR=18:
- EADL1 degrades Oracle CRC by 18pp (data corruption) and AMC CRC by 85pp (control-plane collapse)
- DeepFool degrades Oracle CRC by 24pp (data corruption) but AMC CRC only by 29pp (weak control-plane attack)
- DeepFool's end-to-end CRC (62.0%) is actually worse than its Oracle CRC (71.6%), showing both attack vectors compound

### 7.2 PAM4: Worst Data-Plane Corruption

PAM4 shows the most severe data-plane corruption of any modulation.

| Scenario | SNR=0 | SNR=18 |
|----------|:-----:|:------:|
| Clean+Oracle CRC | 22.6% | 100.0% |
| Clean+AMC CRC | 22.6% | 98.8% |
| EADL1 Adv+Oracle CRC | **14.8%** | **67.8%** |
| EADL1 Adv+AMC CRC | **1.0%** | **1.4%** |
| DeepFool Adv+Oracle CRC | **14.0%** | **77.4%** |
| DeepFool Adv+AMC CRC | **12.0%** | **70.8%** |

PAM4 under EADL1 at SNR=18 shows the starkest contrast between Oracle (67.8%) and AMC-guided (1.4%) CRC, confirming the dual attack mechanism: data corruption + control-plane failure.

### 7.3 BPSK: Purely Control-Plane Attack

BPSK represents the ideal case where the attack is entirely control-plane.

| Scenario | SNR=0 | SNR=18 |
|----------|:-----:|:------:|
| Clean+Oracle CRC | 99.8% | 100.0% |
| EADL1 Adv+Oracle CRC | **98.8%** | **100.0%** |
| EADL1 Adv+AMC CRC | **2.6%** | **2.2%** |
| DeepFool Adv+Oracle CRC | **100.0%** | **100.0%** |
| DeepFool Adv+AMC CRC | **93.0%** | **87.8%** |

Both attacks leave Oracle CRC at 100% (no data corruption). The entire end-to-end damage comes from AMC misclassification. DeepFool's weak AMC attack means BPSK retains 88% CRC even under attack.

---

## 8. Key Findings

### 8.1 Data-Plane Corruption Is Attack-Independent

The most significant finding is that **three fundamentally different attack algorithms produce nearly identical data-plane corruption**:

| Modulation | CW Oracle CRC | EADL1 Oracle CRC | DeepFool Oracle CRC | Range |
|------------|:------------:|:----------------:|:-------------------:|:-----:|
| QAM64 (SNR=18) | 76.0% | 77.6% | 71.6% | 6.0pp |
| PAM4 (SNR=18) | 71.0% | 67.8% | 77.4% | 9.6pp |
| All others (SNR=18) | >99% | >99% | >99% | <1pp |

The within-modulation variance across attacks (6--10pp) is small compared to the between-modulation variance (>20pp). This means data-plane vulnerability is a **property of the modulation**, not the attack.

### 8.2 Control-Plane Damage Determines End-to-End Impact

Since data-plane corruption only affects QAM64 and PAM4, the end-to-end CRC is dominated by control-plane effectiveness for most modulations:

| Attack | Mean AMC Acc (SNR=18) | Mean Adv+AMC CRC (SNR=18) | Link Status |
|--------|:---------------------:|:--------------------------:|:-----------:|
| EADL1 | 2.6% | 2.8% | **Dead** |
| CW | ~7% | ~7.8% | **Dead** |
| DeepFool | 86.6% | 80.9% | **Degraded** |

### 8.3 DeepFool Is Not Viable for AMC Attack Evaluation

DeepFool retains 86.6% mean AMC accuracy under attack -- it barely degrades the control plane. For CRC evaluation purposes, DeepFool's primary effect is data-plane corruption on QAM64/PAM4, which is shared by all attacks. **Recommendation**: Use EADL1 or CW for AMC security evaluations; DeepFool underestimates the threat.

### 8.4 EADL1 Is the Strongest Combined Attack

EADL1 maximizes damage on both planes simultaneously:
- Control plane: 0.4--7.6% AMC accuracy (complete misclassification)
- Data plane: 67.8--77.6% Oracle CRC on QAM64/PAM4 (comparable to CW/DeepFool)
- End-to-end: 0.6--7.2% Adv+AMC CRC at SNR=18 (complete link failure)

---

## 9. Conclusions

1. **Data-plane corruption is attack-independent.** CW, EADL1, and DeepFool -- spanning L2-optimized, L1+L2 sparse, and minimum-perturbation approaches -- produce virtually identical data corruption on QAM64 (72--78% Oracle CRC) and PAM4 (68--77% Oracle CRC) at SNR=18. All other modulations are immune to data-plane corruption. This confirms that data-plane vulnerability is an intrinsic property of the modulation's constellation geometry.

2. **EADL1 is the most effective end-to-end attack**, reducing Adv+AMC CRC to 0.6--7.2% across all modulations at SNR=18. Its sparse L1-regularized perturbations are maximally effective on the control plane while causing comparable data-plane corruption to other attacks.

3. **DeepFool is ineffective for AMC attacks**, retaining 87% mean AMC accuracy. It should not be used as a benchmark for evaluating AMC adversarial robustness. Its only notable effect is data-plane corruption on QAM64/PAM4, which is shared by all gradient-based attacks.

4. **The control-plane attack mechanism dominates end-to-end damage.** For 6 of 8 modulations (all except QAM64 and PAM4), the attack is purely control-plane: Oracle CRC remains >99% regardless of attack algorithm. The entire link failure comes from AMC misclassification causing wrong demodulator selection.

5. **PAM4 is the most data-plane vulnerable modulation**, losing 22--32pp Oracle CRC at SNR=18 across all attacks. Its 4-level amplitude constellation has the smallest decision margins relative to perturbation magnitude.

---

## Appendix A: Complete JSON Data Summary

### A.1 EADL1 Attack

| Mod | SNR | Clean+Oracle CRC | Clean+AMC CRC | Adv+Oracle CRC | Adv+AMC CRC | AMC Clean | AMC Adv | n |
|-----|:---:|:-----------------:|:-------------:|:--------------:|:-----------:|:---------:|:-------:|:---:|
| BPSK | 0 | 0.998 | 0.988 | 0.988 | 0.026 | 0.982 | 0.020 | 500 |
| BPSK | 18 | 1.000 | 0.994 | 1.000 | 0.022 | 0.986 | 0.012 | 500 |
| QPSK | 0 | 0.792 | 0.766 | 0.776 | 0.026 | 0.962 | 0.034 | 500 |
| QPSK | 18 | 1.000 | 0.988 | 1.000 | 0.018 | 0.990 | 0.010 | 500 |
| 8PSK | 0 | 0.068 | 0.064 | 0.058 | 0.008 | 0.944 | 0.042 | 500 |
| 8PSK | 18 | 1.000 | 0.980 | 1.000 | 0.034 | 0.980 | 0.030 | 500 |
| QAM16 | 0 | 0.006 | 0.008 | 0.006 | 0.002 | 0.912 | 0.076 | 500 |
| QAM16 | 18 | 1.000 | 0.942 | 0.996 | 0.072 | 0.954 | 0.066 | 500 |
| QAM64 | 0 | 0.002 | 0.002 | 0.002 | 0.002 | 0.934 | 0.040 | 500 |
| QAM64 | 18 | 0.956 | 0.906 | 0.776 | 0.054 | 0.946 | 0.064 | 500 |
| PAM4 | 0 | 0.226 | 0.226 | 0.148 | 0.010 | 0.988 | 0.004 | 500 |
| PAM4 | 18 | 1.000 | 0.988 | 0.678 | 0.014 | 0.992 | 0.004 | 500 |
| GFSK | 0 | 0.058 | 0.056 | 0.058 | 0.000 | 0.980 | 0.008 | 500 |
| GFSK | 18 | 0.994 | 0.994 | 0.994 | 0.016 | 1.000 | 0.016 | 500 |
| CPFSK | 0 | 0.068 | 0.068 | 0.068 | 0.004 | 1.000 | 0.008 | 500 |
| CPFSK | 18 | 1.000 | 1.000 | 1.000 | 0.006 | 1.000 | 0.004 | 500 |

### A.2 DeepFool Attack

| Mod | SNR | Clean+Oracle CRC | Clean+AMC CRC | Adv+Oracle CRC | Adv+AMC CRC | AMC Clean | AMC Adv | n |
|-----|:---:|:-----------------:|:-------------:|:--------------:|:-----------:|:---------:|:-------:|:---:|
| BPSK | 0 | 0.998 | 0.988 | 1.000 | 0.930 | 0.982 | 0.930 | 500 |
| BPSK | 18 | 1.000 | 0.994 | 1.000 | 0.878 | 0.986 | 0.874 | 500 |
| QPSK | 0 | 0.792 | 0.766 | 0.794 | 0.698 | 0.962 | 0.880 | 500 |
| QPSK | 18 | 1.000 | 0.988 | 1.000 | 0.966 | 0.990 | 0.964 | 500 |
| 8PSK | 0 | 0.068 | 0.064 | 0.056 | 0.048 | 0.944 | 0.896 | 500 |
| 8PSK | 18 | 1.000 | 0.980 | 0.996 | 0.844 | 0.980 | 0.846 | 500 |
| QAM16 | 0 | 0.006 | 0.008 | 0.006 | 0.004 | 0.912 | 0.850 | 500 |
| QAM16 | 18 | 1.000 | 0.942 | 1.000 | 0.904 | 0.954 | 0.904 | 500 |
| QAM64 | 0 | 0.002 | 0.002 | 0.002 | 0.000 | 0.934 | 0.822 | 500 |
| QAM64 | 18 | 0.956 | 0.906 | 0.716 | 0.620 | 0.946 | 0.856 | 500 |
| PAM4 | 0 | 0.226 | 0.226 | 0.140 | 0.120 | 0.988 | 0.894 | 500 |
| PAM4 | 18 | 1.000 | 0.988 | 0.774 | 0.708 | 0.992 | 0.916 | 500 |
| GFSK | 0 | 0.058 | 0.056 | 0.058 | 0.052 | 0.980 | 0.840 | 500 |
| GFSK | 18 | 0.994 | 0.994 | 0.994 | 0.694 | 1.000 | 0.700 | 500 |
| CPFSK | 0 | 0.068 | 0.068 | 0.068 | 0.064 | 1.000 | 0.950 | 500 |
| CPFSK | 18 | 1.000 | 1.000 | 1.000 | 0.868 | 1.000 | 0.866 | 500 |

---

## Appendix B: Reproducibility

### B.1 Commands

```bash
# EADL1 attack CRC experiment
python crc_experiment.py --attack eadl1 --ta_box minmax --n_bursts 500 \
  --output_dir results/crc_defense_fec/eadl1

# DeepFool attack CRC experiment
python crc_experiment.py --attack deepfool --ta_box minmax --n_bursts 500 \
  --output_dir results/crc_defense_fec/deepfool
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
| Bursts per cell | 500 |
| SNR levels | 0 dB, 18 dB |

### B.3 Data Sources

| Attack | JSON Path |
|--------|-----------|
| EADL1 | `results/crc_defense_fec/eadl1/crc_vs_amc.json` |
| DeepFool | `results/crc_defense_fec/deepfool/crc_vs_amc.json` |
| CW (prior) | `CRC_EXPERIMENT_REPORT.md` (n=200 per cell) |
