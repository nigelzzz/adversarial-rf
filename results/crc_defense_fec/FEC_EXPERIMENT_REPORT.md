<<<<<<< HEAD
# FEC-Enabled CRC Defense Experiment Report

## Overview

This experiment adds **Forward Error Correction (FEC)** to the CRC defense pipeline and evaluates its impact at **SNR = 0 dB** — a realistic low-SNR operating point where the original pipeline (without FEC) showed near-total CRC failure for higher-order modulations.

### FEC Implementation

- **Code**: Rate-1/2 convolutional code, constraint length K=7, generators [171, 133] octal
- **Decoder**: 64-state Viterbi with soft-decision (max-log-MAP LLR) decoding
- **Interleaver**: 14-column block interleaver (spreads burst errors across ~2x constraint length)
- **Expected coding gain**: ~5-7 dB
- reference:
  - MIT convolutional code lecture (https://ocw.mit.edu/courses/6-02-introduction-to-eecs-ii-digital-communication-systems-fall-2012/fea2b27744e25ff21846374a56ceb256_MIT6_02F12_lec06.pdf)
  - Viterbi decoding tutorial (https://web.mit.edu/6.02/www/f2011/handouts/8.pdf)
  - Princeton Viterbi algorithm lecture notes (https://www.cs.princeton.edu/courses/archive/spring18/cos463/lectures/L09-viterbi.pdf)
### Pipeline

```
TX:  data -> CRC-8 -> conv_encode -> block_interleave -> modulate -> pulse_shape -> channel(SNR)
RX:  matched_filter -> pilot_recovery -> soft_LLR -> block_deinterleave -> viterbi_decode -> CRC_check
```

### Bit Budget

| Mod    | bps | Coded bits (14 syms) | Decoded payload | Data + CRC   | FEC? |
|--------|-----|----------------------|-----------------|--------------|------|
| BPSK   | 1   | 14                   | 1               | impossible   | Skip |
| QPSK   | 2   | 28                   | 8               | 0 + 8 CRC   | Yes  |
| 8PSK   | 3   | 42                   | 15              | 7 + 8 CRC   | Yes  |
| QAM16  | 4   | 56                   | 22              | 14 + 8 CRC  | Yes  |
| QAM64  | 6   | 84                   | 36              | 28 + 8 CRC  | Yes  |
| PAM4   | 2   | 28                   | 8               | 0 + 8 CRC   | Yes  |

BPSK/CPFSK/GFSK (1 bps) cannot support rate-1/2 FEC with CRC in a 14-symbol burst. FEC is automatically disabled for these modulations.

---

## Results at SNR = 0 dB

### Oracle Demodulation (true modulation type known)

| Mod    | Scenario     | noFEC   | FEC      | Gain      |
|--------|-------------|---------|----------|-----------|
| BPSK   | Clean       | 100.0%  | —        | —         |
| QPSK   | Clean       | 83.5%   | **100.0%** | +16.5 pp |
| 8PSK   | Clean       | 6.5%    | **99.5%**  | +93.0 pp |
| QAM16  | Clean       | 0.0%    | **89.5%**  | +89.5 pp |
| QAM64  | Clean       | 0.0%    | **21.0%**  | +21.0 pp |
| PAM4   | Clean       | 22.0%   | **80.5%**  | +58.5 pp |
| | | | | |
| BPSK   | CW attack   | 100.0%  | —        | —         |
| QPSK   | CW attack   | 83.5%   | **100.0%** | +16.5 pp |
| 8PSK   | CW attack   | 6.5%    | **99.5%**  | +93.0 pp |
| QAM16  | CW attack   | 0.0%    | **89.0%**  | +89.0 pp |
| QAM64  | CW attack   | 0.0%    | **21.5%**  | +21.5 pp |
| PAM4   | CW attack   | 18.5%   | **80.0%**  | +61.5 pp |

**Key observation**: CW attack does not degrade CRC when the correct demodulator is used (oracle). This confirms the attack is **control-plane only** — it fools the classifier but does not corrupt the signal waveform.

### CW Attack + FFT Top-K Recovery + Oracle Demod

| Mod    | Top-K | noFEC   | FEC      | Gain      |
|--------|-------|---------|----------|-----------|
| QPSK   | 10    | 18.5%   | **92.0%**  | +73.5 pp |
| QPSK   | 20    | 49.0%   | **98.0%**  | +49.0 pp |
| QPSK   | 50    | 73.0%   | **100.0%** | +27.0 pp |
| 8PSK   | 10    | 0.0%    | **87.0%**  | +87.0 pp |
| 8PSK   | 20    | 2.5%    | **95.0%**  | +92.5 pp |
| 8PSK   | 50    | 5.0%    | **99.5%**  | +94.5 pp |
| QAM16  | 10    | 0.0%    | **45.0%**  | +45.0 pp |
| QAM16  | 20    | 0.0%    | **79.5%**  | +79.5 pp |
| QAM16  | 50    | 0.0%    | **87.5%**  | +87.5 pp |
| QAM64  | 10    | 0.0%    | 0.5%     | +0.5 pp  |
| QAM64  | 20    | 0.0%    | 8.5%     | +8.5 pp  |
| QAM64  | 50    | 0.5%    | 16.0%    | +15.5 pp |
| PAM4   | 10    | 1.0%    | **67.5%**  | +66.5 pp |
| PAM4   | 20    | 13.0%   | **77.5%**  | +64.5 pp |
| PAM4   | 50    | 18.5%   | **79.5%**  | +61.0 pp |

### AMC Demodulation (classifier selects demodulator)

| Mod    | Scenario         | noFEC   | FEC      |
|--------|-----------------|---------|----------|
| QPSK   | Clean+AMC       | 40.5%   | 3.5%     |
| 8PSK   | Clean+AMC       | 2.0%    | **39.5%**  |
| QAM16  | Clean+AMC       | 0.5%    | **54.0%**  |
| QAM64  | Clean+AMC       | 0.0%    | 4.5%     |
| PAM4   | Clean+AMC       | 18.5%   | **40.0%**  |
| | | | |
| QPSK   | CW+Top50+AMC   | 14.5%   | 3.5%     |
| 8PSK   | CW+Top50+AMC   | 2.0%    | **33.0%**  |
| QAM16  | CW+Top50+AMC   | 0.0%    | **52.0%**  |
| QAM64  | CW+Top50+AMC   | 0.0%    | 4.0%     |
| PAM4   | CW+Top50+AMC   | 15.0%   | **37.0%**  |

**Note on QPSK/PAM4 AMC degradation**: FEC-encoded QPSK/PAM4 signals have different statistical properties than standard QPSK/PAM4 (coded symbols look more "random"), causing the AWN classifier to misidentify them. This reduces AMC accuracy from ~46% (noFEC) to ~3.5% (FEC) for QPSK. This is an artifact of the classifier being trained on uncoded signals — a classifier retrained on FEC-coded signals would not have this issue.

---

## Why QAM64 Shows Limited Improvement

QAM64 with rate-1/2 FEC achieves only 21% CRC at SNR=0 dB, compared to 99.5% for 8PSK. This is a **fundamental information-theoretic limitation**, not a code deficiency.

The Shannon capacity limit for rate-1/2 coded modulations:

| Mod    | Spectral Efficiency | Shannon Limit | Gap at SNR=0 | Result       |
|--------|-------------------|---------------|--------------|--------------|
| QPSK   | 1.0 bit/s/Hz      | ~0 dB         | 0 dB         | 100% CRC     |
| 8PSK   | 1.5 bit/s/Hz      | ~2.5 dB       | 2.5 dB       | 99.5% CRC    |
| QAM16  | 2.0 bit/s/Hz      | ~4.8 dB       | 4.8 dB       | 89.5% CRC    |
| QAM64  | 3.0 bit/s/Hz      | **~8.5 dB**   | **8.5 dB**   | 21% CRC      |
| PAM4   | 1.0 bit/s/Hz      | ~0 dB         | 0 dB         | 80.5% CRC    |

QAM64 at SNR=0 dB is **8.5 dB below the Shannon limit** — no FEC code of any rate can reliably deliver 3 bits/symbol/Hz at this SNR. Intermediate SNR experiments confirm FEC works well for QAM64 when above the threshold:

| QAM64 SNR | noFEC Clean | FEC Clean | FEC Gain |
|-----------|-------------|-----------|----------|
| 0 dB      | 0.0%        | 21.0%     | +21 pp   |
| 3 dB      | 0.5%        | 63.0%     | +62.5 pp |
| 6 dB      | 0.5%        | 87.0%     | +86.5 pp |
| 10 dB     | 9.0%        | 98.5%     | +89.5 pp |
| 18 dB     | 94.0%       | 100.0%    | +6 pp    |

In real systems, this is exactly why **adaptive modulation and coding (AMC)** exists: the transmitter drops from QAM64 to QPSK at low SNR.

---

## Comparison: SNR = 18 dB vs SNR = 0 dB

### Oracle Demod, Clean Signal

| Mod    | SNR=18 noFEC | SNR=18 FEC | SNR=0 noFEC | SNR=0 FEC |
|--------|-------------|------------|-------------|-----------|
| BPSK   | 100.0%      | —          | 100.0%      | —         |
| QPSK   | 100.0%      | 100.0%     | 83.5%       | **100.0%**  |
| 8PSK   | 100.0%      | 100.0%     | 6.5%        | **99.5%**   |
| QAM16  | 100.0%      | 100.0%     | 0.0%        | **89.5%**   |
| QAM64  | 94.0%       | **100.0%** | 0.0%        | 21.0%     |
| PAM4   | 100.0%      | 100.0%     | 22.0%       | **80.5%**   |

At SNR=18 dB, FEC primarily helps QAM64 (94% -> 100%). At SNR=0 dB, FEC provides massive gains for all modulations except QAM64 (limited by Shannon bound).

### CW + Top-K Recovery, Oracle Demod

| Mod    | SNR=18 noFEC Top10 | SNR=18 FEC Top10 | SNR=0 noFEC Top10 | SNR=0 FEC Top10 |
|--------|-------------------|-----------------|-------------------|-----------------|
| QPSK   | 80.0%             | **100.0%**      | 18.5%             | **92.0%**       |
| 8PSK   | 7.5%              | **99.5%**       | 0.0%              | **87.0%**       |
| QAM16  | 0.0%              | **90.5%**       | 0.0%              | **45.0%**       |
| QAM64  | 1.5%              | 23.0%           | 0.0%              | 0.5%            |
| PAM4   | 7.0%              | **85.0%**       | 1.0%              | **67.5%**       |

FEC dramatically improves Top-K recovery at both SNRs. The combination of FEC + aggressive Top-K (K=10) at SNR=0 dB achieves 87-92% CRC for QPSK/8PSK — previously impossible without FEC.

---

## Key Findings

1. **FEC provides 15-94 percentage point CRC improvement at SNR=0 dB** for modulations operating near or above their Shannon limit (QPSK, 8PSK, QAM16, PAM4).

2. **CW attack remains control-plane only**: FEC CRC pass rates under CW attack (oracle demod) are identical to clean — the adversarial perturbation does not corrupt the data plane, even at low SNR with FEC.

3. **FEC + Top-K synergy**: FEC allows aggressive spectral filtering (low K) without destroying the signal, since the Viterbi decoder corrects residual errors from the filtering. At SNR=0, 8PSK with FEC + Top-10 achieves 87% CRC vs 0% without FEC.

4. **QAM64 is fundamentally limited at SNR=0**: The 8.5 dB gap below Shannon capacity cannot be bridged by any FEC code. This motivates adaptive modulation as a complementary defense strategy.

5. **AMC classifier degradation with FEC**: FEC-coded signals have different statistical properties that confuse classifiers trained on uncoded signals. This is solvable by retraining on coded signals, but highlights that FEC changes the signal characteristics visible to ML classifiers.

---

## Experimental Setup

- **Dataset**: Synthetic bursts (16 symbols, 2 pilots, 8 sps, RRC beta=0.35)
- **Bursts per condition**: 200
- **Attack**: CW (Carlini-Wagner L2), c=1.0, steps=1000, minmax normalization
- **Defense**: FFT Top-K (K=10, 20, 50)
- **Classifier**: AWN (Adaptive Wavelet Network) pretrained on RML2016.10a
- **FEC**: Rate-1/2 K=7 convolutional code + soft Viterbi + 14-column block interleaver
- **GPU**: NVIDIA GeForce GTX 1070 Ti (CUDA 12.2)
- **Output**: `results/crc_defense_fec/`

### Reproduction

```bash
source venv/bin/activate

# Main experiment: SNR=18 and SNR=0, all mods, FEC vs noFEC
python3 crc_defense_pipeline_fec.py --device cuda --snr 18,0 --n_bursts 200 \
  --topk 10,20,50 --ckpt /path/to/checkpoint/dir

# QAM64 intermediate SNR analysis
python3 crc_defense_pipeline_fec.py --device cuda --snr 10,6,3 --n_bursts 200 \
  --topk 20,50 --mods QAM64 --ckpt /path/to/checkpoint/dir
```

=======
# Spectral Defense Evaluation: IFFT Top-K Filtering Against CW and EADEN Adversarial Attacks on Automatic Modulation Classification

---

## Abstract

We evaluate IFFT Top-K spectral filtering as a defense against Carlini-Wagner (CW) and Elastic-net (EADEN) adversarial attacks on a wavelet-based automatic modulation classification (AMC) model operating on the RML2016.10a dataset. Our experiments across seven modulation types and nine SNR levels reveal a fundamental limitation of fixed-K spectral defenses: **no single K value provides effective recovery across all modulations**. QAM64 achieves greater than 90% classification recovery with Top-10 (approximately 10% of FFT bins), while QAM16 recovers only approximately 47% with Top-20 and 8PSK recovers as little as 3.5% at low SNR. Simple modulations (AM-DSB, PAM4, GFSK) recover well with Top-20 (85--94%). These results expose a chicken-and-egg problem for spectral defenses in cognitive radio: selecting the optimal K requires knowledge of the modulation type, but modulation classification is precisely the capability the adversary has compromised. This finding challenges the viability of SigGuard-style fixed-K spectral defenses and motivates research into adaptive, modulation-aware defense strategies.

---

## 1. Introduction

Automatic Modulation Classification (AMC) is a critical control-plane function in cognitive radio and software-defined radio (SDR) systems. AMC enables receivers to identify the modulation scheme of incoming signals and select the appropriate demodulator, forming a prerequisite for reliable communication in dynamic spectrum environments. Prior work has established that deep learning-based AMC classifiers are vulnerable to adversarial perturbations [Sadeghi & Larsson 2019, Flowers et al. 2019], which can cause the receiver to select an incorrect demodulator and break the data link entirely.

A natural defense strategy is spectral filtering: since adversarial perturbations often introduce energy outside the signal's natural spectral footprint, retaining only the top-K FFT components should strip away attack energy while preserving the underlying signal. This approach, exemplified by SigGuard-style defenses, applies a fixed K value uniformly across all input signals regardless of modulation type.

Our prior experiments (CRC Defense Report, SNR=18 dB) demonstrated that CW attacks are purely control-plane: the perturbation fools the AMC classifier but does not corrupt the underlying data when the correct demodulator is used. We also observed that FFT Top-K defense effectiveness is strongly modulation-dependent. In this report, we extend those findings with a systematic evaluation across:

- **Two attack algorithms**: CW (L2-optimized) and EADEN (elastic-net, L1+L2 regularized)
- **Seven modulation types**: QAM64, QAM16, 8PSK, QPSK, AM-DSB, GFSK, PAM4
- **Nine SNR levels**: 0, 2, 4, 8, 10, 12, 14, 16, 18 dB
- **Multiple K values**: Top-10 and Top-20 IFFT filtering

The central finding is that modulation spectral complexity determines defense effectiveness, and no universal K exists. This creates a fundamental tension: aggressive filtering (low K) removes perturbation but destroys complex signals; conservative filtering (high K) preserves signal fidelity but retains perturbation energy.

---

## 2. Threat Model

### 2.1 Adversary Capabilities

We consider a white-box adversary with full knowledge of the target AMC classifier (architecture, weights, preprocessing). The adversary can compute gradient-based perturbations on intercepted or predicted IQ samples. The attack is applied at the signal level before the receiver processes the waveform.

**Adversary goal**: Cause the AMC classifier to misclassify the modulation type, leading the receiver to select an incorrect demodulator and break the data link. The adversary does not aim to corrupt the underlying data directly.

**Perturbation constraints**: Both attacks minimize perturbation magnitude subject to achieving misclassification:
- **CW**: Minimizes L2 norm of the perturbation
- **EADEN**: Minimizes a weighted combination of L1 and L2 norms (elastic-net regularization), producing sparser perturbations

### 2.2 Defender Capabilities

The defender applies IFFT Top-K spectral filtering as a preprocessing step before classification. The defense:
1. Computes the real FFT of each I/Q channel (128-point signal yields 65 frequency bins per channel)
2. Retains only the K largest-magnitude frequency bins
3. Zeros all remaining bins
4. Applies the inverse FFT to reconstruct a filtered time-domain signal
5. Passes the filtered signal to the AMC classifier

The defender does not know whether a given signal is adversarial. The defense is applied uniformly to all incoming signals.

### 2.3 System Context

The target system is a cognitive radio receiver using an AWN (Adaptive Wavelet Network) classifier trained on the RML2016.10a dataset with 11 modulation classes. The receiver uses AMC output to select from a bank of demodulators. Incorrect classification causes the receiver to apply the wrong demodulator, resulting in near-random bit errors and link failure.

---

## 3. Experimental Setup

### 3.1 Dataset and Model

| Parameter | Value |
|-----------|-------|
| Dataset | RML2016.10a |
| Signal length | 128 IQ samples (2 x 128) |
| Number of classes | 11 modulations |
| Model | AWN (Adaptive Wavelet Network) |
| Checkpoint | Pretrained on RML2016.10a |
| FFT bins per channel | 65 (real FFT of 128 samples) |

### 3.2 Attack Configuration

| Parameter | CW | EADEN |
|-----------|-----|-------|
| Library | torchattacks | torchattacks |
| Box mode | minmax | minmax |
| Norm | L2 | L1 + L2 (elastic-net) |
| Steps | 100 | default |
| Learning rate | 0.001 | default |
| Kappa (confidence) | 1 | default |
| Epsilon | N/A (L2-optimized) | default |

### 3.3 Defense Configuration

| Parameter | Value |
|-----------|-------|
| Defense type | IFFT Top-K spectral filtering |
| Top-K values tested | 10 bins, 20 bins |
| Top-10 as percentage | 10/65 = 15.4% of frequency bins per channel |
| Top-20 as percentage | 20/65 = 30.8% of frequency bins per channel |
| Application | Per-channel (I and Q independently) |

### 3.4 Evaluation Protocol

- **Samples**: 200 samples per (modulation, SNR) cell from the RML2016.10a test set
- **SNR range**: 0, 2, 4, 8, 10, 12, 14, 16, 18 dB
- **Modulations evaluated**: QAM64, QAM16, 8PSK, QPSK, AM-DSB, GFSK, PAM4
- **Metrics**: Classification accuracy (fraction of correctly classified samples)

---

## 4. Results: CW Attack

### 4.1 QAM64 -- Top-10 Achieves Near-Perfect Recovery

QAM64 is a high-order digital modulation (64 constellation points, 6 bits/symbol) with broad spectral content. Despite its complexity, Top-10 filtering provides excellent recovery.

| SNR (dB) | Clean Acc | CW Acc | Top-10 Acc | Recovery Rate |
|----------|-----------|--------|------------|---------------|
| 0 | 0.9300 | 0.0000 | 0.9100 | 97.8% |
| 2 | 0.9350 | 0.0000 | 0.9450 | 101.1% |
| 4 | 0.9450 | 0.0000 | 0.9050 | 95.8% |
| 8 | 0.9350 | 0.0000 | 0.9200 | 98.4% |
| 10 | 0.9550 | 0.0000 | 0.9150 | 95.8% |
| 12 | 0.9250 | 0.0050 | 0.9400 | 101.6% |
| 14 | 0.9500 | 0.0050 | 0.9500 | 100.0% |
| 16 | 0.9250 | 0.0050 | 0.9150 | 98.9% |
| 18 | 0.9550 | 0.0050 | 0.9050 | 94.8% |
| **Mean** | **0.9394** | **0.0022** | **0.9228** | **98.2%** |

*Recovery Rate = Top-K Acc / Clean Acc x 100%*

CW reduces QAM64 accuracy to effectively 0% across all SNR levels. Top-10 IFFT filtering recovers 92--95% accuracy, achieving 94.8--101.6% of clean performance. The defense is remarkably consistent across the entire SNR range.

### 4.2 QAM16 -- Top-20 Recovers Only Half

QAM16 is a moderate-order modulation (16 constellation points, 4 bits/symbol). Despite using twice as many retained bins (Top-20 vs Top-10 for QAM64), recovery is substantially worse.

| SNR (dB) | Clean Acc | CW Acc | Top-20 Acc | Recovery Rate |
|----------|-----------|--------|------------|---------------|
| 0 | 0.9050 | 0.0000 | 0.3900 | 43.1% |
| 2 | 0.9800 | 0.0000 | 0.4450 | 45.4% |
| 4 | 0.9200 | 0.0000 | 0.5100 | 55.4% |
| 8 | 0.9250 | 0.0000 | 0.4650 | 50.3% |
| 10 | 0.9500 | 0.0000 | 0.4850 | 51.1% |
| 12 | 0.9450 | 0.0000 | 0.5700 | 60.3% |
| 14 | 0.9550 | 0.0000 | 0.5500 | 57.6% |
| 16 | 0.9650 | 0.0000 | 0.5150 | 53.4% |
| 18 | 0.9650 | 0.0000 | 0.5700 | 59.1% |
| **Mean** | **0.9456** | **0.0000** | **0.5000** | **52.8%** |

CW achieves 0.00% attack accuracy for all SNR levels on QAM16. Top-20 filtering recovers only approximately 50% of clean accuracy on average -- barely better than random guessing among the 11 RML2016.10a classes (baseline 9.1%). Recovery shows modest improvement with SNR but plateaus around 57%.

### 4.3 8PSK -- Top-20 Recovery Is SNR-Dependent and Insufficient

8PSK (8 Phase-Shift Keying, 3 bits/symbol) occupies a spectral profile intermediate between QAM16 and QPSK. Top-20 recovery is poor at low SNR and only moderately effective at high SNR.

| SNR (dB) | Clean Acc | CW Acc | Top-20 Acc | Recovery Rate |
|----------|-----------|--------|------------|---------------|
| 0 | 0.9250 | 0.0050 | 0.0300 | 3.2% |
| 2 | 0.9900 | 0.0000 | 0.0450 | 4.5% |
| 4 | 0.9800 | 0.0050 | 0.1300 | 13.3% |
| 8 | 0.9900 | 0.0100 | 0.3050 | 30.8% |
| 10 | 0.9850 | 0.0100 | 0.3050 | 31.0% |
| 12 | 0.9950 | 0.0200 | 0.5700 | 57.3% |
| 14 | 0.9900 | 0.0300 | 0.5350 | 54.0% |
| 16 | 0.9850 | 0.0600 | 0.5750 | 58.4% |
| 18 | 0.9750 | 0.0500 | 0.5350 | 54.9% |
| **Mean** | **0.9794** | **0.0211** | **0.3367** | **34.2%** |

Two notable patterns emerge. First, CW attack is slightly less effective against 8PSK at high SNR (5--6% residual accuracy at SNR 16--18) compared to its near-0% effectiveness on QAM modulations. Second, Top-20 recovery improves dramatically with SNR: from 3.2% at SNR 0 to approximately 55% at SNR 14+. However, even at high SNR, recovery remains below 60%.

### 4.4 AM-DSB, GFSK, PAM4 -- Simple Modulations Recover Well

These three modulation types share simpler spectral characteristics and demonstrate strong Top-20 recovery.

**AM-DSB (Amplitude Modulation, Double Sideband)**:

| SNR (dB) | Clean Acc | CW Acc | Top-20 Acc | Recovery Rate |
|----------|-----------|--------|------------|---------------|
| 0 | 0.8950 | 0.1200 | 0.7750 | 86.6% |
| 2 | 0.9600 | 0.2100 | 0.8550 | 89.1% |
| 4 | 0.9900 | 0.1850 | 0.8450 | 85.4% |
| 8 | 1.0000 | 0.5200 | 0.9150 | 91.5% |
| 10 | 1.0000 | 0.8300 | 0.9700 | 97.0% |
| 12 | 1.0000 | 0.9850 | 0.9950 | 99.5% |
| 14 | 1.0000 | 1.0000 | 1.0000 | 100.0% |
| 16 | 1.0000 | 1.0000 | 1.0000 | 100.0% |
| 18 | 1.0000 | 1.0000 | 1.0000 | 100.0% |
| **Mean** | **0.9828** | **0.6500** | **0.9283** | **94.3%** |

AM-DSB is a notable outlier: CW attack is only partially effective (65% average accuracy under attack), and the attack becomes entirely ineffective at SNR >= 14 dB. Top-20 defense achieves near-complete recovery (94.3% of clean). This is because AM-DSB has a distinctive spectral signature dominated by the carrier and sidebands, which the Top-20 filter preserves.

**GFSK (Gaussian Frequency-Shift Keying)**:

| SNR (dB) | Clean Acc | CW Acc | Top-20 Acc | Recovery Rate |
|----------|-----------|--------|------------|---------------|
| 0 | 0.9900 | 0.4600 | 0.8400 | 84.8% |
| 2 | 0.9800 | 0.1500 | 0.7800 | 79.6% |
| 4 | 1.0000 | 0.0300 | 0.8550 | 85.5% |
| 8 | 1.0000 | 0.1500 | 0.9150 | 91.5% |
| 10 | 1.0000 | 0.1750 | 0.8650 | 86.5% |
| 12 | 1.0000 | 0.1150 | 0.8300 | 83.0% |
| 14 | 1.0000 | 0.2450 | 0.8750 | 87.5% |
| 16 | 1.0000 | 0.2450 | 0.8750 | 87.5% |
| 18 | 1.0000 | 0.2550 | 0.9050 | 90.5% |
| **Mean** | **0.9967** | **0.2022** | **0.8600** | **86.3%** |

GFSK is a constant-envelope FSK modulation. CW achieves moderate effectiveness (20% average accuracy under attack) with irregular SNR dependence. Top-20 defense recovers to 86% of clean performance on average.

**PAM4 (4-level Pulse Amplitude Modulation)**:

| SNR (dB) | Clean Acc | CW Acc | Top-20 Acc | Recovery Rate |
|----------|-----------|--------|------------|---------------|
| 0 | 0.9900 | 0.5500 | 0.9400 | 94.9% |
| 2 | 0.9950 | 0.7150 | 0.9600 | 96.5% |
| 4 | 0.9950 | 0.7100 | 0.9550 | 96.0% |
| 8 | 0.9950 | 0.8100 | 0.9900 | 99.5% |
| 10 | 1.0000 | 0.8500 | 0.9850 | 98.5% |
| 12 | 0.9900 | 0.7900 | 0.9800 | 99.0% |
| 14 | 0.9850 | 0.7850 | 0.9800 | 99.5% |
| 16 | 0.9850 | 0.8200 | 0.9700 | 98.5% |
| 18 | 1.0000 | 0.8550 | 1.0000 | 100.0% |
| **Mean** | **0.9928** | **0.7650** | **0.9733** | **98.0%** |

PAM4 shows the highest recovery rate among all modulations tested. CW attack has limited effectiveness (76.5% average accuracy under attack -- the attack barely works), and Top-20 defense achieves 98% of clean performance. This is consistent with PAM4's simple spectral structure.

### 4.5 QPSK -- Moderate Recovery, Strongly SNR-Dependent

| SNR (dB) | Clean Acc | CW Acc | Top-20 Acc | Recovery Rate |
|----------|-----------|--------|------------|---------------|
| 0 | 0.9750 | 0.0050 | 0.1650 | 16.9% |
| 2 | 0.9900 | 0.0400 | 0.4950 | 50.0% |
| 4 | 0.9900 | 0.2500 | 0.7900 | 79.8% |
| 8 | 0.9950 | 0.4450 | 0.8850 | 88.9% |
| 10 | 0.9950 | 0.6050 | 0.9200 | 92.5% |
| 12 | 0.9900 | 0.6150 | 0.9350 | 94.4% |
| 14 | 0.9650 | 0.5550 | 0.8900 | 92.2% |
| 16 | 0.9850 | 0.5450 | 0.9250 | 93.9% |
| 18 | 0.9850 | 0.6100 | 0.9450 | 95.9% |
| **Mean** | **0.9856** | **0.4078** | **0.7722** | **78.3%** |

QPSK presents a distinctive pattern. CW attack effectiveness is irregular: relatively strong at low SNR (0.5% at SNR 0) but degrading to only 55--61% at high SNR. This means CW only partially works against QPSK at high SNR. Top-20 recovery is strongly SNR-dependent: from 16.9% at SNR 0 to 95.9% at SNR 18. Above SNR 8, QPSK recovery exceeds 88%, making the defense viable in typical operating conditions.

### 4.6 CW Attack Summary Table

Mean accuracy across all SNR levels (0--18 dB):

| Modulation | Clean Acc | CW Attack Acc | Defense Acc | Defense K | Recovery Rate | Category |
|------------|-----------|---------------|-------------|-----------|---------------|----------|
| QAM64 | 93.94% | 0.22% | 92.28% | Top-10 | **98.2%** | Excellent |
| PAM4 | 99.28% | 76.50% | 97.33% | Top-20 | **98.0%** | Excellent |
| AM-DSB | 98.28% | 65.00% | 92.83% | Top-20 | **94.3%** | Excellent |
| GFSK | 99.67% | 20.22% | 86.00% | Top-20 | **86.3%** | Good |
| QPSK | 98.56% | 40.78% | 77.22% | Top-20 | **78.3%** | Moderate |
| QAM16 | 94.56% | 0.00% | 50.00% | Top-20 | **52.8%** | Poor |
| 8PSK | 97.94% | 2.11% | 33.67% | Top-20 | **34.2%** | Failed |

---

## 5. Results: EADEN Attack

EADEN (Elastic-net Attack with Decision-based targeting, Elastic Net regularization) combines L1 and L2 penalties, producing sparser perturbations than CW. We evaluate whether this different perturbation geometry affects defense recovery.

### 5.1 QAM64 -- EADEN Top-10 Recovery Matches CW

| SNR (dB) | Clean Acc | EADEN Acc | Top-10 Acc | Recovery Rate |
|----------|-----------|-----------|------------|---------------|
| 0 | 0.9300 | 0.0000 | 0.9300 | 100.0% |
| 2 | 0.9350 | 0.0000 | 0.9500 | 101.6% |
| 4 | 0.9450 | 0.0000 | 0.9400 | 99.5% |
| 6 | 0.9350* | 0.0000 | 0.9300 | 99.5% |
| 8 | 0.9350 | 0.0000 | 0.9150 | 97.9% |
| 10 | 0.9550 | 0.0000 | 0.9050 | 94.8% |
| 12 | 0.9250 | 0.0000 | 0.9550 | 103.2% |
| 14 | 0.9500 | 0.0000 | 0.9150 | 96.3% |
| 16 | 0.9250 | 0.0000 | 0.8750 | 94.6% |
| 18 | 0.9550 | 0.0000 | 0.9200 | 96.3% |
| **Mean** | **0.9390** | **0.0000** | **0.9235** | **98.4%** |

*SNR 6 clean accuracy estimated from adjacent values.

EADEN achieves a perfect 0.00% attack accuracy across all SNR levels for QAM64 (CW left 0.22% residual). Despite this marginally stronger attack, Top-10 defense recovers 98.4% of clean accuracy -- essentially identical to the CW recovery rate. This indicates that the defense effectiveness is determined by the signal's spectral structure, not the specific attack algorithm.

### 5.2 QAM16 -- Top-10 Is Catastrophic, Top-20 Remains Insufficient

**QAM16 with Top-10:**

| SNR (dB) | EADEN Acc | Top-10 Acc |
|----------|-----------|------------|
| 0 | 0.0000 | 0.0950 |
| 2 | 0.0000 | 0.0600 |
| 4 | 0.0000 | 0.1000 |
| 6 | 0.0000 | 0.1100 |
| 8 | 0.0000 | 0.1050 |
| 10 | 0.0000 | 0.0850 |
| 12 | 0.0000 | 0.0350 |
| 14 | 0.0000 | 0.0700 |
| 16 | 0.0000 | 0.1050 |
| 18 | 0.0000 | 0.1050 |
| **Mean** | **0.0000** | **0.0870** |

Top-10 on QAM16 produces only 8.7% average accuracy -- essentially random classification among 11 classes (baseline 9.1%). The defense destroys the signal rather than recovering it.

**QAM16 with Top-20:**

| SNR (dB) | EADEN Acc | Top-20 Acc |
|----------|-----------|------------|
| 0 | 0.0000 | 0.3650 |
| 2 | 0.0000 | 0.4000 |
| 4 | 0.0000 | 0.5000 |
| 6 | 0.0000 | 0.4900 |
| 8 | 0.0000 | 0.4650 |
| 10 | 0.0000 | 0.4800 |
| 12 | 0.0000 | 0.4600 |
| 14 | 0.0000 | 0.4600 |
| 16 | 0.0000 | 0.4650 |
| 18 | 0.0000 | 0.4850 |
| **Mean** | **0.0000** | **0.4570** |

Top-20 improves QAM16 to 45.7% average accuracy -- a significant improvement over Top-10 but still well below the 94.6% clean accuracy. Notably, unlike 8PSK or QPSK, QAM16 recovery does not improve substantially with SNR, suggesting a structural limitation rather than a noise-limited one.

### 5.3 8PSK -- EADEN Top-20 Recovery Is SNR-Dependent

| SNR (dB) | EADEN Acc | Top-20 Acc | Recovery Rate |
|----------|-----------|------------|---------------|
| 0 | 0.0000 | 0.0350 | 3.8% |
| 2 | 0.0000 | 0.0700 | 7.1% |
| 4 | 0.0000 | 0.1300 | 13.3% |
| 6 | 0.0000 | 0.2750 | 28.1%* |
| 8 | 0.0000 | 0.3700 | 37.4% |
| 10 | 0.0000 | 0.4450 | 45.2% |
| 12 | 0.0000 | 0.4400 | 44.2% |
| 14 | 0.0000 | 0.5400 | 54.5% |
| 16 | 0.0000 | 0.5250 | 53.3% |
| 18 | 0.0000 | 0.4700 | 48.2% |
| **Mean** | **0.0000** | **0.3300** | **33.5%** |

*Recovery rate computed against clean accuracy from CW dataset (0.98 estimated for SNR 6).

8PSK under EADEN mirrors the CW pattern: strong SNR dependence, with recovery climbing from 3.8% at SNR 0 to a peak of approximately 54% at SNR 14. The defense is effectively useless below SNR 4 and marginally useful above SNR 10.

### 5.4 AM-DSB, GFSK, PAM4 -- EADEN Top-20 Recovery Remains Strong

**AM-DSB with EADEN + Top-20:**

| SNR (dB) | EADEN Acc | Top-20 Acc | Recovery Rate |
|----------|-----------|------------|---------------|
| 0 | 0.0000 | 0.8450 | 94.4% |
| 2 | 0.0000 | 0.7450 | 77.6% |
| 4 | 0.0000 | 0.8600 | 86.9% |
| 6 | 0.0000 | 0.9350 | 93.5%* |
| 8 | 0.0000 | 0.8950 | 89.5% |
| 10 | 0.0000 | 0.9450 | 94.5% |
| 12 | 0.0000 | 0.9150 | 91.5% |
| 14 | 0.0000 | 0.9700 | 97.0% |
| 16 | 0.0050 | 0.9650 | 96.5% |
| 18 | 0.0050 | 0.9450 | 94.5% |
| **Mean** | **0.0010** | **0.9020** | **91.6%** |

*Clean accuracy estimated from CW data.

Notably, EADEN is far more effective against AM-DSB than CW: EADEN achieves 0.1% average attack accuracy versus CW's 65.0%. Despite this dramatically stronger attack, Top-20 recovery remains robust at 90.2% average.

**GFSK with EADEN + Top-20:**

| SNR (dB) | EADEN Acc | Top-20 Acc | Recovery Rate |
|----------|-----------|------------|---------------|
| 0 | 0.0000 | 0.7600 | 76.8% |
| 2 | 0.0000 | 0.8000 | 81.6% |
| 4 | 0.0000 | 0.8750 | 87.5% |
| 6 | 0.0000 | 0.9200 | 92.0%* |
| 8 | 0.0000 | 0.9250 | 92.5% |
| 10 | 0.0000 | 0.9000 | 90.0% |
| 12 | 0.0000 | 0.9350 | 93.5% |
| 14 | 0.0000 | 0.9150 | 91.5% |
| 16 | 0.0000 | 0.9050 | 90.5% |
| 18 | 0.0000 | 0.8700 | 87.0% |
| **Mean** | **0.0000** | **0.8805** | **88.3%** |

*Clean accuracy estimated.

**PAM4 with EADEN + Top-20:**

| SNR (dB) | EADEN Acc | Top-20 Acc | Recovery Rate |
|----------|-----------|------------|---------------|
| 0 | 0.0000 | 0.8250 | 83.3% |
| 2 | 0.0000 | 0.8950 | 89.9% |
| 4 | 0.0000 | 0.9350 | 94.0% |
| 6 | 0.0000 | 0.9650 | 97.0%* |
| 8 | 0.0000 | 0.9600 | 96.5% |
| 10 | 0.0000 | 0.9550 | 95.5% |
| 12 | 0.0000 | 0.9250 | 93.4% |
| 14 | 0.0000 | 0.9700 | 98.5% |
| 16 | 0.0000 | 0.9450 | 95.9% |
| 18 | 0.0000 | 0.9600 | 96.0% |
| **Mean** | **0.0000** | **0.9335** | **94.0%** |

*Clean accuracy estimated.

All three simple modulations maintain strong recovery under EADEN. PAM4 leads at 94.0%, followed by AM-DSB at 91.6% and GFSK at 88.3%.

### 5.5 EADEN Attack Summary Table

Mean accuracy across all SNR levels:

| Modulation | Clean Acc | EADEN Acc | Defense Acc | Defense K | Recovery Rate | Category |
|------------|-----------|-----------|-------------|-----------|---------------|----------|
| QAM64 | 93.90% | 0.00% | 92.35% | Top-10 | **98.4%** | Excellent |
| PAM4 | 99.28% | 0.00% | 93.35% | Top-20 | **94.0%** | Excellent |
| AM-DSB | 98.28% | 0.10% | 90.20% | Top-20 | **91.6%** | Excellent |
| GFSK | 99.67% | 0.00% | 88.05% | Top-20 | **88.3%** | Good |
| QAM16 (Top-20) | 94.56% | 0.00% | 45.70% | Top-20 | **48.3%** | Poor |
| 8PSK | 97.94% | 0.00% | 33.00% | Top-20 | **33.5%** | Failed |
| QAM16 (Top-10) | 94.56% | 0.00% | 8.70% | Top-10 | **9.2%** | Destroyed |

---

## 6. Cross-Attack Comparison

### 6.1 Attack Effectiveness: CW vs EADEN

| Modulation | CW Mean Acc | EADEN Mean Acc | More Effective |
|------------|-------------|----------------|----------------|
| QAM64 | 0.22% | 0.00% | EADEN |
| QAM16 | 0.00% | 0.00% | Tied |
| 8PSK | 2.11% | 0.00% | EADEN |
| QPSK | 40.78% | N/A | -- |
| AM-DSB | 65.00% | 0.10% | EADEN (dramatically) |
| GFSK | 20.22% | 0.00% | EADEN |
| PAM4 | 76.50% | 0.00% | EADEN (dramatically) |

EADEN is uniformly more effective than CW across all modulations. The difference is most dramatic for AM-DSB (65.00% to 0.10%) and PAM4 (76.50% to 0.00%). This is consistent with EADEN's elastic-net regularization producing perturbations that are optimized across both L1 and L2 norms, enabling more efficient use of the perturbation budget.

### 6.2 Defense Recovery: CW vs EADEN

| Modulation | CW Recovery Rate | EADEN Recovery Rate | Delta |
|------------|-----------------|---------------------|-------|
| QAM64 (Top-10) | 98.2% | 98.4% | +0.2pp |
| PAM4 (Top-20) | 98.0% | 94.0% | -4.0pp |
| AM-DSB (Top-20) | 94.3% | 91.6% | -2.7pp |
| GFSK (Top-20) | 86.3% | 88.3% | +2.0pp |
| QAM16 (Top-20) | 52.8% | 48.3% | -4.5pp |
| 8PSK (Top-20) | 34.2% | 33.5% | -0.7pp |

Defense recovery rates are remarkably consistent between the two attacks, with differences within 5 percentage points for all modulations. This is a significant finding: **the defense recovery rate is determined primarily by the modulation's spectral characteristics, not by the specific attack algorithm**. Both CW and EADEN produce perturbations that interact similarly with spectral filtering, despite their different optimization objectives and perturbation geometries.

### 6.3 The Defense-Attack Independence Principle

The near-identical recovery rates across CW and EADEN attacks support a general principle: IFFT Top-K defense effectiveness is a function of the signal's spectral structure, not the attack's perturbation structure. This has two implications:

1. **For defenders**: Recovery rate can be predicted from modulation type alone, without needing to characterize the specific attack.
2. **For attackers**: Choosing a different attack algorithm does not help evade spectral defenses; the bottleneck is the signal's spectral overlap with the perturbation, not the perturbation's shape.

---

## 7. Analysis: Why No Universal K Exists

### 7.1 Spectral Complexity and Bandwidth Occupancy

The fundamental question is: why does QAM64 recover with Top-10 while QAM16 fails with Top-20? The answer lies in how different modulations distribute their energy across the frequency spectrum and how the AWN classifier uses spectral features.

**Spectral concentration hypothesis**: Higher-order modulations with more constellation points tend to have their classification-relevant features concentrated in fewer dominant spectral components. This seems counterintuitive -- one might expect more complex modulations to need more spectral bins. The explanation is that the AWN classifier's decision boundary for QAM64 relies on a small number of high-energy spectral features (carrier, main lobe), and these survive aggressive Top-K filtering.

For 128-sample IQ signals with 65 real FFT bins per channel:

| K Value | Bins Retained | Percentage of Spectrum |
|---------|---------------|----------------------|
| Top-10 | 10 | 15.4% |
| Top-20 | 20 | 30.8% |
| Top-30 | 30 | 46.2% |
| Top-50 | 50 | 76.9% |

### 7.2 The Perturbation-Signal Overlap Problem

All gradient-based attacks (CW, EADEN, PGD, etc.) optimize perturbations to be maximally effective at the classifier's decision boundary. For AMC classifiers, this means the perturbation energy concentrates in frequency bins where the classifier's features are most sensitive.

For **narrowband modulations** (8PSK, QAM16): The signal occupies relatively few spectral bins, and the classifier's sensitive features overlap heavily with the perturbation's spectral support. Top-K filtering faces a dilemma:
- Low K: Removes perturbation but also removes signal content needed for classification
- High K: Preserves signal but also preserves the perturbation

For **wideband/simple modulations** (AM-DSB, PAM4, GFSK): The signal's dominant spectral features are highly concentrated in a few bins with large magnitude. The perturbation, being L2-constrained, is necessarily much smaller in magnitude. Top-K naturally selects the large signal bins and discards the smaller perturbation bins.

For **QAM64**: Despite being a high-order modulation, QAM64's classification features are concentrated in a few dominant spectral components (likely the pulse-shaped main lobe). Top-10 captures these effectively. The perturbation energy, distributed across many bins, is mostly discarded.

### 7.3 Recovery Rate vs Modulation Order

Plotting recovery rate against modulation complexity reveals a non-monotonic relationship:

```
Recovery Rate (%)     Modulation Order / Spectral Complexity
    |
 98 |  * QAM64 (Top-10)    * PAM4 (Top-20)
 94 |                                         * AM-DSB (Top-20)
 88 |                       * GFSK (Top-20)
 78 |  * QPSK (Top-20)
    |
 53 |                       * QAM16 (Top-20)
 34 |  * 8PSK (Top-20)
    +------------------------------------------------------>
       Simple (AM, FM)      Medium (PSK, PAM)    High (QAM)
```

This non-monotonic pattern suggests that "modulation order" alone does not predict defense effectiveness. The relevant factor is the **ratio of the signal's classification-relevant spectral energy to the total spectral energy** in the top-K bins. QAM64 has this ratio high in the top-10 bins; QAM16 and 8PSK do not.

### 7.4 SNR Dependence as Diagnostic

The strong SNR dependence of 8PSK and QPSK recovery (improving from near-0% at SNR 0 to 50--55% at SNR 14+) provides a diagnostic signal. At higher SNR, the signal's spectral peaks become more prominent relative to both noise and perturbation, making Top-K more effective at selecting signal bins over perturbation bins. For QAM64 and PAM4, the signal peaks are already dominant at all SNR levels, explaining their SNR-independent recovery.

The SNR-independent recovery of QAM16 (flat at approximately 47% across all SNR levels) suggests a different failure mode: the defense distorts the waveform in a way that the classifier cannot accommodate regardless of signal strength. This is consistent with QAM16 having a spectral profile that is fundamentally altered by Top-20 truncation.

---

## 8. The Chicken-and-Egg Problem for Adaptive Spectral Defenses

### 8.1 The Fundamental Paradox

The experimental results clearly show that different modulations require different K values for optimal recovery:

| Modulation | Optimal K | Recovery at Optimal K |
|------------|-----------|----------------------|
| QAM64 | 10 | 98% |
| PAM4 | 20 | 98% |
| AM-DSB | 20 | 94% |
| GFSK | 20 | 86% |
| QPSK | 20 | 78% (SNR >= 8) |
| QAM16 | >30 (untested) | <53% |
| 8PSK | >30 (untested) | <34% |

An adaptive defense would select K based on the modulation type. However, **the AMC classifier is the very capability the adversary has compromised**. If the defender could reliably identify the modulation type, there would be no need for a defense in the first place.

### 8.2 Possible Solutions and Their Limitations

**Multi-pass classification**: Apply Top-K filtering with multiple K values, run the classifier on each filtered version, and select the output with highest classifier confidence (softmax probability). This avoids needing to know the modulation a priori but introduces significant computational overhead (K passes per signal) and may be fooled by adversarial inputs that produce high confidence at incorrect K values.

**Spectral complexity estimation**: Before classification, estimate the signal's spectral complexity (e.g., number of bins above a threshold, spectral entropy) and select K accordingly. This does not require the AMC classifier and could be implemented as a lightweight preprocessing step. However, the adversarial perturbation itself changes the spectral profile, potentially biasing the complexity estimate.

**Ensemble defense**: Run the classifier on both the unfiltered and filtered (at a fixed K) versions. If the two predictions agree, accept the result; if they disagree, flag the signal as potentially adversarial and apply more aggressive filtering. This leverages the observation that Top-K filtering changes adversarial predictions more than clean predictions.

**CRC-based feedback**: In the full communication pipeline, CRC failure after demodulation indicates incorrect modulation selection. The receiver could iteratively try different demodulators until CRC passes. This approach is practical but slow (up to N_mod demodulation attempts) and only works when CRC or similar integrity checks are available.

### 8.3 Implications for SigGuard-Style Defenses

SigGuard and similar approaches propose a fixed Top-K value applied uniformly to all signals. Our results demonstrate that:

1. **No fixed K achieves acceptable recovery for all modulations.** Top-10 works for QAM64 (98%) but destroys QAM16 (9%). Top-20 works for PAM4 (98%) but fails for 8PSK (34%).

2. **Even with optimal per-modulation K, some modulations remain unrecoverable.** QAM16 at Top-20 recovers only 50%, and 8PSK at Top-20 recovers only 34%. These modulations may require K values of 30 or higher, at which point the defense retains enough perturbation energy to be ineffective.

3. **The defense creates a new attack surface.** An adversary who knows the fixed K value can craft perturbations that survive the filtering. The Top-K filter is a deterministic, differentiable operation that can be incorporated into the attack optimization.

---

## 9. Discussion

### 9.1 Security Implications for Cognitive Radio

The combination of effective adversarial attacks (0% AMC accuracy under CW/EADEN) and partially effective spectral defenses creates a real security concern for deployed cognitive radio systems. The attack-defense asymmetry is severe:

- **Attacker advantage**: A white-box adversary can fool AMC for all modulation types with near-100% success
- **Defender limitation**: Spectral defense recovers 4 of 7 modulations to >85% accuracy, but fails for the remaining 3

In a practical deployment, the defender cannot choose which modulation types are in use -- that is determined by the transmitter and channel conditions. A cognitive radio that falls back to 8PSK or QAM16 under certain channel conditions would have no effective spectral defense against adversarial AMC attacks.

### 9.2 Comparison with Prior CRC Defense Results

Our prior CRC defense experiment (synthetic signals, SNR=18 dB) found that CW attack is purely control-plane: it fools the AMC but does not corrupt data. The current IFFT Top-K evaluation complements those findings:

| Prior Finding (CRC Report) | Current Finding (IFFT Report) | Relationship |
|---------------------------|-------------------------------|-------------|
| CW is control-plane only | CW and EADEN achieve 0% AMC accuracy | Confirms severity |
| Top-K CRC preservation is modulation-dependent | Top-K AMC recovery is modulation-dependent | Consistent |
| QAM64 needs Top-50 for CRC | QAM64 needs only Top-10 for AMC | Different metrics |
| 8PSK/QAM16 defense fails | 8PSK/QAM16 defense fails | Consistent |

The difference in optimal K for QAM64 between CRC preservation (Top-50) and AMC recovery (Top-10) highlights an important distinction: the classifier may use different spectral features than the demodulator. A signal that is correctly classified may still have insufficient spectral content for reliable demodulation.

### 9.3 Limitations

1. **White-box assumption**: Both CW and EADEN assume full knowledge of the classifier. Black-box or transfer attacks may produce different perturbation distributions that interact differently with spectral filtering.

2. **Single classifier**: All results are for the AWN classifier. Different AMC architectures (CNN, ResNet, LSTM-based) may have different spectral sensitivities, leading to different defense recovery patterns.

3. **Limited K sweep**: We tested only Top-10 and Top-20. A finer K sweep (e.g., Top-15, Top-25, Top-30) could reveal more about the recovery curve and identify optimal K values for the currently-failing modulations.

4. **RML2016.10a dataset**: The 128-sample signal length and specific SNR/channel model may not generalize to other datasets (e.g., RML2018.01a with 1024 samples) or real-world captures.

5. **No adaptive attack evaluation**: We did not evaluate adversaries that specifically target the spectral defense (e.g., optimizing perturbations to survive Top-K filtering). Such adaptive attacks would likely degrade defense effectiveness further.

### 9.4 Recommended Follow-Up Experiments

1. **Fine-grained K sweep for QAM16 and 8PSK**: Test Top-25, Top-30, Top-35, Top-40 to find if a viable recovery point exists.
2. **QPSK with Top-10 and Top-15**: Given QPSK's good high-SNR recovery with Top-20, test whether lower K improves low-SNR performance.
3. **BPSK evaluation**: BPSK was included in the CRC report but missing from this IFFT study. It should be evaluated for completeness.
4. **Adaptive Top-K**: Implement the multi-pass classifier confidence approach and evaluate its effectiveness.
5. **Adaptive attacks**: Evaluate a CW attack that includes the Top-K filter in its optimization loop.
6. **Cross-architecture evaluation**: Test whether recovery patterns hold for CNN-based and LSTM-based AMC classifiers.

---

## 10. Conclusions

We presented a systematic evaluation of IFFT Top-K spectral filtering as a defense against CW and EADEN adversarial attacks on the AWN automatic modulation classifier. Our experiments across seven modulations and nine SNR levels yield three principal conclusions.

**First**, no single Top-K value provides universal defense. QAM64 recovers 98% of clean accuracy with just 10 retained frequency bins, while QAM16 recovers only 50% with 20 bins and 8PSK recovers only 34%. This modulation-dependent effectiveness is consistent across both CW and EADEN attacks, indicating that recovery rate is determined by signal spectral structure rather than attack algorithm.

**Second**, simple and analog modulations (AM-DSB, PAM4, GFSK) are well-protected by Top-20 filtering with 86--98% recovery, while phase-shift keyed modulations with intermediate complexity (8PSK, QAM16) represent the hardest cases for spectral defenses.

**Third**, the requirement for modulation-aware K selection creates a chicken-and-egg problem: the defense needs modulation information that only the compromised AMC classifier can provide. This fundamental paradox limits the practical applicability of spectral filtering as a standalone defense and motivates research into attack-agnostic defense strategies that do not depend on modulation knowledge.

These findings challenge the viability of fixed-K spectral defenses proposed in prior work and underscore the need for adaptive, possibly multi-modal defense architectures that can operate without accurate modulation identification.

---

## Appendix A: Complete Per-SNR Data Tables

### A.1 CW Attack -- All Modulations, All SNR Levels

| SNR | QAM64 Clean | QAM64 CW | QAM64 T10 | QAM16 Clean | QAM16 CW | QAM16 T20 | 8PSK Clean | 8PSK CW | 8PSK T20 |
|-----|-------------|----------|-----------|-------------|----------|-----------|------------|---------|----------|
| 0 | 0.930 | 0.000 | 0.910 | 0.905 | 0.000 | 0.390 | 0.925 | 0.005 | 0.030 |
| 2 | 0.935 | 0.000 | 0.945 | 0.980 | 0.000 | 0.445 | 0.990 | 0.000 | 0.045 |
| 4 | 0.945 | 0.000 | 0.905 | 0.920 | 0.000 | 0.510 | 0.980 | 0.005 | 0.130 |
| 8 | 0.935 | 0.000 | 0.920 | 0.925 | 0.000 | 0.465 | 0.990 | 0.010 | 0.305 |
| 10 | 0.955 | 0.000 | 0.915 | 0.950 | 0.000 | 0.485 | 0.985 | 0.010 | 0.305 |
| 12 | 0.925 | 0.005 | 0.940 | 0.945 | 0.000 | 0.570 | 0.995 | 0.020 | 0.570 |
| 14 | 0.950 | 0.005 | 0.950 | 0.955 | 0.000 | 0.550 | 0.990 | 0.030 | 0.535 |
| 16 | 0.925 | 0.005 | 0.915 | 0.965 | 0.000 | 0.515 | 0.985 | 0.060 | 0.575 |
| 18 | 0.955 | 0.005 | 0.905 | 0.965 | 0.000 | 0.570 | 0.975 | 0.050 | 0.535 |

| SNR | AM-DSB Clean | AM-DSB CW | AM-DSB T20 | GFSK Clean | GFSK CW | GFSK T20 | PAM4 Clean | PAM4 CW | PAM4 T20 | QPSK Clean | QPSK CW | QPSK T20 |
|-----|-------------|----------|-----------|------------|---------|----------|------------|---------|----------|------------|---------|----------|
| 0 | 0.895 | 0.120 | 0.775 | 0.990 | 0.460 | 0.840 | 0.990 | 0.550 | 0.940 | 0.975 | 0.005 | 0.165 |
| 2 | 0.960 | 0.210 | 0.855 | 0.980 | 0.150 | 0.780 | 0.995 | 0.715 | 0.960 | 0.990 | 0.040 | 0.495 |
| 4 | 0.990 | 0.185 | 0.845 | 1.000 | 0.030 | 0.855 | 0.995 | 0.710 | 0.955 | 0.990 | 0.250 | 0.790 |
| 8 | 1.000 | 0.520 | 0.915 | 1.000 | 0.150 | 0.915 | 0.995 | 0.810 | 0.990 | 0.995 | 0.445 | 0.885 |
| 10 | 1.000 | 0.830 | 0.970 | 1.000 | 0.175 | 0.865 | 1.000 | 0.850 | 0.985 | 0.995 | 0.605 | 0.920 |
| 12 | 1.000 | 0.985 | 0.995 | 1.000 | 0.115 | 0.830 | 0.990 | 0.790 | 0.980 | 0.990 | 0.615 | 0.935 |
| 14 | 1.000 | 1.000 | 1.000 | 1.000 | 0.245 | 0.875 | 0.985 | 0.785 | 0.980 | 0.965 | 0.555 | 0.890 |
| 16 | 1.000 | 1.000 | 1.000 | 1.000 | 0.245 | 0.875 | 0.985 | 0.820 | 0.970 | 0.985 | 0.545 | 0.925 |
| 18 | 1.000 | 1.000 | 1.000 | 1.000 | 0.255 | 0.905 | 1.000 | 0.855 | 1.000 | 0.985 | 0.610 | 0.945 |

### A.2 EADEN Attack -- All Modulations, All SNR Levels

| SNR | QAM64 EADEN | QAM64 T10 | QAM16 EADEN | QAM16 T10 | QAM16 T20 | 8PSK EADEN | 8PSK T20 |
|-----|------------|-----------|------------|-----------|-----------|-----------|----------|
| 0 | 0.000 | 0.930 | 0.000 | 0.095 | 0.365 | 0.000 | 0.035 |
| 2 | 0.000 | 0.950 | 0.000 | 0.060 | 0.400 | 0.000 | 0.070 |
| 4 | 0.000 | 0.940 | 0.000 | 0.100 | 0.500 | 0.000 | 0.130 |
| 6 | 0.000 | 0.930 | 0.000 | 0.110 | 0.490 | 0.000 | 0.275 |
| 8 | 0.000 | 0.915 | 0.000 | 0.105 | 0.465 | 0.000 | 0.370 |
| 10 | 0.000 | 0.905 | 0.000 | 0.085 | 0.480 | 0.000 | 0.445 |
| 12 | 0.000 | 0.955 | 0.000 | 0.035 | 0.460 | 0.000 | 0.440 |
| 14 | 0.000 | 0.915 | 0.000 | 0.070 | 0.460 | 0.000 | 0.540 |
| 16 | 0.000 | 0.875 | 0.000 | 0.105 | 0.465 | 0.000 | 0.525 |
| 18 | 0.000 | 0.920 | 0.000 | 0.105 | 0.485 | 0.000 | 0.470 |

| SNR | AM-DSB EADEN | AM-DSB T20 | GFSK EADEN | GFSK T20 | PAM4 EADEN | PAM4 T20 |
|-----|-------------|-----------|-----------|----------|-----------|----------|
| 0 | 0.000 | 0.845 | 0.000 | 0.760 | 0.000 | 0.825 |
| 2 | 0.000 | 0.745 | 0.000 | 0.800 | 0.000 | 0.895 |
| 4 | 0.000 | 0.860 | 0.000 | 0.875 | 0.000 | 0.935 |
| 6 | 0.000 | 0.935 | 0.000 | 0.920 | 0.000 | 0.965 |
| 8 | 0.000 | 0.895 | 0.000 | 0.925 | 0.000 | 0.960 |
| 10 | 0.000 | 0.945 | 0.000 | 0.900 | 0.000 | 0.955 |
| 12 | 0.000 | 0.915 | 0.000 | 0.935 | 0.000 | 0.925 |
| 14 | 0.000 | 0.970 | 0.000 | 0.915 | 0.000 | 0.970 |
| 16 | 0.005 | 0.965 | 0.000 | 0.905 | 0.000 | 0.945 |
| 18 | 0.005 | 0.945 | 0.000 | 0.870 | 0.000 | 0.960 |

---

## Appendix B: Reproducibility

### B.1 Attack Commands

```bash
# CW attack with IFFT Top-K defense
python main.py --mode multi_attack_eval --dataset 2016.10a --ckpt_path ./checkpoint \
  --attack_list cw --ta_box minmax --cw_kappa 1 --cw_steps 100 --cw_lr 0.001 \
  --mod_filter QAM64 --def_topk 10 --eval_limit_per_cell 200

python main.py --mode multi_attack_eval --dataset 2016.10a --ckpt_path ./checkpoint \
  --attack_list cw --ta_box minmax --cw_kappa 1 --cw_steps 100 --cw_lr 0.001 \
  --mod_filter QAM16 --def_topk 20 --eval_limit_per_cell 200

# EADEN attack with IFFT Top-K defense
python main.py --mode multi_attack_eval --dataset 2016.10a --ckpt_path ./checkpoint \
  --attack_list eaden --ta_box minmax \
  --mod_filter QAM64 --def_topk 10 --eval_limit_per_cell 200

python main.py --mode multi_attack_eval --dataset 2016.10a --ckpt_path ./checkpoint \
  --attack_list eaden --ta_box minmax \
  --mod_filter QAM16 --def_topk 10 --eval_limit_per_cell 200

python main.py --mode multi_attack_eval --dataset 2016.10a --ckpt_path ./checkpoint \
  --attack_list eaden --ta_box minmax \
  --mod_filter QAM16 --def_topk 20 --eval_limit_per_cell 200
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
