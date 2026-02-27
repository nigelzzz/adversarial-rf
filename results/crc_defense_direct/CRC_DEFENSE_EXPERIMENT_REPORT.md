# CRC Defense Experiment Report: Adversarial CW Attack on AMC and Data-Link Integrity

## 1. Objective

Evaluate the impact of CW (Carlini & Wagner) adversarial attack on:
1. **Control plane**: Does CW fool the AMC (Automatic Modulation Classification) model?
2. **Data plane**: Does CW corrupt the actual transmitted data (CRC check)?
3. **Defense**: Can FFT Top-K spectral filtering recover both AMC accuracy and data integrity?

## 2. Pipeline

```
Random bits → CRC-8 → Modulate → RRC pulse-shape → AWGN channel
    → [CW Attack] → [FFT Top-K Defense] → AWN Classifier → Demodulate → CRC Check
```

All signals are synthetic with `target_rms=0.0082` (matching RML2016.10a amplitude distribution), enabling AWN to classify them directly without a Track A/B split.

## 3. Experimental Setup

| Parameter | Value |
|-----------|-------|
| SNR | 18 dB and 0 dB |
| Burst length | 16 symbols (2 pilot + 14 data), 128 samples |
| Samples per symbol | 8 |
| RRC rolloff | 0.35 |
| Target RMS | 0.0082 |
| CW attack | c=1.0, steps=1000, L2 norm, minmax box |
| Top-K values tested | 10, 20, 30, 40, 50 |
| Bursts per modulation | 200 |
| Modulations | BPSK, QPSK, 8PSK, QAM16, QAM64, PAM4, CPFSK, GFSK |
| Seed | 42 |

### Scenario Definitions

| Scenario | Signal | Demod Selection | What It Measures |
|----------|--------|-----------------|------------------|
| **Clean** | Original | Oracle (true mod) | Baseline CRC performance |
| **Clean+AMC** | Original | AWN prediction | AWN accuracy on synthetic signals |
| **CW** | CW-attacked | Oracle (true mod) | Does CW corrupt data? |
| **CW+AMC** | CW-attacked | AWN prediction | Full attack impact on data link |
| **CW+TopK** | CW → Top-K filtered | Oracle (true mod) | Does Top-K recover signal quality? |
| **CW+TopK+AMC** | CW → Top-K filtered | AWN on filtered signal | Full defense effectiveness |
| **Clean+TopK** | Original → Top-K filtered | Oracle (true mod) | Defense cost on clean signal |

---

## 4. Results at SNR = 18 dB

### 4.1 Core Finding: CW Is a Control-Plane Attack

CRC pass rate with **oracle demodulator** (receiver knows the true modulation):

| Mod | Clean | CW + Oracle | Delta |
|-----|-------|-------------|-------|
| BPSK | 100.0% | **100.0%** | 0 |
| QPSK | 100.0% | **100.0%** | 0 |
| 8PSK | 100.0% | **100.0%** | 0 |
| QAM16 | 100.0% | **100.0%** | 0 |
| QAM64 | 89.5% | **86.5%** | -3.0pp |
| PAM4 | 100.0% | **98.5%** | -1.5pp |
| CPFSK | 100.0% | **100.0%** | 0 |
| GFSK | 99.5% | **99.5%** | 0 |

**All 8 modulations pass CRC at 86–100% under CW attack when the correct demodulator is used.** The CW perturbation (L2-optimized) is too small to move constellation points past decision boundaries. It only fools the AWN neural network.

### 4.2 Attack Impact: Wrong Demod Breaks the Link

When AWN is fooled by CW → selects wrong demodulator → data link breaks:

| Mod | Clean+AMC CRC | AWN Clean Acc | CW+AMC CRC | AWN CW Acc |
|-----|---------------|---------------|-------------|------------|
| BPSK | 100.0% | 100.0% | **11.0%** | 10.0% |
| QPSK | 100.0% | 100.0% | **2.5%** | 2.0% |
| 8PSK | 96.5% | 96.0% | **1.0%** | 0.0% |
| QAM16 | 77.5% | 77.5% | **0.5%** | 0.5% |
| QAM64 | 82.5% | 92.5% | **1.5%** | 1.0% |
| PAM4 | 100.0% | 100.0% | **4.0%** | 3.5% |
| CPFSK | 100.0% | 100.0% | **1.0%** | 1.0% |
| GFSK | 94.5% | 1.0%* | **94.5%** | 1.0%* |

*GFSK: AWN misclassifies synthetic GFSK as CPFSK (both FSK), but CPFSK demod still decodes GFSK — so CRC passes despite wrong classification.

**CW attack reduces CRC pass rate from 77–100% to 0.5–11% for 7 of 8 modulations.**

### 4.3 FFT Top-K Defense: Oracle Demod (Signal Quality)

CRC pass rate when the receiver knows the correct modulation, measuring whether Top-K preserves signal integrity:

| Mod | Clean | CW | CW+T10 | CW+T20 | CW+T30 | CW+T40 | CW+T50 |
|-----|-------|----|--------|--------|--------|--------|--------|
| BPSK | 100.0% | 100.0% | 97.5% | 100.0% | 100.0% | 100.0% | 100.0% |
| QPSK | 100.0% | 100.0% | 76.5% | 100.0% | 100.0% | 100.0% | 100.0% |
| 8PSK | 100.0% | 100.0% | 7.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| QAM16 | 100.0% | 100.0% | 1.0% | 94.5% | 99.0% | 99.5% | 100.0% |
| QAM64 | 89.5% | 86.5% | 0.5% | 36.0% | 63.5% | 76.5% | 81.0% |
| PAM4 | 100.0% | 98.5% | 11.5% | 98.5% | 98.5% | 99.0% | 99.5% |
| CPFSK | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| GFSK | 99.5% | 99.5% | 99.5% | 99.5% | 99.5% | 99.5% | 99.5% |

- **Top-10**: Too aggressive — destroys signal for 8PSK (7%), QAM16 (1%), QAM64 (0.5%)
- **Top-20**: Good for low-order mods, but QAM64 only 36%
- **Top-50**: Best overall, QAM64 reaches 81%
- **CPFSK/GFSK**: Constant-envelope FSK is unaffected by Top-K

### 4.4 FFT Top-K Defense: AMC Demod (Full Defense)

CRC pass rate when AWN classifies the Top-K filtered signal and selects the demodulator:

| Mod | Clean+AMC | CW+AMC | CW+T10+AMC | CW+T20+AMC | CW+T30+AMC | CW+T40+AMC | CW+T50+AMC |
|-----|-----------|--------|------------|------------|------------|------------|------------|
| BPSK | 100.0% | 11.0% | 2.0% | **57.5%** | 54.0% | 54.0% | 52.0% |
| QPSK | 100.0% | 2.5% | 0.0% | **49.0%** | 47.5% | 49.0% | 52.0% |
| 8PSK | 96.5% | 1.0% | 0.5% | **22.0%** | 21.5% | 23.0% | 22.5% |
| QAM16 | 77.5% | 0.5% | 0.0% | 28.0% | 23.0% | 28.5% | **30.5%** |
| QAM64 | 82.5% | 1.5% | 0.5% | 31.0% | 55.5% | 63.0% | **61.5%** |
| PAM4 | 100.0% | 4.0% | 11.0% | **93.0%** | 79.5% | 78.0% | 77.0% |
| CPFSK | 100.0% | 1.0% | 3.5% | 25.5% | 32.5% | 36.0% | **41.0%** |
| GFSK | 94.5% | 94.5% | 7.5% | 82.5% | 93.0% | 93.0% | **93.5%** |

**Best defense results by modulation:**

| Mod | Best CRC Recovery | Best K | Practical? |
|-----|-------------------|--------|------------|
| PAM4 | **93.0%** (from 4.0%) | K=20 | Yes — excellent recovery |
| GFSK | **93.5%** (unchanged) | K=50 | N/A — CW doesn't affect GFSK |
| QAM64 | **63.0%** (from 1.5%) | K=40 | Partial — limited by signal distortion |
| BPSK | **57.5%** (from 11.0%) | K=20 | Poor — in-band perturbation |
| QPSK | **52.0%** (from 2.5%) | K=50 | Poor — in-band perturbation |
| CPFSK | **41.0%** (from 1.0%) | K=50 | Poor |
| QAM16 | **30.5%** (from 0.5%) | K=50 | Fails |
| 8PSK | **23.0%** (from 1.0%) | K=40 | Fails |

### 4.5 AWN AMC Accuracy Recovery

| Mod | Clean | CW | CW+T20 | CW+T30 | CW+T40 | CW+T50 |
|-----|-------|----|--------|--------|--------|--------|
| BPSK | 100.0% | 10.0% | 55.5% | 52.0% | 52.0% | 50.0% |
| QPSK | 100.0% | 2.0% | 49.0% | 47.0% | 49.0% | 52.0% |
| 8PSK | 96.0% | 0.0% | 22.0% | 20.0% | 22.0% | 20.5% |
| QAM16 | 77.5% | 0.5% | 29.0% | 23.5% | 28.0% | 29.5% |
| QAM64 | 92.5% | 1.0% | 89.5% | 87.5% | 84.0% | 76.5% |
| PAM4 | 100.0% | 3.5% | 94.5% | 81.0% | 79.0% | 77.5% |
| CPFSK | 100.0% | 1.0% | 25.5% | 32.5% | 36.0% | 41.0% |
| GFSK | 1.0% | 1.0% | 1.0% | 1.0% | 1.0% | 1.0% |

- QAM64 and PAM4 show strong AMC recovery (77–95%) at K=20
- BPSK/QPSK plateau at ~50% regardless of K
- 8PSK/QAM16 never exceed ~30%

### 4.6 Defense Cost on Clean Signal

CRC pass rate when Top-K is applied to clean signals (no attack), with oracle demod:

| Mod | Clean | Clean+T10 | Clean+T20 | Clean+T30 | Clean+T40 | Clean+T50 |
|-----|-------|-----------|-----------|-----------|-----------|-----------|
| BPSK | 100.0% | 98.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| QPSK | 100.0% | 80.5% | 100.0% | 100.0% | 100.0% | 100.0% |
| 8PSK | 100.0% | 10.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| QAM16 | 100.0% | 1.5% | 95.5% | 99.5% | 100.0% | 100.0% |
| QAM64 | 89.5% | 0.5% | 41.5% | 72.5% | 82.5% | 85.0% |
| PAM4 | 100.0% | 12.5% | 100.0% | 100.0% | 100.0% | 100.0% |
| CPFSK | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| GFSK | 99.5% | 99.5% | 99.5% | 99.5% | 99.5% | 99.5% |

- **Top-10 is destructive** for all constellation mods except BPSK
- **Top-20 is safe** for most mods (QAM64 is the exception at 41.5%)
- **QAM64 requires Top-50** to preserve 85% CRC
- **FSK mods (CPFSK, GFSK)** are unaffected by Top-K filtering

---

## 5. Results at SNR = 0 dB

### 5.1 Baseline: Thermal Noise Dominates

| Mod | Clean CRC (Oracle) | Bits/Symbol | Comment |
|-----|-------------------|-------------|---------|
| BPSK | **100.0%** | 1 | Survives SNR=0 |
| QPSK | **80.0%** | 2 | Partially degraded |
| PAM4 | **25.5%** | 2 | Mostly broken |
| 8PSK | **6.0%** | 3 | Broken by noise |
| CPFSK | **7.5%** | 1 | Broken by noise |
| GFSK | **7.0%** | 1 | Broken by noise |
| QAM16 | **0.5%** | 4 | Broken by noise |
| QAM64 | **0.0%** | 6 | Completely broken |

At SNR=0, thermal noise alone destroys higher-order modulations. Only BPSK maintains reliable CRC.

### 5.2 CW Attack at SNR=0: Still Control-Plane Only

| Mod | Clean CRC | CW + Oracle CRC | Delta |
|-----|-----------|-----------------|-------|
| BPSK | 100.0% | **100.0%** | 0 |
| QPSK | 80.0% | **78.0%** | -2.0pp |
| 8PSK | 6.0% | **5.0%** | -1.0pp |
| QAM16 | 0.5% | **0.5%** | 0 |
| QAM64 | 0.0% | **0.0%** | 0 |
| PAM4 | 25.5% | **21.5%** | -4.0pp |
| CPFSK | 7.5% | **7.5%** | 0 |
| GFSK | 7.0% | **7.0%** | 0 |

**CW still does not corrupt data at SNR=0.** The delta between Clean and CW+Oracle is 0–4pp, within noise variation.

### 5.3 Full Results at SNR=0

#### Oracle Demod CRC

| Mod | Clean | CW | CW+T10 | CW+T20 | CW+T30 | CW+T40 | CW+T50 |
|-----|-------|----|--------|--------|--------|--------|--------|
| BPSK | 100.0% | 100.0% | 79.0% | 100.0% | 99.0% | 99.5% | 99.5% |
| QPSK | 80.0% | 78.0% | 14.5% | 55.5% | 64.0% | 71.5% | 73.0% |
| 8PSK | 6.0% | 5.0% | 2.0% | 4.0% | 3.5% | 1.5% | 4.5% |
| QAM16 | 0.5% | 0.5% | 0.5% | 1.0% | 0.0% | 0.5% | 0.5% |
| QAM64 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| PAM4 | 25.5% | 21.5% | 1.5% | 9.5% | 17.5% | 17.5% | 20.5% |
| CPFSK | 7.5% | 7.5% | 7.5% | 7.5% | 7.5% | 7.5% | 7.5% |
| GFSK | 7.0% | 7.0% | 7.0% | 7.0% | 7.0% | 7.0% | 7.0% |

#### AMC Demod CRC

| Mod | Clean+AMC | CW+AMC | CW+T20+AMC | CW+T50+AMC |
|-----|-----------|--------|------------|------------|
| BPSK | 82.0% | 0.5% | 13.0% | **33.5%** |
| QPSK | 40.0% | 1.5% | 2.5% | **23.0%** |
| 8PSK | 4.0% | 1.0% | 0.0% | 2.5% |
| QAM16 | 0.5% | 0.0% | 0.5% | 1.5% |
| QAM64 | 0.5% | 0.5% | 0.0% | 0.0% |
| PAM4 | 21.5% | 1.5% | 6.5% | **15.0%** |
| CPFSK | 2.5% | 0.0% | 1.0% | 1.5% |
| GFSK | 2.5% | 2.5% | 1.0% | 3.5% |

At SNR=0, defense is largely irrelevant — thermal noise is the dominant impairment, not the adversarial attack.

---

## 6. FGSM vs CW Comparison (SNR=18 dB)

Does FGSM also preserve data integrity like CW?

### Oracle CRC under FGSM at various epsilon (SNR=18)

| Mod | Clean | CW | FGSM eps=0.01 | FGSM eps=0.03 | FGSM eps=0.05 | FGSM eps=0.1 |
|-----|-------|----|---------------|---------------|---------------|--------------|
| BPSK | 100% | 100% | 100% | 100% | 100% | **100%** |
| QPSK | 100% | 100% | 100% | 100% | 100% | **100%** |
| 8PSK | 100% | 100% | 100% | 100% | 99.5% | **36%** |
| QAM16 | 100% | 100% | 100% | 97.5% | 72% | **1.5%** |
| QAM64 | 92% | 87% | 86% | 21.5% | 1% | **0%** |
| PAM4 | 100% | 99% | 100% | 100% | 97.5% | **42%** |

### Perturbation size comparison

| Attack | QAM64 L2 norm | AMC Acc | Oracle CRC |
|--------|---------------|---------|------------|
| CW (c=1.0) | **0.0029** | 0% | 87% |
| FGSM eps=0.01 | 0.0041 | 50.5% | 86% |
| FGSM eps=0.03 | **0.0122** (4.2x CW) | 12% | 21.5% |
| FGSM eps=0.1 | **0.0406** (14x CW) | 63% | 0% |

**Key difference:**
- **CW** is L2-optimized: finds the **minimum** perturbation to fool AMC → data stays intact
- **FGSM** is Linf-uniform: applies fixed perturbation in gradient direction → at high eps, perturbation exceeds constellation decision margins → **corrupts data**

At equivalent AMC degradation (e.g., 8PSK AMC ≈ 5%), both attacks preserve CRC. But FGSM can be pushed far beyond what's needed to fool AMC, entering data-corruption territory.

---

## 7. Root Cause Analysis: Why Top-K Fails for Narrowband Mods

### 7.1 Two independent failure modes

| Mod | Clean AMC | Clean+Top20 AMC | CW+Top20 AMC |
|-----|-----------|-----------------|--------------|
| BPSK | 96.5% | **85.5%** (-11pp) | **49.5%** (-36pp more) |
| 8PSK | 95.0% | **70.5%** (-24.5pp) | **20.5%** (-50pp more) |
| QAM64 | 94.0% | **97.5%** (+3.5pp) | **92.5%** (-5pp) |
| PAM4 | 100.0% | **100.0%** (0) | **92.0%** (-8pp) |

1. **Top-K distortion** (Clean → Clean+Top20): Top-K changes the temporal waveform, degrading AWN features. Narrowband mods lose 11–24.5pp; wideband mods unaffected.
2. **Residual CW perturbation** (Clean+Top20 → CW+Top20): CW perturbation survives Top-K filtering. Narrowband mods lose 36–50pp more; wideband mods lose only 5–8pp.

### 7.2 CW perturbation is in-band

Signal bandwidth at sps=8, rolloff=0.35: `(1+0.35)/(2×8) = 0.084` of sample rate (~22 FFT bins).

Bandpass filter test (keeping only signal bandwidth):

| Mod | Top-K K=20 AMC | Bandpass AMC |
|-----|----------------|--------------|
| BPSK | 49.5% | 54.0% |
| QAM64 | 87.5% | 87.5% |
| PAM4 | 92.0% | 96.5% |

Bandpass filtering provides only marginal improvement because **CW perturbation is placed within the signal bandwidth** — the CW optimizer targets frequencies where AWN's classification features are sensitive. No frequency-domain defense can separate in-band perturbation from signal.

### 7.3 No optimal K value exists

| Mod | CW+T20 AMC | CW+T30 AMC | CW+T40 AMC | CW+T50 AMC |
|-----|------------|------------|------------|------------|
| BPSK | **48.0%** | 42.0% | 41.0% | 41.5% |
| PAM4 | **90.5%** | 76.0% | 76.0% | 73.5% |
| QAM64 | **87.5%** | 87.0% | 80.5% | 76.0% |

Increasing K beyond 20 makes recovery **worse** for most mods: more K retains more CW perturbation while providing diminishing signal quality gains.

---

## 8. Summary of Key Conclusions

### Conclusion 1: CW attack is purely control-plane
CW adversarial perturbation fools the AMC classifier (accuracy drops from 77–100% to 0–10%) but does **not** corrupt the transmitted data. With oracle demod, CRC passes at 86–100% for all 8 modulations at SNR=18. This holds at SNR=0 as well.

### Conclusion 2: Wrong demodulator breaks the data link
When AWN is fooled, it selects the wrong demodulator → ~50% BER → 0–11% CRC pass rate. The real-world impact of AMC attacks is through **incorrect demodulator selection**, not signal corruption.

### Conclusion 3: FFT Top-K defense is modulation-dependent
| Category | Mods | Best CRC Recovery | Verdict |
|----------|------|-------------------|---------|
| Good recovery | PAM4 | 93% (K=20) | Defense works |
| Moderate recovery | QAM64 | 63% (K=40) | Partially effective |
| Poor recovery | BPSK, QPSK, CPFSK | 41–58% (K=20) | Near coin-flip |
| Failed recovery | 8PSK, QAM16 | 23–31% (K=20–50) | Defense ineffective |
| Unaffected | GFSK | 94% (N/A) | CW doesn't affect GFSK |

### Conclusion 4: CW perturbation is in-band
The CW optimizer places perturbation energy within the signal's spectral footprint, making it fundamentally resistant to frequency-domain defenses. Bandpass filtering confirms this — even a perfect signal-bandwidth filter cannot remove the perturbation.

### Conclusion 5: FGSM differs from CW at high epsilon
Unlike CW, FGSM at high epsilon (>0.05) corrupts both control-plane and data-plane. At eps=0.1, QAM16 oracle CRC drops to 1.5% and QAM64 to 0%. CW is the "best-case" adversarial attack for an attacker: maximum AMC disruption with zero data corruption.

### Conclusion 6: SNR determines the operating regime
| SNR | Primary threat | Defense relevance |
|-----|---------------|-------------------|
| 18 dB | CW attack (fools AMC) | Top-K partially effective |
| 0 dB | Thermal noise (destroys signal) | Irrelevant — noise dominates |

At low SNR, higher-order modulations (8PSK, QAM16, QAM64) cannot maintain CRC even without any attack. Only BPSK survives SNR=0.

---

## 9. Implications for System Design

1. **CRC as attack detector**: If the receiver applies CRC and the check fails consistently, it indicates either (a) noise exceeds the modulation's capacity, or (b) the AMC classifier is under attack. Combined with SNR estimation, CRC failure at high SNR is a strong indicator of adversarial attack.

2. **Adaptive defense**: Since Top-K effectiveness varies by modulation, the defense should be adaptive. PAM4 and QAM64 benefit from Top-K; BPSK and 8PSK do not. A practical system could use CRC feedback to select the defense strategy.

3. **Defense limitations**: Frequency-domain defenses fundamentally cannot counter in-band adversarial perturbations. Alternative approaches (adversarial training, ensemble classifiers, signal-domain defenses) may be needed for robust AMC.

---

## 10. Reproducibility

```bash
# SNR=18
python crc_defense_pipeline.py --snr 18 --n_bursts 200 --topk 10,20,30,40,50 \
    --mods BPSK,QPSK,8PSK,QAM16,QAM64,PAM4,CPFSK,GFSK \
    --output_dir ./results/crc_defense_direct/snr18

# SNR=0
python crc_defense_pipeline.py --snr 0 --n_bursts 200 --topk 10,20,30,40,50 \
    --mods BPSK,QPSK,8PSK,QAM16,QAM64,PAM4,CPFSK,GFSK \
    --output_dir ./results/crc_defense_direct/snr0
```

## 11. Output Files

```
results/crc_defense_direct/
├── snr18/
│   ├── crc_defense_pipeline.csv
│   ├── crc_defense_pipeline.json
│   └── fig_crc_defense_pipeline.png
├── snr0/
│   ├── crc_defense_pipeline.csv
│   ├── crc_defense_pipeline.json
│   └── fig_crc_defense_pipeline.png
└── CRC_DEFENSE_EXPERIMENT_REPORT.md
```
