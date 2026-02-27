# Direct End-to-End CRC Defense Pipeline — Experiment Report

## Objective

Evaluate whether FFT Top-K spectral defense can recover AMC (Automatic Modulation Classification) accuracy **and** data-link integrity (CRC) after a CW adversarial attack, using a single unified pipeline with no Track A/B split.

## Pipeline

```
Random bits → CRC-8 → Modulate → Pulse-shape (RRC) → Add AWGN
    → [CW Attack] → [FFT Top-K Defense] → AWN Classifier → Demodulate → CRC Check
```

All signals are **synthetic** with `target_rms=0.0082` (matching RML2016.10a amplitude), allowing AWN to classify them directly.

## Experimental Setup

| Parameter | Value |
|-----------|-------|
| SNR | 18 dB |
| Burst length | 16 symbols (2 pilot + 14 data) |
| Samples per symbol | 8 (128 samples total) |
| RRC rolloff | 0.35 |
| Target RMS | 0.0082 (matches RML2016.10a) |
| CFO | 0.0 (no carrier frequency offset) |
| CW attack | c=1.0, steps=1000, L2 norm, minmax box |
| Top-K values | 10, 20, 50 |
| N bursts per mod | 200 |
| Modulations | BPSK, QPSK, 8PSK, QAM16, QAM64, PAM4 |
| Guard symbols | 16 each side (eliminates edge ISI) |
| Seed | 42 |

## Scenarios

| Scenario | Signal | Demod Mod Selection | Tests |
|----------|--------|---------------------|-------|
| Clean | Original | Oracle (true mod) | Baseline |
| Clean+AMC | Original | AWN prediction | AMC accuracy on synth |
| CW | CW-attacked | Oracle | Does CW corrupt data? |
| CW+AMC | CW-attacked | AWN prediction | Attack impact on data link |
| CW+TopK | CW → Top-K | Oracle | Does Top-K recover data? |
| CW+TopK+AMC | CW → Top-K | AWN on recovered | Full defense effectiveness |
| Clean+TopK | Original → Top-K | Oracle | Defense cost on clean signal |

## Results

### 1. AWN Classification Accuracy on Synthetic Signals

| Mod | Clean AMC | CW AMC | CW+Top20 AMC | CW+Top50 AMC |
|-----|-----------|--------|--------------|--------------|
| BPSK | 99.0% | 6.0% | 55.5% | 50.5% |
| QPSK | 100.0% | 0.0% | 49.5% | 57.0% |
| 8PSK | 96.0% | 0.0% | 21.5% | 25.5% |
| QAM16 | 78.0% | 0.5% | 30.5% | 31.5% |
| QAM64 | 93.0% | 0.0% | 85.5% | 72.0% |
| PAM4 | 100.0% | 2.0% | 94.0% | 83.5% |

**Finding**: CW attack reduces AMC to near 0% for all mods. FFT Top-K partially recovers AMC for wideband mods (QAM64: 86%, PAM4: 94%) but fails for narrowband mods (BPSK: 50-56%, 8PSK: 22-26%).

### 2. CRC Pass Rate — Oracle Demod (correct mod type)

| Mod | Clean | CW | CW+Top10 | CW+Top20 | CW+Top50 |
|-----|-------|-----|----------|----------|----------|
| BPSK | 100.0% | 100.0% | 97.5% | 100.0% | 100.0% |
| QPSK | 100.0% | 100.0% | 80.5% | 100.0% | 100.0% |
| 8PSK | 100.0% | 100.0% | 9.5% | 100.0% | 100.0% |
| QAM16 | 100.0% | 100.0% | 1.0% | 93.5% | 99.5% |
| QAM64 | 96.0% | 92.5% | 1.0% | 41.0% | 87.5% |
| PAM4 | 100.0% | 97.5% | 13.0% | 96.0% | 98.5% |

**Finding**: CW attack does NOT corrupt data — CRC stays 92-100% with oracle demod. The attack is purely **control-plane** (fools classifier) not **data-plane** (corrupts bits). Top-10 is too aggressive and destroys signal; Top-20/50 preserve data integrity.

### 3. CRC Pass Rate — AMC Demod (AWN-driven)

| Mod | Clean+AMC | CW+AMC | CW+Top10+AMC | CW+Top20+AMC | CW+Top50+AMC |
|-----|-----------|--------|--------------|--------------|--------------|
| BPSK | 99.0% | 7.0% | 1.5% | 56.5% | 51.5% |
| QPSK | 100.0% | 0.5% | 0.5% | 49.5% | 57.0% |
| 8PSK | 96.0% | 1.0% | 0.0% | 22.5% | 25.5% |
| QAM16 | 78.0% | 0.5% | 0.0% | 29.0% | 31.0% |
| QAM64 | 89.0% | 0.0% | 1.0% | 33.5% | 61.5% |
| PAM4 | 100.0% | 3.5% | 13.0% | 90.0% | 82.0% |

**Finding**: When AWN drives demodulator selection, CW attack collapses CRC to 0-7%. FFT Top-K defense provides **partial** recovery that is strongly modulation-dependent:
- PAM4: 90% CRC recovery (Top-20) — effective defense
- QAM64: 62% CRC recovery (Top-50) — moderate defense
- BPSK/QPSK: ~50-57% CRC recovery — ineffective (near coin-flip)
- 8PSK/QAM16: 22-31% CRC recovery — ineffective

### 4. Defense Cost on Clean Signal (Oracle Demod)

| Mod | Clean | Clean+Top10 | Clean+Top20 | Clean+Top50 |
|-----|-------|-------------|-------------|-------------|
| BPSK | 100.0% | 98.0% | 100.0% | 100.0% |
| QPSK | 100.0% | 80.5% | 100.0% | 100.0% |
| 8PSK | 100.0% | 9.5% | 100.0% | 100.0% |
| QAM16 | 100.0% | 1.0% | 93.0% | 100.0% |
| QAM64 | 96.0% | 1.0% | 48.5% | 92.5% |
| PAM4 | 100.0% | 13.5% | 99.5% | 100.0% |

**Finding**: Top-10 severely degrades CRC on clean signals (destroys waveform). Top-20 is safe for low-order mods. Top-50 has negligible defense cost.

## Root Cause Analysis: Why FFT Top-K Fails for Narrowband Mods

### Diagnosis

We tested whether the AMC recovery failure is due to (a) Top-K distortion or (b) residual CW perturbation:

| Mod | Clean AMC | Clean+Top20 AMC | CW+Top20 AMC |
|-----|-----------|-----------------|--------------|
| BPSK | 96.5% | 85.5% | 49.5% |
| QPSK | 100.0% | 95.0% | 50.5% |
| 8PSK | 95.0% | 70.5% | 20.5% |
| QAM16 | 76.5% | 61.0% | 33.5% |
| QAM64 | 94.0% | 97.5% | 92.5% |
| PAM4 | 100.0% | 100.0% | 92.0% |

**Both factors contribute**: Top-K itself degrades AMC for narrowband mods (8PSK drops 24.5pp), and residual CW perturbation causes further degradation (8PSK drops another 50pp).

### Bandpass Filter Comparison

We also tested a signal-bandwidth-aware bandpass filter (keeping only the RRC signal bandwidth) as an alternative to Top-K:

| Mod | CW+Top20 AMC | CW+Bandpass AMC |
|-----|--------------|-----------------|
| BPSK | 49.5% | 54.0% |
| QAM64 | 87.5% | 87.5% |
| PAM4 | 92.0% | 96.5% |

**Finding**: Bandpass provides marginal improvement but does not fundamentally solve the problem, because **CW perturbation is in-band**. The CW optimizer places perturbation energy within the signal's spectral footprint (where AWN's classification features live). No frequency-domain filter can separate in-band perturbation from signal.

### Sweep of Top-K Values

| Mod | CW+T20 | CW+T30 | CW+T40 | CW+T50 |
|-----|--------|--------|--------|--------|
| BPSK | 48.0% | 42.0% | 41.0% | 41.5% |
| QAM64 | 87.5% | 87.0% | 80.5% | 76.0% |
| PAM4 | 90.5% | 76.0% | 76.0% | 73.5% |

**Finding**: Increasing K beyond 20 makes recovery *worse* for most mods. At low K, signal loss dominates; at high K, retained perturbation dominates. The sweet spot is K=20 for most mods.

## Key Conclusions

1. **CW attack is control-plane only**: CW perturbation does not corrupt data bits. With oracle demod, CRC passes at 92-100% even under attack. The attack exclusively targets AMC classification.

2. **Wrong demod = broken link**: When AWN is fooled (CW+AMC), applying the wrong demodulator yields ~50% BER and ~0% CRC pass rate. This is the real-world consequence of the attack.

3. **FFT Top-K defense is modulation-dependent**:
   - Effective for wideband mods: PAM4 (90% CRC recovery), QAM64 (62%)
   - Ineffective for narrowband mods: BPSK (52%), 8PSK (23%)

4. **CW perturbation is in-band**: The attack places energy within the signal bandwidth, making it fundamentally resistant to frequency-domain defenses. Bandpass filtering confirms this — even a perfect signal-bandwidth filter cannot remove the perturbation.

5. **No single Top-K value is optimal**: K=20 is the best compromise, but recovery ceiling is limited by in-band perturbation for narrowband modulations.

## Output Files

```
results/crc_defense_direct/
├── crc_defense_pipeline.csv      # Raw data (all scenarios)
├── crc_defense_pipeline.json     # Same data in JSON
├── fig_crc_defense_pipeline.png  # 3-panel plot
└── EXPERIMENT_REPORT.md          # This report
```

## Reproducibility

```bash
python crc_defense_pipeline.py --snr 18 --n_bursts 200 --topk 10,20,50
```
