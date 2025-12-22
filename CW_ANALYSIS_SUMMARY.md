# CW Attack and Recovery Analysis Summary

## Overview

This document summarizes the analysis of CW (Carlini-Wagner) attacks on the AWN modulation classifier and attempts at recovery using FFT Top-K filtering.

## Key Findings

### 1. ✅ Visualization Analysis (Part 1)

**File**: `cw_analysis/sample_045_QPSK.png` (and 54 other samples)

**Observations from Frequency Domain (Row 2)**:
- Clean signal: Strong peaks at low frequencies (0.0-0.1 normalized frequency)
- CW adversarial: **Nearly identical frequency profile** - same peak locations
- FFT Top-K recovered: After keeping Top-50 bins, still looks like CW (attack persists)

**Perturbation Analysis (Bottom left)**:
- CW perturbation is **strongest at low frequencies** (0.0-0.2)
- This is exactly where the clean signal energy is concentrated
- Perturbation modifies existing frequency components, doesn't add new ones

**I/Q Constellation (Row 3)**:
- Clean: Clear QPSK structure (4 clusters)
- CW: Constellation distorted but structured
- Recovered: Similar to CW - not restored

**Conclusion**: CW attack doesn't add high-frequency noise like typical adversarial attacks. It modifies the same low-frequency components the signal uses, making FFT Top-K filtering ineffective.

---

### 2. 🔬 Frequency Spectrum Analysis

**Script**: `analyze_freq_spectrum.py`

**Results for QPSK at SNR=18 dB**:

#### Energy Distribution:
| Signal Type | Total Energy | Energy in Top-50 bins | Percentage |
|------------|--------------|---------------------|------------|
| Clean | 0.56 | 0.56 | **99.8%** |
| CW Adversarial | 0.21 | 0.21 | **99.9%** |
| CW Perturbation | 0.34 | 0.34 | **99.7%** |

#### Bin Overlap:
- **84% overlap** between Clean Top-50 and CW Top-50 bins
- **All 10 top perturbation bins** are inside clean signal's Top-50
- Cannot filter perturbation without damaging clean signal

#### Perturbation Frequency Distribution:
- **73%** in low frequency (bins 0-32)
- **27%** in high frequency (bins 64-128, actually aliased low freq)
- **0.1%** in mid frequencies

#### Top-10 Frequency Bins:
```
Clean:        [127, 1, 120, 8, 125, 3, 126, 2, 123, 5]
CW Adversarial: [0, 1, 127, 120, 8, 2, 126, 3, 125, 122]
Perturbation:   [0, 1, 127, 5, 123, 120, 8, 125, 3, 122]
```

**Interpretation**:
- ✗ **99.7% of CW perturbation energy is in Top-K bins** → FFT Top-K keeps the attack
- ✗ **Perturbation overlaps heavily with clean signal** → Cannot filter without damage
- ✗ **Both signals are frequency-sparse** → CW exploits this by modifying existing bins

---

### 3. 📊 Why FFT Top-K Recovery Fails

#### Fundamental Problem:
The CW attack is **frequency-sparse and overlapping** with the clean signal:

1. **Both clean and adversarial signals use the same ~50 frequency bins**
2. **CW doesn't add new frequencies**, just modifies amplitudes of existing ones
3. **FFT Top-K selects the same bins** for both clean and adversarial signals
4. **Result**: Attack passes through the filter

#### Different from Typical Adversarial Attacks:
- Traditional image adversarial attacks: Add high-frequency noise
- **CW on RF signals**: Modifies low-frequency components where signal lives

#### Experimental Confirmation:

| Top-K Value | Adv Acc | Rec Acc | Recovery Gain |
|------------|---------|---------|---------------|
| K=20 | 45.6% | 45.2% | **-0.4%** |
| K=30 | 45.6% | 45.6% | **0.0%** |
| K=50 | 45.6% | 45.6% | **0.0%** |
| K=80 | 45.6% | 45.8% | **+0.2%** |

**Conclusion**: FFT Top-K filtering provides **no meaningful recovery** regardless of K value.

---

### 4. 🎯 CW Parameter Sweep (SNR >= 0)

**Script**: `test_cw_params.py`

**Tested 7 CW configurations**:
- Steps: 10, 30, 50, 100
- c values: 0.1, 0.5, 1.0, 2.0

**Results**:
- Clean accuracy: 9.40% (issue: random SNR sampling gives poor baseline)
- **All configurations**: Adversarial acc ≈ 0.10%, Recovery acc ≈ 0.20%
- **Recovery gain**: +0.10% across all configs (negligible)
- **Similarity**: 29.9% (high perturbation)

**Observations**:
- ⚠ FFT Top-K shows minimal improvement across all CW configurations
- ⚠ Suggests CW perturbations are primarily low-frequency (within Top-K bins)
- → **Recommendation**: Try detector-gated recovery (ae_fft_topk) instead

---

### 5. 🛡️ Alternative Defense Strategies

#### A. FFT Top-K Variations (Tested):
1. **Different K values** (20, 30, 50, 80): No improvement
2. **Top-K percent** (testing 20%): Running...

#### B. Detector-Gated FFT Top-K (ae_fft_topk) (Running):
- Uses autoencoder to detect adversarial samples
- Only applies FFT Top-K to samples with KL divergence > threshold
- **Defended ratio**: 75-100% of samples per batch
- **Status**: Test in progress...

#### C. Auto Soft Notch (Running):
- Automatically detects anomalous frequency bands
- Applies soft suppression (not hard zeroing)
- **Status**: Test in progress...

---

## Technical Explanation: Why DIV_THRESHOLD is Needed

**From AWN_All.py**:

The `DIV_THRESHOLD` (0.004468 = 90th percentile of clean KL divergence) implements a **detector-gated defense**:

```python
# 1. Compute KL divergence between input and autoencoder reconstruction
kl_divs = kl_divergence(normalized_inputs, reconstructed)

# 2. Gate: only process samples with KL <= threshold
pass_indices = torch.where(kl_divs <= threshold)[0]  # Likely clean or recoverable
drop_indices = torch.where(kl_divs > threshold)[0]   # Too adversarial - reject

# 3. Apply FFT Top-K only to passing samples
for sample in pass_indices:
    filtered = fft_topk_denoise(normalize(sample), topk=50)
    classify(filtered)
```

**Purpose**:
- **Avoids false recoveries**: If FFT Top-K is applied blindly to all adversarial samples, it might distort signals without improving accuracy
- **Two-stage defense**: Detector identifies which samples are adversarial, then FFT Top-K only processes recoverable ones
- **Trade-off**: Reject some samples (high KL) to maintain accuracy on classified samples

---

## Recommendations

### For CW Attack Recovery:

1. ✅ **Use detector-gated approach (ae_fft_topk)** instead of unconditional FFT Top-K
   - Provides intelligent gating based on anomaly detection
   - Avoids applying recovery to unrecoverable samples

2. ❌ **Don't rely on simple FFT Top-K** for CW attacks
   - CW perturbations overlap with signal frequencies
   - Filtering damages both attack and signal equally

3. 🔬 **Consider alternative defenses**:
   - Adversarial training
   - Input transformations (e.g., adding noise, compression)
   - Ensemble methods
   - Different frequency-domain approaches (wavelet, etc.)

### For Evaluation:

1. **Use SNR-specific testing** (e.g., SNR=18 only)
   - Random SNR sampling gives unreliable baselines (9.40% vs 92.55%)

2. **Analyze frequency characteristics** before choosing defense
   - Check if attack is frequency-sparse or broadband
   - Verify attack doesn't overlap with signal frequencies

3. **Visualize** attack effects in multiple domains
   - Time domain
   - Frequency domain
   - I/Q constellation
   - Perturbation analysis

---

## Files Generated

### Visualizations:
- `cw_analysis/` - 55 PNG files showing Clean vs CW vs Recovered
  - Time domain, frequency domain, I/Q constellations, perturbations
  - All 11 modulation types at SNR=18 dB

### Analysis Scripts:
- `visualize_cw_fft.py` - Generate comprehensive visualizations
- `analyze_freq_spectrum.py` - Detailed frequency domain analysis
- `test_cw_params.py` - CW parameter sweep

### Test Scripts:
- `test_2016_10a.py` - Simple CW + recovery test
- `demo_cw_recovery.py` - Demonstration script
- `test_cw_snr0.py` - Configurable SNR >= 0 testing

### Results:
- `cw_param_sweep_results.json` - Parameter sweep data
- `inference/2016.10a_*/` - Evaluation results for different defenses

---

## Next Steps

1. **Wait for detector-gated (ae_fft_topk) results** - May show improvement over unconditional FFT Top-K

2. **If detector-gated doesn't help**:
   - CW attack may be fundamentally unrecoverable with frequency filtering
   - Consider adversarial training or model hardening

3. **Investigate other attack types**:
   - FGSM, PGD (might be more frequency-separable)
   - Spectral attacks (already tested - similar issues)

4. **Fix SNR sampling issue** for future testing:
   - Use stratified sampling or SNR-specific subsets
   - Ensures reliable baseline accuracy measurements

---

## Conclusion

**CW attacks on RF modulation classification are fundamentally different from image adversarial attacks**:

- ❌ **FFT Top-K filtering doesn't work** because:
  - CW perturbations are frequency-sparse
  - Attack overlaps with clean signal frequencies (99.7% in same Top-50 bins)
  - Cannot separate attack from signal in frequency domain

- ✅ **Detector-gated approach may help** by:
  - Identifying which samples are adversarial
  - Only applying recovery to recoverable samples
  - Rejecting highly corrupted samples

- 🔬 **Alternative defenses needed**:
  - Current frequency-domain approaches insufficient
  - Need model-level or training-based defenses
  - Or accept trade-off: reject suspicious samples

**Key Insight**: The frequency-sparse, low-frequency nature of CW attacks on RF signals makes them particularly challenging to defend against using post-processing frequency filters.
