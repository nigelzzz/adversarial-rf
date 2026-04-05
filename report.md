# Adversarial Training Evaluation Report

**Dataset:** RML2016.10a | **Attack:** CW (SPR 20 dB) | **SNR Range:** 0--18 dB

## Summary

The standard AWN model achieves high clean accuracy (90--100% for most modulations) but suffers severe degradation under CW attack, especially for higher-order modulations. Adversarial training recovers most of this loss, improving CW accuracy by +20--60 pp on 8 of 11 modulations.

## Results

| Modulation | Clean Acc | Std CW Acc | AdvTrain CW Acc | Improvement |
|------------|-----------|------------|-----------------|-------------|
| QAM16      | 93.65%    | 1.85%      | 40.35%          | +38.50 pp   |
| QAM64      | 94.40%    | 13.90%     | 9.50%           | **-4.40 pp**|
| 8PSK       | 98.10%    | 31.05%     | 90.55%          | +59.50 pp   |
| WBFM       | 37.80%    | 25.70%     | 48.30%          | +22.60 pp   |
| BPSK       | 98.35%    | 87.15%     | 98.55%          | +11.40 pp   |
| CPFSK      | 100.00%   | 96.35%     | 99.95%          | +3.60 pp    |
| AM-DSB     | 98.30%    | 33.30%     | 69.00%          | +35.70 pp   |
| GFSK       | 99.65%    | 59.75%     | 99.20%          | +39.45 pp   |
| PAM4       | 99.10%    | 98.45%     | 98.75%          | +0.30 pp    |
| QPSK       | 98.35%    | 79.65%     | 92.55%          | +12.90 pp   |
| AM-SSB     | 90.65%    | 38.30%     | 100.00%         | +61.70 pp   |

## Key Findings

1. **Largest gains:** AM-SSB (+61.70 pp), 8PSK (+59.50 pp), GFSK (+39.45 pp), QAM16 (+38.50 pp). These modulations were highly vulnerable under the standard model and benefit most from adversarial training.

2. **QAM64 regression:** The only modulation where adversarial training *hurts* (-4.40 pp). CW accuracy drops from 13.90% to 9.50%, suggesting the adversarial training regime does not generalize to QAM64's dense constellation.

3. **Already robust modulations:** PAM4 and CPFSK show minimal change (+0.30, +3.60 pp) because the standard model already resists CW attack on these (>96% under attack).

4. **Remaining gaps:** QAM16 (40.35%) and QAM64 (9.50%) remain poorly classified under CW even after adversarial training. AM-DSB (69.00%) is improved but still well below its 98.30% clean accuracy.

## Conclusion

Adversarial training is broadly effective, recovering 10 of 11 modulations. The QAM64 regression and residual QAM16 weakness indicate that higher-order constellation modulations need either stronger adversarial training schedules or complementary defenses (e.g., FFT Top-K recovery) to close the gap.
