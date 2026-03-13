# RF Adversarial Paper Writer Memory

## Optimal Top-K Values Per Modulation (RML2016.10a, AWN classifier)

| Modulation | Best K | Mean Recovery Rate | Notes |
|------------|--------|-------------------|-------|
| QAM64 | Top-10 | 98% | Excellent, SNR-independent |
| PAM4 | Top-20 | 96-98% | Excellent, SNR-independent |
| AM-DSB | Top-20 | 92-94% | Excellent, CW barely works at high SNR |
| GFSK | Top-20 | 86-88% | Good, constant-envelope helps |
| QPSK | Top-20 | 78% mean, >90% at SNR>=8 | SNR-dependent |
| QAM16 | Top-20 | 48-53% | Poor, SNR-independent plateau |
| 8PSK | Top-20 | 33-34% | Failed, SNR-dependent but never >55% |
| QAM16 | Top-10 | 8.7% | Destroyed (= random) |

## Key Findings for Paper Narrative

1. **Defense-Attack Independence**: Recovery rates nearly identical between CW and EADEN (<5pp difference). Defense effectiveness is determined by signal spectral structure, not attack algorithm.
2. **Chicken-and-Egg Problem**: Adaptive K needs modulation knowledge, but AMC is compromised. Novel insight for NDSS.
3. **CW is control-plane only**: Confirmed in CRC defense report (SNR=18). CW fools AMC but doesn't corrupt data.
4. **EADEN stronger than CW**: EADEN achieves 0% on all mods; CW leaves residual accuracy for AM-DSB (65%), PAM4 (76.5%), QPSK (41%).
5. **SigGuard fixed-K critique**: No single K works. Top-10 destroys QAM16; Top-20 fails for 8PSK.

## Report Files

- CRC defense (prior): `results/crc_defense_direct/CRC_DEFENSE_EXPERIMENT_REPORT.md`
- IFFT Top-K evaluation: `results/crc_defense_fec/FEC_EXPERIMENT_REPORT.md`

## Attack Parameters Used

- CW: torchattacks, minmax box, kappa=1, steps=100, lr=0.001
- EADEN: torchattacks, minmax box, defaults
- 200 samples per (mod, SNR) cell

## Paper Structure Decisions

- Frame as security paper (control-plane attack on cognitive radio)
- Lead with chicken-and-egg paradox as novel contribution
- Use CRC defense report for "CW is control-plane" evidence
- Use IFFT report for "no universal K" evidence
- Propose adaptive defense as future work, not solved problem
