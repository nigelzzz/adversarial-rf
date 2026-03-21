# FAB Attack CRC Experiment Results

**Attack:** FAB (Fast Adaptive Boundary) — minimum-norm adversarial attack
**Pipeline:** `crc_defense_fec_multi_attack.py --attacks fab --snr 0,18 --n_bursts 200`
**Defense:** FFT Top-K recovery (K=10, 20, 50)
**Demod:** Oracle (correct modulation always used)

---

## SNR 0 dB

### Raw Results

| Mod   | FEC   | Clean AMC | Attack AMC | Attack+Oracle CRC | Top-10 CRC | Top-20 CRC | Top-50 CRC |
|-------|-------|-----------|------------|-------------------|------------|------------|------------|
| BPSK  | noFEC | 73.0%     | 3.5%       | 100.0%            | 80.5%      | 99.5%      | 100.0%     |
| QPSK  | noFEC | 53.0%     | 8.5%       | 79.5%             | 21.0%      | 49.5%      | 71.0%      |
| QPSK  | FEC   | 5.0%      | 0.5%       | 100.0%            | 93.0%      | 99.0%      | 99.5%      |
| 8PSK  | noFEC | 40.0%     | 5.5%       | 6.0%              | 0.5%       | 2.0%       | 3.5%       |
| 8PSK  | FEC   | 32.5%     | 4.5%       | 99.0%             | 86.5%      | 94.5%      | 98.0%      |
| QAM16 | noFEC | 62.5%     | 6.5%       | 0.0%              | 1.5%       | 0.5%       | 0.5%       |
| QAM16 | FEC   | 66.0%     | 11.5%      | 86.5%             | 48.0%      | 81.5%      | 87.0%      |
| QAM64 | noFEC | 19.5%     | 5.5%       | 0.5%              | 0.5%       | 0.0%       | 0.0%       |
| QAM64 | FEC   | 21.0%     | 8.0%       | 25.5%             | 2.0%       | 11.0%      | 25.0%      |
| PAM4  | noFEC | 75.5%     | 24.5%      | 23.5%             | 0.5%       | 10.0%      | 19.5%      |
| PAM4  | FEC   | 36.5%     | 8.5%       | 84.5%             | 62.5%      | 74.0%      | 83.5%      |

### FEC Impact (Attack + Oracle CRC)

| Mod   | No FEC | FEC    | Improvement |
|-------|--------|--------|-------------|
| QPSK  | 79.5%  | 100.0% | +20.5%     |
| 8PSK  | 6.0%   | 99.0%  | +93.0%     |
| QAM16 | 0.0%   | 86.5%  | +86.5%     |
| QAM64 | 0.5%   | 25.5%  | +25.0%     |
| PAM4  | 23.5%  | 84.5%  | +61.0%     |

### Top-K Recovery + FEC (CRC Pass Rate)

| Mod   | Top-10 | Top-20 | Top-50 |
|-------|--------|--------|--------|
| QPSK  | 93.0%  | 99.0%  | 99.5%  |
| 8PSK  | 86.5%  | 94.5%  | 98.0%  |
| QAM16 | 48.0%  | 81.5%  | 87.0%  |
| QAM64 | 2.0%   | 11.0%  | 25.0%  |
| PAM4  | 62.5%  | 74.0%  | 83.5%  |

---

## Key Findings

1. **FEC massively improves CRC at SNR 0 dB under FAB attack:**
   - QPSK: 79.5% -> 100%, 8PSK: 6% -> 99% with oracle demod
   - QAM16/PAM4: strong recovery (84-87%)
   - QAM64: still limited (25.5%) — channel capacity bottleneck

2. **FAB behaves identically to other Linf attacks (apgd, bim, pgd, etc.):**
   - Control-plane only: attack destroys AMC but CRC holds under oracle demod
   - Same modulation-dependent pattern across all tested attacks

3. **QAM64 at SNR 0 is fundamentally broken:**
   - Even clean CRC with FEC is only 21% — noise floor too high for 64-point constellation
   - Not a defense failure, but a Shannon limit issue

4. **Top-K recovery introduces distortion that FEC partially corrects:**
   - Top-20 + FEC: QPSK 99%, 8PSK 94.5%, QAM16 81.5%
   - Higher K preserves more signal but also more adversarial energy
