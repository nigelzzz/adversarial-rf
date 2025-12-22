# Paper CW Attack Replication Notes

## What the Paper Shows

**Figure**: I/Q Constellation Plots
- **(a) Intact Data**: Clean I/Q constellations for BPSK, QPSK, QAM16, QAM64
  - BPSK: 2 clusters (horizontal line)
  - QPSK: 4 clusters (corners of square)
  - QAM16: 16 clusters (4×4 grid)
  - QAM64: 64 clusters (8×8 grid)

- **(b) CW Attack**: Distorted constellations
  - Structure is disrupted
  - Points are scattered
  - Clusters overlap or shift

## Our Replication Approach

### Key Improvements to Match Paper:

1. **Use torchattacks CW API** ✓
   - Official CW implementation
   - Parameters: c=1.0, steps=100, lr=1e-2

2. **Select Same Modulations** ✓
   - BPSK, QPSK, QAM16, QAM64
   - High SNR (18 dB) for clear constellations

3. **Generate I/Q Scatter Plots** ✓
   - Extract I and Q channels
   - Flatten all time samples
   - Plot as scatter with transparency

4. **Add Recovery Column** ✓
   - Paper shows (a) Intact, (b) CW Attack
   - We add (c) After Recovery (detector-gated FFT Top-K)
   - This shows the effectiveness of the defense

5. **Use Detector-Gated Approach** ✓
   - AWN_All.py pattern with KL divergence threshold
   - Only apply FFT Top-K to recoverable samples
   - Reject highly corrupted samples

### Script Features:

- **500 samples per modulation** for good constellation density
- **Batch processing** of CW attack generation
- **Accuracy metrics** for each stage
- **Detection statistics** (how many samples defended)
- **High-resolution output** (300 DPI PNG + PDF)

### Expected Results:

Based on our analysis, we expect:

**For BPSK/QPSK (Simple Modulations)**:
- Clean: High accuracy (~95-98%)
- CW Attack: Moderate to low accuracy (~20-60%)
- Recovery: Some improvement if detector gates well

**For QAM16/QAM64 (Complex Modulations)**:
- Clean: Good accuracy (~85-95%)
- CW Attack: Low accuracy (~10-40%)
- Recovery: Limited improvement (perturbations overlap with signal)

### Differences from Paper:

1. **We add recovery column** - Paper only shows attack
2. **We use detector gating** - More sophisticated than simple FFT Top-K
3. **We show quantitative results** - Accuracy percentages, not just visual
4. **We test recovery effectiveness** - Paper focuses on attack visualization

## Why This Matches the Paper Better

1. **Visual Format**: I/Q constellation plots are the standard way to show RF modulation
2. **Same Modulations**: BPSK, QPSK, QAM16, QAM64 cover simple to complex
3. **High SNR**: SNR=18 gives clear constellations like the paper
4. **Proper Scale**: Fixed axis limits (-0.02 to 0.02) for consistency

## Running the Experiment

```bash
python replicate_paper_cw.py
```

**Output**:
- `paper_cw_replication.png` - High-res figure (300 DPI)
- `paper_cw_replication.pdf` - Publication-quality PDF
- `paper_cw_replication_results.json` - Numerical results

**Time**: ~15-20 minutes (4 modulations × 500 samples × 100 CW steps)
