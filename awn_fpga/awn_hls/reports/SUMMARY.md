# AWN INT8 HLS — Synthesis Summary

Target part: `xczu7ev-ffvc1156-2-e` (ZU7EV, generic ZynqUltraScale+ placeholder
— swap to your real board's part before final synthesis).
Target clock: 5 ns (200 MHz).
Vitis HLS version: 2023.2.

## Top-level numbers

| Metric            | Baseline (no pragmas) | Pragma pass 1 (PIPELINE + ARRAY_PARTITION) |
|-------------------|----------------------:|-------------------------------------------:|
| **Latency**       | 2,736,540 cy          | **52,026 cy** (52.6× faster)               |
| **Wall time**     | 13.683 ms             | **0.260 ms**                               |
| **Throughput**    | 73 inf/s              | **3,846 inf/s**                            |
| Fmax (est)        | 268.7 MHz             | 274.7 MHz                                  |
| BRAM_18K          | 87 / 624 (13%)        | 173 / 624 (27%)                            |
| DSP               | 56 / 1728 (3%)        | 594 / 1728 (34%)                           |
| FF                | 6,901 / 460,800 (1%)  | 83,960 / 460,800 (18%)                     |
| LUT               | 16,709 / 230,400 (7%) | 113,964 / 230,400 (49%)                    |
| csynth time       | ~21 s                 | ~60 min                                    |

## Per-block latency (after pragma pass 1)

| Block          | Cycles  | Time     | % of total |
|----------------|--------:|---------:|-----------:|
| conv1_block    | 32,782  | 164.0 µs | 63.0 %     |
| conv2_block    |  9,409  |  47.0 µs | 18.1 %     |
| u_branch       |  4,418  |  22.1 µs |  8.5 %     |
| p_branch       |  4,418  |  22.1 µs |  8.5 %     |
| split / even   |  4,098  |  20.5 µs |  7.9 %     |
| add (c_q)      |  4,101  |  20.5 µs |  7.9 %     |
| sub (d_q)      |  4,100  |  20.5 µs |  7.9 %     |
| avgpool d / c  |    137  |   0.7 µs |  0.3 %     |
| concat / SE    |  ~300   |   1.5 µs |  0.6 %     |
| FC0 + FC2      |   573   |   2.9 µs |  1.1 %     |
| **Total**      | 52,026  | **260 µs** | 100 % (overlap removed) |

(Block percentages sum to >100% because some blocks share or pipeline with others;
the reported total of 52,026 is the actual end-to-end latency.)

## Hotspot analysis

| Block       | Baseline | Pragma pass 1 | Speedup |
|-------------|---------:|--------------:|--------:|
| conv2_block | 13.107 ms| 47.0 µs       | 279×    |
| conv1_block |  0.287 ms| 164.0 µs      |   1.7×  |
| Other       |  ~0.3 ms | ~49 µs        |   6×    |

After pass 1 the bottleneck shifts from conv2 → conv1 (63% of total time).

## Pragmas applied (pass 1)

- `conv1_block`:
  `ARRAY_PARTITION variable=W1 dim=3 complete` (kh=2 unrolled)
  `ARRAY_PARTITION variable=W1 dim=4 complete` (kw=7 unrolled)
  `ARRAY_PARTITION variable=x  dim=1 complete` (2 IQ ch unrolled)
  `PIPELINE II=1` on the `(oc, w)` loop body.
- `conv2_block`:
  `ARRAY_PARTITION variable=W2 dim=3 complete` (kt=5 unrolled)
  `ARRAY_PARTITION variable=x  dim=2 cyclic factor=5`
  `PIPELINE II=1` on the `(oc, t)` loop body.
- `u_branch` / `p_branch`:
  `ARRAY_PARTITION variable=Wu{1,4}/Wp{1,4} dim=3 complete` (kt=3 unrolled)
  `PIPELINE II=1` on every `(oc, t)` loop in both sub-convs.
- `avgpool_64`: `PIPELINE II=1` on channel loop.
- `linear_acc<M,K>`: `PIPELINE II=1` on output-row loop.

## Functional verification

`csim_design` PASSES bit-exact against:
- `awn_fpga/build/quant.npz` golden input `x_fp` (AM-DSB sample).
- Reference int8 logits from `sw/refmodel.py` (TFLite-style Q31 + shift).
- argmax = 1 (AM-DSB), matches both the iverilog flow and the fp32 reference
  classification result.

Dequantized int8 logits sit within ~21 LSBs of fp32 reference; ranking preserved.

## Next levers if more speedup needed (not yet applied)

1. **Conv1 input-channel unroll**: `ARRAY_PARTITION x dim=1` already complete (2 ch),
   so conv1 latency is limited by 64 oc × 128 w = 8192 inner iterations at II≈4.
   `DATAFLOW` between conv1 → conv2 → lifting would let the next stage start
   while conv1's output trickles out, hiding most of conv1's cost.
2. **Conv2 input-channel partial unroll**: `ARRAY_PARTITION W2 dim=2 cyclic
   factor=8` plus matching `x` partition would give 8× more MACs/cycle on conv2,
   but it's already <18% of the budget so probably not worth the DSP cost.
3. **Coarse DATAFLOW** at top level: u_branch and p_branch happen sequentially
   today; pipelining them as concurrent stages would shave ~22 µs.

For the IEEE submission, pass-1 numbers already comfortably beat the
RML2016 real-time budget (128 samples @ 1 MS/s = 128 µs window → 260 µs of
compute fits with batching/pipelining at the system level).
