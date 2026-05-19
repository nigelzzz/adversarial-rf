---
id: S03
parent: M001
milestone: M001
provides:
  - im2col_addr_gen.v: standalone address generator FSM with (addr, zero_flag, valid, done) output interface
  - Test infrastructure: test_im2col.py with compile_sim(), im2col_1d_ref(), im2col_2d_ref(), run_im2col_hw()
requires:
  []
affects:
  - S04 pipeline controller will instantiate im2col_addr_gen to feed A-side of tiled_gemm_s8 for convolution layers
key_files:
  - awn_fpga/rtl/im2col_addr_gen.v
  - awn_fpga/tb/tb_im2col_addr_gen.v
  - awn_fpga/sw/test_im2col.py
  - sw/test_im2col.py
key_decisions:
  - Unified single FSM for all 4 configs via pad_mode register instead of separate modules
  - Running base-address accumulators instead of multipliers for BRAM address computation
  - Split (h_out, w_out) counters instead of flat t_out to support 2D with H_out > 1
  - Random test vectors instead of fixed golden vectors from quant.npz for broader coverage
  - Signed DIM_W+1 bit pos_w for correct negative padding boundary detection
patterns_established:
  - (none)
observability_surfaces:
  - none
drill_down_paths:
  []
duration: ""
verification_result: passed
completed_at: 2026-05-14T04:09:44.299Z
blocker_discovered: false
---

# S03: Hardware im2col Unit

**Unified parameterized im2col address generator FSM supporting all 4 AWN kernel configurations with 68 tests passing byte-exact against numpy golden references**

## What Happened

Built im2col_addr_gen.v as a single parameterized FSM that handles all 4 AWN convolution configurations: Config D (no-padding, 1D k=3), Config B (zero-padding, 1D k=5), Config C (reflection-padding, 1D k=3), and Config A (2D k=2x7). The FSM uses nested counters (kw → kh → cin → w_out → h_out) with running base-address accumulators (cin_base, kh_base, h_base) to avoid multipliers. Padding logic uses signed DIM_W+1 bit arithmetic for correct negative boundary detection. A cfg_pad_mode register selects between no-padding (00), zero-padding (01), and reflection-padding (10).

The 2D support required splitting a flat t_out counter into separate (h_out, w_out) counters — the original design failed for H_out > 1 because the BRAM address needs (h_out+kh)*W_in + w_out + kw, not just kh*W_in + t_out + kw. The h_base running offset (incremented by W_in each time h_out advances) solved this cleanly.

Test infrastructure includes tb_im2col_addr_gen.v (parameterized via $value$plusargs, loads fmap hex, collects output stream) and test_im2col.py (numpy golden references, iverilog compilation, 68 randomized tests across all configs). Column-major output ordering (flatten order='F') matches the FSM's natural iteration pattern for feeding tiled_gemm_s8.

## Verification

68 tests across all 4 configs pass byte-exact: Config D (19 tests, no-padding, cin={1..64}, win={1..128}, kw={1..7}), Config B (17 tests, zero-padding, kw={3..7}), Config C (16 tests, reflection-padding, kw=3), Config A (16 tests, 2D, kh={1..3}, kw={1..7}, H_out={1..4}). Both awn_fpga/sw/test_im2col.py and repo-root sw/test_im2col.py wrappers verified.

## Requirements Advanced

None.

## Requirements Validated

None.

## New Requirements Surfaced

None.

## Requirements Invalidated or Re-scoped

None.

## Operational Readiness

None.

## Deviations

None.

## Known Limitations

Padding only applies to W dimension (sufficient for AWN where H dimension is always pre-padded or trivial). Reflection padding assumes pad_left equals pad_right. AW=14 limits BRAM to 16K entries.

## Follow-ups

S04 (pipeline controller sequences im2col + tiled_gemm for all 38 AWN ops), S05 (AXI interface)

## Files Created/Modified

None.
