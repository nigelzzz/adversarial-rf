---
id: T02
parent: S03
milestone: M001
key_files:
  - awn_fpga/rtl/im2col_addr_gen.v
  - awn_fpga/sw/test_im2col.py
key_decisions:
  - Used DIM_W+1 bit signed pos_w wire for correct negative padding offset detection
  - Introduced neg_pos_w intermediate wire to avoid iverilog bit-select limitation on negation results
  - two_win_m2 = 2*W_in - 2 precomputed at start for reflection address formula
duration: 
verification_result: passed
completed_at: 2026-05-14T04:08:41.114Z
blocker_discovered: false
---

# T02: Added zero-padding (Config B) and reflection-padding (Config C) modes, all 33 tests passing

**Added zero-padding (Config B) and reflection-padding (Config C) modes, all 33 tests passing**

## What Happened

Extended im2col_addr_gen.v with cfg_pad_mode register (00=none, 01=zero, 10=reflect). Zero-padding emits zero_flag=1 when pos_w < 0 or pos_w >= W_in. Reflection-padding computes mirrored address: pos < 0 → -pos, pos >= W_in → 2*(W_in-1) - pos. Used signed arithmetic (DIM_W+1 bit pos_w) for correct negative boundary detection. Introduced intermediate wires (neg_pos_w) to work around iverilog limitation on bit-selecting negation results. W_out computed internally: with padding → win + 2*padl - kw + 1, without → win - kw + 1. Config B (zero-pad, k=5, cin=64, win=128) verified with 17 tests including the largest AWN im2col (40,960 elements). Config C (reflect-pad, k=3, cin=64, win=64) verified with 16 tests.

## Verification

Ran `python sw/test_im2col.py --config B` (17 PASS) and `python sw/test_im2col.py --config C` (16 PASS)

## Verification Evidence

| # | Command | Exit Code | Verdict | Duration |
|---|---------|-----------|---------|----------|
| 1 | `python sw/test_im2col.py --config B` | 0 | PASS | 20000ms |
| 2 | `python sw/test_im2col.py --config C` | 0 | PASS | 18000ms |

## Deviations

None.

## Known Issues

None.

## Files Created/Modified

- `awn_fpga/rtl/im2col_addr_gen.v`
- `awn_fpga/sw/test_im2col.py`
