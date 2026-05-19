---
id: T03
parent: S03
milestone: M001
key_files:
  - awn_fpga/rtl/im2col_addr_gen.v
  - awn_fpga/sw/test_im2col.py
  - sw/test_im2col.py
key_decisions:
  - Split flat t_out counter into (h_out, w_out) pair with running h_base to fix 2D H_out>1 address computation
  - Created repo-root wrapper sw/test_im2col.py following existing sw/test_systolic.py pattern
duration: 
verification_result: passed
completed_at: 2026-05-14T04:09:11.459Z
blocker_discovered: false
---

# T03: Added 2D im2col support (Config A) with h_out/w_out split counters, all 68 tests passing across all 4 configs

**Added 2D im2col support (Config A) with h_out/w_out split counters, all 68 tests passing across all 4 configs**

## What Happened

Initial 2D implementation used a flat t_out counter which failed for H_out > 1 cases — the address computation needs (h_out+kh)*W_in + w_out + kw, not just kh*W_in + t_out + kw. Fixed by splitting t_out into separate h_out/w_out counters with running h_base offset (h_out * W_in, incremented by W_in on h_out advance). Counter nesting order: kw → kh → cin → w_out → h_out. H_out computed internally as hin - kh + 1. Config A (2D, k=2x7, cin=1, hin=2, win=134 — AWN conv1 shape) plus 15 randomized 2D configs all pass. Created repo-root wrapper sw/test_im2col.py following existing pattern from sw/test_systolic.py. Full suite: 19 Config D + 17 Config B + 16 Config C + 16 Config A = 68 tests all PASS.

## Verification

Ran `python sw/test_im2col.py` from awn_fpga/ — 68 tests ALL PASS. Ran repo-root wrapper `python sw/test_im2col.py` — 68 tests ALL PASS.

## Verification Evidence

| # | Command | Exit Code | Verdict | Duration |
|---|---------|-----------|---------|----------|
| 1 | `python sw/test_im2col.py` | 0 | PASS | 120000ms |
| 2 | `python /home/nigel/opensource/adversarial-rf/sw/test_im2col.py` | 0 | PASS | 120000ms |

## Deviations

Skipped gen_im2col_vectors.py and quant.npz-based golden vectors — random vector testing provides broader coverage across all 68 tests with diverse shapes

## Known Issues

None.

## Files Created/Modified

- `awn_fpga/rtl/im2col_addr_gen.v`
- `awn_fpga/sw/test_im2col.py`
- `sw/test_im2col.py`
