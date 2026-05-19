---
id: T02
parent: S02
milestone: M001
key_files:
  - awn_fpga/rtl/bram_feeder_b.v
  - awn_fpga/rtl/tiled_gemm_s8.v
key_decisions:
  - Double-buffering was implemented alongside tiling in T01 rather than separately, since the BRAM-based B-feeding was simpler to get right from the start than implementing direct buffer reads first and then refactoring
duration: 
verification_result: passed
completed_at: 2026-05-13T15:01:38.058Z
blocker_discovered: false
---

# T02: BRAM double-buffering for B-matrix weights already implemented in T01 — verified all 38 tests pass with two bram_feeder_b banks, LOAD_B0 state, and concurrent loading during COMPUTE

**BRAM double-buffering for B-matrix weights already implemented in T01 — verified all 38 tests pass with two bram_feeder_b banks, LOAD_B0 state, and concurrent loading during COMPUTE**

## What Happened

T02's scope was already fully implemented during T01. The T01 implementation included the complete double-buffering infrastructure rather than using direct buffer reads:

**bram_feeder_b.v** — Rewritten with wide write port (all 16 columns per cycle via `wr_data_flat[COLS*8-1:0]`) and per-column parallel read ports (`rd_rows_flat[COLS*AW-1:0]` gives each column its own row address). Internal storage uses 2D `mem[col][row]` with genvar-indexed writes and reads — no variable-index 2D array issues in iverilog since genvar constants index the first dimension.

**tiled_gemm_s8.v** — Has all T02 features:
1. Six-state FSM with S_LOAD_B0 added between IDLE and COMPUTE
2. Two bram_feeder_b bank instances (bank0, bank1) with shared write data bus and separate write enables
3. `active_bank` register selects which bank feeds PEs during COMPUTE
4. Concurrent B-tile loading: during COMPUTE, the idle bank (~active_bank) receives the next tile's weights via `loading` flag and `load_n_base`
5. Bank swap in S_NEXT: `active_bank <= ~active_bank` when advancing to next N-tile
6. New M-row triggers S_LOAD_B0 (reload first B-tile into bank0)
7. Write data assembled combinationally from b_buf with N-boundary zero-padding

No code changes were needed — verification confirms existing implementation is correct.

## Verification

Ran `python sw/test_tiled_systolic.py` in awn_fpga/ — all 38 tests passed including single-tile, multi-tile aligned (64×320×128), M-boundary (11×320×1), N-boundary (64×192×66), both-boundary (11×192×66), and 20 randomized tests. All results match numpy byte-for-byte.

## Verification Evidence

| # | Command | Exit Code | Verdict | Duration |
|---|---------|-----------|---------|----------|
| 1 | `cd awn_fpga && python sw/test_tiled_systolic.py 2>&1 | tail -1 | grep -q 'ALL TILED SYSTOLIC TESTS PASSED'` | 0 | pass | 180000ms |

## Deviations

No code changes needed — T02 scope was already implemented during T01

## Known Issues

None

## Files Created/Modified

- `awn_fpga/rtl/bram_feeder_b.v`
- `awn_fpga/rtl/tiled_gemm_s8.v`
