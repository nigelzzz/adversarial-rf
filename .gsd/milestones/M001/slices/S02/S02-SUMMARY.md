---
id: S02
parent: M001
milestone: M001
provides:
  - tiled_gemm_s8: drop-in replacement for gemm_s8 with tiling and double-buffering
  - bram_feeder_b: wide-write per-column-read BRAM for weight storage
requires:
  - slice: S01
    provides: pe_s8 and systolic_mesh_s8 PE grid pattern
affects:
  []
key_files:
  - awn_fpga/rtl/tiled_gemm_s8.v
  - awn_fpga/rtl/bram_feeder_b.v
  - awn_fpga/tb/tb_tiled_gemm_s8.v
  - awn_fpga/sw/test_tiled_systolic.py
  - sw/test_tiled_systolic.py
key_decisions:
  - Implemented double-buffering in T01 alongside tiling rather than as separate T02 refactor
  - Used flat packed buses for bram_feeder_b ports to avoid iverilog unpacked array port issues
  - Used -I rtl/ include path for gemm_s8 cross-check compilation
patterns_established:
  - BRAM double-buffering pattern: LOAD_B0 primes bank0, COMPUTE reads active bank while loading idle bank, NEXT swaps banks
  - Flat packed bus pattern for multi-port BRAM: wr_data_flat[COLS*8-1:0] and rd_rows_flat[COLS*AW-1:0] avoid iverilog unpacked array port issues
observability_surfaces:
  - none
drill_down_paths:
  []
duration: ""
verification_result: passed
completed_at: 2026-05-13T15:44:36.092Z
blocker_discovered: false
---

# S02: Tile Sequencer FSM + Weight Double-Buffering

**tiled_gemm_s8 decomposes arbitrary M×K×N GEMMs into ceil(M/8)×ceil(N/16) tiles with BRAM double-buffering — 85 tests pass byte-for-byte including all 10 AWN layer shapes and gemm_s8 cross-check**

## What Happened

S02 built the tiling engine that makes the 8×16 PE mesh handle arbitrary GEMM dimensions, with weight double-buffering to hide B-tile load latency.

**T01** created tiled_gemm_s8.v with a 6-state FSM (IDLE→LOAD_B0→COMPUTE→DRAIN→NEXT→DONE), tile-offset address generation in A/B-feed generate blocks, and boundary zero-padding for non-aligned tile dimensions. The implementation included BRAM double-buffering from the start (planned as T02) since the BRAM-based B-feeding was simpler to get right than direct buffer reads followed by a refactor. Also rewrote bram_feeder_b.v with wide write port (all 16 columns per cycle) and per-column parallel read ports. Initial test suite had 38 tests.

**T02** was verified as already complete — all double-buffering infrastructure (two bram_feeder_b banks, LOAD_B0 state, concurrent loading during COMPUTE, active_bank swap in NEXT) was in place from T01. 38 tests confirmed correctness.

**T03** expanded the test suite to 85 tests across four groups:
- 20 AWN GEMM shapes (all 10 layers × with/without bias) including conv2 (64,320,128), lifting layers (64,192,66), SE attention (32/128,128/32,1), FC (320,128,1 and 11,320,1)
- 14 boundary cases covering minimum dimensions, min-K, and M/N just over tile boundary
- 50 randomized tests with diverse M/K/N combinations and 50% bias probability
- 1 cross-check against behavioral gemm_s8 testbench confirming byte-identical output for (64,320,128)

## Verification

Ran `python sw/test_tiled_systolic.py` from both awn_fpga/ and repo root — all 85 tests passed including gemm_s8 cross-check. Covers all 10 AWN inference GEMM shapes, boundary cases, and 50 randomized dimensions.

## Requirements Advanced

- R002 — Tile sequencer FSM drives 8×16 PE mesh through ceil(M/8)×ceil(N/16) tiles with weight double-buffering via two bram_feeder_b banks
- R007 — 85 tests verify bit-exact match against numpy for all 10 AWN layer GEMM shapes plus boundary and randomized cases
- R010 — Two bram_feeder_b banks instantiated in tiled_gemm_s8 with wide write and per-column parallel read
- R017 — Boundary zero-padding verified for M-boundary (11 mod 8 = 3), N-boundary (66 mod 16 = 2), and combined cases

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

A-matrix still reads directly from a_buf (no BRAM feeder for A). S03 will add im2col feeding to the A side.

## Follow-ups

S03: hardware im2col feeds A-matrix to tiled_gemm. S04: pipeline controller sequences tiled_gemm for all 38 AWN ops. S05: AXI interface for PS-PL data transfer.

## Files Created/Modified

- `awn_fpga/rtl/tiled_gemm_s8.v` — 6-state tile sequencer FSM with BRAM double-buffering, boundary zero-padding
- `awn_fpga/rtl/bram_feeder_b.v` — Rewritten with wide write port and per-column parallel read ports
- `awn_fpga/tb/tb_tiled_gemm_s8.v` — Testbench for tiled_gemm_s8 with 200M cycle timeout
- `awn_fpga/sw/test_tiled_systolic.py` — 85-test verification suite covering AWN shapes, boundary, randomized, and gemm_s8 cross-check
- `sw/test_tiled_systolic.py` — Repo-root wrapper for GSD verification gate compatibility
