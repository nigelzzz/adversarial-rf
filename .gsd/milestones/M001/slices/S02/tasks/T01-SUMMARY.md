---
id: T01
parent: S02
milestone: M001
key_files:
  - awn_fpga/rtl/tiled_gemm_s8.v
  - awn_fpga/tb/tb_tiled_gemm_s8.v
  - awn_fpga/sw/test_tiled_systolic.py
key_decisions:
  - Used flat 1D pe_acc array (not 2D) matching systolic_mesh_s8.v for drain indexing consistency
  - N_tile/M_tile recomputation in S_NEXT uses pre-update base values since non-blocking assignments are simultaneous
duration: 
verification_result: passed
completed_at: 2026-05-13T11:21:28.433Z
blocker_discovered: false
---

# T01: Add tiled_gemm_s8.v with 5-state tile-sequencing FSM, boundary zero-padding, and full testbench — all 38 tests pass byte-for-byte vs numpy

**Add tiled_gemm_s8.v with 5-state tile-sequencing FSM, boundary zero-padding, and full testbench — all 38 tests pass byte-for-byte vs numpy**

## What Happened

Created three files implementing the tiling engine that decomposes arbitrary M×K×N GEMMs into ceil(M/8)×ceil(N/16) tiles processed sequentially through an inline 8×16 PE grid.

**tiled_gemm_s8.v** — 5-state FSM (IDLE→COMPUTE→DRAIN→NEXT→DONE) with tile-offset address generation. The A-feed and B-feed generate blocks add `m_base`/`n_base` offsets to global buffer addresses and include boundary checks (`gi < M_tile`, `gj < N_tile`) that zero-pad out-of-range PE rows/columns. The PE grid is copied verbatim from systolic_mesh_s8.v. Key correctness details: c_buf row stride uses global N_reg (not N_tile), bias index uses `m_base + drain_m`, tile counts use ceiling division `(M+7)>>3`, and acc_clear asserts on cycle 0 of every tile.

**tb_tiled_gemm_s8.v** — Testbench copied from tb_systolic_mesh_s8.v with module name changed to tiled_gemm_s8 and timeout increased to 200M cycles for large multi-tile GEMMs.

**test_tiled_systolic.py** — 38 test cases: 8 single-tile (matching systolic_mesh_s8 range), 4 multi-tile aligned (16×32×32, 64×320×128), 2 M-boundary (11×320×1), 2 N-boundary (64×192×66), 2 both-boundary (11×192×66), and 20 randomized with M∈{1..64}, K∈{1..320}, N∈{1..128}, each compared against `A.int32 @ B.int32 + bias`.

## Verification

Ran `python sw/test_tiled_systolic.py` — all 38 tests passed (single-tile, multi-tile aligned, M-boundary, N-boundary, both-boundary, and 20 randomized). Slice verification command `tail -1 | grep -q 'ALL TILED SYSTOLIC TESTS PASSED'` also passed.

## Verification Evidence

| # | Command | Exit Code | Verdict | Duration |
|---|---------|-----------|---------|----------|
| 1 | `cd awn_fpga && python sw/test_tiled_systolic.py 2>&1 | tail -1 | grep -q 'ALL TILED SYSTOLIC TESTS PASSED'` | 0 | ✅ pass | 540000ms |

## Deviations

None

## Known Issues

None

## Files Created/Modified

- `awn_fpga/rtl/tiled_gemm_s8.v`
- `awn_fpga/tb/tb_tiled_gemm_s8.v`
- `awn_fpga/sw/test_tiled_systolic.py`
