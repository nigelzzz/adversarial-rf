---
id: S01
parent: M001
milestone: M001
provides:
  - pe_s8 module (int8 MAC PE with registered passthrough)
  - systolic_mesh_s8 module (8x16 output-stationary array, single-tile M≤8 N≤16 arbitrary K)
  - bram_feeder_a and bram_feeder_b standalone dual-port BRAM modules
  - Randomized verification infrastructure (test_systolic.py + tb_systolic_mesh_s8.v)
requires:
  []
affects:
  - S02: Tile Sequencer will wrap systolic_mesh_s8 with tiling FSM and integrate BRAM feeders for double-buffering
  - S04: Pipeline controller will use the mesh for all GEMM operations in the 38-op AWN inference
key_files:
  - awn_fpga/rtl/pe_s8.v
  - awn_fpga/rtl/systolic_mesh_s8.v
  - awn_fpga/rtl/bram_feeder_a.v
  - awn_fpga/rtl/bram_feeder_b.v
  - awn_fpga/tb/tb_pe_s8.v
  - awn_fpga/tb/tb_systolic_mesh_s8.v
  - awn_fpga/sw/test_systolic.py
  - sw/test_systolic.py
key_decisions:
  - Flat 1D pe_acc array instead of 2D for reliable iverilog variable-index reads during drain phase
  - Nested drain counters (drain_m, drain_n) instead of integer division for portable hardware mapping
  - BRAM feeders created as standalone modules not instantiated in mesh — S02 integrates for double-buffering
  - Global acc_clear pulse on cycle 0 for all PEs (safe because zero inputs produce zero products)
  - Repo-root sw/test_systolic.py wrapper to handle GSD gate CWD splitting
patterns_established:
  - Systolic mesh mirrors gemm_s8.v interface exactly (params, ports, buffer names) for drop-in testbench compatibility
  - Python randomized verification pattern: compile once, run N tests with tempdir hex vectors, assert bit-exact numpy match
  - Flat 1D arrays for any wire/reg that needs variable-index reads in iverilog
observability_surfaces:
  - none
drill_down_paths:
  []
duration: ""
verification_result: passed
completed_at: 2026-05-13T10:44:47.754Z
blocker_discovered: false
---

# S01: Systolic Array PE + 8x16 Mesh

**Built and verified an 8x16 output-stationary systolic array (128 PEs) with int8×int8 MAC, int32 accumulation, skewed feeding, and bias support — 64 randomized tests produce bit-exact int32 results matching numpy for all single-tile sizes (M≤8, N≤16, K up to 320).**

## What Happened

## What This Slice Delivered

Three tasks built the foundational systolic array infrastructure for the AWN FPGA accelerator:

### T01: PE Module (pe_s8.v) + Unit Testbench
Created the single processing element — the atomic building block of the systolic array. Each PE performs one int8×int8 multiply-accumulate per clock cycle with int32 accumulation, using explicit sign extension (`{{16{prod[15]}}, prod}`) matching the behavioral `gemm_s8.v` arithmetic exactly. The PE has registered passthrough (a_in→a_out, b_in→b_out with 1-cycle latency), active-low async reset, enable gating (en=0 freezes all state), and acc_clear (initializes accumulator to current product, not zero, saving one cycle per tile start).

A 9-case self-checking testbench covers: reset, single MAC, accumulation chain, passthrough latency, enable gate freeze, signed extremes (-128×-128=16384), signed cross products, and a 256-iteration stress test. One timing bug was found and fixed: the testbench originally used two `@(negedge clk)` waits per stimulus, causing double-accumulation.

### T02: BRAM Feeders + 8x16 Systolic Mesh (systolic_mesh_s8.v)
Created `bram_feeder_a.v` (dual-port, ROWS=8, DEPTH=512) and `bram_feeder_b.v` (dual-port, COLS=16, DEPTH=512) as standalone modules. These are not instantiated in the mesh for S01 — they exist for S02's double-buffering integration.

The main deliverable is `systolic_mesh_s8.v`: 128 pe_s8 instances in an 8×16 generate grid. The module interface matches `gemm_s8.v` exactly (same parameters, ports, and buffer array names: a_buf, b_buf, bias_buf, c_buf) so the existing testbench infrastructure works unchanged via hierarchy access (DUT.a_buf, etc.).

Key design decisions:
- **Flat 1D pe_acc array** (`pe_acc[0:PM*PN-1]`) instead of 2D — avoids iverilog issues with variable-index reads during the drain phase.
- **Skewed feeding via generate blocks**: row m feeds A[m, t-m] starting at cycle m; col n feeds B[t-n, n] starting at cycle n. Address computation mixes genvar constants with register values.
- **4-state FSM** (IDLE→COMPUTE→DRAIN→DONE): acc_clear=1 on first compute cycle only, then 0 for remaining K+PM+PN-2 cycles. Global acc_clear is safe because zero inputs produce zero products.
- **Nested drain counters** (drain_m, drain_n) instead of integer division for portable hardware mapping.

### T03: Mesh Testbench + Randomized Python Verification
Created `tb_systolic_mesh_s8.v` mirroring `tb_gemm_s8.v` exactly (same plusargs, $readmemh, $fwrite pattern). Created `test_systolic.py` which compiles the simulation once, then runs 64 tests:
- 14 deterministic: sizes (1,1,1) through (8,320,16), each with and without bias
- 50 randomized: M∈{1,2,4,7,8}, K∈{1..320}, N∈{1..16}, 50% bias probability
- All tests use `iohex.py` hex helpers and verify bit-exact match via `np.array_equal`

The (8,320,16) test exercises the largest K dimension in the AWN pipeline (conv2 GEMM).

### Verification Gate Fix
The GSD verification gate splits `&&`-chained commands into separate shell invocations, so `cd awn_fpga && python sw/test_systolic.py` failed because the `cd` didn't persist. Fixed by creating a thin wrapper at `sw/test_systolic.py` (repo root) that delegates to `awn_fpga/sw/test_systolic.py` using absolute paths.

## Verification

## Verification Results

All slice-level verification checks pass:

### 1. PE Unit Tests
```
cd awn_fpga && mkdir -p build && iverilog -g2005-sv -o build/sim_pe_s8 tb/tb_pe_s8.v rtl/pe_s8.v && vvp build/sim_pe_s8 | grep -q "ALL PE TESTS PASSED"
```
**Result:** PASS — all 9 test cases pass (reset, single MAC, accumulate, continue, passthrough, enable gate, signed min×min, signed cross, 256× stress)

### 2. RTL Compilation
```
cd awn_fpga && iverilog -g2005-sv -s bram_feeder_a -o /dev/null rtl/bram_feeder_a.v
cd awn_fpga && iverilog -g2005-sv -s bram_feeder_b -o /dev/null rtl/bram_feeder_b.v
cd awn_fpga && iverilog -g2005-sv -s systolic_mesh_s8 -o build/chk_mesh rtl/systolic_mesh_s8.v rtl/pe_s8.v
```
**Result:** ALL COMPILE OK — all three modules compile cleanly with iverilog -g2005-sv

### 3. Randomized Matrix Tests
```
cd awn_fpga && python sw/test_systolic.py
```
**Result:** ALL SYSTOLIC TESTS PASSED (64 tests) — 14 deterministic + 50 randomized, all bit-exact int32 match against numpy

### 4. Repo-Root Wrapper
```
python sw/test_systolic.py  (from repo root)
```
**Result:** ALL SYSTOLIC TESTS PASSED (64 tests) — wrapper correctly delegates to awn_fpga/sw/test_systolic.py

## Requirements Advanced

- R001 — PE and single-tile 8x16 mesh verified bit-exact; full tiling (S02) needed for complete R001 validation
- R010 — bram_feeder_a.v and bram_feeder_b.v created as standalone dual-port BRAM modules; integration in S02

## Requirements Validated

None.

## New Requirements Surfaced

None.

## Requirements Invalidated or Re-scoped

None.

## Operational Readiness

None.

## Deviations

Testbench timing fix in T01: original testbench used two @(negedge clk) waits per stimulus causing double-MAC. Fixed to single-edge timing. No impact on PE design — only testbench was affected.

## Known Limitations

Single-tile only: M must be ≤8, N must be ≤16. K is arbitrary but the full AWN pipeline has GEMM dimensions up to (64,192,128) which require tiling (S02). BRAM feeders exist but are not wired into the mesh yet.

## Follow-ups

S02 must integrate BRAM feeders into the mesh for weight double-buffering and add tiling FSM for GEMM dimensions exceeding 8×16. The systolic_mesh_s8 interface is ready for drop-in replacement of gemm_s8 in the pipeline controller (S04).

## Files Created/Modified

None.
