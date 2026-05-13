---
estimated_steps: 62
estimated_files: 2
skills_used: []
---

# T03: Comprehensive randomized verification covering all AWN GEMM shapes

## Overview

Expand `test_tiled_systolic.py` to comprehensively verify tiled_gemm_s8 across all 10 AWN GEMM shapes from refmodel.py, all boundary conditions from the real AWN pipeline, and 50+ randomized tests with arbitrary dimensions. Create a repo-root wrapper for GSD verification gate compatibility. This is the final acceptance gate for S02.

Requirements addressed: R007 (bit-exact match verified across all shapes).

## Expand `awn_fpga/sw/test_tiled_systolic.py`

### All 10 AWN GEMM shapes (from awn_fpga/sw/refmodel.py)

These are the exact GEMM dimensions from every layer in the AWN inference pipeline:

| Layer | M | K | N | Notes |
|-------|---|---|---|-------|
| conv1 | 64 | 14 | 128 | First convolution |
| conv2 | 64 | 320 | 128 | Largest GEMM (acceptance criterion) |
| U.conv1 | 64 | 192 | 66 | N-boundary: 66 mod 16 = 2 |
| U.conv2 | 64 | 192 | 64 | Cleanly aligned |
| P.conv1 | 64 | 192 | 66 | Same shape as U.conv1 |
| P.conv2 | 64 | 192 | 64 | Same shape as U.conv2 |
| SE lin0 | 32 | 128 | 1 | Skinny N (N_tile=1) |
| SE lin3 | 128 | 32 | 1 | Large M (16 M-tiles), skinny N |
| fc.0 | 320 | 128 | 1 | Very large M (40 M-tiles) |
| fc.2 | 11 | 320 | 1 | M-boundary: 11 mod 8 = 3 |

Test each shape with and without bias = 20 tests.

### Additional boundary cases

| M | K | N | Why |
|---|---|---|-----|
| 11 | 320 | 1 | M and N boundary simultaneously |
| 64 | 192 | 66 | N-boundary (already in AWN shapes, ensures coverage) |
| 32 | 128 | 1 | SE linear shape |
| 320 | 128 | 1 | Large M (40 tiles), skinny N |
| 1 | 1 | 1 | Minimum possible dimensions |
| 8 | 1 | 16 | Minimum K, full tile |
| 9 | 1 | 17 | Both M and N just over tile boundary |

Each with and without bias = 14 tests.

### 50 randomized tests

M drawn from {1, 2, 3, 7, 8, 9, 11, 16, 32, 48, 64}, K drawn from {1, 4, 14, 32, 64, 128, 192, 320}, N drawn from {1, 2, 8, 15, 16, 17, 32, 64, 66, 128}. Random bias (50% probability). Use `np.random.seed(42)` for reproducibility.

### Cross-check against gemm_s8

For the (64, 320, 128) shape (conv2 — the S02 acceptance criterion), also run through `tb_gemm_s8.v` testbench and verify the output is byte-identical to the tiled version. This confirms the tiling engine matches the behavioral reference exactly.

To do this: compile and run gemm_s8 testbench with the same input data, compare outputs with `np.array_equal`. The gemm_s8 compile command is:
```
iverilog -g2005-sv -o build/sim_gemm_s8 tb/tb_gemm_s8.v rtl/gemm_s8.v
```
(Read `awn_fpga/rtl/gemm_s8.v` and `awn_fpga/tb/tb_gemm_s8.v` if you need to confirm the testbench interface — it uses the same plusargs pattern.)

Note: if `tb_gemm_s8.v` doesn't exist yet, skip this cross-check and verify against numpy only — numpy is the authoritative reference.

### Expected total: 84+ tests

Final output line: `ALL TILED SYSTOLIC TESTS PASSED (N tests)`

## Create `sw/test_tiled_systolic.py` (repo-root wrapper)

Same pattern as `sw/test_systolic.py` — a thin Python script at the repo root that delegates to `awn_fpga/sw/test_tiled_systolic.py` using absolute paths. This ensures `python sw/test_tiled_systolic.py` works from the repo root (needed for GSD verification gate which splits `cd && python` commands).

Read `sw/test_systolic.py` for the exact wrapper pattern to follow.

## Test script structure

Organize tests into labeled groups in the output:
```
--- AWN GEMM shapes ---
PASS T01: M=64 K=14 N=128 bias=False  (conv1)
PASS T02: M=64 K=14 N=128 bias=True   (conv1)
...
--- Boundary cases ---
PASS T21: M=11 K=320 N=1 bias=False
...
--- Randomized ---
PASS T35: M=48 K=64 N=17 bias=True
...
--- Cross-check vs gemm_s8 ---
PASS: (64,320,128) tiled output matches gemm_s8 byte-for-byte

ALL TILED SYSTOLIC TESTS PASSED (87 tests)
```

## Inputs

- ``awn_fpga/sw/test_tiled_systolic.py` — T01/T02 output. Expand with AWN shapes, boundary cases, randomized tests, and cross-check.`
- ``awn_fpga/sw/iohex.py` — hex I/O helpers used by test script. Unchanged.`
- ``awn_fpga/rtl/gemm_s8.v` — behavioral reference for cross-check compilation.`
- ``sw/test_systolic.py` — repo-root wrapper pattern to copy for sw/test_tiled_systolic.py.`

## Expected Output

- ``awn_fpga/sw/test_tiled_systolic.py` — comprehensive test script with 80+ tests: 20 AWN shapes, 14 boundary cases, 50 randomized, cross-check vs gemm_s8`
- ``sw/test_tiled_systolic.py` — repo-root wrapper delegating to awn_fpga/sw/test_tiled_systolic.py`

## Verification

cd awn_fpga && python sw/test_tiled_systolic.py 2>&1 | tail -1 | grep -q 'ALL TILED SYSTOLIC TESTS PASSED' && python /home/nigel/opensource/adversarial-rf/sw/test_tiled_systolic.py 2>&1 | tail -1 | grep -q 'ALL TILED SYSTOLIC TESTS PASSED'
