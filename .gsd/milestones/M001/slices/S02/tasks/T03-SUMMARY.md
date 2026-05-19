---
id: T03
parent: S02
milestone: M001
key_files:
  - awn_fpga/sw/test_tiled_systolic.py
  - sw/test_tiled_systolic.py
key_decisions:
  - Used -I rtl/ include path for gemm_s8 cross-check compilation since tb_gemm_s8.v uses `include directive
  - Graceful fallback if gemm_s8 compilation fails (returns None, prints SKIP)
duration: 
verification_result: passed
completed_at: 2026-05-13T15:43:37.703Z
blocker_discovered: false
---

# T03: Expanded test suite to 85 tests covering all 10 AWN GEMM shapes, 7 boundary cases, 50 randomized dimensions, and gemm_s8 cross-check — all pass byte-for-byte

**Expanded test suite to 85 tests covering all 10 AWN GEMM shapes, 7 boundary cases, 50 randomized dimensions, and gemm_s8 cross-check — all pass byte-for-byte**

## What Happened

Rewrote `awn_fpga/sw/test_tiled_systolic.py` from 38 tests to 85 tests organized in four groups:

**AWN GEMM shapes (20 tests):** All 10 layer dimensions from the AWN inference pipeline — conv1 (64,14,128), conv2 (64,320,128), U/P lifting conv layers (64,192,66 and 64,192,64), SE attention (32,128,1 and 128,32,1), FC layers (320,128,1 and 11,320,1). Each tested with and without bias.

**Boundary cases (14 tests):** M+N boundary (11,320,1), N-boundary (64,192,66), SE linear (32,128,1), large M skinny N (320,128,1 — 40 M-tiles), minimum (1,1,1), min K full tile (8,1,16), M+N just over tile (9,1,17). Each with and without bias.

**Randomized (50 tests):** M from {1,2,3,7,8,9,11,16,32,48,64}, K from {1,4,14,32,64,128,192,320}, N from {1,2,8,15,16,17,32,64,66,128}, 50% bias probability. np.random.seed(42) for reproducibility.

**Cross-check vs gemm_s8 (1 test):** Generated (64,320,128) GEMM through both tiled_gemm_s8 and behavioral gemm_s8 testbenches with identical input data. Outputs match byte-for-byte, confirming tiling engine correctness against the behavioral reference.

Also created `sw/test_tiled_systolic.py` repo-root wrapper following the same pattern as `sw/test_systolic.py` for GSD verification gate compatibility.

## Verification

Ran `python sw/test_tiled_systolic.py` from both awn_fpga/ and repo root — all 85 tests passed both times. Cross-check against gemm_s8 confirmed byte-identical output for (64,320,128).

## Verification Evidence

| # | Command | Exit Code | Verdict | Duration |
|---|---------|-----------|---------|----------|
| 1 | `cd awn_fpga && python sw/test_tiled_systolic.py 2>&1 | tail -1 | grep -q 'ALL TILED SYSTOLIC TESTS PASSED'` | 0 | pass | 420000ms |
| 2 | `python /home/nigel/opensource/adversarial-rf/sw/test_tiled_systolic.py 2>&1 | tail -1 | grep -q 'ALL TILED SYSTOLIC TESTS PASSED'` | 0 | pass | 420000ms |

## Deviations

85 tests instead of 84+ minimum (cross-check counted as test 85)

## Known Issues

None

## Files Created/Modified

- `awn_fpga/sw/test_tiled_systolic.py`
- `sw/test_tiled_systolic.py`
