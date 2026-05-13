---
id: T03
parent: S01
milestone: M001
key_files:
  - awn_fpga/tb/tb_systolic_mesh_s8.v
  - awn_fpga/sw/test_systolic.py
key_decisions:
  - Testbench mirrors tb_gemm_s8.v exactly (plusargs, readmemh, fwrite pattern) for consistency with existing infrastructure
  - Python script uses np.random.randint(-128, 128) for full signed int8 range coverage including -128
duration: 
verification_result: passed
completed_at: 2026-05-13T10:40:39.346Z
blocker_discovered: false
---

# T03: Add mesh testbench (tb_systolic_mesh_s8.v) and randomized Python verification (test_systolic.py) — 64 tests all bit-exact against numpy

**Add mesh testbench (tb_systolic_mesh_s8.v) and randomized Python verification (test_systolic.py) — 64 tests all bit-exact against numpy**

## What Happened

Created `tb_systolic_mesh_s8.v` following the exact `tb_gemm_s8.v` pattern: same plusargs interface (M, K, N, bias, a, b, bi, out), same `$readmemh` loading into DUT.a_buf/b_buf/bias_buf, same `$fwrite` output with `%08x` format masked by `32'hffffffff`, same reset sequence and cycle-count timeout. The testbench instantiates `systolic_mesh_s8` with `DIM_W=16`.

Created `test_systolic.py` which compiles the simulation once via iverilog, then runs 64 tests (14 deterministic + 50 randomized):
- Deterministic sizes: (1,1,1), (2,2,2), (4,4,4), (8,16,16), (8,1,16), (1,64,1), (8,320,16) — each with and without bias
- Randomized: M∈{1,2,4,7,8}, K∈{1,4,16,32,64,128,192,320}, N∈{1,2,8,15,16}, bias 50% probability
- Uses `iohex.py` helpers exclusively for hex I/O (no custom hex code)
- Reference computed as `A.astype(np.int32) @ B.astype(np.int32)` with optional bias broadcast
- Full int8 range via `np.random.randint(-128, 128)`

All 64 tests pass bit-exact. The (8,320,16) test exercises the largest K dimension from the AWN pipeline.

Note: The T02 verification gate failures (`rtl/bram_feeder_a.v: No such file`) are a CWD issue — the gate runs `cd awn_fpga` as a separate command, but subsequent commands execute from the repo root. The RTL files exist correctly at `awn_fpga/rtl/` and compile cleanly. T03's verification uses `cd awn_fpga && python sw/test_systolic.py` with `&&` chaining to avoid this.

## Verification

Ran `cd awn_fpga && python sw/test_systolic.py` which compiled the simulation and executed 64 tests (14 deterministic + 50 randomized). All tests printed PASS with final output "ALL SYSTOLIC TESTS PASSED (64 tests)" and exit code 0. Also verified all RTL modules compile independently with iverilog.

## Verification Evidence

| # | Command | Exit Code | Verdict | Duration |
|---|---------|-----------|---------|----------|
| 1 | `iverilog -g2005-sv -s tb_systolic_mesh_s8 -o build/chk_tb_mesh tb/tb_systolic_mesh_s8.v rtl/systolic_mesh_s8.v rtl/pe_s8.v` | 0 | pass | 500ms |
| 2 | `cd awn_fpga && python sw/test_systolic.py` | 0 | pass | 120000ms |
| 3 | `iverilog -g2005-sv -s bram_feeder_a -o /dev/null awn_fpga/rtl/bram_feeder_a.v` | 0 | pass | 200ms |
| 4 | `iverilog -g2005-sv -s bram_feeder_b -o /dev/null awn_fpga/rtl/bram_feeder_b.v` | 0 | pass | 200ms |
| 5 | `iverilog -g2005-sv -s systolic_mesh_s8 -o /dev/null awn_fpga/rtl/systolic_mesh_s8.v awn_fpga/rtl/pe_s8.v` | 0 | pass | 300ms |

## Deviations

None. Implementation followed the task plan exactly.

## Known Issues

T02 verification gate failures are a CWD issue: the gate splits 'cd awn_fpga' and subsequent iverilog commands into separate shell invocations, so the cd doesn't persist. RTL files at awn_fpga/rtl/ are correct and compile cleanly. Future verification commands should use '&&' chaining (e.g., 'cd awn_fpga && iverilog ...').

## Files Created/Modified

- `awn_fpga/tb/tb_systolic_mesh_s8.v`
- `awn_fpga/sw/test_systolic.py`
