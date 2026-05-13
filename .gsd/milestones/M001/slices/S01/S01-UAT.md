# S01: Systolic Array PE + 8x16 Mesh — UAT

**Milestone:** M001
**Written:** 2026-05-13T10:44:47.754Z

## UAT: S01 — Systolic Array PE + 8x16 Mesh

### Preconditions
- iverilog installed and on PATH (`iverilog -V` succeeds)
- Python 3.6+ with numpy installed
- Working directory: repo root (`/home/nigel/opensource/adversarial-rf`)

### Test 1: PE Unit Tests (T01 deliverable)
**Steps:**
1. `cd awn_fpga && mkdir -p build`
2. `iverilog -g2005-sv -o build/sim_pe_s8 tb/tb_pe_s8.v rtl/pe_s8.v`
3. `vvp build/sim_pe_s8`

**Expected:** Output shows PASS for all 9 tests, ending with `ALL PE TESTS PASSED`. Exit code 0.

**Edge cases verified:**
- Test 7: -128 × -128 = 16384 (signed extremes, positive result)
- Test 8: 16384 + (-128×127) = 128 (signed cross product)
- Test 9: 256 × 16384 = 4,194,304 (accumulation stress, fits int32)
- Test 6: en=0 freezes acc, a_out, b_out even with changing inputs

### Test 2: BRAM Feeder Compilation (T02 deliverable)
**Steps:**
1. `cd awn_fpga`
2. `iverilog -g2005-sv -s bram_feeder_a -o /dev/null rtl/bram_feeder_a.v`
3. `iverilog -g2005-sv -s bram_feeder_b -o /dev/null rtl/bram_feeder_b.v`

**Expected:** Both commands exit 0 with no errors or warnings.

### Test 3: Systolic Mesh Compilation (T02 deliverable)
**Steps:**
1. `cd awn_fpga && mkdir -p build`
2. `iverilog -g2005-sv -s systolic_mesh_s8 -o build/chk_mesh rtl/systolic_mesh_s8.v rtl/pe_s8.v`

**Expected:** Exit 0. The mesh instantiates 128 pe_s8 instances via generate blocks.

### Test 4: Randomized Matrix Verification (T03 deliverable)
**Steps:**
1. `cd awn_fpga && python sw/test_systolic.py`

**Expected:** 64 tests print PASS, ending with `ALL SYSTOLIC TESTS PASSED (64 tests)`. Exit code 0.

**Key test sizes:**
- T01-T02: (1,1,1) — degenerate single multiply, no accumulation
- T07-T08: (8,16,16) — full mesh utilization
- T09-T10: (8,1,16) — K=1, no accumulation needed
- T11-T12: (1,64,1) — column vector, 15 of 16 columns idle
- T13-T14: (8,320,16) — max K dimension from AWN pipeline (conv2), K+PM+PN-2=342 compute cycles
- T15-T64: randomized M∈{1,2,4,7,8}, K∈{1..320}, N∈{1..16}, bias 50%

### Test 5: Repo-Root Wrapper (verification gate fix)
**Steps:**
1. From repo root: `python sw/test_systolic.py`

**Expected:** Same output as Test 4 — wrapper delegates to awn_fpga/sw/test_systolic.py. Exit code 0.

### Test 6: Interface Compatibility with gemm_s8.v
**Steps:**
1. Verify systolic_mesh_s8.v has matching parameters: `grep -c 'parameter.*A_LEN\|parameter.*B_LEN\|parameter.*C_LEN\|parameter.*BIAS_LEN\|parameter.*DIM_W' awn_fpga/rtl/systolic_mesh_s8.v`
2. Verify buffer names: `grep -c 'a_buf\|b_buf\|bias_buf\|c_buf' awn_fpga/rtl/systolic_mesh_s8.v`

**Expected:** Step 1 returns 5 (all parameters present). Step 2 returns multiple matches confirming buffer arrays exist with exact names for testbench hierarchy access.
