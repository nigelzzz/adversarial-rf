---
estimated_steps: 142
estimated_files: 3
skills_used: []
---

# T01: Create tiled_gemm_s8.v with direct buffer reads and testbench

## Overview

Create `tiled_gemm_s8.v` — a tiling engine that decomposes arbitrary M×K×N GEMMs into ceil(M/8) × ceil(N/16) tiles, processing each sequentially through an inline 8×16 PE grid. This version reads A and B directly from global buffer arrays (no BRAM double-buffering — added in T02) to establish tiling correctness in isolation. Also create the testbench and basic verification script.

The module must match `gemm_s8.v` output byte-for-byte for any M, K, N. This is the critical correctness foundation. Requirements addressed: R002 (tile sequencer FSM), R017 (boundary zero-padding).

## File 1: `awn_fpga/rtl/tiled_gemm_s8.v` (~200 lines)

### Module Interface (identical to systolic_mesh_s8.v and gemm_s8.v)

```verilog
module tiled_gemm_s8 #(
    parameter PM       = 8,
    parameter PN       = 16,
    parameter A_LEN    = 65536,
    parameter B_LEN    = 65536,
    parameter C_LEN    = 16384,
    parameter BIAS_LEN = 1024,
    parameter DIM_W    = 16
)(
    input                clk,
    input                rst_n,
    input                start,
    input  [DIM_W-1:0]   M_in, K_in, N_in,
    input                use_bias,
    output reg           done
);
```

### Internal Buffers (testbench loads via $readmemh)

```verilog
reg signed [7:0]  a_buf    [0:A_LEN-1];
reg signed [7:0]  b_buf    [0:B_LEN-1];
reg signed [31:0] bias_buf [0:BIAS_LEN-1];
reg signed [31:0] c_buf    [0:C_LEN-1];
```

Buffer naming MUST match gemm_s8.v exactly — testbench accesses via `DUT.a_buf` etc.

### FSM States (5 states, 3-bit register)

```verilog
localparam S_IDLE    = 3'd0;
localparam S_COMPUTE = 3'd1;
localparam S_DRAIN   = 3'd2;
localparam S_NEXT    = 3'd3;
localparam S_DONE    = 3'd4;
reg [2:0] state;
```

### Tiling Registers

```verilog
reg [DIM_W-1:0] M_reg, K_reg, N_reg;
reg             use_bias_reg;
reg [DIM_W-1:0] mt, nt;              // current tile indices
reg [DIM_W-1:0] mt_count, nt_count;  // total tile counts
reg [DIM_W-1:0] m_base, n_base;      // base offsets into global buffers
reg [DIM_W-1:0] M_tile, N_tile;      // actual tile dimensions (boundary tiles may be smaller)
reg [DIM_W-1:0] cycle_cnt;
reg [DIM_W-1:0] drain_m, drain_n;
reg pe_en, pe_acc_clear;
```

### PE Grid — Copy from systolic_mesh_s8.v

Read `awn_fpga/rtl/systolic_mesh_s8.v` lines 48-74 and copy the following exactly:
- `wire signed [31:0] pe_acc [0:PM*PN-1];` (flat 1D accumulator array)
- `wire signed [7:0] a_wire [0:PM-1][0:PN];` (horizontal activation wires)
- `wire signed [7:0] b_wire [0:PM][0:PN-1];` (vertical weight wires)
- The nested generate block (gi, gj) instantiating `pe_s8` with connections to a_wire, b_wire, pe_acc.

Copy these verbatim — the PE grid is identical to systolic_mesh_s8.

### A-Feeding Generate Block (MODIFIED from systolic_mesh_s8.v)

Each row gi feeds A[(m_base + gi), cycle_cnt - gi] from global a_buf:

```verilog
generate
    for (gi = 0; gi < PM; gi = gi + 1) begin : a_feed
        wire [DIM_W-1:0] a_offset = cycle_cnt - gi[DIM_W-1:0];
        wire [31:0]      a_addr   = (m_base + gi[DIM_W-1:0]) * K_reg + a_offset;
        wire             a_valid  = (state == S_COMPUTE) &&
                                    (cycle_cnt >= gi[DIM_W-1:0]) &&
                                    (a_offset < K_reg) &&
                                    (gi[DIM_W-1:0] < M_tile);  // BOUNDARY CHECK
        assign a_wire[gi][0] = a_valid ? a_buf[a_addr[15:0]] : 8'sd0;
    end
endgenerate
```

**Two critical differences from systolic_mesh_s8.v:** (1) `m_base` offset added to `a_addr`, (2) `gi < M_tile` boundary check ensures out-of-range rows feed zeros.

### B-Feeding Generate Block (MODIFIED from systolic_mesh_s8.v)

Each column gj feeds B[cycle_cnt - gj, n_base + gj] from global b_buf:

```verilog
generate
    for (gj = 0; gj < PN; gj = gj + 1) begin : b_feed
        wire [DIM_W-1:0] b_offset = cycle_cnt - gj[DIM_W-1:0];
        wire [31:0]      b_addr   = b_offset * N_reg + (n_base + gj[DIM_W-1:0]);
        wire             b_valid  = (state == S_COMPUTE) &&
                                    (cycle_cnt >= gj[DIM_W-1:0]) &&
                                    (b_offset < K_reg) &&
                                    (gj[DIM_W-1:0] < N_tile);  // BOUNDARY CHECK
        assign b_wire[0][gj] = b_valid ? b_buf[b_addr[15:0]] : 8'sd0;
    end
endgenerate
```

**Two critical differences:** (1) `n_base` offset in `b_addr`, (2) `gj < N_tile` boundary check.

### Compute Cycle Count

```verilog
wire [DIM_W-1:0] total_cycles = K_reg + PM[DIM_W-1:0] + PN[DIM_W-1:0] - 16'd2;
```

Same formula for ALL tiles regardless of M_tile/N_tile — boundary PEs get zero inputs, extra cycles are harmless.

### FSM Always Block

**S_IDLE:** On start: latch M/K/N/use_bias. Compute tile counts: `mt_count <= (M_in + 16'd7) >> 3; nt_count <= (N_in + 16'd15) >> 4;`. Initialize mt=0, nt=0, m_base=0, n_base=0. Set `M_tile <= (M_in >= PM[DIM_W-1:0]) ? PM[DIM_W-1:0] : M_in;` and same for N_tile. Set cycle_cnt=0, pe_en=1, pe_acc_clear=1. Transition to S_COMPUTE.

**S_COMPUTE:** On cycle_cnt==0: pe_acc_clear <= 0 (acc_clear was set high on entry — PEs capture first product). When cycle_cnt == total_cycles: pe_en <= 0, drain_m <= 0, drain_n <= 0, transition to S_DRAIN. Else: cycle_cnt++.

**S_DRAIN:** Write one output per cycle:
```verilog
c_buf[(m_base + drain_m) * N_reg + (n_base + drain_n)] <=
    pe_acc[drain_m[3:0]*PN + drain_n[4:0]] +
    (use_bias_reg ? bias_buf[m_base + drain_m] : 32'sd0);
```
Advance drain_n. When drain_n reaches N_tile-1: reset drain_n=0, check drain_m. When drain_m reaches M_tile-1: transition to S_NEXT. Else: drain_m++.

**S_NEXT:** Check if more tiles remain:
- If `nt + 1 < nt_count` (more N-tiles in same M-row): nt++, n_base += 16. Recompute N_tile: `N_tile <= ((N_reg - n_base - PN[DIM_W-1:0]) >= PN[DIM_W-1:0]) ? PN[DIM_W-1:0] : (N_reg - n_base - PN[DIM_W-1:0]);` (uses pre-update n_base since register updates are simultaneous). Set cycle_cnt=0, pe_en=1, pe_acc_clear=1, transition to S_COMPUTE.
- Else if `mt + 1 < mt_count` (more M-tile rows): mt++, nt=0, m_base += 8, n_base=0. Recompute M_tile similarly. Reset N_tile to min(PN, N_reg). Set cycle_cnt=0, pe_en=1, pe_acc_clear=1, transition to S_COMPUTE.
- Else: transition to S_DONE.

**S_DONE:** done <= 1. Transition to S_IDLE.

### CRITICAL PITFALLS (executor must check these)

1. **c_buf row stride uses N_reg (global N), NOT N_tile.** `c_buf[(m_base+drain_m) * N_reg + ...]` — using N_tile would scatter outputs to wrong positions.
2. **Bias index is bias_buf[m_base + drain_m], NOT bias_buf[drain_m].** Each M-tile uses a different bias slice.
3. **Tile count: `(M + 7) >> 3`, NOT `M >> 3`.** Truncating division misses the boundary tile.
4. **acc_clear must assert on cycle 0 of EVERY tile.** Set pe_acc_clear=1 in both IDLE→COMPUTE and NEXT→COMPUTE transitions. The PE's acc_clear sets acc=product (not zero), so the first MAC is captured correctly.
5. **In S_NEXT, M_tile/N_tile computation uses pre-update m_base/n_base** because all non-blocking assignments take effect simultaneously at the clock edge.

## File 2: `awn_fpga/tb/tb_tiled_gemm_s8.v` (~70 lines)

Copy `awn_fpga/tb/tb_systolic_mesh_s8.v` and make these changes:
1. Change module instantiation from `systolic_mesh_s8` to `tiled_gemm_s8`.
2. Increase cycle timeout to handle large multi-tile GEMMs. Use a generous timeout like `M*K*N*2 + 200000` or a fixed `200_000_000` cycles.
3. Everything else stays identical: same plusargs (+M, +K, +N, +bias, +a, +b, +bi, +out), same $readmemh pattern (`DUT.a_buf`, `DUT.b_buf`, etc.), same $fwrite output loop.

Read `awn_fpga/tb/tb_systolic_mesh_s8.v` first for the exact template to copy.

## File 3: `awn_fpga/sw/test_tiled_systolic.py` (~100 lines)

Copy the pattern from `awn_fpga/sw/test_systolic.py`. Key changes:

**Compile function:** Use `tiled_gemm_s8.v` and `pe_s8.v` (also include `bram_feeder_b.v` for T02 compatibility — it compiles cleanly even if not instantiated):
```python
iverilog -g2005-sv -o sim
    tb/tb_tiled_gemm_s8.v
    rtl/tiled_gemm_s8.v
    rtl/pe_s8.v
    rtl/bram_feeder_b.v
```

**Test cases:**
- Single-tile (same as systolic_mesh_s8 range): (1,1,1), (4,4,4), (8,16,16), (8,320,16) — each with and without bias
- Multi-tile aligned: (16,32,32), (64,320,128) — each with and without bias
- M-boundary: (11,320,1) — last M-tile has 3 rows — with and without bias
- N-boundary: (64,192,66) — last N-tile has 2 cols — with and without bias
- Both boundary: (11,192,66) — with and without bias
- 20 randomized: M from {1,2,4,7,8,9,11,16,32,64}, K from {1,4,16,32,64,128,192,320}, N from {1,2,8,15,16,17,32,64,66,128}, 50% bias probability

**Verification:** Each test compares hardware output against numpy: `C_ref = A.astype(np.int32) @ B.astype(np.int32)` plus bias if applicable. Assert `np.array_equal(C_ref, C_hw)`. Print PASS/FAIL per test, exit 0 only if all pass.

Read `awn_fpga/sw/test_systolic.py` and `awn_fpga/sw/iohex.py` for the exact run_gemm / test_one pattern to follow.

## Inputs

- ``awn_fpga/rtl/systolic_mesh_s8.v` — reference for PE grid generate block (lines 48-74), A-feeding (lines 76-87), B-feeding (lines 89-100), and FSM pattern (lines 102-170). Copy PE grid verbatim; modify feeding and FSM for tiling.`
- ``awn_fpga/rtl/pe_s8.v` — PE module instantiated by the generate block. Unchanged.`
- ``awn_fpga/rtl/bram_feeder_b.v` — included in compile for T02 compatibility. Not instantiated in T01.`
- ``awn_fpga/tb/tb_systolic_mesh_s8.v` — template for tb_tiled_gemm_s8.v testbench. Copy and change module name.`
- ``awn_fpga/sw/test_systolic.py` — template for test_tiled_systolic.py. Copy pattern, expand test dimensions.`
- ``awn_fpga/sw/iohex.py` — hex I/O helpers. Used unchanged by test script.`

## Expected Output

- ``awn_fpga/rtl/tiled_gemm_s8.v` — tiling engine with inline 8x16 PE grid, 5-state FSM, tile-offset address generation, boundary zero-padding`
- ``awn_fpga/tb/tb_tiled_gemm_s8.v` — testbench following tb_systolic_mesh_s8.v pattern with increased cycle timeout`
- ``awn_fpga/sw/test_tiled_systolic.py` — verification script with single-tile, multi-tile, boundary, and randomized tests`

## Verification

cd awn_fpga && python sw/test_tiled_systolic.py 2>&1 | tail -1 | grep -q 'ALL TILED SYSTOLIC TESTS PASSED'
