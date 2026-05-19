# S02: Tile Sequencer FSM + Weight Double-Buffering

**Goal:** Tile sequencer FSM drives the 8x16 PE mesh through arbitrary-size GEMMs with ceil(M/8) × ceil(N/16) tiling, boundary zero-padding for non-aligned dimensions, and weight double-buffering via two bram_feeder_b banks; output matches behavioral gemm_s8 and numpy byte-for-byte for all 10 AWN layer shapes including boundary cases (N=66, N=1, M=11).
**Demo:** Tile sequencer drives the 8x16 mesh through a full (64,320,128) GEMM with tiling, zero-padding boundary tiles, and weight double-buffering; output matches behavioral hex byte-for-byte

## Must-Haves

- 1. `cd awn_fpga && python sw/test_tiled_systolic.py` prints "ALL TILED SYSTOLIC TESTS PASSED" with 80+ tests.
- 2. All 10 AWN GEMM shapes tested (conv1 through fc.2) with and without bias.
- 3. Boundary cases verified: (11,320,1) M-boundary, (64,192,66) N-boundary, (320,128,1) large-M skinny-N, (32,128,1) SE-linear, (1,1,1) minimum.
- 4. 50+ randomized tests with M∈{1..64}, K∈{1..320}, N∈{1..128}.
- 5. (64,320,128) cross-checked against gemm_s8 testbench for byte-identical output.
- 6. Weight double-buffering via two bram_feeder_b banks integrated and verified (same functional output as direct-read version).

## Proof Level

- This slice proves: This slice proves: contract (bit-exact tiling against behavioral gemm_s8 for all AWN GEMM dimensions). Real runtime required: no (iverilog simulation only). Human/UAT required: no.

## Integration Closure

Upstream surfaces consumed: `awn_fpga/rtl/pe_s8.v` (unchanged PE module), `awn_fpga/rtl/systolic_mesh_s8.v` (reference for PE grid generate block — not modified), `awn_fpga/rtl/gemm_s8.v` (behavioral reference — not modified), `awn_fpga/sw/iohex.py` (hex I/O helpers — unchanged).
New wiring introduced: `tiled_gemm_s8.v` is a drop-in replacement for `gemm_s8.v` with identical port interface (M_in, K_in, N_in, use_bias, start, done) and identical buffer naming (a_buf, b_buf, bias_buf, c_buf). `bram_feeder_b.v` interface expanded for wide write + multi-read, instantiated inside tiled_gemm_s8.
What remains: S03 (hardware im2col feeds A-matrix to tiled_gemm), S04 (pipeline controller sequences tiled_gemm for all 38 AWN ops), S05 (AXI interface for PS-PL data transfer).

## Verification

- Not provided.

## Tasks

- [x] **T01: Create tiled_gemm_s8.v with direct buffer reads and testbench** `est:2h`
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
  - Files: `awn_fpga/rtl/tiled_gemm_s8.v`, `awn_fpga/tb/tb_tiled_gemm_s8.v`, `awn_fpga/sw/test_tiled_systolic.py`
  - Verify: cd awn_fpga && python sw/test_tiled_systolic.py 2>&1 | tail -1 | grep -q 'ALL TILED SYSTOLIC TESTS PASSED'

- [x] **T02: Add BRAM double-buffering for B-matrix weights** `est:2h`
  ## Overview

Add weight double-buffering to `tiled_gemm_s8.v` using two `bram_feeder_b` banks. While the PE grid computes a tile using one BRAM bank, the next tile's B-matrix loads into the other bank, hiding the K-cycle load latency behind the K+22-cycle compute. This requires:
1. Rewriting `bram_feeder_b.v` with wide write port and per-column parallel read ports
2. Instantiating two banks in `tiled_gemm_s8.v`
3. Adding LOAD_B0 state and concurrent B-loading during COMPUTE

Requirements addressed: R002 (weight double-buffering), R010 (dual-port BRAM feeder integration).

## Step 1: Rewrite `awn_fpga/rtl/bram_feeder_b.v`

The current bram_feeder_b has a single-element write port (one col/row per cycle) and single-element read port. Both are inadequate:
- **Write:** Loading a B-tile of K rows × 16 cols takes K×16 cycles with single writes, but compute takes only K+22 cycles. Need wide writes (all 16 cols per cycle) so loading takes K cycles.
- **Read:** 16 PE columns need simultaneous reads at different row addresses (skewed by column index). Need 16 parallel read ports.

New interface using **flat packed buses** (avoids iverilog unpacked array port issues):

```verilog
module bram_feeder_b #(
    parameter COLS  = 16,
    parameter DEPTH = 512,
    parameter AW    = 9
)(
    input                    clk,
    // Wide write port: all COLS values at row wr_k
    input                    wr_en,
    input  [AW-1:0]          wr_k,
    input  [COLS*8-1:0]      wr_data_flat,
    // Per-column parallel read: each column has its own row address
    input  [COLS*AW-1:0]     rd_rows_flat,
    output [COLS*8-1:0]      rd_datas_flat
);
```

Internal storage unchanged: `reg signed [7:0] mem [0:COLS-1][0:DEPTH-1];`

**Wide write** via generate — all columns written simultaneously:
```verilog
genvar wi;
generate
    for (wi = 0; wi < COLS; wi = wi + 1) begin : wr
        always @(posedge clk)
            if (wr_en) mem[wi][wr_k] <= wr_data_flat[wi*8 +: 8];
    end
endgenerate
```

**Per-column read** via generate — each column reads from its own row address:
```verilog
genvar ri;
generate
    for (ri = 0; ri < COLS; ri = ri + 1) begin : rd
        wire [AW-1:0] row_addr = rd_rows_flat[ri*AW +: AW];
        assign rd_datas_flat[ri*8 +: 8] = mem[ri][row_addr];
    end
endgenerate
```

Each column reads `mem[ri]` where `ri` is a genvar constant — no variable-index 2D reads. The row address varies at runtime but indexes a 1D sub-array, which iverilog handles correctly (same pattern proven in S01 for pe_acc).

## Step 2: Modify `awn_fpga/rtl/tiled_gemm_s8.v`

### Add new FSM state and registers

Expand to 6 states:
```verilog
localparam S_IDLE       = 3'd0;
localparam S_LOAD_B0    = 3'd1;
localparam S_COMPUTE    = 3'd2;
localparam S_DRAIN      = 3'd3;
localparam S_NEXT       = 3'd4;
localparam S_DONE       = 3'd5;
```

New registers:
```verilog
reg              active_bank;    // which bank feeds PEs (0 or 1)
reg [DIM_W-1:0]  load_k;         // K counter for B-tile loading
reg              loading;        // concurrent loading active during COMPUTE
reg [DIM_W-1:0]  load_n_base;    // n_base of B-tile being loaded
```

### Instantiate two BRAM banks

Pack read row addresses from PE skew offsets:
```verilog
localparam AW = 9;  // bram_feeder_b address width
wire [PN*AW-1:0] b_rd_rows;
generate
    for (gj = 0; gj < PN; gj = gj + 1) begin : b_addr_gen
        wire [DIM_W-1:0] b_off_j = cycle_cnt - gj[DIM_W-1:0];
        assign b_rd_rows[gj*AW +: AW] = b_off_j[AW-1:0];
    end
endgenerate
```

Instantiate banks:
```verilog
wire [PN*8-1:0] b_rd_data0, b_rd_data1;
wire [PN*8-1:0] wr_data_flat;
wire [AW-1:0]   wr_k_wire;
wire            wr_en0, wr_en1;

bram_feeder_b #(.COLS(PN), .DEPTH(512), .AW(AW)) bank0 (
    .clk(clk), .wr_en(wr_en0), .wr_k(wr_k_wire),
    .wr_data_flat(wr_data_flat),
    .rd_rows_flat(b_rd_rows), .rd_datas_flat(b_rd_data0)
);
bram_feeder_b #(.COLS(PN), .DEPTH(512), .AW(AW)) bank1 (
    .clk(clk), .wr_en(wr_en1), .wr_k(wr_k_wire),
    .wr_data_flat(wr_data_flat),
    .rd_rows_flat(b_rd_rows), .rd_datas_flat(b_rd_data1)
);
```

### Assemble write data combinationally from b_buf

```verilog
wire [DIM_W-1:0] cur_load_nb = (state == S_LOAD_B0) ? n_base : load_n_base;
reg [PN*8-1:0] wr_data_flat_r;
integer ci;
always @(*) begin
    for (ci = 0; ci < PN; ci = ci + 1) begin
        if (cur_load_nb + ci[DIM_W-1:0] < N_reg)
            wr_data_flat_r[ci*8 +: 8] = b_buf[(load_k * N_reg + cur_load_nb + ci[DIM_W-1:0]) & 16'hFFFF];
        else
            wr_data_flat_r[ci*8 +: 8] = 8'd0;  // zero-pad boundary columns
    end
end
assign wr_data_flat = wr_data_flat_r;
```

### Wire write enables

```verilog
wire load_active = (state == S_LOAD_B0) || (state == S_COMPUTE && loading);
wire load_to_bank1 = (state == S_LOAD_B0) ? 1'b0 : ~active_bank;
assign wr_en0 = load_active && !load_to_bank1;
assign wr_en1 = load_active && load_to_bank1;
assign wr_k_wire = load_k[AW-1:0];
```

During LOAD_B0: always loads into bank0. During COMPUTE: loads into idle bank (~active_bank).

### Replace B-feeding with BRAM bank reads

Replace the existing B-feeding generate block (which reads from b_buf) with:
```verilog
generate
    for (gj = 0; gj < PN; gj = gj + 1) begin : b_feed
        wire [DIM_W-1:0] b_offset = cycle_cnt - gj[DIM_W-1:0];
        wire             b_valid  = (state == S_COMPUTE) &&
                                    (cycle_cnt >= gj[DIM_W-1:0]) &&
                                    (b_offset < K_reg) &&
                                    (gj[DIM_W-1:0] < N_tile);
        wire signed [7:0] b_val = active_bank
            ? $signed(b_rd_data1[gj*8 +: 8])
            : $signed(b_rd_data0[gj*8 +: 8]);
        assign b_wire[0][gj] = b_valid ? b_val : 8'sd0;
    end
endgenerate
```

### FSM changes

**S_IDLE → S_LOAD_B0** (not S_COMPUTE). Initialize load_k=0, active_bank=0.

**S_LOAD_B0:** Load first B-tile into bank0. Each cycle: wr_data_flat assembled combinationally from b_buf using n_base and load_k. Increment load_k. When load_k == K_reg - 1: set active_bank=0, cycle_cnt=0, pe_en=1, pe_acc_clear=1. Check if next N-tile exists (nt_count > 1): if yes, start concurrent loading (loading=1, load_k=0, load_n_base=n_base+16). Transition to S_COMPUTE.

**S_COMPUTE:** Same compute logic (pe_acc_clear, cycle_cnt). Additionally: if `loading` is active, increment load_k each cycle. When load_k reaches K_reg-1: loading <= 0. The wr_data_flat assembly uses load_n_base (not n_base) via cur_load_nb MUX. After total_cycles → S_DRAIN.

**S_NEXT changes:**
- If nt+1 < nt_count (more N-tiles, same M-row): swap active_bank (`active_bank <= ~active_bank`). If nt+2 < nt_count: start loading next-next B-tile (loading=1, load_k=0, load_n_base=n_base+32, using pre-update n_base). Set cycle_cnt=0, pe_en=1, pe_acc_clear=1. → S_COMPUTE.
- If mt+1 < mt_count (new M-row): → S_LOAD_B0 (reload first B-tile). Reset load_k=0.
- Else: → S_DONE.

### Double-buffering flow summary

1. LOAD_B0: Load tile 0 into bank0 (K cycles)
2. COMPUTE tile 0 (bank0 active), concurrently load tile 1 into bank1
3. DRAIN → NEXT: swap to bank1, start loading tile 2 into bank0 → COMPUTE
4. COMPUTE tile 1 (bank1 active), concurrently load tile 2 into bank0
5. ... repeat until last N-tile (no concurrent loading)
6. Last N-tile done → advance mt → LOAD_B0 for new M-row (or DONE)

### Update test compile command

If T01 already included bram_feeder_b.v in the compile, no change needed. Otherwise, add it:
```python
iverilog -g2005-sv -o sim tb/tb_tiled_gemm_s8.v rtl/tiled_gemm_s8.v rtl/pe_s8.v rtl/bram_feeder_b.v
```

### CRITICAL PITFALLS

1. **Bank swap timing:** After swapping active_bank in NEXT, the COMPUTE state must read from the NEW active bank. Since active_bank is registered and COMPUTE starts next cycle, the combinational B-feeding MUX uses the updated value. This is correct.
2. **Concurrent load addressing:** load_n_base must be set BEFORE loading starts. In NEXT, `load_n_base <= n_base + 32` uses pre-update n_base (since n_base is also updated with non-blocking assignment in the same block). This correctly computes (nt+2)*16.
3. **LOAD_B0 last write + COMPUTE first cycle:** The last LOAD_B0 write (load_k=K-1 to bank0) and the first COMPUTE cycle happen on consecutive clock edges. Bank0 data is ready for combinational read on the first COMPUTE cycle.
4. **Only 1 N-tile case:** If nt_count=1, LOAD_B0 sets loading=0 (no tile 1 to load). COMPUTE runs without concurrent loading. NEXT advances mt directly.
  - Files: `awn_fpga/rtl/bram_feeder_b.v`, `awn_fpga/rtl/tiled_gemm_s8.v`, `awn_fpga/sw/test_tiled_systolic.py`
  - Verify: cd awn_fpga && python sw/test_tiled_systolic.py 2>&1 | tail -1 | grep -q 'ALL TILED SYSTOLIC TESTS PASSED'

- [x] **T03: Comprehensive randomized verification covering all AWN GEMM shapes** `est:1h`
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
  - Files: `awn_fpga/sw/test_tiled_systolic.py`, `sw/test_tiled_systolic.py`
  - Verify: cd awn_fpga && python sw/test_tiled_systolic.py 2>&1 | tail -1 | grep -q 'ALL TILED SYSTOLIC TESTS PASSED' && python /home/nigel/opensource/adversarial-rf/sw/test_tiled_systolic.py 2>&1 | tail -1 | grep -q 'ALL TILED SYSTOLIC TESTS PASSED'

## Files Likely Touched

- awn_fpga/rtl/tiled_gemm_s8.v
- awn_fpga/tb/tb_tiled_gemm_s8.v
- awn_fpga/sw/test_tiled_systolic.py
- awn_fpga/rtl/bram_feeder_b.v
- sw/test_tiled_systolic.py
