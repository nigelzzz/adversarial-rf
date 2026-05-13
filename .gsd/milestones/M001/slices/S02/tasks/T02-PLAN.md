---
estimated_steps: 162
estimated_files: 3
skills_used: []
---

# T02: Add BRAM double-buffering for B-matrix weights

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

## Inputs

- ``awn_fpga/rtl/tiled_gemm_s8.v` — T01 output. Modify FSM and B-feeding to use BRAM banks instead of direct b_buf reads.`
- ``awn_fpga/rtl/bram_feeder_b.v` — S01 stub with single-element write/read. Rewrite with wide write + multi-read.`
- ``awn_fpga/sw/test_tiled_systolic.py` — T01 output. May need compile command update to include bram_feeder_b.v.`

## Expected Output

- ``awn_fpga/rtl/bram_feeder_b.v` — rewritten with wide write port (COLS*8-bit packed) and per-column parallel read (COLS independent row addresses)`
- ``awn_fpga/rtl/tiled_gemm_s8.v` — modified with two bram_feeder_b banks, LOAD_B0 state, concurrent loading, bank swap logic`
- ``awn_fpga/sw/test_tiled_systolic.py` — compile command includes bram_feeder_b.v if not already present`

## Verification

cd awn_fpga && python sw/test_tiled_systolic.py 2>&1 | tail -1 | grep -q 'ALL TILED SYSTOLIC TESTS PASSED'
