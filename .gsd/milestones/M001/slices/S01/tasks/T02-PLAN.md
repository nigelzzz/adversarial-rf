---
estimated_steps: 186
estimated_files: 3
skills_used: []
---

# T02: Create BRAM feeders and 8x16 systolic mesh module

## Description

Create the BRAM feeder modules and the 8x16 systolic mesh that wires 128 PEs into an output-stationary grid. The mesh has the same external interface as the behavioral `gemm_s8.v` (same parameters, ports, and array names) so existing testbench patterns work unchanged. For S01, the mesh handles single-tile only (M≤8, N≤16, arbitrary K). Full tiling comes in S02.

The BRAM feeders are standalone dual-port memory modules satisfying R010. They are NOT instantiated in the mesh for S01 (the mesh reads directly from a_buf/b_buf with skewed addressing). S02 integrates the feeders for double-buffering.

## Steps

1. Read `awn_fpga/rtl/pe_s8.v` (created in T01) for the PE interface:
   - Ports: `clk, rst_n, en, acc_clear, a_in[7:0], b_in[7:0], a_out[7:0], b_out[7:0], acc[31:0]`
   - All signals `signed`

2. Read `awn_fpga/rtl/gemm_s8.v` for the module interface to match:
   - Parameters: `A_LEN=65536, B_LEN=65536, C_LEN=16384, BIAS_LEN=1024, DIM_W=16`
   - Ports: `clk, rst_n, start, M_in[DIM_W-1:0], K_in[DIM_W-1:0], N_in[DIM_W-1:0], use_bias, done`
   - Arrays: `a_buf` (int8), `b_buf` (int8), `bias_buf` (int32), `c_buf` (int32)
   - **The testbench accesses these arrays via hierarchy** (`DUT.a_buf`, `DUT.c_buf`), so names must match exactly.

3. Create `awn_fpga/rtl/bram_feeder_a.v` — dual-port BRAM for A-matrix row storage.

```verilog
module bram_feeder_a #(
    parameter ROWS  = 8,
    parameter DEPTH = 512,
    parameter AW    = 9
)(
    input               clk,
    // Write port
    input               wr_en,
    input  [2:0]        wr_row,
    input  [AW-1:0]     wr_col,
    input  signed [7:0] wr_data,
    // Read port
    input  [2:0]        rd_row,
    input  [AW-1:0]     rd_col,
    output signed [7:0] rd_data
);
    reg signed [7:0] mem [0:ROWS-1][0:DEPTH-1];
    always @(posedge clk) if (wr_en) mem[wr_row][wr_col] <= wr_data;
    assign rd_data = mem[rd_row][rd_col];  // combinational read
endmodule
```

This is a simple true dual-port: one write port (clocked), one read port (combinational). One read per cycle. The mesh will use multiple instances or direct array reads for parallel feeding.

4. Create `awn_fpga/rtl/bram_feeder_b.v` — same pattern for B-matrix columns:

```verilog
module bram_feeder_b #(
    parameter COLS  = 16,
    parameter DEPTH = 512,
    parameter AW    = 9
)(
    input               clk,
    input               wr_en,
    input  [3:0]        wr_col,
    input  [AW-1:0]     wr_row,
    input  signed [7:0] wr_data,
    input  [3:0]        rd_col,
    input  [AW-1:0]     rd_row,
    output signed [7:0] rd_data
);
    reg signed [7:0] mem [0:COLS-1][0:DEPTH-1];
    always @(posedge clk) if (wr_en) mem[wr_col][wr_row] <= wr_data;
    assign rd_data = mem[rd_col][rd_row];
endmodule
```

5. Create `awn_fpga/rtl/systolic_mesh_s8.v` — the main 8×16 systolic array.

**Interface (MUST match gemm_s8.v):**
```verilog
module systolic_mesh_s8 #(
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
    input  [DIM_W-1:0]   M_in,
    input  [DIM_W-1:0]   K_in,
    input  [DIM_W-1:0]   N_in,
    input                use_bias,
    output reg           done
);
    // MUST use these exact names — testbench accesses via DUT.a_buf etc.
    reg signed [7:0]  a_buf [0:A_LEN-1];
    reg signed [7:0]  b_buf [0:B_LEN-1];
    reg signed [31:0] bias_buf [0:BIAS_LEN-1];
    reg signed [31:0] c_buf [0:C_LEN-1];
```

**PE Grid (generate block):**
```verilog
// Accumulator output — use FLAT 1D array for reliable iverilog variable indexing
wire signed [31:0] pe_acc [0:PM*PN-1];
reg pe_en, pe_acc_clear;

// Horizontal (activation) and vertical (weight) wires
wire signed [7:0] a_wire [0:PM-1][0:PN];   // a_wire[row][0] = input, [PN] = unused
wire signed [7:0] b_wire [0:PM][0:PN-1];   // b_wire[0][col] = input, [PM] = unused

genvar gi, gj;
generate
    for (gi = 0; gi < PM; gi = gi + 1) begin : row
        for (gj = 0; gj < PN; gj = gj + 1) begin : col
            pe_s8 pe_inst (
                .clk(clk), .rst_n(rst_n),
                .en(pe_en), .acc_clear(pe_acc_clear),
                .a_in(a_wire[gi][gj]),
                .b_in(b_wire[gi][gj]),
                .a_out(a_wire[gi][gj+1]),
                .b_out(b_wire[gi+1][gj]),
                .acc(pe_acc[gi*PN+gj])   // flat index
            );
        end
    end
endgenerate
```

**IMPORTANT iverilog note:** Use a flat 1D `pe_acc[gi*PN+gj]` array rather than 2D `pe_acc[gi][gj]` for the accumulator outputs. This ensures reliable variable-index reads during the drain phase (`pe_acc[drain_m*PN+drain_n]`). 2D unpacked arrays with variable indices can be problematic in iverilog. The 2D `a_wire` and `b_wire` arrays are fine because they're only indexed by genvars (constants) inside the generate block.

**Skewed Feeding (generate blocks reading from a_buf/b_buf):**

For row `m` at compute cycle `t`: read `A[m, t-m]` from `a_buf[m*K + (t-m)]` if `t >= m` and `t-m < K`, else feed 0.
For col `n` at compute cycle `t`: read `B[t-n, n]` from `b_buf[(t-n)*N + n]` if `t >= n` and `t-n < K`, else feed 0.

```verilog
// Left edge feeding — one generate per row
generate
    for (gi = 0; gi < PM; gi = gi + 1) begin : a_feed
        assign a_wire[gi][0] = (state == S_COMPUTE && cycle_cnt >= gi &&
                                (cycle_cnt - gi) < K_reg)
            ? a_buf[gi * K_reg + (cycle_cnt - gi)] : 8'sd0;
    end
endgenerate

// Top edge feeding — one generate per column
generate
    for (gj = 0; gj < PN; gj = gj + 1) begin : b_feed
        assign b_wire[0][gj] = (state == S_COMPUTE && cycle_cnt >= gj &&
                                (cycle_cnt - gj) < K_reg)
            ? b_buf[(cycle_cnt - gj) * N_reg + gj] : 8'sd0;
    end
endgenerate
```

**CRITICAL:** The address expressions `gi * K_reg + (cycle_cnt - gi)` mix genvar (constant `gi`) with registers (`K_reg`, `cycle_cnt`). This is valid Verilog. Ensure the computed addresses don't exceed array bounds. For safety, cast to the required width: `a_buf[addr[15:0]]`.

If iverilog rejects `gi * K_reg` in an assign inside generate (unlikely but possible), compute the addresses in an always block using intermediate regs instead.

**FSM:**
```
localparam S_IDLE    = 2'd0;
localparam S_COMPUTE = 2'd1;
localparam S_DRAIN   = 2'd2;
localparam S_DONE    = 2'd3;

reg [1:0] state;
reg [DIM_W-1:0] cycle_cnt;           // compute cycle counter (0 to K+PM+PN-3)
reg [DIM_W-1:0] K_reg, M_reg, N_reg; // latched dimensions
reg use_bias_reg;
reg [DIM_W-1:0] drain_idx;           // flat index for drain: 0 to M*N-1
```

State transitions:
- **S_IDLE → S_COMPUTE**: On `start` pulse, latch M_in→M_reg, K_in→K_reg, N_in→N_reg, use_bias→use_bias_reg. Set cycle_cnt=0, pe_en=1, pe_acc_clear=1, done=0.
- **S_COMPUTE**: Each posedge: if cycle_cnt==0, set pe_acc_clear=0 (it was 1 for just the first cycle). Increment cycle_cnt. When `cycle_cnt == K_reg + PM + PN - 2`, transition to S_DRAIN, set pe_en=0, drain_idx=0.
- **S_DRAIN**: Each posedge: compute `drain_m = drain_idx / N_reg` and `drain_n = drain_idx % N_reg` (or use two counters). Write `c_buf[drain_idx] = pe_acc[drain_m*PN + drain_n] + (use_bias_reg ? bias_buf[drain_m] : 32'sd0)`. Increment drain_idx. When `drain_idx == M_reg * N_reg`, transition to S_DONE.
- **S_DONE**: Assert done=1 for 1 cycle, return to S_IDLE.

**Why acc_clear works globally:** At cycle t=0, acc_clear=1 for ALL PEs. PEs whose inputs are still zero (in the fill phase, where m+n > 0) get acc = 0×0 = 0. PEs with valid data get acc = first product. On all subsequent cycles, acc_clear=0 and PEs accumulate normally. The zero products during fill/drain contribute nothing. This is mathematically equivalent to the behavioral GEMM's sequential accumulation.

**Division in drain:** `drain_m = drain_idx / N_reg` requires a divider. For simulation (iverilog), `/` and `%` operators work. Alternatively, use two nested counters (drain_m, drain_n) to avoid division:
```verilog
reg [DIM_W-1:0] drain_m, drain_n;
// In S_DRAIN:
c_buf[drain_m * N_reg + drain_n] <= pe_acc[drain_m * PN + drain_n] + ...;
if (drain_n == N_reg - 1) begin
    drain_n <= 0;
    if (drain_m == M_reg - 1)
        state <= S_DONE;
    else
        drain_m <= drain_m + 1;
end else
    drain_n <= drain_n + 1;
```

6. Verify all three files compile:
```bash
cd awn_fpga && mkdir -p build
iverilog -g2005-sv -s bram_feeder_a -o /dev/null rtl/bram_feeder_a.v
iverilog -g2005-sv -s bram_feeder_b -o /dev/null rtl/bram_feeder_b.v
iverilog -g2005-sv -s systolic_mesh_s8 -o build/chk_mesh rtl/systolic_mesh_s8.v rtl/pe_s8.v
```

All three commands must complete with zero errors.

## Must-Haves

- [ ] bram_feeder_a.v: dual-port (clocked write, combinational read), ROWS=8, DEPTH=512 default params
- [ ] bram_feeder_b.v: dual-port, COLS=16, DEPTH=512 default params
- [ ] systolic_mesh_s8.v interface matches gemm_s8.v (A_LEN, B_LEN, C_LEN, BIAS_LEN, DIM_W params; clk, rst_n, start, M_in, K_in, N_in, use_bias, done ports)
- [ ] Buffer arrays named exactly: a_buf, b_buf, bias_buf, c_buf (testbench hierarchy access)
- [ ] 8×16 PE grid via generate blocks (128 pe_s8 instances)
- [ ] Flat 1D pe_acc array for reliable variable-index reads during drain
- [ ] Skewed feeding: row m feeds A[m, t-m] starting at cycle m; col n feeds B[t-n, n] starting at cycle n
- [ ] acc_clear=1 on first compute cycle only, then 0
- [ ] Drain writes pe_acc[m*PN+n] + bias (if use_bias) to c_buf[m*N+n] in row-major order
- [ ] All files Verilog-2005 compatible, compile cleanly with iverilog

## Verification

- `cd awn_fpga && mkdir -p build && iverilog -g2005-sv -s bram_feeder_a -o /dev/null rtl/bram_feeder_a.v && iverilog -g2005-sv -s bram_feeder_b -o /dev/null rtl/bram_feeder_b.v && iverilog -g2005-sv -s systolic_mesh_s8 -o build/chk_mesh rtl/systolic_mesh_s8.v rtl/pe_s8.v && echo 'ALL COMPILE OK'`

## Inputs

- ``awn_fpga/rtl/pe_s8.v` — PE module (created in T01, needed for generate block instantiation)`
- ``awn_fpga/rtl/gemm_s8.v` — interface reference (parameters, ports, array names to match)`

## Expected Output

- ``awn_fpga/rtl/bram_feeder_a.v` — dual-port BRAM for A-matrix row storage (PM=8 rows × K_MAX=512 depth)`
- ``awn_fpga/rtl/bram_feeder_b.v` — dual-port BRAM for B-matrix column storage (PN=16 cols × K_MAX=512 depth)`
- ``awn_fpga/rtl/systolic_mesh_s8.v` — 8x16 PE grid with skewed feeding FSM and gemm_s8-compatible interface`

## Verification

cd awn_fpga && mkdir -p build && iverilog -g2005-sv -s bram_feeder_a -o /dev/null rtl/bram_feeder_a.v && iverilog -g2005-sv -s bram_feeder_b -o /dev/null rtl/bram_feeder_b.v && iverilog -g2005-sv -s systolic_mesh_s8 -o build/chk_mesh rtl/systolic_mesh_s8.v rtl/pe_s8.v && echo 'ALL COMPILE OK'
