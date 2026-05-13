# S01: Systolic Array PE + 8x16 Mesh

**Goal:** Build a single PE (pe_s8.v) with int8×int8 MAC and int32 accumulator, wire 128 PEs into an 8x16 output-stationary systolic mesh (systolic_mesh_s8.v), create dedicated BRAM feeder modules (bram_feeder_a.v, bram_feeder_b.v), and verify via per-PE unit tests and randomized matrix multiplication tests producing bit-exact int32 results matching numpy reference. Single-tile only (M≤8, N≤16, arbitrary K); full tiling comes in S02.
**Demo:** Single PE passes unit tests (MAC + accumulate + requantize); 8x16 mesh computes a small matrix multiply bit-exact against behavioral gemm_s8; randomized matrix tests pass

## Must-Haves

- 1. PE unit tests: `cd awn_fpga && mkdir -p build && iverilog -g2005-sv -o build/sim_pe_s8 tb/tb_pe_s8.v rtl/pe_s8.v && vvp build/sim_pe_s8 | grep -q "ALL PE TESTS PASSED"`
- 2. Mesh randomized tests: `cd awn_fpga && python sw/test_systolic.py` exits 0 and prints "ALL SYSTOLIC TESTS PASSED"
- ## Must-Haves
- Single PE performs int8×int8 MAC with int32 accumulator, registered passthrough (R001, R019)
- 8x16 PE grid (128 PEs) via generate blocks with skewed feeding (R001)
- Dedicated dual-port BRAM feeder modules exist as standalone RTL (R010)
- Mesh interface matches gemm_s8.v (same parameters, ports, array names) for drop-in testbench compatibility
- Per-PE unit tests cover reset, accumulate, passthrough, enable gate, signed edge cases (R008)
- Randomized matrix tests (50+ trials) produce bit-exact match against numpy int32 matmul (R007, R008)
- All Verilog is 2005-compatible (iverilog -g2005-sv), no SystemVerilog features
- ## Proof Level
- This slice proves: contract (PE correctness + single-tile mesh correctness via simulation)
- Real runtime required: no (iverilog simulation only)
- Human/UAT required: no
- ## Verification
- `cd awn_fpga && mkdir -p build && iverilog -g2005-sv -o build/sim_pe_s8 tb/tb_pe_s8.v rtl/pe_s8.v && vvp build/sim_pe_s8 | grep -q "ALL PE TESTS PASSED"`
- `cd awn_fpga && python sw/test_systolic.py` — exits 0, prints "ALL SYSTOLIC TESTS PASSED" (62+ tests including deterministic and randomized)
- ## Integration Closure
- Upstream surfaces consumed: none (S01 has no dependencies)
- New wiring: pe_s8 → systolic_mesh_s8 via generate blocks; bram_feeder_a/b created as standalone modules
- Remaining: S02 adds tiling FSM + integrates feeders for double-buffering; S04 integrates mesh into pipeline controller

## Proof Level

- This slice proves: Not provided.

## Integration Closure

Not provided.

## Verification

- Not provided.

## Tasks

- [x] **T01: Create PE module and unit testbench** `est:1h`
  ## Description

Create the single processing element (PE) for the 8x16 systolic array and its self-checking unit testbench. The PE performs one int8×int8 multiply-accumulate per cycle with int32 accumulation and passes inputs to neighbors with 1-cycle delay. This is the atomic building block — correctness here is foundational for the entire systolic array.

All files live under `awn_fpga/`. Read `awn_fpga/rtl/gemm_s8.v` for the existing arithmetic pattern (the PE must match exactly). Read `awn_fpga/tb/tb_gemm_s8.v` for testbench conventions.

## Steps

1. Read `awn_fpga/rtl/gemm_s8.v` to understand the arithmetic: lines ~60-70 show `$signed(a_buf[...]) * $signed(b_buf[...])` producing a 16-bit product, sign-extended via `{{16{prod16[15]}}, prod16}` to 32 bits, accumulated into `acc` (int32). The PE must reproduce this exact pattern.

2. Read `awn_fpga/tb/tb_gemm_s8.v` for testbench conventions: 20ns clock period (10ns half), active-low async reset, `$display` for output.

3. Create `awn_fpga/rtl/pe_s8.v` with this exact interface:

```verilog
module pe_s8 (
    input               clk,
    input               rst_n,      // active-low asynchronous reset
    input               en,         // gate MAC and passthrough
    input               acc_clear,  // 1 = init acc to current product (new tile)
    input  signed [7:0] a_in,       // activation from west
    input  signed [7:0] b_in,       // weight from north
    output reg signed [7:0]  a_out, // pass east (1-cycle latency)
    output reg signed [7:0]  b_out, // pass south (1-cycle latency)
    output reg signed [31:0] acc    // accumulated partial sum
);
```

Internal arithmetic:
```verilog
wire signed [15:0] prod = a_in * b_in;
wire signed [31:0] prod32 = {{16{prod[15]}}, prod};

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        a_out <= 8'sd0;
        b_out <= 8'sd0;
        acc   <= 32'sd0;
    end else if (en) begin
        a_out <= a_in;
        b_out <= b_in;
        acc   <= acc_clear ? prod32 : (acc + prod32);
    end
    // !en: all registers hold (no update)
end
```

4. Create `awn_fpga/tb/tb_pe_s8.v` — self-checking testbench. Clock 20ns period (10ns half). Reset: drive rst_n=0 for >=40ns, then rst_n=1, wait for negedge clk before stimulus.

Test cases (print PASS/FAIL per test via `$display`, call `$finish` on first failure):

| # | Test | Setup | Expected |
|---|------|-------|----------|
| 1 | Reset | After reset | acc==0, a_out==0, b_out==0 |
| 2 | Single MAC | en=1, acc_clear=1, a_in=3, b_in=7 | acc==21 |
| 3 | Accumulate | acc_clear=0, a_in=5, b_in=-3 | acc==6 (21+(-15)) |
| 4 | Continue | a_in=-10, b_in=4 | acc==-34 (6+(-40)) |
| 5 | Passthrough | Feed a_in=42, b_in=-100 | a_out==42, b_out==-100 on NEXT posedge |
| 6 | Enable gate | en=0, change a_in/b_in | acc, a_out, b_out unchanged |
| 7 | Signed min×min | en=1, acc_clear=1, a_in=-128, b_in=-128 | acc==16384 |
| 8 | Signed cross | acc_clear=0, a_in=-128, b_in=127 | acc==128 (16384+(-16256)) |
| 9 | Stress ×256 | acc_clear=1 then 255× acc_clear=0 with a=-128,b=-128 | acc==4194304 |

End with `$display("ALL PE TESTS PASSED");` if all pass.

5. Compile and run:
```bash
cd awn_fpga && mkdir -p build
iverilog -g2005-sv -o build/sim_pe_s8 tb/tb_pe_s8.v rtl/pe_s8.v
vvp build/sim_pe_s8
```

## Must-Haves

- [ ] pe_s8.v uses explicit sign extension `{{16{prod[15]}}, prod}` (not implicit Verilog widening)
- [ ] Active-low async reset: `always @(posedge clk or negedge rst_n)`
- [ ] en=0 holds ALL registers (acc, a_out, b_out) — no state changes
- [ ] acc_clear=1 sets acc to current product (not zero)
- [ ] Verilog-2005 compatible (iverilog -g2005-sv, no SystemVerilog interfaces/logic/structs)
- [ ] All 9 test cases pass

## Verification

- `cd awn_fpga && mkdir -p build && iverilog -g2005-sv -o build/sim_pe_s8 tb/tb_pe_s8.v rtl/pe_s8.v && vvp build/sim_pe_s8 | grep -q "ALL PE TESTS PASSED"`

## Negative Tests

- **Signed extremes**: -128 × -128 = 16384 (positive), -128 × 127 = -16256 (negative)
- **Accumulation overflow boundary**: 256 × 16384 = 4,194,304 fits in int32 (max 2,147,483,647)
- **Enable gate**: en=0 must freeze ALL state, even with changing inputs
  - Files: `awn_fpga/rtl/pe_s8.v`, `awn_fpga/tb/tb_pe_s8.v`
  - Verify: cd awn_fpga && mkdir -p build && iverilog -g2005-sv -o build/sim_pe_s8 tb/tb_pe_s8.v rtl/pe_s8.v && vvp build/sim_pe_s8 | grep -q 'ALL PE TESTS PASSED'

- [x] **T02: Create BRAM feeders and 8x16 systolic mesh module** `est:2h`
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
  - Files: `awn_fpga/rtl/bram_feeder_a.v`, `awn_fpga/rtl/bram_feeder_b.v`, `awn_fpga/rtl/systolic_mesh_s8.v`
  - Verify: cd awn_fpga && mkdir -p build && iverilog -g2005-sv -s bram_feeder_a -o /dev/null rtl/bram_feeder_a.v && iverilog -g2005-sv -s bram_feeder_b -o /dev/null rtl/bram_feeder_b.v && iverilog -g2005-sv -s systolic_mesh_s8 -o build/chk_mesh rtl/systolic_mesh_s8.v rtl/pe_s8.v && echo 'ALL COMPILE OK'

- [x] **T03: Create mesh testbench and randomized verification** `est:1h30m`
  ## Description

Create the mesh testbench and a Python randomized verification script that proves the systolic array produces bit-exact int32 results matching numpy for all single-tile matrix sizes (M≤8, N≤16, arbitrary K). The testbench follows the existing `tb_gemm_s8.v` pattern exactly (plusargs, $readmemh, $fwrite). The Python script generates random int8 matrices, writes hex vectors, runs the iverilog simulation, and asserts zero divergence.

## Steps

1. Read `awn_fpga/tb/tb_gemm_s8.v` for the testbench pattern. Key elements:
   - Plusargs: `$value$plusargs("M=%d", M_arg)`, similarly K, N, bias, a (path), b (path), bi (path), out (path)
   - Data loading: `$readmemh(a_path, DUT.a_buf)` — loads hex file directly into DUT's internal array via hierarchy
   - Output dump: `$fwrite(fout, "%08x\n", DUT.c_buf[k] & 32'hffffffff)` — 8-char hex, one int32 per line
   - Clock: 20ns period (10ns half)
   - Reset: rst_n=0 → wait → rst_n=1 → negedge → start=1 → negedge → start=0
   - Timeout: based on `8 * M * K * N + 100000` cycles

2. Read `awn_fpga/sw/iohex.py` for hex I/O helpers:
   - `write_int8_hex(path, arr)`: writes numpy int8 array as 2-char hex bytes, one per line
   - `write_int32_hex(path, arr)`: writes numpy int32 array as 8-char hex words, one per line
   - `read_int32_hex(path, count)`: reads hex words back to int32 array

3. Create `awn_fpga/tb/tb_systolic_mesh_s8.v`:

```verilog
module tb_systolic_mesh_s8;
    reg clk = 0;
    always #10 clk = ~clk;   // 20ns period
    
    reg rst_n, start;
    reg [15:0] M_arg, K_arg, N_arg;
    reg bias_arg;
    wire done;
    
    systolic_mesh_s8 DUT (
        .clk(clk), .rst_n(rst_n),
        .start(start),
        .M_in(M_arg), .K_in(K_arg), .N_in(N_arg),
        .use_bias(bias_arg),
        .done(done)
    );
    
    reg [256*8-1:0] a_path, b_path, bias_path, out_path;
    integer fout, k;
    
    initial begin
        $value$plusargs("M=%d", M_arg);
        $value$plusargs("K=%d", K_arg);
        $value$plusargs("N=%d", N_arg);
        $value$plusargs("bias=%d", bias_arg);
        $value$plusargs("a=%s", a_path);
        $value$plusargs("b=%s", b_path);
        if (bias_arg) $value$plusargs("bi=%s", bias_path);
        $value$plusargs("out=%s", out_path);
        
        $readmemh(a_path, DUT.a_buf);
        $readmemh(b_path, DUT.b_buf);
        if (bias_arg) $readmemh(bias_path, DUT.bias_buf);
        
        rst_n = 0; start = 0;
        #40;
        rst_n = 1;
        @(negedge clk);
        start = 1;
        @(negedge clk);
        start = 0;
        
        wait (done);
        @(negedge clk);
        
        fout = $fopen(out_path, "w");
        for (k = 0; k < M_arg * N_arg; k = k + 1)
            $fwrite(fout, "%08x\n", DUT.c_buf[k] & 32'hffffffff);
        $fclose(fout);
        $finish;
    end
    
    // Timeout
    initial begin
        #(20 * (8 * M_arg * K_arg * N_arg + 200000));
        $display("TIMEOUT at cycle %0d", $time/20);
        $finish;
    end
endmodule
```

**iverilog notes:** Declare loop variable `k` as `integer k` at module level (not inside for-loop). Use `$value$plusargs` return value or separate `initial begin` block for bias_arg check. If `if (bias_arg)` before `$readmemh` causes issues, use a conditional: always call `$value$plusargs` but only pass to `$readmemh` when bias_arg is set.

4. Create `awn_fpga/sw/test_systolic.py` — randomized verification script:

```python
#!/usr/bin/env python3
"""Randomized verification of systolic_mesh_s8 against numpy."""
import sys, os, subprocess, tempfile
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import iohex

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
AWN_DIR = os.path.join(SCRIPT_DIR, '..')
BUILD_DIR = os.path.join(AWN_DIR, 'build')

def compile_sim():
    os.makedirs(BUILD_DIR, exist_ok=True)
    sim = os.path.join(BUILD_DIR, 'sim_systolic_mesh_s8')
    subprocess.check_call([
        'iverilog', '-g2005-sv', '-o', sim,
        os.path.join(AWN_DIR, 'tb', 'tb_systolic_mesh_s8.v'),
        os.path.join(AWN_DIR, 'rtl', 'systolic_mesh_s8.v'),
        os.path.join(AWN_DIR, 'rtl', 'pe_s8.v'),
    ])
    return sim

def run_gemm(sim, M, K, N, A, B, bias=None):
    with tempfile.TemporaryDirectory() as tmp:
        a_f = os.path.join(tmp, 'a.hex')
        b_f = os.path.join(tmp, 'b.hex')
        o_f = os.path.join(tmp, 'out.hex')
        iohex.write_int8_hex(a_f, A.flatten())
        iohex.write_int8_hex(b_f, B.flatten())
        pa = [f'+M={M}', f'+K={K}', f'+N={N}', f'+a={a_f}', f'+b={b_f}', f'+out={o_f}']
        if bias is not None:
            bi_f = os.path.join(tmp, 'bi.hex')
            iohex.write_int32_hex(bi_f, bias.flatten())
            pa += ['+bias=1', f'+bi={bi_f}']
        else:
            pa += ['+bias=0']
        subprocess.check_call(['vvp', sim] + pa, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return iohex.read_int32_hex(o_f, count=M*N).reshape(M, N)

def test_one(sim, M, K, N, use_bias, label):
    A = np.random.randint(-128, 127, (M, K), dtype=np.int8)
    B = np.random.randint(-128, 127, (K, N), dtype=np.int8)
    bias = np.random.randint(-100000, 100000, (M,), dtype=np.int32) if use_bias else None
    C_ref = A.astype(np.int32) @ B.astype(np.int32)
    if bias is not None:
        C_ref += bias[:, np.newaxis]
    C_hw = run_gemm(sim, M, K, N, A, B, bias)
    ok = np.array_equal(C_ref, C_hw)
    status = 'PASS' if ok else 'FAIL'
    print(f'{status} {label}: M={M} K={K} N={N} bias={use_bias}')
    if not ok:
        diffs = np.argwhere(C_ref != C_hw)
        for idx in diffs[:5]:
            print(f'  C[{idx[0]},{idx[1]}]: expected {C_ref[tuple(idx)]}, got {C_hw[tuple(idx)]}')
    return ok

def main():
    np.random.seed(42)
    sim = compile_sim()
    ok = True
    tid = 0
    # Deterministic tests
    for M, K, N in [(1,1,1), (2,2,2), (4,4,4), (8,16,16), (8,1,16), (1,64,1), (8,320,16)]:
        for ub in [False, True]:
            tid += 1
            ok &= test_one(sim, M, K, N, ub, f'T{tid:02d}')
    # Randomized tests
    for _ in range(50):
        M = int(np.random.choice([1, 2, 4, 7, 8]))
        K = int(np.random.choice([1, 4, 16, 32, 64, 128, 192, 320]))
        N = int(np.random.choice([1, 2, 8, 15, 16]))
        ub = bool(np.random.random() < 0.5)
        tid += 1
        ok &= test_one(sim, M, K, N, ub, f'T{tid:02d}')
    if ok:
        print(f'\nALL SYSTOLIC TESTS PASSED ({tid} tests)')
        sys.exit(0)
    else:
        print(f'\nSOME TESTS FAILED')
        sys.exit(1)

if __name__ == '__main__':
    main()
```

**Key details:**
- Use `iohex.write_int8_hex` / `iohex.write_int32_hex` / `iohex.read_int32_hex` — do NOT rewrite hex I/O
- A is stored row-major: A[m,k] at flat index m*K+k. B is row-major: B[k,n] at flat index k*N+n. This matches gemm_s8.v's indexing.
- numpy int8 range: `np.random.randint(-128, 127)` gives -128..126. For full range use `randint(-128, 128)` which gives -128..127.
- Reference: `C = A.astype(np.int32) @ B.astype(np.int32)` — cast BEFORE matmul to get int32 arithmetic (numpy's int8 @ int8 would overflow)
- Bias: broadcast `bias[m]` across all N columns: `C_ref += bias[:, np.newaxis]`
- Test sizes: M∈{1,2,4,7,8}, K∈{1..320}, N∈{1,2,8,15,16} — all within single-tile limits (M≤8, N≤16)
- The 8,320,16 test exercises the largest K dimension used in the AWN pipeline (conv2 K=320)

5. Run full verification:
```bash
cd awn_fpga && python sw/test_systolic.py
```

Expected: 62+ tests (14 deterministic + 50 randomized), all PASS, exit code 0.

## Must-Haves

- [ ] Testbench uses same plusargs as tb_gemm_s8.v (M, K, N, bias, a, b, bi, out)
- [ ] Testbench loads data via $readmemh into DUT.a_buf, DUT.b_buf, DUT.bias_buf
- [ ] Testbench outputs via $fwrite with %08x format, masked with 32'hffffffff
- [ ] Python script uses iohex.py helpers (no custom hex I/O)
- [ ] 50+ randomized tests covering M∈{1..8}, K∈{1..320}, N∈{1..16}
- [ ] Tests with and without bias
- [ ] All tests bit-exact match: `np.array_equal(C_ref, C_hw)`
- [ ] Script exits 0 and prints "ALL SYSTOLIC TESTS PASSED" on success

## Verification

- `cd awn_fpga && python sw/test_systolic.py` exits 0 and prints "ALL SYSTOLIC TESTS PASSED"

## Negative Tests

- **Degenerate**: M=1, K=1, N=1 (single multiply, no accumulation)
- **Column vector**: M=8, K=320, N=1 (15 of 16 columns idle)
- **Row vector**: M=1, K=64, N=16 (7 of 8 rows idle)
- **Max pipeline depth**: M=8, K=320, N=16 (K+PM+PN-2 = 342 compute cycles)
- **Signed extremes**: covered by randomized tests with np.random.randint(-128, 127)
  - Files: `awn_fpga/tb/tb_systolic_mesh_s8.v`, `awn_fpga/sw/test_systolic.py`
  - Verify: cd awn_fpga && python sw/test_systolic.py

## Files Likely Touched

- awn_fpga/rtl/pe_s8.v
- awn_fpga/tb/tb_pe_s8.v
- awn_fpga/rtl/bram_feeder_a.v
- awn_fpga/rtl/bram_feeder_b.v
- awn_fpga/rtl/systolic_mesh_s8.v
- awn_fpga/tb/tb_systolic_mesh_s8.v
- awn_fpga/sw/test_systolic.py
