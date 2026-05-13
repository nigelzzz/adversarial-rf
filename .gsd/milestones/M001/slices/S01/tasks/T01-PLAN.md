---
estimated_steps: 70
estimated_files: 2
skills_used: []
---

# T01: Create PE module and unit testbench

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

## Inputs

- ``awn_fpga/rtl/gemm_s8.v` — reference arithmetic pattern (sign extension, int32 accumulation)`
- ``awn_fpga/tb/tb_gemm_s8.v` — testbench conventions (clock, reset, plusargs, $display)`

## Expected Output

- ``awn_fpga/rtl/pe_s8.v` — single PE module with int8×int8 MAC, int32 accumulator, registered passthrough`
- ``awn_fpga/tb/tb_pe_s8.v` — self-checking testbench with 9 deterministic test cases`

## Verification

cd awn_fpga && mkdir -p build && iverilog -g2005-sv -o build/sim_pe_s8 tb/tb_pe_s8.v rtl/pe_s8.v && vvp build/sim_pe_s8 | grep -q 'ALL PE TESTS PASSED'
