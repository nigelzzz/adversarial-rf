---
id: T01
parent: S01
milestone: M001
key_files:
  - awn_fpga/rtl/pe_s8.v
  - awn_fpga/tb/tb_pe_s8.v
key_decisions:
  - PE uses single always block with async reset (matches gemm_s8.v pattern) rather than separate combinational + sequential blocks
  - acc_clear=1 loads product directly (not zero-then-add) to save one cycle per tile start
duration: 
verification_result: passed
completed_at: 2026-05-13T10:36:12.353Z
blocker_discovered: false
---

# T01: Add int8×int8 MAC processing element (pe_s8.v) with int32 accumulator and 9-case self-checking testbench

**Add int8×int8 MAC processing element (pe_s8.v) with int32 accumulator and 9-case self-checking testbench**

## What Happened

Created `pe_s8.v` implementing the single processing element for the output-stationary systolic array. The PE performs one int8×int8 multiply-accumulate per clock with int32 accumulation, using explicit sign extension (`{{16{prod[15]}}, prod}`) matching the `gemm_s8.v` reference arithmetic. Registered passthrough delivers `a_in`→`a_out` and `b_in`→`b_out` with 1-cycle latency. Active-low async reset clears all registers; `en=0` holds all state (acc, a_out, b_out); `acc_clear=1` initializes accumulator to the current product (not zero).

Created `tb_pe_s8.v` with 9 deterministic self-checking test cases covering: reset state, single MAC, accumulation, continued accumulation, passthrough latency verification, enable gating (en=0 freezes all registers), signed extremes (-128×-128=16384), signed cross (-128×127=-16256 accumulated), and a 256-iteration stress test (256×16384=4,194,304 fits int32).

Initial testbench had a timing bug — two `@(negedge clk)` waits per stimulus caused double-MAC on accumulate steps. Fixed to single-edge timing: set inputs, one `@(negedge clk)` lets exactly one posedge latch, then check.

## Verification

Compiled with `iverilog -g2005-sv` and ran with `vvp`. All 9 test cases pass, ending with "ALL PE TESTS PASSED". Full verification command `iverilog ... && vvp ... | grep -q "ALL PE TESTS PASSED"` exits 0.

## Verification Evidence

| # | Command | Exit Code | Verdict | Duration |
|---|---------|-----------|---------|----------|
| 1 | `cd awn_fpga && mkdir -p build && iverilog -g2005-sv -o build/sim_pe_s8 tb/tb_pe_s8.v rtl/pe_s8.v && vvp build/sim_pe_s8 | grep -q 'ALL PE TESTS PASSED'` | 0 | pass | 850ms |

## Deviations

Fixed testbench timing from two @(negedge clk) per stimulus to one — the plan's test sequence implicitly assumed single-edge timing but the initial implementation double-clocked accumulate steps.

## Known Issues

None

## Files Created/Modified

- `awn_fpga/rtl/pe_s8.v`
- `awn_fpga/tb/tb_pe_s8.v`
