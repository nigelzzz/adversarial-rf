# Knowledge Base

## iverilog Gotchas (Discovered in S01)

### Flat 1D arrays for variable-index reads
iverilog can fail silently or produce incorrect results when reading 2D unpacked arrays with variable indices (e.g., `pe_acc[drain_m][drain_n]`). Use flat 1D arrays with computed index (`pe_acc[drain_m*PN+drain_n]`) instead. Genvar-indexed 2D arrays in generate blocks are fine since indices are compile-time constants.

### Nested counters instead of integer division
For drain-phase indexing, use two nested counters (`drain_m`, `drain_n`) rather than `drain_idx / N_reg` and `drain_idx % N_reg`. iverilog's `/` and `%` operators work for simulation but nested counters avoid synthesis issues and map better to hardware.

### Verification command CWD splitting
GSD verification gates split `&&`-chained commands into separate shell invocations. `cd awn_fpga && python sw/test_systolic.py` becomes two commands where the `cd` doesn't persist. Fix: either use absolute paths, or provide a wrapper script at the repo root that delegates to the real script. The `sw/test_systolic.py` wrapper at repo root was created for this reason.

### Testbench timing: single-edge stimulus
When writing self-checking testbenches, use one `@(negedge clk)` per stimulus cycle, not two. Two waits causes the DUT to see two posedge clocks per test case, double-accumulating values.

### acc_clear global pulse is safe
In a systolic mesh, asserting `acc_clear=1` for ALL PEs on cycle 0 is mathematically correct even though most PEs receive zero inputs during the fill phase. `0*0 = 0`, so the accumulator initializes to zero for PEs that haven't received valid data yet. No per-PE staggered clear needed.
