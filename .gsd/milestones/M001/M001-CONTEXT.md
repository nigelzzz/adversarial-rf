# M001: Systolic Array AWN Accelerator — Context

**Gathered:** 2026-05-13
**Status:** Ready for planning

## Project Description

Replace the behavioral single-MAC `gemm_s8` module in the AWN FPGA inference pipeline with an 8x16 output-stationary systolic array (128 PEs), add hardware im2col and AXI PS-PL interface, and prove end-to-end inference latency <500 us at 200 MHz in iverilog simulation.

## Why This Milestone

The current behavioral `gemm_s8` processes 1 MAC/cycle, resulting in 30.8 ms inference latency — 61x over WiFi's burst-level classification budget of ~500 us. A systolic array with 128 parallel MACs brings theoretical compute time to ~355 us, within budget. This milestone proves the design works in simulation before committing to board deployment.

## User-Visible Outcome

### When this milestone is complete, the user can:

- Run the full AWN inference pipeline in iverilog simulation using the systolic array path and get byte-identical results to the behavioral reference model
- See per-layer cycle count breakdown and verify total cycles < 100,000 (500 us at 200 MHz)
- Trigger inference via simulated AXI-Lite control, feed IQ data via simulated DMA, and read back 11-class logits

### Entry point / environment

- Entry point: `iverilog` testbench simulation + Python verification scripts
- Environment: local dev (iverilog + Python)
- Live dependencies involved: none (simulation only)

## Completion Class

- Contract complete means: all 38 GEMM ops produce byte-identical output through systolic path vs behavioral baseline; per-PE and randomized matrix tests pass
- Integration complete means: full pipeline controller orchestrates systolic GEMM + hardware im2col + non-GEMM modules end-to-end; AXI interface simulated
- Operational complete means: none (simulation milestone; board deployment is next milestone)

## Final Integrated Acceptance

To call this milestone complete, we must prove:

- Full AWN inference (IQ input → 11-class logits) through the systolic path matches the behavioral refmodel byte-for-byte
- Total cycle count < 100,000 at 200 MHz with per-layer breakdown
- PS-PL round-trip (AXI-Lite control + DMA data movement) works in simulation

## Scope

### In Scope

- 8x16 output-stationary systolic array PE and mesh RTL
- Tile sequencer FSM with weight double-buffering and zero-padding for boundary tiles
- Hardware im2col unit for 3 kernel configurations (2x7, 1x7, 1x17)
- Full AWN pipeline controller sequencing 38 operations
- AXI-Lite control interface + AXI-DMA data interface
- Per-PE unit tests, randomized matrix tests, end-to-end hex pipeline verification
- Per-layer cycle count breakdown
- DMA error flag detection with PS-readable status register

### Out of Scope / Non-Goals

- Real Pynq-Z2 board deployment (next milestone)
- Vivado synthesis and timing closure
- Power/thermal optimization
- Linux runtime driver
- AWN model changes or retraining
- General-purpose accelerator beyond AWN shapes

## Architectural Decisions

### Systolic Array Dataflow: Output-Stationary

**Decision:** Use output-stationary dataflow where each PE accumulates one output element and weights stream through the array.

**Rationale:** AWN GEMM shapes have K (192-320) large relative to M and N (64-128). Output-stationary naturally hides the K-dimension accumulation latency. The accumulation order matches the behavioral model exactly, guaranteeing zero rounding divergence for bit-exact verification.

**Alternatives Considered:**
- Weight-stationary — would simplify double-buffering since weights are static per layer, but doesn't exploit the K-dimension accumulation advantage and changes accumulation order

### Memory Architecture: Dedicated Dual-Port BRAMs

**Decision:** Use dedicated dual-port BRAM blocks for A-matrix (row feeder) and B-matrix (column feeder) rather than reusing the existing single-port `global_buffer.v`.

**Rationale:** The systolic array needs simultaneous row and column reads every cycle. Single-port memory would halve throughput.

**Alternatives Considered:**
- Reuse `global_buffer.v` — single-port limits bandwidth, would need time-multiplexed access

### im2col Placement: Hardware in PL

**Decision:** Implement im2col as a hardware address generation FSM in the PL fabric, not software on the PS ARM core.

**Rationale:** Software im2col adds 350-700 us overhead — nearly the entire 500 us latency budget. Hardware im2col adds ~200 LUTs and eliminates this overhead entirely.

**Alternatives Considered:**
- Software im2col on PS — unacceptable latency overhead for the WiFi budget

### Verification Strategy: Three-Tier

**Decision:** Three-tier verification: (1) per-PE unit tests, (2) randomized matrix tests, (3) end-to-end 38-op hex pipeline.

**Rationale:** The existing hex-file pattern is proven across 9 modules but only covers AWN-specific inputs. Per-PE and randomized tests catch corner cases in accumulation, overflow, and tiling boundaries.

**Alternatives Considered:**
- Hex-file only — insufficient coverage per user requirement

## Error Handling Strategy

- **Arithmetic overflow in PEs:** int8 x int8 → int16 partial products accumulated into int32. int32 holds 2^15 int16 products without overflow — more than sufficient for max K=320. Requantize back to int8 via existing `requantize_s32_s8` module (shift + clamp to [-128, 127]).
- **Tiling boundary conditions:** When M, N, or K don't divide evenly by tile dimensions (8, 16), zero-pad the trailing tile. FSM masks writes for out-of-bounds output positions.
- **DMA transfer errors:** AXI-DMA SlvErr/DecErr flags checked after each transfer. Error status register readable by PS. No automatic retry — PS decides.
- **Simulation mismatches:** Any bit-level difference between systolic output and behavioral baseline is a hard fail. Zero tolerance.
- **im2col out-of-bounds:** Hardware im2col generates zero for padding positions. Address generation FSM clamps to valid feature map bounds.

## Risks and Unknowns

- Hardware im2col address generation FSM for varying kernel sizes (2x7, 1x7, 1x17) with different stride and padding — not yet prototyped, needs careful handling
- Tiling controller complexity for non-aligned GEMM dimensions — all real AWN shapes have remainders
- BRAM utilization may be tight with weight double-buffering — analysis estimates 46% but actual routing may differ

## Existing Codebase / Prior Art

- `awn_fpga/rtl/gemm_s8.v` — behavioral GEMM, 1 MAC/cycle, to be replaced
- `awn_fpga/rtl/requantize_s32_s8.v` — requantization module, reused by systolic PEs
- `awn_fpga/rtl/global_buffer.v` — single-port parameterized RAM, not reused for systolic feeders
- `awn_fpga/rtl/*.v` — 7 other non-GEMM modules, unchanged
- `awn_fpga/tb/tb_gemm_s8.v` — existing testbench pattern to follow
- `awn_fpga/sw/refmodel.py` — Python reference model with 38 op invocations
- `awn_fpga/sw/quantize_awn.py` — quantization pipeline, int8 parameter extraction
- `awn_fpga/vectors/` — 126 hex test vectors
- `awn_fpga/systolic_optimization.md` — design analysis with PE sketch, tiling math, resource estimates

## Relevant Requirements

- R001-R010, R016-R020 — all 15 active requirements are owned by this milestone

## Technical Constraints

- Target: Zynq-7020 (220 DSP48s, 140 BRAM36Ks, 53K LUTs)
- Clock: 200 MHz
- Verification: iverilog simulation only (no Vivado in this milestone)
- All GEMM shapes: M=64, K=192-320, N=64-128
- Existing hex vector format: $readmemh/$fwrite, maintained for compatibility

## Integration Points

- Existing non-GEMM RTL modules — pipeline controller must orchestrate them unchanged
- refmodel.py — generates reference hex vectors for all 38 ops
- quantize_awn.py — extracts int8 weights/biases for hex vector generation

## Testing Requirements

- Per-PE unit tests: verify MAC + accumulate + requantize for edge cases (max positive, max negative, zero, overflow boundary)
- Randomized matrix tests: random int8 matrices of varying dimensions, compare against numpy
- End-to-end pipeline: all 38 ops through systolic path, hex output matches behavioral baseline byte-for-byte
- Non-GEMM regression: all 9 existing testbenches still pass
- im2col verification: hardware output matches software for all 3 kernel configs
- Cycle count measurement: per-layer and aggregate, must total <100K

## Acceptance Criteria

- **S01:** Single PE passes unit tests; 8x16 mesh computes small GEMM bit-exact; randomized tests pass
- **S02:** Tile sequencer drives mesh through full (64,320,128) GEMM with tiling and boundary zero-padding; output matches behavioral hex
- **S03:** im2col FSM generates correct addresses for all 3 kernel configs; output matches software byte-for-byte
- **S04:** All 38 ops execute through systolic path; non-GEMM modules pass existing tests; per-layer cycle counts reported
- **S05:** PS writes IQ data via DMA, triggers inference via AXI-Lite, reads back logits; total <100K cycles; full PS-PL round-trip in simulation

## Open Questions

- Exact BRAM partitioning for weight double-buffering — analysis estimates 46% utilization but actual allocation depends on tile sizes chosen during S02 implementation
