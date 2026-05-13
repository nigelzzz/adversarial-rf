# Requirements

This file is the explicit capability and coverage contract for the project.

## Active

### R001 — 8x16 Output-Stationary Systolic Array (128 PEs)
- Class: core-capability
- Status: active
- Description: An 8-row by 16-column output-stationary systolic array where each PE performs int8 MAC with int32 accumulation and requantization to int8
- Why it matters: 128 parallel MACs are required to bring GEMM latency from 30.8 ms to <500 us
- Source: user
- Primary owning slice: M001/S01
- Supporting slices: none
- Validation: mapped
- Notes: Output-stationary chosen because K is large (192-320) relative to M/N (64-128)

### R002 — Tile Sequencer FSM with Weight Double-Buffering
- Class: core-capability
- Status: active
- Description: FSM that tiles large GEMM operations across the 8x16 array, streaming weight tiles with double-buffering to hide load latency
- Why it matters: No single AWN GEMM fits in the 8x16 array at once; tiling is required for all real workloads
- Source: user
- Primary owning slice: M001/S02
- Supporting slices: none
- Validation: mapped
- Notes: Must handle zero-padding for boundary tiles when M/N/K don't divide evenly

### R003 — Hardware im2col for All Conv/Lifting Kernel Configs
- Class: core-capability
- Status: active
- Description: Hardware address generation unit that linearizes conv2d/conv1d feature maps into GEMM-compatible layout for kernel sizes 2x7 (conv1), 1x7 (conv2), and 1x17 (lifting)
- Why it matters: Software im2col adds 350-700 us overhead, nearly the entire 500 us budget
- Source: user
- Primary owning slice: M001/S03
- Supporting slices: none
- Validation: mapped
- Notes: Must handle stride and padding variations across layers

### R004 — Full 38-Op Pipeline Controller
- Class: core-capability
- Status: active
- Description: Top-level controller that sequences all 38 operations in AWN inference through the systolic GEMM and non-GEMM modules
- Why it matters: The systolic array is useless without orchestration across the full inference pipeline
- Source: user
- Primary owning slice: M001/S04
- Supporting slices: none
- Validation: mapped
- Notes: Must integrate with existing 9 non-GEMM RTL modules without modifying them

### R005 — AXI-Lite Control + AXI-DMA PS-PL Interface
- Class: core-capability
- Status: active
- Description: AXI-Lite register interface for inference control (start/status/error) and AXI-DMA for streaming IQ data in and logits out
- Why it matters: Required for Zynq PS-PL integration; ARM must feed IQ data and read classification results
- Source: user
- Primary owning slice: M001/S05
- Supporting slices: none
- Validation: mapped
- Notes: Standard Zynq pattern

### R006 — End-to-End Latency <500 us at 200 MHz
- Class: quality-attribute
- Status: active
- Description: Total AWN inference cycle count must be <100,000 cycles at 200 MHz, proven in iverilog simulation
- Why it matters: WiFi burst-level classification requires sub-millisecond latency
- Source: user
- Primary owning slice: M001/S05
- Supporting slices: M001/S01, M001/S02, M001/S04
- Validation: mapped
- Notes: 500 us = 100K cycles at 200 MHz

### R007 — Strict Bit-Exact Match vs Behavioral Refmodel
- Class: quality-attribute
- Status: active
- Description: Every GEMM output through the systolic path must be byte-identical to the behavioral gemm_s8 output — zero tolerance
- Why it matters: Any divergence indicates a hardware bug; output-stationary accumulation order matches behavioral model exactly
- Source: user
- Primary owning slice: M001/S01
- Supporting slices: M001/S02, M001/S04
- Validation: mapped
- Notes: Same accumulation order guarantees zero rounding divergence

### R008 — Per-PE Unit Tests + Randomized Matrix Tests
- Class: quality-attribute
- Status: active
- Description: Individual PE verification (MAC + accumulate + requantize) and randomized matrix multiply tests beyond the fixed hex vectors
- Why it matters: Hex vectors only cover the AWN workload; randomized tests catch corner cases in accumulation and boundary handling
- Source: user
- Primary owning slice: M001/S01
- Supporting slices: none
- Validation: mapped
- Notes: User explicitly requested additional verification beyond existing hex pattern

### R009 — Per-Layer Cycle Count Breakdown
- Class: quality-attribute
- Status: active
- Description: Cycle counter per GEMM invocation showing time spent in conv1, conv2, each lifting level, and FC layers
- Why it matters: Identifies bottlenecks; useful for thesis/paper; cheap to implement
- Source: user
- Primary owning slice: M001/S04
- Supporting slices: M001/S05
- Validation: mapped
- Notes: Aggregate and per-layer counts

### R010 — Dedicated Dual-Port BRAMs for Row/Column Feeders
- Class: core-capability
- Status: active
- Description: Separate dual-port BRAM blocks for A-matrix (row feeder) and B-matrix (column feeder) to support simultaneous reads every cycle
- Why it matters: Single-port memory would halve systolic array throughput
- Source: inferred
- Primary owning slice: M001/S01
- Supporting slices: M001/S02
- Validation: mapped
- Notes: Standard FPGA systolic array practice

### R016 — Non-GEMM RTL Modules Unchanged
- Class: constraint
- Status: active
- Description: All 9 existing RTL modules (requantize_s32_s8, leaky_relu_s8, relu_s8, avgpool1d_s8, eltwise_addsub_s8, mul_s8, lut_s8, global_buffer) must remain unmodified and pass their existing testbenches
- Why it matters: These modules are verified and correct; changes risk regressions
- Source: user
- Primary owning slice: M001/S04
- Supporting slices: none
- Validation: mapped
- Notes: Quality bar from discussion

### R017 — Tiling Boundary Zero-Padding
- Class: core-capability
- Status: active
- Description: When M, N, or K don't divide evenly by tile dimensions (8, 16), the FSM zero-pads trailing tiles and masks writes for out-of-bounds output positions
- Why it matters: All real AWN GEMM shapes have non-aligned dimensions; incorrect padding produces wrong results
- Source: user
- Primary owning slice: M001/S02
- Supporting slices: none
- Validation: mapped
- Notes: Error handling decision from Layer 3

### R018 — Hardware im2col Byte-for-Byte Match
- Class: quality-attribute
- Status: active
- Description: Hardware im2col output must match software im2col byte-for-byte for all kernel configurations (2x7, 1x7, 1x17)
- Why it matters: Any mismatch cascades to wrong GEMM results
- Source: user
- Primary owning slice: M001/S03
- Supporting slices: none
- Validation: mapped
- Notes: Quality bar from discussion

### R019 — int32 Accumulators in PEs
- Class: constraint
- Status: active
- Description: Each PE accumulates int8*int8 partial products into int32, preventing overflow for K up to 2^15 products
- Why it matters: int16 accumulators would overflow for K=320; int32 handles all AWN GEMM shapes with margin
- Source: user
- Primary owning slice: M001/S01
- Supporting slices: none
- Validation: mapped
- Notes: Error handling decision from Layer 3

### R020 — DMA Error Flag Detection
- Class: failure-visibility
- Status: active
- Description: AXI-DMA error flags (SlvErr, DecErr) are checked after each transfer; error status register readable by PS
- Why it matters: Silent DMA failures would produce garbage classification results
- Source: user
- Primary owning slice: M001/S05
- Supporting slices: none
- Validation: mapped
- Notes: No automatic retry — PS decides

## Deferred

### R011 — Pynq-Z2 Board Deployment + Vivado Synthesis
- Class: core-capability
- Status: deferred
- Description: Synthesize design in Vivado, meet timing at 200 MHz, deploy to real Pynq-Z2 hardware
- Why it matters: Simulation proof doesn't guarantee real hardware works
- Source: user
- Primary owning slice: none
- Supporting slices: none
- Validation: unmapped
- Notes: Next milestone; depends on M001 completion

### R012 — Power/Thermal Optimization
- Class: quality-attribute
- Status: deferred
- Description: Optimize power consumption and thermal profile for sustained operation
- Why it matters: WiFi classification runs continuously; power matters for deployment
- Source: inferred
- Primary owning slice: none
- Supporting slices: none
- Validation: unmapped
- Notes: Deferred until board deployment

### R013 — Linux Runtime Driver
- Class: operability
- Status: deferred
- Description: Linux kernel driver or userspace DMA interface for production deployment
- Why it matters: Board deployment needs software integration beyond bare-metal
- Source: inferred
- Primary owning slice: none
- Supporting slices: none
- Validation: unmapped
- Notes: Deferred until board deployment

## Out of Scope

### R014 — Accuracy Improvement / Retraining AWN Model
- Class: anti-feature
- Status: out-of-scope
- Description: No changes to the AWN model architecture, weights, or training pipeline
- Why it matters: This milestone is about hardware acceleration, not model improvement
- Source: inferred
- Primary owning slice: none
- Supporting slices: none
- Validation: n/a
- Notes: Model is fixed; hardware must match its behavior exactly

### R015 — Support for Non-AWN Architectures
- Class: anti-feature
- Status: out-of-scope
- Description: The systolic array and pipeline are designed specifically for AWN inference shapes
- Why it matters: Prevents scope creep toward a general-purpose accelerator
- Source: inferred
- Primary owning slice: none
- Supporting slices: none
- Validation: n/a
- Notes: Parameterized RTL could be reused, but generalization is not a goal

## Traceability

| ID | Class | Status | Primary owner | Supporting | Proof |
|---|---|---|---|---|---|
| R001 | core-capability | active | M001/S01 | none | mapped |
| R002 | core-capability | active | M001/S02 | none | mapped |
| R003 | core-capability | active | M001/S03 | none | mapped |
| R004 | core-capability | active | M001/S04 | none | mapped |
| R005 | core-capability | active | M001/S05 | none | mapped |
| R006 | quality-attribute | active | M001/S05 | S01, S02, S04 | mapped |
| R007 | quality-attribute | active | M001/S01 | S02, S04 | mapped |
| R008 | quality-attribute | active | M001/S01 | none | mapped |
| R009 | quality-attribute | active | M001/S04 | S05 | mapped |
| R010 | core-capability | active | M001/S01 | S02 | mapped |
| R016 | constraint | active | M001/S04 | none | mapped |
| R017 | core-capability | active | M001/S02 | none | mapped |
| R018 | quality-attribute | active | M001/S03 | none | mapped |
| R019 | constraint | active | M001/S01 | none | mapped |
| R020 | failure-visibility | active | M001/S05 | none | mapped |
| R011 | core-capability | deferred | none | none | unmapped |
| R012 | quality-attribute | deferred | none | none | unmapped |
| R013 | operability | deferred | none | none | unmapped |
| R014 | anti-feature | out-of-scope | none | none | n/a |
| R015 | anti-feature | out-of-scope | none | none | n/a |

## Coverage Summary

- Active requirements: 15
- Mapped to slices: 15
- Validated: 0
- Unmapped active requirements: 0
