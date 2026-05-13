# Project

## What This Is

AWN FPGA accelerator — a hardware inference engine for the Adaptive Wavelet Network (AWN) automatic modulation classifier. The project takes a pretrained AWN model (11-class AMC on RML2016.10a) and implements it as an int8 quantized FPGA pipeline targeting real-time WiFi burst-level classification (<500 us latency at 200 MHz).

## Core Value

End-to-end AWN inference in hardware fast enough for WiFi burst-level classification — proven bit-exact against the software reference model.

## Current State

- Full int8 quantization pipeline complete (BN folding, per-tensor symmetric, 124K parameters)
- 9 behavioral Verilog RTL modules implemented and verified bit-exact against numpy (gemm_s8, requantize_s32_s8, leaky_relu_s8, relu_s8, avgpool1d_s8, eltwise_addsub_s8, mul_s8, lut_s8, global_buffer)
- End-to-end inference orchestrator (refmodel.py, 38 op invocations) produces correct classification
- 126 hex test vectors for all operations
- Behavioral gemm_s8 achieves correct results but at 1 MAC/cycle — 30.8 ms latency at 200 MHz (61x over budget)
- Systolic array design analysis complete (8x16 output-stationary, tiling math, resource estimates)

## Architecture / Key Patterns

- **Quantization:** int8 weights/activations, int32 accumulation, requantize via shift+clamp
- **RTL style:** Parameterized Verilog modules, hex-file-based testbenches ($readmemh/$fwrite)
- **Verification:** Python numpy reference model → hex vectors → iverilog simulation → byte-for-byte comparison
- **Target:** Zynq-7020 (Pynq-Z2), 200 MHz clock, 220 DSP48s, 140 BRAM36Ks

## Capability Contract

See `.gsd/REQUIREMENTS.md` for the explicit capability contract, requirement status, and coverage mapping.

## Milestone Sequence

- [ ] M001: Systolic Array AWN Accelerator — Replace behavioral GEMM with 8x16 systolic array, hardware im2col, and AXI PS-PL interface; prove <500 us latency in simulation
