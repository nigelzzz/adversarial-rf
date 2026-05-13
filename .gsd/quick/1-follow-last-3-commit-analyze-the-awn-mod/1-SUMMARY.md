# Quick Task: follow last 3 commit, analyze the awn model how to optimize it in fpga make the latency align wifi requirement, using systolic array tpu to improve

**Date:** 2026-05-13
**Branch:** main

## What Changed
- Created comprehensive FPGA optimization analysis (`awn_fpga/systolic_optimization.md`) that:
  - Reviewed last 3 commits: (1) `3ce58d7` added full awn_fpga pipeline with 9 Verilog RTL modules, quantization, and end-to-end int8 inference; (2) `4463987` added analysis notes on operator naming, lifting scheme theory, BRAM estimation; (3) `95b9248` added latency calculation notes showing 30-60 ms @ 100-200 MHz with 1 MAC/cycle behavioral GEMM
  - Defined WiFi latency requirements: 100-500 us for burst-level AMC classification
  - Identified the bottleneck: `gemm_s8` at 1 MAC/cycle consumes 97.2% of compute (5.98M MACs), with conv2 alone at 43.8%
  - Designed 8x16 output-stationary systolic array (128 PEs, 128 DSP48s) achieving ~355 us @ 200 MHz — within 500 us WiFi budget
  - Provided PE Verilog skeleton, tiling strategy (M-tiles x N-tiles with skewed feeding), and resource estimates
  - Showed 87x latency improvement over current behavioral design (30.8 ms -> 355 us)
  - Verified fit on Pynq-Z2 (58% DSP, 46% BRAM)
  - Compared systolic vs simple parallelism: systolic wins on BRAM port feasibility and routing
  - Included 4-phase implementation roadmap

## Files Modified
- `awn_fpga/systolic_optimization.md` (new) — Full optimization analysis with WiFi requirements, systolic array design, resource estimation, and roadmap

## Verification
- Cross-checked MAC counts against `build/ops.txt` and `build/ops.json`
- Verified tiling math: conv2 (M=64, K=320, N=128) with 8x16 array = 8*8 tiles * 342 cycles/tile = 21,888 cycles
- Confirmed DSP/BRAM fit on Zynq-7020 (220 DSP48, 140 BRAM36)
- Latency formula validated: 71,000 cycles / 200 MHz = 355 us < 500 us WiFi budget
