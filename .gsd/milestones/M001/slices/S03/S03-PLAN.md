# S03: Hardware im2col Unit

**Goal:** Unified parameterized im2col address generator FSM that produces correct linearized BRAM addresses for all 4 AWN kernel configurations (2x7 conv1, 1x5 conv2, 1x3 lifting conv1/conv2), supporting no-padding, zero-padding, and reflection-padding modes. Output matches software im2col from refmodel.py byte-for-byte for all 6 convolution-as-GEMM operations.
**Demo:** im2col FSM generates correct linearized addresses for all 3 kernel configs (2x7 conv1, 1x7 conv2, 1x17 lifting); output matches software im2col byte-for-byte via hex comparison

## Must-Haves

- 1. `cd awn_fpga && python sw/test_im2col.py` prints "ALL IM2COL TESTS PASSED" with all 4 configs verified.
- 2. All 6 AWN im2col operations tested against golden vectors from refmodel.py (conv1, conv2, U.op.1, U.op.4, P.op.1, P.op.4).
- 3. Each config verified byte-for-byte: Config D (no-pad, k=3, 64x66→192x64), Config B (zero-pad, k=5, 64x128→320x128), Config C (reflect-pad, k=3, 64x64→192x66), Config A (2D, k=2x7, 1x2x134→14x128).
- 4. Cycle counts reported per config and match expected K*N + overhead.

## Proof Level

- This slice proves: Contract (bit-exact im2col output against refmodel.py golden vectors for all 6 AWN convolution layers). Real runtime required: no (iverilog simulation only). Human/UAT required: no.

## Integration Closure

Upstream surfaces consumed: `awn_fpga/sw/refmodel.py` (golden reference for im2col address patterns and GEMM shapes), `awn_fpga/sw/iohex.py` (hex I/O helpers), `awn_fpga/build/quant.npz` (quantized weights and activations for golden vector generation).
New wiring introduced: `im2col_addr_gen.v` is a standalone address generator FSM with config registers (C_in, T_in, kH, kW, pad_left, pad_mode). It outputs (addr, zero_flag, valid, done) per cycle. S04 pipeline controller will instantiate it to feed the A-side of tiled_gemm_s8 for convolution layers.
What remains: S04 (pipeline controller sequences im2col + tiled_gemm for all 38 AWN ops), S05 (AXI interface).

## Verification

- Not provided.

## Tasks

- [x] **T01: Golden vector generator and core im2col FSM (Config D: no padding)** `est:3h`
  Create gen_im2col_vectors.py that uses refmodel.py's im2col_1d/im2col_2d functions with quant.npz data to generate golden hex vectors for all 4 im2col configs. Create im2col_addr_gen.v with the core unified FSM handling Config D (no padding, simplest case: lifting conv2, input 64x66, k=3, pad=0 → output 192x64). Create tb_im2col_addr_gen.v testbench that loads feature map hex into BRAM model, runs im2col FSM, collects output stream, writes result hex, and compares against golden. The FSM uses nested counters (cin, kw, t_out) with address computation cin*T_in + t_out + kw. Also handle kH counter (set kH=1 for 1D configs). Output interface: addr, zero_flag, valid, done. Verify Config D (U.op.4 and P.op.4) byte-exact against golden vectors.
  - Files: `awn_fpga/sw/gen_im2col_vectors.py`, `awn_fpga/rtl/im2col_addr_gen.v`, `awn_fpga/tb/tb_im2col_addr_gen.v`
  - Verify: cd awn_fpga && python sw/gen_im2col_vectors.py && iverilog -g2005-sv -o build/sim_im2col tb/tb_im2col_addr_gen.v rtl/im2col_addr_gen.v && python sw/test_im2col.py --config D 2>&1 | grep -q PASS

- [x] **T02: Add zero-padding and reflection-padding (Configs B and C)** `est:2h`
  Extend im2col_addr_gen.v to support zero-padding (Config B: conv2, 64x128 input, k=5, zero_pad=2 → 320x128 output) and reflection-padding (Config C: lifting conv1, 64x64 input, k=3, reflect_pad=2 → 192x66 output). Add pad_mode config register (00=none, 01=zero, 10=reflect). For zero padding: emit zero_flag=1 when original_pos < 0 or >= T_in. For reflection: compute reflected address pos < 0 ? -pos : (pos >= T_in ? 2*(T_in-1) - pos : pos). Boundary detection uses signed comparison for left boundary. Verify both configs against golden vectors. Config B is the largest im2col (40,960 elements) and stress-tests the FSM.
  - Files: `awn_fpga/rtl/im2col_addr_gen.v`, `awn_fpga/sw/test_im2col.py`
  - Verify: cd awn_fpga && python sw/test_im2col.py --config B 2>&1 | grep -q PASS && python sw/test_im2col.py --config C 2>&1 | grep -q PASS

- [x] **T03: Add 2D im2col (Config A) and comprehensive all-6-operations verification** `est:2h`
  Add kH loop support to im2col_addr_gen.v for 2D im2col (Config A: conv1, 1x2x134 pre-padded input, k=(2,7), no padding needed since input is pre-padded → 14x128 output). The 2D address formula is kh*W_in + t_out + kw. Create comprehensive test script test_im2col.py that verifies all 6 AWN im2col operations end-to-end: conv1 (Config A), conv2 (Config B), U.op.1 (Config C), U.op.4 (Config D), P.op.1 (Config C), P.op.4 (Config D). Report cycle counts per config. Create repo-root wrapper sw/test_im2col.py for GSD verification gate compatibility.
  - Files: `awn_fpga/rtl/im2col_addr_gen.v`, `awn_fpga/sw/test_im2col.py`, `sw/test_im2col.py`
  - Verify: cd awn_fpga && python sw/test_im2col.py 2>&1 | tail -1 | grep -q 'ALL IM2COL TESTS PASSED' && python /home/nigel/opensource/adversarial-rf/sw/test_im2col.py 2>&1 | tail -1 | grep -q 'ALL IM2COL TESTS PASSED'

## Files Likely Touched

- awn_fpga/sw/gen_im2col_vectors.py
- awn_fpga/rtl/im2col_addr_gen.v
- awn_fpga/tb/tb_im2col_addr_gen.v
- awn_fpga/sw/test_im2col.py
- sw/test_im2col.py
