---
estimated_steps: 1
estimated_files: 3
skills_used: []
---

# T01: Golden vector generator and core im2col FSM (Config D: no padding)

Create gen_im2col_vectors.py that uses refmodel.py's im2col_1d/im2col_2d functions with quant.npz data to generate golden hex vectors for all 4 im2col configs. Create im2col_addr_gen.v with the core unified FSM handling Config D (no padding, simplest case: lifting conv2, input 64x66, k=3, pad=0 → output 192x64). Create tb_im2col_addr_gen.v testbench that loads feature map hex into BRAM model, runs im2col FSM, collects output stream, writes result hex, and compares against golden. The FSM uses nested counters (cin, kw, t_out) with address computation cin*T_in + t_out + kw. Also handle kH counter (set kH=1 for 1D configs). Output interface: addr, zero_flag, valid, done. Verify Config D (U.op.4 and P.op.4) byte-exact against golden vectors.

## Inputs

- `awn_fpga/sw/refmodel.py (im2col_1d, im2col_2d functions)`
- `awn_fpga/build/quant.npz (quantized activations for golden vectors)`
- `awn_fpga/sw/iohex.py (hex I/O helpers)`

## Expected Output

- `awn_fpga/rtl/im2col_addr_gen.v`
- `awn_fpga/tb/tb_im2col_addr_gen.v`
- `awn_fpga/sw/gen_im2col_vectors.py`
- `awn_fpga/vectors/im2col_cfgD_*.hex`

## Verification

cd awn_fpga && python sw/gen_im2col_vectors.py && iverilog -g2005-sv -o build/sim_im2col tb/tb_im2col_addr_gen.v rtl/im2col_addr_gen.v && python sw/test_im2col.py --config D 2>&1 | grep -q PASS
