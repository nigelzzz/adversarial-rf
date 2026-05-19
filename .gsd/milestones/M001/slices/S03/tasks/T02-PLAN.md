---
estimated_steps: 1
estimated_files: 2
skills_used: []
---

# T02: Add zero-padding and reflection-padding (Configs B and C)

Extend im2col_addr_gen.v to support zero-padding (Config B: conv2, 64x128 input, k=5, zero_pad=2 → 320x128 output) and reflection-padding (Config C: lifting conv1, 64x64 input, k=3, reflect_pad=2 → 192x66 output). Add pad_mode config register (00=none, 01=zero, 10=reflect). For zero padding: emit zero_flag=1 when original_pos < 0 or >= T_in. For reflection: compute reflected address pos < 0 ? -pos : (pos >= T_in ? 2*(T_in-1) - pos : pos). Boundary detection uses signed comparison for left boundary. Verify both configs against golden vectors. Config B is the largest im2col (40,960 elements) and stress-tests the FSM.

## Inputs

- `awn_fpga/vectors/im2col_cfgB_*.hex (golden vectors from T01)`
- `awn_fpga/vectors/im2col_cfgC_*.hex (golden vectors from T01)`

## Expected Output

- `awn_fpga/rtl/im2col_addr_gen.v (updated with padding modes)`
- `awn_fpga/vectors/im2col_cfgB_hw_out.hex`
- `awn_fpga/vectors/im2col_cfgC_hw_out.hex`

## Verification

cd awn_fpga && python sw/test_im2col.py --config B 2>&1 | grep -q PASS && python sw/test_im2col.py --config C 2>&1 | grep -q PASS
