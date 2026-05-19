---
estimated_steps: 1
estimated_files: 3
skills_used: []
---

# T03: Add 2D im2col (Config A) and comprehensive all-6-operations verification

Add kH loop support to im2col_addr_gen.v for 2D im2col (Config A: conv1, 1x2x134 pre-padded input, k=(2,7), no padding needed since input is pre-padded → 14x128 output). The 2D address formula is kh*W_in + t_out + kw. Create comprehensive test script test_im2col.py that verifies all 6 AWN im2col operations end-to-end: conv1 (Config A), conv2 (Config B), U.op.1 (Config C), U.op.4 (Config D), P.op.1 (Config C), P.op.4 (Config D). Report cycle counts per config. Create repo-root wrapper sw/test_im2col.py for GSD verification gate compatibility.

## Inputs

- `awn_fpga/build/quant.npz (real AWN activations for end-to-end test)`
- `awn_fpga/sw/refmodel.py (golden im2col reference)`

## Expected Output

- `awn_fpga/rtl/im2col_addr_gen.v (complete with all 4 configs)`
- `awn_fpga/sw/test_im2col.py (comprehensive test suite)`
- `sw/test_im2col.py (repo-root wrapper)`

## Verification

cd awn_fpga && python sw/test_im2col.py 2>&1 | tail -1 | grep -q 'ALL IM2COL TESTS PASSED' && python /home/nigel/opensource/adversarial-rf/sw/test_im2col.py 2>&1 | tail -1 | grep -q 'ALL IM2COL TESTS PASSED'
