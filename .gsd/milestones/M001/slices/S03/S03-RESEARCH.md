# Slice S03 Research: Hardware im2col Unit

## Summary of Findings

The AWN integer forward pass (refmodel.py) performs **6 convolution-as-GEMM operations** that require im2col transformation. These use **4 distinct im2col configurations** (not 3 as originally stated in the slice description). The task description's kernel sizes (2x7, 1x7, 1x17) are partially incorrect -- the actual kernels from the codebase are **2x7 (conv1), 1x5 (conv2), 1x3 (lifting)**. The 4 additional FC/SE layers are pure matrix-vector multiplies and do NOT require im2col.

**Recommendation:** Implement a single parameterized `im2col_addr_gen` FSM that accepts configuration registers (C_in, T_in, kH, kW, pad_left, pad_mode) and generates one im2col column per output position. This avoids duplicating logic for each kernel config. The FSM outputs either a BRAM read address or a zero-flag, one element per cycle, producing K elements per output column and N columns total.

**Key discovery:** Reflection padding (used by lifting conv1 ops) is the hardest part. Only 4 out of 66 output columns touch reflected positions, so a small boundary-correction circuit suffices. The rest is straight address arithmetic.

---

## Implementation Landscape

### Corrected Kernel Configurations (from code, not task description)

| Layer | PyTorch Op | Kernel | Stride | Padding | Source |
|-------|-----------|--------|--------|---------|--------|
| conv1 | Conv2d(1, 64, (2,7)) | 2x7 | (1,1) | ZeroPad2d(3,3,0,0) pre-applied | `models/model.py:69-73` |
| conv2 | Conv1d(64, 64, 5) | 1x5 | 1 | zero, pad=2 | `models/model.py:77-78` |
| lifting conv1 (U.op.1, P.op.1) | Conv1d(64, 64, 3) | 1x3 | 1 | reflect, pad=2 | `models/lifting.py:27-34` |
| lifting conv2 (U.op.4, P.op.4) | Conv1d(64, 64, 3) | 1x3 | 1 | none | `models/lifting.py:36-38` |

**Note:** The task description listed "1x7 (conv2)" and "1x17 (lifting)" but the actual code uses **1x5** and **1x3** respectively. The lifting `pad = (kernel_size - 1) // 2 + 1 = 2` for kernel_size=3.

### All 6 im2col Operations with Exact GEMM Shapes

| # | Layer | im2col Type | Input Shape | K (=Cin*kH*kW) | N (=output positions) | GEMM (M,K,N) | MACs |
|---|-------|-------------|-------------|---|----|--------------|------|
| 1 | conv1 | 2D, pre-padded | (1,2,134) | 14 | 128 | (64,14,128) | 114,688 |
| 2 | conv2 | 1D, zero pad=2 | (64,128) | 320 | 128 | (64,320,128) | 2,621,440 |
| 3 | U.op.1 | 1D, reflect pad=2 | (64,64) | 192 | 66 | (64,192,66) | 811,008 |
| 4 | U.op.4 | 1D, no pad | (64,66) | 192 | 64 | (64,192,64) | 786,432 |
| 5 | P.op.1 | 1D, reflect pad=2 | (64,64) | 192 | 66 | (64,192,66) | 811,008 |
| 6 | P.op.4 | 1D, no pad | (64,66) | 192 | 64 | (64,192,64) | 786,432 |

FC/SE layers (SE Linear x2, fc.0, fc.2) use direct GEMM with no im2col -- they are matrix-vector multiplies with shapes like (32,128,1), (128,32,1), (320,128,1), (11,320,1).

### 4 Distinct im2col Address Generation Configurations

#### Config A: conv1 (2D, pre-padded input)
- **Input BRAM:** 268 bytes (2 rows x 134 cols, zeros already in BRAM)
- **Output:** 14 elements/column x 128 columns
- **Address formula:** `addr = kh * 134 + wout + kw` (kh=0..1, kw=0..6)
- **Zero generation:** None needed (input pre-padded by PS before im2col starts)
- **Simplest config** -- pure sliding window over pre-padded data

#### Config B: conv2 (1D, zero padding)
- **Input BRAM:** 8192 bytes (64 channels x 128 timepoints)
- **Output:** 320 elements/column x 128 columns
- **Address formula:** `original_pos = t + kw - 2; addr = cin * 128 + original_pos`
- **Zero generation:** When `original_pos < 0` or `original_pos >= 128`
- **Zero positions:** Only at boundaries: t=0 (kw=0,1), t=1 (kw=0), t=126 (kw=4), t=127 (kw=3,4)
- **Largest im2col** -- dominates cycle count (320 * 128 = 40,960 address cycles)

#### Config C: lifting conv1 (1D, reflection padding)
- **Input BRAM:** 4096 bytes (64 channels x 64 timepoints)
- **Output:** 192 elements/column x 66 columns
- **Address formula:** `p = t + kw - 2; reflect(p, 64); addr = cin * 64 + p`
- **Reflection logic:**
  - If `p < 0`: `p = -p` (e.g., -1 -> 1, -2 -> 2)
  - If `p >= 64`: `p = 2*63 - p` (e.g., 64 -> 62, 65 -> 61)
- **Boundary columns needing reflection:** Only t=0, t=1 (left), t=64, t=65 (right)
- **Used 2x** (U.op.1 and P.op.1 with identical parameters)

#### Config D: lifting conv2 (1D, no padding)
- **Input BRAM:** 4224 bytes (64 channels x 66 timepoints)
- **Output:** 192 elements/column x 64 columns
- **Address formula:** `addr = cin * 66 + t + kw`
- **Zero generation:** None needed (all positions valid)
- **Used 2x** (U.op.4 and P.op.4)

### Key Files

| File | Role | What to use |
|------|------|-------------|
| `awn_fpga/sw/refmodel.py` | Software reference with all im2col calls | Gold reference for address patterns and GEMM shapes |
| `awn_fpga/sw/iohex.py` | Hex I/O helpers | Pattern for writing test vectors |
| `awn_fpga/rtl/gemm_s8.v` | Behavioral GEMM | Interface pattern (start/done, M/K/N params, BRAM buffers) |
| `awn_fpga/rtl/global_buffer.v` | Parameterized BRAM | Memory interface (addr_bits, data_bits, wr_en, index, data_in/out) |
| `awn_fpga/tb/tb_gemm_s8.v` | GEMM testbench | Pattern for plusargs, $readmemh loading, $fwrite output, cycle counting |
| `awn_fpga/vectors/` | 126 hex test vectors | Format reference (int8: 2-char hex/line, int32: 8-char hex/line) |
| `models/lifting.py` | Lifting scheme PyTorch | Authoritative kernel/pad params |
| `models/model.py` | AWN model PyTorch | Authoritative conv1/conv2 params |

### What to Create

| File | Purpose |
|------|---------|
| `awn_fpga/rtl/im2col_addr_gen.v` | Parameterized im2col address generator FSM |
| `awn_fpga/tb/tb_im2col_addr_gen.v` | Testbench: loads feature map hex, runs im2col, writes output hex |
| `awn_fpga/sw/gen_im2col_vectors.py` | Python script generating golden im2col hex vectors for all 4 configs |

---

## Build Order

### Phase 1: Config D (no padding, simplest)
1. Implement the core FSM with counters (cin, kw, t_out) and address computation `cin * T_in + t_out + kw`
2. Generate golden vectors from refmodel.py for U.op.4 im2col (input: 64x66, k=3, no pad -> output: 192x64)
3. Verify byte-for-byte match in iverilog

**Why first:** No padding logic at all. Pure nested-loop address generation. Validates the core FSM structure, BRAM interface, and hex comparison flow before adding complexity.

### Phase 2: Config B (zero padding)
1. Add zero-flag output and boundary comparison logic: `(t_out + kw - pad_left) < 0 || >= T_in`
2. Generate golden vectors for conv2 im2col (input: 64x128, k=5, zero_pad=2 -> output: 320x128)
3. Verify byte-for-byte match

**Why second:** Zero padding is the most common padding type and relatively simple (just a comparator). This is also the largest im2col (40,960 elements) so it stress-tests the FSM.

### Phase 3: Config C (reflection padding)
1. Add reflection address mapping: `if pos < 0: pos = -pos; if pos >= T_in: pos = 2*(T_in-1) - pos`
2. Generate golden vectors for U.op.1 im2col (input: 64x64, k=3, reflect_pad=2 -> output: 192x66)
3. Verify byte-for-byte match

**Why third:** Reflection is the most complex padding mode. Only 4 of 66 columns need it, but the hardware must handle it correctly.

### Phase 4: Config A (2D im2col)
1. Add kH counter and 2D address formula: `kh * W_in + t_out + kw`
2. Generate golden vectors for conv1 im2col (input: 2x134 pre-padded, k=(2,7) -> output: 14x128)
3. Verify byte-for-byte match

**Why last:** This is the only 2D im2col and actually the simplest operation (tiny K=14, no zero generation because input is pre-padded). The 2D loop adds one more counter (kh) but no boundary logic.

### Phase 5: Integration test
1. Run all 6 im2col operations in sequence, compare each against refmodel.py golden outputs
2. Measure cycle counts per config
3. Verify total im2col cycle overhead fits within the 200 LUT / minimal-latency budget

---

## Verification Approach

### Per-Config Verification (Phases 1-4)

For each of the 4 configs:

1. **Generate golden vectors** (Python script `gen_im2col_vectors.py`):
   - Load `build/quant.npz` for the quantized feature map data
   - Run `im2col_1d()` or `im2col_2d()` from refmodel.py
   - Write input feature map as `im2col_cfgX_input.hex` (int8, 2-char per line)
   - Write expected im2col output as `im2col_cfgX_golden.hex` (int8, 2-char per line, column-major order matching FSM output order)

2. **Testbench pattern** (following `tb_gemm_s8.v`):
   ```
   - $value$plusargs for config params (C_in, T_in, kH, kW, pad_left, pad_mode)
   - $readmemh to load feature map into a BRAM model
   - Instantiate im2col_addr_gen, connect to BRAM
   - Start FSM, collect output stream (addr + zero_flag) cycle by cycle
   - For each output: if zero_flag, emit 0x00; else emit BRAM[addr]
   - $fwrite collected output to result hex file
   - Compare result hex against golden hex (assert in testbench or diff externally)
   ```

3. **Cycle count verification:**
   - Expected cycles per config = K * N + small overhead (FSM transitions)
   - Config A: 14 * 128 = 1,792 cycles
   - Config B: 320 * 128 = 40,960 cycles
   - Config C: 192 * 66 = 12,672 cycles
   - Config D: 192 * 64 = 12,288 cycles

### End-to-End Verification (Phase 5)

Run refmodel.py but replace the `im2col_2d`/`im2col_1d` calls with hardware simulation:
1. For each GEMM step that uses im2col, write the input feature map to hex
2. Run im2col_addr_gen in iverilog to produce the B matrix hex
3. Feed B matrix hex into existing `tb_gemm_s8` along with weight A matrix
4. Compare GEMM output against refmodel.py golden output
5. This proves: **hw_im2col + hw_gemm == sw_im2col + sw_gemm byte-for-byte**

---

## Constraints from Existing Code

### Memory Interface
- `global_buffer.v`: Single-port BRAM, negedge-triggered, 1-cycle read latency
- im2col must account for 1-cycle read latency (emit address on cycle N, get data on cycle N+1)
- Address width: 14 bits sufficient (max 8192 for conv2 input, but 14 bits = 16384 gives headroom)
- Data width: 8 bits (int8)

### Hex Vector Format
- int8: one byte per line, 2 hex chars, lowercase (e.g., `f6`)
- int32: one word per line, 8 hex chars, lowercase (e.g., `ffffffd0`)
- Row-major flattening (C convention)
- `$readmemh` for loading, `$fwrite` for writing

### GEMM Interface Expectations
- `gemm_s8.v` expects A in `a_buf[m*K + k]` and B in `b_buf[k*N + n]`
- im2col output is the B matrix, stored row-major: B[k][n] at index `k*N + n`
- The im2col FSM must output elements in the order they would be written to `b_buf`: iterate k fastest (within a column), then n
- Actually reviewing refmodel.py more carefully: `im2col_1d` returns `(K, N)` array. The GEMM call is `pipe.gemm(A_weight, B_im2col)` where A is `(M, K)` and B is `(K, N)`. So B[k][n] maps to `b_buf[k * N + n]`.

### Testbench Pattern
- Use `$value$plusargs` for all configuration parameters
- Use `$readmemh` to load input data
- Use `$fwrite` to write output data
- Report cycle count via `$display`
- Match existing naming convention: `tb_im2col_addr_gen.v`

### Resource Budget
- Target: ~200 LUTs (from systolic_optimization.md Section 8)
- No DSP48s needed (pure address arithmetic)
- No BRAMs needed (im2col reads from existing feature map BRAM, doesn't allocate its own)
- Counters needed: cin (6 bits, max 64), kw (3 bits, max 7), kh (1 bit, max 2), t_out (7 bits, max 128)
- One 14-bit multiplier for `cin * T_in` (can be replaced with shift-add since T_in is known at config time)

---

## Common Pitfalls

### 1. Off-by-One in Padding Boundaries
- **Zero padding:** The condition is `original_pos < 0 OR original_pos >= T_in`, NOT `> T_in`
- **Reflection padding:** `pos >= T_in` maps to `2*(T_in-1) - pos`, NOT `2*T_in - pos`
- Test: Verify boundary columns explicitly (first 2 and last 2 output columns for each padded config)

### 2. Reflection Padding Edge Cases
- Reflection with pad=2 and T=64: positions -2,-1 map to 2,1 (NOT 1,2)
- Right boundary: position 64 maps to 62, position 65 maps to 61
- Hardware implementation: `reflected = pos < 0 ? -pos : (pos >= T ? 2*(T-1) - pos : pos)`
- This needs a signed comparison for the left boundary check

### 3. Column vs Row Major Ordering
- im2col output is (K, N) stored row-major: element [k][n] at offset `k*N + n`
- But the FSM naturally iterates column-by-column (all K elements for position n=0, then n=1, ...)
- When writing to b_buf for GEMM, must scatter: element at (k, n) goes to address `k*N + n`
- Alternative: if the systolic array can consume column-by-column, no scatter needed

### 4. Multiplication for Address Computation
- `cin * T_in` is up to 63 * 128 = 8064, fits in 13 bits
- Hardware multiplier costs LUTs. Optimization: since T_in is fixed per layer, use shift-add
  - T_in=128: `cin << 7`
  - T_in=134: `cin << 7 + cin << 2 + cin << 1` (128+4+2=134)
  - T_in=64: `cin << 6`
  - T_in=66: `cin << 6 + cin << 1` (64+2=66)
- Alternatively, maintain a running base address and increment by T_in when cin advances

### 5. Pre-Padding for Conv1
- Conv1 input is pre-padded by PS before im2col. The zeros at positions [0..2] and [131..133] in each row are already in BRAM.
- The im2col unit does NOT need to generate zeros for conv1 -- it just reads from the pre-padded BRAM.
- This is a design decision: keep it this way (simpler im2col) vs. handle conv1 padding in hardware too.

### 6. 1-Cycle BRAM Read Latency
- `global_buffer.v` reads on negedge: address presented at posedge, data available at next negedge
- im2col FSM must pipeline: compute address in cycle N, capture data in cycle N+1
- For zero-padding positions, skip the BRAM read entirely and emit 0x00 directly
- This means the FSM output has 1 cycle of latency from address generation to data availability

### 7. Stride
- All AWN convolutions use stride=1. The im2col formulas assume stride=1.
- If stride support is needed later, the output position advance changes from `t+1` to `t+stride`
- For now, hardcode stride=1 to save LUTs.

### 8. Config Register Loading
- The im2col FSM needs to know (C_in, T_in, kH, kW, pad_left, pad_right, pad_mode, N_out) per layer
- These can be loaded via config registers written by the orchestrator FSM before each GEMM call
- Total config: ~8 registers x 8-16 bits = minimal

---

## Resource Estimate

| Component | LUTs (est.) | Notes |
|-----------|-------------|-------|
| Counters (cin, kh, kw, t_out) | 30 | 6+1+3+7 bit counters with compare |
| Address arithmetic (add, mux) | 60 | Base addr + offset, mux for padding modes |
| Boundary comparison (zero/reflect) | 40 | Signed comparators for pad boundaries |
| Reflection mapping | 30 | Negate or 2*(T-1)-pos |
| FSM state logic | 20 | 4-5 states |
| Config registers | 20 | 8 x ~10-bit registers |
| **Total** | **~200** | Within budget |

---

## Design Decision: Unified vs. Per-Config FSM

**Recommended: Unified parameterized FSM.** All 4 configs share the same nested-loop structure (iterate cin, kh, kw for each output position t_out). The differences are:
1. Whether kh loop exists (only Config A; set kH=1 for 1D configs)
2. Padding mode (none, zero, reflect)
3. Specific parameter values (C_in, T_in, kW, pad)

A unified FSM with a 2-bit `pad_mode` register (00=none, 01=zero, 10=reflect) handles all 4 configs. The kH counter naturally collapses to a single iteration when kH=1. This uses fewer LUTs than separate FSMs and simplifies the orchestrator.
