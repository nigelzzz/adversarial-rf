# Slice S01 Research: Systolic Array PE + 8x16 Mesh

## 1. Summary of Findings and Primary Recommendation

The existing AWN FPGA pipeline uses a behavioral `gemm_s8.v` that performs 1 MAC/cycle via a simple FSM iterating over (m, n, k) indices with int8 inputs, int32 accumulation, and row-major buffer storage. This module is the sole compute bottleneck -- 97.2% of the 5,983,680 total MACs are in 5 GEMM calls with shapes M=64, K=192-320, N=64-128.

**Primary recommendation:** Build an output-stationary 8x16 systolic mesh (128 PEs) that is a **drop-in replacement** for `gemm_s8.v` at the GEMM level. The PE performs `int8 x int8 -> int16 -> sign-extend -> int32 accumulate`, matching the exact arithmetic in the behavioral model. After accumulation completes for a tile, the int32 results are drained and either stored directly (for downstream `requantize_s32_s8`) or passed through the existing requantize module. The systolic array reuses the existing `requantize_s32_s8.v` module unchanged.

**Build order:** `pe_s8.v` (single PE) -> `systolic_mesh_s8.v` (8x16 grid) -> `bram_feeder_a.v` / `bram_feeder_b.v` (dual-port row/column feeders) -> `tb_pe_s8.v` + `tb_systolic_mesh_s8.v` (verification).

---

## 2. Implementation Landscape

### 2.1 Existing Files (What Exists)

| File | Purpose | Reuse Strategy |
|------|---------|----------------|
| `rtl/gemm_s8.v` | Behavioral GEMM, 1 MAC/cycle. Row-major A[M,K], B[K,N], C[M,N] int32 output. Uses `$signed` multiplication, `a_buf`/`b_buf`/`c_buf`/`bias_buf` arrays. | **Replace** with systolic. Keep as behavioral reference for verification. |
| `rtl/requantize_s32_s8.v` | TFLite-style requantization: `sat8(round((acc * mul) >> shift) + out_zp)`. Uses `in_buf`/`out_buf` arrays, processes 1 element/cycle. | **Reuse unchanged.** Systolic drains int32 accumulators into `in_buf` of this module. |
| `rtl/global_buffer.v` | Single-port parameterized RAM (negedge clock, synchronous read/write). From AIC2021 TPU project. | **Reference pattern only.** We need dual-port BRAMs for row/column feeders (D002 decision). Do NOT reuse this module directly. |
| `tb/tb_gemm_s8.v` | Testbench: loads hex via `$readmemh`, passes M/K/N/bias via `$value$plusargs`, dumps `c_buf` via `$fwrite` with `%08x` format. | **Follow this pattern** for systolic testbench. Same plusargs interface, same hex I/O format. |
| `sw/refmodel.py` | Python reference model. `Pipeline.gemm()` method writes A/B/bias hex, invokes iverilog, reads output hex, asserts bit-exact match against numpy `A.int32 @ B.int32 + bias`. | **Integration target.** The systolic array must produce identical c_buf contents so refmodel.py passes unchanged. |
| `sw/iohex.py` | Hex I/O helpers: `write_int8_hex` (2-char lowercase, one per line), `write_int32_hex` (8-char lowercase, one per line), readers, assertion helpers. | **Reuse unchanged.** All test vectors use this format. |
| `vectors/` | 126+ hex files. 10 GEMM invocations (steps 01, 04, 07, 10, 15, 18, 27, 30, 34, 37), each with `_a.hex` (int8), `_b.hex` (int8), `_bias.hex` (int32), `_out.hex` (int32). | **Golden references** for bit-exact verification of systolic against behavioral. |

### 2.2 Files to Create

| File | Purpose |
|------|---------|
| `rtl/pe_s8.v` | Single processing element: int8 x int8 MAC with int32 accumulator, registered passthrough of a_in/b_in. |
| `rtl/systolic_mesh_s8.v` | 8x16 PE grid with generate blocks, skewed feeding logic, accumulator clear, and output drain FSM. |
| `rtl/bram_feeder_a.v` | Dual-port BRAM for A-matrix rows. Port A: write (load from external). Port B: read (feed row data with skew). |
| `rtl/bram_feeder_b.v` | Dual-port BRAM for B-matrix columns. Port A: write (load from external). Port B: read (feed column data with skew). |
| `tb/tb_pe_s8.v` | Per-PE unit test: verify single MAC accumulation, passthrough, clear. |
| `tb/tb_systolic_mesh_s8.v` | Mesh testbench: randomized matrix tests + golden vector comparison. |

---

## 3. Detailed Analysis of Existing Components

### 3.1 gemm_s8.v — Behavioral GEMM (to be replaced)

**Interface:**
- Parameters: `A_LEN=65536`, `B_LEN=65536`, `C_LEN=16384`, `BIAS_LEN=1024`, `DIM_W=16`
- Inputs: `clk`, `rst_n` (active-low async reset), `start` (single-cycle pulse), `M_in`/`K_in`/`N_in` (16-bit), `use_bias`
- Output: `done` (single-cycle pulse)
- Storage: `a_buf` (int8), `b_buf` (int8), `bias_buf` (int32), `c_buf` (int32)

**Compute order (CRITICAL for bit-exactness):**
```
for m = 0 to M-1:
  for n = 0 to N-1:
    acc = bias[m] if use_bias else 0    // S_INIT state
    for k = 0 to K-1:
      acc += sign_extend(a_buf[m*K+k] * b_buf[k*N+n])  // S_MAC state
    c_buf[m*N+n] = acc                  // S_WRITE state
```

**Key arithmetic detail:** The multiply is `$signed(a_buf[...]) * $signed(b_buf[...])` producing a 16-bit signed product `prod16`, then sign-extended to 32 bits via `{{16{prod16[15]}}, prod16}`. This is added to `acc` (int32). No intermediate clamping -- pure int32 accumulation.

**Accumulation order:** K dimension iterated innermost, for a fixed (m, n). This means each output element accumulates k=0, k=1, ..., k=K-1 sequentially. The systolic array must produce the **exact same partial sum sequence** to guarantee bit-exact matching.

**Why this matters:** Integer addition is commutative and associative (no rounding), so accumulation order does NOT affect the result for int8*int8->int32 (product fits in 16 bits, sum of up to 65536 products fits in 32 bits). This means the systolic array can accumulate in any order and still match. This is a major simplification vs. floating-point.

### 3.2 requantize_s32_s8.v — Requantization (reused)

**Interface:**
- Parameters: `LEN=8192`, `ADDR_W=16`
- Inputs: `clk`, `rst_n`, `start`, `length` (16-bit), `mul` (signed 32-bit), `shift` (6-bit unsigned), `out_zp` (signed 8-bit), `act_mode` (2-bit: 0=none, 1=ReLU)
- Output: `done`
- Storage: `in_buf` (int32), `out_buf` (int8)

**Arithmetic:**
```
prod = acc * mul                    // 64-bit signed
half = (shift == 0) ? 0 : (1 << (shift-1))   // rounding half
shifted = (prod + half) >>> shift   // arithmetic right shift
biased = shifted + sign_extend(out_zp)
raw = clamp(biased, -128, 127)     // saturate to int8
acted = (act_mode == 1 && raw < out_zp) ? out_zp : raw   // ReLU
```

**Integration point:** The systolic array's c_buf (int32 accumulation results) must be loaded into `requantize_s32_s8.in_buf` before starting requantization. The existing pipeline already does this -- behavioral gemm writes to `c_buf`, then orchestrator copies to requantize's `in_buf`. The systolic array must expose its accumulated results in the same way.

### 3.3 global_buffer.v — Single-Port RAM (reference only)

Single-port RAM on negedge clock. Parameters: `ADDR_BITS`, `DATA_BITS`. Mutually exclusive read/write controlled by `wr_en`.

**Why NOT reused:** Decision D002 specifies dedicated dual-port BRAMs for feeders. The systolic array needs simultaneous read (feeding PEs) and write (loading next tile) for double-buffering. Single-port is insufficient.

**Design reference:** Use the same parameterized style (generate-friendly, `ADDR_BITS`/`DATA_BITS` parameters), but implement true dual-port with separate read/write ports and independent addresses.

### 3.4 Testbench Pattern (tb_gemm_s8.v)

**Key patterns to follow:**
1. **Plusargs for configuration:** `$value$plusargs("M=%d", M_arg)` etc.
2. **Hex file paths as plusargs:** `$value$plusargs("a=%s", a_path)`
3. **Direct hierarchy access for loading:** `$readmemh(a_path, DUT.a_buf)` -- loads test data directly into DUT internal arrays
4. **Direct hierarchy access for output:** `DUT.c_buf[k]` -- reads results directly from DUT internals
5. **Output format:** `$fwrite(fout, "%08x\n", DUT.c_buf[k] & 32'hffffffff)` -- 8-char hex, unsigned mask, one per line
6. **Timeout:** Based on worst-case cycle count `8 * M * K * N + 100000`
7. **Reset sequence:** `rst_n=1 -> wait -> rst_n=0 -> wait -> rst_n=1 -> negedge -> start=1 -> negedge -> start=0`

**For systolic testbench:** Same interface -- the testbench loads A/B into the systolic array's internal buffers (or feeder BRAMs), triggers start, waits for done, reads c_buf, writes hex output. This allows refmodel.py to use the systolic simulation binary as a drop-in replacement.

### 3.5 refmodel.py — Python Reference Pipeline

**10 GEMM invocations in the AWN forward pass:**

| Step | Layer | M | K | N | Bias | A size | B size | C size |
|------|-------|---|---|---|------|--------|--------|--------|
| 01 | conv1 (2D via im2col) | 64 | 14 | 128 | Yes | 896 | 1,792 | 8,192 |
| 04 | conv2 (1D via im2col) | 64 | 320 | 128 | Yes | 20,480 | 40,960 | 8,192 |
| 07 | U.op.1 (lifting conv1) | 64 | 192 | 66 | Yes | 12,288 | 12,672 | 4,224 |
| 10 | U.op.4 (lifting conv2) | 64 | 192 | 64 | Yes | 12,288 | 12,288 | 4,096 |
| 15 | P.op.1 (lifting conv1) | 64 | 192 | 66 | Yes | 12,288 | 12,672 | 4,224 |
| 18 | P.op.4 (lifting conv2) | 64 | 192 | 64 | Yes | 12,288 | 12,288 | 4,096 |
| 27 | SE Linear(128->32) | 32 | 128 | 1 | No | 4,096 | 128 | 32 |
| 30 | SE Linear(32->128) | 128 | 32 | 1 | No | 4,096 | 32 | 128 |
| 34 | fc.0 Linear(128->320) | 320 | 128 | 1 | Yes | 40,960 | 128 | 320 |
| 37 | fc.2 Linear(320->11) | 11 | 320 | 1 | Yes | 3,520 | 320 | 11 |

**Integration requirement:** `refmodel.py:Pipeline.gemm()` invokes `vvp(sim("gemm_s8"), ...)`. To validate the systolic array, we can either:
1. Replace `sim("gemm_s8")` with `sim("systolic_mesh_s8")` (requires same plusargs interface)
2. Add a parallel path that runs both and compares
3. Make the systolic testbench expose the same `a_buf`/`b_buf`/`c_buf` hierarchy names

Option 1 is cleanest for CI. Option 3 is simplest for initial development.

### 3.6 Hex Vector Format

- **int8:** 2 lowercase hex chars per line, unsigned encoding (e.g., `f6` = -10 signed, `3e` = 62 signed). Loaded via `$readmemh` into `reg signed [7:0]` arrays.
- **int32:** 8 lowercase hex chars per line, unsigned encoding (e.g., `fffff5b0` = -2640 signed). Loaded via `$readmemh` into `reg signed [31:0]` arrays.
- **One value per line**, no address prefixes, no comments.

---

## 4. Build Order

### Phase 1: PE (pe_s8.v + tb_pe_s8.v)

**What to build:**
- `pe_s8.v`: Single PE with ports `{clk, rst_n, en, acc_clear, a_in[7:0], b_in[7:0], a_out[7:0], b_out[7:0], acc[31:0]}`
- `en` signal gates whether the PE performs MAC and passthrough (needed for pipeline fill/drain)
- On `acc_clear`: `acc <= sign_extend(a_in * b_in)` (first product of new tile)
- On `!acc_clear && en`: `acc <= acc + sign_extend(a_in * b_in)`
- Registered outputs: `a_out <= a_in`, `b_out <= b_in` (1-cycle latency through PE)

**Verification (tb_pe_s8.v):**
1. Reset test: verify acc=0, a_out=0, b_out=0 after reset
2. Single MAC: feed a_in=X, b_in=Y with acc_clear=1, verify acc = X*Y
3. Accumulate: feed sequence of (a, b) pairs, verify running sum matches numpy
4. Passthrough: verify a_out/b_out are delayed by exactly 1 cycle
5. Overflow boundary: feed int8 extremes (-128 * -128 = 16384, repeated 256 times = 4,194,304, fits int32)
6. Signed edge cases: -128 * 127 = -16256, -128 * -128 = 16384

### Phase 2: Mesh (systolic_mesh_s8.v)

**What to build:**
- Parameters: `PM=8` (rows), `PN=16` (cols)
- 8x16 PE grid via `generate` blocks
- Wiring: `a_wire[row][col]` horizontal, `b_wire[row][col]` vertical
- Row 0 inputs: `a_in[0..PM-1]` (8 values per cycle from A feeder)
- Col 0 inputs: `b_in[0..PN-1]` (16 values per cycle from B feeder)
- `acc_clear` signal broadcast to all PEs
- Output drain: read `acc[row][col]` from PE grid after K cycles + fill time

**Skewed feeding (CRITICAL):**
- Row `m` data is delayed by `m` cycles (row 0 starts at cycle 0, row 7 at cycle 7)
- Column `n` data is delayed by `n` cycles (col 0 starts at cycle 0, col 15 at cycle 15)
- This ensures PE[m][n] receives `A[m, k-m]` and `B[k-n, n]` simultaneously for each k
- Total fill time: `PM + PN - 2 = 22` cycles before all PEs are computing

**Tiling FSM states:**
```
IDLE -> LOAD (load A/B tile into feeders) -> COMPUTE (K + PM + PN - 2 cycles) ->
DRAIN (read PM*PN accumulators) -> NEXT_TILE (increment tile indices) -> LOAD or DONE
```

**For initial implementation:** Start without tiling -- assume M <= PM and N <= PN (works for SE and FC layers). Add tiling FSM in a second pass for the large conv/lifting GEMMs.

### Phase 3: BRAM Feeders (bram_feeder_a.v, bram_feeder_b.v)

**What to build:**
- Dual-port BRAMs, true dual-port (port A write, port B read, independent addresses)
- A feeder: stores `PM` rows of K elements. Read port delivers `PM` values per cycle (one per row), with row `m` reading from offset `k - m` (skew).
- B feeder: stores `PN` columns of K elements. Read port delivers `PN` values per cycle (one per col), with col `n` reading from offset `k - n` (skew).
- Double-buffering: while current tile computes, next tile's data loads into alternate buffer.

**Size per feeder:**
- A feeder: PM * K_MAX = 8 * 512 = 4096 bytes per buffer (x2 for double-buffer = 8 KB)
- B feeder: PN * K_MAX = 16 * 512 = 8192 bytes per buffer (x2 = 16 KB)
- Total: 24 KB = ~6 BRAM36Ks

**Skew implementation options:**
1. **Address offset in feeder:** Each row/col port computes its own read address: `base + (cycle - row_or_col_offset)`. Feed zeros when `cycle < offset` (pipeline fill).
2. **Shift register at mesh input:** Load all data flat, use a shift register chain to delay inputs. Simpler feeder but more LUT usage.
3. **Pre-skewed storage:** Store data with skew already applied in BRAM. Simplest read logic but complex write logic.

**Recommendation:** Option 1 (address offset). Each feeder has PM (or PN) independent read address counters, each starting at a different cycle. This is clean and maps well to BRAM.

### Phase 4: Integration Testbench (tb_systolic_mesh_s8.v)

**What to build:**
- Same plusargs interface as `tb_gemm_s8.v`: `M`, `K`, `N`, `bias`, `a`, `b`, `bi`, `out`
- Same `$readmemh` loading pattern
- Same `$fwrite` output format
- Internal: loads A/B hex into feeder BRAMs, runs systolic computation, drains accumulators, writes c_buf
- Expose `a_buf`, `b_buf`, `c_buf` named arrays (or equivalent) so refmodel.py can use `$readmemh`/direct hierarchy access

**Test cases:**
1. **Tiny sanity:** 2x2 matrix, hand-computed expected output
2. **Per-PE identity:** 1x1 tile (single PE), verify single accumulation
3. **Randomized small:** Random 8x16 * 16x16 matrices, compare against numpy
4. **Golden vectors:** All 10 GEMM hex vectors from `vectors/` directory
5. **Edge cases:** K=1 (single MAC), M=1/N=1 (vector-matrix), large K=320
6. **Tiling:** 64x320x128 (conv2) requiring 8*8=64 tiles

---

## 5. Verification Approach

### 5.1 Three-Tier Verification (D004)

**Tier 1: Per-PE unit tests**
- Deterministic sequences with hand-computed expected values
- Edge cases: min/max int8, zero inputs, alternating signs
- Passthrough timing verification (1-cycle delay)
- `acc_clear` vs accumulate mode

**Tier 2: Randomized matrix tests**
- Generate random int8 A and B matrices of various sizes
- Compute reference C = A.int32 @ B.int32 in Python/numpy
- Write hex vectors, run systolic simulation, compare output
- Cover all 10 GEMM shapes from the AWN pipeline
- Statistical coverage: 100+ random matrices per shape

**Tier 3: End-to-end hex pipeline**
- Run `refmodel.py` with systolic simulation binary replacing behavioral GEMM
- All 38 ops must pass bit-exact assertion
- This is the ultimate correctness gate

### 5.2 Bit-Exact Guarantee

**Why bit-exact is achievable:** All arithmetic is integer (int8 multiply -> int16 product -> int32 accumulate). Integer arithmetic is order-independent (associative + commutative), so:
- Accumulating k=0..K-1 in forward order (behavioral) produces the same int32 sum as any other order
- The systolic array accumulates in the same forward order within each PE (k=0 arrives first due to skewed feeding), but even if order differed, the result would be identical
- No floating-point rounding to worry about

**The one risk:** If the systolic PE uses a different sign-extension pattern for the 16-bit product. The behavioral model does:
```verilog
wire signed [15:0] prod16 = $signed(a) * $signed(b);
wire signed [31:0] prod32 = {{16{prod16[15]}}, prod16};
acc <= acc + prod32;
```
The PE must use the exact same pattern. Since Verilog `$signed` multiplication of two 8-bit values produces a 16-bit result, and sign-extension to 32 bits is straightforward, this is guaranteed as long as we use `$signed` consistently.

### 5.3 Randomized Testing Strategy

```python
# In a new test script (or extension to refmodel.py):
for trial in range(100):
    M = random.choice([1, 8, 11, 32, 64, 128, 320])
    K = random.choice([1, 14, 32, 128, 192, 320])
    N = random.choice([1, 16, 32, 64, 66, 128])
    A = np.random.randint(-128, 128, (M, K), dtype=np.int8)
    B = np.random.randint(-128, 128, (K, N), dtype=np.int8)
    C_ref = A.astype(np.int32) @ B.astype(np.int32)
    # write hex, run systolic sim, read hex, assert match
```

---

## 6. Constraints from Existing Code

### 6.1 Interface Constraints

1. **Buffer naming:** Testbenches access internal arrays via hierarchy (`DUT.a_buf`, `DUT.c_buf`). The systolic module (or its testbench wrapper) must expose arrays with the same names, OR the testbench must be written to bridge between hex files and the systolic's actual storage.

2. **Reset convention:** Active-low asynchronous reset (`negedge rst_n`), consistent across all existing modules. Systolic PE and mesh must use the same convention.

3. **Clock edge:** All computation on `posedge clk` (except `global_buffer.v` which uses `negedge` -- this is an anomaly from the AIC2021 origin, not followed by other modules).

4. **Signal widths:** `DIM_W=16` for dimension inputs (M, K, N). The systolic module should use the same width.

5. **Start/done protocol:** Single-cycle `start` pulse triggers computation. Single-cycle `done` pulse signals completion. No handshaking.

### 6.2 Naming Conventions

- Module names: `snake_case` (e.g., `gemm_s8`, `requantize_s32_s8`, `pe_s8`)
- Testbenches: `tb_<module_name>.v`
- Parameters: `UPPER_CASE` (e.g., `A_LEN`, `DIM_W`)
- Signals: `snake_case` (e.g., `a_in`, `b_out`, `acc_clear`)
- State names: `S_IDLE`, `S_MAC`, etc.

### 6.3 Hex Format Constraints

- int8: `%02x` format (2 hex chars, lowercase)
- int32: `%08x` format (8 hex chars, lowercase, masked with `& 32'hffffffff`)
- One value per line, no blank lines within a file
- Row-major ordering: A[m*K+k] is the (m*K+k)-th line

### 6.4 Simulation Tool Constraints

- **iverilog only** (no Vivado simulation, no Verilator)
- Compile: `iverilog -g2005-sv -I rtl -I tb -o build/sim_<name> tb/tb_<name>.v`
- Run: `vvp build/sim_<name> +M=64 +K=320 +N=128 ...`
- iverilog limitations: no SystemVerilog interfaces, limited struct support, no `logic` type in all contexts. Stick to Verilog-2005 with generate blocks.

---

## 7. Common Pitfalls

### 7.1 Accumulation Overflow

**Maximum accumulation:** The worst case is K=320 with all inputs at int8 extremes. `(-128) * (-128) = 16,384`. Sum of 320 such products = `5,242,880`. This fits comfortably in int32 (max 2,147,483,647). Even K=65536 would be safe: `65536 * 16384 = 1,073,741,824 < 2^31`. So int32 accumulators are sufficient for all AWN shapes. No overflow risk.

### 7.2 Sign Extension in Multiply

The product of two `signed [7:0]` values in Verilog is `signed [15:0]` (16 bits). When adding to a 32-bit accumulator, the 16-bit product must be sign-extended to 32 bits. The behavioral model does this explicitly:
```verilog
wire signed [15:0] prod16 = $signed(a) * $signed(b);
wire signed [31:0] prod32 = {{16{prod16[15]}}, prod16};
```
Alternatively, if both operands and the result wire are declared `signed`, Verilog handles sign extension automatically:
```verilog
wire signed [7:0] a_in, b_in;
wire signed [31:0] acc_next = acc + a_in * b_in;  // Verilog auto-extends
```
However, to be safe and explicit (and match iverilog behavior), use the explicit sign-extension pattern from the behavioral model.

### 7.3 Skewed Feeding Off-by-One

The most common bug in systolic array implementations is off-by-one in the skew delay. For output-stationary:
- Row `m` should receive `A[m, 0]` at cycle `m` (not `m+1`)
- Col `n` should receive `B[0, n]` at cycle `n`
- PE[m][n] computes its first product at cycle `m + n` (when both A[m,0] and B[0,n] arrive)
- PE[m][n] computes product for k at cycle `m + n + k`
- Total compute cycles: `K + PM + PN - 2` (last PE[PM-1][PN-1] finishes at cycle `K + PM + PN - 3`, so K valid products, first at cycle PM+PN-2, last at cycle K+PM+PN-3)

**Feeding zeros during fill:** During cycles 0..m-1, row m should receive `a_in = 0`. During cycles 0..n-1, col n should receive `b_in = 0`. These zeros do not affect accumulation (0 * anything = 0), but `acc_clear` timing must account for this -- clear should happen when the first REAL data arrives, not when zeros flow through.

**Alternative (simpler):** Do NOT use `acc_clear` timed per-PE. Instead, clear ALL accumulators at the start of each tile (one global `acc_clear` pulse before feeding begins), then let the zero-padding naturally contribute nothing. This avoids per-PE clear timing entirely.

### 7.4 Requantize Integration

The existing pipeline assumes `c_buf` is a flat array indexed by `[m*N+n]`. After systolic computation, accumulators are distributed across `PM*PN` PEs. The drain phase must read them out in the correct order (row-major: PE[0][0], PE[0][1], ..., PE[0][PN-1], PE[1][0], ...) and store into a flat c_buf array that requantize can consume.

### 7.5 Tiling Accumulation

For tiles where M > PM or N > PN, each tile computes a PARTIAL result for a subset of output elements. The accumulator for PE[m][n] corresponds to output element C[m_tile*PM + m, n_tile*PN + n]. Between tiles that map to DIFFERENT output elements, accumulators must be cleared. Between tiles that map to the SAME output elements (only possible if we tile K, which we don't need for AWN since K fits in one pass), accumulators would need to be preserved.

For AWN: K is always streamed in one pass (K_MAX=320, no K-tiling needed). So each tile produces a complete output sub-matrix, and accumulators are always cleared between tiles.

### 7.6 Bias Addition

The behavioral model adds bias in `S_INIT`: `acc <= use_bias ? bias_buf[m] : 0`. This initializes the accumulator with bias before MAC begins. The systolic array can do the same by initializing PE accumulators with the appropriate bias value instead of zero. Since bias is per-row (broadcast across N), each PE in row `m` gets `bias[m_tile * PM + m]`.

**Alternative:** Add bias after accumulation (during drain). This is simpler and avoids complicating the PE clear logic. Since `C = A@B + bias` and bias is int32, just add it when draining: `c_buf[idx] = PE.acc + bias[m]`. The result is mathematically identical.

---

## 8. Open Risks

### 8.1 iverilog Performance

The behavioral GEMM for conv2 (M=64, K=320, N=128) takes 2.6M cycles. With iverilog's interpretation overhead, this runs in seconds. The systolic array testbench will be more complex (128 PEs, generate blocks, multiple feeders). iverilog may slow down significantly for large generate blocks. Mitigation: test with small matrices first, use the full pipeline vectors only for final validation.

### 8.2 iverilog Generate Block Support

iverilog's support for `generate for` blocks is generally good, but there are known edge cases with multi-dimensional arrays of module instances and hierarchical access to generated instances. The testbench may need to use `genvar` naming conventions like `DUT.row[0].col[0].pe_inst.acc` to access individual PE accumulators. Need to verify iverilog supports this.

### 8.3 BRAM Inference in iverilog

iverilog does not synthesize to BRAMs -- it simulates arrays as memory. For simulation correctness this is fine, but the dual-port BRAM modules should be written in a style that Vivado would infer as BRAM for eventual synthesis (separate always blocks for each port, synchronous read/write). This is a coding style concern, not a simulation risk.

### 8.4 Tiling Controller Complexity

The tiling FSM for M=64, PM=8 requires 8 M-tiles. For N=128, PN=16 requires 8 N-tiles. Total 64 tiles for conv2. The FSM must correctly manage:
- Tile index iteration (m_tile * n_tile nested loops)
- Feeder address computation per tile
- Accumulator clear between tiles
- Output drain ordering (which c_buf region to write)

This is the most complex FSM in the design and should be thoroughly tested with the golden vectors.

### 8.5 M Not Divisible by PM

For fc.0 (M=320, PM=8): 320/8 = 40 tiles, divides evenly. For fc.2 (M=11, PM=8): 11/8 = 1.375, need 2 M-tiles with the second tile only using 3 of 8 rows. The systolic array must handle this by masking unused rows (feed zeros, ignore their output). Similarly for N: SE.0 has N=1, PN=16, so 15 of 16 columns are unused.

For the SE/FC layers with N=1, the systolic array is massively underutilized (only 1 of 16 columns active). This is acceptable since these layers are <1% of total MACs. The alternative (bypassing systolic for small GEMMs and using behavioral) could be considered for optimization.

### 8.6 Double-Buffering Necessity

For initial implementation, single-buffering is sufficient -- load tile, compute, drain, load next tile. Double-buffering (load next tile while current computes) overlaps latency and is important for meeting the 355 us target but adds complexity. Recommend: single-buffer first, verify correctness, add double-buffering as optimization.

---

## 9. GEMM Shape Summary for Tiling

| Layer | M | K | N | M-tiles (PM=8) | N-tiles (PN=16) | Total tiles | Cycles/tile | Partial last M-tile | Partial last N-tile |
|-------|---|---|---|-----------------|-----------------|-------------|-------------|---------------------|---------------------|
| conv1 | 64 | 14 | 128 | 8 | 8 | 64 | 36 | No | No |
| conv2 | 64 | 320 | 128 | 8 | 8 | 64 | 342 | No | No |
| U.op1 | 64 | 192 | 66 | 8 | 5 | 40 | 214 | No | Yes (N%16=2) |
| U.op4 | 64 | 192 | 64 | 8 | 4 | 32 | 214 | No | No |
| P.op1 | 64 | 192 | 66 | 8 | 5 | 40 | 214 | No | Yes (N%16=2) |
| P.op4 | 64 | 192 | 64 | 8 | 4 | 32 | 214 | No | No |
| SE.0 | 32 | 128 | 1 | 4 | 1 | 4 | 150 | No | Yes (N%16=1) |
| SE.3 | 128 | 32 | 1 | 16 | 1 | 16 | 54 | No | Yes (N%16=1) |
| fc.0 | 320 | 128 | 1 | 40 | 1 | 40 | 150 | No | Yes (N%16=1) |
| fc.2 | 11 | 320 | 1 | 2 | 1 | 2 | 342 | Yes (M%8=3) | Yes (N%16=1) |

Key observations:
- **All M values are multiples of 8** except fc.2 (M=11, remainder 3). Only 1 GEMM has partial M-tiles.
- **N=66 (U.op1, P.op1):** 66/16 = 4 remainder 2. The 5th N-tile only uses 2 of 16 columns.
- **N=1 (SE, FC layers):** Only 1 column active. 15/16 columns idle. Acceptable for <1% MACs.
- **No K-tiling needed:** K_MAX=320 < any reasonable K buffer size. All K values fit in a single pass.

---

## 10. Recommended PE Design

```
module pe_s8 (
    input               clk,
    input               rst_n,
    input               en,         // gate MAC and passthrough
    input               acc_clear,  // clear accumulator (start of new tile)
    input  signed [7:0] a_in,       // activation from west
    input  signed [7:0] b_in,       // weight from north
    output reg signed [7:0]  a_out, // pass east (1-cycle delay)
    output reg signed [7:0]  b_out, // pass south (1-cycle delay)
    output reg signed [31:0] acc    // accumulated partial sum
);
    wire signed [15:0] prod = a_in * b_in;
    wire signed [31:0] prod32 = {{16{prod[15]}}, prod};

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            a_out <= 8'sd0;
            b_out <= 8'sd0;
            acc   <= 32'sd0;
        end else if (en) begin
            a_out <= a_in;
            b_out <= b_in;
            if (acc_clear)
                acc <= prod32;
            else
                acc <= acc + prod32;
        end
    end
endmodule
```

**Notes:**
- `en` gate prevents spurious accumulation during tile load/drain phases
- `acc_clear` resets accumulator to first product (not zero) to save 1 cycle; alternatively clear to zero and let first MAC add naturally (simpler, 0 cycle penalty since clear can happen during load)
- Explicit sign extension `{{16{prod[15]}}, prod}` matches behavioral model exactly
- Single DSP48 slice per PE for the multiply (accumulate uses fabric or DSP cascade)
