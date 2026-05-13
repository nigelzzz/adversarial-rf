# S02: Tile Sequencer FSM + Weight Double-Buffering — Research

**Date:** 2026-05-13
**Depth:** Targeted research — known systolic array patterns applied to established codebase, moderate integration complexity with BRAM feeders and boundary handling.

## Summary

S02 wraps the proven 8x16 single-tile systolic mesh from S01 with a tiling FSM that decomposes arbitrary M×K×N GEMMs into ceil(M/8) × ceil(N/16) tiles, each processed sequentially through the PE grid. The tile sequencer manages B-matrix (weight) double-buffering via two bram_feeder_b banks to hide load latency, reads A-matrix directly from global buffers with tile offsets, handles boundary zero-padding when M or N don't divide evenly, and drains PE accumulators to the correct global c_buf positions with bias addition.

The primary deliverable is `tiled_gemm_s8.v` — a drop-in replacement for `gemm_s8.v` with an identical port interface (same a_buf, b_buf, bias_buf, c_buf layout) so existing testbenches and refmodel.py work unchanged. The module contains its own 8x16 PE grid (same generate blocks as systolic_mesh_s8) with modified address generation that adds tile offsets, plus two bram_feeder_b instances for weight double-buffering.

The verification acceptance criterion is a full (64,320,128) GEMM — conv2, the largest AWN layer — producing byte-identical int32 output to gemm_s8 and numpy. Additional tests must cover all boundary cases from the AWN pipeline: N=66 (last tile has 2 columns), N=1 (single column), M=11 (last tile has 3 rows).

## Recommendation

**Create `tiled_gemm_s8.v` with inline PE grid and direct global buffer addressing for A, double-buffered BRAM for B.** Do NOT wrap systolic_mesh_s8 as a black box — the feeding logic must be modified to add tile offsets, which requires access to the PE grid's input wires. Copying the PE generate block (25 lines) is cleaner than hierarchical cross-module writes.

Build order: (1) basic tiling FSM with direct buffer reads and no double-buffering, verify bit-exact on (64,320,128); (2) add BRAM double-buffering for B; (3) add randomized tests covering all boundary cases. This isolates correctness from double-buffer complexity.

## Implementation Landscape

### Key Files

- `awn_fpga/rtl/systolic_mesh_s8.v` — S01 single-tile engine. Lines 56-100 contain the PE grid generate block and feeding logic to copy. The skewed address formulas (line 80-85 for A, line 92-98 for B) need tile-offset modification. **Do not modify this file** — it remains the single-tile reference.
- `awn_fpga/rtl/pe_s8.v` — PE module, unchanged. Instantiated by the new tiled module's generate block.
- `awn_fpga/rtl/bram_feeder_b.v` — Current stub: single-element write port (`wr_col[3:0]`, `wr_row[AW-1:0]`, `wr_data[7:0]`). **Needs expansion** to support wide writes (all 16 columns per cycle) to match compute throughput. Without wide writes, loading a B-tile takes K×16 cycles vs K+22 compute cycles — loading becomes the bottleneck and double-buffering can't hide it.
- `awn_fpga/rtl/bram_feeder_a.v` — Current stub: single-element write port. Could be used for A-matrix staging, but for S02 direct global a_buf reads are simpler and sufficient. **Defer bram_feeder_a integration** — A data repeats across all N-tiles within an M-tile, so no double-buffering benefit.
- `awn_fpga/rtl/gemm_s8.v` — Behavioral reference. tiled_gemm_s8 must match its output byte-for-byte. Key: same row-major layout for a_buf (`a_buf[m*K+k]`), b_buf (`b_buf[k*N+n]`), c_buf (`c_buf[m*N+n]`), bias_buf (`bias_buf[m]`).
- `awn_fpga/tb/tb_systolic_mesh_s8.v` — S01 testbench pattern to replicate for tiled version. Uses plusargs for dimensions, $readmemh for inputs, $fwrite for outputs.
- `awn_fpga/sw/test_systolic.py` — S01 randomized verification. **Extend** to test tiled module with M>8, N>16 dimensions.
- `awn_fpga/sw/iohex.py` — Hex I/O helpers. Reuse unchanged.

### New Files

- `awn_fpga/rtl/tiled_gemm_s8.v` — Main deliverable. Drop-in replacement for gemm_s8.v.
- `awn_fpga/tb/tb_tiled_gemm_s8.v` — Testbench (same harness pattern as tb_gemm_s8.v).
- `awn_fpga/sw/test_tiled_systolic.py` — Randomized verification script.

### AWN GEMM Shapes and Tiling Analysis

All 10 GEMM calls from refmodel.py with tiling breakdown:

| Layer | M | K | N | M-tiles | N-tiles | Total tiles | Boundary M | Boundary N |
|-------|---|---|---|---------|---------|-------------|------------|------------|
| conv1 | 64 | 14 | 128 | 8 | 8 | 64 | — | — |
| conv2 | 64 | 320 | 128 | 8 | 8 | 64 | — | — |
| U.conv1 | 64 | 192 | 66 | 8 | 5 | 40 | — | last tile N_tile=2 |
| U.conv2 | 64 | 192 | 64 | 8 | 4 | 32 | — | — |
| P.conv1 | 64 | 192 | 66 | 8 | 5 | 40 | — | last tile N_tile=2 |
| P.conv2 | 64 | 192 | 64 | 8 | 4 | 32 | — | — |
| SE lin0 | 32 | 128 | 1 | 4 | 1 | 4 | — | N_tile=1 |
| SE lin3 | 128 | 32 | 1 | 16 | 1 | 16 | — | N_tile=1 |
| fc.0 | 320 | 128 | 1 | 40 | 1 | 40 | — | N_tile=1 |
| fc.2 | 11 | 320 | 1 | 2 | 1 | 2 | last tile M_tile=3 | N_tile=1 |

**Boundary cases requiring zero-padding:**
- N=66: 66 mod 16 = 2 → last N-tile has only 2 valid columns (14 PEs feed zeros)
- N=1: All tiles have only 1 valid column
- M=11: 11 mod 8 = 3 → last M-tile has only 3 valid rows (5 PEs feed zeros)
- All other shapes are cleanly divisible

### Architecture: tiled_gemm_s8.v

```
┌─────────────────────────────────────────────────────────────────┐
│ tiled_gemm_s8 (same port interface as gemm_s8)                 │
│                                                                 │
│  Global buffers: a_buf, b_buf, bias_buf, c_buf                 │
│  (testbench loads via $readmemh, same as gemm_s8)              │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Tile Sequencer FSM                                       │  │
│  │  IDLE → LOAD_B0 → COMPUTE_LOAD → DRAIN → NEXT → DONE   │  │
│  │  Manages: mt, nt, m_base, n_base, M_tile, N_tile        │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│  A-feeding: a_buf[(m_base + row) * K + (cycle - row)]          │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ 8x16 PE Grid (generate block from systolic_mesh_s8)     │  │
│  └──────────────────────────────────────────────────────────┘  │
│  B-feeding: from active BRAM bank                              │
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐                            │
│  │ bram_feeder_b│  │ bram_feeder_b│  ← weight double-buffer   │
│  │   Bank 0     │  │   Bank 1     │                            │
│  └──────────────┘  └──────────────┘                            │
│  Active bank → PE top-edge    Idle bank ← load from b_buf     │
│                                                                 │
│  Output drain: c_buf[(m_base+dm)*N + (n_base+dn)]             │
│  Bias: bias_buf[m_base + dm]                                   │
└─────────────────────────────────────────────────────────────────┘
```

### FSM States

```
IDLE:         Wait for start pulse. Capture M, K, N, use_bias.
              Compute mt_count = ceil(M/8), nt_count = ceil(N/16).
              Set mt=0, nt=0, m_base=0, n_base=0.
              Compute M_tile = min(8, M), N_tile = min(16, N).
              Transition → LOAD_B0.

LOAD_B0:      Load first B-tile into BRAM bank 0.
              Write one K-row per cycle (wide write: all N_tile columns).
              After K cycles → COMPUTE_LOAD, set active_bank=0.

COMPUTE_LOAD: Run PE grid for K + PM + PN - 2 cycles (same as systolic_mesh_s8).
              A reads from global a_buf with m_base offset.
              B reads from active BRAM bank.
              Simultaneously load next B-tile into idle BRAM bank
              (if next tile exists). Loading takes K cycles; compute
              takes K+22 cycles, so load always finishes first.
              After compute → DRAIN.

DRAIN:        Read pe_acc[dm*PN + dn], add bias_buf[m_base + dm] if use_bias.
              Write to c_buf[(m_base + dm) * N + (n_base + dn)].
              Only write valid positions (dm < M_tile, dn < N_tile).
              After M_tile × N_tile cycles → NEXT.

NEXT:         Advance nt. If nt < nt_count:
                n_base += 16, recompute N_tile = min(16, N - n_base).
                Swap active_bank. → COMPUTE_LOAD.
              Else: reset nt=0, n_base=0, advance mt.
                If mt < mt_count:
                  m_base += 8, recompute M_tile = min(8, M - m_base).
                  N_tile = min(16, N). → LOAD_B0 (reload first B-tile for new M-tile).
                Else: → DONE.

DONE:         Assert done=1 for one cycle → IDLE.
```

### Cycle Count Per Tile

```
compute = K + 8 + 16 - 2 = K + 22
drain   = M_tile × N_tile  (≤ 128)
total   = K + 22 + M_tile × N_tile
```

For conv2 (64,320,128) — 64 tiles, all full-size:
- Per tile: 320 + 22 + 128 = 470 cycles
- First tile adds K=320 for initial B load
- Total: 320 + 64 × 470 = 30,400 cycles

This is well within the 100K budget for the full pipeline.

### bram_feeder_b Modification

Current interface writes one element per cycle. Need wide write (all COLS values per cycle) to keep loading time ≤ K cycles:

```verilog
// BEFORE (S01 stub):
input               wr_en,
input  [3:0]        wr_col,
input  [AW-1:0]     wr_row,
input  signed [7:0] wr_data,

// AFTER (S02 expanded):
input               wr_en,
input  [AW-1:0]     wr_k,              // K index (row in B-tile)
input  signed [7:0] wr_data [0:COLS-1], // all 16 columns at once
```

Write logic becomes: `for col in 0..COLS-1: mem[col][wr_k] <= wr_data[col]`

Read port stays the same: single-column, single-row combinational read for PE feeding.

### Address Generation (Modified from systolic_mesh_s8)

**A-feeding (left edge, row m):**
```verilog
// Original (systolic_mesh_s8 line 80-85):
wire [DIM_W-1:0] a_offset = cycle_cnt - gi;
wire [31:0]      a_addr   = gi * K_reg + a_offset;

// Tiled version — add m_base:
wire [DIM_W-1:0] a_offset = cycle_cnt - gi[DIM_W-1:0];
wire [31:0]      a_addr   = (m_base + gi[DIM_W-1:0]) * K_reg + a_offset;
wire             a_valid  = (state == S_COMPUTE_LOAD) &&
                            (cycle_cnt >= gi[DIM_W-1:0]) &&
                            (a_offset < K_reg) &&
                            (gi[DIM_W-1:0] < M_tile);  // boundary check
```

**B-feeding (top edge, column n) — reads from BRAM bank:**
```verilog
// Original (systolic_mesh_s8 line 92-98):
wire [31:0] b_addr = b_offset * N_reg + gj;

// Tiled version — read from active BRAM bank:
wire [DIM_W-1:0] b_offset = cycle_cnt - gj[DIM_W-1:0];
wire             b_valid  = (state == S_COMPUTE_LOAD) &&
                            (cycle_cnt >= gj[DIM_W-1:0]) &&
                            (b_offset < K_reg) &&
                            (gj[DIM_W-1:0] < N_tile);  // boundary check
// Read: bram_b_bank[active_bank].mem[gj][b_offset]
```

**Key difference from systolic_mesh_s8:** A-feeding adds m_base offset to global a_buf address. B-feeding reads from BRAM bank instead of global b_buf. Both add boundary validity checks (gi < M_tile, gj < N_tile) for zero-padding.

### B-Tile Loading Logic

Loading B-tile (nt) into BRAM bank: read from global b_buf, write one K-row per cycle:
```verilog
// For each k in 0..K-1, write all 16 column values:
for (col = 0; col < 16; col = col + 1)
    if (n_base + col < N)
        wr_data[col] = b_buf[k * N + n_base + col];
    else
        wr_data[col] = 8'sd0;  // zero-pad boundary columns
```

### Build Order

1. **Task 1: Basic tiled_gemm_s8 with direct reads (no double-buffer)**
   - Create tiled_gemm_s8.v with inline PE grid
   - A reads from global a_buf, B reads from global b_buf (both with tile offsets)
   - Tiling FSM: IDLE → COMPUTE → DRAIN → NEXT → DONE
   - Verify (8,320,16) single-tile matches systolic_mesh_s8
   - Verify (64,320,128) multi-tile matches gemm_s8 and numpy

2. **Task 2: Add BRAM double-buffering for B-matrix**
   - Expand bram_feeder_b with wide write port
   - Instantiate two banks in tiled_gemm_s8
   - Add LOAD_B0 state; modify COMPUTE to load next tile into idle bank
   - Bank swap on tile advance
   - Re-verify all tests pass

3. **Task 3: Testbench + randomized verification**
   - Create tb_tiled_gemm_s8.v (same harness as tb_gemm_s8.v)
   - Create test_tiled_systolic.py with:
     - All 10 AWN GEMM shapes from refmodel.py
     - Boundary cases: (11,320,1), (64,192,66), (32,128,1)
     - 50+ randomized tests: M∈{1..64}, K∈{1..320}, N∈{1..128}
     - With and without bias
   - Assert bit-exact int32 match against numpy for every test

### Verification Approach

**Primary verification command:**
```bash
cd awn_fpga && python sw/test_tiled_systolic.py
```

**Expected output:** `ALL TILED SYSTOLIC TESTS PASSED (N tests)`

**Mandatory test cases:**
1. Single-tile (should match systolic_mesh_s8): (8,320,16), (4,4,4), (1,1,1)
2. Aligned multi-tile: (64,320,128) — conv2, the S02 acceptance criterion
3. N-boundary: (64,192,66) — last N-tile has 2 columns
4. M-boundary: (11,320,1) — last M-tile has 3 rows
5. Skinny N: (320,128,1) — fc.0 shape, all tiles have N_tile=1
6. Small total: (32,128,1) — SE linear shape
7. 50+ randomized with arbitrary M, K, N

**Cross-check:** For the (64,320,128) case, also run through gemm_s8 testbench and compare outputs byte-for-byte.

## Constraints

- **iverilog 2D array limitation:** S01 discovered that iverilog has issues with variable-index reads on 2D wire/reg arrays. The PE accumulator array was changed to flat 1D (`pe_acc[m*PN+n]`). bram_feeder_b uses 2D arrays (`mem[col][row]`) — since the read port uses genvar-derived column indices (constant per PE), this should be safe. But if variable-index reads are needed, flatten to 1D.
- **No hierarchical writes in synthesizable RTL:** Cannot write into a sub-module's reg arrays from the parent. This is why tiled_gemm_s8 must include the PE grid directly rather than wrapping systolic_mesh_s8.
- **Accumulator must drain before next tile:** PE accumulators are overwritten by acc_clear on the next tile's first cycle. The drain phase must fully complete before the next COMPUTE_LOAD begins. Output double-buffering (drain to a staging buffer while next tile computes) is a future optimization, not needed for S02.
- **bram_feeder_b array port limitation:** Verilog-2005 doesn't support array ports directly. The wide write can use a packed bus (`input [COLS*8-1:0] wr_data_flat`) and unpack inside the module, or use a generate block with individual byte enables. iverilog with `-g2005-sv` supports unpacked array ports in some cases — test during implementation.

## Common Pitfalls

- **c_buf address using N_tile instead of N_global** — The drain must write `c_buf[(m_base+dm) * N_global + (n_base+dn)]`, using the FULL N dimension for row stride. Using N_tile would scatter outputs to wrong positions. The gemm_s8 reference uses `c_buf[m * N_in + n]` — same pattern with global coordinates.
- **Bias indexing error** — Bias is per output row: `bias_buf[m_base + dm]`, NOT `bias_buf[dm]`. Each M-tile uses a different bias slice.
- **Off-by-one in tile count** — `ceil(M/8)` in integer arithmetic is `(M + 7) / 8` (unsigned) or `(M + PM - 1) / PM`. Using `M / PM` truncates and misses the boundary tile.
- **Skew validity for boundary tiles** — When M_tile < 8, rows gi >= M_tile must feed zeros. The a_valid signal must check `gi < M_tile` (not just `gi < PM`). Same for N_tile < 16 on the B side.
- **acc_clear timing** — Must assert acc_clear on the FIRST compute cycle of each tile (not just the first tile). Each tile starts a fresh accumulation. The PE's acc_clear sets acc = current product (not zero), so the first MAC is captured correctly.
- **BRAM bank swap** — After swapping active bank, the COMPUTE_LOAD state must read from the newly active bank. If the swap happens in NEXT and COMPUTE_LOAD starts next cycle, ensure the read address generation uses the updated bank signal combinationally (not registered).

## Open Risks

- **Wide write port in iverilog:** Unpacked array ports (`input signed [7:0] wr_data [0:15]`) may not be supported in iverilog's Verilog-2005 mode. If not, use a flat packed bus (`input [127:0] wr_data_flat`) and slice inside the module. Test early.
- **Simulation speed for randomized tests:** The (64,320,128) GEMM requires ~30K compute cycles. With 50+ randomized tests of similar size, iverilog simulation may take several minutes. Use `--eval_limit` style per-test timeouts or limit large-dimension tests to 10 runs.
- **bram_feeder_b 2D array synthesis:** The `reg [7:0] mem [0:COLS-1][0:DEPTH-1]` pattern infers distributed RAM in Xilinx, not block RAM. For synthesis (next milestone), may need `(* ram_style = "block" *)` attributes or restructured storage. Not a blocker for simulation.

## Sources

- `awn_fpga/systolic_optimization.md` — Tiling math, cycle count formulas, resource estimates. Confirmed 8×16 array at 200 MHz fits within 500 us budget with ~71K total cycles.
- `awn_fpga/sw/refmodel.py` — All 10 GEMM shapes extracted from the AWN forward pass (lines 351-527). Conv2 (64,320,128) is the largest; fc.2 (11,320,1) has the only M-boundary case.
