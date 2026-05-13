# AWN FPGA Optimization: Systolic Array TPU for WiFi-Latency Alignment

**Date:** 2026-05-13
**Context:** Last 3 commits (95b9248, 4463987, 3ce58d7) established the AWN int8 FPGA pipeline with per-op Verilog modules, latency analysis, and BRAM estimation. This document analyzes how to replace the behavioral `gemm_s8` (1 MAC/cycle) with a systolic array to meet WiFi real-time requirements.

---

## 1. WiFi Latency Requirements

### 1.1 Target Budgets

| WiFi Standard | Frame Duration | Classification Budget (10%) | Notes |
|---|---|---|---|
| 802.11a/g (OFDM) | 4 us symbol | **0.4 us** | Per-symbol decision |
| 802.11n (HT) | 3.6 us symbol | **0.36 us** | Short GI |
| 802.11ac/ax | 3.2-3.6 us symbol | **0.32-0.36 us** | Beamforming adds budget |
| **Practical burst-level** | **~1 ms per burst** | **100-500 us** | Classify after preamble |
| **Spectrum sensing** | 10-100 ms dwell | **1-10 ms** | Most relaxed |

**Realistic target for AMC:** The AWN classifier operates on 128-sample IQ bursts (not per-symbol). The practical target is **burst-level classification within 100-500 us** to fit in the preamble-to-payload gap of a WiFi frame exchange.

### 1.2 Current Latency Gap

| Configuration | Latency | WiFi Budget Gap |
|---|---|---|
| Current RTL (P=1, 100 MHz) | 61.5 ms | **123x over 500 us target** |
| Optimistic clock (P=1, 300 MHz) | 20.5 ms | 41x over |
| Latency notes P=16, 100 MHz | 3.8 ms | 7.6x over |
| Latency notes P=64, 100 MHz | 0.96 ms | 1.9x over |
| **Required** | **< 500 us** | — |

The 1 MAC/cycle behavioral GEMM must be replaced with a high-throughput engine. A systolic array is the natural fit.

---

## 2. AWN Compute Profile (Bottleneck Analysis)

### 2.1 MAC Distribution

From `build/ops.json`, total 5,983,680 MACs across 7 GEMM invocations (conv1/2 lowered via im2col, 4 lifting convs, 4 FC layers):

| Layer | GEMM Shape (M,K,N) | MACs | % Total | Notes |
|---|---|---|---|---|
| conv2 | (64, 320, 128) | 2,621,440 | **43.8%** | Dominant bottleneck |
| U.op.1 | (64, 192, 66) | 811,008 | 13.6% | Lifting Updator conv 1 |
| P.op.1 | (64, 192, 66) | 811,008 | 13.6% | Lifting Predictor conv 1 |
| U.op.4 | (64, 192, 64) | 786,432 | 13.1% | Lifting Updator conv 2 |
| P.op.4 | (64, 192, 64) | 786,432 | 13.1% | Lifting Predictor conv 2 |
| conv1 | (64, 14, 128) | 114,688 | 1.9% | Small, quick |
| FC layers (x4) | various small | 52,672 | <1% | Negligible |

**Key insight:** conv2 alone is 43.8% of compute. The 4 lifting convolutions are 53.4%. Together with conv2, 97.2% of MACs are in 5 GEMM calls with K=192-320 and M,N=64-128.

### 2.2 GEMM Shape Characteristics

All major GEMMs share a pattern: **M=64, K=192-320, N=64-128**. This means:
- **K is the contraction dimension** (inner product length): 192-320
- **M is output channels**: always 64
- **N is spatial/temporal**: 64-128

A systolic array sized for M=64 and with good K-streaming would cover every major GEMM without reconfiguration.

---

## 3. Systolic Array Design for AWN

### 3.1 Architecture: Output-Stationary 2D Systolic Array

The TPU-style output-stationary systolic array computes `C[m,n] = sum_k A[m,k] * B[k,n]`:

```
              B[:,0]  B[:,1]  B[:,2] ... B[:,Pn-1]    (weights streamed down)
                |       |       |          |
                v       v       v          v
A[0,:] --> [ PE ] --> [ PE ] --> [ PE ] ... [ PE ]    --> C[0,0..Pn-1]
A[1,:] --> [ PE ] --> [ PE ] --> [ PE ] ... [ PE ]    --> C[1,0..Pn-1]
  :          :         :         :          :
A[Pm-1,:]--> [ PE ] --> [ PE ] --> [ PE ] ... [ PE ]    --> C[Pm-1,0..Pn-1]
```

Each Processing Element (PE):
```
PE state: accumulator acc (int32)
Each cycle:
  acc += a_in * b_in        // int8 x int8 -> int32 accumulate
  a_out = a_in              // pass A east
  b_out = b_in              // pass B south
```

### 3.2 Systolic Array Sizing

For AWN, the optimal array size balances DSP usage vs. latency:

| Array Size (Pm x Pn) | PEs | DSP48 | conv2 Cycles | Total Inference Cycles | Latency @200MHz |
|---|---|---|---|---|---|
| 8 x 8 | 64 | 64 | ~40,960 | ~95,000 | **475 us** |
| 8 x 16 | 128 | 128 | ~20,480 | ~50,000 | **250 us** |
| 16 x 8 | 128 | 128 | ~20,480 | ~50,000 | **250 us** |
| 16 x 16 | 256 | 256 | ~10,240 | ~28,000 | **140 us** |
| 32 x 8 | 256 | 256 | ~10,240 | ~28,000 | **140 us** |

**Recommended: 8x16 or 16x8 array (128 PEs) at 200 MHz -> ~250 us latency.**

This fits within the 500 us WiFi burst budget with margin for DMA and orchestrator overhead.

### 3.3 Tiling Strategy

Since AWN's GEMMs have M=64 > Pm and N=64-128 > Pn, we tile:

```
For GEMM(M=64, K=320, N=128) with array Pm=8, Pn=16:
  M-tiles = ceil(64/8)  = 8
  N-tiles = ceil(128/16) = 8
  Total tiles = 8 * 8 = 64
  Cycles per tile = K + Pm + Pn - 2 = 320 + 8 + 16 - 2 = 342
  (pipeline fill + K MACs + drain)
  Total cycles = 64 * 342 = 21,888
```

Detailed per-layer breakdown with 8x16 systolic array:

| Layer | M | K | N | M-tiles | N-tiles | Cycles/tile | Total Cycles |
|---|---|---|---|---|---|---|---|
| conv1 | 64 | 14 | 128 | 8 | 8 | 36 | 2,304 |
| conv2 | 64 | 320 | 128 | 8 | 8 | 342 | **21,888** |
| U.op.1 | 64 | 192 | 66 | 8 | 5 | 214 | 8,560 |
| U.op.4 | 64 | 192 | 64 | 8 | 4 | 214 | 6,848 |
| P.op.1 | 64 | 192 | 66 | 8 | 5 | 214 | 8,560 |
| P.op.4 | 64 | 192 | 64 | 8 | 4 | 214 | 6,848 |
| SE Linear x2 | 32/128 | 128/32 | 1 | 4/16 | 1 | 150/54 | 1,464 |
| fc.0 | 320 | 128 | 1 | 40 | 1 | 150 | 6,000 |
| fc.2 | 11 | 320 | 1 | 2 | 1 | 342 | 684 |
| **GEMM subtotal** | | | | | | | **63,156** |
| Non-GEMM ops | | | | | | | ~8,000 |
| **Total** | | | | | | | **~71,000** |

**At 200 MHz: 71,000 / 200e6 = 355 us** -- within the 500 us WiFi budget.

### 3.4 Cycle Formula

```
T_inference = sum_layers( ceil(M/Pm) * ceil(N/Pn) * (K + Pm + Pn - 2) ) / f_clk
            + non_GEMM_cycles / f_clk
            + overhead

For 8x16 array @ 200 MHz:
  = 63,156 / 200e6 + 8,000 / 200e6 + ~10 us
  = 316 us + 40 us + 10 us
  = ~366 us
```

---

## 4. Systolic Array RTL Design

### 4.1 PE Module

```verilog
module pe_s8 (
    input               clk, rst_n,
    input  signed [7:0] a_in,       // activation from west
    input  signed [7:0] b_in,       // weight from north
    input               acc_clear,  // reset accumulator for new tile
    output reg signed [7:0]  a_out, // pass east
    output reg signed [7:0]  b_out, // pass south
    output reg signed [31:0] acc    // accumulated partial sum
);
    wire signed [15:0] prod = a_in * b_in;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            a_out <= 0; b_out <= 0; acc <= 0;
        end else begin
            a_out <= a_in;
            b_out <= b_in;
            if (acc_clear)
                acc <= {{16{prod[15]}}, prod};
            else
                acc <= acc + {{16{prod[15]}}, prod};
        end
    end
endmodule
```

### 4.2 Systolic Array Top Module

```verilog
module systolic_array_s8 #(
    parameter PM = 8,    // rows (M-dimension)
    parameter PN = 16,   // cols (N-dimension)
    parameter K_MAX = 512
)(
    input               clk, rst_n, start,
    input  [15:0]       M_in, K_in, N_in,
    // Weight loading port (double-buffered)
    input               w_load_en,
    input  [15:0]       w_addr,
    input  signed [7:0] w_data,
    // Activation streaming port
    input               a_valid,
    input  signed [7:0] a_data [0:PM-1],
    // Output
    output reg          done,
    output reg          out_valid,
    output reg signed [31:0] out_data [0:PM-1]  // one column at a time
);
    // PM x PN PE grid
    wire signed [7:0]  a_wire [0:PM-1][0:PN];    // horizontal wires
    wire signed [7:0]  b_wire [0:PM][0:PN-1];    // vertical wires
    wire signed [31:0] acc_out [0:PM-1][0:PN-1];
    reg                acc_clear;

    // Weight buffer (double-buffered for overlap)
    reg signed [7:0] w_buf [0:K_MAX*PN-1];

    genvar gi, gj;
    generate
        for (gi = 0; gi < PM; gi = gi + 1) begin : row
            for (gj = 0; gj < PN; gj = gj + 1) begin : col
                pe_s8 pe_inst (
                    .clk(clk), .rst_n(rst_n),
                    .a_in(a_wire[gi][gj]),
                    .b_in(b_wire[gi][gj]),
                    .acc_clear(acc_clear),
                    .a_out(a_wire[gi][gj+1]),
                    .b_out(b_wire[gi+1][gj]),
                    .acc(acc_out[gi][gj])
                );
            end
        end
    endgenerate
    // ... FSM for tile iteration, data feeding, and output draining ...
endmodule
```

### 4.3 Data Flow for One Tile

For a tile computing `C_tile[Pm, Pn] = A_tile[Pm, K] @ B_tile[K, Pn]`:

```
Cycle 0:    Feed A[0,0] to row 0, B[0,0] to col 0
Cycle 1:    Feed A[0,1] to row 0, A[1,0] to row 1, B[1,0] to col 0, B[0,1] to col 1
...
Cycle k:    A row m gets A[m, k-m], B col n gets B[k-n, n] (with skew)
...
Cycle K+Pm+Pn-3: Last PE finishes accumulating
Cycle K+Pm+Pn-2: Drain output row by row
```

**Skewed feeding** is critical: row m's data is delayed by m cycles, col n's data is delayed by n cycles. This ensures each PE receives the correct (a,b) pair at the correct time.

---

## 5. Resource Estimation

### 5.1 DSP48 Usage

| Component | DSP48 Count | Notes |
|---|---|---|
| 8x16 systolic array | 128 | 1 DSP per PE (int8 x int8 multiply) |
| Requantize unit | 1 | Shared, pipelined |
| **Total** | **129** | |

### 5.2 BRAM Usage

| Buffer | Size | 36Kb BRAMs |
|---|---|---|
| Weight double-buffer (2 x K_MAX x PN) | 2 x 512 x 16 = 16 KB | 4 |
| Activation buffer (ping-pong) | 2 x 64 x 320 = 40 KB | 10 |
| Output buffer | 64 x 128 x 4 = 32 KB | 8 |
| Weight ROM (all AWN weights) | 124 KB | 28 |
| im2col buffer | 40 KB | 10 |
| Non-GEMM op buffers | 16 KB | 4 |
| **Total** | **~268 KB** | **~64** |

### 5.3 FPGA Fit

| FPGA | DSP48 Available | BRAM 36Kb | Fit? |
|---|---|---|---|
| Pynq-Z2 (Zynq 7020) | 220 | 140 | Yes (59% DSP, 46% BRAM) |
| Arty Z7-20 (Zynq 7020) | 220 | 140 | Yes |
| ZCU104 (ZU7EV) | 1,728 | 312 | Easy (7% DSP) |
| Ultra96-V2 (ZU3EG) | 360 | 216 | Yes (36% DSP, 30% BRAM) |

**The 8x16 systolic array fits comfortably on a Pynq-Z2** (the cheapest Zynq board), using 59% of DSPs and 46% of BRAMs.

---

## 6. Comparison: Systolic vs. Current vs. Alternatives

### 6.1 Latency Comparison

| Design | Throughput | Latency @200MHz | DSP | BRAM | Power Est. |
|---|---|---|---|---|---|
| Current (1 MAC/cycle) | 1 MAC/clk | 30.8 ms | 1 | ~74 | ~0.3 W |
| Parallel GEMM (P=64) | 64 MAC/clk | 480 us | 64 | ~74 | ~0.8 W |
| **Systolic 8x16** | **128 MAC/clk** | **~355 us** | **128** | **~64** | **~1.2 W** |
| Systolic 16x16 | 256 MAC/clk | ~185 us | 256 | ~72 | ~2.0 W |

### 6.2 Why Systolic > Simple Parallelism

| Property | Simple P=128 | Systolic 8x16 |
|---|---|---|
| Data reuse | None (128 reads/cycle from BRAM) | A reused across Pn, B reused across Pm |
| BRAM ports needed | 128 read ports (impossible) | 8+16 = 24 ports (feasible) |
| Wiring complexity | High (128 multipliers fan-in) | Regular 2D grid (local connections) |
| Routing on FPGA | Very difficult | Clean, place-and-route friendly |
| Power efficiency | High toggle rate | Low: each wire toggles once per cycle |

The simple parallelism approach from `latency_notes.md` (P=64, P=128) is **physically unrealizable** because it requires too many simultaneous BRAM reads. A systolic array achieves the same throughput with only local connections and O(Pm+Pn) memory bandwidth.

---

## 7. Optimization Roadmap

### Phase 1: Systolic Core (Week 1-2)

1. **Implement `pe_s8.v`** -- single PE with registered I/O and int32 accumulator
2. **Implement `systolic_8x16_s8.v`** -- 8x16 PE grid with generate blocks
3. **Testbench** -- verify against numpy GEMM reference using existing vector infrastructure
4. **Validate** -- run conv2 (M=64, K=320, N=128) through systolic, compare to behavioral `gemm_s8`

### Phase 2: Tiling Controller (Week 2-3)

1. **Tile sequencer FSM** -- iterate M-tiles x N-tiles, manage skewed feeding
2. **Weight double-buffer** -- load next tile's weights while current tile computes
3. **im2col integration** -- can remain in software (PS side) or add a simple HW im2col
4. **End-to-end test** -- full AWN inference through systolic, match fp32 argmax

### Phase 3: Pipeline and Integration (Week 3-4)

1. **Non-GEMM op pipelining** -- chain requantize/activation ops after systolic drain
2. **AXI-Lite control** -- PS orchestrates tile sequence, reads output logits
3. **DMA integration** -- stream IQ input from PS DDR to PL, results back
4. **Latency measurement** -- hardware cycle counter, compare to estimates

### Phase 4: WiFi Integration Demo (Week 4)

1. **Full pipeline benchmark** -- measure actual end-to-end latency
2. **Throughput test** -- continuous burst classification rate
3. **Power measurement** -- verify within board thermal limits

---

## 8. Alternative: im2col in Hardware

Currently im2col runs in software. For the tightest latency:

**Software im2col overhead:**
- conv2 im2col: reshape 64x128 -> 320x128 = 40 KB data movement
- PS overhead: ~50-100 us per GEMM call for im2col + DMA setup
- Total PS overhead: ~350-700 us (7 GEMM calls)

This overhead **nearly doubles** the 355 us compute time. Two options:

1. **HW im2col unit**: Small address generator that reads conv input from BRAM and feeds the systolic array in im2col order. Adds ~200 LUTs, eliminates PS overhead entirely. **Recommended.**

2. **Direct conv in systolic**: Modify the systolic array input staging to handle convolution directly (sliding window over time). More complex but avoids im2col storage.

With HW im2col, total latency becomes pure compute: **~355 us at 200 MHz**.

---

## 9. Sensitivity Analysis

### 9.1 Clock Frequency Impact

| f_clk | 8x16 Systolic Latency | Meets 500 us? | Meets 100 us? |
|---|---|---|---|
| 100 MHz | 710 us | No | No |
| 150 MHz | 473 us | Yes | No |
| 200 MHz | 355 us | Yes | No |
| 250 MHz | 284 us | Yes | No |
| 300 MHz | 237 us | Yes | No |

**200 MHz is achievable on Zynq-7020** (Pynq-Z2) and meets the 500 us target.

### 9.2 Array Size Impact

| Array | PEs | Latency @200MHz | DSP% (Z7020) | Meets 500 us? |
|---|---|---|---|---|
| 4x8 | 32 | 1,420 us | 15% | No |
| 8x8 | 64 | 710 us | 29% | No |
| **8x16** | **128** | **355 us** | **58%** | **Yes** |
| 16x16 | 256 | 185 us | 116% | No (doesn't fit Z7020) |
| 8x16 on ZU3EG | 128 | 355 us | 36% | Yes, lots of headroom |

The 8x16 array is the sweet spot for Zynq-7020. For sub-100 us, move to UltraScale+.

---

## 10. Summary

| Metric | Current | Systolic 8x16 @200MHz | Improvement |
|---|---|---|---|
| Inference latency | 30.8 ms | **~355 us** | **87x** |
| WiFi burst budget | 61x over | **Within 500 us** | Meets requirement |
| MAC throughput | 100M MAC/s | **25.6G MAC/s** | 256x |
| DSP utilization | 1 | 128 (58% of Z7020) | Efficient |
| BRAM utilization | 74 blocks | 64 blocks | Better (weight ROM shared) |
| Power estimate | 0.3 W | ~1.2 W | Acceptable for edge |

**Bottom line:** Replacing the behavioral `gemm_s8` with an 8x16 output-stationary systolic array running at 200 MHz brings AWN inference from 30.8 ms down to ~355 us, well within the 500 us WiFi burst-level classification budget. The design fits on a Pynq-Z2 (Zynq-7020) using 58% of DSPs and 46% of BRAMs.
