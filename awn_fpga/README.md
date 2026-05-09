# awn_fpga — AWN model on int8 Verilog hardware, end-to-end

A from-scratch hardware-verification pipeline for the **AWN** (Adaptive
Wavelet Network) RF-modulation classifier. Every arithmetic op of AWN's
forward pass is implemented as a Verilog module, tested in iverilog, and
chained by a Python orchestrator into a full int8 inference that produces the
correct predicted class against the original fp32 PyTorch model.

This repo is *not* a Lab3-style assignment skeleton — Lab3 was the reference
for the GEMM datapath only. Everything below is built from zero.

---

## Three deliverables

1. **Operator profile** — every op AWN executes during inference, with shapes,
   parameter counts, and MACs.
2. **Per-op iverilog tests** — every distinct op has a Verilog module and a
   testbench that compares hardware output to a numpy reference *bit-exactly*.
3. **End-to-end inference** — the full AWN forward pass replayed through the
   real iverilog binaries, producing int8 logits whose argmax matches the
   fp32 model.

---

## TL;DR results

```
Profiler:    34 ops, 5,983,680 MACs, 9 distinct hardware primitives
Op tests:    10 / 10 PASS  (every Verilog module bit-exact vs numpy ref)
Inference:   38 hw ops invoked, all bit-exact at every step
             argmax (int8 hw)  = class 1 (AM-DSB)
             argmax (fp32 ref) = class 1 (AM-DSB)
             argmax-match: TRUE
```

---

## The AWN model

AWN is "Adaptive Wavelet Network" — a CNN whose front end performs a learned
wavelet decomposition (the lifting scheme with learnable Predict/Update
operators). The checkpoint `2016.10a_AWN.pkl` is trained on RadioML 2016.10a
to classify 11 modulations from 128-sample I/Q vectors:

```
8PSK · AM-DSB · AM-SSB · BPSK · CPFSK · GFSK · PAM4 · QAM16 · QAM64 · QPSK · WBFM
```

### Topology

```
input [1, 2, 128]                      # I/Q samples
   │
   ├── unsqueeze → [1, 1, 2, 128]
   │
   ▼
ZeroPad2d(3,3,0,0)                     # pad time dim
Conv2d(1→64, kernel=(2,7))             # squashes I,Q into 64 ch
BatchNorm2d(64)                        # foldable
LeakyReLU(0.01)
squeeze → [1, 64, 128]
   │
   ▼
Conv1d(64→64, k=5, pad=2)
BatchNorm1d(64)                        # foldable
LeakyReLU(0.01)
   │
   ▼  ── lifting scheme (1 level) ───────────────────────────────────
   │     even = x[:, :, 0::2]              # [1,64,64]
   │     odd  = x[:, :, 1::2]              # [1,64,64]
   │     U(odd):  ReflPad(2) → Conv1d k=3 → LeakyReLU → Conv1d k=3 → Tanh
   │     c    = even + U(odd)              # approx coeffs
   │     P(c):   ReflPad(2) → Conv1d k=3 → LeakyReLU → Conv1d k=3 → Tanh
   │     d    = odd  − P(c)                # detail  coeffs
   │
   ▼
AdaptiveAvgPool1d(d) → [1, 64, 1]
AdaptiveAvgPool1d(c) → [1, 64, 1]
Concat dim=1         → [1, 128]
   │
   ▼
SE-Attention:
  Linear(128→32) → ReLU → Linear(32→128) → Sigmoid → multiply with input
   │
   ▼
fc.0:  Linear(128→320) → LeakyReLU
fc.2:  Linear(320→11)                  # logits, no activation
   │
   ▼
output [1, 11] logits → argmax
```

### Why a wavelet front-end

Wavelets give multi-scale time-frequency structure that's well suited to
modulation discrimination. Classical wavelets use *fixed* P/U filters (e.g.
Daubechies). AWN's contribution is learnable P, U built from small CNN blocks
constrained by a final Tanh — the network learns which wavelet best separates
the 11 modulation classes, instead of using one off the shelf.

The lifting block dominates compute: ~3.2 M MACs of the 6 M total live in the
two operator chains (U and P), ~54 % of the whole network.

---

## (1) Operator profile

`sw/profile_awn.py` instruments the AWN forward pass with hooks on every
`nn.Module` and monkey-patches on `__add__`, `__sub__`, `torch.cat`,
`torch.mul` to also catch the functional ops in `forward`. Output:
`build/ops.json` (machine) and `build/ops.txt` (table).

**MAC budget concentrates entirely in convolutions:**

| Rank | Op | Layer | MACs | % |
|---|---|---|---|---|
| 1 | Conv1d 64→64 k=5 | conv2 | 2,621,440 | **43.8 %** |
| 2 | Conv1d 64→64 k=3 | levels.0.U.op.1 | 811,008 | 13.6 % |
| 3 | Conv1d 64→64 k=3 | levels.0.P.op.1 | 811,008 | 13.6 % |
| 4 | Conv1d 64→64 k=3 | levels.0.U.op.4 | 786,432 | 13.1 % |
| 5 | Conv1d 64→64 k=3 | levels.0.P.op.4 | 786,432 | 13.1 % |
| 6 | Conv2d 1→64 k=(2,7) | conv1 | 114,688 | 1.9 % |
| 7–10 | Linear (4 layers) | fc + SE | 52,672 | < 1 % |

**Op-kind histogram (count, not cost):**

```
LeakyReLU x5  ·  Conv1d x5  ·  Linear x4  ·  Dropout x3  ·  ReflectionPad1d x2
Tanh x2  ·  Add x2  ·  AdaptiveAvgPool1d x2
ZeroPad2d, Conv2d, BatchNorm1d, BatchNorm2d, Sub, Concat, ReLU, Sigmoid, Mul  x1 each
```

**After lowering:** every one of the 34 ops collapses into **9 hardware
primitives** (BN folded, Dropout/Pad/Concat/Reshape are sw data movement):

```
gemm_s8  ·  requantize_s32_s8  ·  leaky_relu_s8  ·  relu_s8
lut_s8 (tanh)  ·  lut_s8 (sigmoid)  ·  avgpool1d_s8
eltwise_addsub_s8  ·  mul_s8
```

---

## (2) Per-op iverilog tests

10 hardware modules, each with a numpy reference that uses *exactly* the same
fixed-point arithmetic (`(x*mul + half) >>> shift` with saturation). Run
`python3 sw/run_op_test.py all` → all PASS.

| Op | RTL module | TB | Test config |
|---|---|---|---|
| ReLU | `relu_s8.v` | `tb_relu_s8.v` | 257 elem random int8 |
| LeakyReLU (α=0.01) | `leaky_relu_s8.v` | `tb_leaky_relu_s8.v` | 257 elem, α as Q15 |
| Requantize int32→int8 | `requantize_s32_s8.v` | `tb_requantize_s32_s8.v` | 200 elem random int32 |
| Eltwise Add | `eltwise_addsub_s8.v` (op=0) | shared TB | 200 elem |
| Eltwise Sub | `eltwise_addsub_s8.v` (op=1) | shared TB | 200 elem |
| Mul (scaled) | `mul_s8.v` | `tb_mul_s8.v` | 200 elem, mul+shift |
| AvgPool1d | `avgpool1d_s8.v` | `tb_avgpool1d_s8.v` | 64 ch × 64 samples |
| Tanh LUT | `lut_s8.v` (table=tanh) | `tb_lut_s8.v` | 257 elem |
| Sigmoid LUT | `lut_s8.v` (table=sigmoid) | shared TB | 257 elem |
| GEMM int8 | `gemm_s8.v` | `tb_gemm_s8.v` | M=8, K=12, N=6, +bias |

### Hardware module conventions

- **Buffers internal to the module** (`reg signed [7:0] in_buf [...]`,
  `out_buf`, optionally `lut`, `a_buf`, `b_buf`, `bias_buf`, `c_buf`).
  Loaded by the testbench via `$readmemh("vectors/...hex", DUT.in_buf)`.
- **Handshake:** pulse `start` for one cycle while length/dims are valid,
  wait for `done`. No backpressure, no AXI — minimal protocol.
- **Numerics:** all arithmetic is signed two's complement. Symmetric
  per-tensor quantization (zero-point = 0) so ReLU is just `max(0, x)`.
- **Requantize math:** `out = sat_int8( ((acc * mul) + (1<<(shift-1))) >>> shift + zp )`,
  matching the TFLite Micro fixed-point fast path.

### Per-op test driver

`sw/run_op_test.py` for each op:

1. Generates random tensors that exercise the full int8 / int32 range.
2. Computes the numpy reference using the *same* `(x*mul+half)>>shift` math.
3. Writes input hex files into `vectors/`.
4. Compiles the testbench with `iverilog`.
5. Runs `vvp` with `+plusargs` for length / scales / file paths.
6. Reads back the hardware output hex and asserts it matches the numpy
   reference *element-by-element*.

Anything that fails throws an `AssertionError` with the first divergent
index — no silent passes possible.

---

## (3) End-to-end inference

`sw/refmodel.py` runs the integer AWN forward pass through the real iverilog
modules.

### Quantization scheme (`sw/quantize_awn.py`)

- **Weights:** per-tensor symmetric int8.
  `scale_w = max(|W|) / 127`, `W_q = round(W / scale_w).clip(-127, 127)`.
- **BN folding:** every Conv2d/Conv1d that's followed by a BatchNorm gets
  the BN parameters folded in offline:
  `W' = W · γ/√(σ² + ε)`, `b' = (b − μ) · γ/√(σ² + ε) + β`.
  After folding: only Conv + bias, no separate BN at inference.
- **Activations:** scales calibrated by running one fp32 forward on a fixed
  synthetic IQ input (seed 7 Gaussian) and recording `max(|a|)/127` at each
  intermediate.
- **Bias:** quantized at the int32 accumulator scale `s_bias = s_in · s_w`.
- **Cross-layer rescale:** when two tensors entering an eltwise op have
  different scales (e.g. `c = even + U(odd)`, where `even` and `Tanh(...)`
  have different scales), one of them is rescaled with a `requantize_s32_s8`
  invocation (sign-extended int8 → int32 → requant → int8).
- **Tanh / Sigmoid LUT:** 256 entries, `LUT[i + 128] = round(fn(i · s_in) / s_out)`.
- **Requant ratio encoding:** `M = (s_in · s_w) / s_out` is split with
  `math.frexp` into mantissa × 2^exponent → `mul = round(m · 2^31)`,
  `shift = 31 − exponent`. Hardware shift fits in 6 bits.

### The integer pipeline (`sw/refmodel.py`)

A single `Pipeline` class with one method per hardware op (`gemm`, `relu`,
`leaky_relu`, `requant_s32_s8`, `add_s8`, `sub_s8`, `mul_s8`, `avgpool1d`,
`lut`). Each method:

1. Computes the correct int8/int32 result with numpy.
2. Writes its inputs to `vectors/<step>_<role>.hex`.
3. Invokes the corresponding iverilog binary via `vvp`.
4. Reads back the hardware output.
5. **Asserts bit-exact match** against the numpy reference.

The forward pass itself is just the natural unrolling of AWN's `forward()`,
calling these methods in order. Convolutions are lowered to GEMM via an
`im2col_1d` / `im2col_2d` helper (im2col runs in software since it's pure
data movement, not arithmetic).

### Result

```
input quantized: range -127..88  scale=0.012801
after conv1+leaky: range -1..127
after conv2+leaky: range -1..123

hardware ops invoked: 38
int8 logits:        [-11, 26, -4, -44, -53, -60, -71, -37, -45, -33, -106]
dequantized:        [-207, +490, -75, -829, -998, -1130, -1337, -697, -848, -622, -1997]
fp32 reference:     [-259, +600, -85, -993, -1194, -1375, -1609, -858, -1035, -746, -2392]
argmax (hw):        class 1 (AM-DSB)
argmax (fp32):      class 1 (AM-DSB)
argmax-match: 1
```

**38 hardware op invocations were verified bit-exactly against the numpy
reference at every intermediate step.** The pipeline cannot reach the next
op unless the current one's iverilog output matches numpy. The final int8
logits, when dequantized, track the fp32 reference closely enough to give
the same argmax — the int8 model classifies the input identically to the
fp32 model.

---

## How to run

```bash
cd awn_fpga

# (1) Operator profile
python3 sw/profile_awn.py
#   → build/ops.json, build/ops.txt

# (2) Per-op iverilog tests (10 ops)
python3 sw/run_op_test.py all
#   → 10/10 PASS

# (3) End-to-end inference
python3 sw/quantize_awn.py        # one-time: calibrate + save build/quant.npz
python3 sw/refmodel.py            # run the full pipeline through iverilog
```

Run a single op test:

```bash
python3 sw/run_op_test.py gemm
python3 sw/run_op_test.py tanh sigmoid
```

---

## Project structure

```
awn_fpga/
├── README.md                       # this file
├── 2016.10a_AWN.pkl                # fp32 trained AWN checkpoint (provided)
├── model.py, lifting.py            # AWN architecture (provided)
│
├── rtl/                            # 9 hardware modules
│   ├── gemm_s8.v                   # int8 × int8 → int32 GEMM with optional bias
│   ├── requantize_s32_s8.v         # int32 → int8 with TFLite-style mul+shift
│   ├── relu_s8.v
│   ├── leaky_relu_s8.v             # parameterized α via Q-format
│   ├── eltwise_addsub_s8.v         # op_sel selects + or −
│   ├── mul_s8.v                    # scaled int8 multiply
│   ├── avgpool1d_s8.v              # per-channel mean over time
│   ├── lut_s8.v                    # 256-entry int8→int8 (used for tanh, sigmoid)
│   └── global_buffer.v             # (kept for compatibility with Lab3 style)
│
├── tb/                             # one testbench per RTL op
│   ├── tb_gemm_s8.v
│   ├── tb_requantize_s32_s8.v
│   ├── tb_relu_s8.v
│   ├── tb_leaky_relu_s8.v
│   ├── tb_eltwise_addsub_s8.v
│   ├── tb_mul_s8.v
│   ├── tb_avgpool1d_s8.v
│   └── tb_lut_s8.v
│
├── sw/
│   ├── profile_awn.py              # (1) operator profiler
│   ├── quantize_awn.py             # weight quantization, BN folding, calibration
│   ├── refmodel.py                 # (3) integer forward pass + iverilog orchestrator
│   ├── run_op_test.py              # (2) per-op test driver
│   └── iohex.py                    # shared int8/int32 ↔ hex helpers
│
├── vectors/                        # input/golden/output hex files (regenerated)
└── build/
    ├── ops.json, ops.txt           # operator profile
    ├── quant.npz                   # quantized weights + scales
    └── sim_*                       # compiled iverilog binaries (cached)
```

---

## Hardware op interface — example

```verilog
// rtl/relu_s8.v
module relu_s8 #(parameter LEN = 8192, parameter ADDR_W = 16) (
    input               clk, rst_n, start,
    input [ADDR_W-1:0]  length,
    output reg          done
);
    reg signed [7:0] in_buf  [0:LEN-1];
    reg signed [7:0] out_buf [0:LEN-1];
    // ... FSM that walks `length` elements doing max(0, x) ...
endmodule
```

```verilog
// tb/tb_relu_s8.v  (excerpt)
$readmemh("vectors/relu_in.hex", DUT.in_buf);   // load tensor
@(negedge clk); start = 1; @(negedge clk); start = 0;
while (!done) @(negedge clk);                   // run
for (k = 0; k < length; k = k + 1)              // dump
    $fwrite(fout, "%02x\n", DUT.out_buf[k] & 8'hff);
```

```python
# sw/run_op_test.py  (excerpt)
x = rng.integers(-128, 128, size=257, dtype=np.int8)
io.write_int8_hex("vectors/relu_in.hex", x)
run_sim(compile_tb("relu_s8"), {"len": 257, "in": "vectors/relu_in.hex",
                                  "out": "vectors/relu_out.hex"})
hw  = io.read_int8_hex("vectors/relu_out.hex", count=257)
ref = np.where(x > 0, x, 0).astype(np.int8)
io.assert_equal_int8(hw, ref, "relu_s8")        # bit-exact assertion
```

Same pattern for every op.

---

## Wavelet block in hardware (worth a closer look)

The lifting scheme is what makes AWN distinctive and what required the
non-GEMM hardware ops. One level of lifting maps to:

```
split (sw)
 ├── even ────────────────────────── requantize_s32_s8 ──┐
 │                                   (rescale to s_lift) │
 │                                                       ▼
 └── odd  → ReflectionPad1d (sw) → gemm → requant → leaky_relu
                                 → gemm → requant → tanh-LUT
                                                          │
                                                          ▼
                                                    eltwise_add_s8 → c
                                                          │
                                                          ▼ (P operator chain)
                                       gemm → requant → leaky_relu
                                       gemm → requant → tanh-LUT
                                                          │
                              odd → requantize_s32_s8 ────┤
                              (rescale to s_lift_d)       │
                                                          ▼
                                                    eltwise_sub_s8 → d
```

Without this block the network would be just `conv → conv → fc` and
hardware would need only GEMM + LeakyReLU + requantize. The wavelet block
is why we need cross-scale requantize, both Tanh-LUTs, and eltwise
add/sub primitives.

---

## Why this checkpoint and not another

`2016.10a_AWN.pkl` is the clean baseline. The other AWN variants
(`_at` adversarial-trained, `_ft` fine-tuned) use the *same architecture*,
so the entire pipeline runs unchanged on either one — just swap the path in
`quantize_awn.py` and `profile_awn.py`. The other models in the directory
(`MCLDNN`, `VTCNN2`) are different architectures and would need their own
`model.py` and a rewritten forward in `refmodel.py`.

`AWN_quan_int8_simple.tflite` is an already-quantized version. We could use
its scales directly instead of my single-sample calibration — same hardware,
better accuracy.

---

## Caveats and what's intentionally out of scope

- **Calibration is single-sample**, on a synthetic Gaussian IQ input (seed 7).
  Activation scales are crude; that's why the int8 logits' absolute values
  are ~17 % smaller than fp32 in the dequantized comparison. **Argmax is
  preserved** (which is what classification needs); for tighter logit
  agreement use a real RadioML calibration set.
- **im2col runs in software** in the orchestrator. Convolutions become GEMM
  by reshape + index gather; only the GEMM math runs in iverilog. A
  hardware im2col unit would be straightforward to add.
- **`gemm_s8` is behavioural** (one MAC per cycle, ~2.6 M cycles for the
  conv2 GEMM). Production silicon would need a systolic array or output-
  stationary engine; Lab3's TPU.v shows the structure.
- **No single Verilog top-level** that wraps every op into one
  always-running RTL pipeline. The orchestrator chains ops via files at
  the Python level — agreed scope. A real RTL top is the natural next step.
- **Symmetric per-tensor quantization** (zero-point fixed at 0) — simpler
  hardware (no zp arithmetic, ReLU is `max(0, x)`) at the cost of some
  accuracy versus per-channel asymmetric. TFLite Micro typically uses
  asymmetric per-channel for weights; adding it is mostly a software
  refactor.

---

## Lineage / credits

- Lab3 (NYCU AAML 2025) — reference for the GEMM datapath shape and the
  testbench-driven iverilog flow.
- Sweldens 1996 — the lifting scheme.
- Bastidas et al. (DAWN, WACV 2020) — the LiftingScheme module that AWN
  adapts (`lifting.py` notes this).
- TFLite Micro — the `(x*mul + half) >>> shift` requant fast path used by
  every hardware op here.
