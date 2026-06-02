"""
Phase 3 — emit C headers from awn_fpga/build/quant.npz so the HLS code can
csim against bit-exact int8 golden references.

Outputs (written to this directory):

  awn_dims.h       — array dimension #defines
  awn_weights.h    — all int8 weight tensors
  awn_biases.h     — all int32 bias tensors
  awn_qparams.h    — (mul_q31, right_shift) per requantize op + tanh/sigmoid LUTs
  golden_io.h      — x_fp (fp32 input), s_input, fp_logits, expected_argmax

The (mul, shift) pairs reproduce sw/refmodel.py:q_multiplier exactly, so HLS
csim and the existing iverilog flow share identical fixed-point arithmetic.
"""
import math
import os
import sys
import textwrap

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
NPZ = os.path.normpath(os.path.join(HERE, "..", "..", "build", "quant.npz"))


# ----- mirror of sw/refmodel.py:q_multiplier (kept verbatim) -----------------
def q_multiplier(M):
    if M <= 0:
        return 0, 31
    m, e = math.frexp(M)
    mul_q = round(m * (1 << 31))
    if mul_q == (1 << 31):
        mul_q //= 2
        e += 1
    shift = 31 - e
    if shift < 0:
        raise ValueError(f"M={M} too large for Q31")
    if shift > 60:
        shift = 60
        mul_q = max(1, round(M * (1 << shift)))
    return int(mul_q), int(shift)


# ----- C array emitters ------------------------------------------------------
def c_array_int(name, arr, ctype, per_line=16):
    """Emit `static const ctype name[d0][d1]... = { ... };` for any shape."""
    dims = "".join(f"[{d}]" for d in arr.shape)
    flat = arr.ravel().tolist()
    lines = []
    for i in range(0, len(flat), per_line):
        chunk = ", ".join(f"{int(v):>5}" for v in flat[i:i + per_line])
        lines.append("    " + chunk + ",")
    body = "\n".join(lines).rstrip(",")
    return f"static const {ctype} {name}{dims} = {{\n{body}\n}};\n"


def c_array_float(name, arr, per_line=8):
    dims = "".join(f"[{d}]" for d in arr.shape)
    flat = arr.ravel().tolist()
    lines = []
    for i in range(0, len(flat), per_line):
        chunk = ", ".join(f"{float(v): .8e}f" for v in flat[i:i + per_line])
        lines.append("    " + chunk + ",")
    body = "\n".join(lines).rstrip(",")
    return f"static const float {name}{dims} = {{\n{body}\n}};\n"


# ----- LUT builders (mirror refmodel.make_lut) -------------------------------
def make_lut(fn, in_scale, out_scale):
    """256-entry int8 -> int8 LUT.  Index i maps int8 i-128 (or similar)."""
    xs = (np.arange(-128, 128).astype(np.float64)) * in_scale
    ys = fn(xs) / out_scale
    return np.clip(np.round(ys), -128, 127).astype(np.int8)


# ----- main ------------------------------------------------------------------
def main():
    Q = np.load(NPZ)
    out_h = {}

    # ---------- awn_dims.h ----------
    dims = textwrap.dedent("""\
    #ifndef AWN_DIMS_H
    #define AWN_DIMS_H

    // RML2016.10a: 2 IQ channels, 128 samples, 11 modulation classes.
    #define AWN_IN_CH       2
    #define AWN_IN_LEN      128
    #define AWN_NUM_CLASSES 11

    // conv1: in=1 (IQ stacked as 2x7 kernel), out=64
    #define CONV1_OUT_CH    64
    #define CONV1_KH        2
    #define CONV1_KW        7

    // conv2 (1D): 64 -> 64, kernel 5
    #define CONV2_OUT_CH    64
    #define CONV2_K         5

    // lifting Updator / Predictor: 64 -> 64, kernel 3
    #define LIFT_CH         64
    #define LIFT_K          3

    // SE attention
    #define SE_IN           128
    #define SE_HID          32

    // FC
    #define FC0_IN          128
    #define FC0_OUT         320
    #define FC2_OUT         AWN_NUM_CLASSES

    // LUT size (full int8 domain)
    #define LUT_SIZE        256

    #endif
    """)
    out_h["awn_dims.h"] = dims

    # ---------- awn_weights.h ----------
    weight_keys_int8 = [
        ("W1", "int8_t"),       # (64,1,2,7)
        ("W2", "int8_t"),       # (64,64,5)
        ("Wu1", "int8_t"),      # (64,64,3)
        ("Wu4", "int8_t"),      # (64,64,3)
        ("Wp1", "int8_t"),      # (64,64,3)
        ("Wp4", "int8_t"),      # (64,64,3)
        ("Wse0", "int8_t"),     # (32,128)
        ("Wse3", "int8_t"),     # (128,32)
        ("Wfc0", "int8_t"),     # (320,128)
        ("Wfc2", "int8_t"),     # (11,320)
    ]
    parts = ["#ifndef AWN_WEIGHTS_H\n#define AWN_WEIGHTS_H\n",
             "#include <stdint.h>\n#include \"awn_dims.h\"\n"]
    for k, ctype in weight_keys_int8:
        parts.append(c_array_int(k, np.asarray(Q[k]), ctype))
    parts.append("#endif\n")
    out_h["awn_weights.h"] = "\n".join(parts)

    # ---------- awn_biases.h ----------
    bias_keys = ["b1", "b2", "bu1", "bu4", "bp1", "bp4", "bfc0", "bfc2"]
    parts = ["#ifndef AWN_BIASES_H\n#define AWN_BIASES_H\n",
             "#include <stdint.h>\n#include \"awn_dims.h\"\n"]
    for k in bias_keys:
        parts.append(c_array_int(k, np.asarray(Q[k]), "int32_t"))
    parts.append("#endif\n")
    out_h["awn_biases.h"] = "\n".join(parts)

    # ---------- awn_qparams.h ----------
    # Reproduce the M = (s_in * s_w) / s_out chain from sw/refmodel.py
    f = lambda k: float(Q[k])

    qm_pairs = []
    def add(name, M):
        mul, sh = q_multiplier(M)
        qm_pairs.append((name, mul, sh, M))

    add("CONV1",   f("s_input")     * f("s_W1")   / f("s_conv1_out"))
    add("CONV2",   f("s_conv1_act") * f("s_W2")   / f("s_conv2_out"))
    add("U1",      f("s_conv2_act") * f("s_Wu1")  / f("s_u_conv1_out"))
    add("U4",      f("s_u_conv1_act") * f("s_Wu4")/ f("s_u_conv2_out"))
    add("EVEN",    f("s_conv2_act") / f("s_lift_c"))
    add("P1",      f("s_lift_c")    * f("s_Wp1")  / f("s_p_conv1_out"))
    add("P4",      f("s_p_conv1_act") * f("s_Wp4")/ f("s_p_conv2_out"))
    add("ODD",     f("s_conv2_act") / f("s_lift_d"))
    add("AVG_D",   f("s_lift_d") / (f("s_avg_d") * 64.0))
    add("AVG_C",   f("s_lift_c") / (f("s_avg_c") * 64.0))
    add("D_TO_CONCAT", f("s_avg_d") / f("s_concat"))
    add("C_TO_CONCAT", f("s_avg_c") / f("s_concat"))
    add("SE0",     f("s_concat")   * f("s_Wse0") / f("s_se0_out"))
    add("SE3",     f("s_se0_act")  * f("s_Wse3") / f("s_se3_out"))
    add("SE_MUL",  f("s_se3_act")  * f("s_concat") / f("s_se_mul"))
    add("FC0",     f("s_se_mul")   * f("s_Wfc0") / f("s_fc0_out"))
    add("FC2",     f("s_fc0_act")  * f("s_Wfc2") / f("s_fc2_out"))

    # Tanh LUTs: input at s_u_conv2_out / s_p_conv2_out, output at s_lift_*.
    tanh_u_lut = make_lut(np.tanh, f("s_u_conv2_out"), f("s_lift_c"))
    tanh_p_lut = make_lut(np.tanh, f("s_p_conv2_out"), f("s_lift_d"))

    # Sigmoid LUT for SE branch.
    sig_lut = make_lut(lambda v: 1.0 / (1.0 + np.exp(-v)),
                       f("s_se3_out"), f("s_se3_act"))

    parts = [
        "#ifndef AWN_QPARAMS_H\n#define AWN_QPARAMS_H\n",
        "#include <stdint.h>\n#include \"awn_dims.h\"\n",
        "// (mul_q31, right_shift) pairs reproduce sw/refmodel.py exactly.\n",
        "// Output int32 = (acc_i32 * MUL + half) >> SHIFT, then saturate to int8.\n\n",
    ]
    for name, mul, sh, M in qm_pairs:
        parts.append(f"// M = {M:.6e}\n")
        parts.append(f"#define {name}_MUL    {mul}\n")
        parts.append(f"#define {name}_SHIFT  {sh}\n\n")

    # Emit LUTs.  Index i in 0..255 corresponds to int8 input value (i - 128).
    parts.append(c_array_int("TANH_U_LUT", tanh_u_lut, "int8_t"))
    parts.append(c_array_int("TANH_P_LUT", tanh_p_lut, "int8_t"))
    parts.append(c_array_int("SIGMOID_LUT", sig_lut, "int8_t"))
    parts.append("\n#endif\n")
    out_h["awn_qparams.h"] = "".join(parts)

    # ---------- golden_io.h ----------
    x_fp = np.asarray(Q["x_fp"]).astype(np.float32)        # (1,2,128)
    fp_logits = np.asarray(Q["fp_logits"]).astype(np.float32)
    s_in = float(Q["s_input"])

    # Pre-quantize the input so HLS can skip the float->int8 step if desired.
    x_q = np.clip(np.round(x_fp / s_in), -128, 127).astype(np.int8)

    expected_arg = int(np.argmax(fp_logits))

    parts = [
        "#ifndef GOLDEN_IO_H\n#define GOLDEN_IO_H\n",
        "#include <stdint.h>\n#include \"awn_dims.h\"\n",
        f"\n// Scale for input quantization (q = round(x_fp / S_INPUT))\n",
        f"#define S_INPUT   {s_in:.17e}f\n",
        f"// Scale to dequantize final logits: logits_fp = logits_q * S_FC2_OUT\n",
        f"#define S_FC2_OUT {float(Q['s_fc2_out']):.17e}f\n\n",
    ]
    parts.append(c_array_float("X_FP", x_fp[0]))            # (2,128)
    parts.append(c_array_int("X_Q",  x_q[0],  "int8_t"))    # (2,128)
    parts.append(c_array_float("FP_LOGITS", fp_logits))     # (11,)
    parts.append(f"\n#define EXPECTED_ARGMAX  {expected_arg}\n")
    parts.append("\n#endif\n")
    out_h["golden_io.h"] = "".join(parts)

    # ---- write ----
    for fname, content in out_h.items():
        path = os.path.join(HERE, fname)
        with open(path, "w") as fh:
            fh.write(content)
        print(f"wrote {fname:20s} ({len(content):>7d} bytes)")

    # Summary of qparams
    print("\n-- requantize multipliers --")
    for name, mul, sh, M in qm_pairs:
        print(f"  {name:14s}  M={M:.4e}  mul={mul:>11d}  shift={sh}")
    print(f"\nexpected argmax = {expected_arg} (class index)")


if __name__ == "__main__":
    main()
