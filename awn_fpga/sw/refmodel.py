"""Integer-only AWN forward pass + iverilog orchestrator.

This is the source of truth that proves: int8 hardware outputs == int8 software
reference outputs == correct AWN logits.

For every arithmetic op, this script:
  1. Computes the correct int8/int32 result in numpy.
  2. Writes the inputs to vectors/<step>_<role>.hex.
  3. Invokes the corresponding iverilog binary via vvp.
  4. Reads the hardware output.
  5. Asserts bit-exact match against the numpy result.

Run:  python3 sw/refmodel.py
"""
from __future__ import annotations
import os, sys, math, subprocess
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import iohex as io  # noqa: E402

VEC = os.path.join(ROOT, "vectors")
BUILD = os.path.join(ROOT, "build")
RTL = os.path.join(ROOT, "rtl")
TB = os.path.join(ROOT, "tb")
os.makedirs(VEC, exist_ok=True)
os.makedirs(BUILD, exist_ok=True)


# ---- compile and cache simulation binaries ---------------------------------

_SIM_CACHE: dict[str, str] = {}

def sim(name: str) -> str:
    if name in _SIM_CACHE:
        return _SIM_CACHE[name]
    src = os.path.join(TB, f"tb_{name}.v")
    out = os.path.join(BUILD, f"sim_{name}")
    if (not os.path.exists(out) or
        os.path.getmtime(out) < os.path.getmtime(src) or
        os.path.getmtime(out) < os.path.getmtime(os.path.join(RTL, f"{name}.v"))):
        cmd = ["iverilog", "-g2005-sv", "-I", RTL, "-I", TB, "-o", out, src]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout); print(r.stderr)
            raise RuntimeError(f"iverilog failed for {name}")
    _SIM_CACHE[name] = out
    return out


def vvp(bin_path: str, plusargs: dict, quiet=True):
    cmd = ["vvp", bin_path] + [f"+{k}={v}" for k, v in plusargs.items()]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    if r.returncode != 0:
        print(r.stdout); print(r.stderr)
        raise RuntimeError(f"vvp failed: {' '.join(cmd)}")
    if not quiet:
        print(r.stdout.strip())


# ---- TFLite-style quantized multiplier -------------------------------------

def q_multiplier(M: float):
    """Return (mul_q31, right_shift) such that v*M ≈ ((v*mul + half) >>> shift).
    For 0 < M < 2^16 this fits in shift in [0, 63] and mul in [2^30, 2^31)."""
    if M <= 0:
        return 0, 31
    m, e = math.frexp(M)         # M = m * 2^e, m in [0.5, 1)
    mul_q = round(m * (1 << 31))
    if mul_q == (1 << 31):
        mul_q //= 2
        e += 1
    shift = 31 - e
    if shift < 0:
        # M >= 2 — we'd need a left shift; AWN shouldn't hit this.
        raise ValueError(f"M={M} too large to fit Q31 mul/shift")
    if shift > 60:
        # avoid 64-bit overflow safety margin
        shift = 60
        mul_q = max(1, round(M * (1 << shift)))
    return int(mul_q), int(shift)


def srdhm(x, mul, shift):
    """Scalar reference of the hardware's (x*mul + half) >>> shift (signed)."""
    half = (1 << (shift - 1)) if shift > 0 else 0
    return (int(x) * int(mul) + half) >> shift  # python's >> is arith for signed


def sat8(v):
    return int(max(-128, min(127, v)))


# ---- per-op hardware-replay primitives -------------------------------------

class Pipeline:
    """Records every hardware op invocation. Each op:
       - writes its inputs to vectors/<step>_<role>.hex
       - invokes iverilog
       - reads the hardware output
       - asserts bit-exact match against numpy reference
    """
    def __init__(self):
        self.step = 0
        self.log = []

    def _fname(self, role: str, ext: str = "hex"):
        self.step  # ensure set externally before
        return os.path.join(VEC, f"{self._tag}_{role}.{ext}")

    def _begin(self, kind: str):
        self.step += 1
        self._tag = f"{self.step:02d}_{kind}"
        self.log.append({"step": self.step, "kind": kind})

    # ----- simple int8 elementwise -----
    def relu(self, x: np.ndarray) -> np.ndarray:
        self._begin("relu")
        x = x.astype(np.int8).reshape(-1)
        ref = np.where(x > 0, x, 0).astype(np.int8)
        in_p = self._fname("in"); out_p = self._fname("out")
        io.write_int8_hex(in_p, x)
        vvp(sim("relu_s8"), {"len": len(x),
                              "in": os.path.relpath(in_p, ROOT),
                              "out": os.path.relpath(out_p, ROOT)})
        hw = io.read_int8_hex(out_p, count=len(x))
        io.assert_equal_int8(hw, ref, f"step{self.step} relu")
        return hw

    def leaky_relu(self, x: np.ndarray, alpha_mul=328, alpha_shift=15) -> np.ndarray:
        self._begin("leaky_relu")
        x = x.astype(np.int8).reshape(-1)
        # ref: same Verilog math
        ref = np.empty_like(x)
        for i, v in enumerate(x):
            if v >= 0:
                ref[i] = v
            else:
                ref[i] = sat8(srdhm(int(v), alpha_mul, alpha_shift))
        in_p = self._fname("in"); out_p = self._fname("out")
        io.write_int8_hex(in_p, x)
        vvp(sim("leaky_relu_s8"), {"len": len(x), "amul": alpha_mul,
                                    "ashift": alpha_shift,
                                    "in": os.path.relpath(in_p, ROOT),
                                    "out": os.path.relpath(out_p, ROOT)})
        hw = io.read_int8_hex(out_p, count=len(x))
        io.assert_equal_int8(hw, ref, f"step{self.step} leaky_relu")
        return hw

    def requant_s32_s8(self, acc: np.ndarray, mul: int, shift: int,
                       out_zp: int = 0, act: int = 0) -> np.ndarray:
        self._begin("requant")
        acc = acc.astype(np.int32).reshape(-1)
        ref = np.empty(len(acc), dtype=np.int8)
        for i, v in enumerate(acc):
            r = srdhm(int(v), mul, shift) + out_zp
            s = sat8(r)
            if act == 1 and s < out_zp:
                s = out_zp
            ref[i] = s
        in_p = self._fname("in"); out_p = self._fname("out")
        io.write_int32_hex(in_p, acc)
        vvp(sim("requantize_s32_s8"), {"len": len(acc), "mul": mul, "shift": shift,
                                        "zp": out_zp, "act": act,
                                        "in": os.path.relpath(in_p, ROOT),
                                        "out": os.path.relpath(out_p, ROOT)})
        hw = io.read_int8_hex(out_p, count=len(acc))
        io.assert_equal_int8(hw, ref, f"step{self.step} requant")
        return hw

    def add_s8(self, a, b):
        self._begin("add")
        a = a.astype(np.int8).reshape(-1); b = b.astype(np.int8).reshape(-1)
        ref = np.clip(a.astype(np.int32) + b.astype(np.int32), -128, 127).astype(np.int8)
        a_p = self._fname("a"); b_p = self._fname("b"); out_p = self._fname("out")
        io.write_int8_hex(a_p, a); io.write_int8_hex(b_p, b)
        vvp(sim("eltwise_addsub_s8"), {"len": len(a), "op": 0,
                                         "a": os.path.relpath(a_p, ROOT),
                                         "b": os.path.relpath(b_p, ROOT),
                                         "out": os.path.relpath(out_p, ROOT)})
        hw = io.read_int8_hex(out_p, count=len(a))
        io.assert_equal_int8(hw, ref, f"step{self.step} add")
        return hw

    def sub_s8(self, a, b):
        self._begin("sub")
        a = a.astype(np.int8).reshape(-1); b = b.astype(np.int8).reshape(-1)
        ref = np.clip(a.astype(np.int32) - b.astype(np.int32), -128, 127).astype(np.int8)
        a_p = self._fname("a"); b_p = self._fname("b"); out_p = self._fname("out")
        io.write_int8_hex(a_p, a); io.write_int8_hex(b_p, b)
        vvp(sim("eltwise_addsub_s8"), {"len": len(a), "op": 1,
                                         "a": os.path.relpath(a_p, ROOT),
                                         "b": os.path.relpath(b_p, ROOT),
                                         "out": os.path.relpath(out_p, ROOT)})
        hw = io.read_int8_hex(out_p, count=len(a))
        io.assert_equal_int8(hw, ref, f"step{self.step} sub")
        return hw

    def mul_s8(self, a, b, mul, shift):
        self._begin("mul")
        a = a.astype(np.int8).reshape(-1); b = b.astype(np.int8).reshape(-1)
        ref = np.empty(len(a), dtype=np.int8)
        for i in range(len(a)):
            ref[i] = sat8(srdhm(int(a[i]) * int(b[i]), mul, shift))
        a_p = self._fname("a"); b_p = self._fname("b"); out_p = self._fname("out")
        io.write_int8_hex(a_p, a); io.write_int8_hex(b_p, b)
        vvp(sim("mul_s8"), {"len": len(a), "mul": mul, "shift": shift,
                              "a": os.path.relpath(a_p, ROOT),
                              "b": os.path.relpath(b_p, ROOT),
                              "out": os.path.relpath(out_p, ROOT)})
        hw = io.read_int8_hex(out_p, count=len(a))
        io.assert_equal_int8(hw, ref, f"step{self.step} mul")
        return hw

    def avgpool1d(self, x, channels, plen, mul, shift):
        self._begin("avgpool")
        x = x.astype(np.int8).reshape(-1)
        ref = np.empty(channels, dtype=np.int8)
        for c in range(channels):
            s = int(sum(int(v) for v in x[c*plen:(c+1)*plen]))
            ref[c] = sat8(srdhm(s, mul, shift))
        in_p = self._fname("in"); out_p = self._fname("out")
        io.write_int8_hex(in_p, x)
        vvp(sim("avgpool1d_s8"), {"chans": channels, "plen": plen,
                                    "mul": mul, "shift": shift,
                                    "in": os.path.relpath(in_p, ROOT),
                                    "out": os.path.relpath(out_p, ROOT)})
        hw = io.read_int8_hex(out_p, count=channels)
        io.assert_equal_int8(hw, ref, f"step{self.step} avgpool")
        return hw

    def lut(self, x, lut_table, name="lut"):
        self._begin(name)
        x = x.astype(np.int8).reshape(-1)
        ref = lut_table[(x.astype(np.int32) + 128).clip(0, 255)].astype(np.int8)
        in_p = self._fname("in"); out_p = self._fname("out")
        lut_p = self._fname("table")
        io.write_int8_hex(in_p, x)
        io.write_int8_hex(lut_p, lut_table)
        vvp(sim("lut_s8"), {"len": len(x),
                              "lut": os.path.relpath(lut_p, ROOT),
                              "in":  os.path.relpath(in_p, ROOT),
                              "out": os.path.relpath(out_p, ROOT)})
        hw = io.read_int8_hex(out_p, count=len(x))
        io.assert_equal_int8(hw, ref, f"step{self.step} {name}")
        return hw

    def gemm(self, A, B, bias=None):
        """A: (M,K) int8; B: (K,N) int8; bias: (M,) int32 or None.
        Returns C int32 (M,N)."""
        self._begin("gemm")
        A = A.astype(np.int8); B = B.astype(np.int8)
        M, K = A.shape; K2, N = B.shape
        assert K == K2, (A.shape, B.shape)
        ref = (A.astype(np.int32) @ B.astype(np.int32))
        if bias is not None:
            ref = ref + bias.astype(np.int32).reshape(M, 1)
        a_p = self._fname("a"); b_p = self._fname("b")
        bi_p = self._fname("bias"); out_p = self._fname("out")
        io.write_int8_hex(a_p, A.reshape(-1))
        io.write_int8_hex(b_p, B.reshape(-1))
        if bias is not None:
            io.write_int32_hex(bi_p, bias.astype(np.int32))
        else:
            io.write_int32_hex(bi_p, np.zeros(M, dtype=np.int32))
        vvp(sim("gemm_s8"), {"M": M, "K": K, "N": N,
                              "bias": 1 if bias is not None else 0,
                              "a": os.path.relpath(a_p, ROOT),
                              "b": os.path.relpath(b_p, ROOT),
                              "bi": os.path.relpath(bi_p, ROOT),
                              "out": os.path.relpath(out_p, ROOT)})
        hw = io.read_int32_hex(out_p, count=M*N).reshape(M, N)
        io.assert_equal_int32(hw, ref, f"step{self.step} gemm")
        return hw


# ---- helpers: convert convs to GEMM via im2col -----------------------------

def im2col_2d(x, kh, kw, padding):
    """x: (N, Cin, H, W). pad spec is torch-style (left, right, top, bottom).
    Returns (Cin*kh*kw, N*Hout*Wout)."""
    N, Cin, H, W = x.shape
    pl, pr, pt, pb = padding
    xp = np.pad(x, ((0,0),(0,0),(pt,pb),(pl,pr)), mode='constant')
    H2 = H + pt + pb; W2 = W + pl + pr
    Hout = H2 - kh + 1
    Wout = W2 - kw + 1
    out = np.zeros((Cin*kh*kw, N*Hout*Wout), dtype=x.dtype)
    col = 0
    for n in range(N):
        for h in range(Hout):
            for w in range(Wout):
                patch = xp[n, :, h:h+kh, w:w+kw]   # (Cin, kh, kw)
                out[:, col] = patch.reshape(-1)
                col += 1
    return out, (N, Hout, Wout)


def im2col_1d(x, k, padding_mode="constant", pad=0, ref_pad=None):
    """x: (N, Cin, T). Returns (Cin*k, N*Tout)."""
    if ref_pad is not None:
        xp = np.pad(x, ((0,0),(0,0),(ref_pad, ref_pad)), mode='reflect')
    else:
        xp = np.pad(x, ((0,0),(0,0),(pad, pad)), mode=padding_mode)
    N, Cin, T = xp.shape
    Tout = T - k + 1
    out = np.zeros((Cin*k, N*Tout), dtype=x.dtype)
    col = 0
    for n in range(N):
        for t in range(Tout):
            out[:, col] = xp[n, :, t:t+k].reshape(-1)
            col += 1
    return out, (N, Tout)


def make_lut(fn, in_scale, out_scale):
    idx = np.arange(-128, 128)
    y = fn(idx.astype(np.float64) * in_scale)
    q = np.round(y / out_scale).clip(-128, 127).astype(np.int8)
    return q


def quantize(x, scale):
    return np.round(x / scale).clip(-128, 127).astype(np.int8)


# ---- the integer forward pass ----------------------------------------------

def main():
    Q = np.load(os.path.join(BUILD, "quant.npz"))
    pipe = Pipeline()

    # Input: fp32 -> int8 with s_input
    x_fp = Q["x_fp"]
    s_in = float(Q["s_input"])
    x_q = quantize(x_fp, s_in)                      # (1,2,128) int8
    print(f"input quantized: range {x_q.min()}..{x_q.max()}  scale={s_in:.6f}")

    # === conv1: ZeroPad((3,3,0,0)) + Conv2d(1->64, kernel (2,7)) + LeakyReLU ===
    x4 = x_q[:, np.newaxis, :, :]                   # (1,1,2,128)
    x4_pad = np.pad(x4, ((0,0),(0,0),(0,0),(3,3)), mode='constant')   # (1,1,2,134)
    # Important: padding adds zeros (in int8 quantized space, zero == 0 since zp=0).

    W1 = Q["W1"].astype(np.int8)                    # (64,1,2,7)
    b1 = Q["b1"].astype(np.int32)                   # (64,)
    A_w = W1.reshape(64, 1*2*7)                     # (64, 14)
    B_x, (Nb, Hout, Wout) = im2col_2d(x4_pad, 2, 7, padding=(0,0,0,0))
    # B_x shape: (14, 1*1*128) = (14, 128); Hout=1, Wout=128
    acc = pipe.gemm(A_w, B_x, b1)                   # (64, 128) int32

    # Requantize to s_conv1_out
    s_W1 = float(Q["s_W1"])
    s_o1 = float(Q["s_conv1_out"])
    M1 = (s_in * s_W1) / s_o1
    mul1, shift1 = q_multiplier(M1)
    y1_q = pipe.requant_s32_s8(acc.reshape(-1), mul1, shift1)
    y1_q = y1_q.reshape(64, 128)
    # LeakyReLU (alpha=0.01); in int8 same scale
    y1_q = pipe.leaky_relu(y1_q.reshape(-1)).reshape(64, 128)
    print(f"after conv1+leaky: range {y1_q.min()}..{y1_q.max()}")

    # === conv2: Conv1d(64->64, k=5, pad=2) + LeakyReLU ===
    y1_q3 = y1_q[np.newaxis, :, :]                  # (1,64,128)
    W2 = Q["W2"].astype(np.int8)                    # (64,64,5)
    b2 = Q["b2"].astype(np.int32)
    A2 = W2.reshape(64, 64*5)
    B2, _ = im2col_1d(y1_q3, k=5, padding_mode="constant", pad=2)
    acc2 = pipe.gemm(A2, B2, b2)                    # (64, 128)
    s_W2 = float(Q["s_W2"])
    s_o2 = float(Q["s_conv2_out"])
    M2 = (float(Q["s_conv1_act"]) * s_W2) / s_o2
    mul2, shift2 = q_multiplier(M2)
    y2_q = pipe.requant_s32_s8(acc2.reshape(-1), mul2, shift2)
    y2_q = pipe.leaky_relu(y2_q).reshape(64, 128)
    print(f"after conv2+leaky: range {y2_q.min()}..{y2_q.max()}")

    # === Splitting: even (0,2,4,...) and odd (1,3,5,...) ===
    even_q = y2_q[:, ::2]                           # (64, 64)
    odd_q  = y2_q[:, 1::2]                          # (64, 64)

    # === U(odd): ReflectionPad1d(2) + Conv1d k=3 + LeakyReLU + Conv1d k=3 + Tanh ===
    odd_q3 = odd_q[np.newaxis, :, :]                # (1,64,64)
    Wu1 = Q["Wu1"].astype(np.int8); bu1 = Q["bu1"].astype(np.int32)
    Au1 = Wu1.reshape(64, 64*3)
    Bu1, _ = im2col_1d(odd_q3, k=3, ref_pad=2)
    acc_u1 = pipe.gemm(Au1, Bu1, bu1)
    s_Wu1 = float(Q["s_Wu1"]); s_uo1 = float(Q["s_u_conv1_out"])
    M_u1 = (float(Q["s_conv2_act"]) * s_Wu1) / s_uo1
    mul_u1, shift_u1 = q_multiplier(M_u1)
    yu1 = pipe.requant_s32_s8(acc_u1.reshape(-1), mul_u1, shift_u1)
    yu1 = pipe.leaky_relu(yu1).reshape(64, 66)

    yu1_3 = yu1[np.newaxis, :, :]                   # (1,64,66) — no further pad
    Wu4 = Q["Wu4"].astype(np.int8); bu4 = Q["bu4"].astype(np.int32)
    Au4 = Wu4.reshape(64, 64*3)
    Bu4, _ = im2col_1d(yu1_3, k=3, padding_mode="constant", pad=0)
    acc_u4 = pipe.gemm(Au4, Bu4, bu4)               # (64, 64)
    s_Wu4 = float(Q["s_Wu4"]); s_uo2 = float(Q["s_u_conv2_out"])
    M_u4 = (float(Q["s_u_conv1_act"]) * s_Wu4) / s_uo2
    mul_u4, shift_u4 = q_multiplier(M_u4)
    yu4 = pipe.requant_s32_s8(acc_u4.reshape(-1), mul_u4, shift_u4)
    yu4 = yu4.reshape(64, 64)

    # Tanh LUT — input scale s_uo2, output scale we choose s_lift (= s_lift_c)
    # so the subsequent add (even + Tanh(...)) works at one scale.
    s_lift = float(Q["s_lift_c"])  # scale of the post-add tensor
    # We need both `even` and the Tanh output at scale s_lift.
    # Simplest: rescale `even` and `yu4 (Tanh applied at this scale)` to s_lift.

    # Tanh first: input at s_uo2, output at s_lift.
    tanh_lut = make_lut(np.tanh, s_uo2, s_lift)
    yu_tanh = pipe.lut(yu4.reshape(-1), tanh_lut, name="tanh").reshape(64, 64)

    # Rescale `even` from s_conv2_act to s_lift via requantize_s32_s8 fed with
    # sign-extended int8.
    M_even = float(Q["s_conv2_act"]) / s_lift
    mul_e, shift_e = q_multiplier(M_even)
    even_rs = pipe.requant_s32_s8(even_q.reshape(-1).astype(np.int32),
                                   mul_e, shift_e)
    even_rs = even_rs.reshape(64, 64)

    # c = even + U(odd)
    c_q = pipe.add_s8(even_rs.reshape(-1), yu_tanh.reshape(-1)).reshape(64, 64)

    # === P(c): same op pattern as U ===
    c_q3 = c_q[np.newaxis, :, :]
    Wp1 = Q["Wp1"].astype(np.int8); bp1 = Q["bp1"].astype(np.int32)
    Ap1 = Wp1.reshape(64, 64*3)
    Bp1, _ = im2col_1d(c_q3, k=3, ref_pad=2)
    acc_p1 = pipe.gemm(Ap1, Bp1, bp1)
    s_Wp1 = float(Q["s_Wp1"]); s_po1 = float(Q["s_p_conv1_out"])
    M_p1 = (s_lift * s_Wp1) / s_po1
    mul_p1, shift_p1 = q_multiplier(M_p1)
    yp1 = pipe.requant_s32_s8(acc_p1.reshape(-1), mul_p1, shift_p1)
    yp1 = pipe.leaky_relu(yp1).reshape(64, 66)

    yp1_3 = yp1[np.newaxis, :, :]
    Wp4 = Q["Wp4"].astype(np.int8); bp4 = Q["bp4"].astype(np.int32)
    Ap4 = Wp4.reshape(64, 64*3)
    Bp4, _ = im2col_1d(yp1_3, k=3, padding_mode="constant", pad=0)
    acc_p4 = pipe.gemm(Ap4, Bp4, bp4)
    s_Wp4 = float(Q["s_Wp4"]); s_po2 = float(Q["s_p_conv2_out"])
    M_p4 = (float(Q["s_p_conv1_act"]) * s_Wp4) / s_po2
    mul_p4, shift_p4 = q_multiplier(M_p4)
    yp4 = pipe.requant_s32_s8(acc_p4.reshape(-1), mul_p4, shift_p4)
    yp4 = yp4.reshape(64, 64)

    # Tanh -> rescale to s_lift_d (= s of d tensor)
    s_lift_d = float(Q["s_lift_d"])
    tanh_lut_p = make_lut(np.tanh, s_po2, s_lift_d)
    yp_tanh = pipe.lut(yp4.reshape(-1), tanh_lut_p, name="tanh").reshape(64, 64)

    # Rescale odd_q from s_conv2_act to s_lift_d
    M_odd = float(Q["s_conv2_act"]) / s_lift_d
    mul_o, shift_o = q_multiplier(M_odd)
    odd_rs = pipe.requant_s32_s8(odd_q.reshape(-1).astype(np.int32),
                                  mul_o, shift_o)
    odd_rs = odd_rs.reshape(64, 64)

    d_q = pipe.sub_s8(odd_rs.reshape(-1), yp_tanh.reshape(-1)).reshape(64, 64)

    # === AvgPool over time (length 64) for d and c ===
    s_avg_d = float(Q["s_avg_d"])
    M_avg_d = s_lift_d / (s_avg_d * 64.0)
    mul_ad, shift_ad = q_multiplier(M_avg_d)
    avg_d_q = pipe.avgpool1d(d_q.reshape(-1), 64, 64, mul_ad, shift_ad)

    s_avg_c = float(Q["s_avg_c"])
    M_avg_c = s_lift / (s_avg_c * 64.0)
    mul_ac, shift_ac = q_multiplier(M_avg_c)
    avg_c_q = pipe.avgpool1d(c_q.reshape(-1), 64, 64, mul_ac, shift_ac)

    # Concat dim 1: cat = [avg_d (64), avg_c (64)] -> (128,). Need same scale.
    # Pick s_concat = max(s_avg_d, s_avg_c). Rescale each.
    s_concat = float(Q["s_concat"])
    M_d_to_c = s_avg_d / s_concat
    mul_dc, shift_dc = q_multiplier(M_d_to_c)
    avg_d_rs = pipe.requant_s32_s8(avg_d_q.astype(np.int32), mul_dc, shift_dc)

    M_c_to_c = s_avg_c / s_concat
    mul_cc, shift_cc = q_multiplier(M_c_to_c)
    avg_c_rs = pipe.requant_s32_s8(avg_c_q.astype(np.int32), mul_cc, shift_cc)

    cat_q = np.concatenate([avg_d_rs, avg_c_rs])    # (128,) — pure data move

    # === SE attention: Linear(128, 32) + ReLU + Linear(32, 128) + Sigmoid ===
    Wse0 = Q["Wse0"].astype(np.int8)                 # (32, 128)
    # GEMM: A=(32,128)(weight), B=(128,1)(input as column vector) -> (32,1)
    se0_acc = pipe.gemm(Wse0, cat_q.reshape(128, 1))
    s_Wse0 = float(Q["s_Wse0"]); s_se0_out = float(Q["s_se0_out"])
    M_se0 = (s_concat * s_Wse0) / s_se0_out
    mul_se0, shift_se0 = q_multiplier(M_se0)
    se0_q = pipe.requant_s32_s8(se0_acc.reshape(-1), mul_se0, shift_se0).reshape(32)
    se0_q = pipe.relu(se0_q)

    Wse3 = Q["Wse3"].astype(np.int8)                 # (128, 32)
    se3_acc = pipe.gemm(Wse3, se0_q.reshape(32, 1))
    s_Wse3 = float(Q["s_Wse3"]); s_se3_out = float(Q["s_se3_out"])
    M_se3 = (float(Q["s_se0_act"]) * s_Wse3) / s_se3_out
    mul_se3, shift_se3 = q_multiplier(M_se3)
    se3_q = pipe.requant_s32_s8(se3_acc.reshape(-1), mul_se3, shift_se3)

    # Sigmoid LUT: input s_se3_out, output s_se3_act (~1/127)
    s_se3_act = float(Q["s_se3_act"])
    sig_lut = make_lut(lambda v: 1.0/(1.0+np.exp(-v)), s_se3_out, s_se3_act)
    sig_q = pipe.lut(se3_q, sig_lut, name="sigmoid")

    # se_mul = sig_q * cat_q (element-wise). Output scale s_se_mul.
    s_se_mul = float(Q["s_se_mul"])
    M_mul = (s_se3_act * s_concat) / s_se_mul
    mul_m, shift_m = q_multiplier(M_mul)
    se_mul_q = pipe.mul_s8(sig_q, cat_q, mul_m, shift_m)

    # === fc.0: Linear(128, 320) + LeakyReLU ===
    Wfc0 = Q["Wfc0"].astype(np.int8); bfc0 = Q["bfc0"].astype(np.int32)
    fc0_acc = pipe.gemm(Wfc0, se_mul_q.reshape(128, 1), bfc0)
    s_Wfc0 = float(Q["s_Wfc0"]); s_fc0_out = float(Q["s_fc0_out"])
    M_fc0 = (s_se_mul * s_Wfc0) / s_fc0_out
    mul_fc0, shift_fc0 = q_multiplier(M_fc0)
    fc0_q = pipe.requant_s32_s8(fc0_acc.reshape(-1), mul_fc0, shift_fc0).reshape(320)
    fc0_q = pipe.leaky_relu(fc0_q)

    # === fc.2: Linear(320, 11) — produces logits (no activation) ===
    Wfc2 = Q["Wfc2"].astype(np.int8); bfc2 = Q["bfc2"].astype(np.int32)
    fc2_acc = pipe.gemm(Wfc2, fc0_q.reshape(320, 1), bfc2)
    s_Wfc2 = float(Q["s_Wfc2"]); s_fc2_out = float(Q["s_fc2_out"])
    M_fc2 = (float(Q["s_fc0_act"]) * s_Wfc2) / s_fc2_out
    mul_fc2, shift_fc2 = q_multiplier(M_fc2)
    logits_q = pipe.requant_s32_s8(fc2_acc.reshape(-1), mul_fc2, shift_fc2)

    # Dequantize for human readability
    logits_fp = logits_q.astype(np.float32) * s_fc2_out
    print("\n" + "=" * 70)
    print(f"hardware ops invoked: {pipe.step}")
    print(f"int8 logits:         {list(logits_q.astype(int))}")
    print(f"dequantized logits:  {[f'{v:+.3f}' for v in logits_fp]}")
    fp_logits = Q["fp_logits"]
    print(f"fp32 reference:      {[f'{v:+.3f}' for v in fp_logits]}")
    print(f"argmax (hw):         class {int(np.argmax(logits_q))}")
    print(f"argmax (fp32):       class {int(np.argmax(fp_logits))}")
    err = float(np.max(np.abs(logits_fp - fp_logits)))
    print(f"max-abs deviation in dequantized logits: {err:.3f}")
    print(f"argmax-match: {int(np.argmax(logits_q) == np.argmax(fp_logits))}")


if __name__ == "__main__":
    main()
