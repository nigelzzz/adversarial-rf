"""Per-op iverilog test driver.

Each op gets a numpy reference, a vector generator, and an iverilog test:
  python3 sw/run_op_test.py relu | leaky_relu | requant | add | sub | mul |
                            avgpool1d | tanh | sigmoid | gemm | all
"""
from __future__ import annotations
import os, sys, subprocess
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


def compile_tb(name):
    src = os.path.join(TB, f"tb_{name}.v")
    out = os.path.join(BUILD, f"sim_{name}")
    cmd = ["iverilog", "-g2005-sv", "-I", RTL, "-I", TB, "-o", out, src]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout); print(r.stderr)
        raise RuntimeError(f"iverilog failed for {name}")
    return out


def run_sim(sim_bin, plusargs):
    cmd = ["vvp", sim_bin] + [f"+{k}={v}" for k, v in plusargs.items()]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    if r.returncode != 0:
        print(r.stdout); print(r.stderr)
        raise RuntimeError(f"vvp failed: {' '.join(cmd)}")
    return r.stdout


# ------------ refs and helpers (must match the Verilog math exactly) -------

def sat_int8(v):
    return int(max(-128, min(127, v)))


def srdhm_round_shift(x, mul, shift):
    """Compute sat_int8( round((x*mul) >> shift) ) the same way the Verilog does:
    half = (1 << (shift-1)) if shift>0 else 0; result = (x*mul + half) >> shift."""
    prod = int(x) * int(mul)
    half = (1 << (shift - 1)) if shift > 0 else 0
    if prod + half >= 0:
        sh = (prod + half) >> shift
    else:
        # arithmetic right shift for negative: round toward -inf (>>>)
        sh = -((-(prod + half) + ((1 << shift) - 1)) >> shift)
        # Above is wrong for arith shift. Use proper signed >> via integer math:
        sh = (prod + half) >> shift  # Python's >> on negatives is arith shift
    return sh


def ref_relu(x):       return np.where(x > 0, x, 0).astype(np.int8)


def ref_leaky_relu(x, alpha_mul, alpha_shift):
    out = np.empty_like(x)
    for i, v in enumerate(x):
        if v >= 0:
            out[i] = v
        else:
            r = srdhm_round_shift(int(v), alpha_mul, alpha_shift)
            out[i] = sat_int8(r)
    return out


def ref_requant(x_int32, mul, shift, out_zp=0, act=0):
    out = np.empty(len(x_int32), dtype=np.int8)
    for i, v in enumerate(x_int32):
        r = srdhm_round_shift(int(v), mul, shift) + out_zp
        s = sat_int8(r)
        if act == 1 and s < out_zp:
            s = out_zp
        out[i] = s
    return out


def ref_addsub(a, b, op):
    s = a.astype(np.int32) + (b.astype(np.int32) if op == 0 else -b.astype(np.int32))
    return np.clip(s, -128, 127).astype(np.int8)


def ref_mul(a, b, mul, shift):
    out = np.empty(len(a), dtype=np.int8)
    for i in range(len(a)):
        prod16 = int(a[i]) * int(b[i])
        r = srdhm_round_shift(prod16, mul, shift)
        out[i] = sat_int8(r)
    return out


def ref_avgpool1d(x, channels, plen, mul, shift):
    out = np.empty(channels, dtype=np.int8)
    for c in range(channels):
        s = int(sum(int(v) for v in x[c*plen:(c+1)*plen]))
        out[c] = sat_int8(srdhm_round_shift(s, mul, shift))
    return out


def ref_lut(x, lut):
    return lut[(x.astype(np.int32) + 128).clip(0, 255)].astype(np.int8)


def ref_gemm(A, B, bias=None):
    """A: (M,K) int8; B: (K,N) int8; bias: (M,) int32 or None."""
    M, K = A.shape; K2, N = B.shape
    assert K == K2
    Ai = A.astype(np.int32); Bi = B.astype(np.int32)
    C = Ai @ Bi
    if bias is not None:
        C = C + bias.reshape(M, 1)
    return C.astype(np.int32)


# ------------ tests --------------------------------------------------------

def test_relu(seed=0):
    rng = np.random.default_rng(seed)
    n = 257
    x = rng.integers(-128, 128, size=n, dtype=np.int8)
    io.write_int8_hex(f"{VEC}/relu_in.hex", x)
    sim = compile_tb("relu_s8")
    print(run_sim(sim, dict(len=n,
                            **{"in": "vectors/relu_in.hex",
                               "out": "vectors/relu_out.hex"})).strip())
    hw = io.read_int8_hex(f"{VEC}/relu_out.hex", count=n)
    io.assert_equal_int8(hw, ref_relu(x), "relu_s8")
    print("PASS  relu_s8")


def test_leaky_relu(seed=0):
    rng = np.random.default_rng(seed)
    n = 257
    x = rng.integers(-128, 128, size=n, dtype=np.int8)
    am, ash = 328, 15  # alpha=0.01
    io.write_int8_hex(f"{VEC}/lrelu_in.hex", x)
    sim = compile_tb("leaky_relu_s8")
    print(run_sim(sim, dict(len=n, amul=am, ashift=ash,
                            **{"in": "vectors/lrelu_in.hex",
                               "out": "vectors/lrelu_out.hex"})).strip())
    hw = io.read_int8_hex(f"{VEC}/lrelu_out.hex", count=n)
    io.assert_equal_int8(hw, ref_leaky_relu(x, am, ash), "leaky_relu_s8")
    print("PASS  leaky_relu_s8")


def test_requant(seed=0):
    rng = np.random.default_rng(seed)
    n = 200
    x = rng.integers(-(1 << 28), (1 << 28), size=n, dtype=np.int32)
    mul, shift = 1234567, 25
    io.write_int32_hex(f"{VEC}/rq_in.hex", x)
    sim = compile_tb("requantize_s32_s8")
    print(run_sim(sim, dict(len=n, mul=mul, shift=shift, zp=0, act=0,
                            **{"in": "vectors/rq_in.hex",
                               "out": "vectors/rq_out.hex"})).strip())
    hw = io.read_int8_hex(f"{VEC}/rq_out.hex", count=n)
    io.assert_equal_int8(hw, ref_requant(x, mul, shift), "requantize_s32_s8")
    print("PASS  requantize_s32_s8")


def test_add(seed=0):    return _test_addsub(seed, op=0)
def test_sub(seed=0):    return _test_addsub(seed, op=1)


def _test_addsub(seed, op):
    rng = np.random.default_rng(seed + op)
    n = 200
    a = rng.integers(-128, 128, size=n, dtype=np.int8)
    b = rng.integers(-128, 128, size=n, dtype=np.int8)
    io.write_int8_hex(f"{VEC}/elt_a.hex", a)
    io.write_int8_hex(f"{VEC}/elt_b.hex", b)
    sim = compile_tb("eltwise_addsub_s8")
    print(run_sim(sim, dict(len=n, op=op,
                            a="vectors/elt_a.hex",
                            b="vectors/elt_b.hex",
                            **{"out": "vectors/elt_out.hex"})).strip())
    hw = io.read_int8_hex(f"{VEC}/elt_out.hex", count=n)
    io.assert_equal_int8(hw, ref_addsub(a, b, op), f"eltwise_{'add' if op==0 else 'sub'}")
    print(f"PASS  eltwise_{'add' if op==0 else 'sub'}_s8")


def test_mul(seed=0):
    rng = np.random.default_rng(seed)
    n = 200
    a = rng.integers(-128, 128, size=n, dtype=np.int8)
    b = rng.integers(-128, 128, size=n, dtype=np.int8)
    mul, shift = 100000000, 22  # ~roughly 0.024 effective scale
    io.write_int8_hex(f"{VEC}/mul_a.hex", a)
    io.write_int8_hex(f"{VEC}/mul_b.hex", b)
    sim = compile_tb("mul_s8")
    print(run_sim(sim, dict(len=n, mul=mul, shift=shift,
                            a="vectors/mul_a.hex",
                            b="vectors/mul_b.hex",
                            **{"out": "vectors/mul_out.hex"})).strip())
    hw = io.read_int8_hex(f"{VEC}/mul_out.hex", count=n)
    io.assert_equal_int8(hw, ref_mul(a, b, mul, shift), "mul_s8")
    print("PASS  mul_s8")


def test_avgpool1d(seed=0):
    rng = np.random.default_rng(seed)
    chans, plen = 64, 64  # AWN's actual avgpool config
    x = rng.integers(-128, 128, size=chans*plen, dtype=np.int8)
    # Pick mul,shift so effective scale ≈ 1/64 (the average): 1/64 ≈ 0.015625.
    # Q31 multiplier for 1/64 = round(2^31 / 64) = 33554432; shift=31.
    mul, shift = 33554432, 31
    io.write_int8_hex(f"{VEC}/ap_in.hex", x)
    sim = compile_tb("avgpool1d_s8")
    print(run_sim(sim, dict(chans=chans, plen=plen, mul=mul, shift=shift,
                            **{"in": "vectors/ap_in.hex",
                               "out": "vectors/ap_out.hex"})).strip())
    hw = io.read_int8_hex(f"{VEC}/ap_out.hex", count=chans)
    io.assert_equal_int8(hw, ref_avgpool1d(x, chans, plen, mul, shift), "avgpool1d_s8")
    print("PASS  avgpool1d_s8")


def _make_lut(fn, in_scale=1.0/64, out_scale=1.0/127):
    # Map int8 i in [-128, 127] -> fn(i*in_scale) -> quantize to int8 with out_scale
    idx = np.arange(-128, 128)
    y = fn(idx.astype(np.float32) * in_scale)
    q = np.round(y / out_scale).clip(-128, 127).astype(np.int8)
    return q  # shape (256,) indexed as i+128


def test_tanh(seed=0):    return _test_lut("tanh", np.tanh, seed)
def test_sigmoid(seed=0): return _test_lut("sigmoid", lambda v: 1.0/(1.0+np.exp(-v)), seed)


def _test_lut(name, fn, seed):
    rng = np.random.default_rng(seed)
    n = 257
    x = rng.integers(-128, 128, size=n, dtype=np.int8)
    lut = _make_lut(fn)
    io.write_int8_hex(f"{VEC}/{name}_lut.hex", lut)
    io.write_int8_hex(f"{VEC}/{name}_in.hex", x)
    sim = compile_tb("lut_s8")
    print(run_sim(sim, dict(len=n,
                            lut=f"vectors/{name}_lut.hex",
                            **{"in":  f"vectors/{name}_in.hex",
                               "out": f"vectors/{name}_out.hex"})).strip())
    hw = io.read_int8_hex(f"{VEC}/{name}_out.hex", count=n)
    io.assert_equal_int8(hw, ref_lut(x, lut), f"{name}_lut_s8")
    print(f"PASS  {name}_lut_s8")


def test_gemm(seed=0):
    rng = np.random.default_rng(seed)
    M, K, N = 8, 12, 6
    A = rng.integers(-32, 32, size=(M, K), dtype=np.int8)
    B = rng.integers(-32, 32, size=(K, N), dtype=np.int8)
    bias = rng.integers(-1000, 1000, size=M, dtype=np.int32)
    io.write_int8_hex(f"{VEC}/gemm_a.hex", A.reshape(-1))
    io.write_int8_hex(f"{VEC}/gemm_b.hex", B.reshape(-1))
    io.write_int32_hex(f"{VEC}/gemm_bias.hex", bias)
    sim = compile_tb("gemm_s8")
    print(run_sim(sim, dict(M=M, K=K, N=N, bias=1,
                            a="vectors/gemm_a.hex",
                            b="vectors/gemm_b.hex",
                            bi="vectors/gemm_bias.hex",
                            **{"out": "vectors/gemm_out.hex"})).strip())
    hw = io.read_int32_hex(f"{VEC}/gemm_out.hex", count=M*N).reshape(M, N)
    gold = ref_gemm(A, B, bias)
    io.assert_equal_int32(hw, gold, "gemm_s8")
    print(f"PASS  gemm_s8 (M={M},K={K},N={N})")


REGISTRY = {
    "relu":        test_relu,
    "leaky_relu":  test_leaky_relu,
    "requant":     test_requant,
    "add":         test_add,
    "sub":         test_sub,
    "mul":         test_mul,
    "avgpool1d":   test_avgpool1d,
    "tanh":        test_tanh,
    "sigmoid":     test_sigmoid,
    "gemm":        test_gemm,
}


def main():
    args = sys.argv[1:] or ["all"]
    targets = list(REGISTRY) if args == ["all"] else args
    fails = []
    for t in targets:
        if t not in REGISTRY:
            print(f"unknown op: {t}; have: {list(REGISTRY)}")
            sys.exit(1)
        print(f"\n==> {t}")
        try:
            REGISTRY[t]()
        except AssertionError as e:
            print(f"FAIL  {t}: {e}")
            fails.append(t)
    print("\n" + "=" * 60)
    if fails:
        print(f"FAILED: {fails}")
        sys.exit(1)
    print(f"All {len(targets)} ops PASS")


if __name__ == "__main__":
    main()
