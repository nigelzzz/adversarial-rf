"""Pull a real AWN linear layer's weights, uint8-quantize them, and emit a
Lab3-TPU `input.txt` that drives the iverilog testbench end-to-end.

AWN layer chosen: SE_attention_score.0  (Linear 128 -> 32, no bias).
TPU contract: C[m,n] = sum_k A[k,m] * B[k,n], with K,M,N <= 255.

Mapping:
  K = in_features  = 128  (contraction dim)
  M = out_features = 32
  N = batch        = 4    (one C-tile worth)
  A[k,m] = W_q[m,k]   (transposed quantized weights)
  B[k,n] = X_q[n,k]   (transposed quantized activations)
  golden C[m,n] = y_q[n,m] = sum_k A[k,m]*B[k,n]   (uint32 unsigned)
"""
import sys, types, os
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))

# Shim `from models.lifting import LiftingScheme` so model.py loads
sys.path.insert(0, HERE)
pkg = types.ModuleType("models"); pkg.__path__ = [HERE]
sys.modules["models"] = pkg
import lifting as _lift
sys.modules["models.lifting"] = _lift


def quant_uint8_symmetric(x: np.ndarray):
    """Map x's [min, max] range linearly to uint8 [0, 255]. Returns (xq, scale, zp)."""
    lo, hi = float(x.min()), float(x.max())
    if hi == lo:
        return np.zeros_like(x, dtype=np.uint8), 1.0, 0
    scale = (hi - lo) / 255.0
    zp = -lo / scale
    xq = np.clip(np.round(x / scale + zp), 0, 255).astype(np.uint8)
    return xq, scale, zp


def write_uint8_matrix(fd, m: np.ndarray):
    """Match data_generator.write_matrix layout: 4 col-bytes per file line,
    walking col-stripes outer, rows inner."""
    r, c = m.shape
    calign = ((c + 3) // 4) * 4
    M = np.zeros((r, calign), dtype=np.uint8)
    M[:, :c] = m
    fd.write("\n")
    for cptr in range(0, calign, 4):
        for dr in range(r):
            fd.write("{0:2x} {1:2x} {2:2x} {3:2x}\n".format(
                M[dr, cptr+0], M[dr, cptr+1], M[dr, cptr+2], M[dr, cptr+3]))
    fd.write("\n")


def write_uint32_matrix(fd, m: np.ndarray):
    r, c = m.shape
    calign = ((c + 3) // 4) * 4
    M = np.zeros((r, calign), dtype=np.uint32)
    M[:, :c] = m
    fd.write("\n")
    for cptr in range(0, calign, 4):
        for dr in range(r):
            fd.write("{0:8x} {1:8x} {2:8x} {3:8x}\n".format(
                int(M[dr, cptr+0]), int(M[dr, cptr+1]),
                int(M[dr, cptr+2]), int(M[dr, cptr+3])))
    fd.write("\n")


def main(out_path):
    sd = torch.load(os.path.join(HERE, "2016.10a_AWN.pkl"),
                    map_location="cpu", weights_only=False)

    # Picked layer: SE_attention_score.0.weight, shape (32, 128) = (out, in)
    W = sd["SE_attention_score.0.weight"].numpy()
    out_f, in_f = W.shape
    assert (out_f, in_f) == (32, 128), f"unexpected shape {W.shape}"

    # Synthetic but reproducible activations in a plausible range
    rng = np.random.default_rng(0)
    batch = 4
    X = rng.normal(loc=0.5, scale=0.3, size=(batch, in_f)).astype(np.float32)

    # Quantize both to uint8 (symmetric to range)
    Wq, w_scale, w_zp = quant_uint8_symmetric(W)
    Xq, x_scale, x_zp = quant_uint8_symmetric(X)

    print(f"[layer] SE_attention_score.0.weight  shape={W.shape}  "
          f"range=[{W.min():.4f}, {W.max():.4f}]")
    print(f"[quant] W: scale={w_scale:.6f}  zp={w_zp:.2f}")
    print(f"[quant] X: scale={x_scale:.6f}  zp={x_zp:.2f}")
    print(f"[Wq] min={Wq.min()} max={Wq.max()} mean={Wq.mean():.2f}")
    print(f"[Xq] min={Xq.min()} max={Xq.max()} mean={Xq.mean():.2f}")

    # Lay out for TPU: C[m,n] = sum_k A[k,m]*B[k,n]
    K, M, N = in_f, out_f, batch       # 128, 32, 4
    A = Wq.T.copy()                    # shape (K=128, M=32) ; A[k,m] = Wq[m,k]
    B = Xq.T.copy()                    # shape (K=128, N=4)  ; B[k,n] = Xq[n,k]

    # Golden: uint32 unsigned reference (this is exactly what the TPU computes)
    golden = (A.astype(np.uint32).T @ B.astype(np.uint32))  # (M, N)
    assert golden.shape == (M, N)

    # Optional sanity: reconstruct float y and compare to fp reference
    y_fp_ref = X @ W.T                                  # (batch, out_f)
    y_int  = (Wq.astype(np.int32) - w_zp).astype(np.float32) * w_scale
    x_int  = (Xq.astype(np.int32) - x_zp).astype(np.float32) * x_scale
    y_fp_q = x_int @ y_int.T  # quantized-then-dequantized fp reference
    err = np.abs(y_fp_ref - y_fp_q).max()
    print(f"[fp-vs-quant] max-abs error in dequantized y: {err:.4f}")

    with open(out_path, "w") as fd:
        fd.write("1\n")                                   # PATNUM = 1
        fd.write(f"{K:3x} {M:3x} {N:3x}\n")               # K M N
        write_uint8_matrix(fd, A)
        write_uint8_matrix(fd, B)
        write_uint32_matrix(fd, golden)

    print(f"[ok] wrote {out_path}  (K={K}, M={M}, N={N})")
    print(f"[ok] golden first row (uint32): {golden[0].tolist()}")


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "awn_case_input.txt"
    main(out)
