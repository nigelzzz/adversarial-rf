"""Load AWN.pkl, fold BN into preceding conv, calibrate per-tensor symmetric
int8 scales using a fixed synthetic input, and write build/quant.npz.

Quantization scheme:
  - Weights: per-tensor symmetric. scale_w = max(|W|) / 127.  W_q in [-127, 127].
  - Activations: per-tensor symmetric. scale_a = max(|a|) / 127, calibrated
    from a single-shot fp32 forward on a synthetic input.
  - Bias (folded): scale_b = scale_in * scale_w; b_q = round(b_fold / scale_b),
    stored as int32. Saturated to int32 just in case.
  - Tanh/Sigmoid output scales fixed: tanh -> 1/127 (range [-1,1]),
    sigmoid -> 1/255 (range [0,1] mapped roughly to [0, 127]).
  - Multiplier+shift representation for any rescale ratio M < 1:
      mul = round(M * 2^31); shift = 31. Hardware does (x*mul + 2^30) >>> 31.

Output build/quant.npz contains:
  weights/biases (int8/int32) for each conv/linear (BN-folded)
  scale_a[<tensor name>] = float
  Plus the calibrated input tensor itself (so inference is reproducible).
"""
from __future__ import annotations
import os, sys, types, json, math
import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
pkg = types.ModuleType("models"); pkg.__path__ = [ROOT]
sys.modules["models"] = pkg
import lifting as _lift
sys.modules["models.lifting"] = _lift
import model as awn_mod


def quant_w(W: np.ndarray):
    s = float(np.max(np.abs(W))) / 127.0 if np.any(W) else 1.0
    Wq = np.round(W / s).clip(-127, 127).astype(np.int8)
    return Wq, s


def quant_a(a: np.ndarray):
    s = float(np.max(np.abs(a))) / 127.0 if np.any(a) else 1.0
    return s


def fold_bn_into_conv(W, b, bn_w, bn_b, bn_mean, bn_var, eps):
    """Fold BatchNorm parameters into the preceding conv weights+bias.
    BN: y = bn_w * (x - mean)/sqrt(var+eps) + bn_b
    Folded: W' = W * (bn_w/sqrt(var+eps))[reshape], b' = (b-mean)*bn_w/sqrt(...) + bn_b
    """
    inv = bn_w / np.sqrt(bn_var + eps)
    if W.ndim == 4:    # Conv2d: (out, in, kh, kw)
        Wf = W * inv.reshape(-1, 1, 1, 1)
    else:              # Conv1d: (out, in, k)
        Wf = W * inv.reshape(-1, 1, 1)
    if b is None:
        b = np.zeros_like(bn_b)
    bf = (b - bn_mean) * inv + bn_b
    return Wf.astype(np.float32), bf.astype(np.float32)


def main():
    sd = torch.load(os.path.join(ROOT, "2016.10a_AWN.pkl"),
                    map_location="cpu", weights_only=False)

    eps = 1e-5  # PyTorch default

    # --- Fold BN into conv1 (Conv2d 1->64, kernel (2,7)) ---
    W1 = sd["conv1.1.weight"].numpy()                 # (64,1,2,7)
    bn1_w = sd["conv1.2.weight"].numpy()
    bn1_b = sd["conv1.2.bias"].numpy()
    bn1_m = sd["conv1.2.running_mean"].numpy()
    bn1_v = sd["conv1.2.running_var"].numpy()
    W1f, b1f = fold_bn_into_conv(W1, None, bn1_w, bn1_b, bn1_m, bn1_v, eps)

    # --- Fold BN into conv2 (Conv1d 64->64, k=5) ---
    W2 = sd["conv2.0.weight"].numpy()                 # (64,64,5)
    bn2_w = sd["conv2.1.weight"].numpy()
    bn2_b = sd["conv2.1.bias"].numpy()
    bn2_m = sd["conv2.1.running_mean"].numpy()
    bn2_v = sd["conv2.1.running_var"].numpy()
    W2f, b2f = fold_bn_into_conv(W2, None, bn2_w, bn2_b, bn2_m, bn2_v, eps)

    # --- Inner Conv1d ops have biases, no BN ---
    Wu1 = sd["levels.level_0.wavelet.U.operator.1.weight"].numpy()
    bu1 = sd["levels.level_0.wavelet.U.operator.1.bias"].numpy()
    Wu4 = sd["levels.level_0.wavelet.U.operator.4.weight"].numpy()
    bu4 = sd["levels.level_0.wavelet.U.operator.4.bias"].numpy()
    Wp1 = sd["levels.level_0.wavelet.P.operator.1.weight"].numpy()
    bp1 = sd["levels.level_0.wavelet.P.operator.1.bias"].numpy()
    Wp4 = sd["levels.level_0.wavelet.P.operator.4.weight"].numpy()
    bp4 = sd["levels.level_0.wavelet.P.operator.4.bias"].numpy()

    # --- Linear layers ---
    W_se0 = sd["SE_attention_score.0.weight"].numpy()           # (32, 128)
    W_se3 = sd["SE_attention_score.3.weight"].numpy()           # (128, 32)
    W_fc0 = sd["fc.0.weight"].numpy()                           # (320, 128)
    b_fc0 = sd["fc.0.bias"].numpy()                             # (320,)
    W_fc2 = sd["fc.2.weight"].numpy()                           # (11, 320)
    b_fc2 = sd["fc.2.bias"].numpy()                             # (11,)

    # --- Build a fp32 model with folded-in BN params for activation calibration ---
    model = awn_mod.AWN(num_classes=11, num_levels=1, in_channels=64,
                        kernel_size=3, latent_dim=320,
                        regu_details=0.01, regu_approx=0.01).eval()
    model.load_state_dict(sd)

    # Synthetic but reproducible IQ-like input.
    rng = np.random.default_rng(7)
    x_fp = rng.normal(0.0, 0.5, size=(1, 2, 128)).astype(np.float32)

    # Activation calibration: capture every intermediate via a sequential walk
    # that mirrors what refmodel.py will do.
    acts = {}

    def cal(x, name):
        acts[name] = float(np.max(np.abs(x))) if np.any(x) else 1e-6

    x = x_fp                                          # input
    cal(x, "input")

    # conv1: ZeroPad((3,3,0,0)) on last dim, Conv2d, BN, LeakyReLU
    x_pad = np.pad(x, ((0,0),(0,0),(0,0)), mode='constant')   # NCHW; need 4D
    x4 = x[:, np.newaxis, :, :]                                # (1,1,2,128)
    x4_pad = np.pad(x4, ((0,0),(0,0),(0,0),(3,3)), mode='constant')
    cal(x4_pad, "conv1_in")
    y_c1 = F.conv2d(torch.tensor(x4_pad), torch.tensor(W1f),
                    bias=torch.tensor(b1f)).numpy()           # (1,64,1,128)
    cal(y_c1, "conv1_out")
    y_c1_a = np.where(y_c1 >= 0, y_c1, 0.01 * y_c1)
    cal(y_c1_a, "conv1_act")
    y_c1_a = y_c1_a.squeeze(2)                                # (1,64,128)

    # conv2
    cal(y_c1_a, "conv2_in")
    y_c2 = F.conv1d(torch.tensor(y_c1_a), torch.tensor(W2f),
                    bias=torch.tensor(b2f), padding=2).numpy()
    cal(y_c2, "conv2_out")
    y_c2_a = np.where(y_c2 >= 0, y_c2, 0.01 * y_c2)
    cal(y_c2_a, "conv2_act")

    # Lifting: split
    even = y_c2_a[:, :, ::2]                                  # (1,64,64)
    odd  = y_c2_a[:, :, 1::2]                                 # (1,64,64)
    cal(even, "lift_even"); cal(odd, "lift_odd")

    # U(odd): ReflPad -> Conv1d k=3 -> LeakyReLU -> Conv1d k=3 -> Tanh
    odd_p = np.pad(odd, ((0,0),(0,0),(2,2)), mode='reflect')
    cal(odd_p, "u_pad")
    yu1 = F.conv1d(torch.tensor(odd_p), torch.tensor(Wu1),
                   bias=torch.tensor(bu1)).numpy()             # (1,64,66)
    cal(yu1, "u_conv1_out")
    yu1_a = np.where(yu1 >= 0, yu1, 0.01 * yu1)
    cal(yu1_a, "u_conv1_act")
    yu4 = F.conv1d(torch.tensor(yu1_a), torch.tensor(Wu4),
                   bias=torch.tensor(bu4)).numpy()             # (1,64,64)
    cal(yu4, "u_conv2_out")
    yu = np.tanh(yu4)
    cal(yu, "u_out")  # range [-1,1]

    # c = even + U(odd). Scale must match for hardware add: pick max of two.
    c = even + yu
    cal(c, "lift_c")

    # P(c)
    c_p = np.pad(c, ((0,0),(0,0),(2,2)), mode='reflect')
    cal(c_p, "p_pad")
    yp1 = F.conv1d(torch.tensor(c_p), torch.tensor(Wp1),
                   bias=torch.tensor(bp1)).numpy()
    cal(yp1, "p_conv1_out")
    yp1_a = np.where(yp1 >= 0, yp1, 0.01 * yp1)
    cal(yp1_a, "p_conv1_act")
    yp4 = F.conv1d(torch.tensor(yp1_a), torch.tensor(Wp4),
                   bias=torch.tensor(bp4)).numpy()
    cal(yp4, "p_conv2_out")
    yp = np.tanh(yp4)
    cal(yp, "p_out")

    d = odd - yp
    cal(d, "lift_d")

    # AvgPool over time (length 64)
    avg_d = d.mean(axis=2, keepdims=True)
    avg_c = c.mean(axis=2, keepdims=True)
    cal(avg_d, "avg_d"); cal(avg_c, "avg_c")

    # Concat dim 1 -> (1, 128, 1) then view (1, 128)
    cat = np.concatenate([avg_d, avg_c], axis=1).reshape(1, 128)
    cal(cat, "concat")

    # SE
    se0 = cat @ W_se0.T
    cal(se0, "se0_out")
    se0_a = np.maximum(se0, 0)
    cal(se0_a, "se0_act")
    se3 = se0_a @ W_se3.T
    cal(se3, "se3_out")
    se3_a = 1.0 / (1.0 + np.exp(-se3))
    cal(se3_a, "se3_act")  # in [0,1]

    se_mul = se3_a * cat
    cal(se_mul, "se_mul")

    # fc
    fc0 = se_mul @ W_fc0.T + b_fc0
    cal(fc0, "fc0_out")
    fc0_a = np.where(fc0 >= 0, fc0, 0.01 * fc0)
    cal(fc0_a, "fc0_act")
    fc2 = fc0_a @ W_fc2.T + b_fc2
    cal(fc2, "fc2_out")  # logits

    # Compare against PyTorch's full forward
    with torch.no_grad():
        y_ref, _ = model(torch.tensor(x_fp))
    fp_logits = y_ref.numpy().reshape(-1)
    err = np.max(np.abs(fp_logits - fc2.reshape(-1)))
    print(f"[fp32 sanity] our manual forward vs torch forward: max abs err = {err:.6f}")
    assert err < 1e-3, "fp32 forward mismatch"

    # Quantize all weights
    def qw(W): Wq, s = quant_w(W); return Wq, s
    W1q, s_W1 = qw(W1f); W2q, s_W2 = qw(W2f)
    Wu1q, s_Wu1 = qw(Wu1); Wu4q, s_Wu4 = qw(Wu4)
    Wp1q, s_Wp1 = qw(Wp1); Wp4q, s_Wp4 = qw(Wp4)
    Wse0q, s_Wse0 = qw(W_se0); Wse3q, s_Wse3 = qw(W_se3)
    Wfc0q, s_Wfc0 = qw(W_fc0); Wfc2q, s_Wfc2 = qw(W_fc2)

    # Activation scales = max(|.|)/127
    s_a = {k: v / 127.0 for k, v in acts.items()}

    # Quantize biases at scale = s_in * s_w
    def qb(b, s_in, s_w):
        sb = s_in * s_w
        bq = np.round(b / sb).clip(-(1<<31)+1, (1<<31)-1).astype(np.int32)
        return bq, sb

    b1q,  s_b1  = qb(b1f,   s_a["conv1_in"],  s_W1)
    b2q,  s_b2  = qb(b2f,   s_a["conv2_in"],  s_W2)
    bu1q, s_bu1 = qb(bu1,   s_a["u_pad"],     s_Wu1)
    bu4q, s_bu4 = qb(bu4,   s_a["u_conv1_act"], s_Wu4)
    bp1q, s_bp1 = qb(bp1,   s_a["p_pad"],     s_Wp1)
    bp4q, s_bp4 = qb(bp4,   s_a["p_conv1_act"], s_Wp4)
    bfc0q, s_bfc0 = qb(b_fc0, s_a["se_mul"],  s_Wfc0)
    bfc2q, s_bfc2 = qb(b_fc2, s_a["fc0_act"], s_Wfc2)

    out = {
        "x_fp": x_fp,
        "fp_logits": fp_logits,
        # weights
        "W1": W1q, "b1": b1q, "s_W1": s_W1, "s_b1": s_b1,
        "W2": W2q, "b2": b2q, "s_W2": s_W2, "s_b2": s_b2,
        "Wu1": Wu1q, "bu1": bu1q, "s_Wu1": s_Wu1,
        "Wu4": Wu4q, "bu4": bu4q, "s_Wu4": s_Wu4,
        "Wp1": Wp1q, "bp1": bp1q, "s_Wp1": s_Wp1,
        "Wp4": Wp4q, "bp4": bp4q, "s_Wp4": s_Wp4,
        "Wse0": Wse0q, "s_Wse0": s_Wse0,
        "Wse3": Wse3q, "s_Wse3": s_Wse3,
        "Wfc0": Wfc0q, "bfc0": bfc0q, "s_Wfc0": s_Wfc0,
        "Wfc2": Wfc2q, "bfc2": bfc2q, "s_Wfc2": s_Wfc2,
    }
    # activation scales
    for k, v in s_a.items():
        out[f"s_{k}"] = v
    np.savez(os.path.join(ROOT, "build", "quant.npz"), **out)
    print(f"[ok] wrote build/quant.npz with {len(out)} arrays/scalars")
    print(f"     W1 range: {W1q.min()}..{W1q.max()}; W_fc2 range: {Wfc2q.min()}..{Wfc2q.max()}")


if __name__ == "__main__":
    main()
