"""Operator profiling for the AWN model.

Walks the AWN forward graph with a synthetic [1, 2, 128] input (the RadioML
2016.10a I/Q sample shape) and records every op the hardware will need to
implement. Functional ops inside `forward` (split, add, sub, mul, cat, view,
unsqueeze, squeeze, avgpool) are captured by monkey-patching torch primitives.

Outputs:
  build/ops.json   — machine-readable list of ops in execution order
  build/ops.txt    — human table

Usage:
  python3 sw/profile_awn.py
"""
from __future__ import annotations
import os, sys, json, types
from collections import OrderedDict
from typing import Any

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# Shim model imports
sys.path.insert(0, ROOT)
pkg = types.ModuleType("models"); pkg.__path__ = [ROOT]
sys.modules["models"] = pkg
import lifting as _lift
sys.modules["models.lifting"] = _lift
import model as awn_mod


# --- op record store ----------------------------------------------------------

OPS: list[dict] = []
SEQ = [0]


def _shape(t):
    if isinstance(t, torch.Tensor):
        return list(t.shape)
    return None


def _record(kind: str, name: str, in_shapes, out_shape, **extra):
    SEQ[0] += 1
    rec = {
        "seq":   SEQ[0],
        "kind":  kind,
        "name":  name,
        "in":    in_shapes,
        "out":   out_shape,
    }
    rec.update(extra)
    OPS.append(rec)


# --- nn.Module hooks ----------------------------------------------------------

def hook_module(name: str, module: nn.Module):
    def _h(mod, inputs, output):
        in_shapes = [_shape(i) for i in inputs]
        out_shape = _shape(output) if isinstance(output, torch.Tensor) else \
                    [_shape(o) for o in output]
        kind = type(mod).__name__
        extra: dict[str, Any] = {}

        if isinstance(mod, nn.Conv2d):
            kh, kw = mod.kernel_size
            cin, cout = mod.in_channels, mod.out_channels
            extra.update(in_ch=cin, out_ch=cout, kernel=[kh, kw],
                         stride=list(mod.stride), padding=list(mod.padding),
                         bias=mod.bias is not None,
                         w_shape=list(mod.weight.shape))
            # MACs: out_h * out_w * cout * cin * kh * kw
            _, _, oh, ow = output.shape
            extra["macs"] = oh * ow * cout * cin * kh * kw
        elif isinstance(mod, nn.Conv1d):
            (k,) = mod.kernel_size
            cin, cout = mod.in_channels, mod.out_channels
            extra.update(in_ch=cin, out_ch=cout, kernel=k,
                         stride=list(mod.stride), padding=list(mod.padding),
                         bias=mod.bias is not None,
                         w_shape=list(mod.weight.shape))
            _, _, ot = output.shape
            extra["macs"] = ot * cout * cin * k
        elif isinstance(mod, nn.Linear):
            extra.update(in_f=mod.in_features, out_f=mod.out_features,
                         bias=mod.bias is not None,
                         w_shape=list(mod.weight.shape))
            extra["macs"] = mod.in_features * mod.out_features * (
                output.shape[0] if output.dim() > 1 else 1)
        elif isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d)):
            extra.update(num_features=mod.num_features,
                         note="fold into preceding conv")
        elif isinstance(mod, nn.LeakyReLU):
            extra.update(negative_slope=mod.negative_slope)
        elif isinstance(mod, nn.ReflectionPad1d):
            extra.update(padding=mod.padding)
        elif isinstance(mod, nn.ZeroPad2d):
            extra.update(padding=list(mod.padding))
        elif isinstance(mod, nn.AdaptiveAvgPool1d):
            extra.update(output_size=mod.output_size)
        # Containers: skip
        if isinstance(mod, (nn.Sequential, nn.ModuleList,
                            awn_mod.AWN, awn_mod.LevelTWaveNet,
                            _lift.LiftingScheme, _lift.Operator,
                            _lift.Splitting)):
            return

        _record(kind, name, in_shapes, out_shape, **extra)
    return module.register_forward_hook(_h)


# --- functional-op patches ----------------------------------------------------

_orig_add  = torch.Tensor.__add__
_orig_sub  = torch.Tensor.__sub__
_orig_mul  = torch.Tensor.__mul__
_orig_cat  = torch.cat
_orig_mul_fn = torch.mul

_TRACE_FUNCTIONAL = [True]  # off-switch during model construction


def _w_add(self, other):
    out = _orig_add(self, other)
    if _TRACE_FUNCTIONAL[0] and isinstance(other, torch.Tensor):
        _record("Add", "elementwise_add",
                [_shape(self), _shape(other)], _shape(out))
    return out


def _w_sub(self, other):
    out = _orig_sub(self, other)
    if _TRACE_FUNCTIONAL[0] and isinstance(other, torch.Tensor):
        _record("Sub", "elementwise_sub",
                [_shape(self), _shape(other)], _shape(out))
    return out


def _w_mul(self, other):
    out = _orig_mul(self, other)
    if _TRACE_FUNCTIONAL[0] and isinstance(other, torch.Tensor):
        _record("Mul", "elementwise_mul",
                [_shape(self), _shape(other)], _shape(out))
    return out


def _w_cat(tensors, dim=0):
    out = _orig_cat(tensors, dim=dim)
    if _TRACE_FUNCTIONAL[0]:
        _record("Concat", f"concat_dim{dim}",
                [_shape(t) for t in tensors], _shape(out), dim=dim)
    return out


def _w_mul_fn(a, b):
    out = _orig_mul_fn(a, b)
    if _TRACE_FUNCTIONAL[0] and isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        _record("Mul", "torch_mul",
                [_shape(a), _shape(b)], _shape(out))
    return out


def install_patches():
    torch.Tensor.__add__ = _w_add
    torch.Tensor.__sub__ = _w_sub
    torch.Tensor.__mul__ = _w_mul
    torch.cat = _w_cat
    torch.mul = _w_mul_fn


def remove_patches():
    torch.Tensor.__add__ = _orig_add
    torch.Tensor.__sub__ = _orig_sub
    torch.Tensor.__mul__ = _orig_mul
    torch.cat = _orig_cat
    torch.mul = _orig_mul_fn


# --- main ---------------------------------------------------------------------

def build_model():
    # Match the saved checkpoint shape: 11 classes, 64 ch, 1 level
    return awn_mod.AWN(num_classes=11, num_levels=1, in_channels=64,
                       kernel_size=3, latent_dim=320,
                       regu_details=0.01, regu_approx=0.01).eval()


def main():
    _TRACE_FUNCTIONAL[0] = False
    model = build_model()

    # Load the trained weights so shapes and BN stats are realistic
    sd = torch.load(os.path.join(ROOT, "2016.10a_AWN.pkl"),
                    map_location="cpu", weights_only=False)
    model.load_state_dict(sd)

    # Register hooks on every leaf module
    handles = []
    for name, m in model.named_modules():
        if name == "":
            continue
        # Skip pure containers; we want leaves and the lifting scheme components
        if isinstance(m, (nn.Sequential, nn.ModuleList)):
            continue
        handles.append(hook_module(name, m))

    install_patches()
    _TRACE_FUNCTIONAL[0] = True

    x = torch.zeros(1, 2, 128)
    with torch.no_grad():
        y, _ = model(x)
    print(f"Forward OK. logits shape: {tuple(y.shape)}")

    _TRACE_FUNCTIONAL[0] = False
    remove_patches()
    for h in handles:
        h.remove()

    out_dir = os.path.join(ROOT, "build")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "ops.json"), "w") as f:
        json.dump(OPS, f, indent=2)

    # Tally + write human table
    total_macs = sum(o.get("macs", 0) for o in OPS)
    kinds = {}
    for o in OPS:
        kinds.setdefault(o["kind"], 0)
        kinds[o["kind"]] += 1

    with open(os.path.join(out_dir, "ops.txt"), "w") as f:
        f.write(f"AWN forward graph — {len(OPS)} ops, {total_macs:,} MACs total\n\n")
        f.write(f"{'#':>3}  {'kind':<22} {'name':<48} {'in':<28} {'out':<22} {'macs':>12}\n")
        f.write("-" * 140 + "\n")
        for o in OPS:
            in_repr  = ",".join(str(s) for s in o["in"]) if o["in"] else "-"
            out_repr = str(o["out"])
            f.write(f"{o['seq']:>3}  {o['kind']:<22} {o['name']:<48} "
                    f"{in_repr:<28} {out_repr:<22} {o.get('macs', 0):>12,}\n")
        f.write("\n")
        f.write("By kind:\n")
        for k in sorted(kinds, key=lambda k: -kinds[k]):
            f.write(f"  {k:<24} x {kinds[k]}\n")

    print(f"\nWrote {out_dir}/ops.json and {out_dir}/ops.txt")
    print(f"Op count: {len(OPS)}   Total MACs: {total_macs:,}")
    print("By kind:")
    for k in sorted(kinds, key=lambda k: -kinds[k]):
        print(f"  {k:<24} x {kinds[k]}")


if __name__ == "__main__":
    main()
