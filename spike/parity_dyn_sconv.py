#!/usr/bin/env python3
"""Parity for DynamicShortConvolution (layer 5 k_sconv)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_vlm import load

ROOT = Path(__file__).resolve().parents[1]
SNAP = (
    Path.home()
    / ".cache/huggingface/hub/models--thinkingmachines--Inkling-2-Mini"
    / "snapshots/ed40e7da625f594fc9c4990f5f66be24873c22d7"
)
K = 4
N_KV = 8
CH = 1024  # N_KV * DH


def dyn_sconv(x, dyn_in, static_w, weight_proj, n_heads=N_KV):
    """x, dyn_in: [T, C] / [T, H]; static_w [K, C]; weight_proj [n_heads*K, H]."""
    t, c = x.shape
    assert c == CH
    xf = x.astype(np.float32)
    state = np.zeros((K - 1, c), np.float32)
    ctx = np.concatenate([state, xf], 0)  # [K-1+T, C]
    # dynamic: [T, n_heads, 1, K]
    dyn = (dyn_in @ weight_proj.T).reshape(t, n_heads, 1, K).astype(np.float32)
    static = static_w.T.reshape(n_heads, -1, K)  # [n_heads, C/n_heads, K]
    weights = static[None] + dyn  # broadcast [T, n_heads, C/n_heads, K]
    out = np.zeros((t, n_heads, c // n_heads), np.float32)
    for lag in range(K):
        values = ctx[K - 1 - lag : K - 1 - lag + t].reshape(t, n_heads, c // n_heads)
        out = out + values * weights[..., lag]
    out = out.reshape(t, c)
    return ((out + xf) / math.sqrt(2 * K + 1)).astype(np.float32)


def main():
    print("loading…", flush=True)
    model, _ = load(str(SNAP))
    layer = model.language_model.model.layers[5]
    conv = layer.self_attn.k_sconv
    # synthetic but model weights
    rng = np.random.default_rng(0)
    t = 14
    x = rng.standard_normal((1, t, CH), dtype=np.float32) * 0.1
    dyn = rng.standard_normal((1, t, 2304), dtype=np.float32) * 0.1
    ref = conv(mx.array(x).astype(mx.bfloat16), mx.array(dyn).astype(mx.bfloat16))
    mx.eval(ref)
    ref = np.array(ref.astype(mx.float32))[0]

    static = np.array(conv.static_weight.astype(mx.float32))  # [K, C]
    wproj = np.array(conv.weight_proj.weight.astype(mx.float32))  # [n_heads*K, H]
    pred = dyn_sconv(x[0], dyn[0], static, wproj)
    # compare bf16-rounded
    pred_bf = np.array(mx.array(pred).astype(mx.bfloat16).astype(mx.float32))
    err = np.abs(pred_bf - ref)
    report = {
        "maxabs": float(err.max()),
        "meanabs": float(err.mean()),
        "static_shape": list(static.shape),
        "wproj_shape": list(wproj.shape),
    }
    print(json.dumps(report, indent=2))
    np.savez_compressed(
        ROOT / "goldens/dyn_sconv.npz",
        x=x[0],
        dyn_in=dyn[0],
        static_w=static,
        weight_proj=wproj,
        out_f32=pred,
        out_bf16=ref,
    )
    if report["maxabs"] > 0.05:
        raise SystemExit("dyn sconv parity failed")
    print("OK")


if __name__ == "__main__":
    main()
