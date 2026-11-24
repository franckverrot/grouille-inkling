#!/usr/bin/env python3
"""Full layer-0 f32 reference (CPU) for Rust parity."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from parity_layer0_pieces import (
    ROOT,
    attention_residual,
    load_weights,
    rms_norm,
    short_conv,
    sliding_attn,
)
from parity_moe_layer0 import dequant_nvfp4_rows, softcapped_swiglu
import mlx.core as mx

SNAP = (
    Path.home()
    / ".cache/huggingface/hub/models--thinkingmachines--Inkling-2-Mini"
    / "snapshots/ed40e7da625f594fc9c4990f5f66be24873c22d7"
)


def route(x: np.ndarray, gate_w: np.ndarray, gate_b: np.ndarray, top_k=8, route_scale=8.0):
    scores = 1.0 / (1.0 + np.exp(-(x.astype(np.float32) @ gate_w.T)))
    scores_b = scores + gate_b
    indices = np.argpartition(-scores_b, top_k - 1, axis=-1)[..., :top_k]
    # stable order: sort selected by score desc for determinism
    gathered = np.take_along_axis(scores, indices, axis=-1)
    order = np.argsort(-gathered, axis=-1)
    indices = np.take_along_axis(indices, order, axis=-1)
    gathered = np.take_along_axis(scores, indices, axis=-1)
    weights = gathered / (gathered.sum(axis=-1, keepdims=True) + 1e-20) * route_scale
    return indices.astype(np.int32), weights.astype(np.float32)


def run_moe(moe_in: np.ndarray, indices: np.ndarray, weights: np.ndarray) -> np.ndarray:
    tensors = mx.load(str(SNAP / "model-00001-of-00001.safetensors"))
    p = "model.llm.layers.0.mlp.experts."
    w13 = np.array(tensors[p + "w13_weight"])
    s13 = np.array(tensors[p + "w13_weight_scale"])
    a13 = np.array(tensors[p + "w13_weight_amax"], dtype=np.float32)
    w2 = np.array(tensors[p + "w2_weight"])
    s2 = np.array(tensors[p + "w2_weight_scale"])
    a2 = np.array(tensors[p + "w2_weight_amax"], dtype=np.float32)
    gate_w, up_w = w13[:, 0::2, :], w13[:, 1::2, :]
    gate_s, up_s = s13[:, 0::2, :], s13[:, 1::2, :]
    unique = sorted(set(int(i) for i in indices.reshape(-1)))
    cache = {}
    for e in unique:
        cache[e] = {
            "gate": dequant_nvfp4_rows(gate_w[e], gate_s[e], float(a13[e])),
            "up": dequant_nvfp4_rows(up_w[e], up_s[e], float(a13[e])),
            "down": dequant_nvfp4_rows(w2[e], s2[e], float(a2[e])),
        }
    t, h = moe_in.shape
    out = np.zeros_like(moe_in)
    for i in range(t):
        acc = np.zeros(h, np.float32)
        for r in range(8):
            e = int(indices[i, r])
            w = float(weights[i, r])
            hid = softcapped_swiglu(
                cache[e]["up"] @ moe_in[i], cache[e]["gate"] @ moe_in[i]
            )
            acc += w * (cache[e]["down"] @ hid)
        out[i] = acc
    return out


def main() -> None:
    g = np.load(ROOT / "goldens/layer0.npz")
    w = load_weights()
    x_in = g["x_in"][0].astype(np.float32)
    tensors = mx.load(str(SNAP / "model-00001-of-00001.safetensors"))
    gate_w = np.array(tensors["model.llm.layers.0.mlp.gate.weight"].astype(mx.float32))
    gate_b = np.array(tensors["model.llm.layers.0.mlp.gate.bias"].astype(mx.float32))

    x_norm = rms_norm(x_in, w["attn_norm"])
    attn = sliding_attn(x_norm, w["attn"])
    residual = short_conv(attn, w["attn_sconv"])
    mixed = attention_residual([x_in], residual, w["mlp_query"], True)
    moe_in = rms_norm(mixed, w["mlp_norm"])
    indices, weights = route(moe_in, gate_w, gate_b)
    moe_out = run_moe(moe_in, indices, weights)
    x_out = residual + short_conv(moe_out, w["mlp_sconv"])

    out = ROOT / "goldens/layer0_f32.npz"
    np.savez_compressed(
        out,
        x_in=x_in,
        attn_out=attn,
        residual_after_attn=residual,
        mlp_input_mixed=mixed,
        moe_in=moe_in,
        moe_out=moe_out,
        x_out=x_out,
        route_indices=indices,
        route_weights=weights,
        input_ids=g["input_ids"],
    )
    meta = {
        "path": str(out.relative_to(ROOT)),
        "x_out_absmax": float(np.abs(x_out).max()),
        "vs_mlx_bf16_x_out_maxabs": float(
            np.abs(x_out - g["x_out"][0].astype(np.float32)).max()
        ),
        "bytes": out.stat().st_size,
    }
    (ROOT / "goldens/layer0_f32.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
