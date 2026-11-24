#!/usr/bin/env python3
"""f32 CPU reference for sliding layers 0 → 1."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "spike"))
from export_layer0_f32 import route  # noqa: E402
from parity_layer0_pieces import (  # noqa: E402
    attention_residual,
    rms_norm,
    short_conv,
    sliding_attn,
)
from parity_moe_layer0 import dequant_nvfp4_rows, softcapped_swiglu  # noqa: E402

SNAP = (
    Path.home()
    / ".cache/huggingface/hub/models--thinkingmachines--Inkling-2-Mini"
    / "snapshots/ed40e7da625f594fc9c4990f5f66be24873c22d7"
)
H = 2304


def load_tensors():
    return mx.load(str(SNAP / "model-00001-of-00001.safetensors"))


def bf(tensors, name):
    return np.array(tensors[name].astype(mx.float32))


def taps(tensors, name):
    t = bf(tensors, name)
    return t[:, 0, :]


def layer_bundle(tensors, i: int):
    p = f"model.llm.layers.{i}."
    attn = {
        "wq": bf(tensors, p + "attn.wq_du.weight"),
        "wk": bf(tensors, p + "attn.wk_dv.weight"),
        "wv": bf(tensors, p + "attn.wv_dv.weight"),
        "wr": bf(tensors, p + "attn.wr_du.weight"),
        "wo": bf(tensors, p + "attn.wo_ud.weight"),
        "gate": bf(tensors, p + "attn.gate_attn.weight"),
        "sink": bf(tensors, p + "attn.attn_sink_proj.weight"),
        "q_scale": bf(tensors, p + "attn.q_l2_clip.scale"),
        "rel_scale": bf(tensors, p + "attn.rel_l2_clip.scale"),
        "rel_proj": bf(tensors, p + "attn.rel_logits_proj.proj"),
        "k_sconv": taps(tensors, p + "attn.k_sconv.weight"),
        "v_sconv": taps(tensors, p + "attn.v_sconv.weight"),
    }
    attn_query = None
    if i > 0:
        attn_query = bf(tensors, f"model.llm.attn_res.attn_queries.{i - 1}")
    return {
        "attn": attn,
        "attn_norm": bf(tensors, p + "attn_norm.weight"),
        "mlp_norm": bf(tensors, p + "mlp_norm.weight"),
        "attn_sconv": taps(tensors, p + "attn_sconv.weight"),
        "mlp_sconv": taps(tensors, p + "mlp_sconv.weight"),
        "gate_w": bf(tensors, p + "mlp.gate.weight"),
        "gate_b": bf(tensors, p + "mlp.gate.bias"),
        "attn_query": attn_query,
        "mlp_query": bf(tensors, f"model.llm.attn_res.mlp_queries.{i}"),
        "boundary": (i % 6) == 0,
        "layer_idx": i,
    }


def run_moe(tensors, layer_idx, moe_in, indices, weights):
    p = f"model.llm.layers.{layer_idx}.mlp.experts."
    w13 = np.array(tensors[p + "w13_weight"])
    s13 = np.array(tensors[p + "w13_weight_scale"])
    a13 = np.array(tensors[p + "w13_weight_amax"], dtype=np.float32)
    w2 = np.array(tensors[p + "w2_weight"])
    s2 = np.array(tensors[p + "w2_weight_scale"])
    a2 = np.array(tensors[p + "w2_weight_amax"], dtype=np.float32)
    gate_w, up_w = w13[:, 0::2, :], w13[:, 1::2, :]
    gate_s, up_s = s13[:, 0::2, :], s13[:, 1::2, :]
    unique = sorted(set(int(i) for i in indices.reshape(-1)))
    cache = {
        e: {
            "gate": dequant_nvfp4_rows(gate_w[e], gate_s[e], float(a13[e])),
            "up": dequant_nvfp4_rows(up_w[e], up_s[e], float(a13[e])),
            "down": dequant_nvfp4_rows(w2[e], s2[e], float(a2[e])),
        }
        for e in unique
    }
    t = moe_in.shape[0]
    out = np.zeros_like(moe_in)
    for i in range(t):
        acc = np.zeros(H, np.float32)
        for r in range(8):
            e = int(indices[i, r])
            w = float(weights[i, r])
            h = softcapped_swiglu(
                cache[e]["up"] @ moe_in[i], cache[e]["gate"] @ moe_in[i]
            )
            acc += w * (cache[e]["down"] @ h)
        out[i] = acc
    return out


def run_sliding(tensors, x, blocks, w):
    if w["attn_query"] is None:
        xin = rms_norm(x, w["attn_norm"])
    else:
        mixed = attention_residual(blocks, x, w["attn_query"], True)
        xin = rms_norm(mixed, w["attn_norm"])
    attn = sliding_attn(xin, w["attn"])
    residual = short_conv(attn, w["attn_sconv"])
    if not w["boundary"]:
        residual = residual + x
    mlp_blocks = list(blocks)
    if w["boundary"]:
        mlp_blocks = mlp_blocks + [x]
    mixed = attention_residual(mlp_blocks, residual, w["mlp_query"], True)
    moe_in = rms_norm(mixed, w["mlp_norm"])
    idx, wt = route(moe_in, w["gate_w"], w["gate_b"])
    moe_out = run_moe(tensors, w["layer_idx"], moe_in, idx, wt)
    out = residual + short_conv(moe_out, w["mlp_sconv"])
    next_blocks = list(blocks)
    if w["boundary"]:
        next_blocks = next_blocks + [x]
    return out, next_blocks


def main():
    tensors = load_tensors()
    g = np.load(ROOT / "goldens/layer0_f32.npz")
    x = g["x_in"].astype(np.float32)
    w0, w1 = layer_bundle(tensors, 0), layer_bundle(tensors, 1)
    x1, blocks = run_sliding(tensors, x, [], w0)
    x2, _ = run_sliding(tensors, x1, blocks, w1)
    out = ROOT / "goldens/layer01_f32.npz"
    np.savez_compressed(out, x_in=x, after_layer0=x1, x_out=x2)
    l1 = np.load(ROOT / "goldens/layer1.npz")
    meta = {
        "path": str(out.relative_to(ROOT)),
        "vs_mlx_layer1_maxabs": float(np.abs(x2 - l1["x_out"]).max()),
        "x_out_absmax": float(np.abs(x2).max()),
        "bytes": out.stat().st_size,
    }
    (ROOT / "goldens/layer01_f32.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
