#!/usr/bin/env python3
"""f32 CPU reference for layers 0 → 5 (sliding prelude + first full attn)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "spike"))
from export_layer0_f32 import route  # noqa: E402
from export_layer01_f32 import layer_bundle, load_tensors, run_sliding  # noqa: E402
from parity_full_attn import full_attn  # noqa: E402
from parity_layer0_pieces import attention_residual, rms_norm, short_conv  # noqa: E402
from parity_moe_layer0 import dequant_nvfp4_rows, softcapped_swiglu  # noqa: E402

H = 2304


def bf(tensors, name):
    return np.array(tensors[name].astype(mx.float32))


def taps(tensors, name):
    return bf(tensors, name)[:, 0, :]


def full_bundle(tensors, i: int):
    p = f"model.llm.layers.{i}."
    return {
        "attn": {
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
            "k_static": bf(tensors, p + "attn.k_sconv.static_w"),
            "k_wproj": bf(tensors, p + "attn.k_sconv.weight_proj"),
            "v_static": bf(tensors, p + "attn.v_sconv.static_w"),
            "v_wproj": bf(tensors, p + "attn.v_sconv.weight_proj"),
        },
        "q_norm": bf(tensors, p + "q_input_norm.weight"),
        "k_norm": bf(tensors, p + "k_input_norm.weight"),
        "v_norm": bf(tensors, p + "v_input_norm.weight"),
        "mlp_norm": bf(tensors, p + "mlp_norm.weight"),
        "attn_sconv": taps(tensors, p + "attn_sconv.weight"),
        "mlp_sconv": taps(tensors, p + "mlp_sconv.weight"),
        "gate_w": bf(tensors, p + "mlp.gate.weight"),
        "gate_b": np.array(tensors[p + "mlp.gate.bias"], dtype=np.float32),
        "attn_queries": bf(tensors, f"model.llm.attn_res.attn_queries.{i - 1}"),
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
    out = np.zeros_like(moe_in)
    for i in range(moe_in.shape[0]):
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


def run_full(tensors, x, blocks, w):
    mixed = attention_residual(blocks, x, w["attn_queries"], True)  # [T,3,H]
    q_in = rms_norm(mixed[:, 0], w["q_norm"])
    k_in = rms_norm(mixed[:, 1], w["k_norm"])
    v_in = rms_norm(mixed[:, 2], w["v_norm"])
    attn = full_attn(q_in, k_in, v_in, w["attn"])
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
    print("loading…", flush=True)
    tensors = load_tensors()
    g = np.load(ROOT / "goldens/layer0_f32.npz")
    x = g["x_in"].astype(np.float32)
    blocks: list[np.ndarray] = []
    after = {}
    for i in range(5):
        x, blocks = run_sliding(tensors, x, blocks, layer_bundle(tensors, i))
        after[f"after_{i}"] = x.copy()
        print(f"layer {i} done", flush=True)
    x, blocks = run_full(tensors, x, blocks, full_bundle(tensors, 5))
    after["after_5"] = x.copy()
    print("layer 5 done", flush=True)
    out = ROOT / "goldens/layer05_f32.npz"
    np.savez_compressed(out, x_in=g["x_in"].astype(np.float32), x_out=x, **after)
    meta = {
        "path": str(out.relative_to(ROOT)),
        "x_out_absmax": float(np.abs(x).max()),
        "x_out_norm": float(np.linalg.norm(x)),
        "bytes": out.stat().st_size,
    }
    (ROOT / "goldens/layer05_f32.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
