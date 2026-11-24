#!/usr/bin/env python3
"""f32 CPU reference for full-attn layer 5 (isolated, synthetic blocks)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "spike"))
from export_layer0_f32 import route  # noqa: E402
from parity_full_attn import full_attn  # noqa: E402
from parity_layer0_pieces import (  # noqa: E402
    attention_residual,
    rms_norm,
    short_conv,
)
from parity_moe_layer0 import dequant_nvfp4_rows, softcapped_swiglu  # noqa: E402

SNAP = (
    Path.home()
    / ".cache/huggingface/hub/models--thinkingmachines--Inkling-2-Mini"
    / "snapshots/ed40e7da625f594fc9c4990f5f66be24873c22d7"
)
H = 2304
LAYER = 5


def bf(tensors, name):
    return np.array(tensors[name].astype(mx.float32))


def taps(tensors, name):
    t = bf(tensors, name)
    return t[:, 0, :]


def run_moe(tensors, moe_in, indices, weights):
    p = f"model.llm.layers.{LAYER}.mlp.experts."
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


def main():
    print("loading weights…", flush=True)
    tensors = mx.load(str(SNAP / "model-00001-of-00001.safetensors"))
    p = f"model.llm.layers.{LAYER}."
    attn_w = {
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
    }
    q_norm = bf(tensors, p + "q_input_norm.weight")
    k_norm = bf(tensors, p + "k_input_norm.weight")
    v_norm = bf(tensors, p + "v_input_norm.weight")
    mlp_norm = bf(tensors, p + "mlp_norm.weight")
    attn_sconv = taps(tensors, p + "attn_sconv.weight")
    mlp_sconv = taps(tensors, p + "mlp_sconv.weight")
    attn_queries = bf(tensors, "model.llm.attn_res.attn_queries.4")  # (3,H)
    mlp_query = bf(tensors, f"model.llm.attn_res.mlp_queries.{LAYER}")
    gate_w = bf(tensors, p + "mlp.gate.weight")
    gate_b = np.array(tensors[p + "mlp.gate.bias"], dtype=np.float32)

    rng = np.random.default_rng(5)
    t = 14
    # one prior block state (as if after boundary layer 0)
    block = rng.standard_normal((t, H), np.float32) * 0.05
    x_in = rng.standard_normal((t, H), np.float32) * 0.05

    mixed = attention_residual([block], x_in, attn_queries, True)  # [T,3,H]
    q_in = rms_norm(mixed[:, 0], q_norm)
    k_in = rms_norm(mixed[:, 1], k_norm)
    v_in = rms_norm(mixed[:, 2], v_norm)
    attn = full_attn(q_in, k_in, v_in, attn_w)
    residual = short_conv(attn, attn_sconv) + x_in  # not block boundary
    mlp_mixed = attention_residual([block], residual, mlp_query, True)
    moe_in = rms_norm(mlp_mixed, mlp_norm)
    indices, weights = route(moe_in, gate_w, gate_b)
    moe_out = run_moe(tensors, moe_in, indices, weights)
    x_out = residual + short_conv(moe_out, mlp_sconv)

    out_path = ROOT / "goldens/layer5_f32.npz"
    np.savez_compressed(
        out_path,
        block=block,
        x_in=x_in,
        q_in=q_in,
        k_in=k_in,
        v_in=v_in,
        attn_out=attn,
        residual=residual,
        moe_in=moe_in,
        moe_out=moe_out,
        x_out=x_out,
        route_indices=indices,
        route_weights=weights,
    )
    report = {
        "tokens": t,
        "x_out_norm": float(np.linalg.norm(x_out)),
        "path": str(out_path.relative_to(ROOT)),
    }
    print(json.dumps(report, indent=2))
    print("OK")


if __name__ == "__main__":
    main()
