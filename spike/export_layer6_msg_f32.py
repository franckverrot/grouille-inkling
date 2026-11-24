#!/usr/bin/env python3
"""f32 CPU reference for layer-6 second pass (with messages)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "spike"))
from export_layer0_f32 import route  # noqa: E402
from export_layer01_f32 import layer_bundle, load_tensors, run_moe  # noqa: E402
from parity_layer0_pieces import (  # noqa: E402
    attention_residual,
    rms_norm,
    short_conv,
    sliding_attn,
)

H = 2304


def bf(tensors, name):
    return np.array(tensors[name].astype(mx.float32))


def ctrl(tensors, layer, name):
    p = f"model.llm.layers.{layer}.controllers.{name}."
    return {
        "down": bf(tensors, p + "down_weight"),
        "forget": bf(tensors, p + "forget_weight"),
        "gain": bf(tensors, p + "gain_weight"),
    }


def message_controller(x, message, w):
    xn = rms_norm(x, None)
    mn = rms_norm(message, None)
    pair = np.concatenate([xn, mn], axis=-1)
    hidden = pair @ w["down"].T
    forget = np.tanh(hidden @ w["forget"].T)
    gain = np.tanh(hidden @ w["gain"].T)
    return (1.0 - forget) * x + gain * message


def sliding_attn_qkv(q_in, k_in, v_in, w):
    """Like sliding_attn but Q from q_in, K/V from k_in/v_in."""
    from parity_layer0_pieces import (  # local import of helpers
        DH,
        D_REL,
        N_HEADS,
        N_KV,
        QK_CLIP,
        REL_CLIP,
        REL_EXTENT,
        SWA,
        _softmax,
        l2_clip,
    )

    t = q_in.shape[0]
    q = (q_in @ w["wq"].T).reshape(t, N_HEADS, DH)
    k = short_conv(k_in @ w["wk"].T, w["k_sconv"])
    v = short_conv(v_in @ w["wv"].T, w["v_sconv"])
    q = l2_clip(q, QK_CLIP, w["q_scale"])
    k = l2_clip(k.reshape(t, N_KV, DH), QK_CLIP)
    v = rms_norm(v.reshape(t, N_KV, DH), None)
    r = l2_clip((q_in @ w["wr"].T).reshape(t, N_HEADS, D_REL), REL_CLIP, w["rel_scale"])
    qh = q.transpose(1, 0, 2)
    kh = np.repeat(k.transpose(1, 0, 2), N_HEADS // N_KV, 0)
    vh = np.repeat(v.transpose(1, 0, 2), N_HEADS // N_KV, 0)
    mask = np.full((N_HEADS, t, t), -1e30, np.float32)
    for i in range(t):
        for j in range(t):
            dist = i - j
            if dist < 0 or (SWA > 0 and dist >= SWA):
                continue
            if dist < REL_EXTENT:
                mask[:, i, j] = r[i] @ w["rel_proj"][:, dist]
            else:
                mask[:, i, j] = 0.0
    sinks = (q_in @ w["sink"].T).T
    mask_s = np.concatenate([mask, sinks[:, :, None]], -1)
    kh_s = np.concatenate([kh, np.zeros((N_HEADS, 1, DH), np.float32)], 1)
    vh_s = np.concatenate([vh, np.zeros((N_HEADS, 1, DH), np.float32)], 1)
    scores = np.einsum("htd,hsd->hts", qh, kh_s) * (DH**-0.5) + mask_s
    attn = _softmax(scores, -1)
    out = np.einsum("hts,hsd->htd", attn, vh_s).transpose(1, 0, 2).reshape(t, N_HEADS * DH)
    gate = 1.0 / (1.0 + np.exp(-(q_in @ w["gate"].T)))
    return (out * gate) @ w["wo"].T


def main():
    g = np.load(ROOT / "goldens/layer6_msg.npz")
    x = g["x_in"].astype(np.float32)
    messages = g["messages"].astype(np.float32)  # [T,4,H]
    blocks_arr = g["blocks"].astype(np.float32)  # [T,n,H]
    t = x.shape[0]
    blocks = [blocks_arr[:, i, :] for i in range(blocks_arr.shape[1])]

    tensors = load_tensors()
    w = layer_bundle(tensors, 6)
    ca, ck, cv, cm = (ctrl(tensors, 6, n) for n in ("attn", "k", "v", "mlp"))

    mixed = attention_residual(blocks, x, w["attn_query"], True)  # [T,H]
    q_raw = message_controller(mixed, messages[:, 0], ca)
    k_raw = message_controller(mixed, messages[:, 1], ck)
    v_raw = message_controller(mixed, messages[:, 2], cv)
    q_in = rms_norm(q_raw, w["attn_norm"])
    k_in = rms_norm(k_raw, w["attn_norm"])
    v_in = rms_norm(v_raw, w["attn_norm"])
    attn = sliding_attn_qkv(q_in, k_in, v_in, w["attn"])
    residual = short_conv(attn, w["attn_sconv"])
    # layer 6 is a block boundary, so no residual add of x
    assert w["boundary"]
    mlp_blocks = blocks + [x]
    mlp_mixed = attention_residual(mlp_blocks, residual, w["mlp_query"], True)
    mlp_raw = message_controller(mlp_mixed, messages[:, 3], cm)
    moe_in = rms_norm(mlp_raw, w["mlp_norm"])
    idx, wt = route(moe_in, w["gate_w"], w["gate_b"])
    moe_out = run_moe(tensors, 6, moe_in, idx, wt)
    x_out = residual + short_conv(moe_out, w["mlp_sconv"])

    out = ROOT / "goldens/layer6_msg_f32.npz"
    np.savez_compressed(
        out,
        x_in=x,
        messages=messages,
        blocks=blocks_arr,
        x_out=x_out,
        vs_mlx=g["x_out"].astype(np.float32),
    )
    err = np.abs(x_out - g["x_out"].astype(np.float32))
    meta = {
        "path": str(out.relative_to(ROOT)),
        "vs_mlx_maxabs": float(err.max()),
        "vs_mlx_meanabs": float(err.mean()),
        "x_out_norm": float(np.linalg.norm(x_out)),
    }
    print(json.dumps(meta, indent=2))
    if meta["vs_mlx_maxabs"] > 0.5:
        raise SystemExit("layer6 f32 vs mlx drift too high")
    print("OK")


if __name__ == "__main__":
    main()
