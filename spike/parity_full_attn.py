#!/usr/bin/env python3
"""f32 CPU full attention (layer 5) vs MLX bf16."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_vlm import load

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "spike"))
from parity_dyn_sconv import dyn_sconv  # noqa: E402
from parity_layer0_pieces import l2_clip, rms_norm, _softmax  # noqa: E402

SNAP = (
    Path.home()
    / ".cache/huggingface/hub/models--thinkingmachines--Inkling-2-Mini"
    / "snapshots/ed40e7da625f594fc9c4990f5f66be24873c22d7"
)
H = 2304
N_HEADS = 24
N_KV = 8
DH = 128
D_REL = 16
REL_EXTENT = 1024
QK_CLIP = 22.627416997969522
REL_CLIP = 8.0
LOG_N = 8192.0
LOG_A = 0.1


def full_attn(q_in, k_in, v_in, w):
    """q/k/v_in: [T,H] already normed. Returns [T,H] f32."""
    t = q_in.shape[0]
    q = (q_in @ w["wq"].T).reshape(t, N_HEADS, DH)
    k = k_in @ w["wk"].T
    v = v_in @ w["wv"].T
    k = dyn_sconv(k, k_in, w["k_static"], w["k_wproj"])
    v = dyn_sconv(v, v_in, w["v_static"], w["v_wproj"])
    q = l2_clip(q, QK_CLIP, w["q_scale"])
    k = l2_clip(k.reshape(t, N_KV, DH), QK_CLIP)
    v = rms_norm(v.reshape(t, N_KV, DH), None)
    r = l2_clip((q_in @ w["wr"].T).reshape(t, N_HEADS, D_REL), REL_CLIP, w["rel_scale"])
    # log scaling
    pos = np.arange(1, t + 1, dtype=np.float32)
    tau = 1.0 + LOG_A * np.log(np.maximum(pos / LOG_N, 1.0))
    q = q * tau[:, None, None]
    r = r * tau[:, None, None]

    qh = q.transpose(1, 0, 2)
    kh = np.repeat(k.transpose(1, 0, 2), N_HEADS // N_KV, axis=0)
    vh = np.repeat(v.transpose(1, 0, 2), N_HEADS // N_KV, axis=0)
    mask = np.full((N_HEADS, t, t), -1e30, np.float32)
    for i in range(t):
        for j in range(t):
            dist = i - j
            if dist < 0:
                continue
            if dist < REL_EXTENT:
                mask[:, i, j] = r[i] @ w["rel_proj"][:, dist]
            else:
                mask[:, i, j] = 0.0
    sinks = (q_in @ w["sink"].T).T  # [H,T]
    mask_s = np.concatenate([mask, sinks[:, :, None]], axis=-1)
    kh_s = np.concatenate([kh, np.zeros((N_HEADS, 1, DH), np.float32)], 1)
    vh_s = np.concatenate([vh, np.zeros((N_HEADS, 1, DH), np.float32)], 1)
    scores = np.einsum("htd,hsd->hts", qh, kh_s) * (DH**-0.5) + mask_s
    attn = _softmax(scores, axis=-1)
    out = np.einsum("hts,hsd->htd", attn, vh_s).transpose(1, 0, 2).reshape(t, N_HEADS * DH)
    gate = 1.0 / (1.0 + np.exp(-(q_in @ w["gate"].T)))
    return ((out * gate) @ w["wo"].T).astype(np.float32)


def main():
    print("loading…", flush=True)
    model, _ = load(str(SNAP))
    layer = model.language_model.model.layers[5]
    attn = layer.self_attn
    rng = np.random.default_rng(1)
    t = 14
    q_in = rng.standard_normal((1, t, H), np.float32) * 0.05
    k_in = rng.standard_normal((1, t, H), np.float32) * 0.05
    v_in = rng.standard_normal((1, t, H), np.float32) * 0.05

    ref = attn(
        mx.array(q_in).astype(mx.bfloat16),
        mx.array(k_in).astype(mx.bfloat16),
        mx.array(v_in).astype(mx.bfloat16),
    )
    mx.eval(ref)
    ref = np.array(ref.astype(mx.float32))[0]

    def bf(a):
        return np.array(a.astype(mx.float32))

    w = {
        "wq": bf(attn.q_proj.weight),
        "wk": bf(attn.k_proj.weight),
        "wv": bf(attn.v_proj.weight),
        "wr": bf(attn.r_proj.weight),
        "wo": bf(attn.o_proj.weight),
        "gate": bf(attn.gate_proj.weight),
        "sink": bf(attn.sink_proj.weight),
        "q_scale": bf(attn.q_scale),
        "rel_scale": bf(attn.rel_scale),
        "rel_proj": bf(attn.rel_proj),
        "k_static": bf(attn.k_sconv.static_weight),
        "k_wproj": bf(attn.k_sconv.weight_proj.weight),
        "v_static": bf(attn.v_sconv.static_weight),
        "v_wproj": bf(attn.v_sconv.weight_proj.weight),
    }
    pred = full_attn(q_in[0], k_in[0], v_in[0], w)
    pred_bf = np.array(mx.array(pred).astype(mx.bfloat16).astype(mx.float32))
    err = np.abs(pred_bf - ref)
    report = {"maxabs": float(err.max()), "meanabs": float(err.mean())}
    print(json.dumps(report, indent=2))
    np.savez_compressed(
        ROOT / "goldens/full_attn.npz",
        q_in=q_in[0],
        k_in=k_in[0],
        v_in=v_in[0],
        out_f32=pred,
        out_bf16=ref,
        **w,
    )
    if report["maxabs"] > 0.08:
        raise SystemExit("full attn parity failed")
    print("OK")


if __name__ == "__main__":
    main()
