#!/usr/bin/env python3
"""Piecewise layer-0 parity against goldens/layer0.npz (f32 CPU)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SNAP = (
    Path.home()
    / ".cache/huggingface/hub/models--thinkingmachines--Inkling-2-Mini"
    / "snapshots/ed40e7da625f594fc9c4990f5f66be24873c22d7"
)
EPS = 1e-6
K = 4
H = 2304
N_HEADS = 24
N_KV = 8
DH = 128
D_REL = 16
REL_EXTENT = 1024
SWA = 1024
QK_CLIP = 22.627416997969522
REL_CLIP = 8.0


def rms_norm(x: np.ndarray, weight: np.ndarray | None = None, eps: float = EPS) -> np.ndarray:
    x = x.astype(np.float32)
    var = np.mean(x * x, axis=-1, keepdims=True)
    y = x * np.reciprocal(np.sqrt(var + eps))
    if weight is not None:
        y = y * weight.astype(np.float32)
    return y


def short_conv(x: np.ndarray, taps: np.ndarray, state: np.ndarray | None = None) -> np.ndarray:
    """Depthwise causal conv L=4 + residual, /sqrt(5). taps: [C, K] float32."""
    # x: [T, C], taps from HF (C, 1, K) → use [C, K]
    t, c = x.shape
    assert taps.shape == (c, K)
    if state is None:
        state = np.zeros((K - 1, c), dtype=np.float32)
    xf = x.astype(np.float32)
    ctx = np.concatenate([state, xf], axis=0)  # [K-1+T, C]
    out = np.zeros_like(xf)
    for i in range(t):
        window = ctx[i : i + K]  # [K, C]
        # depthwise: out[i,c] = sum_k taps[c,k] * window[k,c]
        out[i] = np.sum(taps.T * window, axis=0)  # taps.T [K,C] * window
    return ((out + xf) / math.sqrt(K + 1)).astype(np.float32)


def attention_residual(
    blocks: list[np.ndarray], partial: np.ndarray, query: np.ndarray, normalize_values: bool
) -> np.ndarray:
    # each [T,H] → stack [T, n, H]
    states = np.stack([*blocks, partial], axis=-2).astype(np.float32)
    keys = rms_norm(states, None)
    # scores: [T, n, 1]
    q = query.astype(np.float32).reshape(-1, H)  # [1, H] or [3,H]
    scale = H**-0.5
    scores = np.einsum("...nd,qd->...nq", keys, q) * scale
    weights = _softmax(scores, axis=-2)  # over n
    values = keys if normalize_values else states
    # weights [T,n,Q] → [T,Q,n]; values [T,n,H]
    out = np.einsum("...nq,...nd->...qd", weights, values)
    return out[..., 0, :] if out.shape[-2] == 1 else out


def _softmax(x, axis):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def l2_clip(x: np.ndarray, threshold: float, scale: np.ndarray | None = None) -> np.ndarray:
    x = x.astype(np.float32)
    if scale is not None:
        x = x * scale.astype(np.float32)
    norm = np.sqrt(np.sum(x * x, axis=-1, keepdims=True))
    return x * np.minimum(threshold / (norm + 1e-6), 1.0)


def banded_mask(rel: np.ndarray, proj: np.ndarray, t: int) -> np.ndarray:
    """rel [T, H, D_REL], proj [D_REL, REL_EXTENT] → mask [H, T, T] additive."""
    mask = np.full((N_HEADS, t, t), -1e30, dtype=np.float32)
    for i in range(t):
        for j in range(t):
            dist = i - j
            if dist < 0:
                continue
            if SWA > 0 and dist >= SWA:
                continue
            if dist < REL_EXTENT:
                mask[:, i, j] = rel[i] @ proj[:, dist]
            else:
                mask[:, i, j] = 0.0
    return mask


def sliding_attn(
    x: np.ndarray,
    w: dict[str, np.ndarray],
) -> np.ndarray:
    """Layer-0 sliding attention, no KV cache (prefill). x [T,H]."""
    t = x.shape[0]
    # projections: HF names; weights are [out, in]
    q = (x @ w["wq"].T).reshape(t, N_HEADS, DH)
    k = x @ w["wk"].T  # [T, N_KV*DH]
    v = x @ w["wv"].T
    # k/v short conv on channels N_KV*DH
    k = short_conv(k, w["k_sconv"])
    v = short_conv(v, w["v_sconv"])
    q = l2_clip(q, QK_CLIP, w["q_scale"])
    k = l2_clip(k.reshape(t, N_KV, DH), QK_CLIP)
    v = v.reshape(t, N_KV, DH)
    v = rms_norm(v, None)  # value norm no gamma
    r = l2_clip((x @ w["wr"].T).reshape(t, N_HEADS, D_REL), REL_CLIP, w["rel_scale"])
    # [H,T,D]
    qh = q.transpose(1, 0, 2)
    kh = k.transpose(1, 0, 2)
    vh = v.transpose(1, 0, 2)
    # GQA: repeat kv
    rep = N_HEADS // N_KV
    kh = np.repeat(kh, rep, axis=0)
    vh = np.repeat(vh, rep, axis=0)
    mask = banded_mask(r, w["rel_proj"], t)  # [H,T,T]
    sinks = (x @ w["sink"].T).T  # [H, T]
    # append sink column
    mask_s = np.concatenate([mask, sinks[:, :, None]], axis=-1)  # [H,T,T+1]
    kh_s = np.concatenate([kh, np.zeros((N_HEADS, 1, DH), np.float32)], axis=1)
    vh_s = np.concatenate([vh, np.zeros((N_HEADS, 1, DH), np.float32)], axis=1)
    scale = DH**-0.5
    scores = np.einsum("htd,hsd->hts", qh, kh_s) * scale + mask_s
    attn = _softmax(scores, axis=-1)
    out = np.einsum("hts,hsd->htd", attn, vh_s)  # [H,T,D]
    out = out.transpose(1, 0, 2).reshape(t, N_HEADS * DH)
    gate = 1.0 / (1.0 + np.exp(-(x @ w["gate"].T)))
    out = out * gate
    return out @ w["wo"].T


def report(name: str, pred: np.ndarray, ref: np.ndarray) -> dict:
    err = np.abs(pred.astype(np.float32) - ref.astype(np.float32))
    return {
        "name": name,
        "maxabs": float(err.max()),
        "meanabs": float(err.mean()),
        "pred_absmax": float(np.abs(pred).max()),
        "ref_absmax": float(np.abs(ref).max()),
    }


def load_weights():
    tensors = mx.load(str(SNAP / "model-00001-of-00001.safetensors"))

    def bf(name):
        return np.array(tensors[name].astype(mx.float32))

    def f32(name):
        return np.array(tensors[name].astype(mx.float32))

    p = "model.llm.layers.0."
    # conv taps HF (C,1,K) → (C,K)
    def taps(name):
        t = bf(name)
        assert t.shape[1] == 1
        return t[:, 0, :]

    attn = {
        "wq": bf(p + "attn.wq_du.weight"),
        "wk": bf(p + "attn.wk_dv.weight"),
        "wv": bf(p + "attn.wv_dv.weight"),
        "wr": bf(p + "attn.wr_du.weight"),
        "wo": bf(p + "attn.wo_ud.weight"),
        "gate": bf(p + "attn.gate_attn.weight"),
        "sink": bf(p + "attn.attn_sink_proj.weight"),
        "q_scale": bf(p + "attn.q_l2_clip.scale"),
        "rel_scale": bf(p + "attn.rel_l2_clip.scale"),
        "rel_proj": bf(p + "attn.rel_logits_proj.proj"),
        "k_sconv": taps(p + "attn.k_sconv.weight"),
        "v_sconv": taps(p + "attn.v_sconv.weight"),
    }
    return {
        "attn": attn,
        "attn_norm": bf(p + "attn_norm.weight"),
        "mlp_norm": bf(p + "mlp_norm.weight"),
        "attn_sconv": taps(p + "attn_sconv.weight"),
        "mlp_sconv": taps(p + "mlp_sconv.weight"),
        "mlp_query": f32("model.llm.attn_res.mlp_queries.0"),
    }


def main():
    g = np.load(ROOT / "goldens/layer0.npz")
    w = load_weights()
    x_in = g["x_in"][0].astype(np.float32)
    results = []

    # 1) attn from x_in
    x_norm = rms_norm(x_in, w["attn_norm"])
    attn_pred = sliding_attn(x_norm, w["attn"])
    # compare in bf16 space roughly
    attn_ref = g["attn_out"][0].astype(np.float32)
    results.append(report("attn_out", attn_pred, attn_ref))

    # 2) short conv residual from golden attn_out
    res_pred = short_conv(attn_ref, w["attn_sconv"])
    # layer0 block boundary: residual is conv(attn) WITHOUT adding x
    results.append(report("residual_after_attn", res_pred, g["residual_after_attn"][0]))

    # 3) mlp_input mixer
    mixed = attention_residual(
        [x_in], res_pred if False else g["residual_after_attn"][0].astype(np.float32),
        w["mlp_query"],
        True,
    )
    # wait: mlp_input uses residual from the layer (after attn_sconv). Use golden residual.
    mixed = attention_residual(
        [x_in],
        g["residual_after_attn"][0].astype(np.float32),
        w["mlp_query"],
        True,
    )
    results.append(report("mlp_input_mixed", mixed, g["mlp_input_mixed"][0]))

    # 4) moe_in = rms_norm(mixed)
    moe_in = rms_norm(mixed, w["mlp_norm"])
    results.append(report("moe_in", moe_in, g["moe_in"][0]))

    # 5) x_out = residual + mlp_sconv(moe_out)  (block boundary: no +x on attn residual path)
    moe_out = g["moe_out"][0].astype(np.float32)
    # golden moe_out is bf16-rounded; use f32 ref if present
    if "moe_out_f32" in g.files:
        moe_out = g["moe_out_f32"].astype(np.float32)
    mlp_c = short_conv(moe_out, w["mlp_sconv"])
    x_out = g["residual_after_attn"][0].astype(np.float32) + mlp_c
    results.append(report("x_out_from_golden_parts", x_out, g["x_out"][0]))

    print(json.dumps(results, indent=2))
    bad = [r for r in results if r["maxabs"] > 0.1]
    if bad:
        raise SystemExit("piece parity failed: " + ", ".join(b["name"] for b in bad))
    print("OK")


if __name__ == "__main__":
    main()
