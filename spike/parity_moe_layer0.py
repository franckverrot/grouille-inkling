#!/usr/bin/env python3
"""CPU MoE parity for layer 0 against goldens/layer0.npz.

Reimplements NVFP4 (E2M1 × E4M3, lo-nibble first) + softcapped SwiGLU +
top-8 combine. Must match mlx-vlm's moe_out before we trust the Rust port.
"""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SNAP = (
    Path.home()
    / ".cache/huggingface/hub/models--thinkingmachines--Inkling-2-Mini"
    / "snapshots/ed40e7da625f594fc9c4990f5f66be24873c22d7"
)

E2M1 = np.array(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=np.float32,
)
NVFP4_MAX = 2688.0  # 6 * 448


def e4m3_from_bits(bits: np.ndarray) -> np.ndarray:
    b = bits.astype(np.uint8)
    sign = (b >> 7) & 1
    exp = ((b >> 3) & 0xF).astype(np.float32)
    mant = (b & 0x7).astype(np.float32)
    normal = exp != 0
    out = np.where(
        normal,
        (1.0 + mant / 8.0) * (2.0 ** (exp - 7.0)),
        (mant / 8.0) * (2.0 ** (1.0 - 7.0)),
    )
    return np.where(sign.astype(bool), -out, out).astype(np.float32)


def dequant_nvfp4_rows(packed_u8: np.ndarray, scales_u8: np.ndarray, amax: float) -> np.ndarray:
    """packed [out, in/2], scales [out, in/16] → float32 [out, in]."""
    out_f, packed_k = packed_u8.shape
    in_f = packed_k * 2
    assert scales_u8.shape == (out_f, in_f // 16)
    lo = packed_u8 & 0x0F
    hi = packed_u8 >> 4
    nib = np.empty((out_f, in_f), dtype=np.uint8)
    nib[:, 0::2] = lo
    nib[:, 1::2] = hi
    sc = np.repeat(e4m3_from_bits(scales_u8), 16, axis=1)
    return E2M1[nib] * sc * (amax / NVFP4_MAX)


def softcapped_swiglu(up, gate, gate_cap=4.0, up_cap=25.0):
    gate = gate.astype(np.float32)
    up = up.astype(np.float32)
    return (
        np.tanh(gate / gate_cap)
        * gate_cap
        * (1.0 / (1.0 + np.exp(-gate)))
        * np.tanh(up / up_cap)
        * up_cap
    )


def main() -> None:
    golden = np.load(ROOT / "goldens/layer0.npz")
    moe_in = golden["moe_in"][0]  # [T, H]
    moe_out = golden["moe_out"][0]
    indices = golden["route_indices"][0]  # [T, 8]
    weights = golden["route_weights"][0]

    snap = DEFAULT_SNAP
    tensors = mx.load(str(snap / "model-00001-of-00001.safetensors"))
    prefix = "model.llm.layers.0.mlp.experts."
    w13 = np.array(tensors[prefix + "w13_weight"])  # [E, 2304, 1152] u8
    s13 = np.array(tensors[prefix + "w13_weight_scale"])  # [E, 2304, 144] u8
    a13 = np.array(tensors[prefix + "w13_weight_amax"], dtype=np.float32)
    w2 = np.array(tensors[prefix + "w2_weight"])  # [E, 2304, 576]
    s2 = np.array(tensors[prefix + "w2_weight_scale"])  # [E, 2304, 72]
    a2 = np.array(tensors[prefix + "w2_weight_amax"], dtype=np.float32)

    # _split_gate_up: reshape (E, 2304, K) → (E, 1152, 2, K); gate=[...,0,:], up=[...,1,:].
    gate_w, up_w = w13[:, 0::2, :], w13[:, 1::2, :]
    gate_s, up_s = s13[:, 0::2, :], s13[:, 1::2, :]

    T, H = moe_in.shape
    assert H == 2304
    pred = np.zeros_like(moe_out)
    unique = sorted(set(int(i) for i in indices.reshape(-1)))
    print(f"tokens={T} unique_experts={len(unique)}", flush=True)

    cache_w = {}
    for e in unique:
        cache_w[e] = {
            "gate": dequant_nvfp4_rows(gate_w[e], gate_s[e], float(a13[e])),
            "up": dequant_nvfp4_rows(up_w[e], up_s[e], float(a13[e])),
            "down": dequant_nvfp4_rows(w2[e], s2[e], float(a2[e])),
        }

    for t in range(T):
        x = moe_in[t]
        acc = np.zeros(H, dtype=np.float32)
        for r in range(8):
            e = int(indices[t, r])
            w = float(weights[t, r])
            g = cache_w[e]["gate"] @ x  # [1152]
            u = cache_w[e]["up"] @ x
            h = softcapped_swiglu(u, g)
            acc += w * (cache_w[e]["down"] @ h)
        pred[t] = acc

    # Golden MoE ran in bf16; compare after rounding the f32 accumulation.
    pred_bf = np.array(mx.array(pred).astype(mx.bfloat16).astype(mx.float32))
    err = np.abs(pred_bf - moe_out)
    report = {
        "maxabs": float(err.max()),
        "meanabs": float(err.mean()),
        "rel_rms": float(
            np.sqrt((err**2).mean()) / (np.sqrt((moe_out**2).mean()) + 1e-12)
        ),
        "pred_absmax": float(np.abs(pred_bf).max()),
        "ref_absmax": float(np.abs(moe_out).max()),
    }
    print(json.dumps(report, indent=2))
    # One bf16 ulp near |x|≈16 is 0.0625; MLX accumulates experts in bf16.
    if report["maxabs"] > 0.07:
        raise SystemExit("MoE parity failed")
    print("OK", flush=True)


if __name__ == "__main__":
    main()
