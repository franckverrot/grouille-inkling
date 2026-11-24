#!/usr/bin/env python3
"""Export a self-contained layer-0 MoE fixture for Rust parity tests.

Writes goldens/moe_fixture.npz with activations, routes, the f32 CPU reference
output, and packed NVFP4 tensors only for experts touched by the golden prompt.
"""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from parity_moe_layer0 import (
    DEFAULT_SNAP,
    ROOT,
    dequant_nvfp4_rows,
    softcapped_swiglu,
)

OUT = ROOT / "goldens" / "moe_fixture.npz"
META = ROOT / "goldens" / "moe_fixture.json"


def main() -> None:
    golden = np.load(ROOT / "goldens/layer0.npz")
    moe_in = golden["moe_in"][0].astype(np.float32)
    indices = golden["route_indices"][0].astype(np.int32)
    weights = golden["route_weights"][0].astype(np.float32)
    unique = np.array(sorted(set(int(i) for i in indices.reshape(-1))), dtype=np.int32)

    tensors = mx.load(str(DEFAULT_SNAP / "model-00001-of-00001.safetensors"))
    p = "model.llm.layers.0.mlp.experts."
    w13 = np.array(tensors[p + "w13_weight"])
    s13 = np.array(tensors[p + "w13_weight_scale"])
    a13 = np.array(tensors[p + "w13_weight_amax"], dtype=np.float32)
    w2 = np.array(tensors[p + "w2_weight"])
    s2 = np.array(tensors[p + "w2_weight_scale"])
    a2 = np.array(tensors[p + "w2_weight_amax"], dtype=np.float32)

    gate_w = np.stack([w13[e, 0::2, :] for e in unique], axis=0)
    up_w = np.stack([w13[e, 1::2, :] for e in unique], axis=0)
    gate_s = np.stack([s13[e, 0::2, :] for e in unique], axis=0)
    up_s = np.stack([s13[e, 1::2, :] for e in unique], axis=0)
    down_w = np.stack([w2[e] for e in unique], axis=0)
    down_s = np.stack([s2[e] for e in unique], axis=0)
    gate_up_amax = np.array([a13[e] for e in unique], dtype=np.float32)
    down_amax = np.array([a2[e] for e in unique], dtype=np.float32)

    # Map global expert id → row in the fixture stacks.
    remap = {int(e): i for i, e in enumerate(unique)}
    local_indices = np.vectorize(remap.__getitem__)(indices).astype(np.int32)

    T, H = moe_in.shape
    pred = np.zeros_like(moe_in)
    for t in range(T):
        acc = np.zeros(H, dtype=np.float32)
        for r in range(8):
            li = int(local_indices[t, r])
            w = float(weights[t, r])
            g = (
                dequant_nvfp4_rows(gate_w[li], gate_s[li], float(gate_up_amax[li]))
                @ moe_in[t]
            )
            u = (
                dequant_nvfp4_rows(up_w[li], up_s[li], float(gate_up_amax[li]))
                @ moe_in[t]
            )
            h = softcapped_swiglu(u, g)
            acc += w * (
                dequant_nvfp4_rows(down_w[li], down_s[li], float(down_amax[li])) @ h
            )
        pred[t] = acc

    np.savez_compressed(
        OUT,
        moe_in=moe_in,
        moe_out_f32=pred,
        route_weights=weights,
        local_indices=local_indices,
        expert_ids=unique,
        gate_w=gate_w,
        up_w=up_w,
        down_w=down_w,
        gate_s=gate_s,
        up_s=up_s,
        down_s=down_s,
        gate_up_amax=gate_up_amax,
        down_amax=down_amax,
    )
    meta = {
        "path": str(OUT.relative_to(ROOT)),
        "n_tokens": int(T),
        "hidden": int(H),
        "n_experts_fixture": int(len(unique)),
        "moe_out_f32_absmax": float(np.abs(pred).max()),
        "bytes": OUT.stat().st_size,
    }
    META.write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
