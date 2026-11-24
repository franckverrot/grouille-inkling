#!/usr/bin/env python3
"""f32 CPU reference for layer-7 second pass (with messages)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "spike"))
from export_layer0_f32 import route  # noqa: E402
from export_layer01_f32 import layer_bundle, load_tensors, run_moe  # noqa: E402
from export_layer6_msg_f32 import ctrl, message_controller, sliding_attn_qkv  # noqa: E402
from parity_layer0_pieces import attention_residual, rms_norm, short_conv  # noqa: E402


def main():
    g = np.load(ROOT / "goldens/layer7_msg.npz")
    x = g["x_in"].astype(np.float32)
    messages = g["messages"].astype(np.float32)
    blocks_arr = g["blocks"].astype(np.float32)
    blocks = [blocks_arr[:, i, :] for i in range(blocks_arr.shape[1])]

    tensors = load_tensors()
    w = layer_bundle(tensors, 7)
    ca, ck, cv, cm = (ctrl(tensors, 7, n) for n in ("attn", "k", "v", "mlp"))

    mixed = attention_residual(blocks, x, w["attn_query"], True)
    q_in = rms_norm(message_controller(mixed, messages[:, 0], ca), w["attn_norm"])
    k_in = rms_norm(message_controller(mixed, messages[:, 1], ck), w["attn_norm"])
    v_in = rms_norm(message_controller(mixed, messages[:, 2], cv), w["attn_norm"])
    attn = sliding_attn_qkv(q_in, k_in, v_in, w["attn"])
    residual = short_conv(attn, w["attn_sconv"]) + x
    mlp_mixed = attention_residual(blocks, residual, w["mlp_query"], True)
    moe_in = rms_norm(message_controller(mlp_mixed, messages[:, 3], cm), w["mlp_norm"])
    idx, wt = route(moe_in, w["gate_w"], w["gate_b"])
    moe_out = run_moe(tensors, 7, moe_in, idx, wt)
    x_out = residual + short_conv(moe_out, w["mlp_sconv"])

    out = ROOT / "goldens/layer7_msg_f32.npz"
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
        "x_out_norm": float(np.linalg.norm(x_out)),
    }
    print(json.dumps(meta, indent=2))
    if meta["vs_mlx_maxabs"] > 0.5:
        raise SystemExit("layer7 f32 vs mlx drift too high")
    print("OK")


if __name__ == "__main__":
    main()
