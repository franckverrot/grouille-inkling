#!/usr/bin/env python3
"""Dump Inkling-2-Mini layer-0 decode goldens from mlx-vlm.

Loads the local HF checkpoint (~20 GB). Writes goldens/layer0.npz plus a
small JSON sidecar. Text-only; no vision / MTP.

Captures enough intermediates to parity the layer piecewise:
x_in → attn_out → residual_after_attn_sconv → moe_in → moe_out → x_out.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SNAP = (
    Path.home()
    / ".cache/huggingface/hub/models--thinkingmachines--Inkling-2-Mini"
    / "snapshots/ed40e7da625f594fc9c4990f5f66be24873c22d7"
)
PROMPT = "Hi"


def _to_numpy(x: mx.array) -> np.ndarray:
    return np.array(x.astype(mx.float32))


def main() -> None:
    snap = Path(DEFAULT_SNAP)
    if not (snap / "config.json").exists():
        raise SystemExit(f"missing checkpoint at {snap}")

    from mlx_vlm import load
    from mlx_vlm.models.inkling2.language import _cache_padding_mask, _route
    from mlx_vlm.prompt_utils import apply_chat_template

    print(f"loading {snap} …", flush=True)
    model, processor = load(str(snap))
    text_cfg = model.config.text_config
    print(
        f"loaded  hidden={text_cfg.hidden_size} layers={text_cfg.num_hidden_layers} "
        f"visits={len(text_cfg.layer_indices)}",
        flush=True,
    )

    prompt = apply_chat_template(
        processor,
        model.config,
        PROMPT,
        reasoning_effort="none",
    )
    if isinstance(prompt, str):
        ids = processor.tokenizer.encode(prompt)
    else:
        ids = prompt
    input_ids = mx.array([ids], dtype=mx.int32)
    print(f"prompt_tokens={input_ids.shape[1]}", flush=True)

    llm = model.language_model.model
    layer0 = llm.layers[0]
    taps: dict[str, mx.array] = {}

    class CaptureMoE(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def __call__(self, x):
            indices, weights = _route(
                x,
                self.inner.gate_weight,
                self.inner.e_score_correction_bias,
                self.inner.top_k,
                self.inner.route_scale,
            )
            out = self.inner(x)
            # Materialize copies now so later graph reuse cannot drift the golden.
            mx.eval(x, indices, weights, out)
            taps["moe_in"] = mx.array(x)
            taps["route_indices"] = mx.array(indices)
            taps["route_weights"] = mx.array(weights)
            taps["moe_out"] = mx.array(out)
            return out

    class CaptureAttn(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def __call__(self, *args, **kwargs):
            out = self.inner(*args, **kwargs)
            taps["attn_out"] = out
            return out

    layer0.self_attn = CaptureAttn(layer0.self_attn)
    layer0.mlp = CaptureMoE(layer0.mlp)

    cache = model.make_cache()
    x = llm.embed(input_ids)
    blocks: list = []
    taps["x_in"] = x

    # Manual mirror of DecoderLayer.__call__ for residual taps, then trust
    # the wrapped modules for attn/moe.
    conv = cache[0][1]
    mask = _cache_padding_mask(conv, x.shape[1])
    mixed = x[..., None, :]
    attn = layer0.self_attn(
        layer0.input_layernorm(mixed[..., 0, :]).astype(x.dtype),
        cache=cache[0],
        conv_mask=mask,
    )
    residual = layer0.attn_sconv(attn, conv, mask).astype(mx.float32)
    taps["residual_after_attn"] = residual
    blocks.append(x)
    hidden = layer0.mlp_input(blocks, residual)[..., 0, :]
    taps["mlp_input_mixed"] = hidden
    _ = layer0.mlp(layer0.post_attention_layernorm(hidden).astype(x.dtype))
    x_out = (
        residual + layer0.mlp_sconv(taps["moe_out"], conv, mask).astype(mx.float32)
    ).astype(x.dtype)
    if conv is not None:
        conv.advance(x.shape[1])
    taps["x_out"] = x_out

    mx.eval(*taps.values())

    out_dir = ROOT / "goldens"
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / "layer0.npz"
    meta_path = out_dir / "layer0.json"

    payload = {
        "x_in": _to_numpy(taps["x_in"]),
        "attn_out": _to_numpy(taps["attn_out"]),
        "residual_after_attn": _to_numpy(taps["residual_after_attn"]),
        "mlp_input_mixed": _to_numpy(taps["mlp_input_mixed"]),
        "moe_in": _to_numpy(taps["moe_in"]),
        "moe_out": _to_numpy(taps["moe_out"]),
        "x_out": _to_numpy(taps["x_out"]),
        "route_indices": _to_numpy(taps["route_indices"]).astype(np.int32),
        "route_weights": _to_numpy(taps["route_weights"]),
        "input_ids": np.array(ids, dtype=np.int32),
    }
    np.savez_compressed(npz_path, **payload)

    meta = {
        "checkpoint": f"thinkingmachines/Inkling-2-Mini@{snap.name}",
        "prompt": PROMPT,
        "n_tokens": int(input_ids.shape[1]),
        "layer_idx": 0,
        "is_sliding": bool(text_cfg.layer_is_sliding(0)),
        "shapes": {k: list(v.shape) for k, v in payload.items()},
        "x_out_stats": {
            "mean": float(payload["x_out"].mean()),
            "std": float(payload["x_out"].std()),
            "absmax": float(np.abs(payload["x_out"]).max()),
        },
        "route_indices_flat_unique": int(np.unique(payload["route_indices"]).size),
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))
    print(f"wrote {npz_path}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        raise
