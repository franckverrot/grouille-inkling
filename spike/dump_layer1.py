#!/usr/bin/env python3
"""Dump layer-1 taps + f32 CPU reference for residual-add path."""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_vlm import load
from mlx_vlm.models.inkling2.language import _cache_padding_mask
from mlx_vlm.prompt_utils import apply_chat_template

ROOT = Path(__file__).resolve().parents[1]
SNAP = (
    Path.home()
    / ".cache/huggingface/hub/models--thinkingmachines--Inkling-2-Mini"
    / "snapshots/ed40e7da625f594fc9c4990f5f66be24873c22d7"
)


def main() -> None:
    print("loading…", flush=True)
    model, processor = load(str(SNAP))
    llm = model.language_model.model
    prompt = apply_chat_template(
        processor, model.config, "Hi", reasoning_effort="none"
    )
    ids = processor.tokenizer.encode(prompt) if isinstance(prompt, str) else prompt
    input_ids = mx.array([ids], dtype=mx.int32)
    cache = model.make_cache()
    x = llm.embed(input_ids)
    blocks = []
    # layer 0
    x0_in = x
    x = llm.layers[0](x, blocks, cache[0], None)
    mx.eval(x, x0_in)
    x1_in = mx.array(x)
    layer = llm.layers[1]
    taps = {}

    class CapAttn:
        def __init__(self, inner):
            self.inner = inner

        def __call__(self, *a, **k):
            out = self.inner(*a, **k)
            taps["attn_out"] = out
            return out

    class CapMoe:
        def __init__(self, inner):
            self.inner = inner

        def __call__(self, x):
            taps["moe_in"] = x
            out = self.inner(x)
            taps["moe_out"] = out
            return out

    layer.self_attn = CapAttn(layer.self_attn)
    layer.mlp = CapMoe(layer.mlp)

    conv = cache[1][1]
    mask = _cache_padding_mask(conv, x1_in.shape[1])
    mixed = layer.attention_input(blocks, x1_in)
    attn = layer.self_attn(
        layer.input_layernorm(mixed[..., 0, :]).astype(x1_in.dtype),
        cache=cache[1],
        conv_mask=mask,
    )
    residual = layer.attn_sconv(attn, conv, mask).astype(mx.float32)
    residual = residual + x1_in.astype(mx.float32)
    taps["residual"] = residual
    hidden = layer.mlp_input(blocks, residual)[..., 0, :]
    taps["mlp_mixed"] = hidden
    _ = layer.mlp(layer.post_attention_layernorm(hidden).astype(x1_in.dtype))
    x1_out = (
        residual + layer.mlp_sconv(taps["moe_out"], conv, mask).astype(mx.float32)
    ).astype(x1_in.dtype)
    if conv is not None:
        conv.advance(x1_in.shape[1])
    taps["x_out"] = x1_out
    mx.eval(*taps.values(), x1_in)

    def f32(a):
        return np.array(a.astype(mx.float32))

    out = ROOT / "goldens/layer1.npz"
    np.savez_compressed(
        out,
        x0_in=f32(x0_in)[0],
        x1_in=f32(x1_in)[0],
        attn_out=f32(taps["attn_out"])[0],
        residual=f32(taps["residual"])[0],
        mlp_mixed=f32(taps["mlp_mixed"])[0],
        moe_in=f32(taps["moe_in"])[0],
        moe_out=f32(taps["moe_out"])[0],
        x_out=f32(taps["x_out"])[0],
        attn_query=f32(layer.attention_input.query),
        mlp_query=f32(layer.mlp_input.query),
    )
    meta = {
        "path": str(out.relative_to(ROOT)),
        "n_tokens": int(x1_in.shape[1]),
        "x_out_absmax": float(np.abs(f32(taps["x_out"])).max()),
        "bytes": out.stat().st_size,
    }
    (ROOT / "goldens/layer1.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
