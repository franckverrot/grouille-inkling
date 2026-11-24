#!/usr/bin/env python3
"""Dump layer-6 second-pass (with messages) inputs/outputs for Rust parity."""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_vlm import load
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
    prelude = prelude_blocks = bank = None
    loop = llm.loop
    captured = None
    for visit, index in enumerate(llm.layer_indices):
        if index == loop.core_start:
            if visit == index:
                prelude, prelude_blocks = x, list(blocks)
            else:
                states = mx.stack(
                    [*blocks[len(prelude_blocks) + 1 :], x], axis=-2
                )
                bank = mx.fast.rms_norm(states.astype(mx.float32), None, 1e-6)
                x, blocks = prelude, list(prelude_blocks)
        messages = None
        if bank is not None and index < loop.core_end:
            query = llm.message_queries[index - loop.core_start]
            scores = (query @ bank.swapaxes(-1, -2)) * x.shape[-1] ** -0.5
            messages = mx.softmax(scores, axis=-1) @ bank
        if index == 6 and messages is not None and captured is None:
            x_in = mx.array(x)
            blocks_in = [mx.array(b) for b in blocks]
            msgs = mx.array(messages)
            x_out = llm.layers[6](x, blocks, cache[visit], messages)
            mx.eval(x_in, msgs, x_out, *blocks_in)
            captured = {
                "x_in": np.array(x_in.astype(mx.float32))[0],
                "x_out": np.array(x_out.astype(mx.float32))[0],
                "messages": np.array(msgs.astype(mx.float32))[0],  # [T,4,H]
                "n_blocks": len(blocks_in),
                "blocks": np.stack(
                    [np.array(b.astype(mx.float32))[0] for b in blocks_in], axis=1
                ),  # [T, n, H]
            }
            break
        x = llm.layers[index](x, blocks, cache[visit], messages)

    assert captured is not None
    out = ROOT / "goldens/layer6_msg.npz"
    np.savez_compressed(out, **captured)
    meta = {
        "path": str(out.relative_to(ROOT)),
        "tokens": int(captured["x_in"].shape[0]),
        "n_blocks": int(captured["n_blocks"]),
        "x_out_absmax": float(np.abs(captured["x_out"]).max()),
        "bytes": out.stat().st_size,
    }
    (ROOT / "goldens/layer6_msg.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
