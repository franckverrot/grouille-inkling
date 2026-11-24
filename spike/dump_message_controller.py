#!/usr/bin/env python3
"""Unit golden for MessageController (layer 6, attn slot) + message gather."""

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
    layer = llm.layers[6]
    ctrl = layer.controllers["attn"]

    prompt = apply_chat_template(
        processor, model.config, "Hi", reasoning_effort="none"
    )
    ids = (
        processor.tokenizer.encode(prompt)
        if isinstance(prompt, str)
        else prompt
    )
    input_ids = mx.array([ids], dtype=mx.int32)
    cache = model.make_cache()

    # Run through visits until we finish first core pass (visit where index
    # returns to 6 for pass 2); capture bank + messages at second entry to 6.
    x = llm.embed(input_ids)
    blocks = []
    prelude = prelude_blocks = bank = None
    captured_raw = None
    captured_prelude_n = 0
    loop = llm.loop
    captured = {}
    for visit, index in enumerate(llm.layer_indices):
        if index == loop.core_start:
            if visit == index:
                prelude, prelude_blocks = x, blocks.copy()
            else:
                raw_states = mx.stack(
                    [*blocks[len(prelude_blocks) + 1 :], x], axis=-2
                )
                bank = mx.fast.rms_norm(raw_states.astype(mx.float32), None, 1e-6)
                captured_prelude_n = len(prelude_blocks)
                captured_raw = raw_states
                x, blocks = prelude, prelude_blocks.copy()
        messages = None
        if bank is not None and index < loop.core_end:
            query = llm.message_queries[index - loop.core_start]
            scores = (query @ bank.swapaxes(-1, -2)) * x.shape[-1] ** -0.5
            messages = mx.softmax(scores, axis=-1) @ bank
        if index == 6 and messages is not None and "ctrl" not in captured:
            # First token of second pass through layer 6
            mixed = layer.attention_input(blocks, x)
            part = mixed[..., 0, :]
            msg = messages[..., 0, :]
            out = ctrl(part, msg)
            mx.eval(part, msg, out, bank, messages, captured_raw)
            captured["ctrl"] = {
                "x": np.array(part.astype(mx.float32))[0, 0],
                "message": np.array(msg.astype(mx.float32))[0, 0],
                "out": np.array(out.astype(mx.float32))[0, 0],
                "bank": np.array(bank.astype(mx.float32))[0, 0],  # [n,H] first batch
                "raw_states": np.array(captured_raw.astype(mx.float32))[0, 0],
                "prelude_n_blocks": np.array([captured_prelude_n], np.int32),
                "messages": np.array(messages.astype(mx.float32))[0, 0],  # [4,H]
                "queries": np.array(
                    llm.message_queries[0].astype(mx.float32)
                ),  # layer6 → index 0 in message_queries
            }
            break
        x = layer if False else llm.layers[index](
            x, blocks, cache[visit], messages
        )

    assert captured, "did not reach second pass of layer 6"
    c = captured["ctrl"]
    # weights
    tensors = mx.load(str(SNAP / "model-00001-of-00001.safetensors"))
    pref = "model.llm.layers.6.controllers.attn."
    w = {
        "down": np.array(tensors[pref + "down_weight"].astype(mx.float32)),
        "forget": np.array(tensors[pref + "forget_weight"].astype(mx.float32)),
        "gain": np.array(tensors[pref + "gain_weight"].astype(mx.float32)),
    }

    out_path = ROOT / "goldens/message_controller.npz"
    np.savez_compressed(
        out_path,
        x=c["x"],
        message=c["message"],
        out=c["out"],
        bank=c["bank"],
        raw_states=c["raw_states"],
        prelude_n_blocks=c["prelude_n_blocks"],
        messages=c["messages"],
        queries=c["queries"],
        down=w["down"],
        forget=w["forget"],
        gain=w["gain"],
    )
    meta = {
        "path": str(out_path.relative_to(ROOT)),
        "bank_shape": list(c["bank"].shape),
        "raw_states_shape": list(c["raw_states"].shape),
        "prelude_n_blocks": int(c["prelude_n_blocks"][0]),
        "messages_shape": list(c["messages"].shape),
        "out_absmax": float(np.abs(c["out"]).max()),
        "bytes": out_path.stat().st_size,
    }
    (ROOT / "goldens/message_controller.json").write_text(
        json.dumps(meta, indent=2) + "\n"
    )
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
