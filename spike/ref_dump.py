#!/usr/bin/env python3
"""Dump mlx-vlm reference states for a token sequence (teacher-forced, one pass).

    python3 spike/ref_dump.py OUT_DIR trace.json [trace.json ...]

For each trace writes OUT_DIR/<name>.ref.npz with
  visits  [48, T, 2304] float32  residual stream after every layer visit
  logits  [T, unpadded_vocab] float32
where T = len(prompt_ids) + len(gen_ids) - 1. Also OUT_DIR/<name>.logits.ref.npy
(logits for the positions that predict each generated token) for spike/ref_cmp.py.
Run alone: ~21 GB. Nothing is downloaded.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402

SNAP = (
    Path.home()
    / ".cache/huggingface/hub/models--thinkingmachines--Inkling-2-Mini"
    / "snapshots/ed40e7da625f594fc9c4990f5f66be24873c22d7"
)
UNPADDED_VOCAB = 200058


def main() -> int:
    out_dir = Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)
    from mlx_vlm import load

    model, _ = load(str(SNAP))
    lm = model.language_model
    if os.environ.get("REF_F32"):
        lm.set_dtype(mx.float32)
    from mlx_vlm.models.inkling2 import language

    sink: list = []
    orig_call = language.DecoderLayer.__call__

    def tapped(self, *args, **kwargs):
        out = orig_call(self, *args, **kwargs)
        sink.append(out)
        return out

    language.DecoderLayer.__call__ = tapped
    for p in sys.argv[2:]:
        tr = json.loads(Path(p).read_text())
        prompt, gen = tr["prompt_ids"], tr["gen_ids"]
        ids = prompt + gen[:-1]
        sink.clear()
        out = lm(mx.array([ids], dtype=mx.int32), cache=model.make_cache())
        logits = np.array(out.logits[0].astype(mx.float32))[:, :UNPADDED_VOCAB]
        visits = np.stack([np.array(x[0].astype(mx.float32)) for x in sink])
        name = Path(p).name.split(".")[0]
        np.savez(out_dir / f"{name}.ref.npz", visits=visits, logits=logits,
                 prompt_len=len(prompt))
        np.save(out_dir / f"{name}.logits.ref.npy", logits[len(prompt) - 1:])
        print(f"{name}: visits {visits.shape}, logits {logits.shape}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
