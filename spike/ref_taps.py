#!/usr/bin/env python3
"""The drafter's four target states from mlx-vlm in float32, for a recorded run.

    python3 spike/ref_taps.py <trace dir> [tokens]

Reads `trace.json` of a recorded drafter run, rebuilds the context tokens (the
prompt, then each round's anchor and kept proposals), sends the first `tokens` of
them (900 by default, under the window) through the local model in one float32
pass and writes the stream entering visits 5, 23 and 41 and the final normed
state to `feats_f32.bin`, float32, `[tokens][4 * 2304]`.

One model load, nothing downloaded.  Run it alone: it holds about 21 GB.
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
TAPS = [5, 23, 41, 48]


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    trace_dir = Path(sys.argv[1])
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 900
    trace = json.loads((trace_dir / "trace.json").read_text())
    ids = list(trace["prompt"])
    for r in trace["rounds"]:
        ids.append(r["anchor"])
        ids.extend(r["draft"][: r["accepted"]])
    ids = ids[:limit]

    from mlx_vlm import load

    model, _ = load(str(SNAP))
    lm = model.language_model
    lm.set_dtype(mx.float32)
    out = lm(mx.array([ids], dtype=mx.int32), cache=model.make_cache(), capture_layer_ids=TAPS)
    states = [np.array(h[0].astype(mx.float32)) for h in out.hidden_states]
    feats = np.concatenate(states, axis=-1).astype("<f4")
    (trace_dir / "feats_f32.bin").write_bytes(feats.tobytes())
    print(f"{len(ids)} tokens, {feats.shape[1]} floats each -> {trace_dir / 'feats_f32.bin'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
