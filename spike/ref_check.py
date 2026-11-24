#!/usr/bin/env python3
"""Teacher-forced reference check against mlx-vlm.

Replays token sequences the engine produced (INKLING_TRACE json files) through
the local mlx-vlm model in one forward pass each, then compares, position by
position, the reference argmax and logit values with what the engine chose.

    python3 spike/ref_check.py goldens/regress/*.logits.json

One model load, nothing downloaded. Run it alone: it holds ~21 GB, and ~23 GB on
a 5,500-token trace. Keep traces to a few thousand tokens.

    REF_BY_CONTEXT=n   also print the worst logit difference per n steps
    REF_WORST=1        also print the worst step, token by token
    REF_STEPS_OUT=f    also save the per-step differences as .npy
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
WINDOW = 1024  # sliding window and ring size


def main() -> int:
    paths = [Path(p) for p in sys.argv[1:]]
    if not paths:
        print(__doc__)
        return 2
    from mlx_vlm import load

    model, _ = load(str(SNAP))
    lm = model.language_model
    # The engine keeps activations in f32; the shipped BF16 activations add ~0.1-1.0
    # of logit noise, so compare against an f32 cast unless REF_BF16 is set.
    if not os.environ.get("REF_BF16"):
        lm.set_dtype(mx.float32)
    bad = 0
    for path in paths:
        tr = json.loads(path.read_text())
        prompt, gen = tr["prompt_ids"], tr["gen_ids"]
        ids = prompt + gen[:-1]
        out = lm(mx.array([ids], dtype=mx.int32), cache=model.make_cache())
        logits = np.array(out.logits[0].astype(mx.float32))[:, :UNPADDED_VOCAB]
        n = len(gen)
        agree = 0
        worst_margin = 0.0  # how far the reference preferred another token
        max_diff = 0.0  # |engine logit - reference logit| over the engine's top-8
        per_step = []
        # A teacher-forced trace records what the engine would have picked separately.
        picks = tr.get("argmax_ids") or gen
        for i in range(n):
            row = logits[len(prompt) - 1 + i]
            ref = int(row.argmax())
            if ref == picks[i]:
                agree += 1
            else:
                worst_margin = max(worst_margin, float(row[ref] - row[picks[i]]))
            if i < len(tr.get("top", [])):
                step = max(abs(float(val) - float(row[tid])) for tid, val in tr["top"][i])
                max_diff = max(max_diff, step)
                per_step.append(step)
        if os.environ.get("REF_WORST") and per_step:
            # The step with the largest logit difference, token by token.
            i = int(np.argmax(per_step))
            row = logits[len(prompt) - 1 + i]
            print(f"  worst step {i} (context {len(prompt) + i}): reference argmax {int(row.argmax())} at {float(row.max()):.3f}")
            for tid, val in tr["top"][i]:
                print(f"    token {tid}: engine {float(val):8.3f}  reference {float(row[tid]):8.3f}")
        if os.environ.get("REF_STEPS_OUT"):
            np.save(os.environ["REF_STEPS_OUT"], np.array(per_step, dtype=np.float32))
        if os.environ.get("REF_BY_CONTEXT"):
            # Worst logit difference per stretch of context length, to see where drift starts.
            width = int(os.environ["REF_BY_CONTEXT"])
            base = len(prompt)
            for lo in range(0, len(per_step), width):
                chunk = per_step[lo : lo + width]
                print(f"  context {base + lo:6d}..{base + lo + len(chunk) - 1:6d}: {max(chunk):.4f}")
        if len(ids) <= WINDOW:
            ok = worst_margin < 0.05 and max_diff < 0.05
            rule = ""
        else:
            # Past the window two float32 runs stop agreeing digit for digit, the
            # reference with itself included: a routing near-tie falls the other way
            # and the logits shift.  What has to hold is that the picks agree and
            # that a disagreement is a close call.
            ok = agree >= 0.99 * n and worst_margin < 0.5
            rule = " (longer than the window: judged on picks)"
        bad += 0 if ok else 1
        print(
            f"{'ok  ' if ok else 'FAIL'} {path.name}: argmax agrees {agree}/{n}, "
            f"worst reference margin on a disagreement {worst_margin:.3f}, "
            f"max |logit diff| on engine top-8 {max_diff:.3f}{rule}"
        )
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
