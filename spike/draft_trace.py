#!/usr/bin/env python3
"""Record what mlx-vlm's Inkling drafter sees and proposes, round by round.

    TRACE_DIR=<dir> python3 spike/draft_trace.py --model thinkingmachines/Inkling-2-Mini \
        --draft-model thinkingmachines/Inkling-2-Mini --reasoning-effort none \
        --prompt "..." --temperature 0 --max-tokens 1000

Writes `trace.json` (the prompt ids and, per round, the anchor, the number of
context rows, the proposed tokens and the tokens the target kept) and `feats.bin`
(the target states handed to the drafter, float32, `[rows][4 * 2304]`, in round
order).  The arguments are those of `mlx_vlm.generate`, which it wraps without
changing it.  Use the Python mlx-vlm is installed in.

One model load, nothing downloaded.  Run it alone: it holds about 22 GB.
"""

import atexit
import json
import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import mlx.core as mx
import numpy as np
import mlx_vlm.speculative.dflash as dflash
from mlx_vlm.models.inkling2 import language
from mlx_vlm.speculative.drafters.inkling2_dspark import dspark

out_dir = os.environ["TRACE_DIR"]
os.makedirs(out_dir, exist_ok=True)
feats = open(os.path.join(out_dir, "feats.bin"), "wb")
trace = {"prompt": [], "rounds": []}
state = {"drafting": False}

orig_draft = dspark.Model.draft_block
orig_walk = dflash._speculative_walk
orig_lm = language.LanguageModel.__call__


def lm_call(self, inputs=None, *args, **kwargs):
    # Everything the target reads before the first draft is the prompt.
    if inputs is not None and not state["drafting"]:
        trace["prompt"].extend(int(t) for t in np.array(inputs).reshape(-1))
    return orig_lm(self, inputs, *args, **kwargs)


def draft_block(self, last_bonus, hidden, cache, block_size, sampler, token_dtype=mx.int32):
    state["drafting"] = True
    h = np.array(hidden.astype(mx.float32))
    feats.write(h.reshape(-1, h.shape[-1]).astype("<f4").tobytes())
    out = orig_draft(self, last_bonus, hidden, cache, block_size, sampler, token_dtype)
    anchor = last_bonus if isinstance(last_bonus, int) else int(np.array(last_bonus).reshape(-1)[0])
    trace["rounds"].append(
        {
            "anchor": int(anchor),
            "rows": int(h.shape[1]),
            "block": int(block_size),
            "draft": [int(t) for t in np.array(out).reshape(-1)],
        }
    )
    return out


def walk(draft_tokens, target_tokens, budget):
    accepted, new_tokens = orig_walk(draft_tokens, target_tokens, budget)
    trace["rounds"][-1]["accepted"] = int(accepted)
    trace["rounds"][-1]["new"] = [int(t) for t in new_tokens]
    return accepted, new_tokens


def dump():
    feats.close()
    with open(os.path.join(out_dir, "trace.json"), "w") as f:
        json.dump(trace, f)


language.LanguageModel.__call__ = lm_call
dspark.Model.draft_block = draft_block
dflash._speculative_walk = walk
atexit.register(dump)

if __name__ == "__main__":
    from mlx_vlm.generate import main

    sys.argv[0] = "mlx_vlm.generate"
    sys.exit(main())
