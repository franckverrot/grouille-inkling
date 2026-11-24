#!/usr/bin/env python3
"""Decode regression check. Runs one engine process at a time.

    python3 spike/regress.py --save      # record goldens/regress/*.json from the current build
    python3 spike/regress.py             # compare the current build against them
    python3 spike/regress.py --bin PATH  # use another binary

For each prompt it runs the engine twice: once on the full-logits path (ids + top-8
logits per step) and once on the fast greedy path (GPU argmax). It then checks

  * fast-path ids == logits-path ids (same build),
  * ids == golden ids,
  * max |logit - golden logit| over the recorded top-8 stays small.

Goldens are only as good as the build that wrote them: certify that build with
spike/ref_check.py (mlx-vlm, teacher-forced) before trusting them.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "goldens" / "regress"
# A long prompt, so decode runs with several hundred tokens of history.
_PARA = (
    "Rivers begin as thin threads of meltwater and rain high in the hills, gather in valleys, "
    "and widen as they cross the plains toward the sea. Along the way they carve canyons, "
    "spread silt over floodplains, and feed wetlands that shelter birds and fish. People have "
    "settled beside them for water, for food, and for the easy road a current provides. "
)
PROMPTS = {
    "france": ("What's the capital of France?", 40),
    "sky": ("Explain in three sentences why the sky is blue.", 96),
    "rivers": ("Write a short essay about rivers.", 160),
    "long": ("Write a 1000-word short story about a lighthouse keeper.", 400),
    "context": (_PARA * 6 + "\n\nSummarize the text above in two sentences.", 64),
    # Longer than the 1024-token window: the prompt goes through in chunks and the
    # full-attention visits attend past the ring.
    "window": (
        (GOLD / "window.prompt.txt").read_text() + "\n\nSummarize the text above in three sentences.",
        96,
    ),
}
LOGIT_TOL = 0.05


def run(binary: Path, prompt: str, max_new: int, trace: Path, logits: bool) -> dict:
    env = dict(os.environ, INKLING_TRACE=str(trace))
    env.pop("INKLING_TRACE_LOGITS", None)
    if logits:
        env["INKLING_TRACE_LOGITS"] = "1"
    subprocess.run(
        [str(binary), "--no-stream", "-q", "--max", str(max_new), prompt],
        env=env,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return json.loads(trace.read_text())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--bin", default=str(ROOT / "target/release/inkling-generate"))
    ap.add_argument("--out", default=None, help="where to write this run's traces")
    args = ap.parse_args()
    binary = Path(args.bin)
    out_dir = GOLD if args.save else Path(args.out or (ROOT / "target" / "regress-last"))
    out_dir.mkdir(parents=True, exist_ok=True)
    failed = False
    for name, (prompt, max_new) in PROMPTS.items():
        lg = run(binary, prompt, max_new, out_dir / f"{name}.logits.json", True)
        fast = run(binary, prompt, max_new, out_dir / f"{name}.fast.json", False)
        msgs = []
        if fast["gen_ids"] != lg["gen_ids"]:
            failed = True
            msgs.append("fast path ids differ from logits path")
        if not args.save:
            gold = json.loads((GOLD / f"{name}.logits.json").read_text())
            same = gold["gen_ids"] == lg["gen_ids"]
            first = next(
                (i for i, (a, b) in enumerate(zip(gold["gen_ids"], lg["gen_ids"])) if a != b),
                min(len(gold["gen_ids"]), len(lg["gen_ids"])),
            )
            diff = 0.0
            for i in range(min(first + 1, len(gold["top"]), len(lg["top"]))):
                g = dict((int(t), v) for t, v in gold["top"][i])
                for t, v in lg["top"][i]:
                    if int(t) in g:
                        diff = max(diff, abs(v - g[int(t)]))
            if not same:
                failed = True
                msgs.append(f"ids diverge from golden at step {first}")
            if diff > LOGIT_TOL:
                failed = True
                msgs.append("logit drift over tolerance")
            msgs.append(f"max logit diff vs golden {diff:.4f}")
        print(f"{'FAIL' if any('differ' in m or 'diverge' in m or 'drift' in m for m in msgs) else 'ok  '} "
              f"{name}: {len(lg['gen_ids'])} tokens" + ("; " + "; ".join(msgs) if msgs else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
