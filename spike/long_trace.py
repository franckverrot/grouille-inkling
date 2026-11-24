#!/usr/bin/env python3
"""Build a long teacher-forced trace from text files, for checks past the window.

    python3 spike/long_trace.py OUT.json FILE [FILE ...]

The text is cut into pieces and each piece is tokenized by the engine.  The first
piece becomes the prompt.  The token ids of the rest are fed to the engine one by
one (INKLING_FORCE_IDS) while it records its own pick and top-8 logits at every
step.  spike/ref_check.py then replays OUT.json through mlx-vlm.

Environment: LONG_PROMPT_PIECES (pieces that go into the prompt, default 1),
LONG_MAX_TOKENS (cap on forced tokens).  Runs one engine process at a time.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BIN = ROOT / "target" / "release" / "inkling-generate"
SPECIAL = 200000  # ids from here up are control tokens
PIECE_WORDS = 380


def pieces(text: str) -> list[str]:
    out, cur, n = [], [], 0
    for para in text.split("\n\n"):
        words = len(para.split())
        if cur and n + words > PIECE_WORDS:
            out.append("\n\n".join(cur))
            cur, n = [], 0
        cur.append(para)
        n += words
    if cur:
        out.append("\n\n".join(cur))
    return out


def engine(prompt: str, max_new: int, env: dict) -> None:
    subprocess.run(
        [str(BIN), "--no-stream", "-q", "--max", str(max_new), prompt],
        env=dict(os.environ, **env),
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    out = Path(sys.argv[1])
    text = "\n\n".join(Path(p).read_text() for p in sys.argv[2:])
    parts = pieces(text)
    n_prompt = int(os.environ.get("LONG_PROMPT_PIECES", "1"))
    prompt = "\n\n".join(parts[:n_prompt])
    forced: list[int] = []
    with tempfile.TemporaryDirectory() as tmp:
        tok = Path(tmp) / "tok.json"
        for part in parts[n_prompt:]:
            engine(part, 1, {"INKLING_TRACE": str(tok)})
            ids = json.loads(tok.read_text())["prompt_ids"]
            # Drop the chat template: keep what follows the user header.
            body = ids[ids.index(200000) + 2 :] if 200000 in ids else ids
            forced += [i for i in body if i < SPECIAL]
        cap = int(os.environ.get("LONG_MAX_TOKENS", "0"))
        if cap:
            forced = forced[:cap]
        ids_file = Path(tmp) / "forced.json"
        ids_file.write_text(json.dumps({"gen_ids": forced}, separators=(",", ":")))
        engine(
            prompt,
            len(forced),
            {
                "INKLING_FORCE_IDS": str(ids_file),
                "INKLING_TRACE": str(out),
                "INKLING_TRACE_LOGITS": "1",
            },
        )
    tr = json.loads(out.read_text())
    print(f"{out.name}: prompt {len(tr['prompt_ids'])} tokens, forced {len(tr['gen_ids'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
