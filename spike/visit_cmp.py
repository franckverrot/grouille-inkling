#!/usr/bin/env python3
"""Per-visit residual-stream error of the engine against the mlx-vlm reference.

    python3 spike/visit_cmp.py REF.npz DUMP_DIR

REF.npz from spike/ref_dump.py; DUMP_DIR from INKLING_DUMP_X on the same tokens.
Prints relative L2 error after each of the 48 visits for some prompt positions
(prefill path) and some generated positions (decode path).
"""
import os
import sys
from pathlib import Path

import numpy as np

H = 2304
LAYERS = list(range(6)) + list(range(6, 18)) * 3 + list(range(18, 24))


def rel(a, b):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))


def main() -> int:
    ref = np.load(sys.argv[1])
    visits, p = ref["visits"], int(ref["prompt_len"])
    d = Path(sys.argv[2])
    # A prompt longer than the chunk size is dumped chunk by chunk, 48 visits each.
    cap = min(int(os.environ.get("INKLING_PROMPT_CHUNK", "1024")), 1024)
    per = -(-p // -(-p // cap))
    raw = np.fromfile(d / "prefill.bin", dtype=np.float32)
    pre = np.empty((48, p, H), dtype=np.float32)
    off = 0
    for lo in range(0, p, per):
        n = min(per, p - lo)
        pre[:, lo : lo + n] = raw[off : off + 48 * n * H].reshape(48, n, H)
        off += 48 * n * H
    dec = np.fromfile(d / "decode.bin", dtype=np.float32)
    n_dec = dec.size // (48 * H)
    dec = dec[: n_dec * 48 * H].reshape(n_dec, 48, H)
    pre_pos = sorted({0, p // 2, p - 1})
    dec_steps = sorted({0, 1, n_dec // 4, n_dec // 2, n_dec - 1})
    print("relative L2 error of x after each visit (engine vs reference)")
    print("visit layer | " + " ".join(f"pre@{q:<3d}" for q in pre_pos) + " | "
          + " ".join(f"dec+{s:<3d}" for s in dec_steps))
    for v in range(48):
        a = " ".join(f"{rel(pre[v, q], visits[v, q]):7.4f}" for q in pre_pos)
        b = " ".join(f"{rel(dec[s, v], visits[v, p + s]):7.4f}" for s in dec_steps)
        print(f"{v:5d} {LAYERS[v]:5d} | {a} | {b}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
