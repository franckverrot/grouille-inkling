#!/usr/bin/env python3
"""Compare engine traces against saved mlx-vlm reference logits (no model load).

    python3 spike/ref_cmp.py REF_DIR trace.json [trace.json ...]

REF_DIR holds <name>.ref.npy written by spike/ref_dump.py: reference logits for the
positions that predict each generated token of the golden trace <name>. Traces must be
teacher-forced on that same token sequence (INKLING_FORCE_IDS).
"""
import json
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    ref_dir = Path(sys.argv[1])
    for p in sys.argv[2:]:
        tr = json.loads(Path(p).read_text())
        name = Path(p).name.split(".")[0]
        ref = np.load(ref_dir / f"{name}.logits.ref.npy")
        n = min(len(tr["argmax_ids"]), len(ref))
        agree = sum(int(ref[i].argmax()) == tr["argmax_ids"][i] for i in range(n))
        line = f"{Path(p).name}: engine argmax == reference argmax {agree}/{n}"
        if tr.get("top"):
            d1 = np.array([abs(tr["top"][i][0][1] - ref[i][tr["top"][i][0][0]]) for i in range(n)])
            d8 = np.array([max(abs(v - ref[i][t]) for t, v in tr["top"][i]) for i in range(n)])
            q = max(n // 4, 1)
            line += (
                f" | top-1 diff: first quarter {d1[:q].mean():.3f}, last quarter {d1[-q:].mean():.3f}, max {d1.max():.2f}"
                f" | top-8 diff: mean {d8.mean():.3f}, max {d8.max():.2f}, steps over 2.0: {(d8 > 2).sum()}"
            )
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
