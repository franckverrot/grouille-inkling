#!/usr/bin/env python3
"""f32 first core pass fragment: layers 6→11 (no messages)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "spike"))
from export_layer01_f32 import layer_bundle, load_tensors, run_sliding  # noqa: E402
from export_layer05_f32 import full_bundle, run_full  # noqa: E402


def main():
    print("loading…", flush=True)
    tensors = load_tensors()
    g = np.load(ROOT / "goldens/layer06_f32.npz")
    x = g["x_out"].astype(np.float32)
    # after L6 boundary: blocks = [block0, after_5]
    blocks = [g["block0"].astype(np.float32), g["after_5"].astype(np.float32)]
    after = {"after_6": x.copy()}
    for i in range(7, 11):
        x, blocks = run_sliding(tensors, x, blocks, layer_bundle(tensors, i))
        after[f"after_{i}"] = x.copy()
        print(f"layer {i} done n_blocks={len(blocks)}", flush=True)
    x, blocks = run_full(tensors, x, blocks, full_bundle(tensors, 11))
    after["after_11"] = x.copy()
    print(f"layer 11 done n_blocks={len(blocks)}", flush=True)
    out = ROOT / "goldens/core_6_11_f32.npz"
    np.savez_compressed(
        out,
        x_in=g["x_out"].astype(np.float32),
        block0=g["block0"].astype(np.float32),
        block1=g["after_5"].astype(np.float32),
        x_out=x,
        **after,
    )
    meta = {
        "path": str(out.relative_to(ROOT)),
        "n_blocks": len(blocks),
        "x_out_norm": float(np.linalg.norm(x)),
        "x_out_absmax": float(np.abs(x).max()),
    }
    (ROOT / "goldens/core_6_11_f32.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
