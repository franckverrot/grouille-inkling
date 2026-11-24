#!/usr/bin/env python3
"""f32 first core pass tail: layers 12→17 (no messages)."""

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
    g6 = np.load(ROOT / "goldens/layer06_f32.npz")
    g11 = np.load(ROOT / "goldens/core_6_11_f32.npz")
    x = g11["x_out"].astype(np.float32)
    # after L11: still 2 blocks (L12 is next boundary)
    blocks = [g6["block0"].astype(np.float32), g6["after_5"].astype(np.float32)]
    after = {"after_11": x.copy()}
    # L12 is boundary
    x, blocks = run_sliding(tensors, x, blocks, layer_bundle(tensors, 12))
    after["after_12"] = x.copy()
    print(f"layer 12 done n_blocks={len(blocks)}", flush=True)
    for i in range(13, 17):
        x, blocks = run_sliding(tensors, x, blocks, layer_bundle(tensors, i))
        after[f"after_{i}"] = x.copy()
        print(f"layer {i} done n_blocks={len(blocks)}", flush=True)
    x, blocks = run_full(tensors, x, blocks, full_bundle(tensors, 17))
    after["after_17"] = x.copy()
    print(f"layer 17 done n_blocks={len(blocks)}", flush=True)
    blocks_in = np.stack(
        [g6["block0"].astype(np.float32), g6["after_5"].astype(np.float32)], 1
    )
    blocks_out = np.stack(blocks, axis=1)
    out = ROOT / "goldens/core_12_17_f32.npz"
    np.savez_compressed(
        out,
        x_in=g11["x_out"].astype(np.float32),
        blocks_in=blocks_in,
        x_out=x,
        blocks_out=blocks_out,
        **after,
    )
    meta = {
        "path": str(out.relative_to(ROOT)),
        "n_blocks_out": int(blocks_out.shape[1]),
        "x_out_norm": float(np.linalg.norm(x)),
        "x_out_absmax": float(np.abs(x).max()),
    }
    (ROOT / "goldens/core_12_17_f32.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
