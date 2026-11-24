#!/usr/bin/env python3
"""f32 CPU reference for layers 0 → 6 (first core entry, no messages)."""

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
    g = np.load(ROOT / "goldens/layer05_f32.npz")
    x = g["after_5"].astype(np.float32)
    # Reconstruct blocks after layer 5: only layer-0 boundary state.
    # After L0 boundary, blocks=[x0_in]; L1-5 non-boundary keep that.
    x0 = g["x_in"].astype(np.float32)
    # Re-run 0..5 quickly for exact blocks, or derive: after L0, block is x_in.
    blocks = [x0]
    x, blocks = run_sliding(tensors, x, blocks, layer_bundle(tensors, 6))
    print("layer 6 done", flush=True)
    out = ROOT / "goldens/layer06_f32.npz"
    np.savez_compressed(
        out,
        x_in=g["x_in"].astype(np.float32),
        after_5=g["after_5"].astype(np.float32),
        x_out=x,
        block0=x0,
    )
    meta = {
        "path": str(out.relative_to(ROOT)),
        "x_out_absmax": float(np.abs(x).max()),
        "x_out_norm": float(np.linalg.norm(x)),
        "n_blocks_after": len(blocks),
    }
    (ROOT / "goldens/layer06_f32.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
