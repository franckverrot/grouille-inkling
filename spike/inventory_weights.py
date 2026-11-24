#!/usr/bin/env python3
"""Inventory Inkling-2-Mini safetensors into docs/weight_inventory.json."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SNAP = (
    Path.home()
    / ".cache/huggingface/hub/models--thinkingmachines--Inkling-2-Mini"
    / "snapshots/ed40e7da625f594fc9c4990f5f66be24873c22d7"
)


def nbytes(shape: tuple[int, ...], dtype: str) -> int:
    bits = {
        "F32": 32,
        "F16": 16,
        "BF16": 16,
        "F64": 64,
        "I8": 8,
        "U8": 8,
        "I32": 32,
        "I16": 16,
        "BOOL": 1,
        "F8_E4M3": 8,
        "F8_E5M2": 8,
    }.get(dtype, 16)
    n = 1
    for d in shape:
        n *= d
    return n * bits // 8


def family_key(name: str) -> str:
    parts = name.split(".")
    if "layers" in parts:
        i = parts.index("layers")
        return "layers.*." + ".".join(parts[i + 2 :])
    return name


def main() -> None:
    snap = Path(DEFAULT_SNAP)
    cfg = json.loads((snap / "config.json").read_text())
    idx = json.loads((snap / "model.safetensors.index.json").read_text())
    weights = (snap / "model-00001-of-00001.safetensors").resolve()
    if not weights.exists():
        raise SystemExit(f"missing weights at {weights}")

    shapes: dict[str, tuple[int, ...]] = {}
    dtypes: dict[str, str] = {}
    with safe_open(str(weights), framework="pt", device="cpu") as f:
        for key in f.keys():
            sl = f.get_slice(key)
            shapes[key] = tuple(sl.get_shape())
            dtypes[key] = str(sl.get_dtype())

    layer0 = sorted(k for k in shapes if k.startswith("model.llm.layers.0."))
    fam: dict[str, dict] = defaultdict(
        lambda: {"count": 0, "bytes_est": 0, "dtypes": set(), "example": None}
    )
    for key, shape in shapes.items():
        entry = fam[family_key(key)]
        entry["count"] += 1
        entry["bytes_est"] += nbytes(shape, dtypes[key])
        entry["dtypes"].add(dtypes[key])
        entry["example"] = {
            "key": key,
            "shape": list(shape),
            "dtype": dtypes[key],
        }

    payload = {
        "checkpoint": f"thinkingmachines/Inkling-2-Mini@{snap.name}",
        "weights_path": weights.name,
        "total_size": idx["metadata"]["total_size"],
        "text_config": cfg["text_config"],
        "n_tensors": len(shapes),
        "layer0": {
            k: {"shape": list(shapes[k]), "dtype": dtypes[k]} for k in layer0
        },
        "families": {
            k: {**v, "dtypes": sorted(v["dtypes"])}
            for k, v in sorted(fam.items(), key=lambda kv: -kv[1]["bytes_est"])
        },
    }
    out = ROOT / "docs" / "weight_inventory.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")

    print(f"weights {weights}")
    print(f"size_gb {weights.stat().st_size / 1e9:.2f}")
    print(f"tensors {len(shapes)}  layer0 {len(layer0)}")
    print(f"wrote {out}")
    print("layer0:")
    for key in layer0:
        print(f"  {dtypes[key]:12s} {str(shapes[key]):28s} {key}")
    print("top families by bytes:")
    for key, entry in sorted(fam.items(), key=lambda kv: -kv[1]["bytes_est"])[:25]:
        print(
            f"  {entry['bytes_est'] / 1e9:7.3f} GB  n={entry['count']:4d}  {key}"
        )


if __name__ == "__main__":
    main()
