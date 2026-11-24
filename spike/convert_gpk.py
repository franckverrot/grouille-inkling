#!/usr/bin/env python3
"""Convert Inkling-2-Mini HF safetensors → page-aligned .gpk (v1).

See header comment in prior revision for layout. Uses PyTorch for bf16.
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

MAGIC = b"GPK1"
VERSION = 1
PAGE = 16384
ALIGN = 256
HIDDEN = 2304
N_EXP = 128

DTYPE_U8 = 1
DTYPE_F16 = 2
DTYPE_F32 = 3


def align_up(n: int, a: int = ALIGN) -> int:
    return (n + a - 1) // a * a


def split_gate_up(fused: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """fused [E, rows, stride] interleaved gate/up → contiguous halves."""
    return np.ascontiguousarray(fused[:, 0::2, :]), np.ascontiguousarray(fused[:, 1::2, :])


def t_f16(t: torch.Tensor) -> np.ndarray:
    return t.detach().to(torch.float16).cpu().numpy()


def t_f32(t: torch.Tensor) -> np.ndarray:
    return t.detach().to(torch.float32).cpu().numpy()


def t_u8(t: torch.Tensor) -> np.ndarray:
    # Packed experts are uint8; scales are often Float8_e4m3: view as bytes.
    return t.detach().contiguous().view(torch.uint8).cpu().numpy().reshape(t.shape)


class Writer:
    def __init__(self):
        self.entries: list[tuple[str, int, bytes, tuple[int, ...]]] = []

    def add(self, name: str, dtype: int, arr: np.ndarray):
        if arr.size == 0:
            payload, shape = b"", (0,)
        else:
            if not arr.flags["C_CONTIGUOUS"]:
                arr = np.ascontiguousarray(arr)
            payload, shape = arr.tobytes(), tuple(int(x) for x in arr.shape)
        self.entries.append((name, dtype, payload, shape))

    def write(self, path: Path):
        toc_meta = []
        for name, dtype, payload, shape in self.entries:
            nb = name.encode("utf-8")
            dims = list(shape) + [0] * (4 - len(shape))
            assert len(shape) <= 4
            toc_meta.append((nb, dtype, len(shape), dims, len(payload)))

        fixed = 32
        toc_size = sum(2 + len(nb) + 4 + 4 + 16 + 8 + 8 for nb, *_ in toc_meta)
        header_size = align_up(fixed + toc_size, PAGE)

        cursor = header_size
        offsets = []
        for *_, nbytes in toc_meta:
            off = align_up(cursor, ALIGN)
            offsets.append(off)
            cursor = off + nbytes
        file_size = align_up(cursor, PAGE)

        buf = bytearray(file_size)
        buf[0:4] = MAGIC
        struct.pack_into("<I", buf, 4, VERSION)
        struct.pack_into("<I", buf, 8, len(self.entries))
        struct.pack_into("<I", buf, 12, header_size)
        struct.pack_into("<Q", buf, 16, file_size)
        struct.pack_into("<Q", buf, 24, 0)

        pos = 32
        for i, (nb, dtype, ndim, dims, nbytes) in enumerate(toc_meta):
            struct.pack_into("<H", buf, pos, len(nb))
            pos += 2
            buf[pos : pos + len(nb)] = nb
            pos += len(nb)
            struct.pack_into("<I", buf, pos, dtype)
            pos += 4
            struct.pack_into("<I", buf, pos, ndim)
            pos += 4
            struct.pack_into("<4I", buf, pos, *dims)
            pos += 16
            struct.pack_into("<Q", buf, pos, offsets[i])
            pos += 8
            struct.pack_into("<Q", buf, pos, nbytes)
            pos += 8

        for i, (_, _, payload, _) in enumerate(self.entries):
            if payload:
                off = offsets[i]
                buf[off : off + len(payload)] = payload

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(buf)
        print(f"wrote {path} ({file_size / 1e9:.2f} GB, {len(self.entries)} tensors)", flush=True)


def default_ckpt() -> Path:
    return (
        Path.home()
        / ".cache/huggingface/hub/models--thinkingmachines--Inkling-2-Mini"
        / "snapshots/ed40e7da625f594fc9c4990f5f66be24873c22d7"
        / "model-00001-of-00001.safetensors"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=default_ckpt())
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if not args.src.exists():
        sys.exit(f"missing {args.src}")

    w = Writer()
    print(f"reading {args.src}", flush=True)
    with safe_open(str(args.src), framework="pt") as f:
        def T(name: str) -> torch.Tensor:
            return f.get_tensor(name)

        print("head…", flush=True)
        w.add("head.embed", DTYPE_F16, t_f16(T("model.llm.embed.weight")))
        w.add("head.unembed", DTYPE_F16, t_f16(T("model.llm.unembed.weight")))
        w.add("head.embed_norm", DTYPE_F32, t_f32(T("model.llm.embed_norm.weight")))
        w.add("head.final_norm", DTYPE_F32, t_f32(T("model.llm.norm.weight")))
        w.add("head.output_query", DTYPE_F32, t_f32(T("model.llm.attn_res.output_query")))
        w.add("msg.cross", DTYPE_F32, t_f32(T("model.llm.middle_cycle_cross_pass_queries")))
        w.add("msg.kv", DTYPE_F32, t_f32(T("model.llm.middle_cycle_cross_pass_kv_queries")))

        f16_alias = {
            "attn.wq_du.weight": "wq",
            "attn.wk_dv.weight": "wk",
            "attn.wv_dv.weight": "wv",
            "attn.wr_du.weight": "wr",
            "attn.wo_ud.weight": "wo",
            "attn.gate_attn.weight": "gate",
            "attn.attn_sink_proj.weight": "sink",
            "attn.q_l2_clip.scale": "q_scale",
            "attn.rel_l2_clip.scale": "rel_scale",
            "attn.rel_logits_proj.proj": "rel_proj",
            "mlp.gate.weight": "router_w",
        }

        for i in range(24):
            print(f"layer {i}…", flush=True)
            p = f"model.llm.layers.{i}."
            ep = f"{p}mlp.experts."
            mx = i >= 20
            w13 = t_u8(T(f"{ep}w13_weight"))
            s13 = t_u8(T(f"{ep}w13_weight_scale"))
            w_stride = HIDDEN if mx else HIDDEN // 2
            s_stride = HIDDEN // 32 if mx else HIDDEN // 16
            w13 = w13.reshape(N_EXP, 2304, w_stride)
            s13 = s13.reshape(N_EXP, 2304, s_stride)
            gate_w, up_w = split_gate_up(w13)
            gate_s, up_s = split_gate_up(s13)
            pref = f"L{i:02d}"
            w.add(f"{pref}.gate_w", DTYPE_U8, gate_w)
            w.add(f"{pref}.gate_s", DTYPE_U8, gate_s)
            w.add(f"{pref}.up_w", DTYPE_U8, up_w)
            w.add(f"{pref}.up_s", DTYPE_U8, up_s)
            w.add(f"{pref}.down_w", DTYPE_U8, t_u8(T(f"{ep}w2_weight")))
            w.add(f"{pref}.down_s", DTYPE_U8, t_u8(T(f"{ep}w2_weight_scale")))
            if mx:
                w.add(f"{pref}.gate_up_amax", DTYPE_F32, np.zeros(0, np.float32))
                w.add(f"{pref}.down_amax", DTYPE_F32, np.zeros(0, np.float32))
            else:
                w.add(f"{pref}.gate_up_amax", DTYPE_F32, t_f32(T(f"{ep}w13_weight_amax")))
                w.add(f"{pref}.down_amax", DTYPE_F32, t_f32(T(f"{ep}w2_weight_amax")))

            for src, alias in f16_alias.items():
                w.add(f"{pref}.{alias}", DTYPE_F16, t_f16(T(f"{p}{src}")))

            is_full = i in (5, 11, 17, 23)
            w.add(f"{pref}.mlp_norm", DTYPE_F32, t_f32(T(f"{p}mlp_norm.weight")))
            w.add(f"{pref}.router_b", DTYPE_F32, t_f32(T(f"{p}mlp.gate.bias")))
            w.add(f"{pref}.attn_sconv", DTYPE_F32, t_f32(T(f"{p}attn_sconv.weight")))
            w.add(f"{pref}.mlp_sconv", DTYPE_F32, t_f32(T(f"{p}mlp_sconv.weight")))
            w.add(f"{pref}.mlp_query", DTYPE_F32, t_f32(T(f"model.llm.attn_res.mlp_queries.{i}")))
            if is_full:
                # Dyn short-conv + per-input norms (no sliding attn_norm / k_sconv taps).
                for tag, suffix in [
                    ("k_static", "attn.k_sconv.static_w"),
                    ("k_wproj", "attn.k_sconv.weight_proj"),
                    ("v_static", "attn.v_sconv.static_w"),
                    ("v_wproj", "attn.v_sconv.weight_proj"),
                    ("q_norm", "q_input_norm.weight"),
                    ("k_norm", "k_input_norm.weight"),
                    ("v_norm", "v_input_norm.weight"),
                ]:
                    w.add(f"{pref}.{tag}", DTYPE_F32, t_f32(T(f"{p}{suffix}")))
                w.add(
                    f"{pref}.attn_query",
                    DTYPE_F32,
                    t_f32(T(f"model.llm.attn_res.attn_queries.{i - 1}")),
                )
            else:
                w.add(f"{pref}.attn_norm", DTYPE_F32, t_f32(T(f"{p}attn_norm.weight")))
                w.add(f"{pref}.k_sconv", DTYPE_F32, t_f32(T(f"{p}attn.k_sconv.weight")))
                w.add(f"{pref}.v_sconv", DTYPE_F32, t_f32(T(f"{p}attn.v_sconv.weight")))
                if i > 0:
                    w.add(
                        f"{pref}.attn_query",
                        DTYPE_F32,
                        t_f32(T(f"model.llm.attn_res.attn_queries.{i - 1}")),
                    )

            if 6 <= i < 18:
                for cname in ("attn", "k", "v", "mlp"):
                    cp = f"{p}controllers.{cname}."
                    w.add(f"{pref}.ctrl_{cname}_down", DTYPE_F32, t_f32(T(f"{cp}down_weight")))
                    w.add(f"{pref}.ctrl_{cname}_forget", DTYPE_F32, t_f32(T(f"{cp}forget_weight")))
                    w.add(f"{pref}.ctrl_{cname}_gain", DTYPE_F32, t_f32(T(f"{cp}gain_weight")))

    w.write(args.out)


if __name__ == "__main__":
    main()
