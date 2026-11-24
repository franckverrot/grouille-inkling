//! Layers 0→5 f32 parity vs goldens/layer05_f32.npz.

use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use grouille_inkling::attn::{FullAttnWeights, SlidingAttnWeights};
use grouille_inkling::decoder::{layers_0_5, SlidingLayerWeights};
use grouille_inkling::layer5::FullLayerWeights;
use grouille_inkling::moe::ExpertPack;
use grouille_inkling::ops::HIDDEN;
use half::bf16;
use safetensors::tensor::SafeTensors;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .to_path_buf()
}

fn ckpt() -> PathBuf {
    PathBuf::from(std::env::var_os("HOME").unwrap())
        .join(".cache/huggingface/hub/models--thinkingmachines--Inkling-2-Mini")
        .join("snapshots/ed40e7da625f594fc9c4990f5f66be24873c22d7")
        .join("model-00001-of-00001.safetensors")
}

fn load_npz_f32(path: &std::path::Path, key: &str) -> Result<Vec<f32>> {
    let f = std::fs::File::open(path)?;
    let mut npz = npyz::npz::NpzArchive::new(f)?;
    let arr = npz
        .by_name(key)?
        .with_context(|| format!("missing {key}"))?;
    Ok(arr.into_vec()?)
}

fn bf16_f32(st: &SafeTensors, name: &str) -> Result<Vec<f32>> {
    let t = st.tensor(name)?;
    Ok(t.data()
        .chunks_exact(2)
        .map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
        .collect())
}

fn f32s(st: &SafeTensors, name: &str) -> Result<Vec<f32>> {
    let t = st.tensor(name)?;
    Ok(t.data()
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect())
}

fn u8s(st: &SafeTensors, name: &str) -> Result<Vec<u8>> {
    Ok(st.tensor(name)?.data().to_vec())
}

fn taps(w: &[f32]) -> Vec<f32> {
    w.to_vec()
}

fn split_gate_up(fused: &[u8], n: usize, rows: usize, stride: usize) -> (Vec<u8>, Vec<u8>) {
    let half = rows / 2;
    let mut gate = vec![0u8; n * half * stride];
    let mut up = vec![0u8; n * half * stride];
    for e in 0..n {
        for i in 0..half {
            let sg = ((e * rows) + i * 2) * stride;
            let su = ((e * rows) + i * 2 + 1) * stride;
            let d = (e * half + i) * stride;
            gate[d..d + stride].copy_from_slice(&fused[sg..sg + stride]);
            up[d..d + stride].copy_from_slice(&fused[su..su + stride]);
        }
    }
    (gate, up)
}

struct SlidingBuf {
    attn_norm: Vec<f32>,
    mlp_norm: Vec<f32>,
    attn_sconv: Vec<f32>,
    mlp_sconv: Vec<f32>,
    attn_query: Option<Vec<f32>>,
    mlp_query: Vec<f32>,
    router_w: Vec<f32>,
    router_b: Vec<f32>,
    wq: Vec<f32>,
    wk: Vec<f32>,
    wv: Vec<f32>,
    wr: Vec<f32>,
    wo: Vec<f32>,
    gate: Vec<f32>,
    sink: Vec<f32>,
    q_scale: Vec<f32>,
    rel_scale: Vec<f32>,
    rel_proj: Vec<f32>,
    k_sconv: Vec<f32>,
    v_sconv: Vec<f32>,
    gate_w: Vec<u8>,
    gate_s: Vec<u8>,
    up_w: Vec<u8>,
    up_s: Vec<u8>,
    down_w: Vec<u8>,
    down_s: Vec<u8>,
    gate_up_amax: Vec<f32>,
    down_amax: Vec<f32>,
    boundary: bool,
}

struct FullBuf {
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    v_norm: Vec<f32>,
    mlp_norm: Vec<f32>,
    attn_sconv: Vec<f32>,
    mlp_sconv: Vec<f32>,
    attn_queries: Vec<f32>,
    mlp_query: Vec<f32>,
    router_w: Vec<f32>,
    router_b: Vec<f32>,
    wq: Vec<f32>,
    wk: Vec<f32>,
    wv: Vec<f32>,
    wr: Vec<f32>,
    wo: Vec<f32>,
    gate: Vec<f32>,
    sink: Vec<f32>,
    q_scale: Vec<f32>,
    rel_scale: Vec<f32>,
    rel_proj: Vec<f32>,
    k_static: Vec<f32>,
    k_wproj: Vec<f32>,
    v_static: Vec<f32>,
    v_wproj: Vec<f32>,
    gate_w: Vec<u8>,
    gate_s: Vec<u8>,
    up_w: Vec<u8>,
    up_s: Vec<u8>,
    down_w: Vec<u8>,
    down_s: Vec<u8>,
    gate_up_amax: Vec<f32>,
    down_amax: Vec<f32>,
    boundary: bool,
}

fn load_sliding(st: &SafeTensors, i: usize) -> Result<SlidingBuf> {
    let p = format!("model.llm.layers.{i}.");
    let ep = format!("{p}mlp.experts.");
    let w13 = u8s(st, &format!("{ep}w13_weight"))?;
    let s13 = u8s(st, &format!("{ep}w13_weight_scale"))?;
    let (gate_w, up_w) = split_gate_up(&w13, 128, 2304, HIDDEN / 2);
    let (gate_s, up_s) = split_gate_up(&s13, 128, 2304, HIDDEN / 16);
    let attn_query = if i == 0 {
        None
    } else {
        Some(f32s(
            st,
            &format!("model.llm.attn_res.attn_queries.{}", i - 1),
        )?)
    };
    Ok(SlidingBuf {
        attn_norm: bf16_f32(st, &format!("{p}attn_norm.weight"))?,
        mlp_norm: bf16_f32(st, &format!("{p}mlp_norm.weight"))?,
        attn_sconv: taps(&bf16_f32(st, &format!("{p}attn_sconv.weight"))?),
        mlp_sconv: taps(&bf16_f32(st, &format!("{p}mlp_sconv.weight"))?),
        attn_query,
        mlp_query: f32s(st, &format!("model.llm.attn_res.mlp_queries.{i}"))?,
        router_w: bf16_f32(st, &format!("{p}mlp.gate.weight"))?,
        router_b: f32s(st, &format!("{p}mlp.gate.bias"))?,
        wq: bf16_f32(st, &format!("{p}attn.wq_du.weight"))?,
        wk: bf16_f32(st, &format!("{p}attn.wk_dv.weight"))?,
        wv: bf16_f32(st, &format!("{p}attn.wv_dv.weight"))?,
        wr: bf16_f32(st, &format!("{p}attn.wr_du.weight"))?,
        wo: bf16_f32(st, &format!("{p}attn.wo_ud.weight"))?,
        gate: bf16_f32(st, &format!("{p}attn.gate_attn.weight"))?,
        sink: bf16_f32(st, &format!("{p}attn.attn_sink_proj.weight"))?,
        q_scale: bf16_f32(st, &format!("{p}attn.q_l2_clip.scale"))?,
        rel_scale: bf16_f32(st, &format!("{p}attn.rel_l2_clip.scale"))?,
        rel_proj: bf16_f32(st, &format!("{p}attn.rel_logits_proj.proj"))?,
        k_sconv: taps(&bf16_f32(st, &format!("{p}attn.k_sconv.weight"))?),
        v_sconv: taps(&bf16_f32(st, &format!("{p}attn.v_sconv.weight"))?),
        gate_w,
        gate_s,
        up_w,
        up_s,
        down_w: u8s(st, &format!("{ep}w2_weight"))?,
        down_s: u8s(st, &format!("{ep}w2_weight_scale"))?,
        gate_up_amax: f32s(st, &format!("{ep}w13_weight_amax"))?,
        down_amax: f32s(st, &format!("{ep}w2_weight_amax"))?,
        boundary: i.is_multiple_of(6),
    })
}

fn load_full(st: &SafeTensors, i: usize) -> Result<FullBuf> {
    let p = format!("model.llm.layers.{i}.");
    let ep = format!("{p}mlp.experts.");
    let w13 = u8s(st, &format!("{ep}w13_weight"))?;
    let s13 = u8s(st, &format!("{ep}w13_weight_scale"))?;
    let (gate_w, up_w) = split_gate_up(&w13, 128, 2304, HIDDEN / 2);
    let (gate_s, up_s) = split_gate_up(&s13, 128, 2304, HIDDEN / 16);
    Ok(FullBuf {
        q_norm: bf16_f32(st, &format!("{p}q_input_norm.weight"))?,
        k_norm: bf16_f32(st, &format!("{p}k_input_norm.weight"))?,
        v_norm: bf16_f32(st, &format!("{p}v_input_norm.weight"))?,
        mlp_norm: bf16_f32(st, &format!("{p}mlp_norm.weight"))?,
        attn_sconv: taps(&bf16_f32(st, &format!("{p}attn_sconv.weight"))?),
        mlp_sconv: taps(&bf16_f32(st, &format!("{p}mlp_sconv.weight"))?),
        attn_queries: f32s(st, &format!("model.llm.attn_res.attn_queries.{}", i - 1))?,
        mlp_query: f32s(st, &format!("model.llm.attn_res.mlp_queries.{i}"))?,
        router_w: bf16_f32(st, &format!("{p}mlp.gate.weight"))?,
        router_b: f32s(st, &format!("{p}mlp.gate.bias"))?,
        wq: bf16_f32(st, &format!("{p}attn.wq_du.weight"))?,
        wk: bf16_f32(st, &format!("{p}attn.wk_dv.weight"))?,
        wv: bf16_f32(st, &format!("{p}attn.wv_dv.weight"))?,
        wr: bf16_f32(st, &format!("{p}attn.wr_du.weight"))?,
        wo: bf16_f32(st, &format!("{p}attn.wo_ud.weight"))?,
        gate: bf16_f32(st, &format!("{p}attn.gate_attn.weight"))?,
        sink: bf16_f32(st, &format!("{p}attn.attn_sink_proj.weight"))?,
        q_scale: bf16_f32(st, &format!("{p}attn.q_l2_clip.scale"))?,
        rel_scale: bf16_f32(st, &format!("{p}attn.rel_l2_clip.scale"))?,
        rel_proj: bf16_f32(st, &format!("{p}attn.rel_logits_proj.proj"))?,
        k_static: bf16_f32(st, &format!("{p}attn.k_sconv.static_w"))?,
        k_wproj: bf16_f32(st, &format!("{p}attn.k_sconv.weight_proj"))?,
        v_static: bf16_f32(st, &format!("{p}attn.v_sconv.static_w"))?,
        v_wproj: bf16_f32(st, &format!("{p}attn.v_sconv.weight_proj"))?,
        gate_w,
        gate_s,
        up_w,
        up_s,
        down_w: u8s(st, &format!("{ep}w2_weight"))?,
        down_s: u8s(st, &format!("{ep}w2_weight_scale"))?,
        gate_up_amax: f32s(st, &format!("{ep}w13_weight_amax"))?,
        down_amax: f32s(st, &format!("{ep}w2_weight_amax"))?,
        boundary: i.is_multiple_of(6),
    })
}

fn as_sliding<'a>(b: &'a SlidingBuf) -> SlidingLayerWeights<'a> {
    SlidingLayerWeights {
        attn_norm: &b.attn_norm,
        mlp_norm: &b.mlp_norm,
        attn_sconv: &b.attn_sconv,
        mlp_sconv: &b.mlp_sconv,
        attn_query: b.attn_query.as_deref(),
        mlp_query: &b.mlp_query,
        router_w: &b.router_w,
        router_b: &b.router_b,
        attn: SlidingAttnWeights {
        layer: None,
            wq: &b.wq,
            wk: &b.wk,
            wv: &b.wv,
            wr: &b.wr,
            wo: &b.wo,
            gate: &b.gate,
            sink: &b.sink,
            q_scale: &b.q_scale,
            rel_scale: &b.rel_scale,
            rel_proj: &b.rel_proj,
            k_sconv: &b.k_sconv,
            v_sconv: &b.v_sconv,
        },
        experts: ExpertPack {
            gate_w: &b.gate_w,
            gate_s: &b.gate_s,
            up_w: &b.up_w,
            up_s: &b.up_s,
            down_w: &b.down_w,
            down_s: &b.down_s,
            gate_up_amax: &b.gate_up_amax,
            down_amax: &b.down_amax,
            n_experts: 128,
            mxfp8: false,
            layer: None,
        },
        is_block_boundary: b.boundary,
        ctrl_attn: None,
        ctrl_k: None,
        ctrl_v: None,
        ctrl_mlp: None,
    }
}

fn as_full<'a>(b: &'a FullBuf) -> FullLayerWeights<'a> {
    FullLayerWeights {
        q_norm: &b.q_norm,
        k_norm: &b.k_norm,
        v_norm: &b.v_norm,
        mlp_norm: &b.mlp_norm,
        attn_sconv: &b.attn_sconv,
        mlp_sconv: &b.mlp_sconv,
        attn_queries: &b.attn_queries,
        mlp_query: &b.mlp_query,
        router_w: &b.router_w,
        router_b: &b.router_b,
        attn: FullAttnWeights {
        layer: None,
            wq: &b.wq,
            wk: &b.wk,
            wv: &b.wv,
            wr: &b.wr,
            wo: &b.wo,
            gate: &b.gate,
            sink: &b.sink,
            q_scale: &b.q_scale,
            rel_scale: &b.rel_scale,
            rel_proj: &b.rel_proj,
            k_static: &b.k_static,
            k_wproj: &b.k_wproj,
            v_static: &b.v_static,
            v_wproj: &b.v_wproj,
        },
        experts: ExpertPack {
            gate_w: &b.gate_w,
            gate_s: &b.gate_s,
            up_w: &b.up_w,
            up_s: &b.up_s,
            down_w: &b.down_w,
            down_s: &b.down_s,
            gate_up_amax: &b.gate_up_amax,
            down_amax: &b.down_amax,
            n_experts: 128,
            mxfp8: false,
            layer: None,
        },
        is_block_boundary: b.boundary,
        ctrl_attn: None,
        ctrl_k: None,
        ctrl_v: None,
        ctrl_mlp: None,
    }
}

#[test]
fn layers_0_5_match_f32_golden() -> Result<()> {
    let golden = root().join("goldens/layer05_f32.npz");
    if !golden.exists() {
        bail!(
            "missing {}; run python3 spike/export_layer05_f32.py",
            golden.display()
        );
    }
    let x_in = load_npz_f32(&golden, "x_in")?;
    let x_ref = load_npz_f32(&golden, "x_out")?;
    let tokens = x_in.len() / HIDDEN;

    let file = std::fs::File::open(ckpt())?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    let st = SafeTensors::deserialize(&mmap)?;
    let s0 = load_sliding(&st, 0)?;
    let s1 = load_sliding(&st, 1)?;
    let s2 = load_sliding(&st, 2)?;
    let s3 = load_sliding(&st, 3)?;
    let s4 = load_sliding(&st, 4)?;
    let f5 = load_full(&st, 5)?;
    let w0 = as_sliding(&s0);
    let w1 = as_sliding(&s1);
    let w2 = as_sliding(&s2);
    let w3 = as_sliding(&s3);
    let w4 = as_sliding(&s4);
    let wf = as_full(&f5);
    let pred = layers_0_5(&x_in, tokens, &[&w0, &w1, &w2, &w3, &w4], &wf);

    let mut maxabs = 0f32;
    for (a, b) in pred.iter().zip(x_ref.iter()) {
        maxabs = maxabs.max((a - b).abs());
    }
    eprintln!("layers 0→5 parity maxabs={maxabs}");
    assert!(maxabs < 1e-3, "layers05 maxabs={maxabs}");
    Ok(())
}
