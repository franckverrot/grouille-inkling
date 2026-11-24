//! Full-attn layer-5 f32 parity vs `goldens/layer5_f32.npz`.

use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use grouille_inkling::attn::FullAttnWeights;
use grouille_inkling::layer5::{full_layer_prefill, FullLayerWeights};
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

#[test]
fn layer5_matches_f32_golden() -> Result<()> {
    let golden = root().join("goldens/layer5_f32.npz");
    if !golden.exists() {
        bail!(
            "missing {}; run python3 spike/export_layer5_f32.py",
            golden.display()
        );
    }
    let block = load_npz_f32(&golden, "block")?;
    let x_in = load_npz_f32(&golden, "x_in")?;
    let x_ref = load_npz_f32(&golden, "x_out")?;
    let tokens = x_in.len() / HIDDEN;
    assert_eq!(tokens, 14);

    let file = std::fs::File::open(ckpt())?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    let st = SafeTensors::deserialize(&mmap)?;
    let p = "model.llm.layers.5.";

    let q_norm = bf16_f32(&st, &format!("{p}q_input_norm.weight"))?;
    let k_norm = bf16_f32(&st, &format!("{p}k_input_norm.weight"))?;
    let v_norm = bf16_f32(&st, &format!("{p}v_input_norm.weight"))?;
    let mlp_norm = bf16_f32(&st, &format!("{p}mlp_norm.weight"))?;
    let attn_sconv = taps(&bf16_f32(&st, &format!("{p}attn_sconv.weight"))?);
    let mlp_sconv = taps(&bf16_f32(&st, &format!("{p}mlp_sconv.weight"))?);
    let attn_queries = f32s(&st, "model.llm.attn_res.attn_queries.4")?;
    let mlp_query = f32s(&st, "model.llm.attn_res.mlp_queries.5")?;
    let router_w = bf16_f32(&st, &format!("{p}mlp.gate.weight"))?;
    let router_b = f32s(&st, &format!("{p}mlp.gate.bias"))?;

    let wq = bf16_f32(&st, &format!("{p}attn.wq_du.weight"))?;
    let wk = bf16_f32(&st, &format!("{p}attn.wk_dv.weight"))?;
    let wv = bf16_f32(&st, &format!("{p}attn.wv_dv.weight"))?;
    let wr = bf16_f32(&st, &format!("{p}attn.wr_du.weight"))?;
    let wo = bf16_f32(&st, &format!("{p}attn.wo_ud.weight"))?;
    let gate = bf16_f32(&st, &format!("{p}attn.gate_attn.weight"))?;
    let sink = bf16_f32(&st, &format!("{p}attn.attn_sink_proj.weight"))?;
    let q_scale = bf16_f32(&st, &format!("{p}attn.q_l2_clip.scale"))?;
    let rel_scale = bf16_f32(&st, &format!("{p}attn.rel_l2_clip.scale"))?;
    let rel_proj = bf16_f32(&st, &format!("{p}attn.rel_logits_proj.proj"))?;
    let k_static = bf16_f32(&st, &format!("{p}attn.k_sconv.static_w"))?;
    let k_wproj = bf16_f32(&st, &format!("{p}attn.k_sconv.weight_proj"))?;
    let v_static = bf16_f32(&st, &format!("{p}attn.v_sconv.static_w"))?;
    let v_wproj = bf16_f32(&st, &format!("{p}attn.v_sconv.weight_proj"))?;

    let ep = "model.llm.layers.5.mlp.experts.";
    let w13 = u8s(&st, &format!("{ep}w13_weight"))?;
    let s13 = u8s(&st, &format!("{ep}w13_weight_scale"))?;
    let a13 = f32s(&st, &format!("{ep}w13_weight_amax"))?;
    let w2 = u8s(&st, &format!("{ep}w2_weight"))?;
    let s2 = u8s(&st, &format!("{ep}w2_weight_scale"))?;
    let a2 = f32s(&st, &format!("{ep}w2_weight_amax"))?;
    let (gate_w, up_w) = split_gate_up(&w13, 128, 2304, HIDDEN / 2);
    let (gate_s, up_s) = split_gate_up(&s13, 128, 2304, HIDDEN / 16);

    let weights = FullLayerWeights {
        q_norm: &q_norm,
        k_norm: &k_norm,
        v_norm: &v_norm,
        mlp_norm: &mlp_norm,
        attn_sconv: &attn_sconv,
        mlp_sconv: &mlp_sconv,
        attn_queries: &attn_queries,
        mlp_query: &mlp_query,
        router_w: &router_w,
        router_b: &router_b,
        attn: FullAttnWeights {
        layer: None,
            wq: &wq,
            wk: &wk,
            wv: &wv,
            wr: &wr,
            wo: &wo,
            gate: &gate,
            sink: &sink,
            q_scale: &q_scale,
            rel_scale: &rel_scale,
            rel_proj: &rel_proj,
            k_static: &k_static,
            k_wproj: &k_wproj,
            v_static: &v_static,
            v_wproj: &v_wproj,
        },
        experts: ExpertPack {
            gate_w: &gate_w,
            gate_s: &gate_s,
            up_w: &up_w,
            up_s: &up_s,
            down_w: &w2,
            down_s: &s2,
            gate_up_amax: &a13,
            down_amax: &a2,
            n_experts: 128,
            mxfp8: false,
            layer: None,
        },
        is_block_boundary: false,
        ctrl_attn: None,
        ctrl_k: None,
        ctrl_v: None,
        ctrl_mlp: None,
    };

    let blocks: Vec<Vec<f32>> = (0..tokens)
        .map(|t| block[t * HIDDEN..(t + 1) * HIDDEN].to_vec())
        .collect();
    let (pred, _) = full_layer_prefill(&x_in, tokens, &blocks, &weights);
    let mut maxabs = 0f32;
    let mut sumabs = 0f32;
    for (p, r) in pred.iter().zip(x_ref.iter()) {
        let d = (p - r).abs();
        maxabs = maxabs.max(d);
        sumabs += d;
    }
    let meanabs = sumabs / pred.len() as f32;
    eprintln!("layer5 parity maxabs={maxabs} meanabs={meanabs}");
    assert!(maxabs < 1e-3, "layer5 f32 parity failed: maxabs={maxabs}");
    Ok(())
}
