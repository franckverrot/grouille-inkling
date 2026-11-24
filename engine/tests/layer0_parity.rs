//! Full layer-0 f32 parity vs `goldens/layer0_f32.npz`.

use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use grouille_inkling::attn::SlidingAttnWeights;
use grouille_inkling::layer0::{layer0_prefill, Layer0Weights};
use grouille_inkling::moe::ExpertPack;
use grouille_inkling::ops::HIDDEN;
use half::bf16;
use safetensors::tensor::SafeTensors;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .to_path_buf()
}

fn default_checkpoint() -> PathBuf {
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

fn tensor_bf16_as_f32(st: &SafeTensors, name: &str) -> Result<Vec<f32>> {
    let t = st.tensor(name)?;
    let bytes = t.data();
    let mut out = Vec::with_capacity(bytes.len() / 2);
    for chunk in bytes.chunks_exact(2) {
        let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
        out.push(bf16::from_bits(bits).to_f32());
    }
    Ok(out)
}

fn tensor_f32(st: &SafeTensors, name: &str) -> Result<Vec<f32>> {
    let t = st.tensor(name)?;
    let bytes = t.data();
    let mut out = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        out.push(f32::from_le_bytes(chunk.try_into().unwrap()));
    }
    Ok(out)
}

fn tensor_u8(st: &SafeTensors, name: &str) -> Result<Vec<u8>> {
    Ok(st.tensor(name)?.data().to_vec())
}

fn taps_c1k_to_ck(w: &[f32], channels: usize, k: usize) -> Vec<f32> {
    assert_eq!(w.len(), channels * k);
    w.to_vec()
}

fn split_gate_up(
    fused: &[u8],
    n_experts: usize,
    rows: usize,
    row_stride: usize,
) -> (Vec<u8>, Vec<u8>) {
    let mut gate = vec![0u8; n_experts * (rows / 2) * row_stride];
    let mut up = vec![0u8; n_experts * (rows / 2) * row_stride];
    let half = rows / 2;
    for e in 0..n_experts {
        for i in 0..half {
            let src_g = ((e * rows) + (i * 2)) * row_stride;
            let src_u = ((e * rows) + (i * 2 + 1)) * row_stride;
            let dst = (e * half + i) * row_stride;
            gate[dst..dst + row_stride].copy_from_slice(&fused[src_g..src_g + row_stride]);
            up[dst..dst + row_stride].copy_from_slice(&fused[src_u..src_u + row_stride]);
        }
    }
    (gate, up)
}

#[test]
fn layer0_matches_f32_golden() -> Result<()> {
    let root = repo_root();
    let golden = root.join("goldens/layer0_f32.npz");
    if !golden.exists() {
        bail!(
            "missing {}; run python3 spike/export_layer0_f32.py",
            golden.display()
        );
    }
    let x_in = load_npz_f32(&golden, "x_in")?;
    let x_ref = load_npz_f32(&golden, "x_out")?;
    let tokens = x_in.len() / HIDDEN;
    assert_eq!(tokens, 14);

    let ckpt = default_checkpoint();
    let file = std::fs::File::open(&ckpt)?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    let st = SafeTensors::deserialize(&mmap)?;

    let p = "model.llm.layers.0.";
    let attn_norm = tensor_bf16_as_f32(&st, &format!("{p}attn_norm.weight"))?;
    let mlp_norm = tensor_bf16_as_f32(&st, &format!("{p}mlp_norm.weight"))?;
    let attn_sconv = taps_c1k_to_ck(
        &tensor_bf16_as_f32(&st, &format!("{p}attn_sconv.weight"))?,
        HIDDEN,
        4,
    );
    let mlp_sconv = taps_c1k_to_ck(
        &tensor_bf16_as_f32(&st, &format!("{p}mlp_sconv.weight"))?,
        HIDDEN,
        4,
    );
    let mlp_query = tensor_f32(&st, "model.llm.attn_res.mlp_queries.0")?;
    let router_w = tensor_bf16_as_f32(&st, &format!("{p}mlp.gate.weight"))?;
    let router_b = tensor_f32(&st, &format!("{p}mlp.gate.bias"))?;

    let wq = tensor_bf16_as_f32(&st, &format!("{p}attn.wq_du.weight"))?;
    let wk = tensor_bf16_as_f32(&st, &format!("{p}attn.wk_dv.weight"))?;
    let wv = tensor_bf16_as_f32(&st, &format!("{p}attn.wv_dv.weight"))?;
    let wr = tensor_bf16_as_f32(&st, &format!("{p}attn.wr_du.weight"))?;
    let wo = tensor_bf16_as_f32(&st, &format!("{p}attn.wo_ud.weight"))?;
    let gate = tensor_bf16_as_f32(&st, &format!("{p}attn.gate_attn.weight"))?;
    let sink = tensor_bf16_as_f32(&st, &format!("{p}attn.attn_sink_proj.weight"))?;
    let q_scale = tensor_bf16_as_f32(&st, &format!("{p}attn.q_l2_clip.scale"))?;
    let rel_scale = tensor_bf16_as_f32(&st, &format!("{p}attn.rel_l2_clip.scale"))?;
    let rel_proj = tensor_bf16_as_f32(&st, &format!("{p}attn.rel_logits_proj.proj"))?;
    let k_sconv = taps_c1k_to_ck(
        &tensor_bf16_as_f32(&st, &format!("{p}attn.k_sconv.weight"))?,
        8 * 128,
        4,
    );
    let v_sconv = taps_c1k_to_ck(
        &tensor_bf16_as_f32(&st, &format!("{p}attn.v_sconv.weight"))?,
        8 * 128,
        4,
    );

    let ep = "model.llm.layers.0.mlp.experts.";
    let w13 = tensor_u8(&st, &format!("{ep}w13_weight"))?;
    let s13 = tensor_u8(&st, &format!("{ep}w13_weight_scale"))?;
    let a13 = tensor_f32(&st, &format!("{ep}w13_weight_amax"))?;
    let w2 = tensor_u8(&st, &format!("{ep}w2_weight"))?;
    let s2 = tensor_u8(&st, &format!("{ep}w2_weight_scale"))?;
    let a2 = tensor_f32(&st, &format!("{ep}w2_weight_amax"))?;
    let (gate_w, up_w) = split_gate_up(&w13, 128, 2304, HIDDEN / 2);
    let (gate_s, up_s) = split_gate_up(&s13, 128, 2304, HIDDEN / 16);

    let weights = Layer0Weights {
        attn_norm: &attn_norm,
        mlp_norm: &mlp_norm,
        attn_sconv: &attn_sconv,
        mlp_sconv: &mlp_sconv,
        mlp_query: &mlp_query,
        router_w: &router_w,
        router_b: &router_b,
        attn: SlidingAttnWeights {
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
            k_sconv: &k_sconv,
            v_sconv: &v_sconv,
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
    };

    let pred = layer0_prefill(&x_in, tokens, &weights);
    let mut maxabs = 0f32;
    let mut sumabs = 0f32;
    for (p, r) in pred.iter().zip(x_ref.iter()) {
        let d = (p - r).abs();
        maxabs = maxabs.max(d);
        sumabs += d;
    }
    let meanabs = sumabs / pred.len() as f32;
    eprintln!("layer0 parity maxabs={maxabs} meanabs={meanabs}");
    assert!(maxabs < 1e-3, "layer0 f32 parity failed: maxabs={maxabs}");
    Ok(())
}
