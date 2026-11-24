//! Layer-6 second-pass (with messages) f32 parity vs goldens/layer6_pass3_msg_f32.npz.

use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use grouille_inkling::attn::SlidingAttnWeights;
use grouille_inkling::decoder::{sliding_layer_prefill_msgs, SlidingLayerWeights};
use grouille_inkling::loop_msg::{MessageControllerWeights, MSG_SLOTS};
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

fn load_ctrl(
    st: &SafeTensors,
    layer: usize,
    name: &str,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    let p = format!("model.llm.layers.{layer}.controllers.{name}.");
    Ok((
        bf16_f32(st, &format!("{p}down_weight"))?,
        bf16_f32(st, &format!("{p}forget_weight"))?,
        bf16_f32(st, &format!("{p}gain_weight"))?,
    ))
}

#[test]
fn layer6_pass3_msg_matches_f32_golden() -> Result<()> {
    let golden = root().join("goldens/layer6_pass3_msg_f32.npz");
    if !golden.exists() {
        bail!(
            "missing {}; run spike/export_layer6_pass3_msg_f32.py",
            golden.display()
        );
    }
    let x_in = load_npz_f32(&golden, "x_in")?;
    let x_ref = load_npz_f32(&golden, "x_out")?;
    let messages = load_npz_f32(&golden, "messages")?;
    let blocks_flat = load_npz_f32(&golden, "blocks")?;
    let tokens = x_in.len() / HIDDEN;
    assert_eq!(messages.len(), tokens * MSG_SLOTS * HIDDEN);
    let n_blocks = blocks_flat.len() / (tokens * HIDDEN);
    let blocks: Vec<Vec<f32>> = (0..tokens)
        .map(|t| {
            let mut s = Vec::with_capacity(n_blocks * HIDDEN);
            for b in 0..n_blocks {
                // npz layout [T, n, H] row-major
                let off = t * n_blocks * HIDDEN + b * HIDDEN;
                s.extend_from_slice(&blocks_flat[off..off + HIDDEN]);
            }
            s
        })
        .collect();

    let file = std::fs::File::open(ckpt())?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    let st = SafeTensors::deserialize(&mmap)?;
    let p = "model.llm.layers.6.";
    let ep = format!("{p}mlp.experts.");
    let w13 = u8s(&st, &format!("{ep}w13_weight"))?;
    let s13 = u8s(&st, &format!("{ep}w13_weight_scale"))?;
    let (gate_w, up_w) = split_gate_up(&w13, 128, 2304, HIDDEN / 2);
    let (gate_s, up_s) = split_gate_up(&s13, 128, 2304, HIDDEN / 16);

    let (attn_d, attn_f, attn_g) = load_ctrl(&st, 6, "attn")?;
    let (k_d, k_f, k_g) = load_ctrl(&st, 6, "k")?;
    let (v_d, v_f, v_g) = load_ctrl(&st, 6, "v")?;
    let (mlp_d, mlp_f, mlp_g) = load_ctrl(&st, 6, "mlp")?;

    let attn_norm = bf16_f32(&st, &format!("{p}attn_norm.weight"))?;
    let mlp_norm = bf16_f32(&st, &format!("{p}mlp_norm.weight"))?;
    let attn_sconv = taps(&bf16_f32(&st, &format!("{p}attn_sconv.weight"))?);
    let mlp_sconv = taps(&bf16_f32(&st, &format!("{p}mlp_sconv.weight"))?);
    let attn_query = f32s(&st, "model.llm.attn_res.attn_queries.5")?;
    let mlp_query = f32s(&st, "model.llm.attn_res.mlp_queries.6")?;
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
    let k_sconv = taps(&bf16_f32(&st, &format!("{p}attn.k_sconv.weight"))?);
    let v_sconv = taps(&bf16_f32(&st, &format!("{p}attn.v_sconv.weight"))?);
    let down_w = u8s(&st, &format!("{ep}w2_weight"))?;
    let down_s = u8s(&st, &format!("{ep}w2_weight_scale"))?;
    let gate_up_amax = f32s(&st, &format!("{ep}w13_weight_amax"))?;
    let down_amax = f32s(&st, &format!("{ep}w2_weight_amax"))?;

    let w = SlidingLayerWeights {
        attn_norm: &attn_norm,
        mlp_norm: &mlp_norm,
        attn_sconv: &attn_sconv,
        mlp_sconv: &mlp_sconv,
        attn_query: Some(&attn_query),
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
            down_w: &down_w,
            down_s: &down_s,
            gate_up_amax: &gate_up_amax,
            down_amax: &down_amax,
            n_experts: 128,
            mxfp8: false,
            layer: None,
        },
        is_block_boundary: true,
        ctrl_attn: Some(MessageControllerWeights {
            down: &attn_d,
            forget: &attn_f,
            gain: &attn_g,
        }),
        ctrl_k: Some(MessageControllerWeights {
            down: &k_d,
            forget: &k_f,
            gain: &k_g,
        }),
        ctrl_v: Some(MessageControllerWeights {
            down: &v_d,
            forget: &v_f,
            gain: &v_g,
        }),
        ctrl_mlp: Some(MessageControllerWeights {
            down: &mlp_d,
            forget: &mlp_f,
            gain: &mlp_g,
        }),
    };

    let (pred, _) = sliding_layer_prefill_msgs(&x_in, tokens, &blocks, &w, Some(&messages));
    let mut maxabs = 0f32;
    for (a, b) in pred.iter().zip(x_ref.iter()) {
        maxabs = maxabs.max((a - b).abs());
    }
    eprintln!("layer6_pass3_msg parity maxabs={maxabs}");
    assert!(maxabs < 1e-3, "layer6_msg maxabs={maxabs}");
    Ok(())
}
