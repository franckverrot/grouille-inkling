//! Layer-0 MoE parity: Rust NVFP4 path vs `goldens/layer0.npz::moe_out_f32`.

use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use grouille_inkling::moe::{sparse_moe_layer0, ExpertPack, HIDDEN, INTER, TOP_K};
use safetensors::tensor::SafeTensors;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("engine/")
        .to_path_buf()
}

fn default_checkpoint() -> PathBuf {
    dirs_next_home()
        .join(".cache/huggingface/hub/models--thinkingmachines--Inkling-2-Mini")
        .join("snapshots/ed40e7da625f594fc9c4990f5f66be24873c22d7")
        .join("model-00001-of-00001.safetensors")
}

fn dirs_next_home() -> PathBuf {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .expect("HOME")
}

fn load_npz_f32(path: &std::path::Path, key: &str) -> Result<(Vec<f32>, Vec<usize>)> {
    let f = std::fs::File::open(path)?;
    let mut npz = npyz::npz::NpzArchive::new(f)?;
    let arr = npz
        .by_name(key)?
        .with_context(|| format!("missing {key}"))?;
    let shape: Vec<usize> = arr.shape().iter().map(|d| *d as usize).collect();
    let data: Vec<f32> = arr.into_vec()?;
    Ok((data, shape))
}

fn load_npz_i32(path: &std::path::Path, key: &str) -> Result<(Vec<i32>, Vec<usize>)> {
    let f = std::fs::File::open(path)?;
    let mut npz = npyz::npz::NpzArchive::new(f)?;
    let arr = npz
        .by_name(key)?
        .with_context(|| format!("missing {key}"))?;
    let shape: Vec<usize> = arr.shape().iter().map(|d| *d as usize).collect();
    let data: Vec<i32> = arr.into_vec()?;
    Ok((data, shape))
}

fn tensor_u8(st: &SafeTensors, name: &str) -> Result<Vec<u8>> {
    let t = st.tensor(name)?;
    Ok(t.data().to_vec())
}

fn tensor_f32(st: &SafeTensors, name: &str) -> Result<Vec<f32>> {
    let t = st.tensor(name)?;
    let bytes = t.data();
    let mut out = vec![0f32; bytes.len() / 4];
    for (i, chunk) in bytes.chunks_exact(4).enumerate() {
        out[i] = f32::from_le_bytes(chunk.try_into().unwrap());
    }
    Ok(out)
}

/// Split fused `[E, 2*INTER, K]` into gate/up with interleaved rows (MLX `_split_gate_up`).
fn split_gate_up(fused: &[u8], n_experts: usize, rows: usize, row_stride: usize) -> (Vec<u8>, Vec<u8>) {
    assert_eq!(fused.len(), n_experts * rows * row_stride);
    assert_eq!(rows, 2 * INTER);
    let mut gate = vec![0u8; n_experts * INTER * row_stride];
    let mut up = vec![0u8; n_experts * INTER * row_stride];
    for e in 0..n_experts {
        for i in 0..INTER {
            let src_g = ((e * rows) + (i * 2)) * row_stride;
            let src_u = ((e * rows) + (i * 2 + 1)) * row_stride;
            let dst = (e * INTER + i) * row_stride;
            gate[dst..dst + row_stride].copy_from_slice(&fused[src_g..src_g + row_stride]);
            up[dst..dst + row_stride].copy_from_slice(&fused[src_u..src_u + row_stride]);
        }
    }
    (gate, up)
}

#[test]
fn moe_layer0_matches_f32_golden() -> Result<()> {
    let root = repo_root();
    let golden = root.join("goldens/layer0.npz");
    let (moe_in, in_shape) = load_npz_f32(&golden, "moe_in")?;
    let (moe_ref, _) = load_npz_f32(&golden, "moe_out_f32")?;
    let (indices, _) = load_npz_i32(&golden, "route_indices")?;
    let (weights, _) = load_npz_f32(&golden, "route_weights")?;
    assert_eq!(in_shape, vec![1, 14, HIDDEN]);
    let tokens = 14;
    // Drop batch dim.
    let indices: Vec<i32> = indices;
    let weights: Vec<f32> = weights;

    let ckpt = default_checkpoint();
    if !ckpt.exists() {
        bail!("missing checkpoint at {}", ckpt.display());
    }
    let file = std::fs::File::open(&ckpt)?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    let st = SafeTensors::deserialize(&mmap)?;

    let prefix = "model.llm.layers.0.mlp.experts.";
    let w13 = tensor_u8(&st, &format!("{prefix}w13_weight"))?;
    let s13 = tensor_u8(&st, &format!("{prefix}w13_weight_scale"))?;
    let a13 = tensor_f32(&st, &format!("{prefix}w13_weight_amax"))?;
    let w2 = tensor_u8(&st, &format!("{prefix}w2_weight"))?;
    let s2 = tensor_u8(&st, &format!("{prefix}w2_weight_scale"))?;
    let a2 = tensor_f32(&st, &format!("{prefix}w2_weight_amax"))?;

    let n_experts = 128;
    assert_eq!(w13.len(), n_experts * 2304 * 1152);
    let (gate_w, up_w) = split_gate_up(&w13, n_experts, 2304, HIDDEN / 2);
    let (gate_s, up_s) = split_gate_up(&s13, n_experts, 2304, HIDDEN / 16);

    let pack = ExpertPack {
        gate_w: &gate_w,
        gate_s: &gate_s,
        up_w: &up_w,
        up_s: &up_s,
        down_w: &w2,
        down_s: &s2,
        gate_up_amax: &a13,
        down_amax: &a2,
        n_experts,
        mxfp8: false,
        layer: None,
    };

    let pred = sparse_moe_layer0(&moe_in, &indices, &weights, tokens, &pack);
    assert_eq!(pred.len(), moe_ref.len());

    let mut maxabs = 0f32;
    let mut sumabs = 0f32;
    for (p, r) in pred.iter().zip(moe_ref.iter()) {
        let d = (p - r).abs();
        maxabs = maxabs.max(d);
        sumabs += d;
    }
    let meanabs = sumabs / pred.len() as f32;
    eprintln!("moe parity maxabs={maxabs} meanabs={meanabs}");
    assert!(
        maxabs < 1e-4,
        "MoE f32 parity failed: maxabs={maxabs} (expected bit-exact-ish vs Python CPU ref)"
    );
    // silence unused
    let _ = TOP_K;
    let _ = INTER;
    Ok(())
}
