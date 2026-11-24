//! A full-attention layer on the GPU prompt path vs the host reference and an f32 golden.

use anyhow::{bail, Context, Result};
use grouille_inkling::layer5::full_layer_prefill;
use grouille_inkling::metal_prompt::PromptLayer;
use grouille_inkling::visit::MetalAccel;
use grouille_inkling::ops::HIDDEN;
use grouille_inkling::weights::FullLayerBufs;
use safetensors::tensor::SafeTensors;
use std::path::PathBuf;

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

fn load_f32(path: &std::path::Path, key: &str) -> Result<Vec<f32>> {
    let f = std::fs::File::open(path)?;
    let mut npz = npyz::npz::NpzArchive::new(f)?;
    let arr = npz
        .by_name(key)?
        .with_context(|| format!("missing {key}"))?;
    Ok(arr.into_vec()?)
}

#[test]
fn layer5_gpu_matches_cpu() -> Result<()> {
    let golden = root().join("goldens/layer5_f32.npz");
    if !golden.exists() {
        bail!("missing {}", golden.display());
    }
    let x_in = load_f32(&golden, "x_in")?;
    let block = load_f32(&golden, "block")?;
    let tokens = x_in.len() / HIDDEN;
    let blocks: Vec<Vec<f32>> = (0..tokens)
        .map(|t| block[t * HIDDEN..(t + 1) * HIDDEN].to_vec())
        .collect();

    let file = std::fs::File::open(ckpt())?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    let st = SafeTensors::deserialize(&mmap)?;
    let buf = FullLayerBufs::load(&st, 5)?;
    let w = buf.as_weights();

    let (cpu, _) = full_layer_prefill(&x_in, tokens, &blocks, &w);
    let mut accel = MetalAccel::new()?;
    let (gpu, _) =
        accel.ops.prompt_layer_host(&mut accel.moe, &PromptLayer::full(&w), &x_in, tokens, &blocks)?;

    let mut maxabs = 0f32;
    for (a, b) in cpu.iter().zip(gpu.iter()) {
        maxabs = maxabs.max((a - b).abs());
    }
    eprintln!("layer5 gpu vs cpu maxabs={maxabs}");
    assert!(maxabs < 5e-3, "layer5 gpu maxabs={maxabs}");
    Ok(())
}

#[test]
fn layer23_gpu_matches_f32() -> Result<()> {
    let golden = root().join("goldens/layer23_postlude_f32.npz");
    if !golden.exists() {
        bail!("missing {}", golden.display());
    }
    let x_in = load_f32(&golden, "x_in")?;
    let x_ref = load_f32(&golden, "x_out")?;
    let blocks_flat = load_f32(&golden, "blocks")?;
    let tokens = x_in.len() / HIDDEN;
    let n_blocks = blocks_flat.len() / (tokens * HIDDEN);
    let blocks: Vec<Vec<f32>> = (0..tokens)
        .map(|t| {
            let mut s = Vec::with_capacity(n_blocks * HIDDEN);
            for b in 0..n_blocks {
                let off = t * n_blocks * HIDDEN + b * HIDDEN;
                s.extend_from_slice(&blocks_flat[off..off + HIDDEN]);
            }
            s
        })
        .collect();

    let file = std::fs::File::open(ckpt())?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    let st = SafeTensors::deserialize(&mmap)?;
    let buf = FullLayerBufs::load(&st, 23)?;
    let w = buf.as_weights();
    let mut accel = MetalAccel::new()?;
    let (pred, _) =
        accel.ops.prompt_layer_host(&mut accel.moe, &PromptLayer::full(&w), &x_in, tokens, &blocks)?;
    let mut maxabs = 0f32;
    for (a, b) in pred.iter().zip(x_ref.iter()) {
        maxabs = maxabs.max((a - b).abs());
    }
    eprintln!("layer23 gpu vs f32 maxabs={maxabs}");
    assert!(maxabs < 1e-3, "layer23 gpu maxabs={maxabs}");
    Ok(())
}
