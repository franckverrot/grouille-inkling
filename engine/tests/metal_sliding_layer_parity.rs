//! A sliding layer on the GPU prompt path vs the host reference and an f32 golden.

use anyhow::{bail, Context, Result};
use grouille_inkling::decoder::sliding_layer_prefill;
use grouille_inkling::metal_prompt::PromptLayer;
use grouille_inkling::visit::MetalAccel;
use grouille_inkling::ops::HIDDEN;
use grouille_inkling::weights::SlidingLayerBufs;
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

#[test]
fn layer0_gpu_matches_cpu() -> Result<()> {
    let file = std::fs::File::open(ckpt())?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    let st = SafeTensors::deserialize(&mmap)?;
    let buf = SlidingLayerBufs::load(&st, 0, false)?;
    let w = buf.as_weights();

    let tokens = 14usize;
    let mut x = vec![0f32; tokens * HIDDEN];
    for (i, v) in x.iter_mut().enumerate() {
        *v = ((i % 29) as f32) * 0.007 - 0.05;
    }
    let empty: Vec<Vec<f32>> = (0..tokens).map(|_| Vec::new()).collect();

    let (cpu, cpu_blocks) = sliding_layer_prefill(&x, tokens, &empty, &w);
    let mut accel = MetalAccel::new()?;
    let (gpu, gpu_blocks) =
        accel.ops.prompt_layer_host(&mut accel.moe, &PromptLayer::sliding(&w), &x, tokens, &empty)?;

    let mut maxabs = 0f32;
    for (a, b) in cpu.iter().zip(gpu.iter()) {
        maxabs = maxabs.max((a - b).abs());
    }
    eprintln!("layer0 gpu vs cpu maxabs={maxabs}");
    assert!(maxabs < 5e-3, "layer0 gpu maxabs={maxabs}");
    // Layer 0 opens a block: the stream it came in with is now on every token's stack.
    assert_eq!(cpu_blocks, gpu_blocks);
    Ok(())
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
fn layer20_gpu_matches_f32() -> Result<()> {
    let golden = root().join("goldens/layer20_postlude_f32.npz");
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
    let buf = SlidingLayerBufs::load(&st, 20, false)?;
    let w = buf.as_weights();
    let mut accel = MetalAccel::new()?;
    let (pred, _) =
        accel.ops.prompt_layer_host(&mut accel.moe, &PromptLayer::sliding(&w), &x_in, tokens, &blocks)?;
    let mut maxabs = 0f32;
    for (a, b) in pred.iter().zip(x_ref.iter()) {
        maxabs = maxabs.max((a - b).abs());
    }
    eprintln!("layer20 gpu vs f32 maxabs={maxabs}");
    assert!(maxabs < 1e-3, "layer20 gpu maxabs={maxabs}");
    Ok(())
}
