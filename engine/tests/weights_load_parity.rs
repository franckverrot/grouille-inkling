//! Shared weight loader vs layer-6 first-pass golden.

use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use grouille_inkling::decoder::sliding_layer_prefill;
use grouille_inkling::ops::HIDDEN;
use grouille_inkling::weights::SlidingLayerBufs;
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

#[test]
fn sliding_loader_layer6_matches_golden() -> Result<()> {
    let golden = root().join("goldens/layer06_f32.npz");
    if !golden.exists() {
        bail!("missing {}", golden.display());
    }
    let after_5 = load_npz_f32(&golden, "after_5")?;
    let block0 = load_npz_f32(&golden, "block0")?;
    let x_ref = load_npz_f32(&golden, "x_out")?;
    let tokens = after_5.len() / HIDDEN;
    let blocks: Vec<Vec<f32>> = (0..tokens)
        .map(|t| block0[t * HIDDEN..(t + 1) * HIDDEN].to_vec())
        .collect();

    let file = std::fs::File::open(ckpt())?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    let st = SafeTensors::deserialize(&mmap)?;
    let buf = SlidingLayerBufs::load(&st, 6, false)?;
    let w = buf.as_weights();
    let (pred, _) = sliding_layer_prefill(&after_5, tokens, &blocks, &w);
    let mut maxabs = 0f32;
    for (a, b) in pred.iter().zip(x_ref.iter()) {
        maxabs = maxabs.max((a - b).abs());
    }
    eprintln!("weights loader layer6 maxabs={maxabs}");
    assert!(maxabs < 1e-3);
    Ok(())
}
