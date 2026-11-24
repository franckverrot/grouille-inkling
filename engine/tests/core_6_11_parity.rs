//! First-core fragment layers 7→11 after layer-6 boundary.

use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use grouille_inkling::decoder::sliding_layer_prefill;
use grouille_inkling::layer5::full_layer_prefill;
use grouille_inkling::ops::HIDDEN;
use grouille_inkling::weights::{FullLayerBufs, SlidingLayerBufs};
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
fn core_layers_7_to_11_match_f32() -> Result<()> {
    let golden = root().join("goldens/core_6_11_f32.npz");
    if !golden.exists() {
        bail!(
            "missing {}; run python3 spike/export_core_first_f32.py",
            golden.display()
        );
    }
    let x_in = load_npz_f32(&golden, "x_in")?;
    let block0 = load_npz_f32(&golden, "block0")?;
    let block1 = load_npz_f32(&golden, "block1")?;
    let x_ref = load_npz_f32(&golden, "x_out")?;
    let tokens = x_in.len() / HIDDEN;
    let mut blocks: Vec<Vec<f32>> = (0..tokens)
        .map(|t| {
            let mut s = Vec::with_capacity(2 * HIDDEN);
            s.extend_from_slice(&block0[t * HIDDEN..(t + 1) * HIDDEN]);
            s.extend_from_slice(&block1[t * HIDDEN..(t + 1) * HIDDEN]);
            s
        })
        .collect();

    let file = std::fs::File::open(ckpt())?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    let st = SafeTensors::deserialize(&mmap)?;
    let s7 = SlidingLayerBufs::load(&st, 7, false)?;
    let s8 = SlidingLayerBufs::load(&st, 8, false)?;
    let s9 = SlidingLayerBufs::load(&st, 9, false)?;
    let s10 = SlidingLayerBufs::load(&st, 10, false)?;
    let f11 = FullLayerBufs::load(&st, 11)?;

    let mut x = x_in;
    for w in [
        s7.as_weights(),
        s8.as_weights(),
        s9.as_weights(),
        s10.as_weights(),
    ] {
        let (nx, nb) = sliding_layer_prefill(&x, tokens, &blocks, &w);
        x = nx;
        blocks = nb;
    }
    let (pred, next_blocks) = full_layer_prefill(&x, tokens, &blocks, &f11.as_weights());
    assert_eq!(next_blocks[0].len() / HIDDEN, 2); // L11 not boundary

    let mut maxabs = 0f32;
    for (a, b) in pred.iter().zip(x_ref.iter()) {
        maxabs = maxabs.max((a - b).abs());
    }
    eprintln!("core 7→11 parity maxabs={maxabs}");
    assert!(maxabs < 1e-3, "core 7→11 maxabs={maxabs}");
    Ok(())
}
