//! First-core tail layers 12→17 after layer 11.

use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use grouille_inkling::decoder::sliding_layer_prefill;
use grouille_inkling::layer5::full_layer_prefill;
use grouille_inkling::loop_sched::rebuild_message_bank;
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
fn core_layers_12_to_17_match_f32() -> Result<()> {
    let golden = root().join("goldens/core_12_17_f32.npz");
    if !golden.exists() {
        bail!(
            "missing {}; run python3 spike/export_core_12_17_f32.py",
            golden.display()
        );
    }
    let x_in = load_npz_f32(&golden, "x_in")?;
    let blocks_in = load_npz_f32(&golden, "blocks_in")?;
    let x_ref = load_npz_f32(&golden, "x_out")?;
    let blocks_out_ref = load_npz_f32(&golden, "blocks_out")?;
    let tokens = x_in.len() / HIDDEN;
    let n_in = blocks_in.len() / (tokens * HIDDEN);
    let mut blocks: Vec<Vec<f32>> = (0..tokens)
        .map(|t| {
            let mut s = Vec::with_capacity(n_in * HIDDEN);
            for b in 0..n_in {
                let off = t * n_in * HIDDEN + b * HIDDEN;
                s.extend_from_slice(&blocks_in[off..off + HIDDEN]);
            }
            s
        })
        .collect();

    let file = std::fs::File::open(ckpt())?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    let st = SafeTensors::deserialize(&mmap)?;
    let s12 = SlidingLayerBufs::load(&st, 12, false)?;
    let s13 = SlidingLayerBufs::load(&st, 13, false)?;
    let s14 = SlidingLayerBufs::load(&st, 14, false)?;
    let s15 = SlidingLayerBufs::load(&st, 15, false)?;
    let s16 = SlidingLayerBufs::load(&st, 16, false)?;
    let f17 = FullLayerBufs::load(&st, 17)?;

    let mut x = x_in;
    for w in [
        s12.as_weights(),
        s13.as_weights(),
        s14.as_weights(),
        s15.as_weights(),
        s16.as_weights(),
    ] {
        let (nx, nb) = sliding_layer_prefill(&x, tokens, &blocks, &w);
        x = nx;
        blocks = nb;
    }
    let (pred, next_blocks) = full_layer_prefill(&x, tokens, &blocks, &f17.as_weights());
    assert_eq!(next_blocks[0].len() / HIDDEN, 3); // +L12 boundary

    let mut maxabs = 0f32;
    for (a, b) in pred.iter().zip(x_ref.iter()) {
        maxabs = maxabs.max((a - b).abs());
    }
    eprintln!("core 12→17 parity maxabs={maxabs}");
    assert!(maxabs < 1e-3, "core 12→17 maxabs={maxabs}");

    // bank at reentry: prelude_n=1
    let bank = rebuild_message_bank(&next_blocks[0], 1, &pred[..HIDDEN]);
    assert_eq!(bank.len() / HIDDEN, 2);
    let _ = blocks_out_ref;
    Ok(())
}
