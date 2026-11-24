//! Layer-6 decode with messages vs prefill msgs (one token at a time).

use anyhow::{bail, Context, Result};
use grouille_inkling::cache::VisitCache;
use grouille_inkling::decoder::{sliding_layer_decode_step_msgs, sliding_layer_prefill_msgs};
use grouille_inkling::loop_msg::MSG_SLOTS;
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

fn load_f32(path: &std::path::Path, key: &str) -> Result<Vec<f32>> {
    let f = std::fs::File::open(path)?;
    let mut npz = npyz::npz::NpzArchive::new(f)?;
    let arr = npz
        .by_name(key)?
        .with_context(|| format!("missing {key}"))?;
    Ok(arr.into_vec()?)
}

#[test]
fn layer6_msg_decode_matches_prefill() -> Result<()> {
    let golden = root().join("goldens/layer6_msg_f32.npz");
    if !golden.exists() {
        bail!("missing {}", golden.display());
    }
    let x_in = load_f32(&golden, "x_in")?;
    let messages = load_f32(&golden, "messages")?;
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
    let buf = SlidingLayerBufs::load(&st, 6, true)?;
    let w = buf.as_weights();

    let (pref, _) =
        sliding_layer_prefill_msgs(&x_in, tokens, &blocks, &w, Some(&messages));

    let mut cache = VisitCache::sliding();
    let mut step = vec![0f32; tokens * HIDDEN];
    for t in 0..tokens {
        let mut b = blocks[t].clone();
        let msg = &messages[t * MSG_SLOTS * HIDDEN..(t + 1) * MSG_SLOTS * HIDDEN];
        let y = sliding_layer_decode_step_msgs(
            &x_in[t * HIDDEN..(t + 1) * HIDDEN],
            &mut b,
            &w,
            &mut cache,
            Some(msg),
        )?;
        step[t * HIDDEN..(t + 1) * HIDDEN].copy_from_slice(&y);
    }

    let mut maxabs = 0f32;
    for (a, b) in pref.iter().zip(step.iter()) {
        maxabs = maxabs.max((a - b).abs());
    }
    eprintln!("layer6 msg decode vs prefill maxabs={maxabs}");
    // The two host paths add in a different order, so they can land a few float
    // steps apart.  A step is the gap between neighbouring f32 values at the
    // stream's magnitude.
    let scale = pref.iter().fold(0f32, |m, v| m.max(v.abs()));
    let step = f32::EPSILON * 2f32.powi(scale.log2().floor() as i32);
    assert!(
        maxabs <= 8.0 * step,
        "layer6 msg decode maxabs={maxabs} ({} float steps)",
        maxabs / step
    );
    Ok(())
}
