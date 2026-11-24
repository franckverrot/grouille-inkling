//! Full-layer prefill-into-cache then decode vs full prefill.

use anyhow::Result;
use grouille_inkling::cache::VisitCache;
use grouille_inkling::layer5::{
    full_layer_decode_step, full_layer_prefill, full_layer_prefill_into_cache,
};
use grouille_inkling::ops::HIDDEN;
use grouille_inkling::weights::FullLayerBufs;
use safetensors::tensor::SafeTensors;
use std::path::PathBuf;

fn ckpt() -> PathBuf {
    PathBuf::from(std::env::var_os("HOME").unwrap())
        .join(".cache/huggingface/hub/models--thinkingmachines--Inkling-2-Mini")
        .join("snapshots/ed40e7da625f594fc9c4990f5f66be24873c22d7")
        .join("model-00001-of-00001.safetensors")
}

#[test]
fn layer5_prefill_into_cache_then_decode_matches() -> Result<()> {
    let file = std::fs::File::open(ckpt())?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    let st = SafeTensors::deserialize(&mmap)?;
    let buf = FullLayerBufs::load(&st, 5)?;
    let w = buf.as_weights();

    let tokens = 4usize;
    let mut x = vec![0f32; tokens * HIDDEN];
    for (i, v) in x.iter_mut().enumerate() {
        *v = ((i % 13) as f32) * 0.01 - 0.03;
    }
    // Synthetic prior-block stacks (same for prefill and decode entry).
    let blocks: Vec<Vec<f32>> = (0..tokens)
        .map(|t| {
            let mut b = vec![0f32; HIDDEN];
            for i in 0..HIDDEN {
                b[i] = ((t * 3 + i) % 9) as f32 * 0.01;
            }
            b
        })
        .collect();

    let (pref, _) = full_layer_prefill(&x, tokens, &blocks, &w);

    let warm = tokens - 1;
    let mut cache = VisitCache::full(tokens + 4);
    full_layer_prefill_into_cache(
        &x[..warm * HIDDEN],
        warm,
        &blocks[..warm],
        &w,
        &mut cache,
    )?;

    let mut b = blocks[warm].clone();
    let y = full_layer_decode_step(
        &x[warm * HIDDEN..(warm + 1) * HIDDEN],
        &mut b,
        &w,
        &mut cache,
    )?;

    let mut maxabs = 0f32;
    for i in 0..HIDDEN {
        maxabs = maxabs.max((pref[warm * HIDDEN + i] - y[i]).abs());
    }
    eprintln!("layer5 prefill-cache→decode vs prefill maxabs={maxabs}");
    // The two host paths add in a different order, so they can land a few float
    // steps apart.  A step is the gap between neighbouring f32 values at the
    // stream's magnitude.
    let scale = pref[warm * HIDDEN..].iter().fold(0f32, |m, v| m.max(v.abs()));
    let step = f32::EPSILON * 2f32.powi(scale.log2().floor() as i32);
    assert!(
        maxabs <= 8.0 * step,
        "layer5 prefill-cache decode maxabs={maxabs} ({} float steps)",
        maxabs / step
    );
    Ok(())
}
