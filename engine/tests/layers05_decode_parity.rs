//! Stepped decode through layers 0-5 vs prefill `layers_0_5`.

use anyhow::Result;
use grouille_inkling::cache::VisitCache;
use grouille_inkling::decoder::{layers_0_5, sliding_layer_decode_step};
use grouille_inkling::layer5::full_layer_decode_step;
use grouille_inkling::ops::HIDDEN;
use grouille_inkling::weights::{FullLayerBufs, SlidingLayerBufs};
use safetensors::tensor::SafeTensors;
use std::path::PathBuf;

fn ckpt() -> PathBuf {
    PathBuf::from(std::env::var_os("HOME").unwrap())
        .join(".cache/huggingface/hub/models--thinkingmachines--Inkling-2-Mini")
        .join("snapshots/ed40e7da625f594fc9c4990f5f66be24873c22d7")
        .join("model-00001-of-00001.safetensors")
}

#[test]
fn layers_0_5_decode_matches_prefill() -> Result<()> {
    let file = std::fs::File::open(ckpt())?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    let st = SafeTensors::deserialize(&mmap)?;
    let s0 = SlidingLayerBufs::load(&st, 0, false)?;
    let s1 = SlidingLayerBufs::load(&st, 1, false)?;
    let s2 = SlidingLayerBufs::load(&st, 2, false)?;
    let s3 = SlidingLayerBufs::load(&st, 3, false)?;
    let s4 = SlidingLayerBufs::load(&st, 4, false)?;
    let f5 = FullLayerBufs::load(&st, 5)?;
    let w0 = s0.as_weights();
    let w1 = s1.as_weights();
    let w2 = s2.as_weights();
    let w3 = s3.as_weights();
    let w4 = s4.as_weights();
    let wf = f5.as_weights();
    let sliding = [&w0, &w1, &w2, &w3, &w4];

    let tokens = 4usize;
    let mut x = vec![0f32; tokens * HIDDEN];
    for (i, v) in x.iter_mut().enumerate() {
        *v = ((i % 11) as f32) * 0.01 - 0.02;
    }
    let pref = layers_0_5(&x, tokens, &sliding, &wf);

    let mut caches: Vec<VisitCache> = (0..5).map(|_| VisitCache::sliding()).collect();
    caches.push(VisitCache::full(tokens + 4));
    let mut step = vec![0f32; tokens * HIDDEN];
    for t in 0..tokens {
        let mut blocks = Vec::new();
        let mut h = x[t * HIDDEN..(t + 1) * HIDDEN].to_vec();
        for (i, w) in sliding.iter().enumerate() {
            h = sliding_layer_decode_step(&h, &mut blocks, w, &mut caches[i])?.to_vec();
        }
        let y = full_layer_decode_step(&h, &mut blocks, &wf, &mut caches[5])?;
        step[t * HIDDEN..(t + 1) * HIDDEN].copy_from_slice(&y);
    }

    let mut maxabs = 0f32;
    for (a, b) in pref.iter().zip(step.iter()) {
        maxabs = maxabs.max((a - b).abs());
    }
    eprintln!("layers 0→5 decode vs prefill maxabs={maxabs}");
    // The two host paths add in a different order, so they can land a few float
    // steps apart.  A step is the gap between neighbouring f32 values at the
    // stream's magnitude.
    let scale = pref.iter().fold(0f32, |m, v| m.max(v.abs()));
    let step = f32::EPSILON * 2f32.powi(scale.log2().floor() as i32);
    assert!(
        maxabs <= 8.0 * step,
        "layers05 decode maxabs={maxabs} ({} float steps)",
        maxabs / step
    );
    Ok(())
}
