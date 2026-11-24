//! Stepped sliding-layer decode vs prefill on layer 0.

use anyhow::Result;
use grouille_inkling::cache::VisitCache;
use grouille_inkling::decoder::{
    sliding_layer_decode_step, sliding_layer_prefill, sliding_layer_prefill_into_cache,
};
use grouille_inkling::ops::HIDDEN;
use grouille_inkling::weights::SlidingLayerBufs;
use safetensors::tensor::SafeTensors;
use std::path::PathBuf;

fn ckpt() -> PathBuf {
    PathBuf::from(std::env::var_os("HOME").unwrap())
        .join(".cache/huggingface/hub/models--thinkingmachines--Inkling-2-Mini")
        .join("snapshots/ed40e7da625f594fc9c4990f5f66be24873c22d7")
        .join("model-00001-of-00001.safetensors")
}

#[test]
fn layer0_decode_steps_match_prefill() -> Result<()> {
    let file = std::fs::File::open(ckpt())?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    let st = SafeTensors::deserialize(&mmap)?;
    let buf = SlidingLayerBufs::load(&st, 0, false)?;
    let w = buf.as_weights();

    let tokens = 8usize;
    let mut x = vec![0f32; tokens * HIDDEN];
    for (i, v) in x.iter_mut().enumerate() {
        *v = ((i % 17) as f32) * 0.008 - 0.04;
    }
    let empty: Vec<Vec<f32>> = (0..tokens).map(|_| Vec::new()).collect();
    let (pref, _) = sliding_layer_prefill(&x, tokens, &empty, &w);

    let mut cache = VisitCache::sliding();
    let mut step = vec![0f32; tokens * HIDDEN];
    for t in 0..tokens {
        // Block stacks are per-token; layer-0 entry is empty for each position.
        let mut blocks = Vec::new();
        let xt = &x[t * HIDDEN..(t + 1) * HIDDEN];
        let y = sliding_layer_decode_step(xt, &mut blocks, &w, &mut cache)?;
        step[t * HIDDEN..(t + 1) * HIDDEN].copy_from_slice(&y);
    }

    for t in 0..tokens {
        let mut m = 0f32;
        for i in 0..HIDDEN {
            m = m.max((pref[t * HIDDEN + i] - step[t * HIDDEN + i]).abs());
        }
        eprintln!("token {t} maxabs={m}");
    }
    let mut maxabs = 0f32;
    for (a, b) in pref.iter().zip(step.iter()) {
        maxabs = maxabs.max((a - b).abs());
    }
    eprintln!("layer0 decode vs prefill maxabs={maxabs}");
    assert!(maxabs < 1e-5, "layer0 decode maxabs={maxabs}");
    Ok(())
}

#[test]
fn layer0_prefill_into_cache_then_decode_matches() -> Result<()> {
    let file = std::fs::File::open(ckpt())?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    let st = SafeTensors::deserialize(&mmap)?;
    let buf = SlidingLayerBufs::load(&st, 0, false)?;
    let w = buf.as_weights();

    let tokens = 6usize;
    let mut x = vec![0f32; tokens * HIDDEN];
    for (i, v) in x.iter_mut().enumerate() {
        *v = ((i % 17) as f32) * 0.008 - 0.04;
    }
    let empty: Vec<Vec<f32>> = (0..tokens).map(|_| Vec::new()).collect();
    let (pref, _) = sliding_layer_prefill(&x, tokens, &empty, &w);

    let warm = tokens - 1;
    let mut cache = VisitCache::sliding();
    let empty_w: Vec<Vec<f32>> = (0..warm).map(|_| Vec::new()).collect();
    sliding_layer_prefill_into_cache(&x[..warm * HIDDEN], warm, &empty_w, &w, &mut cache)?;

    let mut blocks = Vec::new();
    let xt = &x[warm * HIDDEN..(warm + 1) * HIDDEN];
    let y = sliding_layer_decode_step(xt, &mut blocks, &w, &mut cache)?;

    let mut maxabs = 0f32;
    for i in 0..HIDDEN {
        maxabs = maxabs.max((pref[warm * HIDDEN + i] - y[i]).abs());
    }
    eprintln!("layer0 prefill-cache→decode vs prefill maxabs={maxabs}");
    assert!(maxabs < 1e-5, "layer0 prefill-cache decode maxabs={maxabs}");
    Ok(())
}
