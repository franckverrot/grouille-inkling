//! Stepped decode through sliding layers 0→1 vs prefill.

use anyhow::Result;
use grouille_inkling::cache::VisitCache;
use grouille_inkling::decoder::{layers_0_1, sliding_layer_decode_step};
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
fn layers_0_1_decode_matches_prefill() -> Result<()> {
    let file = std::fs::File::open(ckpt())?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    let st = SafeTensors::deserialize(&mmap)?;
    let b0 = SlidingLayerBufs::load(&st, 0, false)?;
    let b1 = SlidingLayerBufs::load(&st, 1, false)?;
    let w0 = b0.as_weights();
    let w1 = b1.as_weights();

    let tokens = 6usize;
    let mut x = vec![0f32; tokens * HIDDEN];
    for (i, v) in x.iter_mut().enumerate() {
        *v = ((i % 13) as f32) * 0.009 - 0.03;
    }
    let pref = layers_0_1(&x, tokens, &w0, &w1);

    let mut c0 = VisitCache::sliding();
    let mut c1 = VisitCache::sliding();
    let mut step = vec![0f32; tokens * HIDDEN];
    for t in 0..tokens {
        let mut blocks = Vec::new();
        let xt = &x[t * HIDDEN..(t + 1) * HIDDEN];
        let y0 = sliding_layer_decode_step(xt, &mut blocks, &w0, &mut c0)?;
        let y1 = sliding_layer_decode_step(&y0, &mut blocks, &w1, &mut c1)?;
        step[t * HIDDEN..(t + 1) * HIDDEN].copy_from_slice(&y1);
    }

    let mut maxabs = 0f32;
    for (a, b) in pref.iter().zip(step.iter()) {
        maxabs = maxabs.max((a - b).abs());
    }
    eprintln!("layers 0→1 decode vs prefill maxabs={maxabs}");
    assert!(maxabs < 1e-5, "layers01 decode maxabs={maxabs}");
    Ok(())
}
