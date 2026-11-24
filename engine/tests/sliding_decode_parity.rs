//! Stepped sliding-attn decode vs prefill last-token output.

use anyhow::Result;
use grouille_inkling::attn::{
    sliding_attention, sliding_attention_decode_step, sliding_attention_qkv_prefill_cache,
};
use grouille_inkling::cache::VisitCache;
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
fn sliding_decode_steps_match_prefill() -> Result<()> {
    let file = std::fs::File::open(ckpt())?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    let st = SafeTensors::deserialize(&mmap)?;
    let buf = SlidingLayerBufs::load(&st, 0, false)?;
    let w = buf.as_weights().attn;

    let tokens = 14usize;
    let mut x = vec![0f32; tokens * HIDDEN];
    for (i, v) in x.iter_mut().enumerate() {
        *v = ((i % 19) as f32) * 0.01 - 0.08;
    }

    let mut pref = vec![0f32; tokens * HIDDEN];
    sliding_attention(&x, tokens, &w, &mut pref);

    let mut cache = VisitCache::sliding();
    let mut step_out = vec![0f32; tokens * HIDDEN];
    let mut y = vec![0f32; HIDDEN];
    for t in 0..tokens {
        let xt = &x[t * HIDDEN..(t + 1) * HIDDEN];
        sliding_attention_decode_step(xt, xt, xt, &w, &mut cache, &mut y)?;
        step_out[t * HIDDEN..(t + 1) * HIDDEN].copy_from_slice(&y);
    }

    let mut maxabs = 0f32;
    for (a, b) in pref.iter().zip(step_out.iter()) {
        maxabs = maxabs.max((a - b).abs());
    }
    eprintln!("sliding decode vs prefill maxabs={maxabs}");
    assert!(maxabs < 1e-5, "decode vs prefill maxabs={maxabs}");
    Ok(())
}

/// Batch-prefill first tokens into VisitCache, then decode the next token.
#[test]
fn sliding_prefill_cache_then_decode_matches_prefill() -> Result<()> {
    let file = std::fs::File::open(ckpt())?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    let st = SafeTensors::deserialize(&mmap)?;
    let buf = SlidingLayerBufs::load(&st, 0, false)?;
    let w = buf.as_weights().attn;

    let tokens = 8usize;
    let mut x = vec![0f32; tokens * HIDDEN];
    for (i, v) in x.iter_mut().enumerate() {
        *v = ((i % 19) as f32) * 0.01 - 0.08;
    }

    let mut pref = vec![0f32; tokens * HIDDEN];
    sliding_attention(&x, tokens, &w, &mut pref);

    let warm = tokens - 1;
    let mut cache = VisitCache::sliding();
    let mut warm_out = vec![0f32; warm * HIDDEN];
    sliding_attention_qkv_prefill_cache(
        &x[..warm * HIDDEN],
        &x[..warm * HIDDEN],
        &x[..warm * HIDDEN],
        warm,
        &w,
        &mut warm_out,
        &mut cache,
    );
    assert_eq!(cache.kv.len, warm.min(cache.kv.cap));

    let mut y = vec![0f32; HIDDEN];
    let xt = &x[warm * HIDDEN..(warm + 1) * HIDDEN];
    sliding_attention_decode_step(xt, xt, xt, &w, &mut cache, &mut y)?;

    let mut maxabs = 0f32;
    for i in 0..HIDDEN {
        maxabs = maxabs.max((pref[warm * HIDDEN + i] - y[i]).abs());
    }
    eprintln!("sliding prefill-cache→decode vs prefill maxabs={maxabs}");
    assert!(maxabs < 1e-5, "prefill-cache decode maxabs={maxabs}");
    Ok(())
}
