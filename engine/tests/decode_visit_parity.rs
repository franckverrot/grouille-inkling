//! Decode visit runner vs prefill / CPU (T=1 and stepped multi-token).

use anyhow::Result;
use grouille_inkling::cache::DecodeCaches;
use grouille_inkling::decode_visit::{
    run_decode_visits_cached, run_decode_visits_metal_cached, DecodeTokenState,
};
use grouille_inkling::loop_sched::layer_indices;
use grouille_inkling::ops::HIDDEN;
use grouille_inkling::visit::{
    run_visits_cached, run_visits_into_caches, run_visits_into_caches_metal, MetalAccel,
    VisitState, WeightCache,
};
use safetensors::tensor::SafeTensors;
use std::path::PathBuf;

fn ckpt() -> PathBuf {
    PathBuf::from(std::env::var_os("HOME").unwrap())
        .join(".cache/huggingface/hub/models--thinkingmachines--Inkling-2-Mini")
        .join("snapshots/ed40e7da625f594fc9c4990f5f66be24873c22d7")
        .join("model-00001-of-00001.safetensors")
}

fn seed_x() -> [f32; HIDDEN] {
    let mut x = [0f32; HIDDEN];
    for (i, v) in x.iter_mut().enumerate() {
        *v = ((i % 17) as f32) * 0.01 - 0.05;
    }
    x
}

fn assert_match(pref: &VisitState, dec: &DecodeTokenState, label: &str) {
    let mut maxabs = 0f32;
    for i in 0..HIDDEN {
        maxabs = maxabs.max((pref.x[i] - dec.x[i]).abs());
    }
    eprintln!("decode visit T=1 {label} vs prefill maxabs={maxabs}");
    // Accelerate vs scalar CPU can drift ~2e-5 over long visit chains.
    assert!(maxabs < 3e-4, "decode visit {label} maxabs={maxabs}");
}

#[test]
fn decode_visit_t1_through_l6_matches_prefill() -> Result<()> {
    let file = std::fs::File::open(ckpt())?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    let st = SafeTensors::deserialize(&mmap)?;
    let x = seed_x();
    let mut wcache = WeightCache::new();

    let mut pref = VisitState::new(x.to_vec());
    run_visits_cached(&st, &mut pref, 0, 7, &mut wcache)?;

    let indices = layer_indices();
    let mut caches = DecodeCaches::for_schedule(indices, 16);
    let mut dec = DecodeTokenState::new(x);
    run_decode_visits_cached(&st, &mut dec, &mut caches, 0, 7, &mut wcache)?;

    assert_match(&pref, &dec, "→L6");
    Ok(())
}

/// Through first core reentry (visit 18 = second pass L6) so message bank is live.
#[test]
fn decode_visit_t1_through_msg_reentry_matches_prefill() -> Result<()> {
    let file = std::fs::File::open(ckpt())?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    let st = SafeTensors::deserialize(&mmap)?;
    let x = seed_x();
    let mut wcache = WeightCache::new();

    // Visits 0..19: prelude + first core pass + second-pass L6 (messages on).
    let end = 19;
    let mut pref = VisitState::new(x.to_vec());
    run_visits_cached(&st, &mut pref, 0, end, &mut wcache)?;

    let indices = layer_indices();
    let mut caches = DecodeCaches::for_schedule(indices, 16);
    let mut dec = DecodeTokenState::new(x);
    run_decode_visits_cached(&st, &mut dec, &mut caches, 0, end, &mut wcache)?;

    assert_match(&pref, &dec, "→msg reentry L6");
    assert!(dec.bank.is_some(), "message bank should be set after reentry");
    Ok(())
}

/// Full 48-visit schedule (prelude + 3 core passes + postlude).
#[test]
fn decode_visit_t1_full_schedule_matches_prefill() -> Result<()> {
    let file = std::fs::File::open(ckpt())?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    let st = SafeTensors::deserialize(&mmap)?;
    let x = seed_x();
    let mut wcache = WeightCache::new();

    let indices = layer_indices();
    let end = indices.len();
    let mut pref = VisitState::new(x.to_vec());
    run_visits_cached(&st, &mut pref, 0, end, &mut wcache)?;

    let mut caches = DecodeCaches::for_schedule(indices, 16);
    let mut dec = DecodeTokenState::new(x);
    run_decode_visits_cached(&st, &mut dec, &mut caches, 0, end, &mut wcache)?;

    assert_match(&pref, &dec, "→full 48");
    Ok(())
}

/// Stepped multi-token decode: each position shares visit KV/sconv caches.
#[test]
fn decode_visit_t2_through_l6_matches_prefill() -> Result<()> {
    let file = std::fs::File::open(ckpt())?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    let st = SafeTensors::deserialize(&mmap)?;
    let tokens = 2usize;
    let mut x = vec![0f32; tokens * HIDDEN];
    for (i, v) in x.iter_mut().enumerate() {
        *v = ((i % 17) as f32) * 0.01 - 0.05;
    }
    let mut wcache = WeightCache::new();
    let end = 7;

    let mut pref = VisitState::new(x.clone());
    run_visits_cached(&st, &mut pref, 0, end, &mut wcache)?;

    let indices = layer_indices();
    let mut caches = DecodeCaches::for_schedule(indices, tokens + 4);
    let mut stepped = vec![0f32; tokens * HIDDEN];
    for t in 0..tokens {
        let mut tok = [0f32; HIDDEN];
        tok.copy_from_slice(&x[t * HIDDEN..(t + 1) * HIDDEN]);
        let mut dec = DecodeTokenState::new(tok);
        run_decode_visits_cached(&st, &mut dec, &mut caches, 0, end, &mut wcache)?;
        stepped[t * HIDDEN..(t + 1) * HIDDEN].copy_from_slice(&dec.x);
    }

    let mut maxabs = 0f32;
    for (a, b) in pref.x.iter().zip(stepped.iter()) {
        maxabs = maxabs.max((a - b).abs());
    }
    eprintln!("decode visit T=2 →L6 vs prefill maxabs={maxabs}");
    assert!(maxabs < 1e-5, "decode visit T=2 →L6 maxabs={maxabs}");
    Ok(())
}

/// T=2 through message reentry (second-pass L6).
#[test]
fn decode_visit_t2_through_msg_reentry_matches_prefill() -> Result<()> {
    let file = std::fs::File::open(ckpt())?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    let st = SafeTensors::deserialize(&mmap)?;
    let tokens = 2usize;
    let mut x = vec![0f32; tokens * HIDDEN];
    for (i, v) in x.iter_mut().enumerate() {
        *v = ((i % 17) as f32) * 0.01 - 0.05;
    }
    let mut wcache = WeightCache::new();
    let end = 19;

    let mut pref = VisitState::new(x.clone());
    run_visits_cached(&st, &mut pref, 0, end, &mut wcache)?;

    let indices = layer_indices();
    let mut caches = DecodeCaches::for_schedule(indices, tokens + 4);
    let mut stepped = vec![0f32; tokens * HIDDEN];
    for t in 0..tokens {
        let mut tok = [0f32; HIDDEN];
        tok.copy_from_slice(&x[t * HIDDEN..(t + 1) * HIDDEN]);
        let mut dec = DecodeTokenState::new(tok);
        run_decode_visits_cached(&st, &mut dec, &mut caches, 0, end, &mut wcache)?;
        stepped[t * HIDDEN..(t + 1) * HIDDEN].copy_from_slice(&dec.x);
    }

    let mut maxabs = 0f32;
    for (a, b) in pref.x.iter().zip(stepped.iter()) {
        maxabs = maxabs.max((a - b).abs());
    }
    eprintln!("decode visit T=2 →msg reentry vs prefill maxabs={maxabs}");
    assert!(maxabs < 3e-4, "decode visit T=2 →msg reentry maxabs={maxabs}");
    Ok(())
}

/// Metal MoE decode visits vs CPU decode (attn stays CPU).
#[test]
fn decode_visit_metal_moe_t1_through_l6_matches_cpu() -> Result<()> {
    let file = std::fs::File::open(ckpt())?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    let st = SafeTensors::deserialize(&mmap)?;
    let x = seed_x();
    let mut wcache = WeightCache::new();
    let indices = layer_indices();
    let end = 7;

    let mut caches_cpu = DecodeCaches::for_schedule(indices, 16);
    let mut cpu = DecodeTokenState::new(x);
    run_decode_visits_cached(&st, &mut cpu, &mut caches_cpu, 0, end, &mut wcache)?;

    let mut accel = MetalAccel::new()?;
    let mut caches_gpu = DecodeCaches::for_schedule(indices, 16);
    let mut gpu = DecodeTokenState::new(x);
    run_decode_visits_metal_cached(&st, &mut gpu, &mut caches_gpu, 0, end, &mut wcache, &mut accel)?;

    let mut maxabs = 0f32;
    for i in 0..HIDDEN {
        maxabs = maxabs.max((cpu.x[i] - gpu.x[i]).abs());
    }
    eprintln!("decode visit metal MoE T=1 →L6 vs cpu maxabs={maxabs}");
    assert!(maxabs < 5e-3, "metal decode MoE maxabs={maxabs}");
    Ok(())
}

#[test]
fn decode_visit_metal_moe_t1_through_msg_reentry_matches_cpu() -> Result<()> {
    let file = std::fs::File::open(ckpt())?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    let st = SafeTensors::deserialize(&mmap)?;
    let x = seed_x();
    let mut wcache = WeightCache::new();
    let indices = layer_indices();
    let end = 19;

    let mut caches_cpu = DecodeCaches::for_schedule(indices, 16);
    let mut cpu = DecodeTokenState::new(x);
    run_decode_visits_cached(&st, &mut cpu, &mut caches_cpu, 0, end, &mut wcache)?;

    let mut accel = MetalAccel::new()?;
    let mut caches_gpu = DecodeCaches::for_schedule(indices, 16);
    let mut gpu = DecodeTokenState::new(x);
    run_decode_visits_metal_cached(&st, &mut gpu, &mut caches_gpu, 0, end, &mut wcache, &mut accel)?;

    let mut maxabs = 0f32;
    for i in 0..HIDDEN {
        maxabs = maxabs.max((cpu.x[i] - gpu.x[i]).abs());
    }
    eprintln!("decode visit metal MoE T=1 →msg reentry vs cpu maxabs={maxabs}");
    assert!(maxabs < 5e-3, "metal decode MoE reentry maxabs={maxabs}");
    Ok(())
}

#[test]
fn decode_visit_metal_moe_t1_full_schedule_matches_cpu() -> Result<()> {
    let file = std::fs::File::open(ckpt())?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    let st = SafeTensors::deserialize(&mmap)?;
    let x = seed_x();
    let mut wcache = WeightCache::new();
    let indices = layer_indices();
    let end = indices.len();

    let mut caches_cpu = DecodeCaches::for_schedule(indices, 16);
    let mut cpu = DecodeTokenState::new(x);
    run_decode_visits_cached(&st, &mut cpu, &mut caches_cpu, 0, end, &mut wcache)?;

    let mut accel = MetalAccel::new()?;
    let mut caches_gpu = DecodeCaches::for_schedule(indices, 16);
    let mut gpu = DecodeTokenState::new(x);
    run_decode_visits_metal_cached(&st, &mut gpu, &mut caches_gpu, 0, end, &mut wcache, &mut accel)?;

    let mut maxabs = 0f32;
    for i in 0..HIDDEN {
        maxabs = maxabs.max((cpu.x[i] - gpu.x[i]).abs());
    }
    eprintln!("decode visit metal MoE T=1 →full 48 vs cpu maxabs={maxabs}");
    assert!(maxabs < 5e-3, "metal decode MoE full maxabs={maxabs}");
    Ok(())
}

/// Batch-prefill into DecodeCaches, then decode the next token vs full prefill.
#[test]
fn hybrid_prefill_into_caches_then_decode_l6() -> Result<()> {
    let file = std::fs::File::open(ckpt())?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    let st = SafeTensors::deserialize(&mmap)?;
    let tokens = 3usize;
    let mut x = vec![0f32; tokens * HIDDEN];
    for (i, v) in x.iter_mut().enumerate() {
        *v = ((i % 17) as f32) * 0.01 - 0.05;
    }
    let mut wcache = WeightCache::new();
    let end = 7;

    let mut pref = VisitState::new(x.clone());
    run_visits_cached(&st, &mut pref, 0, end, &mut wcache)?;

    let warm = tokens - 1;
    let indices = layer_indices();
    let mut caches = DecodeCaches::for_schedule(indices, tokens + 4);
    let mut warm_state = VisitState::new(x[..warm * HIDDEN].to_vec());
    run_visits_into_caches(&st, &mut warm_state, 0, end, &mut wcache, &mut caches)?;

    let mut tok = [0f32; HIDDEN];
    tok.copy_from_slice(&x[warm * HIDDEN..(warm + 1) * HIDDEN]);
    let mut dec = DecodeTokenState::new(tok);
    run_decode_visits_cached(&st, &mut dec, &mut caches, 0, end, &mut wcache)?;

    let mut maxabs = 0f32;
    for i in 0..HIDDEN {
        maxabs = maxabs.max((pref.x[warm * HIDDEN + i] - dec.x[i]).abs());
    }
    eprintln!("hybrid into_caches→decode →L6 vs prefill maxabs={maxabs}");
    assert!(maxabs < 1e-5, "hybrid decode maxabs={maxabs}");
    Ok(())
}

#[test]
fn hybrid_prefill_into_caches_then_decode_full_schedule() -> Result<()> {
    let file = std::fs::File::open(ckpt())?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    let st = SafeTensors::deserialize(&mmap)?;
    let tokens = 2usize;
    let mut x = vec![0f32; tokens * HIDDEN];
    for (i, v) in x.iter_mut().enumerate() {
        *v = ((i % 17) as f32) * 0.01 - 0.05;
    }
    let mut wcache = WeightCache::new();
    let indices = layer_indices();
    let end = indices.len();

    let mut pref = VisitState::new(x.clone());
    run_visits_cached(&st, &mut pref, 0, end, &mut wcache)?;

    let warm = tokens - 1;
    let mut caches = DecodeCaches::for_schedule(indices, tokens + 4);
    let mut warm_state = VisitState::new(x[..warm * HIDDEN].to_vec());
    run_visits_into_caches(&st, &mut warm_state, 0, end, &mut wcache, &mut caches)?;

    let mut tok = [0f32; HIDDEN];
    tok.copy_from_slice(&x[warm * HIDDEN..(warm + 1) * HIDDEN]);
    let mut dec = DecodeTokenState::new(tok);
    run_decode_visits_cached(&st, &mut dec, &mut caches, 0, end, &mut wcache)?;

    let mut maxabs = 0f32;
    for i in 0..HIDDEN {
        maxabs = maxabs.max((pref.x[warm * HIDDEN + i] - dec.x[i]).abs());
    }
    eprintln!("hybrid into_caches→decode →full 48 vs prefill maxabs={maxabs}");
    assert!(maxabs < 3e-4, "hybrid full decode maxabs={maxabs}");
    Ok(())
}

#[test]
fn hybrid_metal_moe_into_caches_then_decode_l6() -> Result<()> {
    let file = std::fs::File::open(ckpt())?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    let st = SafeTensors::deserialize(&mmap)?;
    let tokens = 3usize;
    let mut x = vec![0f32; tokens * HIDDEN];
    for (i, v) in x.iter_mut().enumerate() {
        *v = ((i % 17) as f32) * 0.01 - 0.05;
    }
    let mut wcache = WeightCache::new();
    let end = 7;
    let warm = tokens - 1;
    let indices = layer_indices();

    let mut pref = VisitState::new(x.clone());
    run_visits_cached(&st, &mut pref, 0, end, &mut wcache)?;

    let mut accel = MetalAccel::new()?;
    let mut caches = DecodeCaches::for_schedule(indices, tokens + 4);
    let mut warm_state = VisitState::new(x[..warm * HIDDEN].to_vec());
    run_visits_into_caches_metal(
        &st,
        &mut warm_state,
        0,
        end,
        &mut wcache,
        &mut caches,
        &mut accel,
    )?;

    let mut tok = [0f32; HIDDEN];
    tok.copy_from_slice(&x[warm * HIDDEN..(warm + 1) * HIDDEN]);
    let mut dec = DecodeTokenState::new(tok);
    run_decode_visits_metal_cached(&st, &mut dec, &mut caches, 0, end, &mut wcache, &mut accel)?;

    let mut maxabs = 0f32;
    for i in 0..HIDDEN {
        maxabs = maxabs.max((pref.x[warm * HIDDEN + i] - dec.x[i]).abs());
    }
    eprintln!("hybrid metal-moe into_caches→decode →L6 vs prefill maxabs={maxabs}");
    assert!(maxabs < 5e-3, "hybrid metal warm decode maxabs={maxabs}");
    Ok(())
}
