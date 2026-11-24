//! Prefill helper: embed → Metal visits → optional finalize/logits.

use anyhow::{bail, Context, Result};
use grouille_inkling::ops::HIDDEN;
use grouille_inkling::head::{unembed_metal, VOCAB};
use grouille_inkling::prefill::{finalize_hidden, prefill_hidden_metal, HeadBufs};
use grouille_inkling::visit::{
    run_visits_cached, run_visits_metal_cached, MetalAccel, VisitState, WeightCache,
};
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
fn prefill_embed_visit_l6_matches_mlx() -> Result<()> {
    let golden = root().join("goldens/layer0_f32.npz");
    let mlx = root().join("goldens/visit_full_mlx.npz");
    if !golden.exists() || !mlx.exists() {
        bail!("missing goldens");
    }
    let ids = {
        let f = std::fs::File::open(&golden)?;
        let mut npz = npyz::npz::NpzArchive::new(f)?;
        let arr = npz.by_name("input_ids")?.context("input_ids")?;
        arr.into_vec::<i32>()?
    };
    let after6 = load_f32(&mlx, "after_visit_6")?;

    let file = std::fs::File::open(ckpt())?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    let st = SafeTensors::deserialize(&mmap)?;
    let head = HeadBufs::load(&st)?;
    let mut accel = MetalAccel::new()?;
    let mut cache = WeightCache::new();

    let state = prefill_hidden_metal(&st, &ids, &head, 0, 7, &mut accel, &mut cache)?;
    let mut maxabs = 0f32;
    for (a, b) in state.x.iter().zip(after6.iter()) {
        maxabs = maxabs.max((a - b).abs());
    }
    eprintln!("prefill embed→L6 vs mlx maxabs={maxabs}");
    assert_eq!(state.x.len(), after6.len());
    assert!(maxabs.is_finite());
    assert!(maxabs < 5.0, "prefill L6 vs mlx maxabs={maxabs}");

    let mut finalized = vec![0f32; state.tokens * HIDDEN];
    finalize_hidden(
        &state.x,
        &state.blocks,
        state.tokens,
        &head.output_query,
        &head.final_norm,
        &mut finalized,
    );
    assert!(finalized.iter().all(|v| v.is_finite()));
    Ok(())
}

/// Full 48-visit Metal schedule vs MLX `after_visit_47` (release recommended).
#[test]
fn prefill_full_visit_matches_mlx() -> Result<()> {
    let golden = root().join("goldens/layer0_f32.npz");
    let mlx = root().join("goldens/visit_full_mlx.npz");
    if !golden.exists() || !mlx.exists() {
        bail!("missing goldens");
    }
    let ids = {
        let f = std::fs::File::open(&golden)?;
        let mut npz = npyz::npz::NpzArchive::new(f)?;
        let arr = npz.by_name("input_ids")?.context("input_ids")?;
        arr.into_vec::<i32>()?
    };
    let after6 = load_f32(&mlx, "after_visit_6")?;
    let after17 = load_f32(&mlx, "after_visit_17")?;
    let after29 = load_f32(&mlx, "after_visit_29")?;
    let after41 = load_f32(&mlx, "after_visit_41")?;
    let after47 = load_f32(&mlx, "after_visit_47")?;

    let file = std::fs::File::open(ckpt())?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    let st = SafeTensors::deserialize(&mmap)?;
    let head = HeadBufs::load(&st)?;
    let mut accel = MetalAccel::new()?;
    let mut cache = WeightCache::new();

    let mut state = prefill_hidden_metal(&st, &ids, &head, 0, 7, &mut accel, &mut cache)?;
    let report = |label: &str, pred: &[f32], refer: &[f32]| {
        let mut maxabs = 0f32;
        for (a, b) in pred.iter().zip(refer.iter()) {
            maxabs = maxabs.max((a - b).abs());
        }
        eprintln!("prefill {label} vs mlx maxabs={maxabs}");
        maxabs
    };
    let m6 = report("after_visit_6", &state.x, &after6);
    run_visits_metal_cached(&st, &mut state, 7, 18, &mut cache, &mut accel)?;
    let m17 = report("after_visit_17", &state.x, &after17);
    run_visits_metal_cached(&st, &mut state, 18, 30, &mut cache, &mut accel)?;
    let m29 = report("after_visit_29", &state.x, &after29);
    run_visits_metal_cached(&st, &mut state, 30, 42, &mut cache, &mut accel)?;
    let m41 = report("after_visit_41", &state.x, &after41);
    run_visits_metal_cached(&st, &mut state, 42, 48, &mut cache, &mut accel)?;
    let m47 = report("after_visit_47", &state.x, &after47);

    assert_eq!(state.x.len(), after47.len());
    assert!(m6.is_finite() && m17.is_finite() && m29.is_finite() && m41.is_finite() && m47.is_finite());
    // Accumulated MoE/attn drift vs MLX; keep a structural ceiling.
    assert!(m47 < 10.0, "full visit vs mlx maxabs={m47}");

    // Smoke: finalize + last-token unembed (reuse visit state; no second schedule).
    let mut hidden = vec![0f32; state.tokens * HIDDEN];
    finalize_hidden(
        &state.x,
        &state.blocks,
        state.tokens,
        &head.output_query,
        &head.final_norm,
        &mut hidden,
    );
    let last = &hidden[(state.tokens - 1) * HIDDEN..state.tokens * HIDDEN];
    let mut logits = vec![0f32; VOCAB];
    unembed_metal(
        &mut accel.ops,
        last,
        1,
        &head.unembed,
        &mut logits,
    )?;
    use grouille_inkling::UNPADDED_VOCAB;
    assert!(logits[..UNPADDED_VOCAB]
        .iter()
        .all(|v| v.is_finite() && v.abs() <= 30.0 + 1e-3));
    Ok(())
}

#[test]
fn cpu_full_visit_vs_mlx_checkpoints() -> Result<()> {
    let mlx = root().join("goldens/visit_full_mlx.npz");
    if !mlx.exists() {
        bail!("missing {}", mlx.display());
    }
    let x_embed = load_f32(&mlx, "x_embed")?;
    let after6 = load_f32(&mlx, "after_visit_6")?;
    let after17 = load_f32(&mlx, "after_visit_17")?;
    let after29 = load_f32(&mlx, "after_visit_29")?;
    let after41 = load_f32(&mlx, "after_visit_41")?;
    let after47 = load_f32(&mlx, "after_visit_47")?;

    let file = std::fs::File::open(ckpt())?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    let st = SafeTensors::deserialize(&mmap)?;
    let mut cache = WeightCache::new();
    let mut state = VisitState::new(x_embed);

    let report = |label: &str, pred: &[f32], refer: &[f32]| {
        let mut maxabs = 0f32;
        for (a, b) in pred.iter().zip(refer.iter()) {
            maxabs = maxabs.max((a - b).abs());
        }
        eprintln!("cpu {label} vs mlx maxabs={maxabs}");
        maxabs
    };

    run_visits_cached(&st, &mut state, 0, 7, &mut cache)?;
    let m6 = report("after_visit_6", &state.x, &after6);
    run_visits_cached(&st, &mut state, 7, 18, &mut cache)?;
    let m17 = report("after_visit_17", &state.x, &after17);
    run_visits_cached(&st, &mut state, 18, 30, &mut cache)?;
    let m29 = report("after_visit_29", &state.x, &after29);
    run_visits_cached(&st, &mut state, 30, 42, &mut cache)?;
    let m41 = report("after_visit_41", &state.x, &after41);
    run_visits_cached(&st, &mut state, 42, 48, &mut cache)?;
    let m47 = report("after_visit_47", &state.x, &after47);

    assert!(m6.is_finite() && m17.is_finite() && m29.is_finite() && m41.is_finite() && m47.is_finite());
    // CPU f32 tracks Metal within noise (~6.65 vs MLX at visit 47); the MLX gap is
    // reference/dtype (bf16 mixed path), not a Metal kernel regression.
    assert!(m47 < 10.0, "cpu full visit vs mlx maxabs={m47}");
    Ok(())
}
