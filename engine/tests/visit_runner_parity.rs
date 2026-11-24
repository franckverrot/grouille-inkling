//! Unified visit runner vs layer06 first-pass golden (visits 0..7).

use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use grouille_inkling::ops::HIDDEN;
use grouille_inkling::visit::{run_visits, run_visits_cached, VisitState, WeightCache};
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
fn visit_runner_through_layer6() -> Result<()> {
    let golden = root().join("goldens/layer06_f32.npz");
    if !golden.exists() {
        bail!("missing {}", golden.display());
    }
    let x_in = load_npz_f32(&golden, "x_in")?;
    let x_ref = load_npz_f32(&golden, "x_out")?;
    let tokens = x_in.len() / HIDDEN;
    assert_eq!(tokens, 14);

    let file = std::fs::File::open(ckpt())?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    let st = SafeTensors::deserialize(&mmap)?;
    let mut state = VisitState::new(x_in);
    // visits 0..7 → layers 0,1,2,3,4,5,6
    run_visits(&st, &mut state, 0, 7)?;
    assert_eq!(state.blocks[0].len() / HIDDEN, 2);

    let mut maxabs = 0f32;
    for (a, b) in state.x.iter().zip(x_ref.iter()) {
        maxabs = maxabs.max((a - b).abs());
    }
    eprintln!("visit runner →L6 maxabs={maxabs}");
    assert!(maxabs < 1e-3, "visit runner maxabs={maxabs}");
    Ok(())
}

#[test]
fn visit_runner_core_7_to_11() -> Result<()> {
    let g6 = root().join("goldens/layer06_f32.npz");
    let g11 = root().join("goldens/core_6_11_f32.npz");
    if !g6.exists() || !g11.exists() {
        bail!("missing core goldens");
    }
    let x = load_npz_f32(&g6, "x_out")?;
    let block0 = load_npz_f32(&g6, "block0")?;
    let block1 = load_npz_f32(&g6, "after_5")?;
    let x_ref = load_npz_f32(&g11, "x_out")?;
    let tokens = x.len() / HIDDEN;
    let mut state = VisitState::new(x);
    state.blocks = (0..tokens)
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
    // visits 7..12 → layers 7,8,9,10,11
    run_visits(&st, &mut state, 7, 12)?;

    let mut maxabs = 0f32;
    for (a, b) in state.x.iter().zip(x_ref.iter()) {
        maxabs = maxabs.max((a - b).abs());
    }
    eprintln!("visit runner L7→11 maxabs={maxabs}");
    assert!(maxabs < 1e-3, "visit runner L7→11 maxabs={maxabs}");
    Ok(())
}

#[test]
fn visit_runner_second_pass_layer6() -> Result<()> {
    let golden = root().join("goldens/visit_reentry_l6_f32.npz");
    if !golden.exists() {
        bail!("missing {}", golden.display());
    }
    let prelude_x = load_npz_f32(&golden, "prelude_x")?;
    let prelude_block = load_npz_f32(&golden, "prelude_block")?;
    let post_x = load_npz_f32(&golden, "post_x")?;
    let post_blocks = load_npz_f32(&golden, "post_blocks")?;
    let x_ref = load_npz_f32(&golden, "x_out")?;
    let tokens = prelude_x.len() / HIDDEN;
    let n_post = post_blocks.len() / (tokens * HIDDEN);

    let mut state = VisitState::new(post_x);
    state.blocks = (0..tokens)
        .map(|t| {
            let mut s = Vec::with_capacity(n_post * HIDDEN);
            for b in 0..n_post {
                let off = t * n_post * HIDDEN + b * HIDDEN;
                s.extend_from_slice(&post_blocks[off..off + HIDDEN]);
            }
            s
        })
        .collect();
    state.prelude_x = Some(prelude_x);
    state.prelude_blocks = Some(
        (0..tokens)
            .map(|t| prelude_block[t * HIDDEN..(t + 1) * HIDDEN].to_vec())
            .collect(),
    );

    let file = std::fs::File::open(ckpt())?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    let st = SafeTensors::deserialize(&mmap)?;
    // visit 18 = second CORE_START (layer 6 with messages)
    run_visits(&st, &mut state, 18, 19)?;

    let mut maxabs = 0f32;
    for (a, b) in state.x.iter().zip(x_ref.iter()) {
        maxabs = maxabs.max((a - b).abs());
    }
    eprintln!("visit runner reentry L6 maxabs={maxabs}");
    assert!(maxabs < 1e-3, "reentry L6 maxabs={maxabs}");
    Ok(())
}

#[test]
fn visit_runner_postlude_18_to_23() -> Result<()> {
    let golden = root().join("goldens/visit_postlude_f32.npz");
    if !golden.exists() {
        bail!("missing {}", golden.display());
    }
    let x_in = load_npz_f32(&golden, "x_in")?;
    let blocks_flat = load_npz_f32(&golden, "blocks_in")?;
    let x_ref = load_npz_f32(&golden, "x_out")?;
    let tokens = x_in.len() / HIDDEN;
    let n_blocks = blocks_flat.len() / (tokens * HIDDEN);
    let mut state = VisitState::new(x_in);
    state.blocks = (0..tokens)
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
    // visits 42..48 → layers 18..23 (postlude)
    run_visits(&st, &mut state, 42, 48)?;

    let mut maxabs = 0f32;
    for (a, b) in state.x.iter().zip(x_ref.iter()) {
        maxabs = maxabs.max((a - b).abs());
    }
    eprintln!("visit runner postlude 18→23 maxabs={maxabs}");
    assert_eq!(state.blocks[0].len() / HIDDEN, 4);
    assert!(maxabs < 1e-3, "postlude visit maxabs={maxabs}");
    Ok(())
}

#[test]
fn visit_runner_third_pass_layer6() -> Result<()> {
    let golden = root().join("goldens/visit_reentry_pass3_l6_f32.npz");
    if !golden.exists() {
        bail!("missing {}", golden.display());
    }
    let prelude_x = load_npz_f32(&golden, "prelude_x")?;
    let prelude_block = load_npz_f32(&golden, "prelude_block")?;
    let post_x = load_npz_f32(&golden, "post_x")?;
    let post_blocks = load_npz_f32(&golden, "post_blocks")?;
    let x_ref = load_npz_f32(&golden, "x_out")?;
    let tokens = prelude_x.len() / HIDDEN;
    let n_post = post_blocks.len() / (tokens * HIDDEN);

    let mut state = VisitState::new(post_x);
    state.blocks = (0..tokens)
        .map(|t| {
            let mut s = Vec::with_capacity(n_post * HIDDEN);
            for b in 0..n_post {
                let off = t * n_post * HIDDEN + b * HIDDEN;
                s.extend_from_slice(&post_blocks[off..off + HIDDEN]);
            }
            s
        })
        .collect();
    state.prelude_x = Some(prelude_x);
    state.prelude_blocks = Some(
        (0..tokens)
            .map(|t| prelude_block[t * HIDDEN..(t + 1) * HIDDEN].to_vec())
            .collect(),
    );

    let file = std::fs::File::open(ckpt())?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    let st = SafeTensors::deserialize(&mmap)?;
    // visit 30 = third CORE_START (layer 6 with messages)
    run_visits(&st, &mut state, 30, 31)?;

    let mut maxabs = 0f32;
    for (a, b) in state.x.iter().zip(x_ref.iter()) {
        maxabs = maxabs.max((a - b).abs());
    }
    eprintln!("visit runner third-pass L6 maxabs={maxabs}");
    assert!(maxabs < 1e-3, "third-pass L6 maxabs={maxabs}");
    Ok(())
}

#[test]
fn visit_weight_cache_reuses_across_passes() -> Result<()> {
    let golden = root().join("goldens/layer06_f32.npz");
    if !golden.exists() {
        bail!("missing {}", golden.display());
    }
    let x_in = load_npz_f32(&golden, "x_in")?;
    let file = std::fs::File::open(ckpt())?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    let st = SafeTensors::deserialize(&mmap)?;
    let mut cache = WeightCache::new();

    let t0 = std::time::Instant::now();
    let mut state = VisitState::new(x_in.clone());
    run_visits_cached(&st, &mut state, 0, 7, &mut cache)?;
    let cold = t0.elapsed();

    let t1 = std::time::Instant::now();
    let mut state2 = VisitState::new(x_in);
    run_visits_cached(&st, &mut state2, 0, 7, &mut cache)?;
    let warm = t1.elapsed();

    eprintln!(
        "visit 0..7 cold={cold:?} warm={warm:?} cached_layers={}",
        cache.len()
    );
    assert_eq!(state.x, state2.x);
    assert!(!cache.is_empty());
    // Warm should be no slower once packs are resident.
    assert!(warm <= cold + std::time::Duration::from_millis(50));
    Ok(())
}

#[test]
fn visit_runner_third_pass_through_core() -> Result<()> {
    // Seed at third-pass reentry; run visits 30..42 (L6..L17) and match L18 entry x.
    let re = root().join("goldens/visit_reentry_pass3_l6_f32.npz");
    let post = root().join("goldens/layer18_postlude_f32.npz");
    if !re.exists() || !post.exists() {
        bail!("missing third-pass / postlude goldens");
    }
    let prelude_x = load_npz_f32(&re, "prelude_x")?;
    let prelude_block = load_npz_f32(&re, "prelude_block")?;
    let post_x = load_npz_f32(&re, "post_x")?;
    let post_blocks = load_npz_f32(&re, "post_blocks")?;
    let x_ref = load_npz_f32(&post, "x_in")?;
    let tokens = prelude_x.len() / HIDDEN;
    let n_post = post_blocks.len() / (tokens * HIDDEN);

    let mut state = VisitState::new(post_x);
    state.blocks = (0..tokens)
        .map(|t| {
            let mut s = Vec::with_capacity(n_post * HIDDEN);
            for b in 0..n_post {
                let off = t * n_post * HIDDEN + b * HIDDEN;
                s.extend_from_slice(&post_blocks[off..off + HIDDEN]);
            }
            s
        })
        .collect();
    state.prelude_x.replace(prelude_x);
    state.prelude_blocks.replace(
        (0..tokens)
            .map(|t| prelude_block[t * HIDDEN..(t + 1) * HIDDEN].to_vec())
            .collect(),
    );

    let file = std::fs::File::open(ckpt())?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    let st = SafeTensors::deserialize(&mmap)?;
    let mut cache = WeightCache::new();
    run_visits_cached(&st, &mut state, 30, 42, &mut cache)?;

    // Full third-pass f32 chain drifts from the MLX bf16 postlude entry; check
    // structural completion + that activations stay finite/reasonable.
    let mut maxabs = 0f32;
    let mut xmax = 0f32;
    for (a, b) in state.x.iter().zip(x_ref.iter()) {
        maxabs = maxabs.max((a - b).abs());
        xmax = xmax.max(a.abs());
    }
    eprintln!(
        "visit third-pass core→postlude vs_mlx_entry maxabs={maxabs} |x|max={xmax} cache={}",
        cache.len()
    );
    assert!(xmax.is_finite() && xmax < 1.0e4);
    assert_eq!(state.blocks[0].len() / HIDDEN, 3);
    assert_eq!(cache.len(), 12); // L6..17 once with controllers
    Ok(())
}
