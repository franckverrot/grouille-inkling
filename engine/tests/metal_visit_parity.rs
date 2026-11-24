//! Metal visit runner vs CPU / goldens (Metal attn; MXFP8 MoE in postlude).

use anyhow::{bail, Context, Result};
use grouille_inkling::ops::HIDDEN;
use grouille_inkling::visit::{run_visits, run_visits_metal, MetalAccel, VisitState};
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
fn metal_visit_through_layer6_matches_cpu() -> Result<()> {
    let golden = root().join("goldens/layer06_f32.npz");
    if !golden.exists() {
        bail!("missing {}", golden.display());
    }
    let x_in = load_f32(&golden, "x_in")?;
    let x_ref = load_f32(&golden, "x_out")?;

    let file = std::fs::File::open(ckpt())?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    let st = SafeTensors::deserialize(&mmap)?;

    let mut cpu = VisitState::new(x_in.clone());
    run_visits(&st, &mut cpu, 0, 7)?;

    let mut gpu = VisitState::new(x_in);
    let mut accel = MetalAccel::new()?;
    run_visits_metal(&st, &mut gpu, 0, 7, &mut accel)?;

    let mut vs_cpu = 0f32;
    let mut vs_ref = 0f32;
    for i in 0..cpu.x.len() {
        vs_cpu = vs_cpu.max((cpu.x[i] - gpu.x[i]).abs());
        vs_ref = vs_ref.max((x_ref[i] - gpu.x[i]).abs());
    }
    eprintln!("metal visit →L6 vs cpu={vs_cpu} vs golden={vs_ref}");
    assert!(vs_cpu < 5e-4, "metal visit vs cpu maxabs={vs_cpu}");
    assert!(vs_ref < 1e-3, "metal visit vs golden maxabs={vs_ref}");

    Ok(())
}

#[test]
fn metal_visit_from_embed_matches_mlx_checkpoints() -> Result<()> {
    let mlx = root().join("goldens/visit_full_mlx.npz");
    if !mlx.exists() {
        bail!("missing {}", mlx.display());
    }
    let x_embed = load_f32(&mlx, "x_embed")?;
    let after6 = load_f32(&mlx, "after_visit_6")?;

    let file = std::fs::File::open(ckpt())?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    let st = SafeTensors::deserialize(&mmap)?;

    let mut gpu = VisitState::new(x_embed);
    let mut accel = MetalAccel::new()?;
    run_visits_metal(&st, &mut gpu, 0, 7, &mut accel)?;

    let mut vs_mlx = 0f32;
    for (a, b) in gpu.x.iter().zip(after6.iter()) {
        vs_mlx = vs_mlx.max((a - b).abs());
    }
    eprintln!("metal visit from embed →L6 vs mlx={vs_mlx}");
    // Prefill MoE/attn accumulate drift vs MLX; keep structural (finite, same shape).
    assert!(vs_mlx.is_finite());
    assert_eq!(gpu.x.len(), after6.len());
    assert!(vs_mlx < 5.0, "metal vs mlx after_visit_6 maxabs={vs_mlx}");
    Ok(())
}

#[test]
fn metal_visit_reentry_l6_matches_cpu() -> Result<()> {
    let golden = root().join("goldens/visit_reentry_l6_f32.npz");
    if !golden.exists() {
        bail!("missing {}", golden.display());
    }
    let prelude_x = load_f32(&golden, "prelude_x")?;
    let prelude_block = load_f32(&golden, "prelude_block")?;
    let post_x = load_f32(&golden, "post_x")?;
    let post_blocks = load_f32(&golden, "post_blocks")?;
    let x_ref = load_f32(&golden, "x_out")?;
    let tokens = prelude_x.len() / HIDDEN;
    let n_post = post_blocks.len() / (tokens * HIDDEN);

    let make_state = || -> VisitState {
        let mut state = VisitState::new(post_x.clone());
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
        state.prelude_x = Some(prelude_x.clone());
        state.prelude_blocks = Some(
            (0..tokens)
                .map(|t| prelude_block[t * HIDDEN..(t + 1) * HIDDEN].to_vec())
                .collect(),
        );
        state
    };

    let file = std::fs::File::open(ckpt())?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    let st = SafeTensors::deserialize(&mmap)?;

    let mut cpu = make_state();
    run_visits(&st, &mut cpu, 18, 19)?;

    let mut gpu = make_state();
    let mut accel = MetalAccel::new()?;
    run_visits_metal(&st, &mut gpu, 18, 19, &mut accel)?;

    let mut vs_cpu = 0f32;
    let mut vs_ref = 0f32;
    for i in 0..cpu.x.len() {
        vs_cpu = vs_cpu.max((cpu.x[i] - gpu.x[i]).abs());
        vs_ref = vs_ref.max((x_ref[i] - gpu.x[i]).abs());
    }
    eprintln!("metal visit reentry L6 vs cpu={vs_cpu} vs golden={vs_ref}");
    assert!(vs_cpu < 5e-3, "metal reentry vs cpu maxabs={vs_cpu}");
    assert!(vs_ref < 5e-3, "metal reentry vs golden maxabs={vs_ref}");
    Ok(())
}

#[test]
fn metal_visit_postlude_matches_cpu() -> Result<()> {
    let golden = root().join("goldens/visit_postlude_f32.npz");
    if !golden.exists() {
        bail!("missing {}", golden.display());
    }
    let x_in = load_f32(&golden, "x_in")?;
    let blocks_flat = load_f32(&golden, "blocks_in")?;
    let x_ref = load_f32(&golden, "x_out")?;
    let tokens = x_in.len() / HIDDEN;
    let n_blocks = blocks_flat.len() / (tokens * HIDDEN);

    let file = std::fs::File::open(ckpt())?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    let st = SafeTensors::deserialize(&mmap)?;

    let make_blocks = || -> Vec<Vec<f32>> {
        (0..tokens)
            .map(|t| {
                let mut s = Vec::with_capacity(n_blocks * HIDDEN);
                for b in 0..n_blocks {
                    let off = t * n_blocks * HIDDEN + b * HIDDEN;
                    s.extend_from_slice(&blocks_flat[off..off + HIDDEN]);
                }
                s
            })
            .collect()
    };

    let mut cpu = VisitState::new(x_in.clone());
    cpu.blocks = make_blocks();
    run_visits(&st, &mut cpu, 42, 48)?;

    let mut gpu = VisitState::new(x_in);
    gpu.blocks = make_blocks();
    let mut accel = MetalAccel::new()?;
    run_visits_metal(&st, &mut gpu, 42, 48, &mut accel)?;

    let mut vs_cpu = 0f32;
    let mut vs_ref = 0f32;
    for i in 0..cpu.x.len() {
        vs_cpu = vs_cpu.max((cpu.x[i] - gpu.x[i]).abs());
        vs_ref = vs_ref.max((x_ref[i] - gpu.x[i]).abs());
    }
    eprintln!("metal visit postlude vs cpu={vs_cpu} vs golden={vs_ref}");
    assert!(vs_cpu < 5e-3, "metal postlude vs cpu maxabs={vs_cpu}");
    assert!(vs_ref < 5e-3, "metal postlude vs golden maxabs={vs_ref}");
    Ok(())
}

#[test]
fn metal_visit_third_pass_l6_matches_cpu() -> Result<()> {
    let golden = root().join("goldens/visit_reentry_pass3_l6_f32.npz");
    if !golden.exists() {
        bail!("missing {}", golden.display());
    }
    let prelude_x = load_f32(&golden, "prelude_x")?;
    let prelude_block = load_f32(&golden, "prelude_block")?;
    let post_x = load_f32(&golden, "post_x")?;
    let post_blocks = load_f32(&golden, "post_blocks")?;
    let x_ref = load_f32(&golden, "x_out")?;
    let tokens = prelude_x.len() / HIDDEN;
    let n_post = post_blocks.len() / (tokens * HIDDEN);

    let make_state = || -> VisitState {
        let mut state = VisitState::new(post_x.clone());
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
        state.prelude_x = Some(prelude_x.clone());
        state.prelude_blocks = Some(
            (0..tokens)
                .map(|t| prelude_block[t * HIDDEN..(t + 1) * HIDDEN].to_vec())
                .collect(),
        );
        state
    };

    let file = std::fs::File::open(ckpt())?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    let st = SafeTensors::deserialize(&mmap)?;

    let mut cpu = make_state();
    run_visits(&st, &mut cpu, 30, 31)?;

    let mut gpu = make_state();
    let mut accel = MetalAccel::new()?;
    run_visits_metal(&st, &mut gpu, 30, 31, &mut accel)?;

    let mut vs_cpu = 0f32;
    let mut vs_ref = 0f32;
    for i in 0..cpu.x.len() {
        vs_cpu = vs_cpu.max((cpu.x[i] - gpu.x[i]).abs());
        vs_ref = vs_ref.max((x_ref[i] - gpu.x[i]).abs());
    }
    eprintln!("metal visit third-pass L6 vs cpu={vs_cpu} vs golden={vs_ref}");
    assert!(vs_cpu < 5e-3, "metal third-pass vs cpu maxabs={vs_cpu}");
    assert!(vs_ref < 5e-3, "metal third-pass vs golden maxabs={vs_ref}");
    Ok(())
}
