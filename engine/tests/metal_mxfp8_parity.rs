//! Metal MXFP8 GEMV vs CPU reference.

use anyhow::Result;
use grouille_inkling::metal_mxfp8::Mxfp8Gemv;
use grouille_inkling::mxfp8::mxfp8_gemv;

#[test]
fn metal_matches_cpu_random_tile() -> Result<()> {
    let out_dim = 64usize;
    let in_dim = 256usize;
    let mut packed = vec![0u8; out_dim * in_dim];
    let mut scales = vec![0u8; out_dim * (in_dim / 32)];
    let mut x = vec![0f32; in_dim];
    for (i, p) in packed.iter_mut().enumerate() {
        *p = ((i * 37 + 11) % 255) as u8;
    }
    for (i, s) in scales.iter_mut().enumerate() {
        // Prefer mid-range E8M0 exponents.
        *s = (120 + (i % 8) as u8).min(200);
    }
    for (i, v) in x.iter_mut().enumerate() {
        *v = ((i % 17) as f32) * 0.1 - 0.5;
    }

    let cpu = mxfp8_gemv(&packed, &scales, out_dim, &x);
    let mut gpu_out = vec![0f32; out_dim];
    let mut gemv = Mxfp8Gemv::new()?;
    eprintln!("metal device: {}", gemv.device_name());
    gemv.gemv(&packed, &scales, out_dim, in_dim, &x, &mut gpu_out)?;

    let mut maxabs = 0f32;
    for (a, b) in cpu.iter().zip(gpu_out.iter()) {
        maxabs = maxabs.max((a - b).abs());
    }
    eprintln!("metal mxfp8 gemv maxabs={maxabs}");
    assert!(maxabs < 1e-4, "metal vs cpu maxabs={maxabs}");
    Ok(())
}

#[test]
fn metal_matches_cpu_fixture_row() -> Result<()> {
    use std::path::PathBuf;
    use anyhow::{bail, Context};

    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .to_path_buf();
    let golden = root.join("goldens/mxfp8_gemv.npz");
    if !golden.exists() {
        bail!("missing {}", golden.display());
    }
    let f = std::fs::File::open(&golden)?;
    let mut npz = npyz::npz::NpzArchive::new(f)?;
    let packed: Vec<u8> = npz.by_name("packed")?.context("packed")?.into_vec()?;
    let scales: Vec<u8> = npz.by_name("scales")?.context("scales")?.into_vec()?;
    let x: Vec<f32> = npz.by_name("x")?.context("x")?.into_vec()?;
    let y_ref: Vec<f32> = npz.by_name("y")?.context("y")?.into_vec()?;
    let out_dim = 1usize;
    let in_dim = x.len();

    let cpu = mxfp8_gemv(&packed, &scales, out_dim, &x);
    let mut gpu_out = vec![0f32; out_dim];
    let mut gemv = Mxfp8Gemv::new()?;
    gemv.gemv(&packed, &scales, out_dim, in_dim, &x, &mut gpu_out)?;

    let mut max_cpu = 0f32;
    let mut max_gpu = 0f32;
    for i in 0..out_dim {
        max_cpu = max_cpu.max((cpu[i] - y_ref[i]).abs());
        max_gpu = max_gpu.max((gpu_out[i] - y_ref[i]).abs());
    }
    eprintln!("metal mxfp8 fixture cpu_maxabs={max_cpu} gpu_maxabs={max_gpu}");
    assert!(max_cpu < 1e-5);
    assert!(max_gpu < 1e-4);
    Ok(())
}
