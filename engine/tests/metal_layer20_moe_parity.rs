//! Metal sparse_moe_mxfp8 vs goldens/layer20_moe_f32.npz on real L20 weights.

use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use grouille_inkling::metal_moe::MetalMoe;
use grouille_inkling::moe::{sparse_moe_dispatch, HIDDEN, INTER, TOP_K};
use grouille_inkling::weights::SlidingLayerBufs;
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

fn load_f32(path: &std::path::Path, key: &str) -> Result<Vec<f32>> {
    let f = std::fs::File::open(path)?;
    let mut npz = npyz::npz::NpzArchive::new(f)?;
    let arr = npz
        .by_name(key)?
        .with_context(|| format!("missing {key}"))?;
    Ok(arr.into_vec()?)
}

fn load_i32(path: &std::path::Path, key: &str) -> Result<Vec<i32>> {
    let f = std::fs::File::open(path)?;
    let mut npz = npyz::npz::NpzArchive::new(f)?;
    let arr = npz
        .by_name(key)?
        .with_context(|| format!("missing {key}"))?;
    Ok(arr.into_vec()?)
}

#[test]
fn metal_layer20_moe_matches_f32() -> Result<()> {
    let golden = root().join("goldens/layer20_moe_f32.npz");
    if !golden.exists() {
        bail!("missing {}", golden.display());
    }
    let moe_in = load_f32(&golden, "moe_in")?;
    let indices = load_i32(&golden, "indices")?;
    let weights = load_f32(&golden, "weights")?;
    let moe_ref = load_f32(&golden, "moe_out")?;
    let tokens = moe_in.len() / HIDDEN;
    assert_eq!(indices.len(), tokens * TOP_K);
    assert_eq!(weights.len(), tokens * TOP_K);

    let file = std::fs::File::open(ckpt())?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    let st = SafeTensors::deserialize(&mmap)?;
    let buf = SlidingLayerBufs::load(&st, 20, false)?;
    assert!(buf.mxfp8);

    let mut gpu = vec![0f32; tokens * HIDDEN];
    let mut moe = MetalMoe::new()?;
    eprintln!("metal device: {}", moe.device_name());
    moe.sparse_moe_mxfp8(
        &buf.gate_w,
        &buf.gate_s,
        &buf.up_w,
        &buf.up_s,
        &buf.down_w,
        &buf.down_s,
        128,
        HIDDEN,
        INTER,
        &indices,
        &weights,
        tokens,
        TOP_K,
        &moe_in,
        &mut gpu,
        None,
    )?;

    let mut maxabs = 0f32;
    for (a, b) in gpu.iter().zip(moe_ref.iter()) {
        maxabs = maxabs.max((a - b).abs());
    }
    eprintln!("metal L20 moe maxabs={maxabs}");
    assert!(maxabs < 1e-3, "metal L20 moe maxabs={maxabs}");

    // Also exercise sparse_moe_dispatch.
    let mut moe2 = MetalMoe::new()?;
    let via = sparse_moe_dispatch(
        &moe_in,
        &indices,
        &weights,
        tokens,
        &buf.as_weights().experts,
        Some(&mut moe2),
    )?;
    let mut dmax = 0f32;
    for (a, b) in via.iter().zip(moe_ref.iter()) {
        dmax = dmax.max((a - b).abs());
    }
    eprintln!("dispatch L20 moe maxabs={dmax}");
    assert!(dmax < 1e-3);
    Ok(())
}
