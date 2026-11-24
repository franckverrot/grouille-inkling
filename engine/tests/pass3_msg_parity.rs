//! Third-pass core layers with messages: L7 (sliding), L11/L17 (full).

use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use grouille_inkling::decoder::sliding_layer_prefill_msgs;
use grouille_inkling::layer5::full_layer_prefill_msgs;
use grouille_inkling::loop_msg::MSG_SLOTS;
use grouille_inkling::ops::HIDDEN;
use grouille_inkling::weights::{FullLayerBufs, SlidingLayerBufs};
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

fn load_blocks(blocks_flat: &[f32], tokens: usize) -> Vec<Vec<f32>> {
    let n_blocks = blocks_flat.len() / (tokens * HIDDEN);
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
}

fn check_sliding(layer: usize) -> Result<()> {
    let golden = root().join(format!("goldens/layer{layer}_pass3_msg_f32.npz"));
    if !golden.exists() {
        bail!("missing {}", golden.display());
    }
    let x_in = load_npz_f32(&golden, "x_in")?;
    let x_ref = load_npz_f32(&golden, "x_out")?;
    let messages = load_npz_f32(&golden, "messages")?;
    let blocks_flat = load_npz_f32(&golden, "blocks")?;
    let tokens = x_in.len() / HIDDEN;
    assert_eq!(messages.len(), tokens * MSG_SLOTS * HIDDEN);
    let blocks = load_blocks(&blocks_flat, tokens);

    let file = std::fs::File::open(ckpt())?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    let st = SafeTensors::deserialize(&mmap)?;
    let buf = SlidingLayerBufs::load(&st, layer, true)?;
    let (pred, _) =
        sliding_layer_prefill_msgs(&x_in, tokens, &blocks, &buf.as_weights(), Some(&messages));
    let mut maxabs = 0f32;
    for (a, b) in pred.iter().zip(x_ref.iter()) {
        maxabs = maxabs.max((a - b).abs());
    }
    eprintln!("layer{layer}_pass3_msg maxabs={maxabs}");
    assert!(maxabs < 1e-3, "layer{layer}_pass3 maxabs={maxabs}");
    Ok(())
}

#[test]
fn layer7_pass3_msg_matches_f32() -> Result<()> {
    check_sliding(7)
}

#[test]
fn layer8_pass3_msg_matches_f32() -> Result<()> {
    check_sliding(8)
}

#[test]
fn layer9_pass3_msg_matches_f32() -> Result<()> {
    check_sliding(9)
}

#[test]
fn layer10_pass3_msg_matches_f32() -> Result<()> {
    check_sliding(10)
}

#[test]
fn layer12_pass3_msg_matches_f32() -> Result<()> {
    check_sliding(12)
}

#[test]
fn layer13_pass3_msg_matches_f32() -> Result<()> {
    check_sliding(13)
}

#[test]
fn layer14_pass3_msg_matches_f32() -> Result<()> {
    check_sliding(14)
}

#[test]
fn layer15_pass3_msg_matches_f32() -> Result<()> {
    check_sliding(15)
}

#[test]
fn layer16_pass3_msg_matches_f32() -> Result<()> {
    check_sliding(16)
}

fn check_full(layer: usize) -> Result<()> {
    let golden = root().join(format!("goldens/layer{layer}_pass3_msg_f32.npz"));
    if !golden.exists() {
        bail!("missing {}", golden.display());
    }
    let x_in = load_npz_f32(&golden, "x_in")?;
    let x_ref = load_npz_f32(&golden, "x_out")?;
    let messages = load_npz_f32(&golden, "messages")?;
    let blocks_flat = load_npz_f32(&golden, "blocks")?;
    let tokens = x_in.len() / HIDDEN;
    assert_eq!(messages.len(), tokens * MSG_SLOTS * HIDDEN);
    let blocks = load_blocks(&blocks_flat, tokens);

    let file = std::fs::File::open(ckpt())?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    let st = SafeTensors::deserialize(&mmap)?;
    let buf = FullLayerBufs::load_with_controllers(&st, layer, true)?;
    let (pred, _) =
        full_layer_prefill_msgs(&x_in, tokens, &blocks, &buf.as_weights(), Some(&messages));
    let mut maxabs = 0f32;
    for (a, b) in pred.iter().zip(x_ref.iter()) {
        maxabs = maxabs.max((a - b).abs());
    }
    eprintln!("layer{layer}_pass3_msg maxabs={maxabs}");
    assert!(maxabs < 1e-3, "layer{layer}_pass3 maxabs={maxabs}");
    Ok(())
}

#[test]
fn layer11_pass3_msg_matches_f32() -> Result<()> {
    check_full(11)
}

#[test]
fn layer17_pass3_msg_matches_f32() -> Result<()> {
    check_full(17)
}
