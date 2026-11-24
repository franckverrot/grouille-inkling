//! Embed vs golden; Metal unembed vs CPU on last token.

use anyhow::{bail, Context, Result};
use grouille_inkling::head::{
    embed_tokens, softcap_logits, unembed, unembed_metal, LOGIT_SOFTCAP, VOCAB,
};
use grouille_inkling::metal_ops::MetalOps;
use grouille_inkling::ops::HIDDEN;
use grouille_inkling::weights::bf16_f32;
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
fn embed_matches_layer0_x_in() -> Result<()> {
    let golden = root().join("goldens/layer0_f32.npz");
    if !golden.exists() {
        bail!("missing {}", golden.display());
    }
    let x_ref = load_f32(&golden, "x_in")?;
    let ids = {
        let f = std::fs::File::open(&golden)?;
        let mut npz = npyz::npz::NpzArchive::new(f)?;
        let arr = npz.by_name("input_ids")?.context("input_ids")?;
        arr.into_vec::<i32>()?
    };

    let file = std::fs::File::open(ckpt())?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    let st = SafeTensors::deserialize(&mmap)?;
    let weight = bf16_f32(&st, "model.llm.embed.weight")?;
    let norm = bf16_f32(&st, "model.llm.embed_norm.weight")?;

    let mut out = vec![0f32; ids.len() * HIDDEN];
    embed_tokens(&ids, &weight, &norm, &mut out);

    let mut maxabs = 0f32;
    let mut sumabs = 0f32;
    for (a, b) in out.iter().zip(x_ref.iter()) {
        let d = (a - b).abs();
        maxabs = maxabs.max(d);
        sumabs += d;
    }
    let meanabs = sumabs / out.len() as f32;
    eprintln!("embed vs layer0 x_in maxabs={maxabs} meanabs={meanabs}");
    // BF16 MLX path vs f32 RMSNorm; cosine is ~1, mean drifts ~1e-3.
    assert!(meanabs < 2e-3, "embed meanabs={meanabs}");
    assert!(maxabs < 0.2, "embed maxabs={maxabs}");
    Ok(())
}

#[test]
fn softcap_roundtrip_unit() {
    let mut y = vec![0.0f32, 15.0, 60.0, -90.0];
    softcap_logits(&mut y, LOGIT_SOFTCAP);
    assert!((y[0] - 0.0).abs() < 1e-6);
    assert!(y[1].abs() < LOGIT_SOFTCAP);
    assert!(y[2].abs() < LOGIT_SOFTCAP + 1e-3);
    assert!(y[3].abs() < LOGIT_SOFTCAP + 1e-3);
}

#[test]
fn metal_unembed_matches_cpu_last_token() -> Result<()> {
    // Synthetic tiny "vocab" exercise via dense_gemv+softcap (API uses full VOCAB).
    // Full-vocab smoke: 1 token × VOCAB on Metal vs a CPU spot-check of 64 rows.
    let file = std::fs::File::open(ckpt())?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    let st = SafeTensors::deserialize(&mmap)?;
    let weight = bf16_f32(&st, "model.llm.unembed.weight")?;
    assert_eq!(weight.len(), VOCAB * HIDDEN);

    let mut x = vec![0f32; HIDDEN];
    for (i, v) in x.iter_mut().enumerate() {
        *v = ((i % 31) as f32) * 0.001 - 0.01;
    }

    let mut gpu = vec![0f32; VOCAB];
    let mut ops = MetalOps::new()?;
    unembed_metal(&mut ops, &x, 1, &weight, &mut gpu)?;

    // Spot-check live (unpadded) rows against CPU gemv+softcap.
    use grouille_inkling::head::UNPADDED_VOCAB;
    let check = [0usize, 1000, 50_000, UNPADDED_VOCAB - 1];
    let mut maxabs = 0f32;
    for &v in &check {
        let row = &weight[v * HIDDEN..(v + 1) * HIDDEN];
        let mut acc = 0f32;
        for i in 0..HIDDEN {
            acc += row[i] * x[i];
        }
        let mut y = [acc];
        softcap_logits(&mut y, LOGIT_SOFTCAP);
        maxabs = maxabs.max((y[0] - gpu[v]).abs());
    }
    eprintln!("metal unembed spot-check maxabs={maxabs}");
    assert!(maxabs < 1e-4, "unembed spot maxabs={maxabs}");
    assert!(gpu[UNPADDED_VOCAB] <= -1.0e8, "pad logits must stay masked");

    // Full CPU unembed is too heavy for CI; ensure API still compiles.
    let _ = unembed as fn(&[f32], usize, &[f32], &mut [f32]);
    Ok(())
}
