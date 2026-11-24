//! Full attention on the GPU prompt path vs CPU `full_attention` / golden.

use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use grouille_inkling::attn::{full_attention, FullAttnWeights};
use grouille_inkling::metal_ops::MetalOps;
use grouille_inkling::metal_prompt::PromptAttn;
use grouille_inkling::ops::HIDDEN;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .to_path_buf()
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
fn metal_full_attn_matches_cpu() -> Result<()> {
    let golden = root().join("goldens/full_attn.npz");
    if !golden.exists() {
        bail!("missing {}", golden.display());
    }
    let q_in = load_npz_f32(&golden, "q_in")?;
    let k_in = load_npz_f32(&golden, "k_in")?;
    let v_in = load_npz_f32(&golden, "v_in")?;
    let out_ref = load_npz_f32(&golden, "out_f32")?;
    let tokens = q_in.len() / HIDDEN;

    let wq = load_npz_f32(&golden, "wq")?;
    let wk = load_npz_f32(&golden, "wk")?;
    let wv = load_npz_f32(&golden, "wv")?;
    let wr = load_npz_f32(&golden, "wr")?;
    let wo = load_npz_f32(&golden, "wo")?;
    let gate = load_npz_f32(&golden, "gate")?;
    let sink = load_npz_f32(&golden, "sink")?;
    let q_scale = load_npz_f32(&golden, "q_scale")?;
    let rel_scale = load_npz_f32(&golden, "rel_scale")?;
    let rel_proj = load_npz_f32(&golden, "rel_proj")?;
    let k_static = load_npz_f32(&golden, "k_static")?;
    let k_wproj = load_npz_f32(&golden, "k_wproj")?;
    let v_static = load_npz_f32(&golden, "v_static")?;
    let v_wproj = load_npz_f32(&golden, "v_wproj")?;
    let w = FullAttnWeights {
        layer: None,
        wq: &wq,
        wk: &wk,
        wv: &wv,
        wr: &wr,
        wo: &wo,
        gate: &gate,
        sink: &sink,
        q_scale: &q_scale,
        rel_scale: &rel_scale,
        rel_proj: &rel_proj,
        k_static: &k_static,
        k_wproj: &k_wproj,
        v_static: &v_static,
        v_wproj: &v_wproj,
    };

    let mut cpu = vec![0f32; tokens * HIDDEN];
    full_attention(&q_in, &k_in, &v_in, tokens, &w, &mut cpu);

    let mut metal = MetalOps::new()?;
    eprintln!("metal device: {}", metal.device_name());
    let gpu = metal.prompt_attention_host(&PromptAttn::full(&w), &q_in, &k_in, &v_in, tokens)?;

    let mut max_cpu = 0f32;
    let mut max_ref = 0f32;
    for i in 0..cpu.len() {
        max_cpu = max_cpu.max((cpu[i] - gpu[i]).abs());
        max_ref = max_ref.max((out_ref[i] - gpu[i]).abs());
    }
    eprintln!("metal full_attn vs cpu={max_cpu} vs golden={max_ref}");
    assert!(max_cpu < 1e-4, "vs cpu maxabs={max_cpu}");
    assert!(max_ref < 1e-4, "vs golden maxabs={max_ref}");
    Ok(())
}
