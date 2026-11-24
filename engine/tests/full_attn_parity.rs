//! Full attention (layer-5 style) parity vs `goldens/full_attn.npz`.

use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use grouille_inkling::attn::{full_attention, FullAttnWeights};
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
fn full_attn_matches_f32_golden() -> Result<()> {
    let golden = root().join("goldens/full_attn.npz");
    if !golden.exists() {
        bail!(
            "missing {}; run python3 spike/parity_full_attn.py",
            golden.display()
        );
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
    let mut out = vec![0f32; tokens * HIDDEN];
    full_attention(&q_in, &k_in, &v_in, tokens, &w, &mut out);
    let mut maxabs = 0f32;
    for (p, r) in out.iter().zip(out_ref.iter()) {
        maxabs = maxabs.max((p - r).abs());
    }
    eprintln!("full_attn maxabs={maxabs}");
    assert!(maxabs < 1e-4, "full_attn parity failed: maxabs={maxabs}");
    Ok(())
}
