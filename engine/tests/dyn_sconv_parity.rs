//! Dynamic short-conv parity vs `goldens/dyn_sconv.npz`.

use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use grouille_inkling::ops::{dynamic_short_conv, CONV_K, HIDDEN};

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
fn dyn_sconv_matches_f32_golden() -> Result<()> {
    let golden = root().join("goldens/dyn_sconv.npz");
    if !golden.exists() {
        bail!(
            "missing {}; run python3 spike/parity_dyn_sconv.py",
            golden.display()
        );
    }
    let x = load_npz_f32(&golden, "x")?;
    let dyn_in = load_npz_f32(&golden, "dyn_in")?;
    let static_w = load_npz_f32(&golden, "static_w")?;
    let weight_proj = load_npz_f32(&golden, "weight_proj")?;
    let out_ref = load_npz_f32(&golden, "out_f32")?;
    let channels = 1024;
    let n_heads = 8;
    let tokens = x.len() / channels;
    assert_eq!(dyn_in.len(), tokens * HIDDEN);
    assert_eq!(static_w.len(), CONV_K * channels);
    assert_eq!(weight_proj.len(), n_heads * CONV_K * HIDDEN);

    let mut out = vec![0f32; tokens * channels];
    dynamic_short_conv(
        &x,
        &dyn_in,
        &static_w,
        &weight_proj,
        tokens,
        channels,
        HIDDEN,
        n_heads,
        CONV_K,
        &mut out,
    );
    let mut maxabs = 0f32;
    for (p, r) in out.iter().zip(out_ref.iter()) {
        maxabs = maxabs.max((p - r).abs());
    }
    eprintln!("dyn_sconv maxabs={maxabs}");
    assert!(maxabs < 1e-5, "dyn_sconv parity failed: maxabs={maxabs}");
    Ok(())
}
