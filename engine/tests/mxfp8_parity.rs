//! MXFP8 GEMV vs goldens/mxfp8_gemv.npz (MLX-matched dequant).

use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use grouille_inkling::mxfp8::mxfp8_gemv;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .to_path_buf()
}

fn load_u8(path: &std::path::Path, key: &str) -> Result<Vec<u8>> {
    let f = std::fs::File::open(path)?;
    let mut npz = npyz::npz::NpzArchive::new(f)?;
    let arr = npz
        .by_name(key)?
        .with_context(|| format!("missing {key}"))?;
    Ok(arr.into_vec()?)
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
fn mxfp8_gemv_matches_fixture() -> Result<()> {
    let golden = root().join("goldens/mxfp8_gemv.npz");
    if !golden.exists() {
        bail!("missing {}", golden.display());
    }
    let packed = load_u8(&golden, "packed")?;
    let scales = load_u8(&golden, "scales")?;
    let x = load_f32(&golden, "x")?;
    let y_ref = load_f32(&golden, "y")?;
    assert_eq!(packed.len(), 2304);
    assert_eq!(scales.len(), 72);
    assert_eq!(x.len(), 2304);
    let y = mxfp8_gemv(&packed, &scales, 1, &x);
    let mut maxabs = 0f32;
    for (a, b) in y.iter().zip(y_ref.iter()) {
        maxabs = maxabs.max((a - b).abs());
    }
    eprintln!("mxfp8 gemv maxabs={maxabs}");
    assert!(maxabs < 1e-5, "mxfp8 gemv maxabs={maxabs}");
    Ok(())
}
