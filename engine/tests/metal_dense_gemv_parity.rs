//! Metal dense GEMV vs CPU `gemv`.

use anyhow::Result;
use grouille_inkling::metal_ops::MetalOps;
use grouille_inkling::ops::gemv;

#[test]
fn metal_dense_gemv_matches_cpu() -> Result<()> {
    let tokens = 5usize;
    let in_dim = 64usize;
    let out_dim = 48usize;
    let mut x = vec![0f32; tokens * in_dim];
    let mut w = vec![0f32; out_dim * in_dim];
    for (i, v) in x.iter_mut().enumerate() {
        *v = ((i % 17) as f32) * 0.01 - 0.05;
    }
    for (i, v) in w.iter_mut().enumerate() {
        *v = ((i % 13) as f32) * 0.02 - 0.1;
    }

    let mut cpu = vec![0f32; tokens * out_dim];
    for t in 0..tokens {
        gemv(
            &w,
            out_dim,
            &x[t * in_dim..(t + 1) * in_dim],
            &mut cpu[t * out_dim..(t + 1) * out_dim],
        );
    }

    let mut gpu = vec![0f32; tokens * out_dim];
    let mut ops = MetalOps::new()?;
    ops.dense_gemv(&x, &w, tokens, out_dim, in_dim, &mut gpu)?;

    let mut maxabs = 0f32;
    for (a, b) in cpu.iter().zip(gpu.iter()) {
        maxabs = maxabs.max((a - b).abs());
    }
    eprintln!("metal dense_gemv maxabs={maxabs}");
    assert!(maxabs < 1e-5, "dense_gemv maxabs={maxabs}");
    Ok(())
}
