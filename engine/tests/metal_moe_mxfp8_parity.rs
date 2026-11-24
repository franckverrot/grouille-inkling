//! Metal gathered MXFP8 GEMV vs CPU mxfp8_gemv.

use anyhow::Result;
use grouille_inkling::metal_moe::MetalMoe;
use grouille_inkling::mxfp8::mxfp8_gemv;

#[test]
fn metal_gather_mxfp8_matches_cpu() -> Result<()> {
    let n_experts = 4usize;
    let tokens = 2usize;
    let routes = 2usize;
    let out_dim = 32usize;
    let in_dim = 64usize;
    let mut packed = vec![0u8; n_experts * out_dim * in_dim];
    let mut scales = vec![0u8; n_experts * out_dim * (in_dim / 32)];
    for (i, p) in packed.iter_mut().enumerate() {
        *p = ((i * 41 + 7) % 255) as u8;
    }
    for (i, s) in scales.iter_mut().enumerate() {
        *s = (118 + (i % 10) as u8).min(200);
    }
    let x: Vec<f32> = (0..tokens * in_dim)
        .map(|i| ((i % 11) as f32) * 0.05 - 0.2)
        .collect();
    let indices: Vec<i32> = vec![0, 2, 1, 3]; // [T=2, R=2]

    let mut cpu = vec![0f32; tokens * routes * out_dim];
    for t in 0..tokens {
        let xt = &x[t * in_dim..(t + 1) * in_dim];
        for r in 0..routes {
            let e = indices[t * routes + r] as usize;
            let w_off = e * out_dim * in_dim;
            let s_off = e * out_dim * (in_dim / 32);
            let y = mxfp8_gemv(
                &packed[w_off..w_off + out_dim * in_dim],
                &scales[s_off..s_off + out_dim * (in_dim / 32)],
                out_dim,
                xt,
            );
            let dst = (t * routes + r) * out_dim;
            cpu[dst..dst + out_dim].copy_from_slice(&y);
        }
    }

    let mut gpu = vec![0f32; tokens * routes * out_dim];
    let mut moe = MetalMoe::new()?;
    eprintln!("metal device: {}", moe.device_name());
    moe.gather_gemv_mxfp8(
        &packed, &scales, n_experts, out_dim, in_dim, &indices, tokens, routes, &x, &mut gpu,
    )?;

    let mut maxabs = 0f32;
    for (a, b) in cpu.iter().zip(gpu.iter()) {
        maxabs = maxabs.max((a - b).abs());
    }
    eprintln!("metal gather mxfp8 maxabs={maxabs}");
    assert!(maxabs < 1e-4, "gather mxfp8 maxabs={maxabs}");
    Ok(())
}

#[test]
fn metal_sparse_moe_mxfp8_matches_cpu() -> Result<()> {
    let n_experts = 4usize;
    let tokens = 2usize;
    let routes = 2usize;
    let hidden = 64usize;
    let inter = 32usize;
    let pack_gate = |e: usize| e * inter * hidden;
    let scale_gate = |e: usize| e * inter * (hidden / 32);
    let pack_down = |e: usize| e * hidden * inter;
    let scale_down = |e: usize| e * hidden * (inter / 32);

    let mut gate_w = vec![0u8; n_experts * inter * hidden];
    let mut gate_s = vec![0u8; n_experts * inter * (hidden / 32)];
    let mut up_w = gate_w.clone();
    let mut up_s = gate_s.clone();
    let mut down_w = vec![0u8; n_experts * hidden * inter];
    let mut down_s = vec![0u8; n_experts * hidden * (inter / 32)];
    for (i, p) in gate_w.iter_mut().enumerate() {
        *p = ((i * 13) % 255) as u8;
    }
    for (i, p) in up_w.iter_mut().enumerate() {
        *p = ((i * 17 + 1) % 255) as u8;
    }
    for (i, p) in down_w.iter_mut().enumerate() {
        *p = ((i * 19 + 2) % 255) as u8;
    }
    for s in [&mut gate_s, &mut up_s, &mut down_s] {
        for (i, v) in s.iter_mut().enumerate() {
            *v = (118 + (i % 8) as u8).min(200);
        }
    }
    let mut x = vec![0f32; tokens * hidden];
    for (i, v) in x.iter_mut().enumerate() {
        *v = ((i % 5) as f32) * 0.1 - 0.2;
    }
    let mut indices = vec![0i32; tokens * routes];
    let mut weights = vec![0f32; tokens * routes];
    for t in 0..tokens {
        let mut sum = 0f32;
        for r in 0..routes {
            indices[t * routes + r] = ((t + r) % n_experts) as i32;
            let w = 1.0 + (r as f32) * 0.1;
            weights[t * routes + r] = w;
            sum += w;
        }
        for r in 0..routes {
            weights[t * routes + r] *= 8.0 / sum;
        }
    }

    let mut cpu = vec![0f32; tokens * hidden];
    for t in 0..tokens {
        let xt = &x[t * hidden..(t + 1) * hidden];
        let mut acc = vec![0f32; hidden];
        for r in 0..routes {
            let e = indices[t * routes + r] as usize;
            let w = weights[t * routes + r];
            let g = mxfp8_gemv(
                &gate_w[pack_gate(e)..pack_gate(e + 1)],
                &gate_s[scale_gate(e)..scale_gate(e + 1)],
                inter,
                xt,
            );
            let u = mxfp8_gemv(
                &up_w[pack_gate(e)..pack_gate(e + 1)],
                &up_s[scale_gate(e)..scale_gate(e + 1)],
                inter,
                xt,
            );
            let mut h = vec![0f32; inter];
            for i in 0..inter {
                let gv = g[i];
                let uv = u[i];
                let sg = (gv / 4.0).tanh() * 4.0;
                let silu = sg / (1.0 + (-gv).exp());
                let su = (uv / 25.0).tanh() * 25.0;
                h[i] = silu * su;
            }
            let d = mxfp8_gemv(
                &down_w[pack_down(e)..pack_down(e + 1)],
                &down_s[scale_down(e)..scale_down(e + 1)],
                hidden,
                &h,
            );
            for i in 0..hidden {
                acc[i] += w * d[i];
            }
        }
        cpu[t * hidden..(t + 1) * hidden].copy_from_slice(&acc);
    }

    let mut gpu = vec![0f32; cpu.len()];
    let mut moe = MetalMoe::new()?;
    moe.sparse_moe_mxfp8(
        &gate_w, &gate_s, &up_w, &up_s, &down_w, &down_s, n_experts, hidden, inter, &indices,
        &weights, tokens, routes, &x, &mut gpu,
        None,
    )?;
    let mut maxabs = 0f32;
    for (a, b) in cpu.iter().zip(gpu.iter()) {
        maxabs = maxabs.max((a - b).abs());
    }
    eprintln!("metal sparse_moe_mxfp8 maxabs={maxabs}");
    assert!(maxabs < 1e-4, "sparse_moe_mxfp8 maxabs={maxabs}");
    Ok(())
}
