//! Metal gathered NVFP4 GEMV vs CPU.

use anyhow::Result;
use grouille_inkling::metal_moe::MetalMoe;
use grouille_inkling::nvfp4::nvfp4_gemv;

#[test]
fn metal_gather_gemv_matches_cpu() -> Result<()> {
    let n_experts = 4usize;
    let tokens = 2usize;
    let routes = 8usize;
    let out_dim = 64usize;
    let in_dim = 128usize;
    let mut packed = vec![0u8; n_experts * out_dim * (in_dim / 2)];
    let mut scales = vec![0u8; n_experts * out_dim * (in_dim / 16)];
    let mut amax = vec![0.1f32; n_experts];
    for (i, p) in packed.iter_mut().enumerate() {
        *p = ((i * 17 + 3) % 255) as u8;
    }
    for (i, s) in scales.iter_mut().enumerate() {
        *s = (0b0011_1000 + (i % 5) as u8) & 0x7F;
    }
    for (e, a) in amax.iter_mut().enumerate() {
        *a = 0.05 + 0.02 * e as f32;
    }
    let mut x = vec![0f32; tokens * in_dim];
    for (i, v) in x.iter_mut().enumerate() {
        *v = ((i % 9) as f32) * 0.05 - 0.2;
    }
    let mut indices = vec![0i32; tokens * routes];
    for t in 0..tokens {
        for r in 0..routes {
            indices[t * routes + r] = ((t + r) % n_experts) as i32;
        }
    }

    // CPU reference
    let mut cpu = vec![0f32; tokens * routes * out_dim];
    for t in 0..tokens {
        let xt = &x[t * in_dim..(t + 1) * in_dim];
        for r in 0..routes {
            let e = indices[t * routes + r] as usize;
            let w_stride = out_dim * (in_dim / 2);
            let s_stride = out_dim * (in_dim / 16);
            let y = nvfp4_gemv(
                &packed[e * w_stride..(e + 1) * w_stride],
                &scales[e * s_stride..(e + 1) * s_stride],
                out_dim,
                amax[e],
                xt,
            );
            cpu[(t * routes + r) * out_dim..(t * routes + r + 1) * out_dim].copy_from_slice(&y);
        }
    }

    let mut gpu = vec![0f32; cpu.len()];
    let mut moe = MetalMoe::new()?;
    eprintln!("metal device: {}", moe.device_name());
    moe.gather_gemv(
        &packed, &scales, &amax, n_experts, out_dim, in_dim, &indices, tokens, routes, &x,
        &mut gpu,
    )?;

    let mut maxabs = 0f32;
    for (a, b) in cpu.iter().zip(gpu.iter()) {
        maxabs = maxabs.max((a - b).abs());
    }
    eprintln!("metal gather moe gemv maxabs={maxabs}");
    assert!(maxabs < 1e-4, "gather maxabs={maxabs}");
    Ok(())
}

#[test]
fn metal_swiglu_and_combine_match_cpu() -> Result<()> {
    let tokens = 3usize;
    let routes = 8usize;
    let inter = 64usize;
    let hidden = 128usize;
    let n = tokens * routes * inter;
    let mut up = vec![0f32; n];
    let mut gate = vec![0f32; n];
    for i in 0..n {
        up[i] = ((i % 11) as f32) * 0.3 - 1.0;
        gate[i] = ((i % 7) as f32) * 0.2 - 0.5;
    }
    let gate_cap = 4.0f32;
    let up_cap = 25.0f32;
    let mut cpu_h = vec![0f32; n];
    for i in 0..n {
        let g = gate[i];
        let u = up[i];
        let sg = (g / gate_cap).tanh() * gate_cap;
        let silu = sg / (1.0 + (-g).exp());
        let su = (u / up_cap).tanh() * up_cap;
        cpu_h[i] = silu * su;
    }
    let mut gpu_h = vec![0f32; n];
    let mut moe = MetalMoe::new()?;
    moe.softcapped_swiglu(&up, &gate, tokens, routes, inter, gate_cap, up_cap, &mut gpu_h)?;
    let mut max_h = 0f32;
    for (a, b) in cpu_h.iter().zip(gpu_h.iter()) {
        max_h = max_h.max((a - b).abs());
    }
    eprintln!("metal swiglu maxabs={max_h}");
    assert!(max_h < 1e-5, "swiglu maxabs={max_h}");

    let mut routes_out = vec![0f32; tokens * routes * hidden];
    let mut weights = vec![0f32; tokens * routes];
    for t in 0..tokens {
        let mut sum = 0f32;
        for r in 0..routes {
            let w = ((t + r) % 5) as f32 + 0.5;
            weights[t * routes + r] = w;
            sum += w;
            for d in 0..hidden {
                routes_out[(t * routes + r) * hidden + d] = ((r + d) % 9) as f32 * 0.1;
            }
        }
        for r in 0..routes {
            weights[t * routes + r] *= 8.0 / sum;
        }
    }
    let mut cpu_o = vec![0f32; tokens * hidden];
    for t in 0..tokens {
        for d in 0..hidden {
            let mut acc = 0f32;
            for r in 0..routes {
                acc += weights[t * routes + r] * routes_out[(t * routes + r) * hidden + d];
            }
            cpu_o[t * hidden + d] = acc;
        }
    }
    let mut gpu_o = vec![0f32; cpu_o.len()];
    moe.combine(&routes_out, &weights, tokens, routes, hidden, &mut gpu_o)?;
    let mut max_c = 0f32;
    for (a, b) in cpu_o.iter().zip(gpu_o.iter()) {
        max_c = max_c.max((a - b).abs());
    }
    eprintln!("metal combine maxabs={max_c}");
    assert!(max_c < 1e-5, "combine maxabs={max_c}");
    Ok(())
}

#[test]
fn metal_sparse_moe_matches_cpu() -> Result<()> {
    let n_experts = 4usize;
    let tokens = 2usize;
    let routes = 8usize;
    let hidden = 64usize;
    let inter = 32usize;
    let pack_gate = |e: usize| e * inter * (hidden / 2);
    let scale_gate = |e: usize| e * inter * (hidden / 16);
    let pack_down = |e: usize| e * hidden * (inter / 2);
    let scale_down = |e: usize| e * hidden * (inter / 16);

    let mut gate_w = vec![0u8; n_experts * inter * (hidden / 2)];
    let mut gate_s = vec![0u8; n_experts * inter * (hidden / 16)];
    let mut up_w = gate_w.clone();
    let mut up_s = gate_s.clone();
    let mut down_w = vec![0u8; n_experts * hidden * (inter / 2)];
    let mut down_s = vec![0u8; n_experts * hidden * (inter / 16)];
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
            *v = (0b0011_1000 + (i % 4) as u8) & 0x7F;
        }
    }
    let gate_up_amax: Vec<f32> = (0..n_experts).map(|e| 0.1 + 0.01 * e as f32).collect();
    let down_amax = gate_up_amax.clone();
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

    // CPU path expects HIDDEN=2304 / INTER=1152 constants in ExpertPack strides.
    // Use Metal-only path vs explicit CPU gemv loop for this small geometry.
    let mut cpu = vec![0f32; tokens * hidden];
    for t in 0..tokens {
        let xt = &x[t * hidden..(t + 1) * hidden];
        let mut acc = vec![0f32; hidden];
        for r in 0..routes {
            let e = indices[t * routes + r] as usize;
            let w = weights[t * routes + r];
            let g = nvfp4_gemv(
                &gate_w[pack_gate(e)..pack_gate(e + 1)],
                &gate_s[scale_gate(e)..scale_gate(e + 1)],
                inter,
                gate_up_amax[e],
                xt,
            );
            let u = nvfp4_gemv(
                &up_w[pack_gate(e)..pack_gate(e + 1)],
                &up_s[scale_gate(e)..scale_gate(e + 1)],
                inter,
                gate_up_amax[e],
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
            let d = nvfp4_gemv(
                &down_w[pack_down(e)..pack_down(e + 1)],
                &down_s[scale_down(e)..scale_down(e + 1)],
                hidden,
                down_amax[e],
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
    moe.sparse_moe(
        &gate_w,
        &gate_s,
        &up_w,
        &up_s,
        &down_w,
        &down_s,
        &gate_up_amax,
        &down_amax,
        n_experts,
        hidden,
        inter,
        &indices,
        &weights,
        tokens,
        routes,
        &x,
        &mut gpu,
        None,
    )?;
    let mut maxabs = 0f32;
    for (a, b) in cpu.iter().zip(gpu.iter()) {
        maxabs = maxabs.max((a - b).abs());
    }
    eprintln!("metal sparse_moe maxabs={maxabs}");
    assert!(maxabs < 1e-4, "sparse_moe maxabs={maxabs}");
    Ok(())
}
