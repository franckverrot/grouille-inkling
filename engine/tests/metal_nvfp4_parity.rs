//! Metal NVFP4 GEMV vs CPU reference.

use std::path::PathBuf;

use anyhow::Result;
use grouille_inkling::metal_nvfp4::Nvfp4Gemv;
use grouille_inkling::nvfp4::nvfp4_gemv;
use safetensors::tensor::SafeTensors;

#[test]
fn metal_matches_cpu_random_tile() -> Result<()> {
    let out_dim = 128usize;
    let in_dim = 256usize;
    let mut packed = vec![0u8; out_dim * (in_dim / 2)];
    let mut scales = vec![0u8; out_dim * (in_dim / 16)];
    let mut x = vec![0f32; in_dim];
    // Deterministic pseudo-random nibbles / scales / activations.
    for (i, p) in packed.iter_mut().enumerate() {
        *p = ((i * 37 + 11) % 255) as u8;
    }
    for (i, s) in scales.iter_mut().enumerate() {
        // Prefer finite e4m3-ish codes (avoid all-ones nibble patterns).
        *s = (0b0011_1000 + (i % 7) as u8) & 0x7F;
    }
    for (i, v) in x.iter_mut().enumerate() {
        *v = ((i % 17) as f32) * 0.1 - 0.5;
    }
    let amax = 0.25f32;

    let mut cpu = vec![0f32; out_dim];
    cpu.copy_from_slice(&nvfp4_gemv(&packed, &scales, out_dim, amax, &x));

    let mut gpu_out = vec![0f32; out_dim];
    let mut gemv = Nvfp4Gemv::new()?;
    eprintln!("metal device: {}", gemv.device_name());
    gemv.gemv(&packed, &scales, out_dim, in_dim, amax, &x, &mut gpu_out)?;

    let mut maxabs = 0f32;
    for (a, b) in cpu.iter().zip(gpu_out.iter()) {
        maxabs = maxabs.max((a - b).abs());
    }
    eprintln!("metal nvfp4 gemv maxabs={maxabs}");
    assert!(maxabs < 1e-4, "metal vs cpu maxabs={maxabs}");
    Ok(())
}

#[test]
fn metal_matches_cpu_real_expert0_gate() -> Result<()> {
    let ckpt = PathBuf::from(std::env::var_os("HOME").unwrap())
        .join(".cache/huggingface/hub/models--thinkingmachines--Inkling-2-Mini")
        .join("snapshots/ed40e7da625f594fc9c4990f5f66be24873c22d7")
        .join("model-00001-of-00001.safetensors");
    let file = std::fs::File::open(&ckpt)?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    let st = SafeTensors::deserialize(&mmap)?;
    let w13 = st.tensor("model.llm.layers.0.mlp.experts.w13_weight")?;
    let s13 = st.tensor("model.llm.layers.0.mlp.experts.w13_weight_scale")?;
    let a13 = st.tensor("model.llm.layers.0.mlp.experts.w13_weight_amax")?;
    // Expert 0, gate rows (even indices of 2304): take first INTER=1152 rows' packed slice.
    // Layout [E=128, 2304, 1152] u8: gate = rows 0,2,4,... so row r maps to fused row 2r.
    let out_dim = 1152usize;
    let in_dim = 2304usize;
    let fused_row_stride = 1152usize;
    let mut packed = vec![0u8; out_dim * (in_dim / 2)];
    let mut scales = vec![0u8; out_dim * (in_dim / 16)];
    let wbytes = w13.data();
    let sbytes = s13.data();
    let expert0 = 0usize;
    let w_base = expert0 * 2304 * fused_row_stride;
    let s_base = expert0 * 2304 * (in_dim / 16);
    for i in 0..out_dim {
        let src = w_base + (i * 2) * fused_row_stride;
        packed[i * fused_row_stride..(i + 1) * fused_row_stride]
            .copy_from_slice(&wbytes[src..src + fused_row_stride]);
        let ss = s_base + (i * 2) * (in_dim / 16);
        scales[i * (in_dim / 16)..(i + 1) * (in_dim / 16)]
            .copy_from_slice(&sbytes[ss..ss + (in_dim / 16)]);
    }
    let amax = {
        let b = a13.data();
        f32::from_le_bytes([b[0], b[1], b[2], b[3]])
    };
    let x: Vec<f32> = (0..in_dim).map(|i| ((i % 13) as f32) * 0.01 - 0.05).collect();
    let cpu = nvfp4_gemv(&packed, &scales, out_dim, amax, &x);
    let mut gpu_out = vec![0f32; out_dim];
    let mut gemv = Nvfp4Gemv::new()?;
    gemv.gemv(&packed, &scales, out_dim, in_dim, amax, &x, &mut gpu_out)?;
    let mut maxabs = 0f32;
    for (a, b) in cpu.iter().zip(gpu_out.iter()) {
        maxabs = maxabs.max((a - b).abs());
    }
    eprintln!("metal real expert0 gate maxabs={maxabs} amax={amax}");
    assert!(maxabs < 1e-3, "real expert tile maxabs={maxabs}");
    Ok(())
}
