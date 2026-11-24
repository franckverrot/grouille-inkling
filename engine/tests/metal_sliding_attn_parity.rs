//! Sliding attention on the GPU prompt path vs CPU `sliding_attention`.

use anyhow::Result;
use grouille_inkling::attn::sliding_attention;
use grouille_inkling::metal_ops::MetalOps;
use grouille_inkling::metal_prompt::PromptAttn;
use grouille_inkling::ops::HIDDEN;
use grouille_inkling::weights::SlidingLayerBufs;
use safetensors::tensor::SafeTensors;
use std::path::PathBuf;

#[test]
fn metal_sliding_attn_matches_cpu_layer0() -> Result<()> {
    let ckpt = PathBuf::from(std::env::var_os("HOME").unwrap())
        .join(".cache/huggingface/hub/models--thinkingmachines--Inkling-2-Mini")
        .join("snapshots/ed40e7da625f594fc9c4990f5f66be24873c22d7")
        .join("model-00001-of-00001.safetensors");
    let file = std::fs::File::open(&ckpt)?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    let st = SafeTensors::deserialize(&mmap)?;
    let buf = SlidingLayerBufs::load(&st, 0, false)?;
    let layer = buf.as_weights();

    let tokens = 14usize;
    let mut x = vec![0f32; tokens * HIDDEN];
    for (i, v) in x.iter_mut().enumerate() {
        *v = ((i % 23) as f32) * 0.01 - 0.1;
    }

    let mut cpu = vec![0f32; tokens * HIDDEN];
    sliding_attention(&x, tokens, &layer.attn, &mut cpu);

    let mut metal = MetalOps::new()?;
    eprintln!("metal device: {}", metal.device_name());
    let gpu = metal.prompt_attention_host(&PromptAttn::sliding(&layer.attn), &x, &x, &x, tokens)?;

    let mut maxabs = 0f32;
    for (a, b) in cpu.iter().zip(gpu.iter()) {
        maxabs = maxabs.max((a - b).abs());
    }
    eprintln!("metal sliding_attn vs cpu maxabs={maxabs}");
    assert!(maxabs < 1e-4, "sliding_attn maxabs={maxabs}");
    Ok(())
}
