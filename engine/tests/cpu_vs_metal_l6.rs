use anyhow::Result;
use grouille_inkling::head::embed_tokens;
use grouille_inkling::ops::HIDDEN;
use grouille_inkling::prefill::HeadBufs;
use grouille_inkling::visit::{
    run_visits_cached, run_visits_metal_cached, MetalAccel, VisitState, WeightCache,
};
use safetensors::tensor::SafeTensors;
use std::path::PathBuf;

fn ckpt() -> PathBuf {
    PathBuf::from(std::env::var_os("HOME").unwrap())
        .join(".cache/huggingface/hub/models--thinkingmachines--Inkling-2-Mini")
        .join("snapshots/ed40e7da625f594fc9c4990f5f66be24873c22d7")
        .join("model-00001-of-00001.safetensors")
}

#[test]
fn cpu_vs_metal_after_l6() -> Result<()> {
    let ids = vec![
        200002, 200004, 34182, 279, 154554, 150628, 25, 220, 15, 200010, 200000, 200004, 4896,
        200010,
    ];
    let file = std::fs::File::open(ckpt())?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    let st = SafeTensors::deserialize(&mmap)?;
    let head = HeadBufs::load(&st)?;

    let mut x = vec![0f32; ids.len() * HIDDEN];
    embed_tokens(&ids, &head.embed, &head.embed_norm, &mut x);
    let mut cpu = VisitState::new(x.clone());
    let mut metal = VisitState::new(x);
    let mut cache_c = WeightCache::new();
    let mut cache_m = WeightCache::new();
    let mut accel = MetalAccel::new()?;
    run_visits_cached(&st, &mut cpu, 0, 7, &mut cache_c)?;
    run_visits_metal_cached(&st, &mut metal, 0, 7, &mut cache_m, &mut accel)?;
    let mut maxabs = 0f32;
    for (a, b) in cpu.x.iter().zip(metal.x.iter()) {
        maxabs = maxabs.max((a - b).abs());
    }
    eprintln!("cpu vs metal after visit6 maxabs={maxabs}");
    let f = std::fs::File::open("../goldens/visit_full_mlx.npz")?;
    let mut npz = npyz::npz::NpzArchive::new(f)?;
    let after6 = npz.by_name("after_visit_6")?.unwrap().into_vec::<f32>()?;
    let mut vs_mlx_cpu = 0f32;
    let mut vs_mlx_metal = 0f32;
    for i in 0..after6.len() {
        vs_mlx_cpu = vs_mlx_cpu.max((cpu.x[i] - after6[i]).abs());
        vs_mlx_metal = vs_mlx_metal.max((metal.x[i] - after6[i]).abs());
    }
    eprintln!("cpu vs mlx={vs_mlx_cpu} metal vs mlx={vs_mlx_metal}");
    assert!(maxabs < 1e-3, "cpu vs metal {maxabs}");
    Ok(())
}
