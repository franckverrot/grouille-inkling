//! Stepped Metal decode sequence logits vs Metal prefill last-token logits.

use anyhow::{bail, Context, Result};
use grouille_inkling::decode_visit::{
    decode_sequence_logits_metal, generate_argmax_metal, generate_argmax_metal_stop,
    generate_sample_metal, sample_logits, XorShift32,
};
use grouille_inkling::head::VOCAB;
use grouille_inkling::prefill::{prefill_logits_metal, HeadBufs};
use grouille_inkling::visit::{MetalAccel, WeightCache};
use safetensors::tensor::SafeTensors;
use std::path::PathBuf;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .to_path_buf()
}

fn ckpt() -> PathBuf {
    PathBuf::from(std::env::var_os("HOME").unwrap())
        .join(".cache/huggingface/hub/models--thinkingmachines--Inkling-2-Mini")
        .join("snapshots/ed40e7da625f594fc9c4990f5f66be24873c22d7")
        .join("model-00001-of-00001.safetensors")
}

#[test]
fn decode_sequence_logits_match_prefill() -> Result<()> {
    let golden = root().join("goldens/layer0_f32.npz");
    if !golden.exists() {
        bail!("missing {}", golden.display());
    }
    let ids = {
        let f = std::fs::File::open(&golden)?;
        let mut npz = npyz::npz::NpzArchive::new(f)?;
        let arr = npz.by_name("input_ids")?.context("input_ids")?;
        let mut v = arr.into_vec::<i32>()?;
        // Keep short for decode e2e runtime (full schedule × T).
        if v.len() > 2 {
            v.truncate(2);
        }
        v
    };

    let file = std::fs::File::open(ckpt())?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    let st = SafeTensors::deserialize(&mmap)?;
    let head = HeadBufs::load(&st)?;
    let mut accel = MetalAccel::new()?;
    let mut cache = WeightCache::new();

    let pref = prefill_logits_metal(&st, &ids, &head, &mut accel, &mut cache)?;
    let dec = decode_sequence_logits_metal(&st, &ids, &head, &mut accel, &mut cache)?;

    assert_eq!(pref.len(), dec.len());
    let mut maxabs = 0f32;
    for (a, b) in pref.iter().zip(dec.iter()) {
        maxabs = maxabs.max((a - b).abs());
    }
    eprintln!(
        "decode sequence T={} logits vs prefill maxabs={maxabs}",
        ids.len()
    );
    assert!(maxabs < 5e-3, "decode e2e logits maxabs={maxabs}");
    Ok(())
}

#[test]
fn generate_argmax_one_token_matches_prefill_argmax() -> Result<()> {
    let golden = root().join("goldens/layer0_f32.npz");
    if !golden.exists() {
        bail!("missing {}", golden.display());
    }
    let prompt = {
        let f = std::fs::File::open(&golden)?;
        let mut npz = npyz::npz::NpzArchive::new(f)?;
        let arr = npz.by_name("input_ids")?.context("input_ids")?;
        let mut v = arr.into_vec::<i32>()?;
        if v.len() > 2 {
            v.truncate(2);
        }
        v
    };

    let file = std::fs::File::open(ckpt())?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    let st = SafeTensors::deserialize(&mmap)?;
    let head = HeadBufs::load(&st)?;
    let mut accel = MetalAccel::new()?;
    let mut cache = WeightCache::new();

    let pref = prefill_logits_metal(&st, &prompt, &head, &mut accel, &mut cache)?;
    let mut pref_best = 0i32;
    let mut pref_v = f32::NEG_INFINITY;
    for (i, &v) in pref.iter().enumerate() {
        if v > pref_v {
            pref_v = v;
            pref_best = i as i32;
        }
    }
    assert!((0..VOCAB as i32).contains(&pref_best));

    let gen = generate_argmax_metal(&st, &prompt, 1, &head, &mut accel, &mut cache)?;
    assert_eq!(gen.len(), prompt.len() + 1);
    assert_eq!(&gen[..prompt.len()], prompt.as_slice());
    eprintln!(
        "generate argmax next={} prefill_argmax={}",
        gen[prompt.len()],
        pref_best
    );
    assert_eq!(
        gen[prompt.len()],
        pref_best,
        "greedy next token should match prefill argmax"
    );
    Ok(())
}

#[test]
fn generate_argmax_two_tokens_in_vocab() -> Result<()> {
    let golden = root().join("goldens/layer0_f32.npz");
    if !golden.exists() {
        bail!("missing {}", golden.display());
    }
    let prompt = {
        let f = std::fs::File::open(&golden)?;
        let mut npz = npyz::npz::NpzArchive::new(f)?;
        let arr = npz.by_name("input_ids")?.context("input_ids")?;
        let mut v = arr.into_vec::<i32>()?;
        if v.len() > 2 {
            v.truncate(2);
        }
        v
    };

    let file = std::fs::File::open(ckpt())?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    let st = SafeTensors::deserialize(&mmap)?;
    let head = HeadBufs::load(&st)?;
    let mut accel = MetalAccel::new()?;
    let mut cache = WeightCache::new();

    let gen = generate_argmax_metal(&st, &prompt, 2, &head, &mut accel, &mut cache)?;
    assert_eq!(gen.len(), prompt.len() + 2);
    for &id in &gen[prompt.len()..] {
        assert!(
            (0..VOCAB as i32).contains(&id),
            "generated id {id} out of vocab"
        );
    }
    eprintln!(
        "generate +2 tokens: {:?} → {:?}",
        prompt,
        &gen[prompt.len()..]
    );

    // Second greedy token should match prefill argmax on prompt+first.
    let mut extended = prompt.clone();
    extended.push(gen[prompt.len()]);
    let pref2 = prefill_logits_metal(&st, &extended, &head, &mut accel, &mut cache)?;
    let mut pref2_best = 0i32;
    let mut pref2_v = f32::NEG_INFINITY;
    for (i, &v) in pref2.iter().enumerate() {
        if v > pref2_v {
            pref2_v = v;
            pref2_best = i as i32;
        }
    }
    eprintln!(
        "second token={} prefill(prompt+t0) argmax={}",
        gen[prompt.len() + 1],
        pref2_best
    );
    assert_eq!(
        gen[prompt.len() + 1],
        pref2_best,
        "second greedy token should match extended prefill argmax"
    );
    Ok(())
}

#[test]
fn generate_stops_on_stop_id() -> Result<()> {
    let golden = root().join("goldens/layer0_f32.npz");
    if !golden.exists() {
        bail!("missing {}", golden.display());
    }
    let prompt = {
        let f = std::fs::File::open(&golden)?;
        let mut npz = npyz::npz::NpzArchive::new(f)?;
        let arr = npz.by_name("input_ids")?.context("input_ids")?;
        let mut v = arr.into_vec::<i32>()?;
        if v.len() > 2 {
            v.truncate(2);
        }
        v
    };

    let file = std::fs::File::open(ckpt())?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    let st = SafeTensors::deserialize(&mmap)?;
    let head = HeadBufs::load(&st)?;
    let mut accel = MetalAccel::new()?;
    let mut cache = WeightCache::new();

    // Discover the first greedy token, then regenerate with that id as the only stop.
    let probe = generate_argmax_metal_stop(
        &st,
        &prompt,
        1,
        &head,
        &mut accel,
        &mut cache,
        &[],
    )?;
    let first = *probe.last().context("empty probe")?;
    let mut cache = WeightCache::new();
    let gen = generate_argmax_metal_stop(
        &st,
        &prompt,
        5,
        &head,
        &mut accel,
        &mut cache,
        &[first],
    )?;
    assert_eq!(gen.len(), prompt.len() + 1);
    assert_eq!(gen[prompt.len()], first);
    eprintln!("stop-id generate truncated to {:?}", &gen[prompt.len()..]);
    Ok(())
}

#[test]
fn sample_logits_zero_temp_is_argmax() {
    let logits = [0.1f32, 5.0, 2.0, -1.0];
    let mut rng = XorShift32::new(42);
    assert_eq!(sample_logits(&logits, 0.0, &mut rng), 1);
    assert_eq!(sample_logits(&logits, -1.0, &mut rng), 1);
}

#[test]
fn generate_sample_zero_temp_matches_argmax() -> Result<()> {
    let golden = root().join("goldens/layer0_f32.npz");
    if !golden.exists() {
        bail!("missing {}", golden.display());
    }
    let prompt = {
        let f = std::fs::File::open(&golden)?;
        let mut npz = npyz::npz::NpzArchive::new(f)?;
        let arr = npz.by_name("input_ids")?.context("input_ids")?;
        let mut v = arr.into_vec::<i32>()?;
        if v.len() > 2 {
            v.truncate(2);
        }
        v
    };

    let file = std::fs::File::open(ckpt())?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    let st = SafeTensors::deserialize(&mmap)?;
    let head = HeadBufs::load(&st)?;
    let mut accel = MetalAccel::new()?;
    let mut cache = WeightCache::new();

    let greedy = generate_argmax_metal(&st, &prompt, 1, &head, &mut accel, &mut cache)?;
    let sampled = generate_sample_metal(
        &st, &prompt, 1, 0.0, 7, &head, &mut accel, &mut cache, &[],
    )?;
    assert_eq!(greedy, sampled);
    eprintln!("sample T=0 matches argmax next={}", greedy[prompt.len()]);
    Ok(())
}
