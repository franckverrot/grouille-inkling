//! A prompt continued after decode vs the same tokens as one prompt.
//!
//! Prompt, a few decoded tokens, then more prompt tokens through
//! `extend_prompt_metal`, as a later chat turn does.  The logits have to match what
//! one prompt over the whole sequence gives.

use anyhow::Result;
use grouille_inkling::decode_visit::{
    decode_next_id_metal, extend_prompt_metal, warm_prompt_metal, EXTEND_MIN,
};
use grouille_inkling::head::VOCAB;
use grouille_inkling::prefill::HeadBufs;
use grouille_inkling::visit::{MetalAccel, WeightCache};
use safetensors::tensor::SafeTensors;
use std::path::PathBuf;

fn ckpt() -> PathBuf {
    PathBuf::from(std::env::var_os("HOME").unwrap())
        .join(".cache/huggingface/hub/models--thinkingmachines--Inkling-2-Mini")
        .join("snapshots/ed40e7da625f594fc9c4990f5f66be24873c22d7")
        .join("model-00001-of-00001.safetensors")
}

fn argmax(v: &[f32]) -> usize {
    v.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0
}

#[test]
fn extended_prompt_matches_single_prompt() -> Result<()> {
    // A system and user turn, as the chat template lays them out.
    let first: Vec<i32> = vec![
        200002, 200004, 34182, 279, 154554, 150628, 25, 220, 15, 200010, 200000, 200004, 4896,
        200010, 200001, 200004,
    ];
    // The next turn: closes the reply, opens a user turn, and is long enough to go
    // through the prompt path rather than decode.
    let mut next: Vec<i32> = vec![200010, 200000, 200004];
    next.extend((0..40).map(|i| 4896 + 37 * i));
    next.extend([200010, 200001, 200004]);
    assert!(next.len() >= EXTEND_MIN);

    let file = std::fs::File::open(ckpt())?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    let st = SafeTensors::deserialize(&mmap)?;
    let head = HeadBufs::load(&st)?;
    let mut accel = MetalAccel::new()?;
    let mut cache = WeightCache::new();

    // Prompt, six decoded tokens, then the next turn on top.
    let mut warm = warm_prompt_metal(&st, &first, &head, &mut accel, &mut cache, 256)?;
    let mut ids = first.clone();
    let mut id = argmax(&warm.logits[..VOCAB]) as i32;
    for _ in 0..6 {
        ids.push(id);
        id = decode_next_id_metal(&st, id, &head, &mut warm.caches, &mut accel, &mut cache)?;
    }
    ids.push(id);
    // The last decoded token has been picked but not fed yet: it leads the extension.
    let mut tail = vec![id];
    tail.extend_from_slice(&next);
    ids.extend_from_slice(&next);
    let mut stepped = vec![0f32; VOCAB];
    extend_prompt_metal(&st, &tail, &head, &mut warm.caches, &mut accel, &mut cache, &mut stepped)?;

    // And decode keeps working from there: one more token after the extension.
    let follow = decode_next_id_metal(
        &st,
        argmax(&stepped) as i32,
        &head,
        &mut warm.caches,
        &mut accel,
        &mut cache,
    )?;

    // The same sequence as one prompt.
    let whole = warm_prompt_metal(&st, &ids, &head, &mut accel, &mut cache, 256)?;
    let mut maxabs = 0f32;
    for (a, b) in stepped.iter().zip(whole.logits.iter()) {
        maxabs = maxabs.max((a - b).abs());
    }
    eprintln!("extended prompt vs single prompt maxabs={maxabs}");
    assert_eq!(argmax(&stepped), argmax(&whole.logits), "picks differ");
    assert!(maxabs < 5e-3, "extended prompt maxabs={maxabs}");

    let mut whole = whole;
    let follow_whole = decode_next_id_metal(
        &st,
        argmax(&whole.logits) as i32,
        &head,
        &mut whole.caches,
        &mut accel,
        &mut cache,
    )?;
    assert_eq!(follow, follow_whole, "the token after the extension differs");
    Ok(())
}
