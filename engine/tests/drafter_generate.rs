//! Generation with the drafter against plain decode.
//!
//! The same tokens have to come out, because the target picks every one of them
//! either way.  And the states the drafter reads have to be the same whether a token
//! went through the prompt path or through a decode step.

use anyhow::Result;
use grouille_inkling::decode_visit::{decode_next_id_metal, generate_argmax_metal, warm_prompt_taps_metal};
use grouille_inkling::drafter::{Drafter, FEAT, TAPS};
use grouille_inkling::ops::HIDDEN;
use grouille_inkling::prefill::HeadBufs;
use grouille_inkling::speculate::{generate_drafted_metal, SpecOptions};
use grouille_inkling::visit::{MetalAccel, WeightCache};
use grouille_inkling::{format_user_prompt, load_default_tokenizer};
use safetensors::tensor::SafeTensors;
use std::path::PathBuf;

fn snapshot() -> PathBuf {
    PathBuf::from(std::env::var_os("HOME").unwrap())
        .join(".cache/huggingface/hub/models--thinkingmachines--Inkling-2-Mini")
        .join("snapshots/ed40e7da625f594fc9c4990f5f66be24873c22d7")
}

const PROMPTS: [&str; 3] = [
    "Write a Python class implementing a binary search tree with insert, delete, search \
     and in-order traversal. Include docstrings. Output only the code.",
    "Output a JSON array with one object per month of the year, each with the keys name, \
     number, days and season (northern hemisphere). Output only the JSON.",
    "Why is the sky blue? Answer at length.",
];

#[test]
fn drafted_generation_matches_plain_decode() -> Result<()> {
    let n_new = 96;
    let tok = load_default_tokenizer()?;
    let file = std::fs::File::open(snapshot().join("model-00001-of-00001.safetensors"))?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    let st = SafeTensors::deserialize(&mmap)?;
    let head = HeadBufs::load(&st)?;
    let mtp_file = std::fs::File::open(snapshot().join("mtp.safetensors"))?;
    let mtp_map = unsafe { memmap2::Mmap::map(&mtp_file)? };
    let mtp = SafeTensors::deserialize(&mtp_map)?;
    let mut accel = MetalAccel::new()?;
    let mut cache = WeightCache::new();
    let mut drafter = Drafter::load(&mtp, &head, &mut accel.ops)?;

    let mut failed = false;
    for text in PROMPTS {
        let prompt = tok.encode(&format_user_prompt(text, 0.0), false)?;
        let plain = generate_argmax_metal(&st, &prompt, n_new, &head, &mut accel, &mut cache)?;
        let plain = &plain[prompt.len()..];
        eprintln!("--- {}", &text[..40.min(text.len())]);
        // Checking every block in full keeps a different number of proposals almost
        // every round.  Left to itself the loop mixes blocks and plain steps.
        for always_full in [true, false] {
            let opts = SpecOptions { always_full, ..Default::default() };
            let (drafted, stats) = generate_drafted_metal(
                &st, &prompt, n_new, &[], &head, &mut drafter, &mut accel, &mut cache, &opts,
            )?;
            let same = plain.iter().zip(&drafted).take_while(|(a, b)| a == b).count();
            eprintln!(
                "  {}: {} blocks gave {} tokens ({} of {} proposals kept), {} plain steps; blocks by tokens gained (1..8): {:?}",
                if always_full { "every block in full" } else { "left to itself" },
                stats.rounds,
                stats.block_tokens(),
                stats.accepted,
                stats.proposed,
                stats.plain_steps,
                &stats.gained[1..]
            );
            if plain != drafted.as_slice() {
                eprintln!("  differs from plain decode at token {same} of {}", plain.len());
                failed = true;
            }
        }
    }
    assert!(!failed, "a drafted generation differs from plain decode");
    Ok(())
}

/// Past the window a block reads a ring that has wrapped and, on the full-attention
/// visits, the archive, and keeping part of a block moves rows into the archive.
#[test]
fn drafted_generation_matches_plain_decode_past_the_window() -> Result<()> {
    let n_new = 40;
    let text = std::fs::read_to_string(
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../goldens/regress/window.prompt.txt"),
    )? + "\n\nSummarize the text above in three sentences.";
    let tok = load_default_tokenizer()?;
    let file = std::fs::File::open(snapshot().join("model-00001-of-00001.safetensors"))?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    let st = SafeTensors::deserialize(&mmap)?;
    let head = HeadBufs::load(&st)?;
    let mtp_file = std::fs::File::open(snapshot().join("mtp.safetensors"))?;
    let mtp_map = unsafe { memmap2::Mmap::map(&mtp_file)? };
    let mtp = SafeTensors::deserialize(&mtp_map)?;
    let mut accel = MetalAccel::new()?;
    let mut cache = WeightCache::new();
    let mut drafter = Drafter::load(&mtp, &head, &mut accel.ops)?;

    let prompt = tok.encode(&format_user_prompt(&text, 0.0), false)?;
    assert!(prompt.len() > 2048, "the prompt has to reach the archive");
    let plain = generate_argmax_metal(&st, &prompt, n_new, &head, &mut accel, &mut cache)?;
    let plain = &plain[prompt.len()..];
    let opts = SpecOptions { always_full: true, ..Default::default() };
    let (drafted, stats) = generate_drafted_metal(
        &st, &prompt, n_new, &[], &head, &mut drafter, &mut accel, &mut cache, &opts,
    )?;
    eprintln!(
        "{} tokens of context: {} blocks gave {} tokens ({} of {} proposals kept); blocks by tokens gained (1..8): {:?}",
        prompt.len(),
        stats.rounds,
        stats.block_tokens(),
        stats.accepted,
        stats.proposed,
        &stats.gained[1..]
    );
    assert!(stats.rounds > 0);
    let same = plain.iter().zip(&drafted).take_while(|(a, b)| a == b).count();
    assert_eq!(plain, drafted.as_slice(), "differs from plain decode at token {same}");
    Ok(())
}

#[test]
fn decode_records_the_states_the_prompt_path_records() -> Result<()> {
    let tok = load_default_tokenizer()?;
    let file = std::fs::File::open(snapshot().join("model-00001-of-00001.safetensors"))?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    let st = SafeTensors::deserialize(&mmap)?;
    let head = HeadBufs::load(&st)?;
    let mut accel = MetalAccel::new()?;
    let mut cache = WeightCache::new();

    // A prompt and the tokens plain decode continues it with.
    let n = 24;
    let prompt = tok.encode(&format_user_prompt(PROMPTS[2], 0.0), false)?;
    let all = generate_argmax_metal(&st, &prompt, n + 1, &head, &mut accel, &mut cache)?;

    // Those tokens as part of the prompt.
    let whole = warm_prompt_taps_metal(&st, &all[..prompt.len() + n], &head, &mut accel, &mut cache, 256, true)?;
    let by_prompt = whole.caches.taps.clone().unwrap();
    drop(whole);

    // And one decode step at a time.
    let mut warm = warm_prompt_taps_metal(&st, &prompt, &head, &mut accel, &mut cache, 256, true)?;
    for &id in &all[prompt.len()..prompt.len() + n] {
        decode_next_id_metal(&st, id, &head, &mut warm.caches, &mut accel, &mut cache)?;
    }
    let by_decode = warm.caches.taps.as_ref().unwrap();
    assert_eq!(by_decode.len(), by_prompt.len());

    for tap in 0..TAPS {
        let mut worst = 0f64;
        for t in prompt.len()..prompt.len() + n {
            let o = t * FEAT + tap * HIDDEN;
            let (mut num, mut den) = (0f64, 0f64);
            for (a, b) in by_decode[o..o + HIDDEN].iter().zip(&by_prompt[o..o + HIDDEN]) {
                num += ((a - b) as f64).powi(2);
                den += (*b as f64).powi(2);
            }
            worst = worst.max((num / den).sqrt());
        }
        eprintln!("state {tap}: decode and prompt differ by {worst:.2e} at worst");
        assert!(worst < 1e-3, "state {tap} differs by {worst}");
    }
    Ok(())
}
