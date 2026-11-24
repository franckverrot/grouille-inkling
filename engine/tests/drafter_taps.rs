//! The target states this engine hands the drafter against the ones mlx-vlm handed
//! its own, for the same tokens.
//!
//! `DRAFT_TRACE` names a recorded mlx-vlm run (see `drafter_replay.rs`).  Its
//! context tokens go through the prompt path here with the taps on.  mlx-vlm
//! computes in BF16, so the two agree to a percent or so, not to the last digit.

use anyhow::{Context, Result};
use grouille_inkling::decode_visit::warm_prompt_taps_metal;
use grouille_inkling::drafter::{FEAT, TAPS};
use grouille_inkling::ops::HIDDEN;
use grouille_inkling::prefill::HeadBufs;
use grouille_inkling::visit::{MetalAccel, WeightCache};
use safetensors::tensor::SafeTensors;
use std::path::PathBuf;

fn snapshot() -> PathBuf {
    PathBuf::from(std::env::var_os("HOME").unwrap())
        .join(".cache/huggingface/hub/models--thinkingmachines--Inkling-2-Mini")
        .join("snapshots/ed40e7da625f594fc9c4990f5f66be24873c22d7")
}

fn int_list(obj: &str, key: &str) -> Vec<i64> {
    let Some(start) = obj.find(&format!("\"{key}\": [")).map(|i| i + key.len() + 5) else {
        return Vec::new();
    };
    let end = start + obj[start..].find(']').unwrap();
    obj[start..end].split(',').filter_map(|v| v.trim().parse().ok()).collect()
}

fn int_field(obj: &str, key: &str) -> i64 {
    let start = obj.find(&format!("\"{key}\": ")).unwrap() + key.len() + 4;
    let end = start + obj[start..].find([',', '}']).unwrap();
    obj[start..end].trim().parse().unwrap()
}

#[test]
#[ignore]
fn taps_match_recorded_run() -> Result<()> {
    let dir = PathBuf::from(std::env::var_os("DRAFT_TRACE").context("set DRAFT_TRACE")?);
    let trace = std::fs::read_to_string(dir.join("trace.json"))?;
    // `feats_f32.bin` (spike/ref_taps.py) is the same run recomputed in float32 and
    // has to match closely.  Without it the BF16 states are the only reference.
    let exact = dir.join("feats_f32.bin").exists();
    let theirs = std::fs::read(dir.join(if exact { "feats_f32.bin" } else { "feats.bin" }))?;
    let theirs: Vec<f32> =
        theirs.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
    let rows = theirs.len() / FEAT;

    // The context in order: the prompt, then each round's anchor and kept proposals.
    let mut tokens: Vec<i32> = int_list(&trace, "prompt").iter().map(|v| *v as i32).collect();
    let rounds = &trace[trace.find("\"rounds\": [").unwrap()..];
    for r in rounds.split("{\"anchor\"").skip(1) {
        let r = format!("{{\"anchor\"{r}");
        tokens.push(int_field(&r, "anchor") as i32);
        let kept = int_field(&r, "accepted") as usize;
        tokens.extend(int_list(&r, "draft").iter().take(kept).map(|v| *v as i32));
    }
    assert!(tokens.len() >= rows);
    tokens.truncate(rows);

    let file = std::fs::File::open(snapshot().join("model-00001-of-00001.safetensors"))?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    let st = SafeTensors::deserialize(&mmap)?;
    let head = HeadBufs::load(&st)?;
    let mut accel = MetalAccel::new()?;
    let mut cache = WeightCache::new();
    let warm = warm_prompt_taps_metal(&st, &tokens, &head, &mut accel, &mut cache, rows, true)?;
    let ours = warm.caches.taps.as_ref().unwrap();
    assert_eq!(ours.len(), theirs.len());

    let mut failed = false;
    for tap in 0..TAPS {
        let (mut num, mut den, mut worst) = (0f64, 0f64, 0f64);
        for t in 0..rows {
            let o = t * FEAT + tap * HIDDEN;
            let (mut n, mut d) = (0f64, 0f64);
            for (a, b) in ours[o..o + HIDDEN].iter().zip(&theirs[o..o + HIDDEN]) {
                n += ((a - b) as f64).powi(2);
                d += (*b as f64).powi(2);
            }
            num += n;
            den += d;
            worst = worst.max((n / d).sqrt());
        }
        let rel = (num / den).sqrt();
        eprintln!("state {tap}: relative difference {rel:.4} overall, {worst:.4} for the worst token");
        failed |= rel >= if exact { 2e-3 } else { 0.2 };
    }
    assert!(!failed, "a state is too far from the recorded one");
    Ok(())
}
