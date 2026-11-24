//! The drafter against a recorded mlx-vlm run.
//!
//! `DRAFT_TRACE` names a directory with `trace.json` and `feats.bin`: for every
//! round of a generation, the target states mlx-vlm handed its drafter, the anchor
//! token and the tokens that drafter proposed.  Fed the same states and anchors,
//! this drafter has to propose the same tokens almost every time.  The two differ
//! in precision (BF16 there, f32 here), so a tie can fall the other way.

use anyhow::{Context, Result};
use grouille_inkling::drafter::{Drafter, BLOCK, FEAT};
use grouille_inkling::prefill::HeadBufs;
use grouille_inkling::visit::MetalAccel;
use safetensors::tensor::SafeTensors;
use std::path::PathBuf;

fn snapshot() -> PathBuf {
    PathBuf::from(std::env::var_os("HOME").unwrap())
        .join(".cache/huggingface/hub/models--thinkingmachines--Inkling-2-Mini")
        .join("snapshots/ed40e7da625f594fc9c4990f5f66be24873c22d7")
}

/// The integers of `"key": [..]` in a flat JSON object.
fn int_list(obj: &str, key: &str) -> Vec<i64> {
    let start = obj.find(&format!("\"{key}\": [")).map(|i| i + key.len() + 5);
    let Some(start) = start else { return Vec::new() };
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
fn proposals_match_recorded_run() -> Result<()> {
    let dir = PathBuf::from(std::env::var_os("DRAFT_TRACE").context("set DRAFT_TRACE")?);
    let trace = std::fs::read_to_string(dir.join("trace.json"))?;
    let feats = std::fs::read(dir.join("feats.bin"))?;
    let feats: Vec<f32> =
        feats.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
    let rounds = &trace[trace.find("\"rounds\": [").unwrap()..];
    let rounds: Vec<&str> = rounds.split("{\"anchor\"").skip(1).collect();

    let file = std::fs::File::open(snapshot().join("model-00001-of-00001.safetensors"))?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    let st = SafeTensors::deserialize(&mmap)?;
    let head = HeadBufs::load(&st)?;
    let mtp_file = std::fs::File::open(snapshot().join("mtp.safetensors"))?;
    let mtp_map = unsafe { memmap2::Mmap::map(&mtp_file)? };
    let mtp = SafeTensors::deserialize(&mtp_map)?;
    let mut accel = MetalAccel::new()?;
    let mut drafter = Drafter::load(&mtp, &head, &mut accel.ops)?;

    let (mut row, mut same_blocks, mut same_tokens, mut tokens, mut same_first) = (0usize, 0, 0, 0, 0);
    let mut n_rounds = 0;
    for r in rounds.iter() {
        let r = format!("{{\"anchor\"{r}");
        let anchor = int_field(&r, "anchor") as i32;
        let rows = int_field(&r, "rows") as usize;
        let block = int_field(&r, "block") as usize;
        let want: Vec<i32> = int_list(&r, "draft").iter().map(|v| *v as i32).collect();
        assert!(block <= BLOCK && want.len() == block - 1);

        drafter.push_context(&feats[row * FEAT..(row + rows) * FEAT], rows, &mut accel.ops)?;
        row += rows;
        let got = drafter.draft(anchor, block - 1, &head, &mut accel.ops)?;

        // Tokens that agree before the first difference: after one, the head is
        // conditioned on a different token and the rest is not comparable.
        let agree = got.tokens.iter().zip(&want).take_while(|(a, b)| a == b).count();
        same_tokens += agree;
        tokens += want.len();
        same_blocks += (agree == want.len()) as usize;
        same_first += (agree > 0) as usize;
        n_rounds += 1;
    }
    eprintln!(
        "{n_rounds} rounds, {} context tokens: identical blocks {same_blocks}, first proposal identical {same_first}, tokens identical up to the first difference {same_tokens}/{tokens}",
        drafter.context_len()
    );
    assert!(same_first * 100 >= n_rounds * 95, "first proposals differ too often");
    assert!(same_blocks * 100 >= n_rounds * 85, "blocks differ too often");
    Ok(())
}
