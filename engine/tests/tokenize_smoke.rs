//! Smoke test: load Inkling tokenizer and round-trip a short string.

use anyhow::{bail, Result};
use grouille_inkling::generate_argmax_metal;
use grouille_inkling::prefill::HeadBufs;
use grouille_inkling::tokenize::{default_tokenizer_path, InklingTokenizer};
use grouille_inkling::visit::{MetalAccel, WeightCache};
use safetensors::tensor::SafeTensors;
use std::path::PathBuf;

fn ckpt() -> PathBuf {
    PathBuf::from(std::env::var_os("HOME").unwrap())
        .join(".cache/huggingface/hub/models--thinkingmachines--Inkling-2-Mini")
        .join("snapshots/ed40e7da625f594fc9c4990f5f66be24873c22d7")
        .join("model-00001-of-00001.safetensors")
}

#[test]
fn tokenizer_roundtrip_smoke() -> Result<()> {
    let path = default_tokenizer_path();
    if !path.exists() {
        bail!("missing tokenizer at {}", path.display());
    }
    let tok = InklingTokenizer::from_file(&path)?;
    let text = "hello";
    let ids = tok.encode(text, false)?;
    assert!(!ids.is_empty());
    let back = tok.decode(&ids, true)?;
    eprintln!("tokenize {text:?} → {ids:?} → {back:?}");
    assert!(back.contains("hello") || back == text);
    Ok(())
}

#[test]
fn tokenize_generate_one_token_decodes() -> Result<()> {
    let path = default_tokenizer_path();
    if !path.exists() {
        bail!("missing tokenizer at {}", path.display());
    }
    let tok = InklingTokenizer::from_file(&path)?;
    // Minimal chat-ish fragment so we get a few tokens.
    let prompt = tok.encode("<|begin_of_text|>hi", false)?;
    assert!(prompt.len() >= 2);

    let file = std::fs::File::open(ckpt())?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    let st = SafeTensors::deserialize(&mmap)?;
    let head = HeadBufs::load(&st)?;
    let mut accel = MetalAccel::new()?;
    let mut cache = WeightCache::new();

    let out = generate_argmax_metal(&st, &prompt, 1, &head, &mut accel, &mut cache)?;
    let text = tok.decode(&out, false)?;
    eprintln!("generate ids={:?} text={text:?}", &out[prompt.len()..]);
    assert_eq!(out.len(), prompt.len() + 1);
    Ok(())
}

#[test]
fn format_effort0_opens_content_text() {
    let p = grouille_inkling::format_user_prompt("hi", 0.0);
    assert!(p.ends_with("<|message_model|><|content_text|>"), "{p}");
    assert!(p.contains("Thinking effort level: 0"));
    assert!(p.contains("Answer briefly and directly."));
}

#[test]
fn format_effort_positive_leaves_model_unopened() {
    let p = grouille_inkling::format_user_prompt("hi", 0.9);
    assert!(!p.contains("<|message_model|>"), "{p}");
    assert!(p.contains("Thinking effort level: 0.9"));
}

#[test]
fn format_chat_turns_history_and_system() {
    use grouille_inkling::{format_chat_turns, ChatRole};
    let turns = [
        (ChatRole::User, "My name is Ada."),
        (ChatRole::Assistant, "Hi Ada."),
        (ChatRole::User, "What is my name?"),
    ];
    let p = format_chat_turns(&turns, 0.0, Some("Be concise."));
    assert!(p.contains("<|message_system|><|content_text|>Be concise.<|end_message|>"));
    assert!(p.contains("<|message_user|><|content_text|>My name is Ada.<|end_message|>"));
    assert!(p.contains("<|message_model|><|content_text|>Hi Ada.<|end_message|>"));
    assert!(p.contains("<|message_user|><|content_text|>What is my name?<|end_message|>"));
    assert!(p.ends_with("<|message_model|><|content_text|>"), "{p}");
}
