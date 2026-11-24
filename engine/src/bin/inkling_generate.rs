//! Minimal greedy / sample generate CLI for Inkling 2 Mini.

use anyhow::{bail, Context, Result};
use grouille_inkling::prefill::HeadBufs;
use grouille_inkling::tokenize::{
    default_tokenizer_path, format_chat_turns, format_user_prompt, format_user_turn_delta,
    ChatRole, InklingTokenizer,
};
use grouille_inkling::visit::{MetalAccel, WeightCache};
use grouille_inkling::{
    decode_logits_metal_into, decode_next_id_metal, extend_prompt_metal, warm_prompt_taps_metal,
    recycle_decode_caches_with_logits,
    sample_logits_top_k_p, CONTENT_TEXT_TOKEN_ID, CONTENT_THINKING_TOKEN_ID,
    DecodeCaches, END_MESSAGE_TOKEN_ID, EOS_TOKEN_ID, MESSAGE_MODEL_TOKEN_ID, PAD_TOKEN_ID,
    UNPADDED_VOCAB, XorShift32,
};
use safetensors::tensor::SafeTensors;
use std::env;
use std::io::{self, Read, Write};
use std::path::PathBuf;
use std::time::Instant;

fn ckpt() -> PathBuf {
    PathBuf::from(env::var_os("HOME").expect("HOME"))
        .join(".cache/huggingface/hub/models--thinkingmachines--Inkling-2-Mini")
        .join("snapshots/ed40e7da625f594fc9c4990f5f66be24873c22d7")
        .join("model-00001-of-00001.safetensors")
}

fn default_gpk() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../weights/inkling2mini.gpk")
}

fn usage() {
    eprintln!(
        "usage: inkling-generate [options] <prompt|- >\n\
         pass - to read the prompt from stdin, or --file PATH\n\
         defaults: chat-wrap prompt, greedy (temp=0), max=128, stop=EOS/PAD or end of content_text, effort=0, penalty=0\n\
         --raw skips chat-template wrapping; --timing prints warm/decode ms; streaming on by default\n\
         --draft lets the bundled drafter propose several tokens at a time for the model to check (greedy only; same output)\n\
         --no-stream buffers the answer; --think prints the scratchpad; --verbose shows the full transcript\n\
         --repl keeps Metal/weights loaded and reads prompts line-by-line (multi-turn history on)\n\
         --no-history disables REPL chat history (each line is a fresh prompt)\n\
         --max-history N keeps only the last N user/assistant pairs in REPL\n\
         --system TEXT adds an extra system message after the effort header\n\
         --bench N repeats the prompt N times (implies --timing --no-stream -q) and prints median tok/s\n\
         --weights PATH  .gpk (packed) or .safetensors; default prefers weights/inkling2mini.gpk if present\n\
         --device prints the Metal GPU name and exits\n\
         env INKLING_PROFILE=1|visits|moe|step|gpu for warm/decode/MoE/step/gpu-split timing\n\
         env INKLING_PROGRESS=1 prints warm visit N/48 and decode N/max counters"
    );
}

/// Top-`n` (id, logit) pairs, descending. Used by `INKLING_TRACE`.
fn top_n(logits: &[f32], n: usize) -> Vec<(usize, f32)> {
    let mut ranked: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    let n = n.min(ranked.len());
    ranked.select_nth_unstable_by(n - 1, |a, b| {
        b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
    });
    ranked.truncate(n);
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    ranked
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Channel {
    None,
    Thinking,
    Text,
}

/// REPL multi-turn KV: caches already hold `ids` (last logits ready to sample).
struct ReplSession {
    caches: DecodeCaches,
    ids: Vec<i32>,
    logits: Vec<f32>,
}

fn main() -> Result<()> {
    let mut args: Vec<String> = env::args().skip(1).collect();
    if args.iter().any(|a| a == "-h" || a == "--help") {
        usage();
        return Ok(());
    }
    let want_repl = args.iter().any(|a| a == "--repl");
    if args.is_empty() && !want_repl {
        usage();
        bail!("missing prompt");
    }

    let mut temperature = 0.0f32;
    let mut draft = false;
    let mut top_k = 0usize;
    let mut top_p = 0.0f32;
    let mut effort = 0.0f32;
    let mut penalty = 0.0f32;
    let mut seed = 1u32;
    let mut max_new = 128usize;
    let mut max_set = false;
    let mut raw = false;
    let mut timing = false;
    let mut stream = true;
    let mut show_think = false;
    let mut verbose = false;
    let mut quiet = false;
    let mut debug_logits = false;
    let mut repl = false;
    let mut history = true;
    let mut system_extra: Option<String> = None;
    let mut bench_n = 0usize;
    let mut max_history = usize::MAX;
    let mut weights_path: Option<PathBuf> = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--weights" => {
                weights_path = Some(PathBuf::from(
                    args.get(i + 1).context("--weights needs a path")?,
                ));
                args.drain(i..i + 2);
            }
            "--max-history" => {
                max_history = args
                    .get(i + 1)
                    .context("--max-history needs a count")?
                    .parse()?;
                args.drain(i..i + 2);
            }
            "--bench" => {
                bench_n = args
                    .get(i + 1)
                    .context("--bench needs a count")?
                    .parse()?;
                args.drain(i..i + 2);
                timing = true;
                stream = false;
                quiet = true;
            }
            "--device" => {
                let accel = MetalAccel::new()?;
                eprintln!("Metal: {}", accel.moe.device_name());
                return Ok(());
            }
            "--repl" => {
                repl = true;
                args.remove(i);
            }
            "--no-history" => {
                history = false;
                args.remove(i);
            }
            "--system" => {
                system_extra = Some(
                    args.get(i + 1)
                        .context("--system needs a value")?
                        .clone(),
                );
                args.drain(i..i + 2);
            }
            "--file" => {
                let path = args.get(i + 1).context("--file needs a path")?.clone();
                let text = std::fs::read_to_string(&path)
                    .with_context(|| format!("read --file {path}"))?;
                args.drain(i..i + 2);
                args.insert(0, text);
            }
            "--raw" => {
                raw = true;
                args.remove(i);
            }
            "--timing" => {
                timing = true;
                args.remove(i);
            }
            "--stream" => {
                stream = true;
                args.remove(i);
            }
            "--no-stream" => {
                stream = false;
                args.remove(i);
            }
            "--think" => {
                show_think = true;
                args.remove(i);
            }
            "--verbose" | "-v" => {
                verbose = true;
                args.remove(i);
            }
            "--quiet" | "-q" => {
                quiet = true;
                args.remove(i);
            }
            "--debug-logits" => {
                debug_logits = true;
                args.remove(i);
            }
            "--draft" => {
                draft = true;
                args.remove(i);
            }
            "--temp" => {
                temperature = args
                    .get(i + 1)
                    .context("--temp needs a value")?
                    .parse()?;
                args.drain(i..i + 2);
            }
            "--top-k" => {
                top_k = args
                    .get(i + 1)
                    .context("--top-k needs a value")?
                    .parse()?;
                args.drain(i..i + 2);
            }
            "--top-p" => {
                top_p = args
                    .get(i + 1)
                    .context("--top-p needs a value")?
                    .parse()?;
                args.drain(i..i + 2);
            }
            "--effort" => {
                effort = args
                    .get(i + 1)
                    .context("--effort needs a value")?
                    .parse()?;
                args.drain(i..i + 2);
            }
            "--penalty" => {
                penalty = args
                    .get(i + 1)
                    .context("--penalty needs a value")?
                    .parse()?;
                args.drain(i..i + 2);
            }
            "--seed" => {
                seed = args
                    .get(i + 1)
                    .context("--seed needs a value")?
                    .parse()?;
                args.drain(i..i + 2);
            }
            "--max" => {
                max_new = args
                    .get(i + 1)
                    .context("--max needs a value")?
                    .parse()?;
                max_set = true;
                args.drain(i..i + 2);
            }
            _ => i += 1,
        }
    }
    if penalty <= 0.0 && effort > 0.0 && !raw {
        penalty = 0.8;
    }
    if effort > 0.0 && !max_set {
        // Default --max; thinking rarely finishes content_text in 128 tokens.
        max_new = 256;
        if !quiet {
            eprintln!("note: raised --max to 256 for --effort (pass --max to override)");
        }
    } else if effort > 0.0 && max_new < 64 && !quiet {
        eprintln!(
            "note: --effort={effort} often needs --max ≥ 64 to finish content_text (have {max_new})"
        );
    }

    let tok = InklingTokenizer::from_file(default_tokenizer_path())?;
    // Always keep HF safetensors mapped for API compatibility / fallback loads.
    let file = std::fs::File::open(ckpt()).context("open checkpoint")?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    let st = SafeTensors::deserialize(&mmap)?;
    let mut accel = MetalAccel::new()?;
    let mut cache = WeightCache::new();
    let weights = weights_path.unwrap_or_else(|| {
        let gpk = default_gpk();
        if gpk.exists() {
            gpk
        } else {
            ckpt()
        }
    });
    let use_gpk = weights
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.eq_ignore_ascii_case("gpk"))
        .unwrap_or(false);
    // Keep `.gpk` alive for the process: Metal buffer is bytesNoCopy over its mmap.
    let _gpk_keep: Option<grouille_inkling::gpk::GpkFile>;
    let head = if use_gpk {
        if !quiet {
            eprintln!("weights: {}", weights.display());
        }
        let gpk = grouille_inkling::gpk::GpkFile::open(&weights, accel.moe.gpu())?;
        grouille_inkling::gpk::install_moe_views(&gpk, &mut accel.moe)?;
        grouille_inkling::gpk::install_attn_views(&gpk, &mut accel.ops)?;
        cache.preload_all_from_gpk(&gpk)?;
        let head = grouille_inkling::gpk::GpkHead::load(&gpk)?.to_head_bufs();
        _gpk_keep = Some(gpk);
        head
    } else {
        if !quiet {
            eprintln!("weights: {}", weights.display());
        }
        _gpk_keep = None;
        HeadBufs::load(&st)?
    };
    let mut rng = XorShift32::new(seed);
    // The drafter proposes several tokens at a time for the target to check.  It
    // only works with greedy picks.
    let spec = std::cell::RefCell::new(if draft {
        let mtp_file = std::fs::File::open(ckpt().with_file_name("mtp.safetensors")).context("open mtp.safetensors")?;
        let mtp_map = unsafe { memmap2::Mmap::map(&mtp_file)? };
        let mtp = SafeTensors::deserialize(&mtp_map)?;
        let drafter = grouille_inkling::drafter::Drafter::load(&mtp, &head, &mut accel.ops)?;
        Some((grouille_inkling::speculate::Speculator::new(Default::default()), drafter))
    } else {
        None
    });

    let run_one = |user_text: &str,
                   prior: &[(ChatRole, String)],
                   effort_now: f32,
                   sys_now: Option<&str>,
                   timing_now: bool,
                   max_now: usize,
                   temp_now: f32,
                   top_k_now: usize,
                   top_p_now: f32,
                   penalty_now: f32,
                   think_now: bool,
                   accel: &mut MetalAccel,
                   cache: &mut WeightCache,
                   rng: &mut XorShift32,
                   session: &mut Option<ReplSession>,
                   save_session: bool|
     -> Result<String> {
        let t_warm = Instant::now();
        let mut continued = false;
        let mut continue_prefix = 0usize;
        let (mut caches, mut logits, prompt) = if raw {
            let prompt = tok.encode(user_text, false)?;
            if !quiet {
                eprintln!("prompt tokens={}", prompt.len());
            }
            if timing_now && !quiet {
                eprintln!("warming prompt ({} tokens)…", prompt.len());
                let _ = io::stderr().flush();
            }
            let warm = warm_prompt_taps_metal(
                &st,
                &prompt,
                &head,
                accel,
                cache,
                prompt.len() + max_now + 64,
                draft,
            )?;
            (warm.caches, warm.logits, prompt)
        } else if save_session {
            match session.take() {
                Some(mut sess) => {
                    let delta =
                        tok.encode(&format_user_turn_delta(user_text, effort_now), false)?;
                    let prefix = sess.ids.len();
                    let mut prompt = std::mem::take(&mut sess.ids);
                    prompt.extend_from_slice(&delta);
                    if !quiet {
                        eprintln!("prompt tokens={}", prompt.len());
                    }
                    // The conversation so far stays where it is; only the new turn is
                    // processed.
                    continued = true;
                    continue_prefix = prefix;
                    if timing_now && !quiet {
                        eprintln!(
                            "continuing ({prefix} → {} tokens)…",
                            prompt.len()
                        );
                        let _ = io::stderr().flush();
                    }
                    extend_prompt_metal(
                        &st,
                        &prompt[prefix..],
                        &head,
                        &mut sess.caches,
                        accel,
                        cache,
                        &mut sess.logits,
                    )?;
                    (sess.caches, sess.logits, prompt)
                }
                None => {
                    let prompt = if prior.is_empty() && sys_now.is_none() {
                        tok.encode(&format_user_prompt(user_text, effort_now), false)?
                    } else {
                        let mut turns: Vec<(ChatRole, &str)> = prior
                            .iter()
                            .map(|(r, t)| (*r, t.as_str()))
                            .collect();
                        turns.push((ChatRole::User, user_text));
                        tok.encode(&format_chat_turns(&turns, effort_now, sys_now), false)?
                    };
                    if !quiet {
                        eprintln!("prompt tokens={}", prompt.len());
                    }
                    if timing_now && !quiet {
                        eprintln!("warming prompt ({} tokens)…", prompt.len());
                        let _ = io::stderr().flush();
                    }
                    let warm = warm_prompt_taps_metal(
                        &st,
                        &prompt,
                        &head,
                        accel,
                        cache,
                        prompt.len() + max_now + 64,
                        draft,
                    )?;
                    (warm.caches, warm.logits, prompt)
                }
            }
        } else if prior.is_empty() && sys_now.is_none() {
            let prompt = tok.encode(&format_user_prompt(user_text, effort_now), false)?;
            if !quiet {
                eprintln!("prompt tokens={}", prompt.len());
            }
            if timing_now && !quiet {
                eprintln!("warming prompt ({} tokens)…", prompt.len());
                let _ = io::stderr().flush();
            }
            let warm = warm_prompt_taps_metal(
                &st,
                &prompt,
                &head,
                accel,
                cache,
                prompt.len() + max_now + 64,
                draft,
            )?;
            (warm.caches, warm.logits, prompt)
        } else {
            let mut turns: Vec<(ChatRole, &str)> = prior
                .iter()
                .map(|(r, t)| (*r, t.as_str()))
                .collect();
            turns.push((ChatRole::User, user_text));
            let prompt = tok.encode(&format_chat_turns(&turns, effort_now, sys_now), false)?;
            if !quiet {
                eprintln!("prompt tokens={}", prompt.len());
            }
            if timing_now && !quiet {
                eprintln!("warming prompt ({} tokens)…", prompt.len());
                let _ = io::stderr().flush();
            }
            let warm = warm_prompt_taps_metal(
                &st,
                &prompt,
                &head,
                accel,
                cache,
                prompt.len() + max_now + 64,
                draft,
            )?;
            (warm.caches, warm.logits, prompt)
        };
        let warm_ms = t_warm.elapsed().as_secs_f64() * 1000.0;
        if debug_logits {
            let mut ranked: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
            ranked.select_nth_unstable_by(9, |a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            });
            ranked.truncate(10);
            ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            eprintln!("--- top logits after warm ---");
            for (i, (id, v)) in ranked.iter().enumerate() {
                let piece = tok.decode(&[*id as i32], false).unwrap_or_default();
                eprintln!("  {i}: id={id} logit={v:.3} text={piece:?}");
            }
        }
        let mut out = prompt;
        let prompt_len = out.len();
        let mut channel = match out.last().copied() {
            Some(CONTENT_TEXT_TOKEN_ID) => Channel::Text,
            Some(CONTENT_THINKING_TOKEN_ID) => Channel::Thinking,
            _ => Channel::None,
        };
        let mut answer = String::new();
        if stream && !quiet {
            eprintln!("--- answer ---");
            let _ = io::stderr().flush();
        }
        let t_dec = Instant::now();
        let mut stopped_on_end_message = false;
        // Fresh logits each step; keep seen ids so we do not rebuild them O(n)
        // per token on long --max runs.
        // Presence over the whole prompt + generated so far (HF-style), not
        // only the new tokens, otherwise --penalty ignores the user turn.
        // Bitset + dense id list; TLS reuses the ~200 KB mask across REPL turns.
        thread_local! {
            static SEEN_MASK: std::cell::RefCell<Vec<bool>> =
                const { std::cell::RefCell::new(Vec::new()) };
            static SEEN_IDS: std::cell::RefCell<Vec<usize>> =
                const { std::cell::RefCell::new(Vec::new()) };
        }
        let (mut seen_mask, mut seen_ids) = if penalty_now > 0.0 {
            SEEN_MASK.with(|m| {
                SEEN_IDS.with(|ids| {
                    let mut mask = std::mem::take(&mut *m.borrow_mut());
                    let mut ids = std::mem::take(&mut *ids.borrow_mut());
                    if mask.len() < UNPADDED_VOCAB {
                        mask.resize(UNPADDED_VOCAB, false);
                    } else {
                        for &i in ids.iter() {
                            if i < mask.len() {
                                mask[i] = false;
                            }
                        }
                    }
                    ids.clear();
                    (mask, ids)
                })
            })
        } else {
            (Vec::new(), Vec::new())
        };
        if penalty_now > 0.0 {
            for &id in &out {
                let id = id as usize;
                if id < UNPADDED_VOCAB && !seen_mask[id] {
                    seen_mask[id] = true;
                    seen_ids.push(id);
                }
            }
        }
        let mut first_decode_ms: Option<f64> = None;
        // Greedy + no presence penalty: decode can return the next id via GPU
        // argmax (skip ~UNPADDED_VOCAB logits readback each step).
        // INKLING_TRACE=path dumps prompt/generated ids as JSON for the reference
        // check (spike/ref_check.py). INKLING_TRACE_LOGITS=1 also forces the full
        // logits path and records the top-8 logits per step.
        let trace_path = std::env::var_os("INKLING_TRACE");
        let trace_logits = trace_path.is_some() && std::env::var_os("INKLING_TRACE_LOGITS").is_some();
        let mut trace_top: Vec<Vec<(usize, f32)>> = Vec::new();
        // INKLING_FORCE_IDS=trace.json teacher-forces that trace's gen_ids (the
        // engine's own choice per step is still recorded as argmax_ids).
        let forced: Option<Vec<i32>> = match std::env::var_os("INKLING_FORCE_IDS") {
            Some(path) => {
                let text = std::fs::read_to_string(&path).context("read INKLING_FORCE_IDS")?;
                let key = "\"gen_ids\":[";
                let start = text.find(key).context("gen_ids missing")? + key.len();
                let end = start + text[start..].find(']').context("gen_ids unterminated")?;
                Some(
                    text[start..end]
                        .split(',')
                        .filter(|t| !t.trim().is_empty())
                        .map(|t| t.trim().parse::<i32>())
                        .collect::<std::result::Result<Vec<_>, _>>()
                        .context("parse gen_ids")?,
                )
            }
            None => None,
        };
        let mut trace_argmax: Vec<i32> = Vec::new();
        let greedy_argmax = temp_now <= 0.0
            && top_k_now == 0
            && top_p_now <= 0.0
            && penalty_now <= 0.0
            && !trace_logits;
        let mut pending_next: Option<i32> = None;
        // With the drafter, tokens arrive several at a time.
        let mut spec_ref = spec.borrow_mut();
        let mut spec_now = spec_ref.as_mut().filter(|_| greedy_argmax && forced.is_none());
        if let Some((sp, drafter)) = spec_now.as_mut() {
            if !continued {
                sp.reset();
                drafter.reset();
            }
        }
        let mut arrived: std::collections::VecDeque<i32> = Default::default();
        for i in 0..max_now {
            if penalty_now > 0.0 {
                for &idx in &seen_ids {
                    logits[idx] -= penalty_now;
                }
            }
            let chosen = if let Some(p) = pending_next.take() {
                p
            } else {
                if trace_logits {
                    trace_top.push(top_n(&logits, 8));
                }
                sample_logits_top_k_p(&logits, temp_now, top_k_now, top_p_now, rng)
            };
            trace_argmax.push(chosen);
            let next = match &forced {
                Some(f) if i < f.len() => f[i],
                Some(_) => break,
                None => chosen,
            };
            out.push(next);
            if penalty_now > 0.0 {
                let id = next as usize;
                if id < UNPADDED_VOCAB && !seen_mask[id] {
                    seen_mask[id] = true;
                    seen_ids.push(id);
                }
            }

            let mut stop_after_answer = false;
            match next {
                MESSAGE_MODEL_TOKEN_ID => channel = Channel::None,
                CONTENT_THINKING_TOKEN_ID => {
                    channel = Channel::Thinking;
                    if think_now {
                        eprint!("[think] ");
                        let _ = io::stderr().flush();
                    }
                }
                CONTENT_TEXT_TOKEN_ID => channel = Channel::Text,
                END_MESSAGE_TOKEN_ID => {
                    if channel == Channel::Text {
                        stop_after_answer = true;
                        stopped_on_end_message = true;
                    }
                    if think_now && channel == Channel::Thinking {
                        eprintln!();
                    }
                    channel = Channel::None;
                }
                _ => {
                    let piece = tok.decode(&[next], false)?;
                    match channel {
                        Channel::Text => {
                            answer.push_str(&piece);
                            if stream {
                                print!("{piece}");
                                let _ = io::stdout().flush();
                            }
                        }
                        Channel::Thinking if think_now => {
                            eprint!("{piece}");
                            let _ = io::stderr().flush();
                        }
                        _ => {}
                    }
                }
            }

            if next == EOS_TOKEN_ID || next == PAD_TOKEN_ID || stop_after_answer {
                break;
            }
            if i + 1 < max_now {
                let t_step = timing_now.then(Instant::now);
                if let Some((sp, drafter)) = spec_now.as_mut() {
                    // Every token that has arrived but the last is already processed.
                    if arrived.is_empty() {
                        arrived.extend(sp.step(
                            &st,
                            next,
                            max_now - i - 1,
                            &[EOS_TOKEN_ID, PAD_TOKEN_ID, END_MESSAGE_TOKEN_ID],
                            &head,
                            drafter,
                            &mut caches,
                            accel,
                            cache,
                        )?);
                    }
                    pending_next = arrived.pop_front();
                } else if greedy_argmax {
                    pending_next = Some(decode_next_id_metal(
                        &st, next, &head, &mut caches, accel, cache,
                    )?);
                } else {
                    decode_logits_metal_into(
                        &st, next, &head, &mut caches, accel, cache, &mut logits,
                    )?;
                }
                if let Some(t0) = t_step {
                    if first_decode_ms.is_none() {
                        first_decode_ms = Some(t0.elapsed().as_secs_f64() * 1000.0);
                    }
                }
                if (grouille_inkling::flags::progress() || (timing_now && !quiet))
                    && (i + 1) % 8 == 0
                {
                    let ms = t_dec.elapsed().as_secs_f64() * 1000.0;
                    let n = (i + 1) as f64;
                    let tok_s = n / (ms / 1000.0).max(1e-6);
                    if grouille_inkling::flags::progress() {
                        eprint!("\rdecode {}/{} [{tok_s:.1} tok/s] ", i + 1, max_now);
                    } else {
                        eprint!("[{tok_s:.1} tok/s] ");
                    }
                    let _ = io::stderr().flush();
                }
            }
        }
        // Fold END_MESSAGE into KV so the next REPL turn can continue as a prefix.
        let can_continue = save_session && stopped_on_end_message;
        if can_continue {
            if let Some(&end_id) = out.last() {
                if caches.taps.is_some() {
                    // The step that also records what the drafter reads.
                    decode_next_id_metal(&st, end_id, &head, &mut caches, accel, cache)?;
                } else {
                    decode_logits_metal_into(
                        &st, end_id, &head, &mut caches, accel, cache, &mut logits,
                    )?;
                }
            }
        }
        let dec_ms = t_dec.elapsed().as_secs_f64() * 1000.0;
        if let Some(path) = &trace_path {
            let ids = |v: &[i32]| v.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(",");
            let top = trace_top
                .iter()
                .map(|step| {
                    let inner = step
                        .iter()
                        .map(|(id, v)| format!("[{id},{v}]"))
                        .collect::<Vec<_>>()
                        .join(",");
                    format!("[{inner}]")
                })
                .collect::<Vec<_>>()
                .join(",");
            let json = format!(
                "{{\"prompt_ids\":[{}],\"gen_ids\":[{}],\"argmax_ids\":[{}],\"top\":[{}]}}\n",
                ids(&out[..prompt_len]),
                ids(&out[prompt_len..]),
                ids(&trace_argmax),
                top
            );
            std::fs::write(path, json).context("write INKLING_TRACE")?;
        }
        if grouille_inkling::flags::progress() {
            eprintln!();
        }
        if stream {
            println!();
        }
        let n_gen = out.len().saturating_sub(prompt_len);
        let decode_steps = n_gen.saturating_sub(1);
        if let (true, Some((sp, _))) = (timing_now, spec_now.as_ref()) {
            let s = &sp.stats;
            eprintln!(
                "drafter: {} blocks gave {} tokens ({} of {} proposals kept), {} plain steps, {} drafts",
                s.rounds,
                s.block_tokens(),
                s.accepted,
                s.proposed,
                s.plain_steps,
                s.drafts
            );
        }
        if timing_now {
            let n_gen_rate = n_gen.max(1);
            let tok_s = if dec_ms > 0.0 {
                (n_gen_rate as f64) / (dec_ms / 1000.0)
            } else {
                0.0
            };
            let warm_label = if continued { "continue" } else { "warm" };
            let step_ms = if decode_steps > 0 {
                dec_ms / decode_steps as f64
            } else {
                0.0
            };
            let first = first_decode_ms
                .map(|ms| format!(", first_decode={ms:.0} ms"))
                .unwrap_or_default();
            if continued {
                eprintln!(
                    "timing: {warm_label}={warm_ms:.0} ms (kept {continue_prefix} tokens), decode={dec_ms:.0} ms total, ~{step_ms:.0} ms/decode-step{first} over {decode_steps} steps ({n_gen_rate} new incl. first, {tok_s:.2} tok/s)"
                );
            } else {
                eprintln!(
                    "timing: {warm_label}={warm_ms:.0} ms, decode={dec_ms:.0} ms total, ~{step_ms:.0} ms/decode-step{first} over {decode_steps} steps ({n_gen_rate} new incl. first, {tok_s:.2} tok/s)"
                );
            }
        } else if !quiet && bench_n == 0 && save_session {
            eprintln!("({n_gen} new tokens, prompt {})", prompt_len);
        }
        if !stream && bench_n == 0 {
            println!("{answer}");
        }
        if answer.is_empty() && think_now && !quiet && bench_n == 0 {
            eprintln!(
                "(no content_text before --max={max_now}; raise --max or lower --effort)"
            );
        }
        if save_session && !can_continue && bench_n == 0 && !raw && (!quiet || timing_now) {
            eprintln!(
                "(no <|end_message|> before --max={max_now}; dropping KV; raise --max to keep multi-turn caches)"
            );
        }
        if verbose {
            let full = tok.decode(&out, false)?;
            eprintln!("--- full ---");
            eprintln!("{full}");
        }
        if penalty_now > 0.0 {
            SEEN_MASK.with(|m| {
                SEEN_IDS.with(|ids| {
                    *m.borrow_mut() = seen_mask;
                    *ids.borrow_mut() = seen_ids;
                })
            });
        }
        if can_continue {
            *session = Some(ReplSession {
                caches,
                ids: out,
                logits,
            });
        } else {
            recycle_decode_caches_with_logits(caches, Some(logits));
            if save_session {
                *session = None;
            }
        }
        Ok(answer)
    };

    if repl {
        eprintln!(
            "inkling-generate REPL on {}: /quit or Ctrl-D to exit",
            accel.moe.device_name()
        );
        eprintln!(
            "commands: /clear  /rewarm  /effort <n>  /max <n>  /temp <n>  /top_p <n>  /top_k <n>  /penalty <n>  /seed <n>  /think  /system <text>  /timing  /device  /status  /history  /quit  /help"
        );
        if history && !raw {
            eprintln!(
                "(multi-turn history on; appends new user turns onto KV when the last answer closed cleanly)"
            );
        }
        let mut prior: Vec<(ChatRole, String)> = Vec::new();
        let mut session: Option<ReplSession> = None;
        let drop_session = |session: &mut Option<ReplSession>| {
            if let Some(s) = session.take() {
                recycle_decode_caches_with_logits(s.caches, Some(s.logits));
            }
        };
        let stdin = io::stdin();
        loop {
            eprint!("> ");
            let _ = io::stderr().flush();
            let mut line = String::new();
            if stdin.read_line(&mut line)? == 0 {
                break;
            }
            let line = line.trim_end();
            if line.is_empty() {
                continue;
            }
            if line.starts_with('/') {
                let cmd = line.split_whitespace().next().unwrap_or("");
                let rest = line[cmd.len()..].trim_start();
                match cmd {
                    "/quit" | "/exit" => break,
                    "/clear" => {
                        prior.clear();
                        drop_session(&mut session);
                        eprintln!("(history + KV cleared)");
                    }
                    "/rewarm" => {
                        drop_session(&mut session);
                        eprintln!("(KV dropped; next turn rewarms from history)");
                    }
                    "/effort" => {
                        if rest.is_empty() {
                            eprintln!("(effort={effort}; usage: /effort <n>)");
                        } else {
                            effort = rest.parse().context("/effort needs a number")?;
                            if effort > 0.0 && max_new < 256 {
                                max_new = 256;
                                eprintln!(
                                    "(effort={effort}; raised max to {max_new} so content_text can finish)"
                                );
                            } else {
                                eprintln!("(effort={effort})");
                            }
                            if effort > 0.0 && penalty <= 0.0 {
                                penalty = 0.8;
                                eprintln!("(penalty={penalty} for thinking)");
                            }
                        }
                    }
                    "/system" => {
                        if rest.is_empty() {
                            system_extra = None;
                            eprintln!("(system message cleared)");
                        } else {
                            system_extra = Some(rest.to_string());
                            eprintln!("(system set)");
                        }
                        // System lives in the prompt prefix; drop KV so the next
                        // turn rewarms with the new header.
                        drop_session(&mut session);
                    }
                    "/max" => {
                        if rest.is_empty() {
                            eprintln!("(max={max_new}; usage: /max <n>)");
                        } else {
                            max_new = rest.parse().context("/max needs a number")?;
                            eprintln!("(max={max_new})");
                        }
                    }
                    "/temp" | "/temperature" => {
                        if rest.is_empty() {
                            eprintln!("(temperature={temperature}; usage: /temp <n>, 0 = greedy)");
                        } else {
                            temperature = rest.parse().context("/temp needs a number")?;
                            eprintln!("(temperature={temperature})");
                        }
                    }
                    "/top_p" | "/top-p" => {
                        if rest.is_empty() {
                            eprintln!("(top_p={top_p}; usage: /top_p <n>, 0 = off)");
                        } else {
                            top_p = rest.parse().context("/top_p needs a number")?;
                            eprintln!("(top_p={top_p})");
                        }
                    }
                    "/top_k" | "/top-k" => {
                        if rest.is_empty() {
                            eprintln!("(top_k={top_k}; usage: /top_k <n>, 0 = off)");
                        } else {
                            top_k = rest.parse().context("/top_k needs a number")?;
                            eprintln!("(top_k={top_k})");
                        }
                    }
                    "/penalty" => {
                        if rest.is_empty() {
                            eprintln!("(penalty={penalty}; usage: /penalty <n>, 0 = off)");
                        } else {
                            penalty = rest.parse().context("/penalty needs a number")?;
                            eprintln!("(penalty={penalty})");
                        }
                    }
                    "/seed" => {
                        if rest.is_empty() {
                            eprintln!("(seed={seed}; usage: /seed <u32>)");
                        } else {
                            seed = rest.parse().context("/seed needs a u32")?;
                            rng = XorShift32::new(seed);
                            eprintln!("(seed={seed})");
                        }
                    }
                    "/think" => {
                        show_think = !show_think;
                        eprintln!("(think={show_think})");
                    }
                    "/timing" => {
                        timing = !timing;
                        eprintln!("(timing={timing})");
                    }
                    "/device" => {
                        eprintln!("(device={})", accel.moe.device_name());
                    }
                    "/status" => {
                        let kv_tokens = session.as_ref().map(|s| s.ids.len()).unwrap_or(0);
                        eprintln!(
                            "(device={} effort={effort} max={max_new} temp={temperature} top_k={top_k} top_p={top_p} penalty={penalty} seed={seed} think={show_think} timing={timing} history={} kv_tokens={kv_tokens} layers={} attn_preload={} warm_pool={})",
                            accel.moe.device_name(),
                            if history && !raw { prior.len() / 2 } else { 0 },
                            cache.len(),
                            cache.attn_gemv_preloaded(),
                            grouille_inkling::warm_caches_pooled(),
                        );
                    }
                    "/history" => {
                        if prior.is_empty() {
                            eprintln!("(no history)");
                        } else {
                            for (i, (role, text)) in prior.iter().enumerate() {
                                let tag = match role {
                                    ChatRole::User => "user",
                                    ChatRole::Assistant => "assistant",
                                };
                                let preview: String = text.chars().take(120).collect();
                                let ellipsis = if text.chars().count() > 120 { "…" } else { "" };
                                eprintln!("  [{i}] {tag}: {preview}{ellipsis}");
                            }
                        }
                    }
                    "/help" => {
                        eprintln!("/clear: drop chat history and KV");
                        eprintln!("/rewarm: drop KV only (keep text history)");
                        eprintln!("/effort <n>: set thinking effort (0 = direct answer)");
                        eprintln!("/max <n>: set max new tokens");
                        eprintln!("/temp <n>: sampling temperature (0 = greedy)");
                        eprintln!("/top_p <n>: nucleus sampling (0 = off)");
                        eprintln!("/top_k <n>: top-k sampling (0 = off)");
                        eprintln!("/penalty <n>: presence penalty over prompt+output (0 = off)");
                        eprintln!("/seed <n>: reset RNG seed");
                        eprintln!("/think: toggle streaming the thinking channel");
                        eprintln!("/system <text>: set/clear extra system message");
                        eprintln!("/timing: toggle warm/decode timing lines");
                        eprintln!("/device: print Metal GPU name");
                        eprintln!("/status: print current sampling / session knobs");
                        eprintln!("/history: list prior user/assistant turns");
                        eprintln!("/help: this text");
                        eprintln!("/quit: exit (also /exit or Ctrl-D)");
                    }
                    other => eprintln!("unknown command {other} (try /help)"),
                }
                continue;
            }
            let keep_kv = history && !raw;
            let answer = run_one(
                line,
                &prior,
                effort,
                system_extra.as_deref(),
                timing,
                max_new,
                temperature,
                top_k,
                top_p,
                penalty,
                show_think,
                &mut accel,
                &mut cache,
                &mut rng,
                &mut session,
                keep_kv,
            )?;
            if history && !raw {
                prior.push((ChatRole::User, line.to_string()));
                prior.push((ChatRole::Assistant, answer));
                if max_history < usize::MAX {
                    let max_entries = max_history.saturating_mul(2);
                    if prior.len() > max_entries {
                        let drop = prior.len() - max_entries;
                        prior.drain(0..drop);
                        drop_session(&mut session); // prefix no longer valid after history trim
                    }
                }
            } else {
                drop_session(&mut session);
            }
        }
        return Ok(());
    }

    let user_text = if args.len() == 1 && args[0] == "-" {
        let mut buf = String::new();
        io::stdin().read_to_string(&mut buf)?;
        buf
    } else {
        args.join(" ")
    };
    let user_text = user_text.trim_end().to_string();
    if user_text.is_empty() {
        bail!("missing prompt");
    }
    if bench_n > 0 {
        if repl {
            bail!("--bench cannot be combined with --repl");
        }
        let mut rates = Vec::with_capacity(bench_n);
        for i in 0..bench_n {
            let t0 = Instant::now();
            let mut no_sess = None;
            let _ = run_one(
                &user_text,
                &[],
                effort,
                system_extra.as_deref(),
                timing,
                max_new,
                temperature,
                top_k,
                top_p,
                penalty,
                show_think,
                &mut accel,
                &mut cache,
                &mut rng,
                &mut no_sess,
                false,
            )?;
            let ms = t0.elapsed().as_secs_f64() * 1000.0;
            // run_one already printed timing lines; collect wall tok/s approx via max_new.
            rates.push(ms);
            eprintln!("bench {i}: wall={ms:.0} ms");
        }
        let steady = if rates.len() > 1 { &rates[1..] } else { &rates[..] };
        let mut sorted = steady.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let med = sorted[sorted.len() / 2];
        let approx_tok_s = if med > 0.0 {
            (max_new as f64) / (med / 1000.0)
        } else {
            0.0
        };
        eprintln!(
            "bench median wall={med:.0} ms (~{approx_tok_s:.2} tok/s over max={max_new}) over {} steady runs (skip cold first; warm+decode)",
            sorted.len()
        );
        return Ok(());
    }
    let mut no_sess = None;
    let _ = run_one(
        &user_text,
        &[],
        effort,
        system_extra.as_deref(),
        timing,
        max_new,
        temperature,
        top_k,
        top_p,
        penalty,
        show_think,
        &mut accel,
        &mut cache,
        &mut rng,
        &mut no_sess,
        false,
    )?;
    Ok(())
}
