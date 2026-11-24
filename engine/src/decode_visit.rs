//! Single-token decode through the visit schedule (48 visits / caches).

use anyhow::Result;
use safetensors::tensor::SafeTensors;

use crate::cache::DecodeCaches;
use crate::decoder::sliding_layer_decode_step_msgs;
use crate::head::{
    embed_tokens, unembed_metal, EOS_TOKEN_ID, LOGIT_SOFTCAP, PAD_TOKEN_ID,
    UNPADDED_VOCAB, VOCAB,
};
use crate::layer5::full_layer_decode_step_msgs;
use crate::loop_msg::{gather_messages, MSG_SLOTS};
use crate::loop_sched::{
    layer_indices, rebuild_message_bank_into, CORE_END, CORE_START, N_VISITS,
};
use crate::ops::HIDDEN;
use crate::prefill::{finalize_hidden, HeadBufs};
use crate::visit::{MetalAccel, WeightCache};
use crate::weights::is_full_attn;

/// Decode one new token through `layer_indices()[start..end)`.
///
/// `blocks` is this token's residual-block stack. `caches` holds all visit
/// KV/sconv states (warmed by prior tokens). Message bank fields mirror
/// [`crate::visit::VisitState`] for core reentry.
pub struct DecodeTokenState {
    pub x: [f32; HIDDEN],
    pub blocks: Vec<f32>,
    pub prelude_x: Option<[f32; HIDDEN]>,
    pub prelude_blocks: Option<Vec<f32>>,
    pub bank: Option<Vec<f32>>,
}

thread_local! {
    /// Spare message-bank buffer between decode tokens (see decode_logits_metal_into).
    static BANK_SCRATCH: std::cell::RefCell<Vec<f32>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

impl DecodeTokenState {
    pub fn new(x: [f32; HIDDEN]) -> Self {
        Self {
            x,
            blocks: Vec::new(),
            prelude_x: None,
            prelude_blocks: None,
            bank: None,
        }
    }
}

pub fn run_decode_visits(
    st: &SafeTensors,
    state: &mut DecodeTokenState,
    caches: &mut DecodeCaches,
    start_visit: usize,
    end_visit: usize,
) -> Result<()> {
    let mut cache = WeightCache::new();
    run_decode_visits_cached(st, state, caches, start_visit, end_visit, &mut cache)
}

/// Same as [`run_decode_visits`] but reuses a caller-owned [`WeightCache`].
pub fn run_decode_visits_cached(
    st: &SafeTensors,
    state: &mut DecodeTokenState,
    caches: &mut DecodeCaches,
    start_visit: usize,
    end_visit: usize,
    cache: &mut WeightCache,
) -> Result<()> {
    run_decode_visits_cached_accel(st, state, caches, start_visit, end_visit, cache, None)
}

/// Decode visits with Metal MoE (attn/sconv stay on CPU decode path).
pub fn run_decode_visits_metal(
    st: &SafeTensors,
    state: &mut DecodeTokenState,
    caches: &mut DecodeCaches,
    start_visit: usize,
    end_visit: usize,
    accel: &mut MetalAccel,
) -> Result<()> {
    let mut cache = WeightCache::new();
    run_decode_visits_cached_accel(
        st,
        state,
        caches,
        start_visit,
        end_visit,
        &mut cache,
        Some(accel),
    )
}

/// Same as [`run_decode_visits_metal`] with a shared [`WeightCache`].
pub fn run_decode_visits_metal_cached(
    st: &SafeTensors,
    state: &mut DecodeTokenState,
    caches: &mut DecodeCaches,
    start_visit: usize,
    end_visit: usize,
    cache: &mut WeightCache,
    accel: &mut MetalAccel,
) -> Result<()> {
    run_decode_visits_cached_accel(
        st,
        state,
        caches,
        start_visit,
        end_visit,
        cache,
        Some(accel),
    )
}

fn run_decode_visits_cached_accel(
    st: &SafeTensors,
    state: &mut DecodeTokenState,
    caches: &mut DecodeCaches,
    start_visit: usize,
    end_visit: usize,
    cache: &mut WeightCache,
    accel: Option<&mut MetalAccel>,
) -> Result<()> {
    if let Some(accel) = accel {
        return run_decode_visits_gpu_x(
            st,
            state,
            caches,
            start_visit,
            end_visit,
            cache,
            accel,
            false,
        );
    }
    // CPU reference: one layer step at a time on the host.
    let indices = layer_indices();
    assert!(end_visit <= indices.len());
    assert_eq!(caches.visits.len(), indices.len());
    let msg_queries = cache.msg_queries(st)?;

    let profile_visits = crate::flags::profile_visits();
    let mut t_visit = std::time::Instant::now();
    for visit in start_visit..end_visit {
        let index = indices[visit];
        if index == CORE_START {
            if visit == index {
                state.prelude_x.replace(state.x);
                match state.prelude_blocks.as_mut() {
                    Some(pb) => pb.clone_from(&state.blocks),
                    None => state.prelude_blocks = Some(state.blocks.clone()),
                }
            } else {
                let pb = state.prelude_blocks.as_ref().expect("prelude");
                let prelude_n = pb.len() / HIDDEN;
                let mut bank = state.bank.take().unwrap_or_else(|| {
                    BANK_SCRATCH.with(|b| std::mem::take(&mut *b.borrow_mut()))
                });
                rebuild_message_bank_into(&state.blocks, prelude_n, &state.x, &mut bank);
                state.bank = Some(bank);
                state.x = state.prelude_x.expect("prelude_x");
                state.blocks.clone_from(pb);
            }
        }

        let messages: Option<[f32; MSG_SLOTS * HIDDEN]> = match state.bank {
            Some(ref bank) if index < CORE_END => {
                let q = &msg_queries[(index - CORE_START) * MSG_SLOTS * HIDDEN
                    ..(index - CORE_START + 1) * MSG_SLOTS * HIDDEN];
                let mut out = [0f32; MSG_SLOTS * HIDDEN];
                gather_messages(bank, bank.len() / HIDDEN, q, &mut out);
                Some(out)
            }
            _ => None,
        };

        let with_ctrl = messages.is_some();
        let visit_cache = &mut caches.visits[visit];
        let msg_ref = messages.as_ref().map(|m| m.as_slice());
        if is_full_attn(index) {
            let buf = cache.full(st, index, with_ctrl)?;
            let w = buf.as_weights();
            state.x =
                full_layer_decode_step_msgs(&state.x, &mut state.blocks, &w, visit_cache, msg_ref)?;
        } else {
            let buf = cache.sliding(st, index, with_ctrl)?;
            let w = buf.as_weights();
            state.x = sliding_layer_decode_step_msgs(
                &state.x,
                &mut state.blocks,
                &w,
                visit_cache,
                msg_ref,
            )?;
        }
        if profile_visits {
            let ms = t_visit.elapsed().as_secs_f64() * 1000.0;
            eprintln!("profile visit={visit} layer={index} {ms:.2} ms");
            t_visit = std::time::Instant::now();
        }
    }
    Ok(())
}

/// Most prompt tokens that go through the 48 visits together.  A longer prompt is
/// split into equal chunks no larger than this.
pub const PROMPT_CHUNK: usize = crate::cache::RING;

/// Metal decode for one token. Every visit is recorded onto the same open compute
/// encoder, with the stream, block stack, KV rings, conv state and message bank
/// resident on the GPU. Starts at visit 0. Unless `skip_end_sync`, waits at the end
/// and copies the stream and block stack back into `state`.
fn run_decode_visits_gpu_x(
    st: &SafeTensors,
    state: &mut DecodeTokenState,
    caches: &mut DecodeCaches,
    start_visit: usize,
    end_visit: usize,
    cache: &mut WeightCache,
    accel: &mut MetalAccel,
    skip_end_sync: bool,
) -> Result<()> {
    let indices = layer_indices();
    assert_eq!(start_visit, 0, "Metal decode records a token from visit 0");
    assert!(end_visit <= indices.len());
    assert_eq!(caches.visits.len(), indices.len());
    let msg_queries = cache.msg_queries(st)?;
    accel.ops
        .ensure_x_buf(Some(state.x.as_slice()))?;
    accel.ops
        .ensure_blocks_buf(&state.blocks)?;

    let profile_visits = crate::flags::profile_visits();
    let profile_step = crate::flags::profile_step();
    let mut t_visit = std::time::Instant::now();
    // Messages of the current core pass, gathered on the GPU. None on the first pass.
    let mut gpu_msgs: Option<crate::gpu::Buffer> = None;
    let mut reentries = 0usize;
    for visit in start_visit..end_visit {
        let index = indices[visit];
        let t_enc = profile_step.then(std::time::Instant::now);
        if index == CORE_START {
            let ops = &mut accel.ops;
            if visit == index {
                // First core visit: keep x and the block stack for the later passes.
                anyhow::ensure!(ops.gpu_stash_ok(), "block stack too deep for the prelude stash");
                ops.stash_prelude_gpu()?;
            } else {
                // Re-entry: build the bank, restore the prelude, gather this pass's messages.
                anyhow::ensure!(ops.gpu_reentry_ok(), "block stack too deep for the message bank");
                gpu_msgs = Some(ops.core_reentry_gpu(&msg_queries[..], reentries)?);
                reentries += 1;
            }
        }

        let msgs = gpu_msgs
            .clone()
            .filter(|_| index < CORE_END)
            .map(|b| (b, (index - CORE_START) * MSG_SLOTS * HIDDEN * 4));
        let with_ctrl = msgs.is_some();
        let visit_cache = &mut caches.visits[visit];
        visit_cache.gpu_kv(&accel.ops.gemv)?;

        if is_full_attn(index) {
            let buf = cache.full(st, index, with_ctrl)?;
            let w = buf.as_weights();
            let fuse_msg = match msgs {
                Some(msgs) => {
                    let (Some(ca), Some(ck), Some(cv), Some(cm)) = (
                        w.ctrl_attn.as_ref(),
                        w.ctrl_k.as_ref(),
                        w.ctrl_v.as_ref(),
                        w.ctrl_mlp.as_ref(),
                    ) else {
                        anyhow::bail!("layer {index} has no message controllers");
                    };
                    Some(crate::metal_fuse::FuseMsgCtrls {
                        msgs,
                        attn: (ca.down, ca.forget, ca.gain),
                        k: (ck.down, ck.forget, ck.gain),
                        v: (cv.down, cv.forget, cv.gain),
                        mlp: (cm.down, cm.forget, cm.gain),
                    })
                }
                None => None,
            };
            accel.ops.decode_full_visit_fused(
            w.experts.layer,
            &mut accel.moe,
            &state.blocks,
            w.attn.wq,
            w.attn.wk,
            w.attn.wv,
            w.attn.wr,
            w.attn.sink,
            w.attn.gate,
            w.attn.k_static,
            w.attn.k_wproj,
            w.attn.v_static,
            w.attn.v_wproj,
            w.attn.q_scale,
            w.attn.rel_scale,
            w.attn.rel_proj,
            w.attn.wo,
            w.attn_sconv,
            w.mlp_sconv,
            w.attn_queries,
            w.q_norm,
            w.k_norm,
            w.v_norm,
            w.mlp_query,
            w.mlp_norm,
            w.router_w,
            w.router_b,
            &w.experts,
            &mut visit_cache.k_sconv,
            &mut visit_cache.v_sconv,
            &mut visit_cache.attn_sconv,
            &mut visit_cache.mlp_sconv,
                visit_cache.gpu.as_mut().unwrap(),
                visit,
                fuse_msg.as_ref(),
                w.is_block_boundary,
            )?;
        } else {
            let buf = cache.sliding(st, index, with_ctrl)?;
            let w = buf.as_weights();
            let fuse_msg = match msgs {
                Some(msgs) => {
                    let (Some(ca), Some(ck), Some(cv), Some(cm)) = (
                        w.ctrl_attn.as_ref(),
                        w.ctrl_k.as_ref(),
                        w.ctrl_v.as_ref(),
                        w.ctrl_mlp.as_ref(),
                    ) else {
                        anyhow::bail!("layer {index} has no message controllers");
                    };
                    Some(crate::metal_fuse::FuseMsgCtrls {
                        msgs,
                        attn: (ca.down, ca.forget, ca.gain),
                        k: (ck.down, ck.forget, ck.gain),
                        v: (cv.down, cv.forget, cv.gain),
                        mlp: (cm.down, cm.forget, cm.gain),
                    })
                }
                None => None,
            };
            accel.ops.decode_sliding_visit_fused(
                w.experts.layer,
                &mut accel.moe,
                &state.blocks,
            w.attn.wq,
            w.attn.wk,
            w.attn.wv,
            w.attn.wr,
            w.attn.sink,
            w.attn.gate,
            w.attn.k_sconv,
            w.attn.v_sconv,
            w.attn.q_scale,
            w.attn.rel_scale,
            w.attn.rel_proj,
            w.attn.wo,
            w.attn_sconv,
            w.mlp_sconv,
            w.attn_query,
            w.attn_norm,
            w.mlp_query,
            w.mlp_norm,
            w.router_w,
            w.router_b,
            &w.experts,
            &mut visit_cache.k_sconv,
            &mut visit_cache.v_sconv,
            &mut visit_cache.attn_sconv,
            &mut visit_cache.mlp_sconv,
                visit_cache.gpu.as_mut().unwrap(),
                visit,
                fuse_msg.as_ref(),
                w.is_block_boundary,
            )?;
        }
        if let Some(t) = t_enc {
            crate::flags::StepAccum::with_mut(|s| {
                s.encode_ms += t.elapsed().as_secs_f64() * 1000.0;
                s.fused_visits += 1;
            });
        }

        if let Some(dir) = crate::flags::dump_dir() {
            accel.ops
                .sync_x_and_blocks(&mut state.x, &mut state.blocks)?;
            crate::flags::dump_f32(dir, "decode.bin", &state.x);
        }

        if profile_visits {
            let ms = t_visit.elapsed().as_secs_f64() * 1000.0;
            eprintln!("profile visit={visit} layer={index} {ms:.2} ms");
            t_visit = std::time::Instant::now();
        }
    }
    if !skip_end_sync {
        let t_end = profile_step.then(std::time::Instant::now);
        accel.ops
            .sync_x_and_blocks(&mut state.x, &mut state.blocks)?;
        if let Some(t) = t_end {
            crate::flags::StepAccum::with_mut(|s| {
                s.end_sync_ms += t.elapsed().as_secs_f64() * 1000.0;
            });
        }
    }
    Ok(())
}

/// Embed `id`, run full Metal decode visits (warming `caches`), return logits.
pub fn decode_logits_metal(
    st: &SafeTensors,
    id: i32,
    head: &HeadBufs,
    caches: &mut DecodeCaches,
    accel: &mut MetalAccel,
    cache: &mut WeightCache,
) -> Result<Vec<f32>> {
    let mut logits = vec![0f32; VOCAB];
    decode_logits_metal_into(st, id, head, caches, accel, cache, &mut logits)?;
    Ok(logits)
}

/// Like [`decode_logits_metal`] writing into a caller-owned logits buffer.
pub fn decode_logits_metal_into(
    st: &SafeTensors,
    id: i32,
    head: &HeadBufs,
    caches: &mut DecodeCaches,
    accel: &mut MetalAccel,
    cache: &mut WeightCache,
    logits: &mut [f32],
) -> Result<()> {
    assert_eq!(logits.len(), VOCAB);
    let profile = crate::flags::profile_any();
    let mut x = [0f32; HIDDEN];
    embed_tokens(&[id], &head.embed, &head.embed_norm, &mut x);
    // Reuse blocks/bank Vec capacity across decode steps on this thread.
    thread_local! {
        static DECODE_STATE: std::cell::RefCell<DecodeTokenState> =
            const { std::cell::RefCell::new(DecodeTokenState {
                x: [0f32; HIDDEN],
                blocks: Vec::new(),
                prelude_x: None,
                prelude_blocks: None,
                bank: None,
            }) };
    }
    let n_visits = N_VISITS;
    let (visits_ms, hidden) = DECODE_STATE.with(|cell| {
        let mut state = cell.borrow_mut();
        state.x = x;
        state.blocks.clear();
        state.prelude_x = None;
        if let Some(pb) = state.prelude_blocks.as_mut() {
            pb.clear();
        }
        // Must be None for the first core pass (controllers start on reentry).
        // Stash the allocation so CORE_START rebuild can reuse it.
        if let Some(mut bank) = state.bank.take() {
            bank.clear();
            BANK_SCRATCH.with(|b| *b.borrow_mut() = bank);
        }
        let t0 = std::time::Instant::now();
        run_decode_visits_metal_cached(st, &mut state, caches, 0, n_visits, cache, accel)?;
        let visits_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let mut hidden = [0f32; HIDDEN];
        {
            use crate::ops::{attention_residual_token_ext, rms_norm};
            let n_blocks = state.blocks.len() / HIDDEN;
            let mut mixed = [0f32; HIDDEN];
            attention_residual_token_ext(
                &state.blocks,
                Some(state.x.as_slice()),
                n_blocks,
                &head.output_query,
                true,
                &mut mixed,
            );
            rms_norm(&mixed, Some(&head.final_norm), &mut hidden);
        }
        Ok::<_, anyhow::Error>((visits_ms, hidden))
    })?;

    let t1 = std::time::Instant::now();
    unembed_metal(
        &mut accel.ops,
        &hidden,
        1,
        &head.unembed,
        logits,
    )?;
    if profile {
        eprintln!(
            "profile decode_token visits={visits_ms:.0} ms unembed={:.0} ms",
            t1.elapsed().as_secs_f64() * 1000.0
        );
    }
    Ok(())
}

/// Greedy decode step: visits, final mix, unembed and argmax on one command buffer,
/// one wait, no logits readback. Returns the next token id.
pub fn decode_next_id_metal(
    st: &SafeTensors,
    id: i32,
    head: &HeadBufs,
    caches: &mut DecodeCaches,
    accel: &mut MetalAccel,
    cache: &mut WeightCache,
) -> Result<i32> {
    let profile = crate::flags::profile_any();
    let profile_step = crate::flags::profile_step();
    let profile_gpu = crate::flags::profile_gpu();
    if profile_step {
        let _ = crate::flags::StepAccum::take_and_reset();
    }
    if profile_gpu {
        let _ = crate::flags::GpuSplitAccum::take_and_reset();
    }
    let mut x = [0f32; HIDDEN];
    embed_tokens(&[id], &head.embed, &head.embed_norm, &mut x);
    thread_local! {
        static DECODE_STATE: std::cell::RefCell<DecodeTokenState> =
            const { std::cell::RefCell::new(DecodeTokenState {
                x: [0f32; HIDDEN],
                blocks: Vec::new(),
                prelude_x: None,
                prelude_blocks: None,
                bank: None,
            }) };
    }
    assert_eq!(head.unembed.len(), VOCAB * HIDDEN);
    let t_tok = std::time::Instant::now();
    accel.ops.record_taps = caches.taps.is_some();
    let next = DECODE_STATE.with(|cell| {
        let mut state = cell.borrow_mut();
        state.x = x;
        state.blocks.clear();
        run_decode_visits_gpu_x(st, &mut state, caches, 0, N_VISITS, cache, accel, true)?;
        let t_fin = profile_step.then(std::time::Instant::now);
        let id = accel.ops.finalize_argmax_from_resident(
            &head.output_query,
            &head.final_norm,
            &head.unembed[..UNPADDED_VOCAB * HIDDEN],
            UNPADDED_VOCAB,
            LOGIT_SOFTCAP,
        )?;
        if let Some(t) = t_fin {
            crate::flags::StepAccum::with_mut(|s| {
                s.end_sync_ms += t.elapsed().as_secs_f64() * 1000.0;
            });
        }
        Ok::<_, anyhow::Error>(id)
    })?;
    if let Some(taps) = caches.taps.as_mut() {
        accel.ops.read_decode_taps(taps)?;
    }
    let step_ms = t_tok.elapsed().as_secs_f64() * 1000.0;
    if profile && !profile_step && !profile_gpu {
        eprintln!("profile decode_token {step_ms:.1} ms");
    }
    if profile_step {
        let s = crate::flags::StepAccum::take_and_reset();
        eprintln!(
            "profile step wall={step_ms:.1} encode={:.1} wait={:.1} visits={}",
            s.encode_ms, s.end_sync_ms, s.fused_visits,
        );
    }
    if profile_gpu {
        let g = crate::flags::GpuSplitAccum::take_and_reset();
        let n = g.visits.max(1) as f64;
        let tot_g = g.attn_gpu_ms() + g.moe_gpu_ms;
        let pct = |x: f64| {
            if tot_g > 0.0 {
                100.0 * x / tot_g
            } else {
                0.0
            }
        };
        eprintln!(
            "profile gpu visits={} proj={:.1}({:.0}%) sdpa={:.1}({:.0}%) premoe={:.1}({:.0}%) moe={:.1}({:.0}%) tot={tot_g:.1} | per_v proj={:.2} sdpa={:.2} premoe={:.2} moe={:.2}",
            g.visits,
            g.proj_gpu_ms,
            pct(g.proj_gpu_ms),
            g.sdpa_gpu_ms,
            pct(g.sdpa_gpu_ms),
            g.premoe_gpu_ms,
            pct(g.premoe_gpu_ms),
            g.moe_gpu_ms,
            pct(g.moe_gpu_ms),
            g.proj_gpu_ms / n,
            g.sdpa_gpu_ms / n,
            g.premoe_gpu_ms / n,
            g.moe_gpu_ms / n,
        );
    }
    Ok(next)
}

/// Step through `ids` one token at a time with shared decode caches; last-token logits.
pub fn decode_sequence_logits_metal(
    st: &SafeTensors,
    ids: &[i32],
    head: &HeadBufs,
    accel: &mut MetalAccel,
    cache: &mut WeightCache,
) -> Result<Vec<f32>> {
    assert!(!ids.is_empty());
    let indices = layer_indices();
    let mut caches = DecodeCaches::for_schedule(indices, ids.len() + 4);
    let mut logits = vec![0f32; VOCAB];
    for &id in ids {
        decode_logits_metal_into(st, id, head, &mut caches, accel, cache, &mut logits)?;
    }
    Ok(logits)
}

fn argmax_i32(logits: &[f32]) -> i32 {
    let n = logits.len().min(crate::head::UNPADDED_VOCAB);
    #[cfg(target_os = "macos")]
    {
        #[link(name = "Accelerate", kind = "framework")]
        extern "C" {
            fn vDSP_maxvi(a: *const f32, ia: i32, c: *mut f32, i: *mut u32, n: u32);
        }
        let mut best_v = 0f32;
        let mut best_i = 0u32;
        unsafe {
            vDSP_maxvi(logits.as_ptr(), 1, &mut best_v, &mut best_i, n as u32);
        }
        best_i as i32
    }
    #[cfg(not(target_os = "macos"))]
    {
        let mut best_i = 0usize;
        let mut best_v = f32::NEG_INFINITY;
        for (i, &v) in logits[..n].iter().enumerate() {
            if v > best_v {
                best_v = v;
                best_i = i;
            }
        }
        best_i as i32
    }
}

/// Subtract `penalty` from logits of tokens already present in `prev` (once each).
///
/// The generate CLI applies this over the full prompt + generated ids (HF-style).
pub fn apply_presence_penalty(logits: &mut [f32], prev: &[i32], penalty: f32) {
    if !(penalty > 0.0) || prev.is_empty() {
        return;
    }
    // Dense id list + bitset beats HashSet; reuse scratch across steps.
    thread_local! {
        static SEEN_MASK: std::cell::RefCell<Vec<bool>> =
            const { std::cell::RefCell::new(Vec::new()) };
        static SEEN_IDS: std::cell::RefCell<Vec<usize>> =
            const { std::cell::RefCell::new(Vec::new()) };
    }
    SEEN_MASK.with(|mask_cell| {
        SEEN_IDS.with(|ids_cell| {
            let mut mask = mask_cell.borrow_mut();
            let mut ids = ids_cell.borrow_mut();
            let n = logits.len().min(crate::head::UNPADDED_VOCAB);
            if mask.len() < n {
                mask.resize(n, false);
            } else {
                for &i in ids.iter() {
                    if i < mask.len() {
                        mask[i] = false;
                    }
                }
            }
            ids.clear();
            for &id in prev {
                let i = id as usize;
                if i < n && !mask[i] {
                    mask[i] = true;
                    ids.push(i);
                }
            }
            for &i in ids.iter() {
                logits[i] -= penalty;
            }
        });
    });
}

/// Tiny xorshift32 RNG for temperature sampling (no extra crate).
pub struct XorShift32(u32);

impl XorShift32 {
    pub fn new(seed: u32) -> Self {
        Self(seed | 1)
    }

    pub fn next_f32(&mut self) -> f32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.0 = x;
        (x as f32) / (u32::MAX as f32)
    }
}

/// Sample from temperature-scaled logits (`temperature <= 0` → argmax).
///
/// When `top_k > 0`, only the top-k logits participate.
pub fn sample_logits(logits: &[f32], temperature: f32, rng: &mut XorShift32) -> i32 {
    sample_logits_top_k(logits, temperature, 0, rng)
}

/// Like [`sample_logits`] with optional top-k truncation (`top_k == 0` → full vocab).
pub fn sample_logits_top_k(
    logits: &[f32],
    temperature: f32,
    top_k: usize,
    rng: &mut XorShift32,
) -> i32 {
    sample_logits_top_k_p(logits, temperature, top_k, 0.0, rng)
}

/// Temperature sampling with optional top-k and nucleus (top-p) truncation.
///
/// `top_k == 0` → no k cutoff. `top_p <= 0` or `>= 1` → no nucleus cutoff.
pub fn sample_logits_top_k_p(
    logits: &[f32],
    temperature: f32,
    top_k: usize,
    top_p: f32,
    rng: &mut XorShift32,
) -> i32 {
    if !(temperature > 0.0) {
        return argmax_i32(logits);
    }
    let inv_t = 1.0 / temperature;
    let nucleus = top_p > 0.0 && top_p < 1.0;
    let n_live = logits.len().min(crate::head::UNPADDED_VOCAB);
    // Full-vocab temperature sample: reused scratch, no 200k index Vec.
    if top_k == 0 && !nucleus {
        thread_local! {
            static SAMPLE_BUF: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) };
        }
        return SAMPLE_BUF.with(|cell| {
            let mut buf = cell.borrow_mut();
            if buf.len() < n_live {
                buf.resize(n_live, 0.0);
            }
            let buf = &mut buf[..n_live];
            #[cfg(target_os = "macos")]
            {
                #[link(name = "Accelerate", kind = "framework")]
                extern "C" {
                    fn vDSP_vsmul(
                        a: *const f32,
                        ia: i32,
                        b: *const f32,
                        c: *mut f32,
                        ic: i32,
                        n: u32,
                    );
                    fn vDSP_maxv(a: *const f32, ia: i32, c: *mut f32, n: u32);
                    fn vDSP_vsadd(
                        a: *const f32,
                        ia: i32,
                        b: *const f32,
                        c: *mut f32,
                        ic: i32,
                        n: u32,
                    );
                    fn vvexpf(y: *mut f32, x: *const f32, n: *const i32);
                    fn vDSP_sve(a: *const f32, ia: i32, c: *mut f32, n: u32);
                }
                let n_u = n_live as u32;
                let n_i = n_live as i32;
                let mut max = 0f32;
                let mut sum = 0f32;
                unsafe {
                    vDSP_vsmul(logits.as_ptr(), 1, &inv_t, buf.as_mut_ptr(), 1, n_u);
                    vDSP_maxv(buf.as_ptr(), 1, &mut max, n_u);
                    let neg_max = -max;
                    vDSP_vsadd(buf.as_ptr(), 1, &neg_max, buf.as_mut_ptr(), 1, n_u);
                    vvexpf(buf.as_mut_ptr(), buf.as_ptr(), &n_i);
                    vDSP_sve(buf.as_ptr(), 1, &mut sum, n_u);
                }
                let u = rng.next_f32() * sum;
                let mut c = 0f32;
                for (i, &p) in buf.iter().enumerate() {
                    c += p;
                    if u <= c {
                        return i as i32;
                    }
                }
                (n_live.saturating_sub(1)) as i32
            }
            #[cfg(not(target_os = "macos"))]
            {
                let mut max = f32::NEG_INFINITY;
                for (b, &v) in buf.iter_mut().zip(&logits[..n_live]) {
                    *b = v * inv_t;
                    max = max.max(*b);
                }
                let mut sum = 0f32;
                for b in &mut *buf {
                    *b = (*b - max).exp();
                    sum += *b;
                }
                let u = rng.next_f32() * sum;
                let mut c = 0f32;
                for (i, &p) in buf.iter().enumerate() {
                    c += p;
                    if u <= c {
                        return i as i32;
                    }
                }
                (n_live.saturating_sub(1)) as i32
            }
        });
    }
    thread_local! {
        static IDXS: std::cell::RefCell<Vec<usize>> =
            const { std::cell::RefCell::new(Vec::new()) };
        static WEIGHTS: std::cell::RefCell<Vec<f32>> =
            const { std::cell::RefCell::new(Vec::new()) };
    }
    IDXS.with(|idxs_cell| {
        WEIGHTS.with(|weights_cell| {
            let mut idxs = idxs_cell.borrow_mut();
            let mut weights = weights_cell.borrow_mut();
            idxs.clear();
            if idxs.capacity() < n_live {
                idxs.reserve(n_live);
            }
            idxs.extend(0..n_live);
            if top_k > 0 && top_k < idxs.len() {
                idxs.select_nth_unstable_by(top_k - 1, |&a, &b| {
                    logits[b]
                        .partial_cmp(&logits[a])
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                idxs.truncate(top_k);
            }
            if nucleus {
                // Need descending order for cumulative mass; if no top-k, still sort all.
                idxs.sort_unstable_by(|&a, &b| {
                    (logits[b] * inv_t)
                        .partial_cmp(&(logits[a] * inv_t))
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            }
            let mut max = f32::NEG_INFINITY;
            for &i in idxs.iter() {
                max = max.max(logits[i] * inv_t);
            }
            weights.clear();
            weights.reserve(idxs.len());
            let mut sum = 0f32;
            for &i in idxs.iter() {
                let e = ((logits[i] * inv_t) - max).exp();
                sum += e;
                weights.push(e);
            }
            if nucleus && sum > 0.0 {
                let mut cum = 0f32;
                let mut cut = idxs.len();
                for (j, &w) in weights.iter().enumerate() {
                    cum += w / sum;
                    if cum >= top_p {
                        cut = j + 1;
                        break;
                    }
                }
                idxs.truncate(cut.max(1));
                weights.truncate(idxs.len());
                sum = weights.iter().sum();
            }
            let u = rng.next_f32() * sum;
            let mut c = 0f32;
            for (j, &i) in idxs.iter().enumerate() {
                c += weights[j];
                if u <= c {
                    return i as i32;
                }
            }
            idxs.last().copied().unwrap_or(0) as i32
        })
    })
}

/// Temperature sampling generate (see [`generate_argmax_metal_stop`] for stop ids).
pub fn generate_sample_metal(
    st: &SafeTensors,
    prompt: &[i32],
    n_new: usize,
    temperature: f32,
    seed: u32,
    head: &HeadBufs,
    accel: &mut MetalAccel,
    cache: &mut WeightCache,
    stop_ids: &[i32],
) -> Result<Vec<i32>> {
    let stops = if stop_ids.is_empty() {
        // <|end_message|> closes content channels mid-turn; do not treat as EOS.
        [EOS_TOKEN_ID, PAD_TOKEN_ID].as_slice()
    } else {
        stop_ids
    };
    generate_sample_metal_top_k(
        st, prompt, n_new, temperature, 0, seed, head, accel, cache, stops,
    )
}

/// Like [`generate_sample_metal`] with top-k truncation (`top_k == 0` → full vocab).
pub fn generate_sample_metal_top_k(
    st: &SafeTensors,
    prompt: &[i32],
    n_new: usize,
    temperature: f32,
    top_k: usize,
    seed: u32,
    head: &HeadBufs,
    accel: &mut MetalAccel,
    cache: &mut WeightCache,
    stop_ids: &[i32],
) -> Result<Vec<i32>> {
    let WarmPrompt {
        mut caches,
        mut logits,
    } = warm_prompt_metal(st, prompt, head, accel, cache, prompt.len() + n_new + 4)?;

    let mut rng = XorShift32::new(seed);
    let mut out = prompt.to_vec();
    for i in 0..n_new {
        let next = sample_logits_top_k(&logits, temperature, top_k, &mut rng);
        out.push(next);
        if stop_ids.contains(&next) {
            break;
        }
        if i + 1 < n_new {
            decode_logits_metal_into(st, next, head, &mut caches, accel, cache, &mut logits)?;
        }
    }
    Ok(out)
}

/// Prompt warm: batch Metal prefill into decode caches + first-token logits.
pub struct WarmPrompt {
    pub caches: DecodeCaches,
    pub logits: Vec<f32>,
}

thread_local! {
    /// Recycled ~60 MB KV/sconv buffers between warms when the REPL drops a session.
    static WARM_CACHES: std::cell::RefCell<Option<DecodeCaches>> =
        const { std::cell::RefCell::new(None) };
    static WARM_LOGITS: std::cell::RefCell<Option<Vec<f32>>> =
        const { std::cell::RefCell::new(None) };
}

/// Return decode caches (and optional logits) to the warm pool when a REPL session is discarded.
pub fn recycle_decode_caches(caches: DecodeCaches) {
    recycle_decode_caches_with_logits(caches, None);
}

/// Like [`recycle_decode_caches`] also returning the logits buffer.
pub fn recycle_decode_caches_with_logits(caches: DecodeCaches, logits: Option<Vec<f32>>) {
    WARM_CACHES.with(|cell| {
        *cell.borrow_mut() = Some(caches);
    });
    if let Some(logits) = logits {
        WARM_LOGITS.with(|cell| {
            *cell.borrow_mut() = Some(logits);
        });
    }
}

/// True when a recycled DecodeCaches sits in the warm pool (next rewarm skips ~60 MB alloc).
pub fn warm_caches_pooled() -> bool {
    WARM_CACHES.with(|cell| cell.borrow().is_some())
}

/// Batch-prefill `prompt` into KV/sconv caches (Metal) and return last-token logits.
pub fn warm_prompt_metal(
    st: &SafeTensors,
    prompt: &[i32],
    head: &HeadBufs,
    accel: &mut MetalAccel,
    cache: &mut WeightCache,
    full_cap: usize,
) -> Result<WarmPrompt> {
    warm_prompt_taps_metal(st, prompt, head, accel, cache, full_cap, false)
}

/// [`warm_prompt_metal`] that, with `taps`, also records in `caches.taps` the target
/// states the drafter reads, for the prompt and for everything processed after it.
pub fn warm_prompt_taps_metal(
    st: &SafeTensors,
    prompt: &[i32],
    head: &HeadBufs,
    accel: &mut MetalAccel,
    cache: &mut WeightCache,
    full_cap: usize,
    taps: bool,
) -> Result<WarmPrompt> {
    assert!(!prompt.is_empty());
    // A new prompt: drop what decode kept on the GPU for the previous one.
    accel.ops.clear_decode_state();
    let indices = layer_indices();
    let mut caches = WARM_CACHES.with(|cell| {
        match cell.borrow_mut().take() {
            Some(mut c) if c.visits.len() == indices.len() => {
                c.reset_for_warm(full_cap);
                c
            }
            _ => DecodeCaches::for_schedule(indices, full_cap),
        }
    });
    caches.taps = taps.then(Vec::new);

    let profile = crate::flags::profile_any();
    let t_pre = std::time::Instant::now();
    // Host layer decode + GPU unembed upload (f16). Unembed is ~VOCAB×H and
    // otherwise lands on the first logits call; doing it here keeps finalize thin.
    cache.preload_all_layers(st)?;
    let w_live = &head.unembed[..crate::head::UNPADDED_VOCAB * HIDDEN];
    accel.ops.preload_weight_f16(w_live)?;
    // Attn proj GEMVs on decode hit Metal; upload once so the first token
    // does not pay 24× weight copies mid-decode.
    cache.preload_attn_gemv(accel)?;
    if profile {
        eprintln!(
            "profile preload={:.0} ms ({} layers + unembed + attn gemv)",
            t_pre.elapsed().as_secs_f64() * 1000.0,
            cache.len()
        );
    }
    let mut logits = WARM_LOGITS.with(|cell| {
        cell.borrow_mut()
            .take()
            .filter(|v| v.len() == VOCAB)
            .unwrap_or_else(|| vec![0f32; VOCAB])
    });
    prompt_chunks(st, prompt, head, &mut caches, accel, cache, &mut logits)?;
    Ok(WarmPrompt { caches, logits })
}

/// Fewest tokens [`extend_prompt_metal`] sends through the prompt path.  A prompt
/// visit has more fixed cost than a decode step, so a handful of tokens is quicker
/// one at a time.
pub const EXTEND_MIN: usize = 8;

/// Process more prompt tokens on top of what `caches` already hold: a later chat
/// turn, a tool result.  `logits` gets the last token's logits.  The cost follows the
/// new tokens, not the conversation so far.
pub fn extend_prompt_metal(
    st: &SafeTensors,
    tokens: &[i32],
    head: &HeadBufs,
    caches: &mut DecodeCaches,
    accel: &mut MetalAccel,
    cache: &mut WeightCache,
    logits: &mut [f32],
) -> Result<()> {
    assert!(!tokens.is_empty());
    // Decode into full logits does not record the drafter's states.
    if tokens.len() < EXTEND_MIN && caches.taps.is_none() {
        for &id in tokens {
            decode_logits_metal_into(st, id, head, caches, accel, cache, logits)?;
        }
        return Ok(());
    }
    // Decode keeps the conv states on the GPU.  The prompt path wants them on the host.
    accel.ops.conv_states_to_host(caches)?;
    prompt_chunks(st, tokens, head, caches, accel, cache, logits)
}

/// The final normed state of every token of a chunk, `[tokens][H]`, which is also
/// the fourth state the drafter reads.
fn finish_rows(rows: &mut crate::visit::ChunkRows, head: &HeadBufs) -> Vec<f32> {
    use crate::drafter::FEAT;
    let n = rows.tokens;
    let mut hidden = vec![0f32; n * HIDDEN];
    finalize_hidden(&rows.x, &rows.stacks, n, &head.output_query, &head.final_norm, &mut hidden);
    for (t, h) in hidden.chunks_exact(HIDDEN).enumerate() {
        rows.taps[t * FEAT + 3 * HIDDEN..(t + 1) * FEAT].copy_from_slice(h);
    }
    hidden
}

/// What the target makes of a block it has not committed to.
pub struct BlockPass {
    /// The target's pick after each token of the block.
    pub picks: Vec<i32>,
    /// The states the drafter reads, `[tokens][FEAT]`.
    pub feats: Vec<f32>,
}

/// Run `tokens` on top of what `caches` hold without committing them, and return
/// the target's pick after every position.  [`commit_block_metal`] then keeps as
/// many leading tokens as turned out right.  Until it is called, `caches` are as
/// they were.
pub fn verify_block_metal(
    st: &SafeTensors,
    tokens: &[i32],
    head: &HeadBufs,
    caches: &mut DecodeCaches,
    accel: &mut MetalAccel,
    cache: &mut WeightCache,
) -> Result<BlockPass> {
    assert!(!tokens.is_empty());
    let n = tokens.len();
    // Decode keeps the conv states on the GPU.  The prompt path wants them on the host.
    accel.ops.conv_states_to_host(caches)?;
    let mut x = vec![0f32; n * HIDDEN];
    embed_tokens(tokens, &head.embed, &head.embed_norm, &mut x);
    if n <= crate::drafter::BLOCK && !crate::flags::block_by_chunk() {
        // A short block: every visit, the final state and the picks in one command
        // buffer.
        let out = crate::visit::run_prompt_block_gpu(st, &x, n, head, cache, caches, accel)?;
        accel.ops.release_prompt_scratch();
        accel.moe.release_prompt_scratch();
        return Ok(BlockPass { picks: out.picks, feats: out.feats });
    }
    caches.hold_block(true);
    let rows = crate::visit::run_prompt_chunk_rows(st, &x, n, cache, caches, accel);
    caches.hold_block(false);
    let mut rows = rows?;
    let hidden = finish_rows(&mut rows, head);
    let mut logits = vec![0f32; n * VOCAB];
    unembed_metal(&mut accel.ops, &hidden, n, &head.unembed, &mut logits)?;
    let mut picks = Vec::with_capacity(n);
    for row in logits.chunks_exact(VOCAB) {
        let (mut best, mut top) = (0usize, f32::MIN);
        for (i, &v) in row[..crate::head::UNPADDED_VOCAB].iter().enumerate() {
            if v > top {
                top = v;
                best = i;
            }
        }
        picks.push(best as i32);
    }
    accel.ops.release_prompt_scratch();
    accel.moe.release_prompt_scratch();
    Ok(BlockPass { picks, feats: rows.taps })
}

/// Keep the first `keep` tokens of the block [`verify_block_metal`] ran.
pub fn commit_block_metal(caches: &mut DecodeCaches, accel: &mut MetalAccel, keep: usize) -> Result<()> {
    caches.commit_block(&accel.ops.gemv, keep)
}

/// Tokens through the 48 visits in chunks, on top of whatever `caches` hold, then
/// the last token's logits.
fn prompt_chunks(
    st: &SafeTensors,
    tokens: &[i32],
    head: &HeadBufs,
    caches: &mut DecodeCaches,
    accel: &mut MetalAccel,
    cache: &mut WeightCache,
    logits: &mut [f32],
) -> Result<()> {
    let profile = crate::flags::profile_any();
    let t1 = std::time::Instant::now();
    // Equal chunks, so a prompt just over the limit does not leave a tiny tail.
    let cap = crate::flags::prompt_chunk().min(PROMPT_CHUNK);
    let per = tokens.len().div_ceil(tokens.len().div_ceil(cap));
    let mut x = Vec::new();
    let mut last = (Vec::new(), Vec::new());
    let mut last_hidden = None;
    for chunk in tokens.chunks(per) {
        x.resize(chunk.len() * HIDDEN, 0.0);
        embed_tokens(chunk, &head.embed, &head.embed_norm, &mut x);
        if caches.taps.is_some() {
            // The drafter wants the states of every token, not only the last.
            let mut rows =
                crate::visit::run_prompt_chunk_rows(st, &x, chunk.len(), cache, caches, accel)?;
            let hidden = finish_rows(&mut rows, head);
            if let Some(taps) = caches.taps.as_mut() {
                taps.extend_from_slice(&rows.taps);
            }
            last_hidden = Some(hidden[(chunk.len() - 1) * HIDDEN..].to_vec());
        } else {
            last = crate::visit::run_prompt_chunk_gpu(st, &x, chunk.len(), cache, caches, accel)?;
        }
    }
    if crate::flags::profile_prefill() {
        let p = crate::flags::prefill_take();
        let parts: Vec<String> = crate::flags::PREFILL_PHASES
            .iter()
            .zip(p)
            .filter(|(_, ms)| *ms >= 0.5)
            .map(|(n, ms)| format!("{n}={ms:.0}"))
            .collect();
        eprintln!("profile prefill (ms) {}", parts.join(" "));
    }
    if profile {
        eprintln!(
            "profile visits={:.0} ms ({} visits)",
            t1.elapsed().as_secs_f64() * 1000.0,
            N_VISITS
        );
    }
    // Only the last token's logits are used for the next sample.
    let mut hidden = [0f32; HIDDEN];
    match last_hidden {
        Some(h) => hidden.copy_from_slice(&h),
        None => finalize_hidden(
            &last.0,
            std::slice::from_ref(&last.1),
            1,
            &head.output_query,
            &head.final_norm,
            &mut hidden,
        ),
    }
    let t2 = std::time::Instant::now();
    unembed_metal(&mut accel.ops, &hidden, 1, &head.unembed, logits)?;
    if profile {
        eprintln!(
            "profile finalize+unembed={:.0} ms",
            t2.elapsed().as_secs_f64() * 1000.0
        );
    }
    // Decode needs none of the prompt-sized GPU scratch.
    accel.ops.release_prompt_scratch();
    accel.moe.release_prompt_scratch();
    Ok(())
}

/// Greedy generate: warm caches on `prompt`, then append up to `n_new` argmax tokens.
///
/// Prompt is batch-prefilled into decode caches (Metal MoE/attn fill); new tokens
/// use Metal decode. Stops early if a generated id is in `stop_ids`.
/// Returns `prompt` + new ids (including the stop id if hit).
pub fn generate_argmax_metal(
    st: &SafeTensors,
    prompt: &[i32],
    n_new: usize,
    head: &HeadBufs,
    accel: &mut MetalAccel,
    cache: &mut WeightCache,
) -> Result<Vec<i32>> {
    generate_argmax_metal_stop(
        st,
        prompt,
        n_new,
        head,
        accel,
        cache,
        &[EOS_TOKEN_ID, PAD_TOKEN_ID],
    )
}

/// Same as [`generate_argmax_metal`] with optional stop ids.
pub fn generate_argmax_metal_stop(
    st: &SafeTensors,
    prompt: &[i32],
    n_new: usize,
    head: &HeadBufs,
    accel: &mut MetalAccel,
    cache: &mut WeightCache,
    stop_ids: &[i32],
) -> Result<Vec<i32>> {
    let WarmPrompt {
        mut caches,
        logits,
    } = warm_prompt_metal(st, prompt, head, accel, cache, prompt.len() + n_new + 4)?;

    let mut out = prompt.to_vec();
    let mut pending: Option<i32> = None;
    for i in 0..n_new {
        let next = pending.take().unwrap_or_else(|| argmax_i32(&logits));
        out.push(next);
        if stop_ids.contains(&next) {
            break;
        }
        if i + 1 < n_new {
            // GPU argmax unembed: skip full logits readback on the greedy path.
            pending = Some(decode_next_id_metal(
                st, next, head, &mut caches, accel, cache,
            )?);
        }
    }
    Ok(out)
}
