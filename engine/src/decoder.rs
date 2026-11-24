//! Generic sliding-window and full-attention decoder layers.

use crate::attn::{
    sliding_attention, sliding_attention_decode_step, sliding_attention_qkv,
    sliding_attention_qkv_prefill_cache, SlidingAttnWeights,
};
use crate::cache::VisitCache;
use crate::layer5::{full_layer_prefill, FullLayerWeights};
use crate::loop_msg::{message_controller, MessageControllerWeights, MSG_SLOTS};
use anyhow::Result;

use crate::moe::{sparse_moe_dispatch_into, ExpertPack, TOP_K};
use crate::ops::{
    attention_residual_token_ext, attention_residual_token_extras, axpy, gemv, rms_norm, short_conv,
    short_conv_decode_inplace, short_conv_with_state, CONV_K, HIDDEN,
};

pub struct SlidingLayerWeights<'a> {
    pub attn_norm: &'a [f32],
    pub mlp_norm: &'a [f32],
    pub attn_sconv: &'a [f32],
    pub mlp_sconv: &'a [f32],
    /// `None` for layer 0.
    pub attn_query: Option<&'a [f32]>,
    pub mlp_query: &'a [f32],
    pub router_w: &'a [f32],
    pub router_b: &'a [f32],
    pub attn: SlidingAttnWeights<'a>,
    pub experts: ExpertPack<'a>,
    pub is_block_boundary: bool,
    /// Core-loop controllers (attn/k/v/mlp); required when messages are passed.
    pub ctrl_attn: Option<MessageControllerWeights<'a>>,
    pub ctrl_k: Option<MessageControllerWeights<'a>>,
    pub ctrl_v: Option<MessageControllerWeights<'a>>,
    pub ctrl_mlp: Option<MessageControllerWeights<'a>>,
}

use crate::route::route_from_logits;

const N_EXPERTS: usize = 128;

fn route(
    x: &[f32],
    tokens: usize,
    w: &[f32],
    b: &[f32],
    indices: &mut [i32],
    weights: &mut [f32],
    logits: &mut [f32],
) {
    if tokens == 1 {
        let mut stack = [0f32; N_EXPERTS];
        gemv(w, N_EXPERTS, x, &mut stack);
        route_from_logits(&stack, 1, b, indices, weights);
        return;
    }
    debug_assert_eq!(logits.len(), tokens * N_EXPERTS);
    for t in 0..tokens {
        gemv(
            w,
            N_EXPERTS,
            &x[t * HIDDEN..(t + 1) * HIDDEN],
            &mut logits[t * N_EXPERTS..(t + 1) * N_EXPERTS],
        );
    }
    route_from_logits(logits, tokens, b, indices, weights);
}

/// Returns `(x_out, next_block_states)` where each next_block_states[t] is the
/// stack of completed residual-block vectors for that token.
///
/// `messages`, when present, is `[T * 4 * H]` (attn/k/v/mlp slots).
pub fn sliding_layer_prefill(
    x_in: &[f32],
    tokens: usize,
    block_states: &[Vec<f32>],
    w: &SlidingLayerWeights<'_>,
) -> (Vec<f32>, Vec<Vec<f32>>) {
    sliding_layer_prefill_msgs(x_in, tokens, block_states, w, None)
}

pub fn sliding_layer_prefill_msgs(
    x_in: &[f32],
    tokens: usize,
    block_states: &[Vec<f32>],
    w: &SlidingLayerWeights<'_>,
    messages: Option<&[f32]>,
) -> (Vec<f32>, Vec<Vec<f32>>) {
    let mut blocks = block_states.to_vec();
    let mut x = Vec::with_capacity(tokens * HIDDEN);
    sliding_layer_prefill_into(x_in, tokens, &mut blocks, w, messages, None, &mut x)
        .expect("host experts");
    (x, blocks)
}

/// Prefill and leave `cache` ready for the next single-token decode step.
pub fn sliding_layer_prefill_into_cache(
    x_in: &[f32],
    tokens: usize,
    block_states: &[Vec<f32>],
    w: &SlidingLayerWeights<'_>,
    cache: &mut VisitCache,
) -> Result<(Vec<f32>, Vec<Vec<f32>>)> {
    sliding_layer_prefill_into_cache_msgs(x_in, tokens, block_states, w, None, cache)
}

pub fn sliding_layer_prefill_into_cache_msgs(
    x_in: &[f32],
    tokens: usize,
    block_states: &[Vec<f32>],
    w: &SlidingLayerWeights<'_>,
    messages: Option<&[f32]>,
    cache: &mut VisitCache,
) -> Result<(Vec<f32>, Vec<Vec<f32>>)> {
    let mut blocks = block_states.to_vec();
    let mut x = Vec::with_capacity(tokens * HIDDEN);
    sliding_layer_prefill_into(x_in, tokens, &mut blocks, w, messages, Some(cache), &mut x)?;
    Ok((x, blocks))
}

/// The layer over `tokens` rows on the host.  This is the reference the GPU prompt
/// path is checked against.  It updates `block_states` in place and writes the
/// stream into `out`.  With `cache` it also leaves the visit ready for host decode.
pub fn sliding_layer_prefill_into(
    x_in: &[f32],
    tokens: usize,
    block_states: &mut [Vec<f32>],
    w: &SlidingLayerWeights<'_>,
    messages: Option<&[f32]>,
    cache: Option<&mut VisitCache>,
    out: &mut Vec<f32>,
) -> Result<()> {
    assert_eq!(x_in.len(), tokens * HIDDEN);
    assert_eq!(block_states.len(), tokens);
    if let Some(m) = messages {
        assert_eq!(m.len(), tokens * MSG_SLOTS * HIDDEN);
        assert!(w.ctrl_attn.is_some() && w.ctrl_k.is_some() && w.ctrl_v.is_some());
        assert!(w.ctrl_mlp.is_some());
    }
    thread_local! {
        static PREFILL: std::cell::RefCell<PrefillScratch> =
            const { std::cell::RefCell::new(PrefillScratch::new()) };
    }
    PREFILL.with(|cell| {
        let mut scratch = cell.borrow_mut();
        scratch.resize(tokens);
        sliding_layer_prefill_msgs_inner_bufs(
            x_in,
            tokens,
            block_states,
            w,
            messages,
            cache,
            &mut scratch,
            out,
        )
    })
}

pub(crate) struct PrefillScratch {
    pub q_in: Vec<f32>,
    pub k_in: Vec<f32>,
    pub v_in: Vec<f32>,
    pub attn: Vec<f32>,
    pub residual: Vec<f32>,
    pub moe_in: Vec<f32>,
    pub mlp_c: Vec<f32>,
    pub route_logits: Vec<f32>,
    pub indices: Vec<i32>,
    pub route_weights: Vec<f32>,
    pub moe_out: Vec<f32>,
}

impl PrefillScratch {
    pub(crate) const fn new() -> Self {
        Self {
            q_in: Vec::new(),
            k_in: Vec::new(),
            v_in: Vec::new(),
            attn: Vec::new(),
            residual: Vec::new(),
            moe_in: Vec::new(),
            mlp_c: Vec::new(),
            route_logits: Vec::new(),
            indices: Vec::new(),
            route_weights: Vec::new(),
            moe_out: Vec::new(),
        }
    }

    pub(crate) fn resize(&mut self, tokens: usize) {
        let n = tokens * HIDDEN;
        for v in [
            &mut self.q_in,
            &mut self.k_in,
            &mut self.v_in,
            &mut self.attn,
            &mut self.residual,
            &mut self.moe_in,
            &mut self.mlp_c,
            &mut self.moe_out,
        ] {
            v.resize(n, 0.0);
        }
        self.route_logits.resize(tokens * N_EXPERTS, 0.0);
        self.indices.resize(tokens * TOP_K, 0);
        self.route_weights.resize(tokens * TOP_K, 0.0);
    }
}

/// Threads for per-token host work in prefill (one for short prompts).
pub(crate) fn host_threads(tokens: usize) -> usize {
    if tokens < 16 {
        return 1;
    }
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .min(8)
}

fn sliding_layer_prefill_msgs_inner_bufs(
    x_in: &[f32],
    tokens: usize,
    block_states: &mut [Vec<f32>],
    w: &SlidingLayerWeights<'_>,
    messages: Option<&[f32]>,
    mut visit_cache: Option<&mut VisitCache>,
    scratch: &mut PrefillScratch,
    out: &mut Vec<f32>,
) -> Result<()> {
    let PrefillScratch {
        q_in,
        k_in,
        v_in,
        attn,
        residual,
        moe_in,
        mlp_c,
        route_logits,
        indices,
        route_weights,
        moe_out,
    } = &mut *scratch;
    // Per-token host work is independent, so it is spread over a few threads.
    let per = tokens.div_ceil(host_threads(tokens));
    std::thread::scope(|scope| {
        let block_states = &*block_states;
        for (((ci, q_c), k_c), v_c) in q_in[..tokens * HIDDEN]
            .chunks_mut(per * HIDDEN)
            .enumerate()
            .zip(k_in[..tokens * HIDDEN].chunks_mut(per * HIDDEN))
            .zip(v_in[..tokens * HIDDEN].chunks_mut(per * HIDDEN))
        {
            scope.spawn(move || {
                for j in 0..q_c.len() / HIDDEN {
                    let t = ci * per + j;
                    let row = j * HIDDEN..(j + 1) * HIDDEN;
                    let mut mixed = [0f32; HIDDEN];
                    if let Some(q) = w.attn_query {
                        let n_blocks = block_states[t].len() / HIDDEN;
                        attention_residual_token_ext(
                            &block_states[t],
                            Some(&x_in[t * HIDDEN..(t + 1) * HIDDEN]),
                            n_blocks,
                            q,
                            true,
                            &mut mixed,
                        );
                    } else {
                        mixed.copy_from_slice(&x_in[t * HIDDEN..(t + 1) * HIDDEN]);
                    }
                    if let Some(msgs) = messages {
                        let base = t * MSG_SLOTS * HIDDEN;
                        let mut blended = [0f32; HIDDEN];
                        for (slot, ctrl, dst) in [
                            (0, &w.ctrl_attn, &mut q_c[row.clone()]),
                            (1, &w.ctrl_k, &mut k_c[row.clone()]),
                            (2, &w.ctrl_v, &mut v_c[row.clone()]),
                        ] {
                            message_controller(
                                &mixed,
                                &msgs[base + slot * HIDDEN..base + (slot + 1) * HIDDEN],
                                ctrl.as_ref().unwrap(),
                                &mut blended,
                            );
                            rms_norm(&blended, Some(w.attn_norm), dst);
                        }
                    } else {
                        rms_norm(&mixed, Some(w.attn_norm), &mut q_c[row.clone()]);
                        k_c[row.clone()].copy_from_slice(&q_c[row.clone()]);
                        v_c[row.clone()].copy_from_slice(&q_c[row.clone()]);
                    }
                }
            });
        }
    });

    attn.fill(0.0);
    match visit_cache {
        Some(ref mut cache) => {
            sliding_attention_qkv_prefill_cache(q_in, k_in, v_in, tokens, &w.attn, attn, cache)
        }
        None if messages.is_some() => sliding_attention_qkv(q_in, k_in, v_in, tokens, &w.attn, attn),
        None => sliding_attention(q_in, tokens, &w.attn, attn),
    }

    residual.fill(0.0);
    match visit_cache {
        Some(ref mut cache) => {
            cache.attn_sconv.resize(HIDDEN * (CONV_K - 1), 0.0);
            short_conv_with_state(
                attn,
                w.attn_sconv,
                tokens,
                HIDDEN,
                None,
                residual,
                Some(cache.attn_sconv.as_mut_slice()),
            );
        }
        None => short_conv(attn, w.attn_sconv, tokens, HIDDEN, residual),
    }
    if !w.is_block_boundary {
        axpy(1.0, x_in, residual);
    }

    moe_in.fill(0.0);
    std::thread::scope(|scope| {
        let block_states = &*block_states;
        let residual = &*residual;
        for (ci, m_c) in moe_in[..tokens * HIDDEN].chunks_mut(per * HIDDEN).enumerate() {
            scope.spawn(move || {
                let mut mixed = [0f32; HIDDEN];
                for j in 0..m_c.len() / HIDDEN {
                    let t = ci * per + j;
                    let dst = &mut m_c[j * HIDDEN..(j + 1) * HIDDEN];
                    let n_blocks = block_states[t].len() / HIDDEN;
                    if w.is_block_boundary {
                        attention_residual_token_extras(
                            &block_states[t],
                            &[
                                &x_in[t * HIDDEN..(t + 1) * HIDDEN],
                                &residual[t * HIDDEN..(t + 1) * HIDDEN],
                            ],
                            n_blocks,
                            w.mlp_query,
                            true,
                            &mut mixed,
                        );
                    } else {
                        attention_residual_token_ext(
                            &block_states[t],
                            Some(&residual[t * HIDDEN..(t + 1) * HIDDEN]),
                            n_blocks,
                            w.mlp_query,
                            true,
                            &mut mixed,
                        );
                    }
                    if let Some(msgs) = messages {
                        let base = t * MSG_SLOTS * HIDDEN + 3 * HIDDEN;
                        let mut blended = [0f32; HIDDEN];
                        message_controller(
                            &mixed,
                            &msgs[base..base + HIDDEN],
                            w.ctrl_mlp.as_ref().unwrap(),
                            &mut blended,
                        );
                        rms_norm(&blended, Some(w.mlp_norm), dst);
                    } else {
                        rms_norm(&mixed, Some(w.mlp_norm), dst);
                    }
                }
            });
        }
    });

    route(
        moe_in,
        tokens,
        w.router_w,
        w.router_b,
        indices,
        route_weights,
        route_logits,
    );
    sparse_moe_dispatch_into(moe_in, indices, route_weights, tokens, &w.experts, None, moe_out)?;
    mlp_c.fill(0.0);
    match visit_cache {
        Some(ref mut cache) => {
            cache.mlp_sconv.resize(HIDDEN * (CONV_K - 1), 0.0);
            short_conv_with_state(
                moe_out,
                w.mlp_sconv,
                tokens,
                HIDDEN,
                None,
                mlp_c,
                Some(cache.mlp_sconv.as_mut_slice()),
            );
        }
        None => short_conv(moe_out, w.mlp_sconv, tokens, HIDDEN, mlp_c),
    }
    out.resize(tokens * HIDDEN, 0.0);
    out.copy_from_slice(residual);
    axpy(1.0, mlp_c, out);

    if w.is_block_boundary {
        for t in 0..tokens {
            block_states[t].extend_from_slice(&x_in[t * HIDDEN..(t + 1) * HIDDEN]);
        }
    }
    Ok(())
}

/// Single-token sliding layer decode on the host. Updates `cache` and `blocks`.
///
/// This is the CPU reference the parity tests compare against. Metal decode does
/// not go through it (see `decode_visit::run_decode_visits_gpu_x`).
pub fn sliding_layer_decode_step(
    x_in: &[f32],
    blocks: &mut Vec<f32>,
    w: &SlidingLayerWeights<'_>,
    cache: &mut VisitCache,
) -> Result<[f32; HIDDEN]> {
    sliding_layer_decode_step_msgs(x_in, blocks, w, cache, None)
}

/// As [`sliding_layer_decode_step`]; `messages`, when present, is `[MSG_SLOTS * H]`.
pub fn sliding_layer_decode_step_msgs(
    x_in: &[f32],
    blocks: &mut Vec<f32>,
    w: &SlidingLayerWeights<'_>,
    cache: &mut VisitCache,
    messages: Option<&[f32]>,
) -> Result<[f32; HIDDEN]> {
    assert_eq!(x_in.len(), HIDDEN);
    if let Some(m) = messages {
        assert_eq!(m.len(), MSG_SLOTS * HIDDEN);
        assert!(w.ctrl_attn.is_some() && w.ctrl_k.is_some() && w.ctrl_v.is_some());
        assert!(w.ctrl_mlp.is_some());
    }

    let mut mixed = [0f32; HIDDEN];
    if let Some(q) = w.attn_query {
        let n_blocks = blocks.len() / HIDDEN;
        attention_residual_token_ext(blocks, Some(x_in), n_blocks, q, true, &mut mixed);
    } else {
        mixed.copy_from_slice(x_in);
    }
    let mut q_in = [0f32; HIDDEN];
    let mut k_in = [0f32; HIDDEN];
    let mut v_in = [0f32; HIDDEN];
    if let Some(msgs) = messages {
        let mut blended = [0f32; HIDDEN];
        message_controller(
            &mixed,
            &msgs[0..HIDDEN],
            w.ctrl_attn.as_ref().unwrap(),
            &mut blended,
        );
        rms_norm(&blended, Some(w.attn_norm), &mut q_in);
        message_controller(
            &mixed,
            &msgs[HIDDEN..2 * HIDDEN],
            w.ctrl_k.as_ref().unwrap(),
            &mut blended,
        );
        rms_norm(&blended, Some(w.attn_norm), &mut k_in);
        message_controller(
            &mixed,
            &msgs[2 * HIDDEN..3 * HIDDEN],
            w.ctrl_v.as_ref().unwrap(),
            &mut blended,
        );
        rms_norm(&blended, Some(w.attn_norm), &mut v_in);
    } else {
        rms_norm(&mixed, Some(w.attn_norm), &mut q_in);
        k_in = q_in;
        v_in = q_in;
    }

    let mut attn = [0f32; HIDDEN];
    sliding_attention_decode_step(&q_in, &k_in, &v_in, &w.attn, cache, &mut attn)?;

    let mut residual = [0f32; HIDDEN];
    short_conv_decode_inplace(
        &attn,
        w.attn_sconv,
        HIDDEN,
        &mut cache.attn_sconv,
        &mut residual,
    );
    if !w.is_block_boundary {
        axpy(1.0, x_in, &mut residual);
    }

    let mut moe_in = [0f32; HIDDEN];
    {
        let n_blocks = blocks.len() / HIDDEN;
        let mut mixed = [0f32; HIDDEN];
        if w.is_block_boundary {
            attention_residual_token_extras(
                blocks,
                &[x_in, residual.as_slice()],
                n_blocks,
                w.mlp_query,
                true,
                &mut mixed,
            );
        } else {
            attention_residual_token_ext(
                blocks,
                Some(residual.as_slice()),
                n_blocks,
                w.mlp_query,
                true,
                &mut mixed,
            );
        }
        if let Some(msgs) = messages {
            let mut blended = [0f32; HIDDEN];
            message_controller(
                &mixed,
                &msgs[3 * HIDDEN..4 * HIDDEN],
                w.ctrl_mlp.as_ref().unwrap(),
                &mut blended,
            );
            rms_norm(&blended, Some(w.mlp_norm), &mut moe_in);
        } else {
            rms_norm(&mixed, Some(w.mlp_norm), &mut moe_in);
        }
    }

    let mut indices = [0i32; TOP_K];
    let mut weights = [0f32; TOP_K];
    route(
        &moe_in,
        1,
        w.router_w,
        w.router_b,
        &mut indices,
        &mut weights,
        &mut [],
    );

    let mut moe_out = [0f32; HIDDEN];
    sparse_moe_dispatch_into(&moe_in, &indices, &weights, 1, &w.experts, None, &mut moe_out)?;

    let mut mlp_c = [0f32; HIDDEN];
    short_conv_decode_inplace(
        &moe_out,
        w.mlp_sconv,
        HIDDEN,
        &mut cache.mlp_sconv,
        &mut mlp_c,
    );

    let mut x_out = residual;
    axpy(1.0, &mlp_c, &mut x_out);
    if w.is_block_boundary {
        blocks.extend_from_slice(x_in);
    }
    Ok(x_out)
}

pub fn layers_0_1(
    x_in: &[f32],
    tokens: usize,
    w0: &SlidingLayerWeights<'_>,
    w1: &SlidingLayerWeights<'_>,
) -> Vec<f32> {
    let empty: Vec<Vec<f32>> = (0..tokens).map(|_| Vec::new()).collect();
    let (x1, blocks) = sliding_layer_prefill(x_in, tokens, &empty, w0);
    let (x2, _) = sliding_layer_prefill(&x1, tokens, &blocks, w1);
    x2
}

/// Sliding layers 0-4 then full-attention layer 5.
pub fn layers_0_5(
    x_in: &[f32],
    tokens: usize,
    sliding: &[&SlidingLayerWeights<'_>; 5],
    full: &FullLayerWeights<'_>,
) -> Vec<f32> {
    let mut blocks: Vec<Vec<f32>> = (0..tokens).map(|_| Vec::new()).collect();
    let mut x = x_in.to_vec();
    for w in sliding {
        let (nx, nb) = sliding_layer_prefill(&x, tokens, &blocks, w);
        x = nx;
        blocks = nb;
    }
    let (x_out, _) = full_layer_prefill(&x, tokens, &blocks, full);
    x_out
}
