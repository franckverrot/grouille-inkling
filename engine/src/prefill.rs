//! End-to-end prefill: embed → visit schedule → output residual → logits.

use anyhow::Result;
use safetensors::tensor::SafeTensors;

use crate::head::{embed_tokens, final_norm, unembed_metal, VOCAB};
use crate::ops::{attention_residual_token_ext, HIDDEN};
use crate::visit::{run_visits_metal_cached, MetalAccel, VisitState, WeightCache};
use crate::weights::{bf16_f32, f32s};

/// Owned weights for embed / final residual / unembed.
pub struct HeadBufs {
    pub embed: Vec<f32>,
    pub embed_norm: Vec<f32>,
    pub output_query: Vec<f32>,
    pub final_norm: Vec<f32>,
    pub unembed: Vec<f32>,
}

impl HeadBufs {
    pub fn load(st: &SafeTensors) -> Result<Self> {
        Ok(Self {
            embed: bf16_f32(st, "model.llm.embed.weight")?,
            embed_norm: bf16_f32(st, "model.llm.embed_norm.weight")?,
            output_query: f32s(st, "model.llm.attn_res.output_query")?,
            final_norm: bf16_f32(st, "model.llm.norm.weight")?,
            unembed: bf16_f32(st, "model.llm.unembed.weight")?,
        })
    }
}

/// Mix block stack + last residual with `output_query`, then final RMSNorm.
pub fn finalize_hidden(
    x: &[f32],
    blocks: &[Vec<f32>],
    tokens: usize,
    output_query: &[f32],
    norm_w: &[f32],
    out: &mut [f32],
) {
    assert_eq!(x.len(), tokens * HIDDEN);
    assert_eq!(blocks.len(), tokens);
    assert_eq!(out.len(), tokens * HIDDEN);
    let mut mixed = [0f32; HIDDEN];
    for t in 0..tokens {
        let n_states = blocks[t].len() / HIDDEN;
        // Append x[t] without cloning the block stack.
        attention_residual_token_ext(
            &blocks[t],
            Some(&x[t * HIDDEN..(t + 1) * HIDDEN]),
            n_states,
            output_query,
            true,
            &mut mixed,
        );
        rms_norm_one(
            &mixed,
            norm_w,
            &mut out[t * HIDDEN..(t + 1) * HIDDEN],
        );
    }
}

fn rms_norm_one(x: &[f32], w: &[f32], out: &mut [f32]) {
    final_norm(x, 1, w, out);
}

/// Metal prefill through the full visit schedule; returns last-token logits.
pub fn prefill_logits_metal(
    st: &SafeTensors,
    ids: &[i32],
    head: &HeadBufs,
    accel: &mut MetalAccel,
    cache: &mut WeightCache,
) -> Result<Vec<f32>> {
    let tokens = ids.len();
    let mut x = vec![0f32; tokens * HIDDEN];
    embed_tokens(ids, &head.embed, &head.embed_norm, &mut x);

    let mut state = VisitState::new(x);
    let n_visits = crate::loop_sched::N_VISITS;
    run_visits_metal_cached(st, &mut state, 0, n_visits, cache, accel)?;

    let last_t = tokens - 1;
    let mut hidden = vec![0f32; HIDDEN];
    finalize_hidden(
        &state.x[last_t * HIDDEN..(last_t + 1) * HIDDEN],
        &state.blocks[last_t..last_t + 1],
        1,
        &head.output_query,
        &head.final_norm,
        &mut hidden,
    );

    let mut logits = vec![0f32; VOCAB];
    unembed_metal(
        &mut accel.ops,
        &hidden,
        1,
        &head.unembed,
        &mut logits,
    )?;
    Ok(logits)
}

/// Embed + Metal visits `[start, end)` only (no unembed). For checkpoint tests.
pub fn prefill_hidden_metal(
    st: &SafeTensors,
    ids: &[i32],
    head: &HeadBufs,
    start_visit: usize,
    end_visit: usize,
    accel: &mut MetalAccel,
    cache: &mut WeightCache,
) -> Result<VisitState> {
    let tokens = ids.len();
    let mut x = vec![0f32; tokens * HIDDEN];
    embed_tokens(ids, &head.embed, &head.embed_norm, &mut x);
    let mut state = VisitState::new(x);
    run_visits_metal_cached(st, &mut state, start_visit, end_visit, cache, accel)?;
    Ok(state)
}
