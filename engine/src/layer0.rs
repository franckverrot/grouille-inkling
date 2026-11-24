//! Full layer-0 prefill forward (sliding attn + MoE), f32 CPU.

use crate::attn::{sliding_attention, SlidingAttnWeights};
use crate::moe::{sparse_moe_layer0, ExpertPack, TOP_K};
use crate::ops::{attention_residual_token, gemv, rms_norm, short_conv, HIDDEN};

pub struct Layer0Weights<'a> {
    pub attn_norm: &'a [f32],
    pub mlp_norm: &'a [f32],
    pub attn_sconv: &'a [f32],
    pub mlp_sconv: &'a [f32],
    pub mlp_query: &'a [f32],
    pub router_w: &'a [f32], // [128, H]
    pub router_b: &'a [f32], // [128]
    pub attn: SlidingAttnWeights<'a>,
    pub experts: ExpertPack<'a>,
}

fn route(
    x: &[f32],
    tokens: usize,
    w: &[f32],
    b: &[f32],
    indices: &mut [i32],
    weights: &mut [f32],
) {
    const N_EXP: usize = 128;
    let mut logits = vec![0f32; tokens * N_EXP];
    for t in 0..tokens {
        gemv(
            w,
            N_EXP,
            &x[t * HIDDEN..(t + 1) * HIDDEN],
            &mut logits[t * N_EXP..(t + 1) * N_EXP],
        );
    }
    crate::route::route_from_logits(&logits, tokens, b, indices, weights);
}

pub fn layer0_prefill(x_in: &[f32], tokens: usize, w: &Layer0Weights<'_>) -> Vec<f32> {
    assert_eq!(x_in.len(), tokens * HIDDEN);
    let mut x_norm = vec![0f32; tokens * HIDDEN];
    for t in 0..tokens {
        rms_norm(
            &x_in[t * HIDDEN..(t + 1) * HIDDEN],
            Some(w.attn_norm),
            &mut x_norm[t * HIDDEN..(t + 1) * HIDDEN],
        );
    }
    let mut attn = vec![0f32; tokens * HIDDEN];
    sliding_attention(&x_norm, tokens, &w.attn, &mut attn);

    let mut residual = vec![0f32; tokens * HIDDEN];
    short_conv(&attn, w.attn_sconv, tokens, HIDDEN, &mut residual);

    let mut moe_in = vec![0f32; tokens * HIDDEN];
    let mut mixed = vec![0f32; HIDDEN];
    let mut states = vec![0f32; 2 * HIDDEN];
    for t in 0..tokens {
        states[..HIDDEN].copy_from_slice(&x_in[t * HIDDEN..(t + 1) * HIDDEN]);
        states[HIDDEN..].copy_from_slice(&residual[t * HIDDEN..(t + 1) * HIDDEN]);
        attention_residual_token(&states, 2, w.mlp_query, true, &mut mixed);
        rms_norm(&mixed, Some(w.mlp_norm), &mut moe_in[t * HIDDEN..(t + 1) * HIDDEN]);
    }

    let mut indices = vec![0i32; tokens * TOP_K];
    let mut weights = vec![0f32; tokens * TOP_K];
    route(
        &moe_in,
        tokens,
        w.router_w,
        w.router_b,
        &mut indices,
        &mut weights,
    );
    let moe_out = sparse_moe_layer0(&moe_in, &indices, &weights, tokens, &w.experts);

    let mut mlp_c = vec![0f32; tokens * HIDDEN];
    short_conv(&moe_out, w.mlp_sconv, tokens, HIDDEN, &mut mlp_c);
    let mut x_out = residual;
    for i in 0..x_out.len() {
        x_out[i] += mlp_c[i];
    }
    x_out
}
