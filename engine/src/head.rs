//! Token embed + final norm + unembed (lm_head) with logit softcap.

use anyhow::Result;

use crate::metal_ops::MetalOps;
use crate::ops::{rms_norm, HIDDEN};

pub const VOCAB: usize = 201_024;
/// Active logit rows (MLX `unpadded_vocab_size`); trailing pad ids are unused.
pub const UNPADDED_VOCAB: usize = 200_058;
pub const LOGIT_SOFTCAP: f32 = 30.0;
/// `<|content_model_end_sampling|>` from generation_config.json.
pub const EOS_TOKEN_ID: i32 = 200_006;
/// `<|endoftext|>` / pad from generation_config.json.
pub const PAD_TOKEN_ID: i32 = 199_999;
/// `<|end_message|>`: closes one content channel (thinking or text).
pub const END_MESSAGE_TOKEN_ID: i32 = 200_010;
/// `<|message_model|>`: opens the assistant turn.
pub const MESSAGE_MODEL_TOKEN_ID: i32 = 200_001;
/// `<|content_text|>`: assistant visible answer channel.
pub const CONTENT_TEXT_TOKEN_ID: i32 = 200_004;
/// `<|content_thinking|>`: assistant scratchpad channel.
pub const CONTENT_THINKING_TOKEN_ID: i32 = 200_008;

/// Lookup `embed.weight[ids]` then RMSNorm with `embed_norm`.
pub fn embed_tokens(ids: &[i32], weight: &[f32], norm: &[f32], out: &mut [f32]) {
    let tokens = ids.len();
    assert_eq!(weight.len(), VOCAB * HIDDEN);
    assert_eq!(norm.len(), HIDDEN);
    assert_eq!(out.len(), tokens * HIDDEN);
    const PARALLEL_AT: usize = 8;
    if tokens < PARALLEL_AT {
        for (t, &id) in ids.iter().enumerate() {
            let id = id as usize;
            assert!(id < VOCAB);
            let row = &weight[id * HIDDEN..(id + 1) * HIDDEN];
            rms_norm(row, Some(norm), &mut out[t * HIDDEN..(t + 1) * HIDDEN]);
        }
        return;
    }
    let n_threads = std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(4)
        .min(tokens)
        .max(1);
    let chunk = tokens.div_ceil(n_threads);
    let out_ptr = out.as_mut_ptr() as usize;
    std::thread::scope(|scope| {
        let mut start = 0usize;
        while start < tokens {
            let end = (start + chunk).min(tokens);
            let ids = &ids[start..end];
            let base = start;
            scope.spawn(move || {
                for (i, &id) in ids.iter().enumerate() {
                    let t = base + i;
                    let id = id as usize;
                    assert!(id < VOCAB);
                    let row = &weight[id * HIDDEN..(id + 1) * HIDDEN];
                    let dst = unsafe {
                        std::slice::from_raw_parts_mut(
                            (out_ptr as *mut f32).add(t * HIDDEN),
                            HIDDEN,
                        )
                    };
                    rms_norm(row, Some(norm), dst);
                }
            });
            start = end;
        }
    });
}

/// Soft-capped logits: `softcap * tanh(y / softcap)`.
pub fn softcap_logits(y: &mut [f32], softcap: f32) {
    if y.is_empty() {
        return;
    }
    #[cfg(target_os = "macos")]
    {
        // vDSP path for the ~200k unpadded vocab row on every decode step.
        extern "C" {
            fn vDSP_vsdiv(a: *const f32, ia: i32, b: *const f32, c: *mut f32, ic: i32, n: u32);
            fn vvtanhf(y: *mut f32, x: *const f32, n: *const i32);
            fn vDSP_vsmul(a: *const f32, ia: i32, b: *const f32, c: *mut f32, ic: i32, n: u32);
        }
        let n = y.len();
        let n_i = n as i32;
        let n_u = n as u32;
        unsafe {
            vDSP_vsdiv(y.as_ptr(), 1, &softcap, y.as_mut_ptr(), 1, n_u);
            vvtanhf(y.as_mut_ptr(), y.as_ptr(), &n_i);
            vDSP_vsmul(y.as_ptr(), 1, &softcap, y.as_mut_ptr(), 1, n_u);
        }
    }
    #[cfg(not(target_os = "macos"))]
    {
        let inv = 1.0 / softcap;
        for v in y.iter_mut() {
            *v = softcap * (*v * inv).tanh();
        }
    }
}

/// CPU unembed for one or more hidden states: `out[t] = softcap(W @ x[t])`.
pub fn unembed(
    x: &[f32],
    tokens: usize,
    weight: &[f32],
    out: &mut [f32],
) {
    assert_eq!(x.len(), tokens * HIDDEN);
    assert_eq!(weight.len(), VOCAB * HIDDEN);
    assert_eq!(out.len(), tokens * VOCAB);
    for t in 0..tokens {
        let xt = &x[t * HIDDEN..(t + 1) * HIDDEN];
        let ot = &mut out[t * VOCAB..(t + 1) * VOCAB];
        for v in 0..VOCAB {
            let row = &weight[v * HIDDEN..(v + 1) * HIDDEN];
            let mut acc = 0f32;
            for i in 0..HIDDEN {
                acc += row[i] * xt[i];
            }
            ot[v] = acc;
        }
        softcap_logits(ot, LOGIT_SOFTCAP);
    }
}

/// Metal unembed (dense GEMV + softcap on GPU). Prefer `tokens == 1` for decode.
///
/// Only the first [`UNPADDED_VOCAB`] rows are projected (matches MLX); the rest of
/// each row is masked so argmax/sample ignore pad ids.
pub fn unembed_metal(
    ops: &mut MetalOps,
    x: &[f32],
    tokens: usize,
    weight: &[f32],
    out: &mut [f32],
) -> Result<()> {
    assert_eq!(x.len(), tokens * HIDDEN);
    assert_eq!(weight.len(), VOCAB * HIDDEN);
    assert_eq!(out.len(), tokens * VOCAB);
    let w_live = &weight[..UNPADDED_VOCAB * HIDDEN];
    if tokens == 1 {
        ops.dense_gemv_softcap(
            x,
            w_live,
            1,
            UNPADDED_VOCAB,
            HIDDEN,
            LOGIT_SOFTCAP,
            &mut out[..UNPADDED_VOCAB],
        )?;
        // Finite mask so softcap-range asserts stay meaningful; sample cuts at UNPADDED.
        out[UNPADDED_VOCAB..].fill(-1.0e9);
        return Ok(());
    }
    let mut live = vec![0f32; tokens * UNPADDED_VOCAB];
    ops.dense_gemv_softcap(
        x,
        w_live,
        tokens,
        UNPADDED_VOCAB,
        HIDDEN,
        LOGIT_SOFTCAP,
        &mut live,
    )?;
    for t in 0..tokens {
        let dst = &mut out[t * VOCAB..(t + 1) * VOCAB];
        dst[..UNPADDED_VOCAB].copy_from_slice(&live[t * UNPADDED_VOCAB..(t + 1) * UNPADDED_VOCAB]);
        dst[UNPADDED_VOCAB..].fill(-1.0e9);
    }
    Ok(())
}

/// Final RMSNorm before unembed (`model.llm.norm`).
pub fn final_norm(x: &[f32], tokens: usize, weight: &[f32], out: &mut [f32]) {
    assert_eq!(x.len(), tokens * HIDDEN);
    assert_eq!(out.len(), tokens * HIDDEN);
    assert_eq!(weight.len(), HIDDEN);
    for t in 0..tokens {
        rms_norm(
            &x[t * HIDDEN..(t + 1) * HIDDEN],
            Some(weight),
            &mut out[t * HIDDEN..(t + 1) * HIDDEN],
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softcap_matches_scalar() {
        let mut y = [0.0f32, 15.0, -45.0, 90.0, -0.5];
        let mut ref_y = y;
        let cap = 30.0f32;
        softcap_logits(&mut y, cap);
        for v in ref_y.iter_mut() {
            *v = cap * (*v / cap).tanh();
        }
        for (a, b) in y.iter().zip(ref_y.iter()) {
            assert!((a - b).abs() < 1e-5, "{a} vs {b}");
        }
    }
}
