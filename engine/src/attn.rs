//! Sliding-window and full attention (CPU f32).

use anyhow::Result;

use crate::cache::VisitCache;
use crate::ops::{
    axpy, dot, dynamic_short_conv, dynamic_short_conv_decode_inplace, gemv, l2_clip, rms_norm,
    short_conv, short_conv_decode_inplace, short_conv_with_state, sigmoid_gate_mul, softmax_inplace,
    CONV_K, HIDDEN,
};

/// `rh · rel_proj[:, dist]` with column stride `REL_EXTENT`.
#[inline]
fn rel_bias(rh: &[f32], rel_proj: &[f32], dist: usize) -> f32 {
    debug_assert_eq!(rh.len(), D_REL);
    debug_assert!(dist < REL_EXTENT);
    // D_REL is 16: a hand loop beats cblas_sdot setup on this tiny strided product.
    // rel_proj layout: [d * REL_EXTENT + dist]
    let mut acc = 0f32;
    let mut off = dist;
    for i in 0..D_REL {
        acc += rh[i] * rel_proj[off];
        off += REL_EXTENT;
    }
    acc
}

pub const N_HEADS: usize = 24;
pub const N_KV: usize = 8;
pub const DH: usize = 128;
pub const D_REL: usize = 16;
pub const REL_EXTENT: usize = 1024;
pub const SWA: usize = 1024;
pub const QK_CLIP: f32 = 22.627_417;
pub const REL_CLIP: f32 = 8.0;

pub struct SlidingAttnWeights<'a> {
    /// `.gpk` layer index when attn tensors live in MetalOps registered f16 buffers
    /// (host `wq`..`gate`/`sink`/`wo` are then empty). `None` for host-f32 weights.
    pub layer: Option<usize>,
    pub wq: &'a [f32],    // [N_HEADS*DH, H]
    pub wk: &'a [f32],    // [N_KV*DH, H]
    pub wv: &'a [f32],    // [N_KV*DH, H]
    pub wr: &'a [f32],    // [N_HEADS*D_REL, H]
    pub wo: &'a [f32],    // [H, N_HEADS*DH]
    pub gate: &'a [f32],  // [N_HEADS*DH, H]
    pub sink: &'a [f32],  // [N_HEADS, H]
    pub q_scale: &'a [f32], // [N_HEADS, DH]
    pub rel_scale: &'a [f32], // [N_HEADS, D_REL]
    pub rel_proj: &'a [f32], // [D_REL, REL_EXTENT]
    pub k_sconv: &'a [f32], // [N_KV*DH, 4]
    pub v_sconv: &'a [f32],
}

pub fn sliding_attention(x: &[f32], tokens: usize, w: &SlidingAttnWeights<'_>, out: &mut [f32]) {
    sliding_attention_qkv(x, x, x, tokens, w, out);
}

/// Sliding attention with separate Q/K/V inputs (message-controller path).
pub fn sliding_attention_qkv(
    q_in: &[f32],
    k_in: &[f32],
    v_in: &[f32],
    tokens: usize,
    w: &SlidingAttnWeights<'_>,
    out: &mut [f32],
) {
    sliding_attention_qkv_inner(q_in, k_in, v_in, tokens, w, out, None);
}

/// Prefill sliding attention and leave `cache` ready for the next decode step.
pub fn sliding_attention_qkv_prefill_cache(
    q_in: &[f32],
    k_in: &[f32],
    v_in: &[f32],
    tokens: usize,
    w: &SlidingAttnWeights<'_>,
    out: &mut [f32],
    cache: &mut VisitCache,
) {
    sliding_attention_qkv_inner(q_in, k_in, v_in, tokens, w, out, Some(cache));
}

fn sliding_attention_qkv_inner(
    q_in: &[f32],
    k_in: &[f32],
    v_in: &[f32],
    tokens: usize,
    w: &SlidingAttnWeights<'_>,
    out: &mut [f32],
    mut cache: Option<&mut VisitCache>,
) {
    assert_eq!(q_in.len(), tokens * HIDDEN);
    assert_eq!(k_in.len(), tokens * HIDDEN);
    assert_eq!(v_in.len(), tokens * HIDDEN);
    assert_eq!(out.len(), tokens * HIDDEN);
    let kv_ch = N_KV * DH;
    let q_ch = N_HEADS * DH;

    let mut q = vec![0f32; tokens * q_ch];
    let mut k = vec![0f32; tokens * kv_ch];
    let mut v = vec![0f32; tokens * kv_ch];
    let mut r = vec![0f32; tokens * N_HEADS * D_REL];
    let mut k_c = vec![0f32; tokens * kv_ch];
    let mut v_c = vec![0f32; tokens * kv_ch];

    for t in 0..tokens {
        gemv(
            w.wq,
            q_ch,
            &q_in[t * HIDDEN..(t + 1) * HIDDEN],
            &mut q[t * q_ch..(t + 1) * q_ch],
        );
        gemv(
            w.wk,
            kv_ch,
            &k_in[t * HIDDEN..(t + 1) * HIDDEN],
            &mut k[t * kv_ch..(t + 1) * kv_ch],
        );
        gemv(
            w.wv,
            kv_ch,
            &v_in[t * HIDDEN..(t + 1) * HIDDEN],
            &mut v[t * kv_ch..(t + 1) * kv_ch],
        );
        gemv(
            w.wr,
            N_HEADS * D_REL,
            &q_in[t * HIDDEN..(t + 1) * HIDDEN],
            &mut r[t * N_HEADS * D_REL..(t + 1) * N_HEADS * D_REL],
        );
    }
    if let Some(cache) = cache.as_mut() {
        let mut k_hist = vec![0f32; kv_ch * (CONV_K - 1)];
        let mut v_hist = vec![0f32; kv_ch * (CONV_K - 1)];
        short_conv_with_state(
            &k,
            w.k_sconv,
            tokens,
            kv_ch,
            None,
            &mut k_c,
            Some(&mut k_hist),
        );
        short_conv_with_state(
            &v,
            w.v_sconv,
            tokens,
            kv_ch,
            None,
            &mut v_c,
            Some(&mut v_hist),
        );
        cache.k_sconv = k_hist;
        cache.v_sconv = v_hist;
    } else {
        short_conv(&k, w.k_sconv, tokens, kv_ch, &mut k_c);
        short_conv(&v, w.v_sconv, tokens, kv_ch, &mut v_c);
    }

    // l2 clip q/k/r; value rms-norm
    for t in 0..tokens {
        for h in 0..N_HEADS {
            let qh = &mut q[t * q_ch + h * DH..t * q_ch + (h + 1) * DH];
            let scale = &w.q_scale[h * DH..(h + 1) * DH];
            l2_clip(qh, QK_CLIP, Some(scale));
            let rh = &mut r[t * N_HEADS * D_REL + h * D_REL
                ..t * N_HEADS * D_REL + (h + 1) * D_REL];
            let rscale = &w.rel_scale[h * D_REL..(h + 1) * D_REL];
            l2_clip(rh, REL_CLIP, Some(rscale));
        }
        for h in 0..N_KV {
            let kh = &mut k_c[t * kv_ch + h * DH..t * kv_ch + (h + 1) * DH];
            l2_clip(kh, QK_CLIP, None);
            let mut vn = [0f32; DH];
            let vh = &v_c[t * kv_ch + h * DH..t * kv_ch + (h + 1) * DH];
            rms_norm(vh, None, &mut vn);
            v_c[t * kv_ch + h * DH..t * kv_ch + (h + 1) * DH].copy_from_slice(&vn);
        }
    }

    if let Some(cache) = cache.as_mut() {
        for t in 0..tokens {
            cache.kv.push(
                &k_c[t * kv_ch..(t + 1) * kv_ch],
                &v_c[t * kv_ch..(t + 1) * kv_ch],
            );
        }
    }

    let scale = (DH as f32).sqrt().recip();
    let rep = N_HEADS / N_KV;
    let s_len = tokens + 1; // + sink
    let mut scores = vec![0f32; s_len];
    let mut ctx = vec![0f32; tokens * q_ch];

    for t in 0..tokens {
        let mut sinks = [0f32; N_HEADS];
        gemv(
            w.sink,
            N_HEADS,
            &q_in[t * HIDDEN..(t + 1) * HIDDEN],
            &mut sinks,
        );

        for h in 0..N_HEADS {
            let kv_h = h / rep;
            let qh = &q[t * q_ch + h * DH..t * q_ch + (h + 1) * DH];
            for j in 0..tokens {
                let dist = t as i32 - j as i32;
                let logit = if dist < 0 || (SWA > 0 && dist >= SWA as i32) {
                    f32::NEG_INFINITY
                } else if (dist as usize) < REL_EXTENT {
                    let rh = &r[t * N_HEADS * D_REL + h * D_REL
                        ..t * N_HEADS * D_REL + (h + 1) * D_REL];
                    rel_bias(rh, w.rel_proj, dist as usize)
                } else {
                    0.0
                };
                let kh = &k_c[j * kv_ch + kv_h * DH..j * kv_ch + (kv_h + 1) * DH];
                scores[j] = dot(qh, kh) * scale + logit;
            }
            scores[tokens] = sinks[h];
            softmax_inplace(&mut scores);
            let out_h = &mut ctx[t * q_ch + h * DH..t * q_ch + (h + 1) * DH];
            out_h.fill(0.0);
            for j in 0..tokens {
                let vh = &v_c[j * kv_ch + kv_h * DH..j * kv_ch + (kv_h + 1) * DH];
                axpy(scores[j], vh, out_h);
            }
        }

        let mut gate = vec![0f32; q_ch];
        gemv(
            w.gate,
            q_ch,
            &q_in[t * HIDDEN..(t + 1) * HIDDEN],
            &mut gate,
        );
        let ctx_t = &mut ctx[t * q_ch..(t + 1) * q_ch];
        sigmoid_gate_mul(&gate, ctx_t);
        gemv(w.wo, HIDDEN, ctx_t, &mut out[t * HIDDEN..(t + 1) * HIDDEN]);
    }
}

/// Single-token sliding attention decode step on the host; updates `cache` in place.
pub fn sliding_attention_decode_step(
    q_in: &[f32],
    k_in: &[f32],
    v_in: &[f32],
    w: &SlidingAttnWeights<'_>,
    cache: &mut VisitCache,
    out: &mut [f32],
) -> Result<()> {
    assert_eq!(q_in.len(), HIDDEN);
    assert_eq!(k_in.len(), HIDDEN);
    assert_eq!(v_in.len(), HIDDEN);
    assert_eq!(out.len(), HIDDEN);
    let kv_ch = N_KV * DH;
    let q_ch = N_HEADS * DH;

    // Stack buffers: decode is T=1 and these dims are tiny vs heap traffic ×48 visits.
    let mut q = [0f32; N_HEADS * DH];
    let mut k = [0f32; N_KV * DH];
    let mut v = [0f32; N_KV * DH];
    let mut r = [0f32; N_HEADS * D_REL];
    let mut sinks = [0f32; N_HEADS];
    let mut gate = [0f32; N_HEADS * DH];
    gemv(w.wq, q_ch, q_in, &mut q);
    gemv(w.wk, kv_ch, k_in, &mut k);
    gemv(w.wv, kv_ch, v_in, &mut v);
    gemv(w.wr, N_HEADS * D_REL, q_in, &mut r);
    gemv(w.sink, N_HEADS, q_in, &mut sinks);
    gemv(w.gate, q_ch, q_in, &mut gate);

    let mut k_c = [0f32; N_KV * DH];
    let mut v_c = [0f32; N_KV * DH];
    short_conv_decode_inplace(&k, w.k_sconv, kv_ch, &mut cache.k_sconv, &mut k_c);
    short_conv_decode_inplace(&v, w.v_sconv, kv_ch, &mut cache.v_sconv, &mut v_c);

    for h in 0..N_HEADS {
        l2_clip(
            &mut q[h * DH..(h + 1) * DH],
            QK_CLIP,
            Some(&w.q_scale[h * DH..(h + 1) * DH]),
        );
        l2_clip(
            &mut r[h * D_REL..(h + 1) * D_REL],
            REL_CLIP,
            Some(&w.rel_scale[h * D_REL..(h + 1) * D_REL]),
        );
    }
    for h in 0..N_KV {
        l2_clip(&mut k_c[h * DH..(h + 1) * DH], QK_CLIP, None);
        let mut vn = [0f32; DH];
        rms_norm(&v_c[h * DH..(h + 1) * DH], None, &mut vn);
        v_c[h * DH..(h + 1) * DH].copy_from_slice(&vn);
    }
    cache.kv.push(&k_c, &v_c);
    let tlen = cache.kv.len;
    assert!(tlen <= SWA);

    let t = tlen - 1; // position of the new token within the ring window
    let scale = (DH as f32).sqrt().recip();
    let rep = N_HEADS / N_KV;
    let mut scores = [0f32; SWA + 1];
    let mut ctx = [0f32; N_HEADS * DH];

    let sdpa_one = |h: usize, scores: &mut [f32], ctx: &mut [f32]| {
        let kv_h = h / rep;
        let qh = &q[h * DH..(h + 1) * DH];
        for j in 0..tlen {
            let dist = t as i32 - j as i32;
            let logit = if dist < 0 || (SWA > 0 && dist >= SWA as i32) {
                f32::NEG_INFINITY
            } else if (dist as usize) < REL_EXTENT {
                rel_bias(
                    &r[h * D_REL..(h + 1) * D_REL],
                    w.rel_proj,
                    dist as usize,
                )
            } else {
                0.0
            };
            let rj = cache.kv.row(j);
            let kh = &cache.kv.k[rj * kv_ch + kv_h * DH..rj * kv_ch + (kv_h + 1) * DH];
            scores[j] = dot(qh, kh) * scale + logit;
        }
        scores[tlen] = sinks[h];
        softmax_inplace(&mut scores[..=tlen]);
        let out_h = &mut ctx[h * DH..(h + 1) * DH];
        out_h.fill(0.0);
        for j in 0..tlen {
            let rj = cache.kv.row(j);
            let vh = &cache.kv.v[rj * kv_ch + kv_h * DH..rj * kv_ch + (kv_h + 1) * DH];
            axpy(scores[j], vh, out_h);
        }
    };

    if tlen >= 16 {
        const WORKERS: usize = 4;
        std::thread::scope(|scope| {
            for w in 0..WORKERS {
                let ctx_ptr = ctx.as_mut_ptr() as usize;
                scope.spawn(move || {
                    let mut local = [0f32; SWA + 1];
                    let ctx =
                        unsafe { std::slice::from_raw_parts_mut(ctx_ptr as *mut f32, N_HEADS * DH) };
                    for h in (0..N_HEADS).filter(|h| h % WORKERS == w) {
                        sdpa_one(h, &mut local, ctx);
                    }
                });
            }
        });
    } else {
        for h in 0..N_HEADS {
            sdpa_one(h, &mut scores, &mut ctx);
        }
    }

    sigmoid_gate_mul(&gate, &mut ctx);
    gemv(w.wo, HIDDEN, &ctx, out);
    Ok(())
}

pub const LOG_SCALE_N_FLOOR: f32 = 8192.0;
pub const LOG_SCALE_ALPHA: f32 = 0.1;

pub struct FullAttnWeights<'a> {
    /// See [`SlidingAttnWeights::layer`].
    pub layer: Option<usize>,
    pub wq: &'a [f32],
    pub wk: &'a [f32],
    pub wv: &'a [f32],
    pub wr: &'a [f32],
    pub wo: &'a [f32],
    pub gate: &'a [f32],
    pub sink: &'a [f32],
    pub q_scale: &'a [f32],
    pub rel_scale: &'a [f32],
    pub rel_proj: &'a [f32],
    /// Dynamic K short-conv: static `[K, C]`, weight_proj `[N_KV*K, H]`.
    pub k_static: &'a [f32],
    pub k_wproj: &'a [f32],
    pub v_static: &'a [f32],
    pub v_wproj: &'a [f32],
}

/// Full (non-sliding) attention with dynamic K/V short-convs.
/// `q_in` / `k_in` / `v_in` are already layernormed `[T*H]`.
pub fn full_attention(
    q_in: &[f32],
    k_in: &[f32],
    v_in: &[f32],
    tokens: usize,
    w: &FullAttnWeights<'_>,
    out: &mut [f32],
) {
    full_attention_inner(q_in, k_in, v_in, tokens, w, out, None);
}

/// Prefill full attention and leave `cache` ready for the next decode step.
pub fn full_attention_prefill_cache(
    q_in: &[f32],
    k_in: &[f32],
    v_in: &[f32],
    tokens: usize,
    w: &FullAttnWeights<'_>,
    out: &mut [f32],
    cache: &mut VisitCache,
) {
    full_attention_inner(q_in, k_in, v_in, tokens, w, out, Some(cache));
}

fn full_attention_inner(
    q_in: &[f32],
    k_in: &[f32],
    v_in: &[f32],
    tokens: usize,
    w: &FullAttnWeights<'_>,
    out: &mut [f32],
    mut cache: Option<&mut VisitCache>,
) {
    assert_eq!(q_in.len(), tokens * HIDDEN);
    assert_eq!(k_in.len(), tokens * HIDDEN);
    assert_eq!(v_in.len(), tokens * HIDDEN);
    assert_eq!(out.len(), tokens * HIDDEN);
    let kv_ch = N_KV * DH;
    let q_ch = N_HEADS * DH;

    let mut q = vec![0f32; tokens * q_ch];
    let mut k = vec![0f32; tokens * kv_ch];
    let mut v = vec![0f32; tokens * kv_ch];
    let mut r = vec![0f32; tokens * N_HEADS * D_REL];
    let mut k_c = vec![0f32; tokens * kv_ch];
    let mut v_c = vec![0f32; tokens * kv_ch];

    for t in 0..tokens {
        gemv(
            w.wq,
            q_ch,
            &q_in[t * HIDDEN..(t + 1) * HIDDEN],
            &mut q[t * q_ch..(t + 1) * q_ch],
        );
        gemv(
            w.wk,
            kv_ch,
            &k_in[t * HIDDEN..(t + 1) * HIDDEN],
            &mut k[t * kv_ch..(t + 1) * kv_ch],
        );
        gemv(
            w.wv,
            kv_ch,
            &v_in[t * HIDDEN..(t + 1) * HIDDEN],
            &mut v[t * kv_ch..(t + 1) * kv_ch],
        );
        gemv(
            w.wr,
            N_HEADS * D_REL,
            &q_in[t * HIDDEN..(t + 1) * HIDDEN],
            &mut r[t * N_HEADS * D_REL..(t + 1) * N_HEADS * D_REL],
        );
    }
    dynamic_short_conv(
        &k, k_in, w.k_static, w.k_wproj, tokens, kv_ch, HIDDEN, N_KV, CONV_K, &mut k_c,
    );
    dynamic_short_conv(
        &v, v_in, w.v_static, w.v_wproj, tokens, kv_ch, HIDDEN, N_KV, CONV_K, &mut v_c,
    );

    if let Some(cache) = cache.as_mut() {
        let hist = CONV_K - 1;
        let mut k_hist = vec![0f32; kv_ch * hist];
        let mut v_hist = vec![0f32; kv_ch * hist];
        for c in 0..kv_ch {
            for age in 0..hist {
                let src = tokens as isize - hist as isize + age as isize;
                let kv = if src < 0 {
                    0.0
                } else {
                    k[src as usize * kv_ch + c]
                };
                let vv = if src < 0 {
                    0.0
                } else {
                    v[src as usize * kv_ch + c]
                };
                k_hist[c * hist + age] = kv;
                v_hist[c * hist + age] = vv;
            }
        }
        cache.k_sconv = k_hist;
        cache.v_sconv = v_hist;
    }

    for t in 0..tokens {
        for h in 0..N_HEADS {
            let qh = &mut q[t * q_ch + h * DH..t * q_ch + (h + 1) * DH];
            let scale = &w.q_scale[h * DH..(h + 1) * DH];
            l2_clip(qh, QK_CLIP, Some(scale));
            let rh = &mut r[t * N_HEADS * D_REL + h * D_REL
                ..t * N_HEADS * D_REL + (h + 1) * D_REL];
            let rscale = &w.rel_scale[h * D_REL..(h + 1) * D_REL];
            l2_clip(rh, REL_CLIP, Some(rscale));
        }
        for h in 0..N_KV {
            let kh = &mut k_c[t * kv_ch + h * DH..t * kv_ch + (h + 1) * DH];
            l2_clip(kh, QK_CLIP, None);
            let mut vn = [0f32; DH];
            let vh = &v_c[t * kv_ch + h * DH..t * kv_ch + (h + 1) * DH];
            rms_norm(vh, None, &mut vn);
            v_c[t * kv_ch + h * DH..t * kv_ch + (h + 1) * DH].copy_from_slice(&vn);
        }
        // log-scaling after clip; tau=1 for positions < n_floor
        let pos = (t + 1) as f32;
        let tau = 1.0 + LOG_SCALE_ALPHA * (pos / LOG_SCALE_N_FLOOR).max(1.0).ln();
        if (tau - 1.0).abs() > 0.0 {
            for h in 0..N_HEADS {
                let qh = &mut q[t * q_ch + h * DH..t * q_ch + (h + 1) * DH];
                for d in 0..DH {
                    qh[d] *= tau;
                }
                let rh = &mut r[t * N_HEADS * D_REL + h * D_REL
                    ..t * N_HEADS * D_REL + (h + 1) * D_REL];
                for d in 0..D_REL {
                    rh[d] *= tau;
                }
            }
        }
    }

    if let Some(cache) = cache.as_mut() {
        for t in 0..tokens {
            cache.kv.push(
                &k_c[t * kv_ch..(t + 1) * kv_ch],
                &v_c[t * kv_ch..(t + 1) * kv_ch],
            );
        }
    }

    let scale = (DH as f32).sqrt().recip();
    let rep = N_HEADS / N_KV;
    let s_len = tokens + 1;
    let mut scores = vec![0f32; s_len];
    let mut ctx = vec![0f32; tokens * q_ch];

    for t in 0..tokens {
        let mut sinks = [0f32; N_HEADS];
        gemv(
            w.sink,
            N_HEADS,
            &q_in[t * HIDDEN..(t + 1) * HIDDEN],
            &mut sinks,
        );

        for h in 0..N_HEADS {
            let kv_h = h / rep;
            let qh = &q[t * q_ch + h * DH..t * q_ch + (h + 1) * DH];
            for j in 0..tokens {
                let dist = t as i32 - j as i32;
                let logit = if dist < 0 {
                    f32::NEG_INFINITY
                } else if (dist as usize) < REL_EXTENT {
                    let rh = &r[t * N_HEADS * D_REL + h * D_REL
                        ..t * N_HEADS * D_REL + (h + 1) * D_REL];
                    rel_bias(rh, w.rel_proj, dist as usize)
                } else {
                    0.0
                };
                let kh = &k_c[j * kv_ch + kv_h * DH..j * kv_ch + (kv_h + 1) * DH];
                scores[j] = dot(qh, kh) * scale + logit;
            }
            scores[tokens] = sinks[h];
            softmax_inplace(&mut scores);
            let out_h = &mut ctx[t * q_ch + h * DH..t * q_ch + (h + 1) * DH];
            out_h.fill(0.0);
            for j in 0..tokens {
                let vh = &v_c[j * kv_ch + kv_h * DH..j * kv_ch + (kv_h + 1) * DH];
                axpy(scores[j], vh, out_h);
            }
        }

        let mut gate = vec![0f32; q_ch];
        gemv(
            w.gate,
            q_ch,
            &q_in[t * HIDDEN..(t + 1) * HIDDEN],
            &mut gate,
        );
        let ctx_t = &mut ctx[t * q_ch..(t + 1) * q_ch];
        sigmoid_gate_mul(&gate, ctx_t);
        gemv(
            w.wo,
            HIDDEN,
            ctx_t,
            &mut out[t * HIDDEN..(t + 1) * HIDDEN],
        );
    }
}

/// Single-token full attention decode; grows `cache.kv` (no SWA ring).
pub fn full_attention_decode_step(
    q_in: &[f32],
    k_in: &[f32],
    v_in: &[f32],
    w: &FullAttnWeights<'_>,
    cache: &mut VisitCache,
    out: &mut [f32],
) -> Result<()> {
    assert_eq!(q_in.len(), HIDDEN);
    assert_eq!(k_in.len(), HIDDEN);
    assert_eq!(v_in.len(), HIDDEN);
    assert_eq!(out.len(), HIDDEN);
    let kv_ch = N_KV * DH;
    let q_ch = N_HEADS * DH;

    let mut q = [0f32; N_HEADS * DH];
    let mut k = [0f32; N_KV * DH];
    let mut v = [0f32; N_KV * DH];
    let mut r = [0f32; N_HEADS * D_REL];
    let mut sinks = [0f32; N_HEADS];
    let mut gate = [0f32; N_HEADS * DH];
    // Same as sliding: six T=1 GEMVs on one Metal CB.
    gemv(w.wq, q_ch, q_in, &mut q);
    gemv(w.wk, kv_ch, k_in, &mut k);
    gemv(w.wv, kv_ch, v_in, &mut v);
    gemv(w.wr, N_HEADS * D_REL, q_in, &mut r);
    gemv(w.sink, N_HEADS, q_in, &mut sinks);
    gemv(w.gate, q_ch, q_in, &mut gate);

    let mut k_c = [0f32; N_KV * DH];
    let mut v_c = [0f32; N_KV * DH];
    dynamic_short_conv_decode_inplace(
        &k,
        k_in,
        w.k_static,
        w.k_wproj,
        kv_ch,
        HIDDEN,
        N_KV,
        &mut cache.k_sconv,
        &mut k_c,
    );
    dynamic_short_conv_decode_inplace(
        &v,
        v_in,
        w.v_static,
        w.v_wproj,
        kv_ch,
        HIDDEN,
        N_KV,
        &mut cache.v_sconv,
        &mut v_c,
    );

    for h in 0..N_HEADS {
        l2_clip(
            &mut q[h * DH..(h + 1) * DH],
            QK_CLIP,
            Some(&w.q_scale[h * DH..(h + 1) * DH]),
        );
        l2_clip(
            &mut r[h * D_REL..(h + 1) * D_REL],
            REL_CLIP,
            Some(&w.rel_scale[h * D_REL..(h + 1) * D_REL]),
        );
    }
    for h in 0..N_KV {
        l2_clip(&mut k_c[h * DH..(h + 1) * DH], QK_CLIP, None);
        let mut vn = [0f32; DH];
        rms_norm(&v_c[h * DH..(h + 1) * DH], None, &mut vn);
        v_c[h * DH..(h + 1) * DH].copy_from_slice(&vn);
    }

    cache.kv.push(&k_c, &v_c);
    let tlen = cache.kv.len;
    let t = tlen - 1;
    let pos = tlen as f32;
    let tau = 1.0 + LOG_SCALE_ALPHA * (pos / LOG_SCALE_N_FLOOR).max(1.0).ln();
    if (tau - 1.0).abs() > 0.0 {
        for h in 0..N_HEADS {
            for d in 0..DH {
                q[h * DH + d] *= tau;
            }
            for d in 0..D_REL {
                r[h * D_REL + d] *= tau;
            }
        }
    }

    // Reuse sliding fused SDPA+gate+wo (SWA=1024 ≥ short decode windows).
    let scale = (DH as f32).sqrt().recip();
    let rep = N_HEADS / N_KV;
    // Short/medium chats stay stack-only; longer prompts fall back to a heap buffer.
    const STACK_SCORES: usize = 512;
    let mut scores_stack = [0f32; STACK_SCORES];
    let mut scores_heap = if tlen + 1 > STACK_SCORES {
        vec![0f32; tlen + 1]
    } else {
        Vec::new()
    };
    let scores: &mut [f32] = if tlen < STACK_SCORES {
        &mut scores_stack[..tlen + 1]
    } else {
        scores_heap.as_mut_slice()
    };
    let mut ctx = [0f32; N_HEADS * DH];

    let sdpa_one = |h: usize, scores: &mut [f32], ctx: &mut [f32]| {
        let kv_h = h / rep;
        let qh = &q[h * DH..(h + 1) * DH];
        for j in 0..tlen {
            let dist = t as i32 - j as i32;
            let logit = if dist < 0 {
                f32::NEG_INFINITY
            } else if (dist as usize) < REL_EXTENT {
                rel_bias(
                    &r[h * D_REL..(h + 1) * D_REL],
                    w.rel_proj,
                    dist as usize,
                )
            } else {
                0.0
            };
            let rj = cache.kv.row(j);
            let kh = &cache.kv.k[rj * kv_ch + kv_h * DH..rj * kv_ch + (kv_h + 1) * DH];
            scores[j] = dot(qh, kh) * scale + logit;
        }
        scores[tlen] = sinks[h];
        softmax_inplace(scores);
        let out_h = &mut ctx[h * DH..(h + 1) * DH];
        out_h.fill(0.0);
        for j in 0..tlen {
            let rj = cache.kv.row(j);
            let vh = &cache.kv.v[rj * kv_ch + kv_h * DH..rj * kv_ch + (kv_h + 1) * DH];
            axpy(scores[j], vh, out_h);
        }
    };

    // Medium/long full-KV: parallelize independent heads (4 workers).
    // Threshold 16: short prompts stay serial (spawn > work); chats past a
    // few turns benefit sooner than the old 32-token cutoff.
    if tlen >= 16 {
        const WORKERS: usize = 4;
        std::thread::scope(|scope| {
            for w in 0..WORKERS {
                let ctx_ptr = ctx.as_mut_ptr() as usize;
                let scores_len = tlen + 1;
                scope.spawn(move || {
                    // Prefer stack scores (same 512 cap as the serial path) so
                    // every decode step does not heap-allocate 4 score buffers.
                    let mut local_stack = [0f32; STACK_SCORES];
                    let mut local_heap = if scores_len > STACK_SCORES {
                        vec![0f32; scores_len]
                    } else {
                        Vec::new()
                    };
                    let local: &mut [f32] = if scores_len <= STACK_SCORES {
                        &mut local_stack[..scores_len]
                    } else {
                        local_heap.as_mut_slice()
                    };
                    let ctx =
                        unsafe { std::slice::from_raw_parts_mut(ctx_ptr as *mut f32, N_HEADS * DH) };
                    for h in (0..N_HEADS).filter(|h| h % WORKERS == w) {
                        sdpa_one(h, local, ctx);
                    }
                });
            }
        });
    } else {
        for h in 0..N_HEADS {
            sdpa_one(h, scores, &mut ctx);
        }
    }

    sigmoid_gate_mul(&gate, &mut ctx);
    gemv(w.wo, HIDDEN, &ctx, out);
    Ok(())
}
