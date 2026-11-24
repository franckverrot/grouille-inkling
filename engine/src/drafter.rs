//! The DSpark drafter that ships with Inkling 2 Mini (`mtp.safetensors`).
//!
//! It proposes up to seven tokens after an anchor token, for the target to check in
//! one pass.  For every token the target has processed it reads four of the
//! target's hidden states (the stream entering visits 5, 23 and 41, and the final
//! normed state) and keeps its own K and V rows for them.  A block is the anchor's
//! embedding followed by seven mask embeddings.  Three layers attend to that
//! context and to the whole block, and a small recurrent head turns the mask
//! positions into tokens one after the other, each step seeing the token before it.
//!
//! Nothing here has to match another runtime bit for bit: a proposal the target
//! disagrees with is dropped, so the drafter only decides how often a round pays.

use anyhow::{ensure, Result};
use safetensors::tensor::SafeTensors;

use crate::head::{embed_tokens, LOGIT_SOFTCAP, UNPADDED_VOCAB, VOCAB};
use crate::metal_draft::{DraftHead, DraftMat, HEAD_ROWS};
use crate::metal_ops::MetalOps;
use crate::ops::HIDDEN;
use crate::prefill::HeadBufs;
use crate::weights::bf16_f32;

/// Target states read per token.
pub const TAPS: usize = 4;
/// The visits whose incoming stream is read.  The fourth state is the final norm.
pub const TAP_VISITS: [usize; 3] = [5, 23, 41];
/// Floats of target state per token.
pub const FEAT: usize = TAPS * HIDDEN;
/// Anchor plus proposals.
pub const BLOCK: usize = 8;

const LAYERS: usize = 3;
/// Layers 0 and 2 see the last `WINDOW` context rows, layer 1 all of them.
const LOCAL: [bool; LAYERS] = [true, false, true];
const WINDOW: usize = 1024;
const HEADS: usize = 24;
const KV_HEADS: usize = 8;
const GROUP: usize = HEADS / KV_HEADS;
const DH: usize = 128;
const Q_CH: usize = HEADS * DH;
const KV_CH: usize = KV_HEADS * DH;
const INTER: usize = 9216;
const RANK: usize = 256;
const EPS: f32 = 1e-6;
const ROPE_BASE: f64 = 1e6;

#[link(name = "Accelerate", kind = "framework")]
extern "C" {
    #[allow(clippy::too_many_arguments)]
    fn cblas_sgemm(
        order: i32,
        trans_a: i32,
        trans_b: i32,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a: *const f32,
        lda: i32,
        b: *const f32,
        ldb: i32,
        beta: f32,
        c: *mut f32,
        ldc: i32,
    );
    #[allow(clippy::too_many_arguments)]
    fn cblas_sgemv(
        order: i32,
        trans: i32,
        m: i32,
        n: i32,
        alpha: f32,
        a: *const f32,
        lda: i32,
        x: *const f32,
        incx: i32,
        beta: f32,
        y: *mut f32,
        incy: i32,
    );
}
const ROW_MAJOR: i32 = 101;
const NO_TRANS: i32 = 111;
const TRANS: i32 = 112;

/// `out[m][n] = x[m][k] * w[n][k]^T`: a linear layer without bias.
fn linear(x: &[f32], w: &[f32], m: usize, n: usize, k: usize, out: &mut [f32]) {
    assert!(x.len() >= m * k && w.len() == n * k && out.len() >= m * n);
    if m == 0 {
        return;
    }
    unsafe {
        cblas_sgemm(
            ROW_MAJOR,
            NO_TRANS,
            TRANS,
            m as i32,
            n as i32,
            k as i32,
            1.0,
            x.as_ptr(),
            k as i32,
            w.as_ptr(),
            k as i32,
            0.0,
            out.as_mut_ptr(),
            n as i32,
        );
    }
}

/// `y[m] = beta * y + a[m][n] * x[n]`.
fn gemv(a: &[f32], x: &[f32], m: usize, n: usize, beta: f32, y: &mut [f32]) {
    assert!(a.len() == m * n && x.len() == n && y.len() == m);
    unsafe {
        cblas_sgemv(
            ROW_MAJOR,
            NO_TRANS,
            m as i32,
            n as i32,
            1.0,
            a.as_ptr(),
            n as i32,
            x.as_ptr(),
            1,
            beta,
            y.as_mut_ptr(),
            1,
        );
    }
}

/// RMS norm with a learned scale, over each `w.len()`-wide piece of `x`.
fn rms_scale(x: &mut [f32], w: &[f32]) {
    for row in x.chunks_exact_mut(w.len()) {
        let ms = row.iter().map(|v| v * v).sum::<f32>() / row.len() as f32;
        let inv = 1.0 / (ms + EPS).sqrt();
        for (v, s) in row.iter_mut().zip(w) {
            *v *= inv * s;
        }
    }
}

struct Layer {
    input_norm: Vec<f32>,
    post_norm: Vec<f32>,
    ctx_norm: Vec<f32>,
    q_proj: DraftMat,
    k_proj: DraftMat,
    v_proj: DraftMat,
    k_ctx: DraftMat,
    v_ctx: DraftMat,
    o_proj: DraftMat,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    k_norm_ctx: Vec<f32>,
    gate: DraftMat,
    up: DraftMat,
    down: DraftMat,
}

/// One layer's K and V rows for the context: K normed and rotated, oldest first.
#[derive(Default)]
struct Context {
    k: Vec<f32>,
    v: Vec<f32>,
    rows: usize,
}

/// Proposals for one block.
pub struct Draft {
    pub tokens: Vec<i32>,
    /// The drafter's probability for each proposal, given the ones before it.
    pub probs: Vec<f32>,
}

pub struct Drafter {
    layers: Vec<Layer>,
    /// Softmaxed fusion weights: one row per layer, then one for the head.
    alpha: [[f32; TAPS]; LAYERS + 1],
    norm: Vec<f32>,
    mask_embed: Vec<f32>,
    head_norm: Vec<f32>,
    /// `[VOCAB][RANK]`: what the head learns from the token before.
    prev_embed: Vec<f32>,
    /// `[3 RANK][2 RANK]`: the recurrent weights on (state, previous token).
    rnn_state: Vec<f32>,
    /// `[3 RANK][2 HIDDEN]`: on (block state, fused target state).
    rnn_input: Vec<f32>,
    rnn_bias: Vec<f32>,
    /// The unembed folded onto the head's output, `[UNPADDED_VOCAB][RANK]` and a
    /// bias, on the GPU.  The last few vocabulary rows stay here: the GPU head
    /// covers a multiple of 8.
    head: DraftHead,
    tail_w: Vec<f32>,
    tail_b: Vec<f32>,
    freqs: [f32; DH / 2],
    context: Vec<Context>,
    /// Tokens of context so far, which is also the anchor's position.
    pos: usize,
    /// The fused target state of the newest context token, for the head.
    last_state: Vec<f32>,
}

impl Drafter {
    /// Load the drafter onto the GPU and fold the target's unembed onto its head.
    pub fn load(mtp: &SafeTensors, head: &HeadBufs, ops: &mut MetalOps) -> Result<Self> {
        let get = |name: &str, len: usize| -> Result<Vec<f32>> {
            let v = bf16_f32(mtp, name)?;
            ensure!(v.len() == len, "{name}: {} values, expected {len}", v.len());
            Ok(v)
        };
        let mut layers = Vec::with_capacity(LAYERS);
        for i in 0..LAYERS {
            let p = |s: &str| format!("layers.{i}.{s}.weight");
            let mut mat = |s: &str, out_dim: usize, in_dim: usize| -> Result<DraftMat> {
                ops.draft_mat(&get(&p(s), out_dim * in_dim)?, out_dim, in_dim)
            };
            layers.push(Layer {
                input_norm: get(&p("input_layernorm"), HIDDEN)?,
                post_norm: get(&p("post_attention_layernorm"), HIDDEN)?,
                ctx_norm: get(&p("dflare_norm"), HIDDEN)?,
                q_proj: mat("self_attn.q_proj", Q_CH, HIDDEN)?,
                k_proj: mat("self_attn.k_proj", KV_CH, HIDDEN)?,
                v_proj: mat("self_attn.v_proj", KV_CH, HIDDEN)?,
                k_ctx: mat("self_attn.k_proj_ctx", KV_CH, HIDDEN)?,
                v_ctx: mat("self_attn.v_proj_ctx", KV_CH, HIDDEN)?,
                o_proj: mat("self_attn.o_proj", HIDDEN, Q_CH)?,
                q_norm: get(&p("self_attn.q_norm"), DH)?,
                k_norm: get(&p("self_attn.k_norm"), DH)?,
                k_norm_ctx: get(&p("self_attn.k_norm_ctx"), DH)?,
                gate: mat("mlp.gate_proj", INTER, HIDDEN)?,
                up: mat("mlp.up_proj", INTER, HIDDEN)?,
                down: mat("mlp.down_proj", HIDDEN, INTER)?,
            });
        }
        let fuse = get("fuse_weight", (LAYERS + 1) * TAPS)?;
        let mut alpha = [[0f32; TAPS]; LAYERS + 1];
        for (row, w) in alpha.iter_mut().zip(fuse.chunks_exact(TAPS)) {
            let max = w.iter().cloned().fold(f32::MIN, f32::max);
            let sum: f32 = w.iter().map(|v| (v - max).exp()).sum();
            for (a, v) in row.iter_mut().zip(w) {
                *a = (v - max).exp() / sum;
            }
        }

        // rnn_proj takes [state, previous token, block state, target state].
        let rnn = get("markov_head.rnn_proj.weight", 3 * RANK * (2 * RANK + 2 * HIDDEN))?;
        let mut rnn_state = Vec::with_capacity(3 * RANK * 2 * RANK);
        let mut rnn_input = Vec::with_capacity(3 * RANK * 2 * HIDDEN);
        for row in rnn.chunks_exact(2 * RANK + 2 * HIDDEN) {
            rnn_state.extend_from_slice(&row[..2 * RANK]);
            rnn_input.extend_from_slice(&row[2 * RANK..]);
        }

        // logits = unembed * (hidden_out * latent + bias): fold the two products.
        let out_w = get("markov_head.hidden_out.weight", HIDDEN * RANK)?;
        let out_b = get("markov_head.hidden_out.bias", HIDDEN)?;
        let unembed = &head.unembed[..UNPADDED_VOCAB * HIDDEN];
        let mut fold_w = vec![0f32; UNPADDED_VOCAB * RANK];
        unsafe {
            cblas_sgemm(
                ROW_MAJOR,
                NO_TRANS,
                NO_TRANS,
                UNPADDED_VOCAB as i32,
                RANK as i32,
                HIDDEN as i32,
                1.0,
                unembed.as_ptr(),
                HIDDEN as i32,
                out_w.as_ptr(),
                RANK as i32,
                0.0,
                fold_w.as_mut_ptr(),
                RANK as i32,
            );
        }
        let mut fold_b = vec![0f32; UNPADDED_VOCAB];
        gemv(unembed, &out_b, UNPADDED_VOCAB, HIDDEN, 0.0, &mut fold_b);
        let head_gpu = ops.draft_head(&fold_w[..HEAD_ROWS * RANK], &fold_b[..HEAD_ROWS], RANK)?;

        let mut freqs = [0f32; DH / 2];
        for (i, f) in freqs.iter_mut().enumerate() {
            *f = (-(i as f64) * ROPE_BASE.ln() / (DH / 2) as f64).exp() as f32;
        }
        Ok(Self {
            layers,
            alpha,
            norm: get("norm.weight", HIDDEN)?,
            mask_embed: get("mask_embed", HIDDEN)?,
            head_norm: get("markov_head.dflare_norm.weight", HIDDEN)?,
            prev_embed: get("markov_head.markov_w1.weight", VOCAB * RANK)?,
            rnn_state,
            rnn_input,
            rnn_bias: get("markov_head.rnn_proj.bias", 3 * RANK)?,
            head: head_gpu,
            tail_w: fold_w[HEAD_ROWS * RANK..].to_vec(),
            tail_b: fold_b[HEAD_ROWS..].to_vec(),
            freqs,
            context: (0..LAYERS).map(|_| Context::default()).collect(),
            pos: 0,
            last_state: vec![0f32; HIDDEN],
        })
    }

    /// Forget the context: a new prompt.
    pub fn reset(&mut self) {
        for c in &mut self.context {
            *c = Context::default();
        }
        self.pos = 0;
        self.last_state.fill(0.0);
    }

    /// Tokens of context the drafter holds.
    pub fn context_len(&self) -> usize {
        self.pos
    }

    /// Rotate each head's `DH` values of a row to position `pos`.
    fn rotate(&self, row: &mut [f32], pos: usize) {
        let half = DH / 2;
        let mut sc = [(0f32, 0f32); DH / 2];
        for (v, f) in sc.iter_mut().zip(&self.freqs) {
            *v = (pos as f32 * f).sin_cos();
        }
        for head in row.chunks_exact_mut(DH) {
            for i in 0..half {
                let (s, c) = sc[i];
                let (a, b) = (head[i], head[half + i]);
                head[i] = a * c - b * s;
                head[half + i] = a * s + b * c;
            }
        }
    }

    /// The four target states of each row, fused with `alpha` and normed.
    fn fuse(feats: &[f32], n: usize, alpha: &[f32; TAPS], norm: &[f32], out: &mut [f32]) {
        assert!(feats.len() == n * FEAT && out.len() == n * HIDDEN);
        for (o, f) in out.chunks_exact_mut(HIDDEN).zip(feats.chunks_exact(FEAT)) {
            o.fill(0.0);
            for (tap, a) in f.chunks_exact(HIDDEN).zip(alpha) {
                for (x, v) in o.iter_mut().zip(tap) {
                    *x += v * a;
                }
            }
        }
        rms_scale(out, norm);
    }

    /// Add the target states of `n` more tokens the target has processed, oldest
    /// first: `feats` is `[n][FEAT]`, the four states of a token side by side.
    pub fn push_context(&mut self, feats: &[f32], n: usize, ops: &mut MetalOps) -> Result<()> {
        assert_eq!(feats.len(), n * FEAT);
        if n == 0 {
            return Ok(());
        }
        let mut fused = vec![0f32; n * HIDDEN];
        for l in 0..LAYERS {
            Self::fuse(feats, n, &self.alpha[l], &self.layers[l].ctx_norm, &mut fused);
            // A local layer never looks further back than its window.
            let skip = if LOCAL[l] { n.saturating_sub(WINDOW) } else { 0 };
            let m = n - skip;
            let x = &fused[skip * HIDDEN..];
            let mut kv = ops.draft_linear(x, m, &[&self.layers[l].k_ctx, &self.layers[l].v_ctx])?;
            let v = kv.pop().unwrap();
            let mut k = kv.pop().unwrap();
            rms_scale(&mut k, &self.layers[l].k_norm_ctx);
            for (i, row) in k.chunks_exact_mut(KV_CH).enumerate() {
                self.rotate(row, self.pos + skip + i);
            }
            let c = &mut self.context[l];
            c.k.extend_from_slice(&k);
            c.v.extend_from_slice(&v);
            c.rows += m;
            if LOCAL[l] && c.rows > WINDOW {
                let drop = (c.rows - WINDOW) * KV_CH;
                c.k.drain(..drop);
                c.v.drain(..drop);
                c.rows = WINDOW;
            }
        }
        Self::fuse(
            &feats[(n - 1) * FEAT..],
            1,
            &self.alpha[LAYERS],
            &self.head_norm,
            &mut self.last_state,
        );
        self.pos += n;
        Ok(())
    }

    /// One layer's attention for the block: every position sees the layer's
    /// context rows and the whole block.  `h` is the normed input, `[BLOCK][HIDDEN]`.
    fn attention(&self, l: usize, h: &[f32], ops: &mut MetalOps) -> Result<Vec<f32>> {
        let w = &self.layers[l];
        let ctx = &self.context[l];
        let mut qkv = ops.draft_linear(h, BLOCK, &[&w.q_proj, &w.k_proj, &w.v_proj])?;
        let bv = qkv.pop().unwrap();
        let mut bk = qkv.pop().unwrap();
        let mut q = qkv.pop().unwrap();
        rms_scale(&mut q, &w.q_norm);
        rms_scale(&mut bk, &w.k_norm);
        for i in 0..BLOCK {
            self.rotate(&mut q[i * Q_CH..(i + 1) * Q_CH], self.pos + i);
            self.rotate(&mut bk[i * KV_CH..(i + 1) * KV_CH], self.pos + i);
        }

        let s = ctx.rows;
        let nk = s + BLOCK;
        let nq = BLOCK * GROUP;
        let scale = (DH as f32).powf(-0.5);
        let mut qg = vec![0f32; nq * DH];
        let mut sc = vec![0f32; nq * nk];
        let mut og = vec![0f32; nq * DH];
        let mut attn = vec![0f32; BLOCK * Q_CH];
        for g in 0..KV_HEADS {
            // The three query heads that share KV head g, for every block row.
            for i in 0..BLOCK {
                for j in 0..GROUP {
                    let src = i * Q_CH + (g * GROUP + j) * DH;
                    qg[(i * GROUP + j) * DH..][..DH].copy_from_slice(&q[src..src + DH]);
                }
            }
            unsafe {
                if s > 0 {
                    cblas_sgemm(
                        ROW_MAJOR,
                        NO_TRANS,
                        TRANS,
                        nq as i32,
                        s as i32,
                        DH as i32,
                        scale,
                        qg.as_ptr(),
                        DH as i32,
                        ctx.k.as_ptr().add(g * DH),
                        KV_CH as i32,
                        0.0,
                        sc.as_mut_ptr(),
                        nk as i32,
                    );
                }
                cblas_sgemm(
                    ROW_MAJOR,
                    NO_TRANS,
                    TRANS,
                    nq as i32,
                    BLOCK as i32,
                    DH as i32,
                    scale,
                    qg.as_ptr(),
                    DH as i32,
                    bk.as_ptr().add(g * DH),
                    KV_CH as i32,
                    0.0,
                    sc.as_mut_ptr().add(s),
                    nk as i32,
                );
            }
            for row in sc.chunks_exact_mut(nk) {
                let max = row.iter().cloned().fold(f32::MIN, f32::max);
                let mut sum = 0f32;
                for v in row.iter_mut() {
                    *v = (*v - max).exp();
                    sum += *v;
                }
                let inv = 1.0 / sum;
                for v in row.iter_mut() {
                    *v *= inv;
                }
            }
            unsafe {
                if s > 0 {
                    cblas_sgemm(
                        ROW_MAJOR,
                        NO_TRANS,
                        NO_TRANS,
                        nq as i32,
                        DH as i32,
                        s as i32,
                        1.0,
                        sc.as_ptr(),
                        nk as i32,
                        ctx.v.as_ptr().add(g * DH),
                        KV_CH as i32,
                        0.0,
                        og.as_mut_ptr(),
                        DH as i32,
                    );
                }
                cblas_sgemm(
                    ROW_MAJOR,
                    NO_TRANS,
                    NO_TRANS,
                    nq as i32,
                    DH as i32,
                    BLOCK as i32,
                    1.0,
                    sc.as_ptr().add(s),
                    nk as i32,
                    bv.as_ptr().add(g * DH),
                    KV_CH as i32,
                    if s > 0 { 1.0 } else { 0.0 },
                    og.as_mut_ptr(),
                    DH as i32,
                );
            }
            for i in 0..BLOCK {
                for j in 0..GROUP {
                    let dst = i * Q_CH + (g * GROUP + j) * DH;
                    attn[dst..dst + DH].copy_from_slice(&og[(i * GROUP + j) * DH..][..DH]);
                }
            }
        }
        Ok(ops.draft_linear(&attn, BLOCK, &[&w.o_proj])?.pop().unwrap())
    }

    /// Propose `count` tokens after `anchor`, the newest token the target has
    /// chosen but not processed yet.  `ops` computes the vocabulary logits.
    pub fn draft(&self, anchor: i32, count: usize, head: &HeadBufs, ops: &mut MetalOps) -> Result<Draft> {
        ensure!(count < BLOCK, "a block holds {} proposals at most", BLOCK - 1);
        if count == 0 {
            return Ok(Draft { tokens: Vec::new(), probs: Vec::new() });
        }
        ensure!(self.pos > 0, "the drafter has no context");

        // The block: the anchor, then masks.  It is always the full trained length,
        // because every position attends to every other.
        let mut x = vec![0f32; BLOCK * HIDDEN];
        embed_tokens(&[anchor], &head.embed, &head.embed_norm, &mut x[..HIDDEN]);
        for row in x[HIDDEN..].chunks_exact_mut(HIDDEN) {
            row.copy_from_slice(&self.mask_embed);
        }
        let mut residual = vec![0f32; BLOCK * HIDDEN];
        let mut h = vec![0f32; BLOCK * HIDDEN];
        for l in 0..LAYERS {
            let w = &self.layers[l];
            for (r, v) in residual.iter_mut().zip(&x) {
                *r = if l == 0 { *v } else { *r + *v };
            }
            h.copy_from_slice(&residual);
            rms_scale(&mut h, &w.input_norm);
            let attn = self.attention(l, &h, ops)?;
            for (r, v) in residual.iter_mut().zip(&attn) {
                *r += *v;
            }
            h.copy_from_slice(&residual);
            rms_scale(&mut h, &w.post_norm);
            x = ops.draft_mlp(&h, BLOCK, &w.gate, &w.up, &w.down)?;
        }
        for (v, r) in x.iter_mut().zip(&residual) {
            *v += r;
        }
        rms_scale(&mut x, &self.norm);
        let states = &x[HIDDEN..(1 + count) * HIDDEN];

        // Vocabulary logits of the mask positions, before the head's correction.
        // The GPU keeps them; the rows it does not cover are done here.
        let unembed = &head.unembed[..UNPADDED_VOCAB * HIDDEN];
        ops.draft_base(&self.head, states, count, unembed)?;
        let tail = UNPADDED_VOCAB - HEAD_ROWS;
        let mut tail_base = vec![0f32; count * tail];
        linear(states, &unembed[HEAD_ROWS * HIDDEN..], count, tail, HIDDEN, &mut tail_base);

        // What each step of the head gets from its block state and the target state.
        let mut input = vec![0f32; count * 2 * HIDDEN];
        for (row, s) in input.chunks_exact_mut(2 * HIDDEN).zip(states.chunks_exact(HIDDEN)) {
            row[..HIDDEN].copy_from_slice(s);
            row[HIDDEN..].copy_from_slice(&self.last_state);
        }
        let mut from_input = vec![0f32; count * 3 * RANK];
        linear(&input, &self.rnn_input, count, 3 * RANK, 2 * HIDDEN, &mut from_input);

        let mut state = vec![0f32; RANK];
        let mut prev = anchor as usize;
        let mut tokens = Vec::with_capacity(count);
        let mut probs = Vec::with_capacity(count);
        let mut sp = vec![0f32; 2 * RANK];
        let mut raw = vec![0f32; 3 * RANK];
        for i in 0..count {
            ensure!(prev < VOCAB, "token {prev} outside the vocabulary");
            sp[..RANK].copy_from_slice(&state);
            sp[RANK..].copy_from_slice(&self.prev_embed[prev * RANK..(prev + 1) * RANK]);
            for ((r, f), b) in raw.iter_mut().zip(&from_input[i * 3 * RANK..]).zip(&self.rnn_bias) {
                *r = f + b;
            }
            gemv(&self.rnn_state, &sp, 3 * RANK, 2 * RANK, 1.0, &mut raw);
            for j in 0..RANK {
                let g = 1.0 / (1.0 + (-raw[j]).exp());
                state[j] = g * state[j] + (1.0 - g) * raw[RANK + j].tanh();
            }
            let latent: Vec<f32> = raw[2 * RANK..].iter().map(|v| v.tanh()).collect();
            // The best token under the capped logits the target also uses, and its
            // probability.
            let pick = ops.draft_pick(&self.head, i, &latent)?;
            let (mut best, mut top, mut sum) = (pick.row, pick.top, pick.sum_exp);
            for r in 0..tail {
                let w = &self.tail_w[r * RANK..(r + 1) * RANK];
                let dot: f32 = w.iter().zip(&latent).map(|(a, b)| a * b).sum();
                let l = tail_base[i * tail + r] + self.tail_b[r] + dot;
                let c = LOGIT_SOFTCAP * (l / LOGIT_SOFTCAP).tanh();
                sum += c.exp();
                if c > top {
                    top = c;
                    best = HEAD_ROWS + r;
                }
            }
            tokens.push(best as i32);
            probs.push(top.exp() / sum);
            prev = best;
        }
        Ok(Draft { tokens, probs })
    }
}
