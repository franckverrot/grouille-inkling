//! Prompt processing with the stream on the GPU.
//!
//! A chunk of prompt tokens goes through a layer visit in three command buffers:
//! everything up to the router logits, the experts, and the output conv.  In between
//! the host only picks each token's experts.  Afterwards it rolls the conv states and
//! files the chunk's K and V rows.
//!
//! A block of at most eight tokens, as the drafter proposes them, takes the same
//! kernels through all 48 visits in one command buffer, with the experts picked on
//! the GPU.  It commits nothing: each visit leaves its rows to be kept or dropped.

use anyhow::{Context, Result};
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLCommandBuffer, MTLCommandEncoder, MTLComputeCommandEncoder, MTLSize};

use crate::attn::{
    FullAttnWeights, SlidingAttnWeights, D_REL, DH, N_HEADS, N_KV, QK_CLIP, REL_CLIP,
};
use crate::cache::{GpuKv, VisitCache};
use crate::decoder::SlidingLayerWeights;
use crate::gpu::{copy_buffer, read_buffer, write_buffer, Buffer, FcValue};
use crate::layer5::FullLayerWeights;
use crate::loop_msg::{MessageControllerWeights, GATE_RANK, MSG_SLOTS};
use crate::metal_glue::ChunkAttn;
use crate::metal_moe::MetalMoe;
use crate::metal_ops::{
    MetalOps, ATTN_GATE, ATTN_ROUTER, ATTN_SINK, ATTN_WK, ATTN_WO, ATTN_WQ, ATTN_WR, ATTN_WV,
};
use crate::moe::{ExpertPack, TOP_K};
use crate::ops::{conv_state_roll, CONV_K, HIDDEN};
use crate::route::route_from_logits;

type Enc = ProtocolObject<dyn MTLComputeCommandEncoder>;

const Q_CH: usize = N_HEADS * DH;
const KV_CH: usize = N_KV * DH;
const N_EXPERTS: usize = 128;
/// Block-stack states a chunk has room for.  The schedule reaches four.
const STACK_CAP: usize = 8;
/// Message-bank states a chunk has room for.  The schedule reaches three.
const BANK_CAP: usize = 4;

/// A layer's attention weights as the prompt path reads them.
pub struct PromptAttn<'a> {
    pub layer: Option<usize>,
    pub full: bool,
    pub wq: &'a [f32],
    pub wk: &'a [f32],
    pub wv: &'a [f32],
    pub wr: &'a [f32],
    pub sink: &'a [f32],
    pub gate: &'a [f32],
    pub wo: &'a [f32],
    pub q_scale: &'a [f32],
    pub rel_scale: &'a [f32],
    pub rel_proj: &'a [f32],
    /// K and V short-conv taps: `[channel][4]` on a sliding layer, `[lag][channel]`
    /// on a full-attention one.
    pub k_taps: &'a [f32],
    pub v_taps: &'a [f32],
    /// Projections for the per-token taps of a full-attention layer.
    pub dyn_taps: Option<(&'a [f32], &'a [f32])>,
}

impl<'a> PromptAttn<'a> {
    pub fn sliding(w: &SlidingAttnWeights<'a>) -> Self {
        Self {
            layer: w.layer,
            full: false,
            wq: w.wq,
            wk: w.wk,
            wv: w.wv,
            wr: w.wr,
            sink: w.sink,
            gate: w.gate,
            wo: w.wo,
            q_scale: w.q_scale,
            rel_scale: w.rel_scale,
            rel_proj: w.rel_proj,
            k_taps: w.k_sconv,
            v_taps: w.v_sconv,
            dyn_taps: None,
        }
    }

    pub fn full(w: &FullAttnWeights<'a>) -> Self {
        Self {
            layer: w.layer,
            full: true,
            wq: w.wq,
            wk: w.wk,
            wv: w.wv,
            wr: w.wr,
            sink: w.sink,
            gate: w.gate,
            wo: w.wo,
            q_scale: w.q_scale,
            rel_scale: w.rel_scale,
            rel_proj: w.rel_proj,
            k_taps: w.k_static,
            v_taps: w.v_static,
            dyn_taps: Some((w.k_wproj, w.v_wproj)),
        }
    }
}

/// One layer's weights as a prompt visit reads them.
pub struct PromptLayer<'a> {
    pub attn: PromptAttn<'a>,
    /// Residual queries for the q, k and v inputs.  A sliding layer has one for all
    /// three, and layer 0 has none.
    pub attn_queries: [Option<&'a [f32]>; 3],
    pub attn_norms: [&'a [f32]; 3],
    /// Message controllers for attention, k, v and the MLP.
    pub ctrl: Option<[&'a MessageControllerWeights<'a>; 4]>,
    pub mlp_query: &'a [f32],
    pub mlp_norm: &'a [f32],
    pub attn_sconv: &'a [f32],
    pub mlp_sconv: &'a [f32],
    pub router_w: &'a [f32],
    pub router_b: &'a [f32],
    pub experts: &'a ExpertPack<'a>,
    pub is_block_boundary: bool,
}

fn controllers<'a>(
    a: &'a Option<MessageControllerWeights<'a>>,
    k: &'a Option<MessageControllerWeights<'a>>,
    v: &'a Option<MessageControllerWeights<'a>>,
    m: &'a Option<MessageControllerWeights<'a>>,
) -> Option<[&'a MessageControllerWeights<'a>; 4]> {
    Some([a.as_ref()?, k.as_ref()?, v.as_ref()?, m.as_ref()?])
}

impl<'a> PromptLayer<'a> {
    pub fn sliding(w: &'a SlidingLayerWeights<'a>) -> Self {
        Self {
            attn: PromptAttn::sliding(&w.attn),
            attn_queries: [w.attn_query; 3],
            attn_norms: [w.attn_norm; 3],
            ctrl: controllers(&w.ctrl_attn, &w.ctrl_k, &w.ctrl_v, &w.ctrl_mlp),
            mlp_query: w.mlp_query,
            mlp_norm: w.mlp_norm,
            attn_sconv: w.attn_sconv,
            mlp_sconv: w.mlp_sconv,
            router_w: w.router_w,
            router_b: w.router_b,
            experts: &w.experts,
            is_block_boundary: w.is_block_boundary,
        }
    }

    pub fn full(w: &'a FullLayerWeights<'a>) -> Self {
        let q = |i: usize| Some(&w.attn_queries[i * HIDDEN..(i + 1) * HIDDEN]);
        Self {
            attn: PromptAttn::full(&w.attn),
            attn_queries: [q(0), q(1), q(2)],
            attn_norms: [w.q_norm, w.k_norm, w.v_norm],
            ctrl: controllers(&w.ctrl_attn, &w.ctrl_k, &w.ctrl_v, &w.ctrl_mlp),
            mlp_query: w.mlp_query,
            mlp_norm: w.mlp_norm,
            attn_sconv: w.attn_sconv,
            mlp_sconv: w.mlp_sconv,
            router_w: w.router_w,
            router_b: w.router_b,
            experts: &w.experts,
            is_block_boundary: w.is_block_boundary,
        }
    }
}

/// A chunk of prompt tokens on the GPU: the stream, the block stack, and what the
/// core passes hand to each other.  Rows are padded to `tp`, a multiple of 32.
pub struct PromptGpu {
    pub tokens: usize,
    tp: usize,
    /// `[tp][H]`
    x: Buffer,
    /// `[STACK_CAP][tp][H]`
    stack: Buffer,
    n_states: usize,
    prelude_x: Buffer,
    prelude_n: usize,
    has_prelude: bool,
    /// `[BANK_CAP][tp][H]`
    bank: Buffer,
    bank_n: usize,
    /// `[tp][MSG_SLOTS][H]`, the messages of the visit being encoded.
    msgs: Buffer,
}

impl PromptGpu {
    /// Whether a core pass has left a message bank.
    pub fn has_bank(&self) -> bool {
        self.bank_n > 0
    }
}

/// What the attention chain leaves for its caller.
struct AttnChain {
    /// Raw K and V projections, whose last rows are the next conv state.
    k: Buffer,
    v: Buffer,
    /// Key buffers: `hist` ring rows, then the chunk's finished K and V rows.
    k_near: Buffer,
    v_near: Buffer,
    hist: usize,
    /// Where the chunk's finished K and V rows can be read once the visit has run:
    /// a buffer and a byte offset.
    k_done: (Buffer, usize),
    v_done: (Buffer, usize),
    /// The output projection, `[tp][H]`.
    out: Buffer,
}

/// What a visit's first half leaves for its second.
struct Front {
    chain: AttnChain,
    /// The stream after attention, the MLP input and the router logits.
    res: Buffer,
    moe_in: Buffer,
    logits: Buffer,
    /// The MLP conv state the visit came in with.
    mlp_state: Buffer,
}

/// Where a block visit leaves the rows a commit needs, each `[tokens][channels]`
/// from the byte offset.
pub(crate) struct BlockRows {
    k: (Buffer, usize),
    v: (Buffer, usize),
    k_in: Buffer,
    v_in: Buffer,
    attn_in: Buffer,
    mlp_in: Buffer,
}

impl MetalOps {
    fn pbuf(&mut self, slot: u16, floats: usize) -> Result<Buffer> {
        self.gemv_scratch_buf(5000 + slot, floats * 4)
    }

    /// A scratch buffer the CPU fills while recording, or whose rows are read back
    /// afterwards.  A block records all its visits before any of them runs, so each
    /// visit needs its own.
    fn pbuf_v(&mut self, slot: u16, floats: usize) -> Result<Buffer> {
        let Some(visit) = self.block_visit else { return self.pbuf(slot, floats) };
        let i = match slot {
            24 => 0,
            25 => 1,
            31 => 2,
            40 => 3,
            41 => 4,
            42 => 5,
            43 => 6,
            46 => 7,
            47 => 8,
            48 => 9,
            _ => unreachable!("slot {slot} is shared between visits"),
        };
        self.gemv_scratch_buf(5200 + visit as u16 * 16 + i, floats * 4)
    }

    /// Copy `floats` values from one buffer to another on the GPU.
    fn enc_copy(
        &mut self,
        enc: &Enc,
        src: (&Buffer, usize),
        dst: (&Buffer, usize),
        floats: usize,
    ) -> Result<()> {
        if floats == 0 {
            return Ok(());
        }
        let meta = self.pmeta(&[floats as u32])?;
        let pso = self.gemv.pso("k_copy_f32", &[])?;
        enc.setComputePipelineState(&pso);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(src.0), src.1, 0);
            enc.setBuffer_offset_atIndex(Some(dst.0), dst.1, 1);
            enc.setBuffer_offset_atIndex(Some(&meta), 0, 2);
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize { width: floats.div_ceil(1024), height: 1, depth: 1 },
            MTLSize { width: 1024, height: 1, depth: 1 },
        );
        Ok(())
    }

    /// A meta buffer for one dispatch.  Every dispatch of a command buffer is encoded
    /// before any of them runs, so each gets its own.
    pub(crate) fn pmeta(&mut self, vals: &[u32]) -> Result<Buffer> {
        let b = self.gemv_scratch_buf(6000 + self.prompt_meta_n, 32)?;
        self.prompt_meta_n += 1;
        write_buffer(&b, 0, vals);
        Ok(b)
    }

    /// One of the `k_tok_*` kernels: buffers in binding order, one simdgroup per
    /// grid cell.
    pub(crate) fn enc_tok(
        &mut self,
        enc: &Enc,
        name: &str,
        fc: &[(usize, FcValue)],
        bufs: &[(&Buffer, usize)],
        width: usize,
        height: usize,
    ) -> Result<()> {
        let pso = self.gemv.pso(name, fc)?;
        enc.setComputePipelineState(&pso);
        for (i, (b, off)) in bufs.iter().enumerate() {
            unsafe { enc.setBuffer_offset_atIndex(Some(b), *off, i) };
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize { width, height, depth: 1 },
            MTLSize { width: 32, height: 1, depth: 1 },
        );
        Ok(())
    }

    /// `y[T][out] = x[T][in] * w[out][in]^T` with f16 weights.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn enc_gemm(
        &mut self,
        enc: &Enc,
        x: &Buffer,
        w: &Buffer,
        y: &Buffer,
        tokens: usize,
        out_dim: usize,
        in_dim: usize,
    ) -> Result<()> {
        let fc = [
            (0, FcValue::U32(8)),
            (1, FcValue::U32(out_dim as u32)),
            (2, FcValue::U32(in_dim as u32)),
            (3, FcValue::F32(0.0)),
        ];
        // A small block takes the 8-token tile: a quarter of the arithmetic.
        let (name, tile) = if tokens <= 8 { ("k_gemm_f16w_mma8", 8) } else { ("k_gemm_f16w_mma", 32) };
        self.enc_tok(enc, name, &fc, &[(x, 0), (w, 0), (y, 0)], out_dim / 8, tokens.div_ceil(tile))
    }

    /// Softmax mix over the block stack and `extras`, into `out`.  With `norm_w` the
    /// result is RMS-normed.  Without a query the single extra passes through.
    #[allow(clippy::too_many_arguments)]
    fn enc_mix(
        &mut self,
        enc: &Enc,
        st: &PromptGpu,
        extras: (&Buffer, Option<&Buffer>),
        query: Option<&[f32]>,
        norm_w: Option<&Buffer>,
        out: &Buffer,
    ) -> Result<()> {
        let (ns, ne) = match query {
            Some(_) => (st.n_states, 1 + extras.1.is_some() as usize),
            None => (0, 1),
        };
        let meta = self.pmeta(&[
            ns as u32,
            ne as u32,
            st.tp as u32,
            HIDDEN as u32,
            norm_w.is_some() as u32,
            query.is_none() as u32,
        ])?;
        let b_query = match query {
            Some(q) => self.weight_buf(q)?,
            None => extras.0.clone(),
        };
        let bufs = [
            (&st.stack, 0),
            (extras.0, 0),
            (extras.1.unwrap_or(extras.0), 0),
            (&b_query, 0),
            (norm_w.unwrap_or(extras.0), 0),
            (out, 0),
            (&meta, 0),
        ];
        if st.tokens > 8 {
            return self.enc_tok(enc, "k_tok_mix", &[], &bufs, st.tokens, 1);
        }
        // A few tokens: spread each over eight simdgroups, for the latency.
        let pso = self.gemv.pso("k_tok_mix_wide", &[])?;
        enc.setComputePipelineState(&pso);
        for (i, (b, off)) in bufs.iter().enumerate() {
            unsafe { enc.setBuffer_offset_atIndex(Some(b), *off, i) };
        }
        unsafe { enc.setThreadgroupMemoryLength_atIndex(8 * 20 * 4, 0) };
        enc.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize { width: st.tokens, height: 1, depth: 1 },
            MTLSize { width: 256, height: 1, depth: 1 },
        );
        Ok(())
    }

    /// Message controller on `x` with the message in `slot`, then the RMS norm.
    #[allow(clippy::too_many_arguments)]
    fn enc_ctrl(
        &mut self,
        enc: &Enc,
        st: &PromptGpu,
        x: &Buffer,
        slot: usize,
        c: &MessageControllerWeights<'_>,
        norm_w: &Buffer,
        out: &Buffer,
    ) -> Result<()> {
        let (t, tp) = (st.tokens, st.tp);
        let b_xm = self.pbuf(14, tp * 2 * HIDDEN)?;
        let b_hid = self.pbuf(15, tp * GATE_RANK)?;
        let b_f = self.pbuf(16, tp * HIDDEN)?;
        let b_g = self.pbuf(17, tp * HIDDEN)?;
        let (down, forget, gain) = (
            self.weight_buf_f16(c.down)?,
            self.weight_buf_f16(c.forget)?,
            self.weight_buf_f16(c.gain)?,
        );
        let meta = self.pmeta(&[(MSG_SLOTS * HIDDEN) as u32, HIDDEN as u32])?;
        let msg_off = slot * HIDDEN * 4;
        self.enc_tok(
            enc,
            "k_tok_ctrl_prep",
            &[],
            &[(x, 0), (&st.msgs, msg_off), (&b_xm, 0), (&meta, 0)],
            t,
            1,
        )?;
        self.enc_gemm(enc, &b_xm, &down, &b_hid, t, GATE_RANK, 2 * HIDDEN)?;
        self.enc_gemm(enc, &b_hid, &forget, &b_f, t, HIDDEN, GATE_RANK)?;
        self.enc_gemm(enc, &b_hid, &gain, &b_g, t, HIDDEN, GATE_RANK)?;
        self.enc_tok(
            enc,
            "k_tok_ctrl_apply",
            &[],
            &[
                (x, 0),
                (&st.msgs, msg_off),
                (&b_f, 0),
                (&b_g, 0),
                (norm_w, 0),
                (out, 0),
                (&meta, 0),
            ],
            t,
            1,
        )
    }

    /// Attention of a chunk from its normed q, k and v inputs: projections, K/V conv
    /// and clips, attention over the history and the chunk itself, output projection.
    /// `states` are the K and V conv states, `history` what the visit already holds.
    fn enc_attn_chain(
        &mut self,
        enc: &Enc,
        w: &PromptAttn<'_>,
        t: usize,
        inputs: (&Buffer, &Buffer, &Buffer),
        states: (&[f32], &[f32]),
        history: Option<&GpuKv>,
    ) -> Result<AttnChain> {
        let tp = t.next_multiple_of(32);
        let (qin, kin, vin) = inputs;
        let pos0 = history.map_or(0, |h| h.len);
        let hist = history.map_or(0, |h| h.ring_rows());
        let far = history
            .filter(|_| w.full)
            .map_or(Vec::new(), |h| h.archive_segments(h.archive_rows()));
        let b_q = self.pbuf(20, tp * Q_CH)?;
        let b_gate = self.pbuf(21, tp * Q_CH)?;
        let b_r = self.pbuf(22, tp * N_HEADS * D_REL)?;
        let b_sink = self.pbuf(23, tp * N_HEADS)?;
        let b_k = self.pbuf_v(24, tp * KV_CH)?;
        let b_v = self.pbuf_v(25, tp * KV_CH)?;
        let b_dk = self.pbuf(26, tp * N_KV * CONV_K)?;
        let b_dv = self.pbuf(27, tp * N_KV * CONV_K)?;
        // Keys of the chunk's first attention block: the ring rows, then its own.
        let b_kn = self.pbuf(28, (hist + tp) * KV_CH)?;
        let b_vn = self.pbuf(29, (hist + tp) * KV_CH)?;
        let b_ctx = self.pbuf(30, tp * Q_CH)?;
        let b_a = self.pbuf_v(31, tp * HIDDEN)?;
        let b_kst = self.pbuf_v(40, states.0.len())?;
        let b_vst = self.pbuf_v(41, states.1.len())?;
        write_buffer(&b_kst, 0, states.0);
        write_buffer(&b_vst, 0, states.1);
        let b_thr_q = self.pbuf(44, 4)?;
        let b_thr_r = self.pbuf(45, 4)?;
        write_buffer(&b_thr_q, 0, &[QK_CLIP]);
        write_buffer(&b_thr_r, 0, &[REL_CLIP]);
        if let Some(h) = history.filter(|_| hist > 0) {
            if self.block_visit.is_some() {
                // Recorded, not run yet: the GPU lays the ring out oldest first.
                let first = (h.len - hist) % crate::cache::RING;
                let head = (crate::cache::RING - first).min(hist);
                for (ring, near) in [(&h.k_ring, &b_kn), (&h.v_ring, &b_vn)] {
                    self.enc_copy(enc, (ring, first * KV_CH * 4), (near, 0), head * KV_CH)?;
                    self.enc_copy(enc, (ring, 0), (near, head * KV_CH * 4), (hist - head) * KV_CH)?;
                }
            } else {
                h.unroll_ring(&b_kn, &b_vn);
            }
        }

        let w_q = self.attn_f16_or_convert(w.layer, ATTN_WQ, w.wq)?;
        let w_k = self.attn_f16_or_convert(w.layer, ATTN_WK, w.wk)?;
        let w_v = self.attn_f16_or_convert(w.layer, ATTN_WV, w.wv)?;
        let w_r = self.attn_f16_or_convert(w.layer, ATTN_WR, w.wr)?;
        let w_sink = self.attn_f16_or_convert(w.layer, ATTN_SINK, w.sink)?;
        let w_gate = self.attn_f16_or_convert(w.layer, ATTN_GATE, w.gate)?;
        let w_o = self.attn_f16_or_convert(w.layer, ATTN_WO, w.wo)?;
        let rel = self.attn_f16_or_convert(w.layer, crate::metal_ops::ATTN_RELPROJ, w.rel_proj)?;
        let q_scale = self.weight_buf(w.q_scale)?;
        let rel_scale = self.weight_buf(w.rel_scale)?;
        let k_taps = self.weight_buf(w.k_taps)?;
        let v_taps = self.weight_buf(w.v_taps)?;

        // Projections.
        self.enc_gemm(enc, qin, &w_q, &b_q, t, Q_CH, HIDDEN)?;
        self.enc_gemm(enc, qin, &w_gate, &b_gate, t, Q_CH, HIDDEN)?;
        self.enc_gemm(enc, qin, &w_r, &b_r, t, N_HEADS * D_REL, HIDDEN)?;
        self.enc_gemm(enc, qin, &w_sink, &b_sink, t, N_HEADS, HIDDEN)?;
        self.enc_gemm(enc, kin, &w_k, &b_k, t, KV_CH, HIDDEN)?;
        self.enc_gemm(enc, vin, &w_v, &b_v, t, KV_CH, HIDDEN)?;
        if let Some((kp, vp)) = w.dyn_taps {
            let (kp, vp) = (self.weight_buf_f16(kp)?, self.weight_buf_f16(vp)?);
            self.enc_gemm(enc, kin, &kp, &b_dk, t, N_KV * CONV_K, HIDDEN)?;
            self.enc_gemm(enc, vin, &vp, &b_dv, t, N_KV * CONV_K, HIDDEN)?;
        }

        // K and V: conv, then clip or norm, written after the ring rows.
        let geom = [(5, FcValue::U32(N_KV as u32)), (6, FcValue::U32(DH as u32))];
        let dynamic = w.dyn_taps.is_some() as u32;
        let near_off = hist * KV_CH * 4;
        for (mode, x, taps, state, dyn_taps, out) in [
            (0u32, &b_k, &k_taps, &b_kst, &b_dk, &b_kn),
            (1u32, &b_v, &v_taps, &b_vst, &b_dv, &b_vn),
        ] {
            let meta = self.pmeta(&[mode, dynamic])?;
            self.enc_tok(
                enc,
                "k_tok_kv_conv",
                &geom,
                &[(x, 0), (taps, 0), (state, 0), (dyn_taps, 0), (out, near_off), (&meta, 0)],
                t,
                N_KV,
            )?;
        }
        for (x, scale, n, thr) in [
            (&b_q, &q_scale, DH, &b_thr_q),
            (&b_r, &rel_scale, D_REL, &b_thr_r),
        ] {
            let meta = self.pmeta(&[N_HEADS as u32, n as u32])?;
            self.enc_tok(
                enc,
                "k_tok_l2_clip",
                &[],
                &[(x, 0), (scale, 0), (&meta, 0), (thr, 0)],
                t,
                N_HEADS,
            )?;
        }

        self.encode_attn_chunk(
            enc,
            &ChunkAttn {
                q: &b_q,
                k_near: &b_kn,
                v_near: &b_vn,
                r: &b_r,
                rel_proj: &rel,
                sinks: &b_sink,
                gate: &b_gate,
                ctx: &b_ctx,
            },
            t,
            pos0,
            hist,
            &far,
            w.full,
        )?;
        self.enc_gemm(enc, &b_ctx, &w_o, &b_a, t, HIDDEN, Q_CH)?;
        // The next visit of a block reuses the key buffers: put its rows aside.
        let (k_done, v_done) = if self.block_visit.is_some() {
            let (hk, hv) = (self.pbuf_v(46, tp * KV_CH)?, self.pbuf_v(47, tp * KV_CH)?);
            self.enc_copy(enc, (&b_kn, near_off), (&hk, 0), t * KV_CH)?;
            self.enc_copy(enc, (&b_vn, near_off), (&hv, 0), t * KV_CH)?;
            ((hk, 0), (hv, 0))
        } else {
            ((b_kn.clone(), near_off), (b_vn.clone(), near_off))
        };
        Ok(AttnChain { k: b_k, v: b_v, k_near: b_kn, v_near: b_vn, hist, k_done, v_done, out: b_a })
    }

    /// A layer's attention over host rows with no history: `q_in`, `k_in` and `v_in`
    /// are the normed inputs, `[tokens][H]`, and the result is the output projection.
    /// The parity tests compare it with the host reference.
    pub fn prompt_attention_host(
        &mut self,
        w: &PromptAttn<'_>,
        q_in: &[f32],
        k_in: &[f32],
        v_in: &[f32],
        tokens: usize,
    ) -> Result<Vec<f32>> {
        let tp = tokens.next_multiple_of(32);
        let b_in = [
            self.pbuf(10, tp * HIDDEN)?,
            self.pbuf(11, tp * HIDDEN)?,
            self.pbuf(12, tp * HIDDEN)?,
        ];
        self.flush_pending()?;
        for (b, rows) in b_in.iter().zip([q_in, k_in, v_in]) {
            assert_eq!(rows.len(), tokens * HIDDEN);
            write_buffer(b, 0, rows);
        }
        let zeros = vec![0f32; KV_CH * (CONV_K - 1)];
        self.prompt_meta_n = 0;
        let cb = self.gemv.command_buffer()?;
        let chain = {
            let enc = cb.computeCommandEncoder().context("encoder")?;
            let chain = self.enc_attn_chain(
                &enc,
                w,
                tokens,
                (&b_in[0], &b_in[1], &b_in[2]),
                (&zeros, &zeros),
                None,
            )?;
            enc.endEncoding();
            chain
        };
        cb.commit();
        cb.waitUntilCompleted();
        let mut out = vec![0f32; tokens * HIDDEN];
        read_buffer(&chain.out, 0, &mut out);
        Ok(out)
    }

    /// One layer over host rows with a fresh history: returns the output stream and
    /// every token's block stack.  The parity tests compare it with the host
    /// reference.
    pub fn prompt_layer_host(
        &mut self,
        moe: &mut MetalMoe,
        w: &PromptLayer<'_>,
        x: &[f32],
        tokens: usize,
        blocks: &[Vec<f32>],
    ) -> Result<(Vec<f32>, Vec<Vec<f32>>)> {
        let mut st = self.prompt_begin(x, tokens)?;
        self.prompt_load_stack(&mut st, blocks)?;
        let mut cache = if w.attn.full { VisitCache::full(tokens) } else { VisitCache::sliding() };
        self.prompt_visit(moe, &mut st, w, &mut cache, None)?;
        let mut out = vec![0f32; tokens * HIDDEN];
        self.prompt_read_x(&st, &mut out);
        Ok((out, self.prompt_read_stack(&st)))
    }

    /// Replace the chunk's block stack with per-token stacks, `blocks[t]` = `[n][H]`.
    pub fn prompt_load_stack(&mut self, st: &mut PromptGpu, blocks: &[Vec<f32>]) -> Result<()> {
        assert_eq!(blocks.len(), st.tokens);
        let n = blocks[0].len() / HIDDEN;
        anyhow::ensure!(n <= STACK_CAP, "block stack deeper than {STACK_CAP}");
        for (t, stack) in blocks.iter().enumerate() {
            assert_eq!(stack.len(), n * HIDDEN, "every token has the same stack depth");
            for (s, row) in stack.chunks(HIDDEN).enumerate() {
                write_buffer(&st.stack, (s * st.tp + t) * HIDDEN * 4, row);
            }
        }
        st.n_states = n;
        Ok(())
    }

    /// Every token's block stack.
    pub fn prompt_read_stack(&self, st: &PromptGpu) -> Vec<Vec<f32>> {
        Self::read_states(&st.stack, st, st.n_states)
    }

    /// Set what the core passes restore on re-entry: the stream and the block stacks
    /// the first core visit started from.  The stacks become the bottom of the
    /// chunk's stack.
    pub fn prompt_load_prelude(&mut self, st: &mut PromptGpu, x: &[f32], blocks: &[Vec<f32>]) {
        write_buffer(&st.prelude_x, 0, &x[..st.tokens * HIDDEN]);
        let n = blocks[0].len() / HIDDEN;
        for (t, stack) in blocks.iter().enumerate() {
            for (s, row) in stack.chunks(HIDDEN).enumerate() {
                write_buffer(&st.stack, (s * st.tp + t) * HIDDEN * 4, row);
            }
        }
        st.prelude_n = n;
        st.has_prelude = true;
    }

    /// The stream and block stacks kept for core re-entry, once there are any.
    pub fn prompt_read_prelude(&self, st: &PromptGpu) -> Option<(Vec<f32>, Vec<Vec<f32>>)> {
        if !st.has_prelude {
            return None;
        }
        let mut x = vec![0f32; st.tokens * HIDDEN];
        read_buffer(&st.prelude_x, 0, &mut x);
        Some((x, Self::read_states(&st.stack, st, st.prelude_n)))
    }

    /// Replace the message bank with per-token banks, `banks[t]` = `[n][H]`.
    pub fn prompt_load_bank(&mut self, st: &mut PromptGpu, banks: &[Vec<f32>]) -> Result<()> {
        assert_eq!(banks.len(), st.tokens);
        let n = banks[0].len() / HIDDEN;
        anyhow::ensure!(n <= BANK_CAP, "message bank deeper than {BANK_CAP}");
        for (t, bank) in banks.iter().enumerate() {
            for (s, row) in bank.chunks(HIDDEN).enumerate() {
                write_buffer(&st.bank, (s * st.tp + t) * HIDDEN * 4, row);
            }
        }
        st.bank_n = n;
        Ok(())
    }

    /// Every token's message bank.
    pub fn prompt_read_bank(&self, st: &PromptGpu) -> Vec<Vec<f32>> {
        Self::read_states(&st.bank, st, st.bank_n)
    }

    /// `n` state-major rows per token, read back token by token.
    fn read_states(buf: &Buffer, st: &PromptGpu, n: usize) -> Vec<Vec<f32>> {
        (0..st.tokens)
            .map(|t| {
                let mut rows = vec![0f32; n * HIDDEN];
                for (s, row) in rows.chunks_mut(HIDDEN).enumerate() {
                    read_buffer(buf, (s * st.tp + t) * HIDDEN * 4, row);
                }
                rows
            })
            .collect()
    }

    /// Put a chunk's embedded tokens on the GPU with an empty block stack.
    pub fn prompt_begin(&mut self, x: &[f32], tokens: usize) -> Result<PromptGpu> {
        assert_eq!(x.len(), tokens * HIDDEN);
        let tp = tokens.next_multiple_of(32);
        let st = PromptGpu {
            tokens,
            tp,
            x: self.pbuf(0, tp * HIDDEN)?,
            stack: self.pbuf(1, STACK_CAP * tp * HIDDEN)?,
            n_states: 0,
            prelude_x: self.pbuf(2, tp * HIDDEN)?,
            prelude_n: 0,
            has_prelude: false,
            bank: self.pbuf(3, BANK_CAP * tp * HIDDEN)?,
            bank_n: 0,
            msgs: self.pbuf(4, tp * MSG_SLOTS * HIDDEN)?,
        };
        self.flush_pending()?;
        write_buffer(&st.x, 0, x);
        Ok(st)
    }

    /// First core visit: keep the stream and the stack depth for the later passes.
    pub fn prompt_stash_prelude(&mut self, st: &mut PromptGpu) {
        copy_buffer(&st.x, 0, &st.prelude_x, 0, st.tokens * HIDDEN * 4);
        st.prelude_n = st.n_states;
        st.has_prelude = true;
    }

    /// Core re-entry: build the message bank from the pass that just ended, then go
    /// back to the prelude's stream and stack.
    pub fn prompt_reenter_core(&mut self, st: &mut PromptGpu) -> Result<()> {
        // The bank is the core blocks this pass completed after its first, plus the
        // stream it ended on, each RMS-normed.
        let skip = st.prelude_n + 1;
        let n_core = st.n_states.saturating_sub(skip);
        anyhow::ensure!(n_core < BANK_CAP, "message bank deeper than {BANK_CAP}");
        let row = st.tp * HIDDEN * 4;
        self.prompt_meta_n = 0;
        let meta = self.pmeta(&[HIDDEN as u32])?;
        let cb = self.gemv.command_buffer()?;
        {
            let enc = cb.computeCommandEncoder().context("encoder")?;
            for i in 0..n_core {
                self.enc_tok(
                    &enc,
                    "k_tok_rms",
                    &[],
                    &[(&st.stack, (skip + i) * row), (&st.bank, i * row), (&meta, 0)],
                    st.tokens,
                    1,
                )?;
            }
            self.enc_tok(
                &enc,
                "k_tok_rms",
                &[],
                &[(&st.x, 0), (&st.bank, n_core * row), (&meta, 0)],
                st.tokens,
                1,
            )?;
            enc.endEncoding();
        }
        cb.commit();
        cb.waitUntilCompleted();
        st.bank_n = n_core + 1;
        copy_buffer(&st.prelude_x, 0, &st.x, 0, st.tokens * HIDDEN * 4);
        st.n_states = st.prelude_n;
        Ok(())
    }

    /// The chunk's stream, token-major.
    pub fn prompt_read_x(&self, st: &PromptGpu, out: &mut [f32]) {
        read_buffer(&st.x, 0, &mut out[..st.tokens * HIDDEN]);
    }

    /// Stream and block stack of the chunk's last token.
    pub fn prompt_last_token(&self, st: &PromptGpu) -> (Vec<f32>, Vec<f32>) {
        let t = st.tokens - 1;
        let mut x = vec![0f32; HIDDEN];
        read_buffer(&st.x, t * HIDDEN * 4, &mut x);
        let mut blocks = vec![0f32; st.n_states * HIDDEN];
        for (s, row) in blocks.chunks_mut(HIDDEN).enumerate() {
            read_buffer(&st.stack, (s * st.tp + t) * HIDDEN * 4, row);
        }
        (x, blocks)
    }

    /// A visit up to the router logits, recorded on `enc`: the attention inputs, the
    /// attention itself and its residual, and the MLP input.
    fn enc_front(
        &mut self,
        enc: &Enc,
        st: &PromptGpu,
        w: &PromptLayer<'_>,
        cache: &mut VisitCache,
        msg_queries: Option<&[f32]>,
    ) -> Result<Front> {
        let (t, tp) = (st.tokens, st.tp);
        let ctrl = match msg_queries {
            Some(_) => Some(w.ctrl.as_ref().context("layer has no message controllers")?),
            None => None,
        };
        let b_in = [
            self.pbuf(10, tp * HIDDEN)?,
            self.pbuf(11, tp * HIDDEN)?,
            self.pbuf(12, tp * HIDDEN)?,
        ];
        let b_mix = self.pbuf(13, tp * HIDDEN)?;
        let b_res = self.pbuf(32, tp * HIDDEN)?;
        let b_moein = self.pbuf(33, tp * HIDDEN)?;
        let b_logits = self.pbuf(34, tp * N_EXPERTS)?;

        // Conv states: the three inputs before this chunk, per channel.
        let hist_len = CONV_K - 1;
        cache.k_sconv.resize(KV_CH * hist_len, 0.0);
        cache.v_sconv.resize(KV_CH * hist_len, 0.0);
        cache.attn_sconv.resize(HIDDEN * hist_len, 0.0);
        cache.mlp_sconv.resize(HIDDEN * hist_len, 0.0);
        let b_ast = self.pbuf_v(42, HIDDEN * hist_len)?;
        let b_mst = self.pbuf_v(43, HIDDEN * hist_len)?;
        write_buffer(&b_ast, 0, &cache.attn_sconv);
        write_buffer(&b_mst, 0, &cache.mlp_sconv);

        let w_router = self.attn_f16_or_convert(w.attn.layer, ATTN_ROUTER, w.router_w)?;
        let attn_taps = self.weight_buf(w.attn_sconv)?;
        let mlp_norm = self.weight_buf(w.mlp_norm)?;

        if let Some(q) = msg_queries {
            let b_queries = self.weight_buf(q)?;
            let meta = self.pmeta(&[st.bank_n as u32, tp as u32, HIDDEN as u32])?;
            self.enc_tok(
                enc,
                "k_tok_gather",
                &[],
                &[(&st.bank, 0), (&b_queries, 0), (&st.msgs, 0), (&meta, 0)],
                t,
                1,
            )?;
        }

        // Attention inputs.  A sliding layer without controllers has one, shared
        // by q, k and v.
        let streams = if w.attn.full || ctrl.is_some() { 3 } else { 1 };
        for i in 0..streams {
            let norm_w = self.weight_buf(w.attn_norms[i])?;
            let query = w.attn_queries[i];
            match ctrl {
                None => self.enc_mix(enc, st, (&st.x, None), query, Some(&norm_w), &b_in[i])?,
                Some(c) => {
                    // A sliding layer mixes once for its three controllers.
                    let mixed = match query {
                        Some(_) => {
                            if w.attn.full || i == 0 {
                                self.enc_mix(enc, st, (&st.x, None), query, None, &b_mix)?;
                            }
                            &b_mix
                        }
                        None => &st.x,
                    };
                    self.enc_ctrl(enc, st, mixed, i, c[i], &norm_w, &b_in[i])?;
                }
            }
        }
        let (qin, kin, vin) = if streams == 3 {
            (&b_in[0], &b_in[1], &b_in[2])
        } else {
            (&b_in[0], &b_in[0], &b_in[0])
        };

        let chain = self.enc_attn_chain(
            enc,
            &w.attn,
            t,
            (qin, kin, vin),
            (&cache.k_sconv, &cache.v_sconv),
            cache.gpu.as_ref(),
        )?;

        // Attention residual.  A layer that opens a block does not add the stream.
        let meta = self.pmeta(&[HIDDEN as u32, !w.is_block_boundary as u32])?;
        self.enc_tok(
            enc,
            "k_tok_sconv_add",
            &[],
            &[(&chain.out, 0), (&attn_taps, 0), (&b_ast, 0), (&st.x, 0), (&b_res, 0), (&meta, 0)],
            t,
            1,
        )?;

        // MLP input: the mix also sees the stream on a layer that opens a block.
        let extras = if w.is_block_boundary { (&st.x, Some(&b_res)) } else { (&b_res, None) };
        match ctrl {
            None => self.enc_mix(enc, st, extras, Some(w.mlp_query), Some(&mlp_norm), &b_moein)?,
            Some(c) => {
                self.enc_mix(enc, st, extras, Some(w.mlp_query), None, &b_mix)?;
                self.enc_ctrl(enc, st, &b_mix, 3, c[3], &mlp_norm, &b_moein)?;
            }
        }
        self.enc_gemm(enc, &b_moein, &w_router, &b_logits, t, N_EXPERTS, HIDDEN)?;
        Ok(Front { chain, res: b_res, moe_in: b_moein, logits: b_logits, mlp_state: b_mst })
    }

    /// The output conv and residual: `x = res + conv(moe)`.
    fn enc_tail(
        &mut self,
        enc: &Enc,
        st: &PromptGpu,
        w: &PromptLayer<'_>,
        front: &Front,
        b_moe: &Buffer,
    ) -> Result<()> {
        let mlp_taps = self.weight_buf(w.mlp_sconv)?;
        let meta = self.pmeta(&[HIDDEN as u32, 1])?;
        self.enc_tok(
            enc,
            "k_tok_sconv_add",
            &[],
            &[(b_moe, 0), (&mlp_taps, 0), (&front.mlp_state, 0), (&front.res, 0), (&st.x, 0), (&meta, 0)],
            st.tokens,
            1,
        )
    }

    /// One layer visit over the chunk.  `msg_queries` are the layer's four message
    /// queries on core passes two and three.
    pub fn prompt_visit(
        &mut self,
        moe: &mut MetalMoe,
        st: &mut PromptGpu,
        w: &PromptLayer<'_>,
        cache: &mut VisitCache,
        msg_queries: Option<&[f32]>,
    ) -> Result<()> {
        let (t, tp) = (st.tokens, st.tp);
        let hb = HIDDEN * 4;
        let mut lap = crate::flags::profile_prefill().then(std::time::Instant::now);
        let continuing = cache.tokens_seen() > 0;
        cache.gpu_kv(&self.gemv)?;
        self.flush_pending()?;

        // ---- Everything up to the router logits.
        self.prompt_meta_n = 0;
        let cb = self.gemv.command_buffer()?;
        let front = {
            let enc = cb.computeCommandEncoder().context("encoder")?;
            let front = self.enc_front(&enc, st, w, cache, msg_queries)?;
            enc.endEncoding();
            front
        };
        cb.commit();
        cb.waitUntilCompleted();
        crate::flags::prefill_lap(&mut lap, 0);

        // ---- The host picks each token's experts.
        let mut logits = vec![0f32; t * N_EXPERTS];
        read_buffer(&front.logits, 0, &mut logits);
        let mut indices = vec![0i32; t * TOP_K];
        let mut weights = vec![0f32; t * TOP_K];
        route_from_logits(&logits, t, w.router_b, &mut indices, &mut weights);
        crate::flags::prefill_lap(&mut lap, 1);

        // ---- Experts, straight from the GPU rows.
        moe.x_override = Some(front.moe_in.clone());
        let b_moe =
            crate::moe::sparse_moe_dispatch_keep_gpu(&[], &indices, &weights, t, w.experts, Some(moe))?
                .context("expert dispatch")?;
        crate::flags::prefill_lap(&mut lap, 2);

        // ---- Output conv and residual.  The stream this visit came in with joins the
        // block stack first when the layer opens a block.
        if w.is_block_boundary {
            anyhow::ensure!(st.n_states < STACK_CAP, "block stack deeper than {STACK_CAP}");
            copy_buffer(&st.x, 0, &st.stack, st.n_states * tp * hb, t * hb);
            st.n_states += 1;
        }
        self.prompt_meta_n = 0;
        let cb = self.gemv.command_buffer()?;
        {
            let enc = cb.computeCommandEncoder().context("encoder")?;
            self.enc_tail(&enc, st, w, &front, &b_moe)?;
            enc.endEncoding();
        }
        cb.commit();
        cb.waitUntilCompleted();
        let chain = &front.chain;

        // ---- A block being checked keeps its rows aside and changes nothing.
        if cache.hold {
            cache.held = Some(Self::read_block_rows(
                &BlockRows {
                    k: chain.k_done.clone(),
                    v: chain.v_done.clone(),
                    k_in: chain.k.clone(),
                    v_in: chain.v.clone(),
                    attn_in: chain.out.clone(),
                    mlp_in: b_moe,
                },
                t,
            ));
            crate::flags::prefill_lap(&mut lap, 3);
            return Ok(());
        }

        // ---- What the next chunk and decode start from.
        let hist_len = CONV_K - 1;
        let tail = hist_len.min(t);
        for (buf, channels, state) in [
            (&chain.k, KV_CH, &mut cache.k_sconv),
            (&chain.v, KV_CH, &mut cache.v_sconv),
            (&chain.out, HIDDEN, &mut cache.attn_sconv),
            (&b_moe, HIDDEN, &mut cache.mlp_sconv),
        ] {
            let mut rows = vec![0f32; tail * channels];
            read_buffer(buf, (t - tail) * channels * 4, &mut rows);
            conv_state_roll(&rows, tail, channels, state, continuing);
        }
        cache
            .gpu
            .as_mut()
            .unwrap()
            .append_buffers(&self.gemv, &chain.k_near, &chain.v_near, chain.hist, t)?;
        crate::flags::prefill_lap(&mut lap, 3);
        Ok(())
    }

    /// The rows a visit left for a block of `t` tokens, read back to the host.
    pub(crate) fn read_block_rows(r: &BlockRows, t: usize) -> crate::cache::HeldRows {
        let rows = |buf: &Buffer, off: usize, channels: usize| {
            let mut v = vec![0f32; t * channels];
            read_buffer(buf, off, &mut v);
            v
        };
        crate::cache::HeldRows {
            tokens: t,
            k: rows(&r.k.0, r.k.1, KV_CH),
            v: rows(&r.v.0, r.v.1, KV_CH),
            k_in: rows(&r.k_in, 0, KV_CH),
            v_in: rows(&r.v_in, 0, KV_CH),
            attn_in: rows(&r.attn_in, 0, HIDDEN),
            mlp_in: rows(&r.mlp_in, 0, HIDDEN),
        }
    }

    // ---- A block of a few tokens, all visits in one command buffer ----
    //
    // The functions below record on an encoder the caller owns and run nothing.  The
    // experts are picked and run by the kernels decode uses, one token at a time, so
    // the host is not needed between the first visit and the last.

    /// First core visit of a block: keep the stream and the stack depth.
    pub(crate) fn block_stash_prelude(&mut self, enc: &Enc, st: &mut PromptGpu) -> Result<()> {
        self.enc_copy(enc, (&st.x, 0), (&st.prelude_x, 0), st.tokens * HIDDEN)?;
        st.prelude_n = st.n_states;
        st.has_prelude = true;
        Ok(())
    }

    /// Core re-entry of a block: the message bank, then back to the prelude.
    pub(crate) fn block_reenter_core(&mut self, enc: &Enc, st: &mut PromptGpu) -> Result<()> {
        let skip = st.prelude_n + 1;
        let n_core = st.n_states.saturating_sub(skip);
        anyhow::ensure!(n_core < BANK_CAP, "message bank deeper than {BANK_CAP}");
        let row = st.tp * HIDDEN * 4;
        let meta = self.pmeta(&[HIDDEN as u32])?;
        for i in 0..n_core {
            self.enc_tok(
                enc,
                "k_tok_rms",
                &[],
                &[(&st.stack, (skip + i) * row), (&st.bank, i * row), (&meta, 0)],
                st.tokens,
                1,
            )?;
        }
        self.enc_tok(
            enc,
            "k_tok_rms",
            &[],
            &[(&st.x, 0), (&st.bank, n_core * row), (&meta, 0)],
            st.tokens,
            1,
        )?;
        st.bank_n = n_core + 1;
        self.enc_copy(enc, (&st.prelude_x, 0), (&st.x, 0), st.tokens * HIDDEN)?;
        st.n_states = st.prelude_n;
        Ok(())
    }

    /// Keep the block's stream as it is now, for tapped state `j`.
    pub(crate) fn block_tap(&mut self, enc: &Enc, st: &PromptGpu, j: usize) -> Result<Buffer> {
        let b = self.pbuf(50 + j as u16, st.tp * HIDDEN)?;
        self.enc_copy(enc, (&st.x, 0), (&b, 0), st.tokens * HIDDEN)?;
        Ok(b)
    }

    /// After a block's last visit: the final normed state of every token, its
    /// vocabulary logits and the best of them.  Returns the states, `[tp][H]`,
    /// and the picks as [`Self::read_block_picks`] wants them.
    pub(crate) fn block_finish(
        &mut self,
        enc: &Enc,
        st: &PromptGpu,
        output_query: &[f32],
        final_norm: &[f32],
        unembed: &[f32],
    ) -> Result<(Buffer, Buffer)> {
        use crate::metal_draft::HEAD_ROWS;
        let t = st.tokens;
        let b_hidden = self.pbuf(53, st.tp * HIDDEN)?;
        let norm = self.weight_buf(final_norm)?;
        self.enc_mix(enc, st, (&st.x, None), Some(output_query), Some(&norm), &b_hidden)?;
        let w = self.weight_buf_f16(unembed)?;
        let b_logits = match self.block_logits.clone() {
            Some(b) => b,
            None => {
                let b = self.gemv.buffer_shared(8 * HEAD_ROWS * 4)?;
                self.block_logits = Some(b.clone());
                b
            }
        };
        self.enc_gemm(enc, &b_hidden, &w, &b_logits, t, HEAD_ROWS, HIDDEN)?;
        let groups = HEAD_ROWS.div_ceil(256);
        let b_part = self.pbuf(54, 8 * groups * 2)?;
        let meta = self.pmeta(&[HEAD_ROWS as u32, groups as u32])?;
        let pso = self.gemv.pso("k_pick_rows", &[])?;
        enc.setComputePipelineState(&pso);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(&b_logits), 0, 0);
            enc.setBuffer_offset_atIndex(Some(&b_part), 0, 1);
            enc.setBuffer_offset_atIndex(Some(&meta), 0, 2);
            enc.setThreadgroupMemoryLength_atIndex(16 * 4, 0);
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize { width: groups, height: t, depth: 1 },
            MTLSize { width: 256, height: 1, depth: 1 },
        );
        Ok((b_hidden, b_part))
    }

    /// For each of a block's `t` tokens: the best vocabulary row among those the
    /// GPU covers, and its logit.
    pub(crate) fn read_block_picks(part: &Buffer, t: usize) -> Vec<(usize, f32)> {
        let groups = crate::metal_draft::HEAD_ROWS.div_ceil(256);
        let mut all = vec![0f32; t * groups * 2];
        read_buffer(part, 0, &mut all);
        all.chunks_exact(groups * 2)
            .map(|rows| {
                let (mut best, mut top) = (0usize, f32::MIN);
                for p in rows.chunks_exact(2) {
                    if p[0] > top {
                        top = p[0];
                        best = p[1] as usize;
                    }
                }
                (best, top)
            })
            .collect()
    }

    /// One layer visit of a block.  Returns where its rows will be.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn block_visit(
        &mut self,
        enc: &Enc,
        moe: &mut MetalMoe,
        st: &mut PromptGpu,
        w: &PromptLayer<'_>,
        cache: &mut VisitCache,
        msg_queries: Option<&[f32]>,
        visit: usize,
    ) -> Result<BlockRows> {
        let (t, tp) = (st.tokens, st.tp);
        anyhow::ensure!(t <= crate::drafter::BLOCK, "a block holds at most 8 tokens");
        cache.gpu_kv(&self.gemv)?;
        self.block_visit = Some(visit);
        let front = self.enc_front(enc, st, w, cache, msg_queries);
        let b_moe = self.pbuf_v(48, tp * HIDDEN);
        self.block_visit = None;
        let (front, b_moe) = (front?, b_moe?);

        // Each token's eight experts, picked the way decode picks them.
        let b_idx = self.pbuf(35, tp * TOP_K)?;
        let b_wt = self.pbuf(36, tp * TOP_K)?;
        let b_bias = self.weight_buf(w.router_b)?;
        let p_route = self.gemv.pso("k_route_topk_t1", &[])?;
        {
            enc.setComputePipelineState(&p_route);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(&front.logits), 0, 0);
                enc.setBuffer_offset_atIndex(Some(&b_bias), 0, 1);
                enc.setBuffer_offset_atIndex(Some(&b_idx), 0, 2);
                enc.setBuffer_offset_atIndex(Some(&b_wt), 0, 3);
                enc.setThreadgroupMemoryLength_atIndex(N_EXPERTS * 4, 0);
                enc.setThreadgroupMemoryLength_atIndex(N_EXPERTS * 4, 1);
                enc.setThreadgroupMemoryLength_atIndex(N_EXPERTS * 4, 2);
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize { width: t, height: 1, depth: 1 },
                MTLSize { width: 32, height: 1, depth: 1 },
            );
        }
        {
            moe.encode_block_mma(
                enc,
                w.experts,
                HIDDEN,
                crate::moe::INTER,
                TOP_K,
                t,
                &front.moe_in,
                &b_idx,
                &b_wt,
                &b_moe,
            )?;
        }

        // The stream this visit came in with joins the block stack when the layer
        // opens a block, before the output conv overwrites it.
        if w.is_block_boundary {
            anyhow::ensure!(st.n_states < STACK_CAP, "block stack deeper than {STACK_CAP}");
            self.enc_copy(enc, (&st.x, 0), (&st.stack, st.n_states * tp * HIDDEN * 4), t * HIDDEN)?;
            st.n_states += 1;
        }
        self.enc_tail(enc, st, w, &front, &b_moe)?;
        let chain = front.chain;
        Ok(BlockRows {
            k: chain.k_done,
            v: chain.v_done,
            k_in: chain.k,
            v_in: chain.v,
            attn_in: chain.out,
            mlp_in: b_moe,
        })
    }
}

#[cfg(test)]
mod tests;
