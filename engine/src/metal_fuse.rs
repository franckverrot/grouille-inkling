//! One-wait sliding visit: attn block → residual → route → MoE → mlp sconv.

use anyhow::{Context, Result};
use objc2_metal::{MTLCommandBuffer, MTLCommandEncoder, MTLComputeCommandEncoder, MTLSize};

use crate::gpu::{write_buffer, Buffer, FcValue};
use crate::metal_moe::MetalMoe;
use crate::metal_ops::{
    ATTN_GATE, ATTN_QSCALE, ATTN_RELPROJ, ATTN_RELSCALE, ATTN_ROUTER, ATTN_SINK, ATTN_WK, ATTN_WO, ATTN_WQ, ATTN_WR, ATTN_WV,
};
use crate::metal_ops::{cb_gpu_ms, profile_phase_restart, MetalOps};
use crate::moe::{ExpertPack, INTER, TOP_K};
use crate::ops::HIDDEN;

/// Message controllers of a core-loop visit on passes 2 and 3.
pub struct FuseMsgCtrls<'a> {
    /// GPU buffer holding this pass's gathered messages, and the byte offset of this
    /// layer's four slots (attn, k, v, mlp; `HIDDEN` floats each).
    pub msgs: (Buffer, usize),
    pub attn: (&'a [f32], &'a [f32], &'a [f32]), // down, forget, gain
    pub k: (&'a [f32], &'a [f32], &'a [f32]),
    pub v: (&'a [f32], &'a [f32], &'a [f32]),
    pub mlp: (&'a [f32], &'a [f32], &'a [f32]),
}

impl MetalOps {
    /// Fused sliding decode visit, recorded onto the open segment encoder.
    ///
    /// Covers: residual mix → message controllers (when `msg_ctrls` is set) → norm
    /// → projections → K/V conv + clips → SDPA → `wo` + attn sconv + residual
    /// → mlp mix / controller / norm → router → top-8 → MoE (NVFP4 or MXFP8)
    /// → mlp sconv + residual → `x`. Stage by stage: see docs/kernel-surface.md.
    #[allow(clippy::too_many_arguments)]
    pub fn decode_sliding_visit_fused(
        &mut self,
        layer: Option<usize>,
        moe: &mut MetalMoe,
        blocks: &[f32],
        wq: &[f32],
        wk: &[f32],
        wv: &[f32],
        wr: &[f32],
        w_sink: &[f32],
        w_gate: &[f32],
        k_taps: &[f32],
        v_taps: &[f32],
        q_scale: &[f32],
        rel_scale: &[f32],
        rel_proj: &[f32],
        wo: &[f32],
        attn_sconv: &[f32],
        mlp_sconv: &[f32],
        attn_query: Option<&[f32]>,
        attn_norm: &[f32],
        mlp_query: &[f32],
        mlp_norm: &[f32],
        router_w: &[f32],
        router_b: &[f32],
        experts: &ExpertPack<'_>,
        k_state: &mut [f32],
        v_state: &mut [f32],
        attn_sconv_state: &mut [f32],
        mlp_sconv_state: &mut [f32],
        kv: &mut crate::cache::GpuKv,
        // Index of this visit in the token: CPU-staged buffers are per slot because
        // every visit is recorded before any of them runs.
        segment_slot: usize,
        msg_ctrls: Option<&FuseMsgCtrls<'_>>,
        is_block_boundary: bool,
    ) -> Result<()> {
        assert!(segment_slot < 48, "segment_slot must be < 48");
        const N_HEADS: usize = 24;
        const N_KV: usize = 8;
        const DH: usize = 128;
        const D_REL: usize = 16;
        const Q_CH: usize = N_HEADS * DH;
        const KV_CH: usize = N_KV * DH;
        const MAX_T: usize = crate::cache::RING;
        const QK_CLIP: f32 = 22.627_417;
        const REL_CLIP: f32 = 8.0;
        // Position of this token and its ring slot.
        let len = kv.len;
        let ring_slot = len % MAX_T;
        if self.blocks_buf.is_none() {
            self.ensure_blocks_buf(blocks)?;
        }
        let n_states = self.blocks_n;

        // Reuse the existing attn-block path for KV-resident proj…wo, but stop
        // before commit by inlining a tailored encode below (one CB with MoE).
        let gemv_pso = |me: &mut Self, name: &str, out_dim: usize, in_dim: usize| {
            me.gemv.pso(
                name,
                &[
                    (0, FcValue::U32(1)),
                    (1, FcValue::U32(out_dim as u32)),
                    (2, FcValue::U32(in_dim as u32)),
                    (3, FcValue::F32(0.0)),
                ],
            )
        };
        // Attn proj weights as f16: halves W bandwidth (the proj bottleneck).
        // tile8: 8 rows share one TG x-cache (unlike per-row tgx which reloads x).
        let p_kv = gemv_pso(self, "k_dense_gemv_reduce_f16w_tile8", KV_CH, HIDDEN)?;
        let p_qg = gemv_pso(self, "k_dense_gemv_reduce_dual_f16w_tile8", Q_CH, HIDDEN)?;
        let p_kvd = gemv_pso(self, "k_dense_gemv_reduce_dual_f16w_tile8", KV_CH, HIDDEN)?;
        let p_r = gemv_pso(self, "k_dense_gemv_reduce_f16w_tile8", N_HEADS * D_REL, HIDDEN)?;
        let p_s = gemv_pso(self, "k_dense_gemv_reduce_f16w_tile8", N_HEADS, HIDDEN)?;
        let p_wo = gemv_pso(self, "k_dense_gemv_reduce_f16w_tile8", HIDDEN, Q_CH)?;
        let p_router = gemv_pso(self, "k_dense_gemv_reduce_f16w_tgx", 128, HIDDEN)?;
        let p_rms_w = self.gemv.pso("k_rms_norm_affine", &[])?;
        let p_copy = self.gemv.pso("k_copy_hidden", &[])?;
        let p_route = self.gemv.pso("k_route_topk_t1", &[])?;

        let b_xq = self.gemv_scratch_buf(0, HIDDEN * 4)?;
        let b_xk = self.gemv_scratch_buf(1, HIDDEN * 4)?;
        let b_xv = self.gemv_scratch_buf(2, HIDDEN * 4)?;
        let b_q = self.gemv_scratch_buf(16, Q_CH * 4)?;
        let b_k = self.gemv_scratch_buf(17, KV_CH * 4)?;
        let b_v = self.gemv_scratch_buf(18, KV_CH * 4)?;
        let b_r = self.gemv_scratch_buf(19, N_HEADS * D_REL * 4)?;
        let b_s = self.gemv_scratch_buf(20, N_HEADS * 4)?;
        let b_g = self.gemv_scratch_buf(21, Q_CH * 4)?;
        let b_ctx = self.gemv_scratch_buf(41, Q_CH * 4)?;
        let b_attn = self.gemv_scratch_buf(42, HIDDEN * 4)?;
        let b_resid = self.gemv_scratch_buf(43, HIDDEN * 4)?;
        let b_mixed = self.gemv_scratch_buf(44, HIDDEN * 4)?;
        let b_moein = self.gemv_scratch_buf(45, HIDDEN * 4)?;
        let b_logits = self.gemv_scratch_buf(46, 128 * 4)?;
        let b_idx = self.gemv_scratch_buf(47, TOP_K * 4)?;
        let b_wt = self.gemv_scratch_buf(48, TOP_K * 4)?;
        let b_moeout = self.gemv_scratch_buf(49, HIDDEN * 4)?;
        let b_nch = self.gemv_scratch_buf(64, 4)?;
        let b_n_hidden = self.gemv_scratch_buf(72, 4)?;
        let b_rms_meta = self.gemv_scratch_buf(74, 8)?;
        let b_pre_mix = self.gemv_scratch_buf(76, HIDDEN * 4)?;
        // Per-segment-slot staging so all 48 visits can share one CB (no mid-flush).
        // Bands are 64-wide so slot∈0..47 never collide.
        let slot = segment_slot as u16;
        debug_assert!(segment_slot < 48);
        let b_tlen = self.gemv_scratch_buf(1000 + slot, 16)?;
        let b_mix_meta = self.gemv_scratch_buf(1100 + slot, 16)?;
        let b_mix_meta_mlp = self.gemv_scratch_buf(1200 + slot, 16)?;
        let b_copy_meta = self.gemv_scratch_buf(1300 + slot, 8)?;

        let kv_key = k_state.as_ptr() as usize;
        let (b_kring, b_vring) = (kv.k_ring.clone(), kv.v_ring.clone());
        let b_m_q = self.gemv_scratch_buf(65, 16)?;
        let b_m_r = self.gemv_scratch_buf(66, 16)?;
        let b_m_k = self.gemv_scratch_buf(67, 16)?;
        let b_m_v = self.gemv_scratch_buf(68, 16)?;
        let b_thr_q = self.gemv_scratch_buf(69, 4)?;
        let b_thr_r = self.gemv_scratch_buf(70, 4)?;
        if !self.attn_block_consts {
            write_buffer(&b_nch, 0, &[KV_CH as u32]);
            write_buffer(&b_m_q, 0, &[N_HEADS as u32, DH as u32, 1u32, 0]);
            write_buffer(&b_m_r, 0, &[N_HEADS as u32, D_REL as u32, 1u32, 0]);
            write_buffer(&b_m_k, 0, &[N_KV as u32, DH as u32, 0u32, 0]);
            write_buffer(&b_m_v, 0, &[N_KV as u32, DH as u32, 0u32, 0]);
            write_buffer(&b_thr_q, 0, &[QK_CLIP]);
            write_buffer(&b_thr_r, 0, &[REL_CLIP]);
            self.attn_block_consts = true;
        }
        write_buffer(&b_n_hidden, 0, &[HIDDEN as u32]);
        // Pre-attn mix: 1 extra (x).
        write_buffer(
            &b_mix_meta,
            0,
            &[n_states as u32, HIDDEN as u32, 1u32, 1u32],
        );
        // MLP mix: 1 extra (residual) or 2 (x_in + residual) on block boundary.
        let mlp_n_extra: u32 = if is_block_boundary { 2 } else { 1 };
        write_buffer(
            &b_mix_meta_mlp,
            0,
            &[n_states as u32, HIDDEN as u32, 1u32, mlp_n_extra],
        );
        write_buffer(&b_rms_meta, 0, &[HIDDEN as u32, 1u32]);

        let b_wq = self.attn_f16_or_convert(layer, ATTN_WQ, wq)?;
        let b_wk = self.attn_f16_or_convert(layer, ATTN_WK, wk)?;
        let b_wv = self.attn_f16_or_convert(layer, ATTN_WV, wv)?;
        let b_wr = self.attn_f16_or_convert(layer, ATTN_WR, wr)?;
        let b_ws = self.attn_f16_or_convert(layer, ATTN_SINK, w_sink)?;
        let b_wg = self.attn_f16_or_convert(layer, ATTN_GATE, w_gate)?;
        let b_ktap = self.weight_buf(k_taps)?;
        let b_vtap = self.weight_buf(v_taps)?;
        let b_qsc = self.attn_f16_or_convert(layer, ATTN_QSCALE, q_scale)?;
        let b_rsc = self.attn_f16_or_convert(layer, ATTN_RELSCALE, rel_scale)?;
        let b_rel = self.attn_f16_or_convert(layer, ATTN_RELPROJ, rel_proj)?;
        let b_wo = self.attn_f16_or_convert(layer, ATTN_WO, wo)?;
        // Opt-in lossy 8-bit projections: [wq, gate, wk, wv, wo].
        let q8w = if crate::flags::attn_q8() {
            Some([
                self.attn_q8(layer, ATTN_WQ, wq, Q_CH, HIDDEN)?,
                self.attn_q8(layer, ATTN_GATE, w_gate, Q_CH, HIDDEN)?,
                self.attn_q8(layer, ATTN_WK, wk, KV_CH, HIDDEN)?,
                self.attn_q8(layer, ATTN_WV, wv, KV_CH, HIDDEN)?,
                self.attn_q8(layer, ATTN_WO, wo, HIDDEN, Q_CH)?,
            ])
        } else {
            None
        };
        let b_asconv = self.weight_buf(attn_sconv)?;
        let b_msconv = self.weight_buf(mlp_sconv)?;
        let b_attn_n = self.weight_buf(attn_norm)?;
        let b_attn_q = match attn_query {
            Some(q) => Some(self.weight_buf(q)?),
            None => None,
        };
        let b_mlp_q = self.weight_buf(mlp_query)?;
        let b_mlp_n = self.weight_buf(mlp_norm)?;
        let b_rw = self.attn_f16_or_convert(layer, ATTN_ROUTER, router_w)?;
        let b_rb = self.weight_buf(router_b)?;
        let mut msg_base = 0usize;
        let msg_bufs: Option<(Buffer, Buffer, Buffer, Buffer, Buffer, Buffer, Buffer, Buffer, Buffer, Buffer, Buffer, Buffer, Buffer)> =
            if let Some(m) = msg_ctrls {
                msg_base = m.msgs.1;
                let b_msgs = m.msgs.0.clone();
                Some((
                    b_msgs,
                    self.weight_buf(m.attn.0)?,
                    self.weight_buf(m.attn.1)?,
                    self.weight_buf(m.attn.2)?,
                    self.weight_buf(m.k.0)?,
                    self.weight_buf(m.k.1)?,
                    self.weight_buf(m.k.2)?,
                    self.weight_buf(m.v.0)?,
                    self.weight_buf(m.v.1)?,
                    self.weight_buf(m.v.2)?,
                    self.weight_buf(m.mlp.0)?,
                    self.weight_buf(m.mlp.1)?,
                    self.weight_buf(m.mlp.2)?,
                ))
            } else {
                None
            };

        let b_xstream = self.ensure_x_buf(None)?;
        // Per-visit sconv state on GPU (scratch slots are shared across visits).
        let sconv_kv_key = kv_key ^ 0xC0FFEE;
        let (b_kst, b_vst) = if let Some((bk, bv, _)) = self.conv_state_bufs.get(&sconv_kv_key).cloned() {
            if len == 0 {
                write_buffer(&bk, 0, k_state);
                write_buffer(&bv, 0, v_state);
            }
            (bk, bv)
        } else {
            let bk = self.gemv.buffer_shared(KV_CH * 3 * 4)?;
            let bv = self.gemv.buffer_shared(KV_CH * 3 * 4)?;
            write_buffer(&bk, 0, k_state);
            write_buffer(&bv, 0, v_state);
            self.conv_state_bufs
                .insert(sconv_kv_key, (bk.clone(), bv.clone(), 1));
            (bk, bv)
        };
        let sconv_am_key = kv_key ^ 0xADD1_EA57;
        let (b_attn_st_res, b_mlp_st_res) =
            if let Some((ba, bm, _)) = self.conv_state_bufs.get(&sconv_am_key).cloned() {
                if len == 0 {
                    write_buffer(&ba, 0, attn_sconv_state);
                    write_buffer(&bm, 0, mlp_sconv_state);
                }
                (ba, bm)
            } else {
                let ba = self.gemv.buffer_shared(HIDDEN * 3 * 4)?;
                let bm = self.gemv.buffer_shared(HIDDEN * 3 * 4)?;
                write_buffer(&ba, 0, attn_sconv_state);
                write_buffer(&bm, 0, mlp_sconv_state);
                self.conv_state_bufs
                    .insert(sconv_am_key, (ba.clone(), bm.clone(), 1));
                (ba, bm)
            };
        // Keep names used below.
        let b_attn_st = b_attn_st_res;
        let b_mlp_st = b_mlp_st_res;
        write_buffer(&b_tlen, 0, &crate::metal_glue::decode_attn_meta(len, false)[..]);
        // A sliding visit has no archive; the attention kernel still wants the binding.
        let b_far = self.gemv_scratch_buf(3430, 16)?;
        // Resident stack; already seeded / grown by ensure_blocks_buf.
        if is_block_boundary && self.blocks_n + 1 > self.blocks_cap {
            // Mid-segment grow needs a flush; prefer large initial cap.
            self.flush_pending()?;
            let mut host = Vec::new();
            self.sync_blocks(&mut host)?;
            self.ensure_blocks_buf(&host)?;
        }
        let b_blocks = self
            .blocks_buf
            .as_ref()
            .context("blocks_buf missing")?
            .clone();
        let new_off = ring_slot * KV_CH * 4;

        let split_gpu = crate::flags::profile_gpu();
        // Continue an open (uncommitted) segment CB when deferring more visits.
        // GPU-split profiling always starts a fresh CB so attn/MoE timings stay clean.
        let mut cb = if split_gpu {
            self.flush_pending()?;
            self.gemv.command_buffer()?
        } else {
            self.take_open_cb()?
        };

        {
            let mut enc = self.take_open_encoder(&cb)?;
            // Order against the previous visit's writes on this concurrent encoder.
            Self::buf_barrier(&enc);
            let one_tg = MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            };
            const TG_N: usize = 256;
            const TG_SIMD: usize = TG_N / 32;
            let reduce = |enc: &objc2::runtime::ProtocolObject<dyn MTLComputeCommandEncoder>,
                          pso: &crate::gpu::Pso,
                          x: &crate::gpu::Buffer,
                          w: &crate::gpu::Buffer,
                          y: &crate::gpu::Buffer,
                          rows: usize,
                          tg_bytes: usize| {
                enc.setComputePipelineState(pso);
                unsafe {
                    enc.setBuffer_offset_atIndex(Some(x), 0, 0);
                    enc.setBuffer_offset_atIndex(Some(w), 0, 1);
                    enc.setBuffer_offset_atIndex(Some(y), 0, 2);
                    enc.setThreadgroupMemoryLength_atIndex(tg_bytes, 0);
                }
                enc.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize {
                        width: rows.div_ceil(8),
                        height: 1,
                        depth: 1,
                    },
                    MTLSize {
                        width: 32,
                        height: 8,
                        depth: 1,
                    },
                );
            };
            let reduce_dual = |enc: &objc2::runtime::ProtocolObject<dyn MTLComputeCommandEncoder>,
                               pso: &crate::gpu::Pso,
                               x: &crate::gpu::Buffer,
                               w0: &crate::gpu::Buffer,
                               w1: &crate::gpu::Buffer,
                               y0: &crate::gpu::Buffer,
                               y1: &crate::gpu::Buffer,
                               rows: usize,
                               tg_bytes: usize| {
                enc.setComputePipelineState(pso);
                unsafe {
                    enc.setBuffer_offset_atIndex(Some(x), 0, 0);
                    enc.setBuffer_offset_atIndex(Some(w0), 0, 1);
                    enc.setBuffer_offset_atIndex(Some(w1), 0, 2);
                    enc.setBuffer_offset_atIndex(Some(y0), 0, 3);
                    enc.setBuffer_offset_atIndex(Some(y1), 0, 4);
                    enc.setThreadgroupMemoryLength_atIndex(tg_bytes, 0);
                }
                enc.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize {
                        width: rows.div_ceil(8),
                        height: 1,
                        depth: 1,
                    },
                    MTLSize {
                        width: 32,
                        height: 8,
                        depth: 1,
                    },
                );
            };
            let tg_x = HIDDEN * 4;
            let tg_x_dual = HIDDEN * 4;
            let tg_wo = Q_CH * 4;
            // GPU pre-attn from resident x when requested.
            let has_msg = msg_bufs.is_some();
            {
                use crate::metal_glue::{CtrlJob, NormJob, NP};
                let b_part = self.gemv_scratch_buf(3300, 3 * NP * 4)?;
                let mixed: &Buffer = if let Some(ref b_aq) = b_attn_q {
                    self.encode_mix(
                        &enc,
                        3000 + slot,
                        &b_blocks,
                        n_states,
                        &b_xstream,
                        None,
                        &[b_aq],
                        &[&b_pre_mix],
                        &b_part,
                    )?;
                    &b_pre_mix
                } else {
                    &b_xstream
                };
                if let Some(ref mb) = msg_bufs {
                    let (ref b_msgs, ref ad, ref af, ref ag, ref kd, ref kf, ref kg, ref vd, ref vf, ref vg, ..) = *mb;
                    self.encode_ctrl_norm(
                        &enc,
                        3210,
                        &[
                            CtrlJob { x: mixed, msgs: b_msgs, msg_off: msg_base, down: ad, forget: af, gain: ag, norm_w: &b_attn_n, out: &b_xq },
                            CtrlJob { x: mixed, msgs: b_msgs, msg_off: HIDDEN * 4 + msg_base, down: kd, forget: kf, gain: kg, norm_w: &b_attn_n, out: &b_xk },
                            CtrlJob { x: mixed, msgs: b_msgs, msg_off: 2 * HIDDEN * 4 + msg_base, down: vd, forget: vf, gain: vg, norm_w: &b_attn_n, out: &b_xv },
                        ],
                    )?;
                } else if b_attn_q.is_some() {
                    self.encode_norm(
                        &enc,
                        &[NormJob { v: mixed, partials: &b_part, part_off: 0, norm_w: &b_attn_n, out: &b_xq }],
                    )?;
                } else {
                    // Layer 0 has no residual mix: plain RMS-affine of the stream.
                    enc.setComputePipelineState(&p_rms_w);
                    unsafe {
                        enc.setBuffer_offset_atIndex(Some(mixed), 0, 0);
                        enc.setBuffer_offset_atIndex(Some(&b_attn_n), 0, 1);
                        enc.setBuffer_offset_atIndex(Some(&b_xq), 0, 2);
                        enc.setBuffer_offset_atIndex(Some(&b_rms_meta), 0, 3);
                        enc.setThreadgroupMemoryLength_atIndex(256 * 4, 0);
                    }
                    enc.dispatchThreadgroups_threadsPerThreadgroup(
                        one_tg,
                        MTLSize { width: 256, height: 1, depth: 1 },
                    );
                    Self::buf_barrier(&enc);
                }
            }
            // Without messages, q=k=v in b_xq; with messages, distinct q/k/v.
            let b_qin = &b_xq;
            let b_kin = if has_msg { &b_xk } else { &b_xq };
            let b_vin = if has_msg { &b_xv } else { &b_xq };
            // Q+gate share x and OUT_DIM: one dual GEMV halves x traffic.
            if let Some(ref q) = q8w {
                self.encode_q8_dual(&enc, b_qin, &q[0], &q[1], &b_q, &b_g, Q_CH, HIDDEN)?;
                self.encode_q8(&enc, b_kin, &q[2], &b_k, KV_CH, HIDDEN, 0.0)?;
                self.encode_q8(&enc, b_vin, &q[3], &b_v, KV_CH, HIDDEN, 0.0)?;
            } else {
                reduce_dual(&enc, &p_qg, b_qin, &b_wq, &b_wg, &b_q, &b_g, Q_CH, tg_x_dual);
                if std::ptr::eq(b_kin, b_vin) {
                    reduce_dual(&enc, &p_kvd, b_kin, &b_wk, &b_wv, &b_k, &b_v, KV_CH, tg_x_dual);
                } else {
                    reduce(&enc, &p_kv, b_kin, &b_wk, &b_k, KV_CH, tg_x);
                    reduce(&enc, &p_kv, b_vin, &b_wv, &b_v, KV_CH, tg_x);
                }
            }
            reduce(&enc, &p_r, b_qin, &b_wr, &b_r, N_HEADS * D_REL, tg_x);
            reduce(&enc, &p_s, b_qin, &b_ws, &b_s, N_HEADS, tg_x);
            // Proj GEMVs are independent of each other; barrier before sconv/clips.
            Self::buf_barrier(&enc);

            self.encode_post_proj(
                &enc,
                &crate::metal_glue::KvConv { x: &b_k, taps: &b_ktap, state: &b_kst, ring: &b_kring, dyn_taps: None },
                &crate::metal_glue::KvConv { x: &b_v, taps: &b_vtap, state: &b_vst, ring: &b_vring, dyn_taps: None },
                new_off,
                (&b_q, &b_qsc, &b_m_q, &b_thr_q),
                (&b_r, &b_rsc, &b_m_r, &b_thr_r),
            )?;
            Self::buf_barrier(&enc);

            if split_gpu {
                enc = profile_phase_restart(&self.gemv, enc, &mut cb, |wall, gpu| {
                    crate::flags::GpuSplitAccum::with_mut(|s| s.add_proj(wall, gpu));
                })?;
            }

            self.encode_decode_attn(
                &enc,
                &crate::metal_glue::DecodeAttn {
                    q: &b_q,
                    k_ring: &b_kring,
                    v_ring: &b_vring,
                    r: &b_r,
                    rel_proj: &b_rel,
                    sinks: &b_s,
                    gate: &b_g,
                    ctx: &b_ctx,
                    meta: &b_tlen,
                    far_parts: &b_far,
                    far: &[],
                },
                len,
            )?;
            Self::buf_barrier(&enc);
            if let Some(ref q) = q8w {
                self.encode_q8(&enc, &b_ctx, &q[4], &b_attn, HIDDEN, Q_CH, 0.0)?;
            } else {
                reduce(&enc, &p_wo, &b_ctx, &b_wo, &b_attn, HIDDEN, tg_wo);
            }
            Self::buf_barrier(&enc);

            if split_gpu {
                enc = profile_phase_restart(&self.gemv, enc, &mut cb, |wall, gpu| {
                    crate::flags::GpuSplitAccum::with_mut(|s| s.add_sdpa(wall, gpu));
                })?;
            }

            // attn short-conv, plus the residual add unless this is a block boundary.
            self.encode_sconv_add(
                &enc,
                &b_attn,
                &b_asconv,
                &b_attn_st,
                &b_resid,
                if is_block_boundary { None } else { Some(&b_xstream) },
            )?;
            Self::buf_barrier(&enc);
            {
                use crate::metal_glue::{CtrlJob, NormJob, NP};
                // mlp residual mix -> (message controller) -> RMS-affine -> moe_in.
                let b_part_mlp = self.gemv_scratch_buf(3301, NP * 4)?;
                let (extra, extra1) = if is_block_boundary {
                    (&b_xstream, Some(&b_resid))
                } else {
                    (&b_resid, None)
                };
                self.encode_mix(
                    &enc,
                    3100 + slot,
                    &b_blocks,
                    n_states,
                    extra,
                    extra1,
                    &[&b_mlp_q],
                    &[&b_mixed],
                    &b_part_mlp,
                )?;
                if let Some(ref mb) = msg_bufs {
                    let (ref b_msgs, _, _, _, _, _, _, _, _, _, ref md, ref mf, ref mg) = *mb;
                    self.encode_ctrl_norm(
                        &enc,
                        3230,
                        &[CtrlJob {
                            x: &b_mixed,
                            msgs: b_msgs,
                            msg_off: 3 * HIDDEN * 4 + msg_base,
                            down: md,
                            forget: mf,
                            gain: mg,
                            norm_w: &b_mlp_n,
                            out: &b_moein,
                        }],
                    )?;
                } else {
                    self.encode_norm(
                        &enc,
                        &[NormJob {
                            v: &b_mixed,
                            partials: &b_part_mlp,
                            part_off: 0,
                            norm_w: &b_mlp_n,
                            out: &b_moein,
                        }],
                    )?;
                }
            }

            Self::buf_barrier(&enc);
            // Router stays per-row tgx (128 rows); tile8 launch would misbind the kernel.
            enc.setComputePipelineState(&p_router);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(&b_moein), 0, 0);
                enc.setBuffer_offset_atIndex(Some(&b_rw), 0, 1);
                enc.setBuffer_offset_atIndex(Some(&b_logits), 0, 2);
                enc.setThreadgroupMemoryLength_atIndex(HIDDEN * 4 + TG_SIMD * 4, 0);
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: 128,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: TG_N,
                    height: 1,
                    depth: 1,
                },
            );
            Self::buf_barrier(&enc);

            enc.setComputePipelineState(&p_route);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(&b_logits), 0, 0);
                enc.setBuffer_offset_atIndex(Some(&b_rb), 0, 1);
                enc.setBuffer_offset_atIndex(Some(&b_idx), 0, 2);
                enc.setBuffer_offset_atIndex(Some(&b_wt), 0, 3);
                enc.setThreadgroupMemoryLength_atIndex(128 * 4, 0);
                enc.setThreadgroupMemoryLength_atIndex(128 * 4, 1);
                enc.setThreadgroupMemoryLength_atIndex(128 * 4, 2);
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                one_tg,
                MTLSize {
                    width: 32,
                    height: 1,
                    depth: 1,
                },
            );

            // Finish sconv→mix→route, then MoE+tail on a fresh CB.
            if split_gpu {
                enc = profile_phase_restart(&self.gemv, enc, &mut cb, |wall, gpu| {
                    crate::flags::GpuSplitAccum::with_mut(|s| s.add_premoe(wall, gpu));
                })?;
            }

            Self::buf_barrier(&enc);
            if experts.mxfp8 {
                moe.encode_mxfp8_resident_t1(
                    &enc,
                    experts.gate_w,
                    experts.gate_s,
                    experts.up_w,
                    experts.up_s,
                    experts.down_w,
                    experts.down_s,
                    128,
                    HIDDEN,
                    INTER,
                    TOP_K,
                    &b_moein,
                    &b_idx,
                    &b_wt,
                    &b_moeout,
                    segment_slot as u8,
                    experts.layer,
                )?;
            } else {
                moe.encode_nvfp4_resident_t1(
                    &enc,
                    experts.gate_w,
                    experts.gate_s,
                    experts.up_w,
                    experts.up_s,
                    experts.down_w,
                    experts.down_s,
                    experts.gate_up_amax,
                    experts.down_amax,
                    128,
                    HIDDEN,
                    INTER,
                    TOP_K,
                    &b_moein,
                    &b_idx,
                    &b_wt,
                    &b_moeout,
                    segment_slot as u8,
                    experts.layer,
                )?;
            }
            Self::buf_barrier(&enc);

            // Boundary: append x_in to the GPU block stack before x is overwritten.
            if is_block_boundary {
                let dst_off = self.blocks_n * HIDDEN;
                write_buffer(&b_copy_meta, 0, &[HIDDEN as u32, dst_off as u32]);
                enc.setComputePipelineState(&p_copy);
                unsafe {
                    enc.setBuffer_offset_atIndex(Some(&b_xstream), 0, 0);
                    enc.setBuffer_offset_atIndex(Some(&b_blocks), 0, 1);
                    enc.setBuffer_offset_atIndex(Some(&b_copy_meta), 0, 2);
                }
                enc.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize { width: HIDDEN.div_ceil(256), height: 1, depth: 1 },
                    MTLSize { width: 256, height: 1, depth: 1 },
                );
                self.blocks_n += 1;
                Self::buf_barrier(&enc);
            }
            // x_out = residual + mlp short-conv, written straight to the stream.
            self.encode_sconv_add(&enc, &b_moeout, &b_msconv, &b_mlp_st, &b_xstream, Some(&b_resid))?;
            if split_gpu {
                enc.endEncoding();
            } else {
                // Leave the encoder open: the next visit appends to the same one.
                self.pending_enc = Some(enc);
            }
        }
        kv.len += 1;
        if split_gpu {
            // Profiling only: one command buffer per phase, so wait here.
            let t_moe = std::time::Instant::now();
            cb.commit();
            cb.waitUntilCompleted();
            crate::flags::GpuSplitAccum::with_mut(|s| {
                s.add_moe(t_moe.elapsed().as_secs_f64() * 1000.0, cb_gpu_ms(&cb));
            });
            self.pending_cb = None;
        } else {
            self.pending_cb = Some(cb);
        }
        self.pending_committed = false;
        let _ = (k_state, v_state, attn_sconv_state, mlp_sconv_state);
        Ok(())
    }
}
