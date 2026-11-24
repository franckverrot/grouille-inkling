//! Staged glue for the fused decode visit: residual mix, message controllers, RMS norm.
//!
//! Each helper appends a short chain of stages to the open segment encoder and ends
//! with a buffer barrier. Every stage is a multi-threadgroup dispatch, and jobs that
//! do not depend on each other (the q/k/v controllers) share a stage, so they overlap
//! on the concurrent encoder instead of running as a serial chain.

use anyhow::Result;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLComputeCommandEncoder, MTLSize};

use crate::gpu::{write_buffer, Buffer};
use crate::metal_ops::MetalOps;
use crate::ops::HIDDEN;

type Enc = ProtocolObject<dyn MTLComputeCommandEncoder>;

/// Sums of squares a stage leaves behind: one per simdgroup of a 256-wide dispatch.
pub(crate) const NP: usize = HIDDEN / 32;

/// Archive rows one threadgroup of `k_decode_far_part` covers.
pub(crate) const FAR_CHUNK: usize = 512;
/// Floats `k_decode_far_part` and `k_decode_ring_part` leave per (chunk, query head).
pub(crate) const FAR_STRIDE: usize = 132;
/// Ring rows one simdgroup of `k_decode_ring_part` covers.
pub(crate) const RING_CHUNK: usize = 32;
/// Ring rows from which decode attention switches to the chunked, two-stage form.
/// Below it the single-stage kernel is faster: the extra stage costs more than the
/// redundant reads.  The two cross near 400 rows on an M2 Max.
pub(crate) const RING_SPLIT: usize = 384;

/// What the attention of a token at position `len` covers, as the kernels read it:
/// `[ring rows valid once the token is in, its ring slot, archive chunks, archive
/// rows]`, then the row count of every archive segment in use.  Archive rows count
/// the row this token pushes out of the ring.
pub(crate) fn decode_attn_meta(len: usize, full: bool) -> Vec<u32> {
    use crate::cache::{ARCH_SEG, RING};
    let far_rows = if full { (len + 1).saturating_sub(RING) } else { 0 };
    let mut meta = vec![
        (len + 1).min(RING) as u32,
        (len % RING) as u32,
        far_rows.div_ceil(FAR_CHUNK) as u32,
        far_rows as u32,
    ];
    meta.extend((0..far_rows.div_ceil(ARCH_SEG)).map(|s| ARCH_SEG.min(far_rows - s * ARCH_SEG) as u32));
    meta
}

/// Buffers of one decode attention step.
pub(crate) struct DecodeAttn<'a> {
    pub q: &'a Buffer,
    pub k_ring: &'a Buffer,
    pub v_ring: &'a Buffer,
    pub r: &'a Buffer,
    pub rel_proj: &'a Buffer,
    pub sinks: &'a Buffer,
    pub gate: &'a Buffer,
    pub ctx: &'a Buffer,
    /// [`decode_attn_meta`], written by the host.
    pub meta: &'a Buffer,
    /// Scratch for the archive partials; any buffer when there is no archive.
    pub far_parts: &'a Buffer,
    /// The archive segments in use, as [`crate::cache::GpuKv::archive_segments`] gives them.
    pub far: &'a [(Buffer, Buffer, usize)],
}

/// One message controller followed by an affine RMS norm.
pub(crate) struct CtrlJob<'a> {
    pub x: &'a Buffer,
    pub msgs: &'a Buffer,
    /// Byte offset of this controller's message slot in `msgs`.
    pub msg_off: usize,
    pub down: &'a Buffer,
    pub forget: &'a Buffer,
    pub gain: &'a Buffer,
    pub norm_w: &'a Buffer,
    pub out: &'a Buffer,
}

/// `out = v * inv_rms(v) * w`, with sums of `v^2` at `partials[part_off..]` (bytes).
pub(crate) struct NormJob<'a> {
    pub v: &'a Buffer,
    pub partials: &'a Buffer,
    pub part_off: usize,
    pub norm_w: &'a Buffer,
    pub out: &'a Buffer,
}

/// K or V projection output on its way into the KV ring.
pub(crate) struct KvConv<'a> {
    pub x: &'a Buffer,
    /// Static taps: `[C, 4]` for sliding layers, `[4, C]` for full-attention layers.
    pub taps: &'a Buffer,
    pub state: &'a Buffer,
    pub ring: &'a Buffer,
    /// Per-token taps (`[n_kv * 4]`) on full-attention layers.
    pub dyn_taps: Option<&'a Buffer>,
}

fn sz(w: usize) -> MTLSize {
    MTLSize {
        width: w,
        height: 1,
        depth: 1,
    }
}

impl MetalOps {
    /// Residual mix over `n_states` resident block states plus one or two extras, in
    /// one stage. `queries` and `outs` have the same length (1 or 3). Sums of `out_j^2`
    /// land in `partials` (`[nq * NP]` floats). `meta_key` must be unique per call site
    /// and visit (CPU-staged).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_mix(
        &mut self,
        enc: &Enc,
        meta_key: u16,
        blocks: &Buffer,
        n_states: usize,
        extra: &Buffer,
        extra1: Option<&Buffer>,
        queries: &[&Buffer],
        outs: &[&Buffer],
        partials: &Buffer,
    ) -> Result<()> {
        let nq = queries.len();
        assert!(nq == 1 || nq == 3);
        assert_eq!(outs.len(), nq);
        let n_extra = if extra1.is_some() { 2 } else { 1 };
        let n = n_states + n_extra;
        assert!(n <= 16, "residual mix supports at most 16 sources");
        let p_mix = self.gemv.pso("k_mix", &[])?;
        let b_meta = self.gemv_scratch_buf(meta_key, 16)?;
        write_buffer(&b_meta, 0, &[n_states as u32, n_extra as u32, nq as u32, 1u32]);
        let e1 = extra1.unwrap_or(extra);
        enc.setComputePipelineState(&p_mix);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(blocks), 0, 0);
            enc.setBuffer_offset_atIndex(Some(extra), 0, 1);
            enc.setBuffer_offset_atIndex(Some(e1), 0, 2);
            enc.setBuffer_offset_atIndex(Some(queries[0]), 0, 3);
            enc.setBuffer_offset_atIndex(Some(queries[1.min(nq - 1)]), 0, 4);
            enc.setBuffer_offset_atIndex(Some(queries[2.min(nq - 1)]), 0, 5);
            enc.setBuffer_offset_atIndex(Some(&b_meta), 0, 6);
            enc.setBuffer_offset_atIndex(Some(outs[0]), 0, 7);
            enc.setBuffer_offset_atIndex(Some(outs[1.min(nq - 1)]), 0, 8);
            enc.setBuffer_offset_atIndex(Some(outs[2.min(nq - 1)]), 0, 9);
            enc.setBuffer_offset_atIndex(Some(partials), 0, 10);
            enc.setThreadgroupMemoryLength_atIndex(64 * 4, 0);
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(sz(HIDDEN / 256), sz(256));
        Self::buf_barrier(enc);
        Ok(())
    }

    /// Message controllers plus affine RMS norm for up to four jobs: three stages
    /// (down, apply, norm). `key` is the first of 12 scratch slots owned by the call site.
    pub(crate) fn encode_ctrl_norm(&mut self, enc: &Enc, key: u16, jobs: &[CtrlJob<'_>]) -> Result<()> {
        assert!(!jobs.is_empty() && jobs.len() <= 4);
        let p_down = self.gemv.pso("k_ctrl_down", &[])?;
        let p_apply = self.gemv.pso("k_ctrl_apply", &[])?;
        let mut scratch = Vec::with_capacity(jobs.len());
        for j in 0..jobs.len() as u16 {
            scratch.push((
                self.gemv_scratch_buf(key + j, 130 * 4)?,
                self.gemv_scratch_buf(key + 4 + j, HIDDEN * 4)?,
                self.gemv_scratch_buf(key + 8 + j, NP * 4)?,
            ));
        }
        enc.setComputePipelineState(&p_down);
        for (job, (dn, _, _)) in jobs.iter().zip(&scratch) {
            unsafe {
                enc.setBuffer_offset_atIndex(Some(job.x), 0, 0);
                enc.setBuffer_offset_atIndex(Some(job.msgs), job.msg_off, 1);
                enc.setBuffer_offset_atIndex(Some(job.down), 0, 2);
                enc.setBuffer_offset_atIndex(Some(dn), 0, 3);
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(sz(65), sz(32));
        }
        Self::buf_barrier(enc);
        enc.setComputePipelineState(&p_apply);
        for (job, (dn, blend, part)) in jobs.iter().zip(&scratch) {
            unsafe {
                enc.setBuffer_offset_atIndex(Some(job.x), 0, 0);
                enc.setBuffer_offset_atIndex(Some(job.msgs), job.msg_off, 1);
                enc.setBuffer_offset_atIndex(Some(job.forget), 0, 2);
                enc.setBuffer_offset_atIndex(Some(job.gain), 0, 3);
                enc.setBuffer_offset_atIndex(Some(dn), 0, 4);
                enc.setBuffer_offset_atIndex(Some(blend), 0, 5);
                enc.setBuffer_offset_atIndex(Some(part), 0, 6);
                enc.setThreadgroupMemoryLength_atIndex(64 * 4, 0);
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(sz(HIDDEN / 256), sz(256));
        }
        Self::buf_barrier(enc);
        let norms: Vec<NormJob<'_>> = jobs
            .iter()
            .zip(&scratch)
            .map(|(job, (_, blend, part))| NormJob {
                v: blend,
                partials: part,
                part_off: 0,
                norm_w: job.norm_w,
                out: job.out,
            })
            .collect();
        self.encode_norm(enc, &norms)
    }

    /// Affine RMS norm from sums left by a previous stage. One stage for all jobs.
    pub(crate) fn encode_norm(&mut self, enc: &Enc, jobs: &[NormJob<'_>]) -> Result<()> {
        let p_norm = self.gemv.pso("k_norm_from_partials", &[])?;
        enc.setComputePipelineState(&p_norm);
        for job in jobs {
            unsafe {
                enc.setBuffer_offset_atIndex(Some(job.v), 0, 0);
                enc.setBuffer_offset_atIndex(Some(job.partials), job.part_off, 1);
                enc.setBuffer_offset_atIndex(Some(job.norm_w), 0, 2);
                enc.setBuffer_offset_atIndex(Some(job.out), 0, 3);
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(sz(HIDDEN / 256), sz(256));
        }
        Self::buf_barrier(enc);
        Ok(())
    }

    /// One stage after the projections: K/V short-conv plus per-head norm written to
    /// the ring row at `ring_off`, and the scaled L2 clips of Q and the relative
    /// logits. `m_q` / `m_r` use the `[n_rows, dh, has_scale, _]` meta layout.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_post_proj(
        &mut self,
        enc: &Enc,
        k: &KvConv<'_>,
        v: &KvConv<'_>,
        ring_off: usize,
        q: (&Buffer, &Buffer, &Buffer, &Buffer),
        r: (&Buffer, &Buffer, &Buffer, &Buffer),
    ) -> Result<()> {
        let p_kv = self.gemv.pso("k_kv_sconv_head", &[])?;
        let p_clip = self.gemv.pso("k_l2_clip_head", &[])?;
        let b_dummy = self.gemv_scratch_buf(3420, 64)?;
        enc.setComputePipelineState(&p_kv);
        for (job, mode) in [(k, 0u32), (v, 1u32)] {
            let dynamic = job.dyn_taps.is_some() as u32;
            // Four constant metas; rewriting the same values is harmless.
            let b_meta = self.gemv_scratch_buf(3410 + (mode * 2 + dynamic) as u16, 8)?;
            write_buffer(&b_meta, 0, &[mode, dynamic]);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(job.x), 0, 0);
                enc.setBuffer_offset_atIndex(Some(job.taps), 0, 1);
                enc.setBuffer_offset_atIndex(Some(job.state), 0, 2);
                enc.setBuffer_offset_atIndex(Some(job.ring), ring_off, 3);
                enc.setBuffer_offset_atIndex(Some(&b_meta), 0, 4);
                enc.setBuffer_offset_atIndex(Some(q.3), 0, 5);
                enc.setBuffer_offset_atIndex(Some(job.dyn_taps.unwrap_or(&b_dummy)), 0, 6);
                enc.setThreadgroupMemoryLength_atIndex(16, 0);
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(sz(8), sz(128));
        }
        enc.setComputePipelineState(&p_clip);
        for ((x, scale, meta, thr), rows, threads) in [(q, 24usize, 128usize), (r, 24, 32)] {
            unsafe {
                enc.setBuffer_offset_atIndex(Some(x), 0, 0);
                enc.setBuffer_offset_atIndex(Some(scale), 0, 1);
                enc.setBuffer_offset_atIndex(Some(meta), 0, 2);
                enc.setBuffer_offset_atIndex(Some(thr), 0, 3);
                enc.setThreadgroupMemoryLength_atIndex(16, 0);
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(sz(rows), sz(threads));
        }
        Self::buf_barrier(enc);
        Ok(())
    }

    /// Move the ring row at `slot` to archive row `arch_row`.  Goes in a stage before
    /// the one that overwrites the slot.
    pub(crate) fn encode_archive_row(
        &mut self,
        enc: &Enc,
        kv: &crate::cache::GpuKv,
        slot: usize,
        arch_row: usize,
    ) -> Result<()> {
        const KV_CH: usize = 8 * 128;
        let p = self.gemv.pso("k_archive_row", &[])?;
        let (seg, dst) = crate::cache::GpuKv::archive_slot(arch_row);
        let (ka, va) = (&kv.k_arch[seg], &kv.v_arch[seg]);
        let src = slot * KV_CH * 4;
        enc.setComputePipelineState(&p);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(&kv.k_ring), src, 0);
            enc.setBuffer_offset_atIndex(Some(&kv.v_ring), src, 1);
            enc.setBuffer_offset_atIndex(Some(ka), dst, 2);
            enc.setBuffer_offset_atIndex(Some(va), dst, 3);
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(sz(1), sz(256));
        Ok(())
    }

    /// Decode attention for the token at position `len`.  No trailing barrier.
    ///
    /// While the history is shorter than [`RING_SPLIT`] rows one kernel does it all
    /// in a single stage.  Beyond that the ring goes through in chunks, three query
    /// heads per KV head, and so does the archive when there is one.  After a
    /// barrier one more kernel folds every chunk and the sink into a softmax and
    /// applies the output gate.  The second form reads every row once and spreads
    /// over the whole GPU, and pays for it with the extra stage.
    pub(crate) fn encode_decode_attn(&mut self, enc: &Enc, a: &DecodeAttn<'_>, len: usize) -> Result<()> {
        use crate::cache::RING;
        use crate::gpu::FcValue;
        const N_HEADS: usize = 24;
        const N_KV: usize = 8;
        const DH: usize = 128;
        const D_REL: usize = 16;
        let ring_rows = (len + 1).min(RING);
        if ring_rows < RING_SPLIT {
            let p = self.gemv.pso(
                "k_decode_attn_small",
                &[
                    (4, FcValue::U32(N_HEADS as u32)),
                    (5, FcValue::U32(N_KV as u32)),
                    (6, FcValue::U32(DH as u32)),
                    (7, FcValue::U32(D_REL as u32)),
                    (8, FcValue::U32(RING as u32)),
                ],
            )?;
            enc.setComputePipelineState(&p);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(a.q), 0, 0);
                enc.setBuffer_offset_atIndex(Some(a.k_ring), 0, 1);
                enc.setBuffer_offset_atIndex(Some(a.v_ring), 0, 2);
                enc.setBuffer_offset_atIndex(Some(a.r), 0, 3);
                enc.setBuffer_offset_atIndex(Some(a.rel_proj), 0, 4);
                enc.setBuffer_offset_atIndex(Some(a.sinks), 0, 5);
                enc.setBuffer_offset_atIndex(Some(a.gate), 0, 6);
                enc.setBuffer_offset_atIndex(Some(a.ctx), 0, 7);
                enc.setBuffer_offset_atIndex(Some(a.meta), 0, 8);
                enc.setThreadgroupMemoryLength_atIndex((RING + 4 + 16 + 8 * DH) * 4, 0);
            }
            // One threadgroup per query head.
            enc.dispatchThreadgroups_threadsPerThreadgroup(sz(N_HEADS), sz(256));
            return Ok(());
        }
        let ring_chunks = ring_rows.div_ceil(RING_CHUNK);
        let fc = [
            (4, FcValue::U32(N_HEADS as u32)),
            (5, FcValue::U32(N_KV as u32)),
            (6, FcValue::U32(DH as u32)),
            (7, FcValue::U32(D_REL as u32)),
            (8, FcValue::U32(RING as u32)),
            (10, FcValue::U32(RING_CHUNK as u32)),
        ];
        // Written and read inside one visit, so every visit can share it.
        let b_ring = self.gemv_scratch_buf(
            3432,
            RING.div_ceil(RING_CHUNK) * N_HEADS * FAR_STRIDE * 4,
        )?;
        let p_ring = self.gemv.pso("k_decode_ring_part", &fc)?;
        enc.setComputePipelineState(&p_ring);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(a.q), 0, 0);
            enc.setBuffer_offset_atIndex(Some(a.k_ring), 0, 1);
            enc.setBuffer_offset_atIndex(Some(a.v_ring), 0, 2);
            enc.setBuffer_offset_atIndex(Some(a.r), 0, 3);
            enc.setBuffer_offset_atIndex(Some(a.rel_proj), 0, 4);
            enc.setBuffer_offset_atIndex(Some(&b_ring), 0, 5);
            enc.setBuffer_offset_atIndex(Some(a.meta), 0, 6);
        }
        // One simdgroup per (ring chunk, KV head).
        enc.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize { width: ring_chunks, height: N_KV, depth: 1 },
            sz(32),
        );
        if !a.far.is_empty() {
            let p_far = self.gemv.pso(
                "k_decode_far_part",
                &[
                    (4, FcValue::U32(N_HEADS as u32)),
                    (5, FcValue::U32(N_KV as u32)),
                    (6, FcValue::U32(DH as u32)),
                ],
            )?;
            enc.setComputePipelineState(&p_far);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(a.q), 0, 0);
                enc.setThreadgroupMemoryLength_atIndex(3 * (FAR_CHUNK + 16 + 8 * DH) * 4, 0);
            }
            // One dispatch per segment, one threadgroup per (archive chunk, KV head).
            // The segments do not depend on each other, nor on the ring.
            let seg_chunks = crate::cache::ARCH_SEG / FAR_CHUNK;
            for (s, (ka, va, rows)) in a.far.iter().enumerate() {
                unsafe {
                    enc.setBuffer_offset_atIndex(Some(ka), 0, 1);
                    enc.setBuffer_offset_atIndex(Some(va), 0, 2);
                    enc.setBuffer_offset_atIndex(
                        Some(a.far_parts),
                        s * seg_chunks * N_HEADS * FAR_STRIDE * 4,
                        3,
                    );
                    enc.setBuffer_offset_atIndex(Some(a.meta), (4 + s) * 4, 4);
                }
                enc.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize { width: rows.div_ceil(FAR_CHUNK), height: N_KV, depth: 1 },
                    sz(256),
                );
            }
        }
        Self::buf_barrier(enc);
        let p_merge = self.gemv.pso("k_decode_merge", &fc)?;
        enc.setComputePipelineState(&p_merge);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(&b_ring), 0, 0);
            enc.setBuffer_offset_atIndex(Some(a.far_parts), 0, 1);
            enc.setBuffer_offset_atIndex(Some(a.sinks), 0, 2);
            enc.setBuffer_offset_atIndex(Some(a.gate), 0, 3);
            enc.setBuffer_offset_atIndex(Some(a.ctx), 0, 4);
            enc.setBuffer_offset_atIndex(Some(a.meta), 0, 5);
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(sz(N_HEADS), sz(32));
        Ok(())
    }

    /// `out = short_conv(x) (+ add)` over the hidden width; updates the rolling state.
    /// No trailing barrier.
    pub(crate) fn encode_sconv_add(
        &mut self,
        enc: &Enc,
        x: &Buffer,
        taps: &Buffer,
        state: &Buffer,
        out: &Buffer,
        add: Option<&Buffer>,
    ) -> Result<()> {
        let p = self.gemv.pso("k_short_conv_add", &[])?;
        let has_add = add.is_some() as u32;
        let b_meta = self.gemv_scratch_buf(3400 + has_add as u16, 8)?;
        write_buffer(&b_meta, 0, &[HIDDEN as u32, has_add]);
        enc.setComputePipelineState(&p);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(x), 0, 0);
            enc.setBuffer_offset_atIndex(Some(taps), 0, 1);
            enc.setBuffer_offset_atIndex(Some(state), 0, 2);
            enc.setBuffer_offset_atIndex(Some(out), 0, 3);
            enc.setBuffer_offset_atIndex(Some(&b_meta), 0, 4);
            enc.setBuffer_offset_atIndex(Some(add.unwrap_or(x)), 0, 5);
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(sz(HIDDEN.div_ceil(64)), sz(64));
        Ok(())
    }

    /// Prompt attention for a chunk of `tokens` queries, from host slices, in its own
    /// command buffer.  The prompt path keeps everything on the GPU and calls
    /// [`Self::encode_attn_chunk`] directly; this is the same thing for a caller that
    /// has the rows on the host.
    ///
    /// Inputs are token-major and already clipped / normalized: `q` `[T, 24*128]`,
    /// `k`/`v` `[T, 8*128]`, `r` `[T, 24*16]`, `sinks` `[T, 24]`, `gate` `[T, 24*128]`.
    /// `history` is what the visit kept from earlier chunks; `k`/`v` are not in it yet.
    /// `ctx` comes back token-major and already multiplied by sigmoid(gate).
    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn attn_chunk(
        &mut self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        r: &[f32],
        rel_proj_f16: &Buffer,
        sinks: &[f32],
        gate: &[f32],
        tokens: usize,
        history: Option<&crate::cache::GpuKv>,
        full: bool,
        ctx: &mut [f32],
    ) -> Result<()> {
        use crate::gpu::read_buffer;
        use objc2_metal::{MTLCommandBuffer, MTLCommandEncoder};
        const Q_CH: usize = 24 * 128;
        const KV_CH: usize = 8 * 128;
        assert_eq!(q.len(), tokens * Q_CH);
        assert_eq!(k.len(), tokens * KV_CH);
        assert_eq!(ctx.len(), tokens * Q_CH);
        let pos0 = history.map_or(0, |h| h.len);
        let hist = history.map_or(0, |h| h.ring_rows());
        let far = history.filter(|_| full).map_or(Vec::new(), |h| h.archive_segments(h.archive_rows()));
        let tp = tokens.next_multiple_of(32);
        let b_q = self.gemv_scratch_buf(3600, tp * Q_CH * 4)?;
        let b_k = self.gemv_scratch_buf(3601, (hist + tp) * KV_CH * 4)?;
        let b_v = self.gemv_scratch_buf(3602, (hist + tp) * KV_CH * 4)?;
        let b_r = self.gemv_scratch_buf(3603, tp * 24 * 16 * 4)?;
        let b_sink = self.gemv_scratch_buf(3604, tp * 24 * 4)?;
        let b_gate = self.gemv_scratch_buf(3605, tp * Q_CH * 4)?;
        let b_ctx = self.gemv_scratch_buf(3608, tp * Q_CH * 4)?;
        write_buffer(&b_q, 0, q);
        if let Some(h) = history.filter(|_| hist > 0) {
            h.unroll_ring(&b_k, &b_v);
        }
        write_buffer(&b_k, hist * KV_CH * 4, k);
        write_buffer(&b_v, hist * KV_CH * 4, v);
        write_buffer(&b_r, 0, r);
        write_buffer(&b_sink, 0, sinks);
        write_buffer(&b_gate, 0, gate);
        self.flush_pending()?;
        let cb = self.gemv.command_buffer()?;
        {
            let enc = cb
                .computeCommandEncoder()
                .ok_or_else(|| anyhow::anyhow!("encoder"))?;
            self.encode_attn_chunk(
                &enc,
                &ChunkAttn {
                    q: &b_q,
                    k_near: &b_k,
                    v_near: &b_v,
                    r: &b_r,
                    rel_proj: rel_proj_f16,
                    sinks: &b_sink,
                    gate: &b_gate,
                    ctx: &b_ctx,
                },
                tokens,
                pos0,
                hist,
                &far,
                full,
            )?;
            enc.endEncoding();
        }
        cb.commit();
        cb.waitUntilCompleted();
        read_buffer(&b_ctx, 0, ctx);
        Ok(())
    }

    /// Encode a chunk's attention on a serial encoder.  `k_near` / `v_near` hold
    /// `hist` ring rows, oldest first, then the chunk's own `tokens` rows; the chunk
    /// starts at position `pos0`.  It attends to those in one f32 block, then, on a
    /// full-attention visit, to the archive segments in `far`, [`FAR_BLOCK`] rows at
    /// a time.  `ctx` is token-major and already multiplied by sigmoid(gate).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_attn_chunk(
        &mut self,
        enc: &Enc,
        a: &ChunkAttn<'_>,
        tokens: usize,
        pos0: usize,
        hist: usize,
        far: &[(Buffer, Buffer, usize)],
        full: bool,
    ) -> Result<()> {
        use crate::cache::{arch_bytes, ARCH_SEG, RING};
        use crate::gpu::FcValue;
        const N_HEADS: usize = 24;
        const N_KV: usize = 8;
        const DH: usize = 128;
        const D_REL: usize = 16;
        /// Archive rows per block.
        const FAR_BLOCK: usize = 2048;
        let kv_ch = N_KV * DH;
        assert!(tokens <= RING, "a prompt chunk covers at most {RING} tokens");
        // Kernels work on tiles of 32 queries by 8 keys; every buffer is padded to that.
        let tp = tokens.next_multiple_of(32);
        let near = hist + tokens;
        let near_p = near.next_multiple_of(8);
        let key_cap = near_p.max(if far.is_empty() { 0 } else { FAR_BLOCK });
        let fc = [
            (4, FcValue::U32(N_HEADS as u32)),
            (5, FcValue::U32(N_KV as u32)),
            (6, FcValue::U32(DH as u32)),
            (7, FcValue::U32(D_REL as u32)),
            (8, FcValue::U32(RING as u32)),
            // A full-attention visit has no window.
            (9, FcValue::U32(if full { 0 } else { RING as u32 })),
        ];
        let p_scores = self.gemv.pso("k_attn_scores_mma", &fc)?;
        let p_soft = self.gemv.pso("k_attn_softmax_rows", &fc)?;
        let p_ctx = self.gemv.pso("k_attn_ctx_mma", &fc)?;
        let b_s = self.gemv_scratch_buf(3606, N_HEADS * tp * key_cap * 4)?;
        let b_state = self.gemv_scratch_buf(3607, N_HEADS * tp * 16)?;

        // (keys, padded keys, position of key 0, K, V, byte offset, archive block?)
        let mut blocks =
            vec![(near, near_p, pos0 - hist, a.k_near.clone(), a.v_near.clone(), 0usize, false)];
        for (seg, (ka, va, rows)) in far.iter().enumerate() {
            for row0 in (0..*rows).step_by(FAR_BLOCK) {
                let n = FAR_BLOCK.min(rows - row0);
                blocks.push((
                    n,
                    n.next_multiple_of(8),
                    seg * ARCH_SEG + row0,
                    ka.clone(),
                    va.clone(),
                    row0 * kv_ch * arch_bytes(),
                    true,
                ));
            }
        }
        let far_psos = if far.is_empty() {
            None
        } else {
            Some((
                self.gemv.pso("k_attn_scores_far_mma", &fc)?,
                self.gemv.pso("k_attn_ctx_far_mma", &fc)?,
            ))
        };
        let simdgroup = sz(32);
        let n_blocks = blocks.len();
        for (i, (nk, nkp, k_pos0, bk, bv, off, is_far)) in blocks.into_iter().enumerate() {
            let flags = (i == 0) as u32 | (((i + 1 == n_blocks) as u32) << 1);
            let b_meta = self.pmeta(&[
                tokens as u32,
                tp as u32,
                nk as u32,
                nkp as u32,
                pos0 as u32,
                k_pos0 as u32,
                flags,
                0u32,
            ])?;
            let (scores, context) = match (is_far, far_psos.as_ref()) {
                (true, Some((ps, pc))) => (ps, pc),
                _ => (&p_scores, &p_ctx),
            };
            enc.setComputePipelineState(scores);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(a.q), 0, 0);
                enc.setBuffer_offset_atIndex(Some(&bk), off, 1);
                enc.setBuffer_offset_atIndex(Some(&b_s), 0, 2);
                enc.setBuffer_offset_atIndex(Some(&b_meta), 0, 3);
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize { width: nkp / 8, height: tp / 32, depth: N_HEADS },
                simdgroup,
            );
            enc.setComputePipelineState(&p_soft);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(&b_s), 0, 0);
                enc.setBuffer_offset_atIndex(Some(a.r), 0, 1);
                enc.setBuffer_offset_atIndex(Some(a.rel_proj), 0, 2);
                enc.setBuffer_offset_atIndex(Some(a.sinks), 0, 3);
                enc.setBuffer_offset_atIndex(Some(&b_state), 0, 4);
                enc.setBuffer_offset_atIndex(Some(&b_meta), 0, 5);
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize { width: tokens, height: N_HEADS, depth: 1 },
                simdgroup,
            );
            enc.setComputePipelineState(context);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(&b_s), 0, 0);
                enc.setBuffer_offset_atIndex(Some(&bv), off, 1);
                enc.setBuffer_offset_atIndex(Some(&b_state), 0, 2);
                enc.setBuffer_offset_atIndex(Some(a.gate), 0, 3);
                enc.setBuffer_offset_atIndex(Some(a.ctx), 0, 4);
                enc.setBuffer_offset_atIndex(Some(&b_meta), 0, 5);
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize { width: DH / 8, height: tp / 32, depth: N_HEADS },
                simdgroup,
            );
        }
        Ok(())
    }
}

/// Buffers of a chunk's attention, all on the GPU and padded to 32-token tiles.
pub(crate) struct ChunkAttn<'a> {
    pub q: &'a Buffer,
    pub k_near: &'a Buffer,
    pub v_near: &'a Buffer,
    pub r: &'a Buffer,
    pub rel_proj: &'a Buffer,
    pub sinks: &'a Buffer,
    pub gate: &'a Buffer,
    pub ctx: &'a Buffer,
}
