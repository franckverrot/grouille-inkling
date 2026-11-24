//! GPU-resident decode helpers: prelude stash, core-reentry message bank + gather,
//! and final mix + norm + unembed argmax, all encoded on the open segment CB so a
//! token needs no host waits between prelude, core passes, postlude and sampling.

use anyhow::{Context, Result};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBarrierScope, MTLCommandBuffer, MTLCommandEncoder, MTLComputeCommandEncoder, MTLDispatchType,
    MTLSize,
};

use crate::gpu::{read_buffer, write_buffer, Buffer, CommandBuffer, FcValue, Pso};
use crate::metal_ops::{cb_gpu_ms, MetalOps};

const H: usize = 2304;
/// Residual-mix kernel handles at most 16 sources (states + extras).
const MIX_MAX: usize = 16;
/// `k_gather_messages_t1` score capacity.
pub const GPU_BANK_MAX: usize = 32;
/// Rows reserved for the prelude block stash.
const PRELUDE_MAX: usize = 32;
/// `[12 layers][4 slots]` queries per core pass.
const N_Q: usize = 48;

type Enc = Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>;

/// Fresh segment compute encoder: Concurrent by default, Serial under `INKLING_SERIAL_ENC=1`.
pub(crate) fn new_segment_encoder(cb: &CommandBuffer) -> Result<Enc> {
    if crate::flags::serial_enc() {
        cb.computeCommandEncoder().context("serial segment encoder")
    } else {
        cb.computeCommandEncoderWithDispatchType(MTLDispatchType::Concurrent)
            .context("concurrent segment encoder")
    }
}

fn tg(w: usize) -> MTLSize {
    MTLSize { width: w, height: 1, depth: 1 }
}

fn encode_copy(
    enc: &Enc,
    pso: &Pso,
    src: &Buffer,
    src_off: usize,
    dst: &Buffer,
    dst_off: usize,
    n_buf: &Buffer,
    n: usize,
) {
    enc.setComputePipelineState(pso);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(src), src_off, 0);
        enc.setBuffer_offset_atIndex(Some(dst), dst_off, 1);
        enc.setBuffer_offset_atIndex(Some(n_buf), 0, 2);
    }
    enc.dispatchThreadgroups_threadsPerThreadgroup(tg(n.div_ceil(256)), tg(256));
}

impl MetalOps {
    /// Open (uncommitted) segment CB, or a fresh one.
    pub(crate) fn take_open_cb(&mut self) -> Result<CommandBuffer> {
        if let Some(prev) = self.pending_cb.take() {
            if self.pending_committed {
                if let Some(e) = self.pending_enc.take() {
                    e.endEncoding();
                }
                prev.waitUntilCompleted();
                self.pending_committed = false;
                self.gemv.command_buffer()
            } else {
                Ok(prev)
            }
        } else {
            self.gemv.command_buffer()
        }
    }

    fn put_open_cb(&mut self, cb: CommandBuffer) {
        self.pending_cb = Some(cb);
        self.pending_committed = false;
    }

    /// The segment-wide compute encoder on `cb`: the stored open one, else a fresh one.
    ///
    /// Concurrent dispatch (unless `INKLING_SERIAL_ENC=1`): every producer→consumer
    /// edge must be separated by [`Self::buf_barrier`]; independent dispatches may overlap.
    /// `cb` must be the CB returned by [`Self::take_open_cb`].
    pub(crate) fn take_open_encoder(&mut self, cb: &CommandBuffer) -> Result<Enc> {
        if let Some(enc) = self.pending_enc.take() {
            return Ok(enc);
        }
        new_segment_encoder(cb)
    }

    /// Buffer-scope memory barrier: all prior dispatches on `enc` complete (and are
    /// visible) before any later dispatch starts. No-op under `INKLING_SERIAL_ENC=1`.
    pub(crate) fn buf_barrier(enc: &ProtocolObject<dyn MTLComputeCommandEncoder>) {
        if !crate::flags::serial_enc() {
            enc.memoryBarrierWithScope(MTLBarrierScope::Buffers);
        }
    }

    /// Park `enc` (not ended) on the pending segment so the next visit appends to it.
    pub(crate) fn put_open_encoder(&mut self, enc: Enc) {
        self.pending_enc = Some(enc);
    }

    /// Whether the resident block stack fits the GPU prelude stash.
    pub fn gpu_stash_ok(&self) -> bool {
        self.blocks_n <= PRELUDE_MAX
    }

    /// Stash resident `x` + block stack (the prelude state at the first `CORE_START`)
    /// into GPU buffers, encoded on the open CB. No wait.
    pub fn stash_prelude_gpu(&mut self) -> Result<()> {
        let p_copy = self.gemv.pso("k_copy_f32", &[])?;
        let b_x = self.ensure_x_buf(None)?;
        if self.blocks_buf.is_none() {
            self.ensure_blocks_buf(&[])?;
        }
        let b_blocks = self.blocks_buf.as_ref().context("blocks_buf")?.clone();
        let n_rows = self.blocks_n;
        self.prelude_n = n_rows;
        let b_px = self.gemv_scratch_buf(1950, H * 4)?;
        let b_pb = self.gemv_scratch_buf(1951, PRELUDE_MAX * H * 4)?;
        let m_x = self.gemv_scratch_buf(1960, 4)?;
        let m_b = self.gemv_scratch_buf(1961, 4)?;
        write_buffer(&m_x, 0, &[H as u32]);
        write_buffer(&m_b, 0, &[(n_rows * H) as u32]);

        let cb = self.take_open_cb()?;
        {
            let enc = self.take_open_encoder(&cb)?;
            // Order against the previous visit's writes to x / blocks.
            Self::buf_barrier(&enc);
            // The two copies are independent (distinct src/dst).
            encode_copy(&enc, &p_copy, &b_x, 0, &b_px, 0, &m_x, H);
            if n_rows > 0 {
                encode_copy(&enc, &p_copy, &b_blocks, 0, &b_pb, 0, &m_b, n_rows * H);
            }
            // Consumers (later restore copies) are ordered by their own leading barrier.
            self.put_open_encoder(enc);
        }
        self.put_open_cb(cb);
        Ok(())
    }

    /// Whether a GPU reentry can handle the current block stack (bank fits the gather kernel).
    pub fn gpu_reentry_ok(&self) -> bool {
        let skip = self.prelude_n + 1;
        let n_bank = self.blocks_n.saturating_sub(skip) + 1;
        n_bank <= GPU_BANK_MAX && self.blocks_buf.is_some()
    }

    /// Core reentry entirely on the open CB (no wait):
    /// 1. rms-normed bank rows from completed core blocks + current `x`;
    /// 2. restore `x` / blocks from the prelude stash (`blocks_n = prelude_n`);
    /// 3. softmax-gather messages for all 12×4 core queries into a GPU buffer.
    ///
    /// `queries`: host `[N_Q * H]` (stable pointer; cached as a weight buffer).
    /// `reentry`: 0-based reentry index within the token (distinct staging slots).
    /// Returns the msgs buffer laid out `[12][4][H]` (offset `(layer-CORE_START)*4*H*4`).
    pub fn core_reentry_gpu(&mut self, queries: &[f32], reentry: usize) -> Result<Buffer> {
        assert_eq!(queries.len(), N_Q * H);
        assert!(self.gpu_reentry_ok());
        let r = reentry.min(3);
        let prelude_n = self.prelude_n;
        let skip = prelude_n + 1;
        let n_core = self.blocks_n.saturating_sub(skip);
        let n_bank = n_core + 1;

        let p_copy = self.gemv.pso("k_copy_f32", &[])?;
        let p_bank = self.gemv.pso("k_build_bank_t1", &[])?;
        let p_gather = self.gemv.pso("k_gather_messages_t1", &[])?;

        let b_x = self.ensure_x_buf(None)?;
        let b_blocks = self.blocks_buf.as_ref().context("blocks_buf")?.clone();
        let b_px = self.gemv_scratch_buf(1950, H * 4)?;
        let b_pb = self.gemv_scratch_buf(1951, PRELUDE_MAX * H * 4)?;
        let m_x = self.gemv_scratch_buf(1960, 4)?;
        let m_b = self.gemv_scratch_buf(1961, 4)?;
        let b_q = self.weight_buf(queries)?;
        let b_bank = self.gemv_scratch_buf(1980 + r as u16, GPU_BANK_MAX * H * 4)?;
        let b_msgs = self.gemv_scratch_buf(1990 + r as u16, N_Q * H * 4)?;
        let m_bank = self.gemv_scratch_buf(1970 + (r * 2) as u16, 16)?;
        let m_gather = self.gemv_scratch_buf(1971 + (r * 2) as u16, 16)?;
        write_buffer(&m_bank, 0, &[n_core as u32, H as u32, skip as u32, 0u32]);
        write_buffer(&m_gather, 0, &[n_bank as u32, H as u32, N_Q as u32, 0u32]);

        let cb = self.take_open_cb()?;
        {
            let enc = self.take_open_encoder(&cb)?;
            // Order against the previous visit's writes to x / blocks.
            Self::buf_barrier(&enc);
            // 1. bank = rms(core blocks) ++ rms(x)
            enc.setComputePipelineState(&p_bank);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(&b_blocks), 0, 0);
                enc.setBuffer_offset_atIndex(Some(&b_x), 0, 1);
                enc.setBuffer_offset_atIndex(Some(&b_bank), 0, 2);
                enc.setBuffer_offset_atIndex(Some(&m_bank), 0, 3);
                enc.setThreadgroupMemoryLength_atIndex(256 * 4, 0);
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(tg(n_bank), tg(256));
            // gather reads bank; restore copies overwrite x / blocks that bank read.
            Self::buf_barrier(&enc);
            // 2. messages for every core layer / slot
            enc.setComputePipelineState(&p_gather);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(&b_bank), 0, 0);
                enc.setBuffer_offset_atIndex(Some(&b_q), 0, 1);
                enc.setBuffer_offset_atIndex(Some(&b_msgs), 0, 2);
                enc.setBuffer_offset_atIndex(Some(&m_gather), 0, 3);
                enc.setThreadgroupMemoryLength_atIndex((256 + 32) * 4, 0);
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(tg(N_Q), tg(256));
            // 3. restore prelude x / blocks (bank already consumed the old ones; the
            //    copies are independent of gather, which reads only bank + queries).
            encode_copy(&enc, &p_copy, &b_px, 0, &b_x, 0, &m_x, H);
            if prelude_n > 0 {
                encode_copy(&enc, &p_copy, &b_pb, 0, &b_blocks, 0, &m_b, prelude_n * H);
            }
            self.put_open_encoder(enc);
        }
        self.put_open_cb(cb);
        self.blocks_n = prelude_n;
        self.bank_buf = Some(b_bank);
        self.bank_n = n_bank;
        Ok(b_msgs)
    }

    /// Final `attention_residual(blocks + x, output_query)` → RMSNorm(final_norm) →
    /// softcap unembed → argmax, all on the open CB; one commit+wait, one i32 read.
    ///
    /// `unembed` is the live `[out_dim * H]` f32 slice (same pointer as
    /// `unembed_metal_argmax`, so the f16 weight cache is shared).
    /// Returns `Ok(None)` (nothing encoded) when resident state can't be used.
    pub fn finalize_argmax_from_resident(
        &mut self,
        output_query: &[f32],
        final_norm: &[f32],
        unembed: &[f32],
        out_dim: usize,
        softcap: f32,
    ) -> Result<i32> {
        assert_eq!(output_query.len(), H);
        assert_eq!(final_norm.len(), H);
        assert_eq!(unembed.len(), out_dim * H);
        let n_states = self.blocks_n;
        anyhow::ensure!(
            self.x_buf.is_some() && self.blocks_buf.is_some() && n_states < MIX_MAX,
            "no resident decode state to finalize ({n_states} block states)"
        );
        let p_mix = self.gemv.pso("k_residual_mix_t1", &[])?;
        let p_rms = self.gemv.pso("k_rms_norm_affine", &[])?;
        let p_un = self.gemv.pso(
            "k_dense_gemv_reduce_f16w",
            &[
                (0, FcValue::U32(1)),
                (1, FcValue::U32(out_dim as u32)),
                (2, FcValue::U32(H as u32)),
                (3, FcValue::F32(softcap)),
            ],
        )?;
        let p_am = self.gemv.pso("k_argmax_partial", &[])?;
        let p_fin = self.gemv.pso("k_argmax_final", &[])?;

        let b_x = self.ensure_x_buf(None)?;
        let b_blocks = self.blocks_buf.as_ref().context("blocks_buf")?.clone();
        let b_oq = self.weight_buf(output_query)?;
        let b_fn = self.weight_buf(final_norm)?;
        let b_w = self.weight_buf_f16(unembed)?;
        // Opt-in lossy 8-bit unembed.
        let q8_un = if crate::flags::attn_q8() {
            Some(self.q8_from_f32(unembed, out_dim, H)?)
        } else {
            None
        };

        const TG: usize = 256;
        let n_tg = out_dim.div_ceil(TG);
        let b_mixed = self.gemv_scratch_buf(1800, H * 4)?;
        let b_hidden = self.gemv_scratch_buf(1801, H * 4)?;
        let b_mix_meta = self.gemv_scratch_buf(1802, 16)?;
        let b_rms_meta = self.gemv_scratch_buf(1803, 8)?;
        let b_dummy = self.gemv_scratch_buf(1804, H * 4)?;
        let b_y = self.gemv_scratch_buf(1805, out_dim * 4)?;
        let b_pidx = self.gemv_scratch_buf(1806, n_tg * 4)?;
        let b_pval = self.gemv_scratch_buf(1807, n_tg * 4)?;
        let b_n = self.gemv_scratch_buf(1808, 4)?;
        let b_out = self.gemv_scratch_buf(1809, 4)?;
        let b_np = self.gemv_scratch_buf(1810, 4)?;
        write_buffer(&b_mix_meta, 0, &[n_states as u32, H as u32, 1u32, 1u32]);
        write_buffer(&b_rms_meta, 0, &[H as u32, 1u32]);
        write_buffer(&b_n, 0, &[out_dim as u32]);
        write_buffer(&b_np, 0, &[n_tg as u32]);
        write_buffer(&b_out, 0, &[-1i32]);

        let cb = self.take_open_cb()?;
        {
            let enc = self.take_open_encoder(&cb)?;
            Self::buf_barrier(&enc);
            // mixed = softmax-attn over normalized(blocks ++ [x]) with output_query
            enc.setComputePipelineState(&p_mix);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(&b_blocks), 0, 0);
                enc.setBuffer_offset_atIndex(Some(&b_x), 0, 1);
                enc.setBuffer_offset_atIndex(Some(&b_oq), 0, 2);
                enc.setBuffer_offset_atIndex(Some(&b_mixed), 0, 3);
                enc.setBuffer_offset_atIndex(Some(&b_mix_meta), 0, 4);
                enc.setBuffer_offset_atIndex(Some(&b_dummy), 0, 5);
                enc.setThreadgroupMemoryLength_atIndex((32 + 256) * 4, 0);
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(tg(1), tg(256));
            Self::buf_barrier(&enc);
            // hidden = rms(mixed) * final_norm
            enc.setComputePipelineState(&p_rms);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(&b_mixed), 0, 0);
                enc.setBuffer_offset_atIndex(Some(&b_fn), 0, 1);
                enc.setBuffer_offset_atIndex(Some(&b_hidden), 0, 2);
                enc.setBuffer_offset_atIndex(Some(&b_rms_meta), 0, 3);
                enc.setThreadgroupMemoryLength_atIndex(256 * 4, 0);
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(tg(1), tg(256));
            Self::buf_barrier(&enc);
            // logits = softcap(W @ hidden)
            if let Some(ref q) = q8_un {
                let pad = out_dim.next_multiple_of(4);
                self.encode_q8(&enc, &b_hidden, q, &b_y, pad, H, softcap)?;
            } else {
                enc.setComputePipelineState(&p_un);
                unsafe {
                    enc.setBuffer_offset_atIndex(Some(&b_hidden), 0, 0);
                    enc.setBuffer_offset_atIndex(Some(&b_w), 0, 1);
                    enc.setBuffer_offset_atIndex(Some(&b_y), 0, 2);
                    enc.setThreadgroupMemoryLength_atIndex((TG / 32) * 4, 0);
                }
                enc.dispatchThreadgroups_threadsPerThreadgroup(tg(out_dim), tg(TG));
            }
            Self::buf_barrier(&enc);
            // argmax: per-TG partials → one final TG → i32 id
            enc.setComputePipelineState(&p_am);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(&b_y), 0, 0);
                enc.setBuffer_offset_atIndex(Some(&b_pidx), 0, 1);
                enc.setBuffer_offset_atIndex(Some(&b_pval), 0, 2);
                enc.setBuffer_offset_atIndex(Some(&b_n), 0, 3);
                enc.setThreadgroupMemoryLength_atIndex(TG * 4, 0);
                enc.setThreadgroupMemoryLength_atIndex(TG * 4, 1);
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(tg(n_tg), tg(TG));
            Self::buf_barrier(&enc);
            enc.setComputePipelineState(&p_fin);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(&b_pidx), 0, 0);
                enc.setBuffer_offset_atIndex(Some(&b_pval), 0, 1);
                enc.setBuffer_offset_atIndex(Some(&b_np), 0, 2);
                enc.setBuffer_offset_atIndex(Some(&b_out), 0, 3);
                enc.setThreadgroupMemoryLength_atIndex(TG * 4, 0);
                enc.setThreadgroupMemoryLength_atIndex(TG * 4, 1);
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(tg(1), tg(TG));
            enc.endEncoding();
        }
        cb.commit();
        cb.waitUntilCompleted();
        self.pending_cb = None;
        self.pending_committed = false;
        if crate::flags::profile_gpu() || crate::flags::profile_step() {
            let gpu_ms = cb_gpu_ms(&cb);
            if gpu_ms > 0.0 {
                crate::flags::GpuSplitAccum::with_mut(|s| s.add_seg_gpu(gpu_ms));
            }
        }
        let mut id = [0i32; 1];
        read_buffer(&b_out, 0, &mut id);
        Ok(id[0])
    }
}
