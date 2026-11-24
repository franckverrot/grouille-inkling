//! Metal ports of shared CPU ops (short-conv + dense GEMV).

use anyhow::{Context, Result};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLCommandBuffer, MTLCommandEncoder, MTLComputeCommandEncoder, MTLSize};
use std::collections::HashMap;

use crate::gpu::{read_buffer, write_buffer, Buffer, CommandBuffer, FcValue, Gpu, Pso};

/// Scratch slots of the stream decode keeps at the tapped visits, and of the final
/// normed state its head computes.
pub(crate) const TAP_SLOT: u16 = 1820;
pub(crate) const FINAL_STATE_SLOT: u16 = 1801;

/// Metal GPU duration in ms for a completed command buffer (`GPUEndTime - GPUStartTime`).
pub(crate) fn cb_gpu_ms(cb: &CommandBuffer) -> f64 {
    let start = cb.GPUStartTime();
    let end = cb.GPUEndTime();
    if end > start && start > 0.0 {
        (end - start) * 1000.0
    } else {
        0.0
    }
}

/// End encoder, commit+wait `cb`, record GPU/wall into `add`, return a fresh encoder.
pub(crate) fn profile_phase_restart(
    gemv: &Gpu,
    enc: Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>,
    cb: &mut CommandBuffer,
    add: impl FnOnce(f64, f64),
) -> Result<Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>> {
    enc.endEncoding();
    let t = std::time::Instant::now();
    cb.commit();
    cb.waitUntilCompleted();
    add(t.elapsed().as_secs_f64() * 1000.0, cb_gpu_ms(cb));
    *cb = gemv.command_buffer()?;
    crate::metal_resident::new_segment_encoder(cb)
}

const MSL_GEMV: &str = include_str!("../kernels/dense_gemv.metal");

/// Registered-f16 attn tensor ids for [`MetalOps::register_attn_f16`].
pub const ATTN_WQ: u8 = 0;
pub const ATTN_WK: u8 = 1;
pub const ATTN_WV: u8 = 2;
pub const ATTN_WR: u8 = 3;
pub const ATTN_WO: u8 = 4;
pub const ATTN_GATE: u8 = 5;
pub const ATTN_SINK: u8 = 6;
pub const ATTN_ROUTER: u8 = 7;
// q_scale / rel_scale / rel_proj: f16 Metal views (promote in-kernel).
pub const ATTN_QSCALE: u8 = 8;
pub const ATTN_RELSCALE: u8 = 9;
pub const ATTN_RELPROJ: u8 = 10;

pub struct MetalOps {
    pub(crate) gemv: Gpu,
    /// Host-weight pointer → Metal shared buffer (skip re-upload on decode).
    weight_bufs: HashMap<(usize, usize), Buffer>,
    /// Registered f16 attn/router weights from `.gpk` keyed `(layer, ATTN_*)`.
    /// Offset-0 shared buffers (f16 bytes copied once; no host f32 inflation).
    pub(crate) attn_f16: HashMap<(usize, u8), Buffer>,
    /// Meta buffers handed out to the prompt visit being encoded.
    pub(crate) prompt_meta_n: u16,
    /// The visit being recorded while a whole block goes into one command buffer.
    pub(crate) block_visit: Option<usize>,
    /// The vocabulary logits of a block's tokens, kept between blocks.
    pub(crate) block_logits: Option<Buffer>,
    /// While set, decode keeps the stream entering each tapped visit for the drafter.
    pub(crate) record_taps: bool,
    /// Opt-in 8-bit requantized weights (`INKLING_ATTN_Q8`), keyed by source buffer.
    pub(crate) q8: crate::metal_q8::Q8Cache,
    /// Reused x/y scratch keyed by `(slot, byte_len)`.
    gemv_scratch: HashMap<(u16, usize), Buffer>,
    /// Scratch for [`Self::dense_gemv_many_ex`] (decode fires this every attn visit).
    gemv_prepared: Vec<(Pso, Buffer, Buffer, Buffer, usize, usize, bool)>,
    gemv_x_bufs: Vec<(usize, usize, Buffer)>,
    pub(crate) attn_block_consts: bool,
    /// A visit's short-conv states on the GPU, in pairs, keyed by the address of the
    /// host state they were seeded from.
    pub(crate) conv_state_bufs: HashMap<usize, (Buffer, Buffer, usize)>,
    /// Decode-token residual stream `[HIDDEN]`: stays on GPU across visits.
    pub(crate) x_buf: Option<Buffer>,
    /// Pending command buffer for deferred wait (GPU-resident x segments).
    pub(crate) pending_cb: Option<crate::gpu::CommandBuffer>,
    /// Whether [`Self::pending_cb`] has already been `commit`ted.
    pub(crate) pending_committed: bool,
    /// Compute encoder left open on [`Self::pending_cb`] across fused visits / GPU bank
    /// reentry. One encoder per token segment keeps dispatches ordered even when scratch
    /// buffers are hazard-untracked (separate encoders would overlap and race).
    pub(crate) pending_enc: Option<Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>>,
    /// GPU-resident residual-block stack `[n_blocks * HIDDEN]` for fused visits.
    pub(crate) blocks_buf: Option<Buffer>,
    /// Rows currently valid in [`Self::blocks_buf`].
    pub(crate) blocks_n: usize,
    /// Row capacity of [`Self::blocks_buf`].
    pub(crate) blocks_cap: usize,
    /// Rows of the GPU prelude block stash (see `metal_resident`).
    pub(crate) prelude_n: usize,
    /// GPU message bank from the last core reentry + its row count.
    pub(crate) bank_buf: Option<Buffer>,
    pub(crate) bank_n: usize,
}

impl MetalOps {
    pub fn new() -> Result<Self> {
        let mut me = Self {
            gemv: {
                // The archive element type is a compile-time choice in the kernels.
                let f16 = if crate::flags::kv_f16() { "#define ARCH_F16 1\n" } else { "" };
                Gpu::new(&format!("{f16}{MSL_GEMV}")).context("compile dense_gemv.metal")?
            },
            weight_bufs: HashMap::new(),
            attn_f16: HashMap::new(),
            prompt_meta_n: 0,
            block_visit: None,
            block_logits: None,
            record_taps: false,
            q8: HashMap::new(),
            gemv_scratch: HashMap::new(),
            gemv_prepared: Vec::with_capacity(6),
            gemv_x_bufs: Vec::with_capacity(6),
            attn_block_consts: false,
            conv_state_bufs: HashMap::new(),
            x_buf: None,
            pending_cb: None,
            pending_committed: false,
            pending_enc: None,
            blocks_buf: None,
            blocks_n: 0,
            blocks_cap: 0,
            prelude_n: 0,
            bank_buf: None,
            bank_n: 0,
        };
        // Prefault T=1 decode GEMV scratch (6 jobs × x/y) so the first decode
        // token after a prompt does not allocate mid-step.
        {
            const H: usize = 2304;
            const Q_CH: usize = 24 * 128; // N_HEADS * DH
            const KV_CH: usize = 8 * 128;
            const D_REL: usize = 16; // matches attn::D_REL
            let xs = [
                H * 4, // q/k/v/r/sink/gate x is always H for T=1
                H * 4,
                H * 4,
                H * 4,
                H * 4,
                H * 4,
            ];
            let ys = [
                Q_CH * 4,
                KV_CH * 4,
                KV_CH * 4,
                24 * D_REL * 4, // r
                24 * 4,        // sinks
                Q_CH * 4,      // gate
            ];
            for (i, &len) in xs.iter().enumerate() {
                let _ = me.gemv_scratch_buf(i as u16, len)?;
            }
            for (i, &len) in ys.iter().enumerate() {
                let _ = me.gemv_scratch_buf(16 + i as u16, len)?;
            }
            // Unembed softcap Y (UNPADDED_VOCAB): first decode token after warm.
            let _ = me.gemv_scratch_buf(16, 200_058 * 4)?;
            let _ = me.gemv_scratch_buf(1805, 200_058 * 4)?; // GPU finalize Y
            for name in ["k_argmax_final", "k_argmax_partial", "k_build_bank_t1", "k_gather_messages_t1", "k_copy_f32"] {
                let _ = me.gemv.pso(name, &[])?;
            }
            // The prompt path: the multi-token GEMM in every shape it is used in
            // (projections, dynamic taps, router, message controllers), the per-token
            // kernels and the attention stages.
            for (out_dim, in_dim) in [
                (24 * 128, H),
                (8 * 128, H),
                (24 * 16, H),
                (24, H),
                (H, 24 * 128),
                (128, H),
                (8 * 4, H),
                (64, 2 * H),
                (H, 64),
            ] {
                let _ = me.gemv.pso(
                    "k_gemm_f16w_mma",
                    &[
                        (0, FcValue::U32(8)),
                        (1, FcValue::U32(out_dim as u32)),
                        (2, FcValue::U32(in_dim as u32)),
                        (3, FcValue::F32(0.0)),
                    ],
                )?;
            }
            for name in [
                "k_tok_mix",
                "k_tok_ctrl_prep",
                "k_tok_ctrl_apply",
                "k_tok_l2_clip",
                "k_tok_sconv_add",
                "k_tok_rms",
                "k_tok_gather",
            ] {
                let _ = me.gemv.pso(name, &[])?;
            }
            let _ = me.gemv.pso(
                "k_tok_kv_conv",
                &[(5, FcValue::U32(8)), (6, FcValue::U32(128))],
            )?;
            // Decode attention, for a short history and for a longer one.
            let attn = [
                (4, FcValue::U32(24)),
                (5, FcValue::U32(8)),
                (6, FcValue::U32(128)),
                (7, FcValue::U32(16)),
                (8, FcValue::U32(1024)),
            ];
            let _ = me.gemv.pso("k_decode_attn_small", &attn)?;
            let chunked = [attn.as_slice(), &[(10, FcValue::U32(crate::metal_glue::RING_CHUNK as u32))]].concat();
            for name in ["k_decode_ring_part", "k_decode_merge"] {
                let _ = me.gemv.pso(name, &chunked)?;
            }
            // What a full-attention visit needs once its history outgrows the ring.
            let geom = [
                (4, FcValue::U32(24)),
                (5, FcValue::U32(8)),
                (6, FcValue::U32(128)),
            ];
            let _ = me.gemv.pso("k_archive_row", &[])?;
            let _ = me.gemv.pso("k_decode_far_part", &geom)?;
            for name in ["k_attn_scores_far_mma", "k_attn_ctx_far_mma"] {
                let _ = me.gemv.pso(
                    name,
                    &[
                        (4, FcValue::U32(24)),
                        (5, FcValue::U32(8)),
                        (6, FcValue::U32(128)),
                        (7, FcValue::U32(16)),
                        (8, FcValue::U32(1024)),
                        (9, FcValue::U32(0)),
                    ],
                )?;
            }
            // A sliding visit has a 1024-token window, a full-attention one has none.
            for swa in [1024u32, 0] {
                for name in ["k_attn_scores_mma", "k_attn_softmax_rows", "k_attn_ctx_mma"] {
                    let _ = me.gemv.pso(
                        name,
                        &[
                            (4, FcValue::U32(24)),
                            (5, FcValue::U32(8)),
                            (6, FcValue::U32(128)),
                            (7, FcValue::U32(16)),
                            (8, FcValue::U32(1024)),
                            (9, FcValue::U32(swa)),
                        ],
                    )?;
                }
            }
            // Warm T=1 decode GEMV PSOs. Warm specializes TOKENS=prompt_len, so
            // the first decode token would otherwise compile these mid-step
            // (same idea as MetalMoe::warmup_decode_psos).
            let soft0 = 0.0f32;
            for &(out_dim, in_dim) in &[
                (Q_CH, H),       // q / gate
                (KV_CH, H),      // k / v
                (24 * D_REL, H), // r
                (24, H),         // sinks
                (H, Q_CH),       // wo
                (128, H),        // router
            ] {
                for name in [
                    "k_dense_gemv",
                    "k_dense_gemv_reduce",
                    "k_dense_gemv_reduce_f16w",
                    "k_dense_gemv_reduce_dual_f16w",
                    "k_dense_gemv_reduce_f16w_tgx",
                    "k_dense_gemv_reduce_dual_f16w_tgx",
                ] {
                    let _ = me.gemv.pso(
                        name,
                        &[
                            (0, FcValue::U32(1)),
                            (1, FcValue::U32(out_dim as u32)),
                            (2, FcValue::U32(in_dim as u32)),
                            (3, FcValue::F32(soft0)),
                        ],
                    )?;
                }
            }
            // Softcap unembed (LOGIT_SOFTCAP = 30); f16w halves W traffic.
            for name in ["k_dense_gemv_reduce_f16w", "k_dense_gemv_reduce_row"] {
                let _ = me.gemv.pso(
                    name,
                    &[
                        (0, FcValue::U32(1)),
                        (1, FcValue::U32(200_058)),
                        (2, FcValue::U32(H as u32)),
                        (3, FcValue::F32(30.0)),
                    ],
                )?;
            }
        }
        Ok(me)
    }

    /// Free the large scratch buffers prompt processing leaves behind.  Decode only
    /// uses small ones.  Call while the GPU is idle.
    pub fn release_prompt_scratch(&mut self) {
        const KEEP_BELOW: usize = 8 << 20;
        self.gemv_scratch.retain(|&(_, len), _| len < KEEP_BELOW);
    }

    /// The four states the drafter reads for the token decode just finished: the
    /// stream entering the tapped visits and the final normed state.  Decode has to
    /// have run with `record_taps` set.
    pub fn read_decode_taps(&mut self, out: &mut Vec<f32>) -> Result<()> {
        use crate::ops::HIDDEN;
        let mut row = vec![0f32; HIDDEN];
        for slot in [TAP_SLOT, TAP_SLOT + 1, TAP_SLOT + 2, FINAL_STATE_SLOT] {
            let b = self.gemv_scratch_buf(slot, HIDDEN * 4)?;
            read_buffer(&b, 0, &mut row);
            out.extend_from_slice(&row);
        }
        Ok(())
    }

    /// Bring every visit's conv state back from the GPU, where decode keeps it, and
    /// forget the GPU copies.  The prompt path reads and writes the host state, and
    /// decode seeds itself from the host again afterwards.
    pub fn conv_states_to_host(&mut self, caches: &mut crate::cache::DecodeCaches) -> Result<()> {
        self.flush_pending()?;
        for v in &mut caches.visits {
            // The fused visits key their state buffers by the host state's address.
            let key = v.k_sconv.as_ptr() as usize;
            if let Some((bk, bv, _)) = self.conv_state_bufs.get(&(key ^ 0xC0FFEE)) {
                read_buffer(bk, 0, &mut v.k_sconv);
                read_buffer(bv, 0, &mut v.v_sconv);
            }
            if let Some((ba, bm, _)) = self.conv_state_bufs.get(&(key ^ 0xADD1_EA57)) {
                read_buffer(ba, 0, &mut v.attn_sconv);
                read_buffer(bm, 0, &mut v.mlp_sconv);
            }
        }
        self.clear_decode_state();
        Ok(())
    }

    /// Drop what decode keeps on the GPU between tokens: conv states, the stream, the
    /// block stack and the bank.  Call when a new prompt replaces the caches.
    pub fn clear_decode_state(&mut self) {
        self.conv_state_bufs.clear();
        self.x_buf = None;
        if let Some(enc) = self.pending_enc.take() {
            enc.endEncoding();
        }
        self.pending_cb = None;
        self.pending_committed = false;
        self.blocks_buf = None;
        self.blocks_n = 0;
        self.blocks_cap = 0;
        self.prelude_n = 0;
        self.bank_buf = None;
        self.bank_n = 0;
    }

    /// Ensure decode `x` buffer exists; optionally seed from host.
    pub fn ensure_x_buf(&mut self, host: Option<&[f32]>) -> Result<Buffer> {
        const H: usize = 2304;
        if self.x_buf.is_none() {
            self.x_buf = Some(self.gemv.buffer_shared(H * 4)?);
        }
        let b = self.x_buf.as_ref().unwrap().clone();
        if let Some(x) = host {
            assert_eq!(x.len(), H);
            write_buffer(&b, 0, x);
        }
        Ok(b)
    }

    /// Seed / replace GPU block stack from host `[n * HIDDEN]`.
    pub fn ensure_blocks_buf(&mut self, host: &[f32]) -> Result<Buffer> {
        const H: usize = 2304;
        assert_eq!(host.len() % H, 0);
        let n = host.len() / H;
        // Cap for a full decode token: prelude + core + boundary appends.
        let need_cap = n.max(32).next_power_of_two().max(32);
        if self.blocks_buf.is_none() || self.blocks_cap < need_cap {
            self.blocks_buf = Some(self.gemv.buffer_shared(need_cap * H * 4)?);
            self.blocks_cap = need_cap;
        }
        let b = self.blocks_buf.as_ref().unwrap().clone();
        if n > 0 {
            write_buffer(&b, 0, host);
        }
        self.blocks_n = n;
        Ok(b)
    }

    /// Read GPU block stack into host vec (after flush).
    pub fn sync_blocks(&mut self, out: &mut Vec<f32>) -> Result<()> {
        const H: usize = 2304;
        out.resize(self.blocks_n * H, 0.0);
        if self.blocks_n == 0 {
            return Ok(());
        }
        let b = self
            .blocks_buf
            .as_ref()
            .context("blocks_buf missing")?
            .clone();
        read_buffer(&b, 0, out);
        Ok(())
    }

    /// Commit+wait any open segment CB (no `x` readback).
    pub fn flush_pending(&mut self) -> Result<()> {
        // Close the segment-wide encoder before commit.
        if let Some(enc) = self.pending_enc.take() {
            enc.endEncoding();
        }
        if let Some(cb) = self.pending_cb.take() {
            if !self.pending_committed {
                cb.commit();
            }
            cb.waitUntilCompleted();
            if crate::flags::profile_gpu() || crate::flags::profile_step() {
                let gpu_ms = cb_gpu_ms(&cb);
                if gpu_ms > 0.0 {
                    crate::flags::GpuSplitAccum::with_mut(|s| s.add_seg_gpu(gpu_ms));
                }
            }
        }
        self.pending_committed = false;
        Ok(())
    }

    /// Wait any deferred segment CB and read `x` into host.
    pub fn sync_x(&mut self, out: &mut [f32]) -> Result<()> {
        self.flush_pending()?;
        let b = self.ensure_x_buf(None)?;
        read_buffer(&b, 0, out);
        Ok(())
    }

    /// Flush, read `x` and block stack (bank rebuild / token barriers).
    pub fn sync_x_and_blocks(&mut self, x: &mut [f32], blocks: &mut Vec<f32>) -> Result<()> {
        self.sync_x(x)?;
        self.sync_blocks(blocks)?;
        Ok(())
    }

    pub(crate) fn gemv_scratch_buf(&mut self, slot: u16, len: usize) -> Result<Buffer> {
        // Pow2 buckets so multi-token warm GEMVs reuse across nearby T.
        let len = len.max(16).next_power_of_two();
        let key = (slot, len);
        if let Some(b) = self.gemv_scratch.get(&key) {
            return Ok(b.clone());
        }
        let b = self.gemv.buffer_shared(len)?;
        self.gemv_scratch.insert(key, b.clone());
        Ok(b)
    }

    pub(crate) fn weight_buf(&mut self, w: &[f32]) -> Result<Buffer> {
        let key = (w.as_ptr() as usize, w.len());
        if let Some(b) = self.weight_bufs.get(&key) {
            return Ok(b.clone());
        }
        let b = self.gemv.buffer_shared_readonly(w.len() * 4)?;
        write_buffer(&b, 0, w);
        self.weight_bufs.insert(key, b.clone());
        Ok(b)
    }

    /// f16 weight buffer for attn proj GEMVs (keyed separately from f32).
    pub(crate) fn weight_buf_f16(&mut self, w: &[f32]) -> Result<Buffer> {
        // Distinguish from f32 cache entries with a high bit on the length key.
        let key = (w.as_ptr() as usize, w.len() | (1usize << 63));
        if let Some(b) = self.weight_bufs.get(&key) {
            return Ok(b.clone());
        }
        // Converted straight into the buffer.  The unembed is 460M values, so a large
        // matrix is split over a few threads.
        let b = self.gemv.buffer_shared_readonly(w.len() * 2)?;
        const PER_THREAD: usize = 1 << 22;
        if w.len() < 2 * PER_THREAD {
            crate::gpu::write_buffer_f16(&b, 0, w);
        } else {
            let threads = std::thread::available_parallelism().map_or(4, |n| n.get()).min(8);
            let per = w.len().div_ceil(threads);
            // A buffer handle cannot cross threads; its memory can.
            let base = objc2_metal::MTLBuffer::contents(&*b).as_ptr() as usize;
            std::thread::scope(|scope| {
                for (i, part) in w.chunks(per).enumerate() {
                    scope.spawn(move || {
                        use half::slice::HalfFloatSliceExt;
                        let dst = (base + i * per * 2) as *mut half::f16;
                        unsafe { std::slice::from_raw_parts_mut(dst, part.len()) }
                            .convert_from_f32_slice(part);
                    });
                }
            });
        }
        self.weight_bufs.insert(key, b.clone());
        Ok(b)
    }

    /// Copy IEEE f16 bytes into a shared Metal buffer (offset 0).
    pub fn attn_f16_buffer_from_bytes(&mut self, bytes: &[u8]) -> Result<Buffer> {
        let b = self.gemv.buffer_shared_readonly(bytes.len().max(16))?;
        write_buffer(&b, 0, bytes);
        Ok(b)
    }

    /// Register f16 attn/router weights for `(layer, id)` (memcpy once from the gpk mmap).
    pub fn register_attn_f16(&mut self, layer: usize, id: u8, bytes: &[u8]) -> Result<()> {
        let b = self.attn_f16_buffer_from_bytes(bytes)?;
        self.attn_f16.insert((layer, id), b);
        Ok(())
    }

    /// Register an already-uploaded buffer (share one copy across several `MetalOps`).
    pub fn register_attn_f16_buffer(&mut self, layer: usize, id: u8, buf: Buffer) {
        self.attn_f16.insert((layer, id), buf);
    }

    /// Registered f16 buffer for `(layer, id)`, else convert `fallback` (host f32) once.
    pub(crate) fn attn_f16_or_convert(
        &mut self,
        layer: Option<usize>,
        id: u8,
        fallback: &[f32],
    ) -> Result<Buffer> {
        if let Some(l) = layer {
            if let Some(b) = self.attn_f16.get(&(l, id)) {
                return Ok(b.clone());
            }
        }
        if fallback.is_empty() {
            anyhow::bail!(
                "attn f16 weight (layer={layer:?}, id={id}) not registered and host copy is empty"
            );
        }
        self.weight_buf_f16(fallback)
    }

    /// Upload `w` as f16 (unembed / attn-proj bandwidth path).
    pub fn preload_weight_f16(&mut self, w: &[f32]) -> Result<()> {
        let _ = self.weight_buf_f16(w)?;
        Ok(())
    }

    pub fn device_name(&self) -> String {
        self.gemv.name()
    }

    /// Batched dense GEMV: `y[t] = W @ x[t]` for each token.
    /// `x`: `[tokens * in_dim]`, `w`: `[out_dim * in_dim]` row-major, `y`: `[tokens * out_dim]`.
    pub fn dense_gemv(
        &mut self,
        x: &[f32],
        w: &[f32],
        tokens: usize,
        out_dim: usize,
        in_dim: usize,
        y: &mut [f32],
    ) -> Result<()> {
        self.dense_gemv_many(&[(x, w, tokens, out_dim, in_dim)], &mut [y])
    }

    /// Like [`dense_gemv`] but applies `softcap * tanh(y / softcap)` on the GPU.
    pub fn dense_gemv_softcap(
        &mut self,
        x: &[f32],
        w: &[f32],
        tokens: usize,
        out_dim: usize,
        in_dim: usize,
        softcap: f32,
        y: &mut [f32],
    ) -> Result<()> {
        if tokens == 1 && crate::flags::attn_q8() {
            return self.q8_gemv_softcap_host(x, w, out_dim, in_dim, softcap, y);
        }
        self.dense_gemv_many_ex(
            &[(x, w, tokens, out_dim, in_dim)],
            &mut [y],
            softcap,
        )
    }

    /// Several GEMVs on one command buffer / one host sync.
    ///
    /// Each job is `(x, w, tokens, out_dim, in_dim)`; `ys[i]` receives job `i`.
    /// Y scratch buffers are keyed by slot: callers must not reuse the same
    /// job index for distinct outputs.
    pub fn dense_gemv_many(
        &mut self,
        jobs: &[(&[f32], &[f32], usize, usize, usize)],
        ys: &mut [&mut [f32]],
    ) -> Result<()> {
        self.dense_gemv_many_ex(jobs, ys, 0.0)
    }

    fn dense_gemv_many_ex(
        &mut self,
        jobs: &[(&[f32], &[f32], usize, usize, usize)],
        ys: &mut [&mut [f32]],
        softcap: f32,
    ) -> Result<()> {
        assert_eq!(jobs.len(), ys.len());
        assert!(!jobs.is_empty());
        self.gemv_prepared.clear();
        self.gemv_x_bufs.clear();
        // T=1 decode: threadgroup-reduce GEMV. Prefer 4-row blocking when OUT%4==0.
        let use_reduce = jobs.iter().all(|&(_, _, tokens, _, _)| tokens == 1);
        // Decode packs 4-6 GEMVs that share the same x pointer; upload once.
        for (i, &(x, w, tokens, out_dim, in_dim)) in jobs.iter().enumerate() {
            assert_eq!(x.len(), tokens * in_dim);
            assert_eq!(ys[i].len(), tokens * out_dim);
            assert_eq!(w.len(), out_dim * in_dim);
            // Multi-token work goes through the simdgroup-matrix GEMM (f16 weights).
            let use_mma =
                !use_reduce && softcap == 0.0 && out_dim % 8 == 0 && in_dim % 8 == 0;
            // Softcap decode unembed: f16 W (~2× less bandwidth). Plain GEMVs stay f32.
            let use_f16w = use_mma || (use_reduce && softcap > 0.0);
            let kernel = if use_mma {
                "k_gemm_f16w_mma"
            } else if !use_reduce {
                "k_dense_gemv"
            } else if use_f16w {
                "k_dense_gemv_reduce_f16w"
            } else if out_dim % 4 == 0 {
                "k_dense_gemv_reduce"
            } else {
                "k_dense_gemv_reduce_row"
            };
            // The mma kernel ignores TOKENS; a fixed value keeps one pipeline per shape
            // instead of one per prompt length.
            let pso = self.gemv.pso(
                kernel,
                &[
                    (0, FcValue::U32(if use_mma { 8 } else { tokens as u32 })),
                    (1, FcValue::U32(out_dim as u32)),
                    (2, FcValue::U32(in_dim as u32)),
                    (3, FcValue::F32(softcap)),
                ],
            )?;
            let b_w = if use_f16w {
                self.weight_buf_f16(w)?
            } else {
                self.weight_buf(w)?
            };
            let x_key = (x.as_ptr() as usize, x.len());
            let b_x = if let Some((_, _, b)) = self
                .gemv_x_bufs
                .iter()
                .find(|(p, n, _)| *p == x_key.0 && *n == x_key.1)
            {
                b.clone()
            } else {
                // The GEMM kernel works on 32-token tiles: leave room for the last one.
                let b = self.gemv_scratch_buf(i as u16, tokens.next_multiple_of(32) * in_dim * 4)?;
                write_buffer(&b, 0, x);
                self.gemv_x_bufs.push((x_key.0, x_key.1, b.clone()));
                b
            };
            let b_y =
                self.gemv_scratch_buf(16 + i as u16, tokens.next_multiple_of(32) * out_dim * 4)?;
            self.gemv_prepared
                .push((pso, b_x, b_w, b_y, tokens, out_dim, use_mma));
        }

        let cb = self.gemv.command_buffer()?;
        {
            let enc = cb.computeCommandEncoder().context("encoder")?;
            for (pso, b_x, b_w, b_y, tokens, out_dim, use_mma) in &self.gemv_prepared {
                enc.setComputePipelineState(pso);
                unsafe {
                    enc.setBuffer_offset_atIndex(Some(b_x), 0, 0);
                    enc.setBuffer_offset_atIndex(Some(b_w), 0, 1);
                    enc.setBuffer_offset_atIndex(Some(b_y), 0, 2);
                }
                if *use_mma {
                    enc.dispatchThreadgroups_threadsPerThreadgroup(
                        MTLSize {
                            width: *out_dim / 8,
                            height: tokens.div_ceil(32),
                            depth: 1,
                        },
                        MTLSize {
                            width: 32,
                            height: 1,
                            depth: 1,
                        },
                    );
                } else if use_reduce {
                    const TG_N: usize = 256;
                    // One TG per output row (simd reduce); tg mem = n_simd floats.
                    let tg_mem = (TG_N / 32) * 4;
                    unsafe { enc.setThreadgroupMemoryLength_atIndex(tg_mem, 0) };
                    enc.dispatchThreadgroups_threadsPerThreadgroup(
                        MTLSize {
                            width: *out_dim,
                            height: 1,
                            depth: 1,
                        },
                        MTLSize {
                            width: TG_N,
                            height: 1,
                            depth: 1,
                        },
                    );
                } else {
                    let tg_x = 8usize.min(*tokens).max(1);
                    let tg_y = 32usize.min(*out_dim).max(1);
                    enc.dispatchThreadgroups_threadsPerThreadgroup(
                        MTLSize {
                            width: tokens.div_ceil(tg_x),
                            height: out_dim.div_ceil(tg_y),
                            depth: 1,
                        },
                        MTLSize {
                            width: tg_x,
                            height: tg_y,
                            depth: 1,
                        },
                    );
                }
            }
            enc.endEncoding();
        }
        cb.commit();
        cb.waitUntilCompleted();
        for (i, (_, _, _, b_y, _, _, _)) in self.gemv_prepared.iter().enumerate() {
            read_buffer(b_y, 0, ys[i]);
        }
        Ok(())
    }

}
