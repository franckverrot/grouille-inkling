//! The drafter's matrix products and vocabulary head on the GPU.

use anyhow::{Context, Result};
use objc2_metal::{MTLCommandBuffer, MTLCommandEncoder, MTLComputeCommandEncoder, MTLSize};

use crate::gpu::{read_buffer, write_buffer, write_buffer_f16, Buffer};
use crate::head::UNPADDED_VOCAB;
use crate::metal_ops::MetalOps;
use crate::ops::HIDDEN;

/// A weight matrix on the GPU as f16, `[out_dim][in_dim]`.
pub struct DraftMat {
    buf: Buffer,
    pub out_dim: usize,
    pub in_dim: usize,
}

/// Vocabulary rows the GPU head covers, a multiple of 8.  The few rows after them
/// are the caller's.
pub const HEAD_ROWS: usize = UNPADDED_VOCAB / 8 * 8;
/// Positions a block can ask logits for.
const HEAD_BLOCK: usize = 8;
const PICK_TG: usize = 256;

/// The drafter's head: the folded unembed, and the logits of a block's positions.
pub struct DraftHead {
    fold: Buffer,
    width: usize,
    bias: Buffer,
    base: Buffer,
    states: Buffer,
    latent: Buffer,
    part: Buffer,
    meta: Buffer,
}

/// What one head step found over the rows the GPU covers.
pub struct Pick {
    pub row: usize,
    /// The best capped logit and the sum of `exp(capped)` over the rows.
    pub top: f32,
    pub sum_exp: f32,
}

impl MetalOps {
    /// Put a weight matrix on the GPU as f16.
    pub fn draft_mat(&mut self, w: &[f32], out_dim: usize, in_dim: usize) -> Result<DraftMat> {
        assert_eq!(w.len(), out_dim * in_dim);
        assert!(out_dim.is_multiple_of(8) && in_dim.is_multiple_of(8));
        let buf = self.gemv.buffer_shared(w.len() * 2)?;
        write_buffer_f16(&buf, 0, w);
        Ok(DraftMat { buf, out_dim, in_dim })
    }

    /// `x`, `[rows][in]`, through matrices that share it as input: `[rows][out]` each.
    pub fn draft_linear(&mut self, x: &[f32], rows: usize, mats: &[&DraftMat]) -> Result<Vec<Vec<f32>>> {
        let in_dim = mats[0].in_dim;
        assert_eq!(x.len(), rows * in_dim);
        let rp = rows.next_multiple_of(if rows <= 8 { 8 } else { 32 });
        self.flush_pending()?;
        let b_x = self.gemv_scratch_buf(7000, rp * in_dim * 4)?;
        write_buffer(&b_x, 0, x);
        let mut outs = Vec::with_capacity(mats.len());
        for (i, m) in mats.iter().enumerate() {
            assert_eq!(m.in_dim, in_dim);
            outs.push(self.gemv_scratch_buf(7001 + i as u16, rp * m.out_dim * 4)?);
        }
        let cb = self.gemv.command_buffer()?;
        {
            let enc = cb.computeCommandEncoder().context("encoder")?;
            for (m, y) in mats.iter().zip(&outs) {
                self.enc_gemm(&enc, &b_x, &m.buf, y, rows, m.out_dim, in_dim)?;
            }
            enc.endEncoding();
        }
        cb.commit();
        cb.waitUntilCompleted();
        Ok(mats
            .iter()
            .zip(&outs)
            .map(|(m, y)| {
                let mut v = vec![0f32; rows * m.out_dim];
                read_buffer(y, 0, &mut v);
                v
            })
            .collect())
    }

    /// `down(silu(gate(h)) * up(h))` for `h`, `[rows][in]`.
    pub fn draft_mlp(
        &mut self,
        h: &[f32],
        rows: usize,
        gate: &DraftMat,
        up: &DraftMat,
        down: &DraftMat,
    ) -> Result<Vec<f32>> {
        let (in_dim, mid) = (gate.in_dim, gate.out_dim);
        assert_eq!(h.len(), rows * in_dim);
        assert!(up.out_dim == mid && down.in_dim == mid);
        let rp = rows.next_multiple_of(if rows <= 8 { 8 } else { 32 });
        self.flush_pending()?;
        let b_x = self.gemv_scratch_buf(7000, rp * in_dim * 4)?;
        let b_gate = self.gemv_scratch_buf(7008, rp * mid * 4)?;
        let b_up = self.gemv_scratch_buf(7009, rp * mid * 4)?;
        let b_out = self.gemv_scratch_buf(7010, rp * down.out_dim * 4)?;
        let b_meta = self.gemv_scratch_buf(7011, 16)?;
        write_buffer(&b_x, 0, h);
        write_buffer(&b_meta, 0, &[(rows * mid) as u32]);
        let silu = self.gemv.pso("k_silu_mul", &[])?;
        let cb = self.gemv.command_buffer()?;
        {
            let enc = cb.computeCommandEncoder().context("encoder")?;
            self.enc_gemm(&enc, &b_x, &gate.buf, &b_gate, rows, mid, in_dim)?;
            self.enc_gemm(&enc, &b_x, &up.buf, &b_up, rows, mid, in_dim)?;
            enc.setComputePipelineState(&silu);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(&b_gate), 0, 0);
                enc.setBuffer_offset_atIndex(Some(&b_up), 0, 1);
                enc.setBuffer_offset_atIndex(Some(&b_meta), 0, 2);
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize { width: (rows * mid).div_ceil(1024), height: 1, depth: 1 },
                MTLSize { width: 1024, height: 1, depth: 1 },
            );
            self.enc_gemm(&enc, &b_gate, &down.buf, &b_out, rows, down.out_dim, mid)?;
            enc.endEncoding();
        }
        cb.commit();
        cb.waitUntilCompleted();
        let mut out = vec![0f32; rows * down.out_dim];
        read_buffer(&b_out, 0, &mut out);
        Ok(out)
    }

    /// The head's GPU side.  `fold` is `[HEAD_ROWS][width]` and `bias` `[HEAD_ROWS]`.
    pub fn draft_head(&mut self, fold: &[f32], bias: &[f32], width: usize) -> Result<DraftHead> {
        assert_eq!(fold.len(), HEAD_ROWS * width);
        assert_eq!(bias.len(), HEAD_ROWS);
        assert!(width.is_multiple_of(4));
        let g = &self.gemv;
        let head = DraftHead {
            fold: g.buffer_shared(fold.len() * 2)?,
            width,
            bias: g.buffer_shared(bias.len() * 4)?,
            base: g.buffer_shared(HEAD_BLOCK * HEAD_ROWS * 4)?,
            states: g.buffer_shared(HEAD_BLOCK * HIDDEN * 4)?,
            latent: g.buffer_shared(width * 4)?,
            part: g.buffer_shared(HEAD_ROWS.div_ceil(PICK_TG) * 4 * 4)?,
            meta: g.buffer_shared(16)?,
        };
        write_buffer_f16(&head.fold, 0, fold);
        write_buffer(&head.bias, 0, bias);
        write_buffer(&head.meta, 0, &[HEAD_ROWS as u32, width as u32]);
        Ok(head)
    }

    /// Vocabulary logits of `rows` states, `[rows][H]`, before the head's correction.
    /// They stay on the GPU for [`Self::draft_pick`].  `unembed` is the target's
    /// live unembed rows.
    pub fn draft_base(&mut self, head: &DraftHead, states: &[f32], rows: usize, unembed: &[f32]) -> Result<()> {
        assert!(rows <= HEAD_BLOCK && states.len() == rows * HIDDEN);
        self.flush_pending()?;
        let w = self.weight_buf_f16(unembed)?;
        write_buffer(&head.states, 0, states);
        let cb = self.gemv.command_buffer()?;
        {
            let enc = cb.computeCommandEncoder().context("encoder")?;
            self.enc_gemm(&enc, &head.states, &w, &head.base, rows, HEAD_ROWS, HIDDEN)?;
            enc.endEncoding();
        }
        cb.commit();
        cb.waitUntilCompleted();
        Ok(())
    }

    /// One head step for position `row` of the block: the best of the GPU's rows.
    pub fn draft_pick(&mut self, head: &DraftHead, row: usize, latent: &[f32]) -> Result<Pick> {
        assert!(row < HEAD_BLOCK && latent.len() == head.width);
        write_buffer(&head.latent, 0, latent);
        let pso = self.gemv.pso("k_draft_pick", &[])?;
        let groups = HEAD_ROWS.div_ceil(PICK_TG);
        let cb = self.gemv.command_buffer()?;
        {
            let enc = cb.computeCommandEncoder().context("encoder")?;
            enc.setComputePipelineState(&pso);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(&head.base), row * HEAD_ROWS * 4, 0);
                enc.setBuffer_offset_atIndex(Some(&head.bias), 0, 1);
                enc.setBuffer_offset_atIndex(Some(&head.fold), 0, 2);
                enc.setBuffer_offset_atIndex(Some(&head.latent), 0, 3);
                enc.setBuffer_offset_atIndex(Some(&head.part), 0, 4);
                enc.setBuffer_offset_atIndex(Some(&head.meta), 0, 5);
                enc.setThreadgroupMemoryLength_atIndex(32 * 4, 0);
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize { width: groups, height: 1, depth: 1 },
                MTLSize { width: PICK_TG, height: 1, depth: 1 },
            );
            enc.endEncoding();
        }
        cb.commit();
        cb.waitUntilCompleted();
        let mut part = vec![0f32; groups * 4];
        read_buffer(&head.part, 0, &mut part);
        let mut pick = Pick { row: 0, top: f32::MIN, sum_exp: 0.0 };
        for p in part.chunks_exact(4) {
            if p[0] > pick.top {
                pick.top = p[0];
                pick.row = p[1] as usize;
            }
            pick.sum_exp += p[2];
        }
        Ok(pick)
    }
}
