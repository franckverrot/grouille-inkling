//! Opt-in 8-bit attention projections and unembed (`INKLING_ATTN_Q8=1`).
//!
//! Lossy. Each row of an f16 weight is requantized to int8 with its own scale the
//! first time it is used, which halves the bytes those matrices stream per token.
//! The default path does not use any of this; measure with spike/ref_check.py
//! before relying on it. A native pack entry would avoid keeping the f16 copy too.

use std::collections::HashMap;

use anyhow::Result;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLComputeCommandEncoder, MTLSize};

use crate::gpu::{write_buffer, Buffer, FcValue};
use crate::metal_ops::MetalOps;

type Enc = ProtocolObject<dyn MTLComputeCommandEncoder>;

/// int8 weights `[out, in]` and f32 scale `[out]`.
pub(crate) type Q8 = (Buffer, Buffer);

pub(crate) type Q8Cache = HashMap<usize, Q8>;

fn quantize<T: Copy + Sync>(
    src: &[T],
    out_dim: usize,
    in_dim: usize,
    to_f32: fn(T) -> f32,
) -> (Vec<i8>, Vec<f32>) {
    let mut q = vec![0i8; out_dim * in_dim];
    let mut scale = vec![0f32; out_dim];
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .min(8);
    let rows_per = out_dim.div_ceil(threads).max(1);
    std::thread::scope(|s| {
        for ((qc, sc), src_c) in q
            .chunks_mut(rows_per * in_dim)
            .zip(scale.chunks_mut(rows_per))
            .zip(src.chunks(rows_per * in_dim))
        {
            s.spawn(move || {
                let mut row = vec![0f32; in_dim];
                for (r, srow) in src_c.chunks(in_dim).enumerate() {
                    let mut m = 0f32;
                    for (d, &v) in row.iter_mut().zip(srow) {
                        *d = to_f32(v);
                        m = m.max(d.abs());
                    }
                    let sc_r = if m > 0.0 { m / 127.0 } else { 1.0 };
                    let inv = 1.0 / sc_r;
                    for (d, &v) in qc[r * in_dim..(r + 1) * in_dim].iter_mut().zip(&row) {
                        *d = (v * inv).round().clamp(-127.0, 127.0) as i8;
                    }
                    sc[r] = sc_r;
                }
            });
        }
    });
    (q, scale)
}

fn grid(rows: usize) -> (MTLSize, MTLSize) {
    (
        MTLSize {
            width: rows.div_ceil(32),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 32,
            height: 8,
            depth: 1,
        },
    )
}

impl MetalOps {
    fn q8_upload(&mut self, key: usize, q: &[i8], scale: &[f32]) -> Result<Q8> {
        let bq = self.gemv.buffer_shared_readonly(q.len())?;
        write_buffer(&bq, 0, q);
        let bs = self.gemv.buffer_shared_readonly(scale.len() * 4)?;
        write_buffer(&bs, 0, scale);
        self.q8.insert(key, (bq.clone(), bs.clone()));
        Ok((bq, bs))
    }

    /// 8-bit view of a registered (or converted) f16 attention weight.
    pub(crate) fn attn_q8(
        &mut self,
        layer: Option<usize>,
        id: u8,
        fallback: &[f32],
        out_dim: usize,
        in_dim: usize,
    ) -> Result<Q8> {
        let src = self.attn_f16_or_convert(layer, id, fallback)?;
        let key = Retained::as_ptr(&src) as *const () as usize;
        if let Some(q) = self.q8.get(&key) {
            return Ok(q.clone());
        }
        let n = out_dim * in_dim;
        assert!(src.length() >= n * 2);
        let bits: &[u16] =
            unsafe { std::slice::from_raw_parts(src.contents().as_ptr() as *const u16, n) };
        let (q, scale) = quantize(bits, out_dim, in_dim, |b| half::f16::from_bits(b).to_f32());
        self.q8_upload(key, &q, &scale)
    }

    /// 8-bit view of a host f32 weight (unembed). Rows are zero-padded to a multiple
    /// of 4 so the 4-rows-per-simdgroup kernel never reads past the buffer.
    pub(crate) fn q8_from_f32(&mut self, w: &[f32], out_dim: usize, in_dim: usize) -> Result<Q8> {
        let key = (w.as_ptr() as usize) ^ (1usize << 62);
        if let Some(q) = self.q8.get(&key) {
            return Ok(q.clone());
        }
        assert_eq!(w.len(), out_dim * in_dim);
        let (mut q, mut scale) = quantize(w, out_dim, in_dim, |v| v);
        let pad = out_dim.next_multiple_of(4);
        q.resize(pad * in_dim, 0);
        scale.resize(pad, 1.0);
        self.q8_upload(key, &q, &scale)
    }

    /// Host-in, host-out softcapped 8-bit GEMV (the full-logits unembed path).
    pub(crate) fn q8_gemv_softcap_host(
        &mut self,
        x: &[f32],
        w: &[f32],
        out_dim: usize,
        in_dim: usize,
        softcap: f32,
        out: &mut [f32],
    ) -> Result<()> {
        use objc2_metal::{MTLCommandBuffer, MTLCommandEncoder};
        let q = self.q8_from_f32(w, out_dim, in_dim)?;
        let pad = out_dim.next_multiple_of(4);
        let b_x = self.gemv_scratch_buf(3500, in_dim * 4)?;
        let b_y = self.gemv_scratch_buf(3501, pad * 4)?;
        write_buffer(&b_x, 0, x);
        self.flush_pending()?;
        let cb = self.gemv.command_buffer()?;
        {
            let enc = cb
                .computeCommandEncoder()
                .ok_or_else(|| anyhow::anyhow!("encoder"))?;
            self.encode_q8(&enc, &b_x, &q, &b_y, pad, in_dim, softcap)?;
            enc.endEncoding();
        }
        cb.commit();
        cb.waitUntilCompleted();
        crate::gpu::read_buffer(&b_y, 0, &mut out[..out_dim]);
        Ok(())
    }

    /// `y = softcap(scale * (W8 x))`. No trailing barrier.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_q8(
        &mut self,
        enc: &Enc,
        x: &Buffer,
        w: &Q8,
        y: &Buffer,
        out_dim: usize,
        in_dim: usize,
        softcap: f32,
    ) -> Result<()> {
        assert!(out_dim.is_multiple_of(4) && in_dim.is_multiple_of(16));
        let pso = self.gemv.pso(
            "k_dense_gemv_q8",
            &[
                (1, FcValue::U32(out_dim as u32)),
                (2, FcValue::U32(in_dim as u32)),
                (3, FcValue::F32(softcap)),
            ],
        )?;
        enc.setComputePipelineState(&pso);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(x), 0, 0);
            enc.setBuffer_offset_atIndex(Some(&w.0), 0, 1);
            enc.setBuffer_offset_atIndex(Some(&w.1), 0, 2);
            enc.setBuffer_offset_atIndex(Some(y), 0, 3);
        }
        let (groups, threads) = grid(out_dim);
        enc.dispatchThreadgroups_threadsPerThreadgroup(groups, threads);
        Ok(())
    }

    /// Two 8-bit GEMVs over the same `x` (q + gate). No trailing barrier.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_q8_dual(
        &mut self,
        enc: &Enc,
        x: &Buffer,
        w0: &Q8,
        w1: &Q8,
        y0: &Buffer,
        y1: &Buffer,
        out_dim: usize,
        in_dim: usize,
    ) -> Result<()> {
        assert!(out_dim.is_multiple_of(4) && in_dim.is_multiple_of(16));
        let pso = self.gemv.pso(
            "k_dense_gemv_dual_q8",
            &[
                (1, FcValue::U32(out_dim as u32)),
                (2, FcValue::U32(in_dim as u32)),
                (3, FcValue::F32(0.0)),
            ],
        )?;
        enc.setComputePipelineState(&pso);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(x), 0, 0);
            enc.setBuffer_offset_atIndex(Some(&w0.0), 0, 1);
            enc.setBuffer_offset_atIndex(Some(&w0.1), 0, 2);
            enc.setBuffer_offset_atIndex(Some(&w1.0), 0, 3);
            enc.setBuffer_offset_atIndex(Some(&w1.1), 0, 4);
            enc.setBuffer_offset_atIndex(Some(y0), 0, 5);
            enc.setBuffer_offset_atIndex(Some(y1), 0, 6);
        }
        let (groups, threads) = grid(out_dim);
        enc.dispatchThreadgroups_threadsPerThreadgroup(groups, threads);
        Ok(())
    }
}
