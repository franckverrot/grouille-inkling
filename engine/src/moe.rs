//! Sparse MoE (Inkling 2 Mini): top-8 softcapped SwiGLU experts.

use anyhow::Result;

use crate::metal_moe::MetalMoe;
use crate::mxfp8::mxfp8_gemv;
use crate::nvfp4::nvfp4_gemv;

pub const HIDDEN: usize = 2304;
pub const INTER: usize = 1152;
pub const TOP_K: usize = 8;

#[inline]
fn softcapped_swiglu(up: &[f32], gate: &[f32], out: &mut [f32]) {
    const GATE_CAP: f32 = 4.0;
    const UP_CAP: f32 = 25.0;
    for i in 0..out.len() {
        let g = gate[i];
        let u = up[i];
        let sg = (g / GATE_CAP).tanh() * GATE_CAP;
        let silu = sg / (1.0 + (-g).exp());
        let su = (u / UP_CAP).tanh() * UP_CAP;
        out[i] = silu * su;
    }
}

/// One layer's expert weights (HF layout, gate/up already split).
///
/// NVFP4 (layers 0-19): packed nibbles + E4M3 scales + amax.
/// MXFP8 (layers 20-23): one E4M3 byte per weight + E8M0 scales (no amax).
pub struct ExpertPack<'a> {
    pub gate_w: &'a [u8],
    pub gate_s: &'a [u8],
    pub up_w: &'a [u8],
    pub up_s: &'a [u8],
    pub down_w: &'a [u8],
    pub down_s: &'a [u8],
    pub gate_up_amax: &'a [f32], // [E]; unused when mxfp8
    pub down_amax: &'a [f32],
    pub n_experts: usize,
    pub mxfp8: bool,
    /// When set, Metal MoE uses `.gpk` layer-keyed resident views (host slices may be empty).
    pub layer: Option<usize>,
}

impl ExpertPack<'_> {
    fn gate_strides(&self) -> (usize, usize) {
        if self.mxfp8 {
            (INTER * HIDDEN, INTER * (HIDDEN / 32))
        } else {
            (INTER * (HIDDEN / 2), INTER * (HIDDEN / 16))
        }
    }

    fn down_strides(&self) -> (usize, usize) {
        if self.mxfp8 {
            (HIDDEN * INTER, HIDDEN * (INTER / 32))
        } else {
            (HIDDEN * (INTER / 2), HIDDEN * (INTER / 16))
        }
    }

    fn gate_row(&self, e: usize) -> (&[u8], &[u8], f32) {
        let (w_stride, s_stride) = self.gate_strides();
        (
            &self.gate_w[e * w_stride..(e + 1) * w_stride],
            &self.gate_s[e * s_stride..(e + 1) * s_stride],
            if self.mxfp8 {
                0.0
            } else {
                self.gate_up_amax[e]
            },
        )
    }

    fn up_row(&self, e: usize) -> (&[u8], &[u8], f32) {
        let (w_stride, s_stride) = self.gate_strides();
        (
            &self.up_w[e * w_stride..(e + 1) * w_stride],
            &self.up_s[e * s_stride..(e + 1) * s_stride],
            if self.mxfp8 {
                0.0
            } else {
                self.gate_up_amax[e]
            },
        )
    }

    fn down_row(&self, e: usize) -> (&[u8], &[u8], f32) {
        let (w_stride, s_stride) = self.down_strides();
        (
            &self.down_w[e * w_stride..(e + 1) * w_stride],
            &self.down_s[e * s_stride..(e + 1) * s_stride],
            if self.mxfp8 {
                0.0
            } else {
                self.down_amax[e]
            },
        )
    }
}

/// `moe_in` and outputs are `[tokens, HIDDEN]` row-major.
/// `indices` / `weights` are `[tokens, TOP_K]`.
pub fn sparse_moe_layer0(
    moe_in: &[f32],
    indices: &[i32],
    weights: &[f32],
    tokens: usize,
    experts: &ExpertPack<'_>,
) -> Vec<f32> {
    assert_eq!(moe_in.len(), tokens * HIDDEN);
    assert_eq!(indices.len(), tokens * TOP_K);
    assert_eq!(weights.len(), tokens * TOP_K);
    let mut out = vec![0f32; tokens * HIDDEN];
    let mut hidden = vec![0f32; INTER];
    for t in 0..tokens {
        let x = &moe_in[t * HIDDEN..(t + 1) * HIDDEN];
        let acc = &mut out[t * HIDDEN..(t + 1) * HIDDEN];
        for r in 0..TOP_K {
            let e = indices[t * TOP_K + r] as usize;
            let w = weights[t * TOP_K + r];
            let (gw, gs, ga) = experts.gate_row(e);
            let (uw, us, ua) = experts.up_row(e);
            let (gate, up) = if experts.mxfp8 {
                (
                    mxfp8_gemv(gw, gs, INTER, x),
                    mxfp8_gemv(uw, us, INTER, x),
                )
            } else {
                debug_assert_eq!(ga, ua);
                (
                    nvfp4_gemv(gw, gs, INTER, ga, x),
                    nvfp4_gemv(uw, us, INTER, ua, x),
                )
            };
            softcapped_swiglu(&up, &gate, &mut hidden);
            let (dw, ds, da) = experts.down_row(e);
            let down = if experts.mxfp8 {
                mxfp8_gemv(dw, ds, HIDDEN, &hidden)
            } else {
                nvfp4_gemv(dw, ds, HIDDEN, da, &hidden)
            };
            for i in 0..HIDDEN {
                acc[i] += w * down[i];
            }
        }
    }
    out
}

/// Dispatch MoE to CPU or Metal when a `MetalMoe` is provided.
pub fn sparse_moe_dispatch(
    moe_in: &[f32],
    indices: &[i32],
    weights: &[f32],
    tokens: usize,
    experts: &ExpertPack<'_>,
    metal: Option<&mut MetalMoe>,
) -> Result<Vec<f32>> {
    let mut out = vec![0f32; tokens * HIDDEN];
    sparse_moe_dispatch_into(moe_in, indices, weights, tokens, experts, metal, &mut out)?;
    Ok(out)
}

/// Like [`sparse_moe_dispatch`] writing into a caller-owned `[tokens * H]` buffer
/// (decode keeps this on the stack for `tokens == 1`).
pub fn sparse_moe_dispatch_into(
    moe_in: &[f32],
    indices: &[i32],
    weights: &[f32],
    tokens: usize,
    experts: &ExpertPack<'_>,
    metal: Option<&mut MetalMoe>,
    out: &mut [f32],
) -> Result<()> {
    assert_eq!(out.len(), tokens * HIDDEN);
    if let Some(m) = metal {
        if experts.mxfp8 {
            m.sparse_moe_mxfp8(
                experts.gate_w,
                experts.gate_s,
                experts.up_w,
                experts.up_s,
                experts.down_w,
                experts.down_s,
                experts.n_experts,
                HIDDEN,
                INTER,
                indices,
                weights,
                tokens,
                TOP_K,
                moe_in,
                out,
                experts.layer,
            )?;
        } else {
            m.sparse_moe(
                experts.gate_w,
                experts.gate_s,
                experts.up_w,
                experts.up_s,
                experts.down_w,
                experts.down_s,
                experts.gate_up_amax,
                experts.down_amax,
                experts.n_experts,
                HIDDEN,
                INTER,
                indices,
                weights,
                tokens,
                TOP_K,
                moe_in,
                out,
                experts.layer,
            )?;
        }
        return Ok(());
    }
    out.copy_from_slice(&sparse_moe_layer0(
        moe_in, indices, weights, tokens, experts,
    ));
    Ok(())
}

/// Prefill/warm: MoE on GPU without host readback of `out`.
///
/// Returns `None` when falling back to the host path (caller should use
/// [`sparse_moe_dispatch_into`]). On success the buffer holds `[tokens * H]` f32.
pub fn sparse_moe_dispatch_keep_gpu(
    moe_in: &[f32],
    indices: &[i32],
    weights: &[f32],
    tokens: usize,
    experts: &ExpertPack<'_>,
    metal: Option<&mut MetalMoe>,
) -> Result<Option<crate::gpu::Buffer>> {
    let Some(m) = metal else {
        return Ok(None);
    };
    if experts.mxfp8 {
        Ok(Some(m.sparse_moe_mxfp8_keep_gpu(
            experts.gate_w,
            experts.gate_s,
            experts.up_w,
            experts.up_s,
            experts.down_w,
            experts.down_s,
            experts.n_experts,
            HIDDEN,
            INTER,
            indices,
            weights,
            tokens,
            TOP_K,
            moe_in,
            experts.layer,
        )?))
    } else {
        Ok(Some(m.sparse_moe_keep_gpu(
            experts.gate_w,
            experts.gate_s,
            experts.up_w,
            experts.up_s,
            experts.down_w,
            experts.down_s,
            experts.gate_up_amax,
            experts.down_amax,
            experts.n_experts,
            HIDDEN,
            INTER,
            indices,
            weights,
            tokens,
            TOP_K,
            moe_in,
            experts.layer,
        )?))
    }
}
