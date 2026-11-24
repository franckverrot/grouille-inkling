//! Metal gathered NVFP4 GEMV for MoE projections.

use anyhow::{Context, Result};
use objc2_metal::{
    MTLCommandBuffer, MTLCommandEncoder, MTLComputeCommandEncoder, MTLSize,
};
use std::collections::HashMap;

use crate::gpu::{read_buffer, write_buffer, Buffer, FcValue, Gpu};

const MSL: &str = include_str!("../kernels/moe.metal");

/// Groups the (token, route) pairs of a visit by expert into chunks of up to 32 pairs
/// for the `*_mma` kernels. Returns the chunk table (`[expert, 32 pair ids]` per chunk,
/// -1 for an empty slot), ordered by how many 8-pair blocks a chunk uses, and
/// `(first_chunk, n_chunks)` for block counts 1..=4.
fn build_chunks(indices: &[i32], n_experts: usize) -> Result<(Vec<i32>, [(usize, usize); 4])> {
    let mut by_expert: Vec<Vec<i32>> = vec![Vec::new(); n_experts];
    for (pid, &e) in indices.iter().enumerate() {
        anyhow::ensure!(
            e >= 0 && (e as usize) < n_experts,
            "routed expert {e} out of range"
        );
        by_expert[e as usize].push(pid as i32);
    }
    let mut sections: [Vec<i32>; 4] = Default::default();
    for (e, pairs) in by_expert.iter().enumerate() {
        for chunk in pairs.chunks(32) {
            let s = &mut sections[chunk.len().div_ceil(8) - 1];
            s.push(e as i32);
            s.extend_from_slice(chunk);
            s.extend(std::iter::repeat_n(-1, 32 - chunk.len()));
        }
    }
    let mut table = Vec::with_capacity(sections.iter().map(Vec::len).sum());
    let mut spans = [(0usize, 0usize); 4];
    for (i, sec) in sections.iter().enumerate() {
        spans[i] = (table.len() / 33, sec.len() / 33);
        table.extend_from_slice(sec);
    }
    Ok((table, spans))
}

/// Full-layer expert tensors resident on GPU.
///
/// `*_off` are byte offsets into the corresponding buffer (0 for owned uploads;
/// non-zero when several tensors share one `.gpk` mmap Metal buffer).
struct ResidentNvfp4 {
    gate_w: Buffer,
    gate_w_off: usize,
    gate_s: Buffer,
    gate_s_off: usize,
    up_w: Buffer,
    up_w_off: usize,
    up_s: Buffer,
    up_s_off: usize,
    down_w: Buffer,
    down_w_off: usize,
    down_s: Buffer,
    down_s_off: usize,
    gate_up_amax: Buffer,
    gate_up_amax_off: usize,
    down_amax: Buffer,
    down_amax_off: usize,
}

struct ResidentMxfp8 {
    gate_w: Buffer,
    gate_w_off: usize,
    gate_s: Buffer,
    gate_s_off: usize,
    up_w: Buffer,
    up_w_off: usize,
    up_s: Buffer,
    up_s_off: usize,
    down_w: Buffer,
    down_w_off: usize,
    down_s: Buffer,
    down_s_off: usize,
}

/// Routed (token, expert) pairs a short block can have: eight tokens, eight routes.
pub(crate) const BLOCK_PAIRS: usize = 64;

pub struct MetalMoe {
    gpu: Gpu,
    /// Reused buffers keyed by `(slot, byte_len)`.
    ///
    /// Must be slot-keyed: gate/up (and gate/up/hid activations) share sizes, so a
    /// length-only pool aliases distinct tensors onto one MTLBuffer and corrupts MoE.
    scratch: HashMap<(u8, usize), Buffer>,
    /// Input rows already on the GPU, for the next multi-token dispatch only.
    pub(crate) x_override: Option<Buffer>,
    /// Full expert weights keyed by host `gate_w` pointer (stable for WeightCache).
    resident_nv: HashMap<usize, ResidentNvfp4>,
    resident_mx: HashMap<usize, ResidentMxfp8>,
    /// Layer-keyed views (`.gpk` no-copy); preferred when `gate_w` host slice is empty.
    resident_nv_layer: HashMap<usize, ResidentNvfp4>,
    resident_mx_layer: HashMap<usize, ResidentMxfp8>,
}

impl MetalMoe {
    pub fn gpu(&self) -> &Gpu {
        &self.gpu
    }

    pub fn new() -> Result<Self> {
        let mut me = Self {
            gpu: Gpu::new(MSL).context("compile moe.metal")?,
            scratch: HashMap::new(),
            x_override: None,
            resident_nv: HashMap::new(),
            resident_mx: HashMap::new(),
            resident_nv_layer: HashMap::new(),
            resident_mx_layer: HashMap::new(),
        };
        // Decode is always T=1 / R=8 / H=2304 / inter=1152; unique experts ∈ [1,8].
        // Building these PSOs once avoids a multi-ms hitch on the first MoE visit.
        me.warmup_decode_psos()?;
        Ok(me)
    }

    fn upload_u8(&self, data: &[u8]) -> Result<Buffer> {
        let b = self.gpu.buffer_shared_write_combined(data.len().max(16))?;
        write_buffer(&b, 0, data);
        Ok(b)
    }

    fn upload_f32(&self, data: &[f32]) -> Result<Buffer> {
        let b = self.gpu.buffer_shared_write_combined((data.len() * 4).max(16))?;
        write_buffer(&b, 0, data);
        Ok(b)
    }

    /// Register NVFP4 experts as offsets into a shared `.gpk` Metal buffer.
    pub fn register_resident_nvfp4_view(
        &mut self,
        layer: usize,
        base: Buffer,
        gate_w_off: usize,
        gate_s_off: usize,
        up_w_off: usize,
        up_s_off: usize,
        down_w_off: usize,
        down_s_off: usize,
        gate_up_amax: &[f32],
        down_amax: &[f32],
    ) -> Result<()> {
        let r = ResidentNvfp4 {
            gate_w: base.clone(),
            gate_w_off,
            gate_s: base.clone(),
            gate_s_off,
            up_w: base.clone(),
            up_w_off,
            up_s: base.clone(),
            up_s_off,
            down_w: base.clone(),
            down_w_off,
            down_s: base,
            down_s_off,
            gate_up_amax: self.upload_f32(gate_up_amax)?,
            gate_up_amax_off: 0,
            down_amax: self.upload_f32(down_amax)?,
            down_amax_off: 0,
        };
        self.resident_nv_layer.insert(layer, r);
        Ok(())
    }

    /// Register MXFP8 experts as offsets into a shared `.gpk` Metal buffer.
    pub fn register_resident_mxfp8_view(
        &mut self,
        layer: usize,
        base: Buffer,
        gate_w_off: usize,
        gate_s_off: usize,
        up_w_off: usize,
        up_s_off: usize,
        down_w_off: usize,
        down_s_off: usize,
    ) -> Result<()> {
        let r = ResidentMxfp8 {
            gate_w: base.clone(),
            gate_w_off,
            gate_s: base.clone(),
            gate_s_off,
            up_w: base.clone(),
            up_w_off,
            up_s: base.clone(),
            up_s_off,
            down_w: base.clone(),
            down_w_off,
            down_s: base,
            down_s_off,
        };
        self.resident_mx_layer.insert(layer, r);
        Ok(())
    }

    fn ensure_resident_nvfp4(
        &mut self,
        gate_w: &[u8],
        gate_s: &[u8],
        up_w: &[u8],
        up_s: &[u8],
        down_w: &[u8],
        down_s: &[u8],
        gate_up_amax: &[f32],
        down_amax: &[f32],
    ) -> Result<&ResidentNvfp4> {
        let key = gate_w.as_ptr() as usize;
        if !self.resident_nv.contains_key(&key) {
            let r = ResidentNvfp4 {
                gate_w: self.upload_u8(gate_w)?,
                gate_w_off: 0,
                gate_s: self.upload_u8(gate_s)?,
                gate_s_off: 0,
                up_w: self.upload_u8(up_w)?,
                up_w_off: 0,
                up_s: self.upload_u8(up_s)?,
                up_s_off: 0,
                down_w: self.upload_u8(down_w)?,
                down_w_off: 0,
                down_s: self.upload_u8(down_s)?,
                down_s_off: 0,
                gate_up_amax: self.upload_f32(gate_up_amax)?,
                gate_up_amax_off: 0,
                down_amax: self.upload_f32(down_amax)?,
                down_amax_off: 0,
            };
            self.resident_nv.insert(key, r);
        }
        Ok(self.resident_nv.get(&key).unwrap())
    }

    fn ensure_resident_mxfp8(
        &mut self,
        gate_w: &[u8],
        gate_s: &[u8],
        up_w: &[u8],
        up_s: &[u8],
        down_w: &[u8],
        down_s: &[u8],
    ) -> Result<&ResidentMxfp8> {
        let key = gate_w.as_ptr() as usize;
        if !self.resident_mx.contains_key(&key) {
            let r = ResidentMxfp8 {
                gate_w: self.upload_u8(gate_w)?,
                gate_w_off: 0,
                gate_s: self.upload_u8(gate_s)?,
                gate_s_off: 0,
                up_w: self.upload_u8(up_w)?,
                up_w_off: 0,
                up_s: self.upload_u8(up_s)?,
                up_s_off: 0,
                down_w: self.upload_u8(down_w)?,
                down_w_off: 0,
                down_s: self.upload_u8(down_s)?,
                down_s_off: 0,
            };
            self.resident_mx.insert(key, r);
        }
        Ok(self.resident_mx.get(&key).unwrap())
    }

    /// Compile the pipelines decode and short prefills use and allocate their scratch
    /// up front, so the first token does not pay for it.
    fn warmup_decode_psos(&mut self) -> Result<()> {
        const H: usize = 2304;
        const INTER: usize = 1152;
        const R: usize = 8;
        const N: u32 = 128;
        let dual = |tokens: u32| {
            [
                (0, FcValue::U32(tokens)),
                (1, FcValue::U32(R as u32)),
                (2, FcValue::U32(INTER as u32)),
                (3, FcValue::U32(H as u32)),
                (4, FcValue::U32(N)),
                (6, FcValue::F32(4.0)),
                (7, FcValue::F32(25.0)),
            ]
        };
        // T=1 decode: the `*_dec` kernels (constants must match the encode functions).
        let down = [
            (1, FcValue::U32(R as u32)),
            (2, FcValue::U32(H as u32)),
            (3, FcValue::U32(INTER as u32)),
            (4, FcValue::U32(N)),
            (10, FcValue::U32(crate::flags::moe_down_rows() as u32)),
        ];
        for name in ["k_nvfp4_dual_swiglu_dec", "k_mxfp8_dual_swiglu_dec"] {
            let _ = self.gpu.pso(name, &dual(1))?;
        }
        for name in ["k_nvfp4_down_combine_dec", "k_mxfp8_down_combine_dec"] {
            let _ = self.gpu.pso(name, &down)?;
        }
        for slot in 0..48u8 {
            let _ = self.scratch_buf(40 + slot, R * INTER * 4)?;
        }
        // Multi-token prefill: one pipeline per block count (see the `*_mma` kernels).
        for nb in 1..=4u32 {
            for name in ["k_nvfp4_dual_mma", "k_mxfp8_dual_mma"] {
                let _ = self.gpu.pso(
                    name,
                    &[
                        (1, FcValue::U32(R as u32)),
                        (2, FcValue::U32(INTER as u32)),
                        (3, FcValue::U32(H as u32)),
                        (6, FcValue::F32(4.0)),
                        (7, FcValue::F32(25.0)),
                        (11, FcValue::U32(nb)),
                    ],
                )?;
            }
            for name in ["k_nvfp4_down_mma", "k_mxfp8_down_mma"] {
                let _ = self.gpu.pso(
                    name,
                    &[
                        (1, FcValue::U32(R as u32)),
                        (2, FcValue::U32(H as u32)),
                        (3, FcValue::U32(INTER as u32)),
                        (11, FcValue::U32(nb)),
                    ],
                )?;
            }
        }
        let _ = self.gpu.pso(
            "k_moe_sum_routes",
            &[(1, FcValue::U32(R as u32)), (2, FcValue::U32(H as u32))],
        )?;
        Ok(())
    }

    /// 256-entry E4M3 table, pre-multiplied by 0.5, for the `*_dec` NVFP4 kernels.
    fn e4m3_half_lut(&mut self) -> Result<Buffer> {
        let key = (250u8, 1024usize);
        if let Some(b) = self.scratch.get(&key) {
            return Ok(b.clone());
        }
        let b = self.gpu.buffer_shared_tracked(1024)?;
        let mut t = [0f32; 256];
        for (i, v) in t.iter_mut().enumerate() {
            let sign = if i & 0x80 != 0 { -1.0f32 } else { 1.0 };
            let e = ((i >> 3) & 0xF) as i32;
            let m = (i & 7) as f32;
            let mag = if e != 0 {
                (1.0 + m / 8.0) * 2f32.powi(e - 7)
            } else {
                (m / 8.0) * 2f32.powi(-6)
            };
            *v = 0.5 * sign * mag;
        }
        write_buffer(&b, 0, &t);
        self.scratch.insert(key, b.clone());
        Ok(b)
    }

    /// Free the large scratch buffers prompt processing leaves behind.  Decode only
    /// uses small ones.  Call while the GPU is idle.
    pub fn release_prompt_scratch(&mut self) {
        self.scratch.retain(|&(_, len), _| len < (8 << 20));
    }

    fn scratch_buf(&mut self, slot: u8, len: usize) -> Result<Buffer> {
        // Pow2 buckets so warm visits with similar unique-expert counts reuse.
        let len = len.max(16).next_power_of_two();
        let key = (slot, len);
        if let Some(b) = self.scratch.get(&key) {
            return Ok(b.clone());
        }
        // Tracked: the 3-encoder MoE path (gate_up → down → combine) shares one CB.
        let b = self.gpu.buffer_shared_tracked(len)?;
        self.scratch.insert(key, b.clone());
        Ok(b)
    }

    pub fn device_name(&self) -> String {
        self.gpu.name()
    }

    /// Gathered NVFP4 GEMV.
    ///
    /// * `packed` `[E, out, in/2]`, `scales` `[E, out, in/16]`, `amax` `[E]`
    /// * `indices` `[tokens * routes]`
    /// * `x` `[tokens * in]`
    /// * `y` `[tokens * routes * out]`
    pub fn gather_gemv(
        &mut self,
        packed: &[u8],
        scales: &[u8],
        amax: &[f32],
        n_experts: usize,
        out_dim: usize,
        in_dim: usize,
        indices: &[i32],
        tokens: usize,
        routes: usize,
        x: &[f32],
        y: &mut [f32],
    ) -> Result<()> {
        self.gather_gemv_ex(
            packed, scales, amax, n_experts, out_dim, in_dim, indices, tokens, routes, x, y,
            false,
        )
    }

    pub fn gather_gemv_ex(
        &mut self,
        packed: &[u8],
        scales: &[u8],
        amax: &[f32],
        n_experts: usize,
        out_dim: usize,
        in_dim: usize,
        indices: &[i32],
        tokens: usize,
        routes: usize,
        x: &[f32],
        y: &mut [f32],
        per_route_x: bool,
    ) -> Result<()> {
        let x_len = if per_route_x {
            tokens * routes * in_dim
        } else {
            tokens * in_dim
        };
        assert_eq!(x.len(), x_len);
        assert_eq!(indices.len(), tokens * routes);
        assert_eq!(y.len(), tokens * routes * out_dim);
        assert_eq!(amax.len(), n_experts);
        assert_eq!(packed.len(), n_experts * out_dim * (in_dim / 2));
        assert_eq!(scales.len(), n_experts * out_dim * (in_dim / 16));

        let pso = self.gpu.pso(
            "k_nvfp4_gather_gemv",
            &[
                (0, FcValue::U32(tokens as u32)),
                (1, FcValue::U32(routes as u32)),
                (2, FcValue::U32(out_dim as u32)),
                (3, FcValue::U32(in_dim as u32)),
                (4, FcValue::U32(n_experts as u32)),
                (9, FcValue::U32(if per_route_x { 1 } else { 0 })),
            ],
        )?;
        let b_p = self.gpu.buffer_shared(packed.len())?;
        let b_s = self.gpu.buffer_shared(scales.len())?;
        let b_a = self.gpu.buffer_shared(amax.len() * 4)?;
        let b_i = self.gpu.buffer_shared(indices.len() * 4)?;
        let b_x = self.gpu.buffer_shared(x.len() * 4)?;
        let b_y = self.gpu.buffer_shared(y.len() * 4)?;
        write_buffer(&b_p, 0, packed);
        write_buffer(&b_s, 0, scales);
        write_buffer(&b_a, 0, amax);
        write_buffer(&b_i, 0, indices);
        write_buffer(&b_x, 0, x);

        let tg = MTLSize {
            width: 4,
            height: 2,
            depth: 16,
        };
        let groups = MTLSize {
            width: tokens.div_ceil(tg.width),
            height: routes.div_ceil(tg.height),
            depth: out_dim.div_ceil(tg.depth),
        };
        let cb = self.gpu.command_buffer()?;
        let enc = cb.computeCommandEncoder().context("encoder")?;
        enc.setComputePipelineState(&pso);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(&b_p), 0, 0);
            enc.setBuffer_offset_atIndex(Some(&b_s), 0, 1);
            enc.setBuffer_offset_atIndex(Some(&b_a), 0, 2);
            enc.setBuffer_offset_atIndex(Some(&b_i), 0, 3);
            enc.setBuffer_offset_atIndex(Some(&b_x), 0, 4);
            enc.setBuffer_offset_atIndex(Some(&b_y), 0, 5);
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(groups, tg);
        enc.endEncoding();
        cb.commit();
        cb.waitUntilCompleted();
        read_buffer(&b_y, 0, y);
        Ok(())
    }

    /// Gathered MXFP8 GEMV (no amax).
    ///
    /// * `packed` `[E, out, in]`, `scales` `[E, out, in/32]`
    pub fn gather_gemv_mxfp8(
        &mut self,
        packed: &[u8],
        scales: &[u8],
        n_experts: usize,
        out_dim: usize,
        in_dim: usize,
        indices: &[i32],
        tokens: usize,
        routes: usize,
        x: &[f32],
        y: &mut [f32],
    ) -> Result<()> {
        self.gather_gemv_mxfp8_ex(
            packed, scales, n_experts, out_dim, in_dim, indices, tokens, routes, x, y, false,
        )
    }

    pub fn gather_gemv_mxfp8_ex(
        &mut self,
        packed: &[u8],
        scales: &[u8],
        n_experts: usize,
        out_dim: usize,
        in_dim: usize,
        indices: &[i32],
        tokens: usize,
        routes: usize,
        x: &[f32],
        y: &mut [f32],
        per_route_x: bool,
    ) -> Result<()> {
        let x_len = if per_route_x {
            tokens * routes * in_dim
        } else {
            tokens * in_dim
        };
        assert_eq!(x.len(), x_len);
        assert_eq!(indices.len(), tokens * routes);
        assert_eq!(y.len(), tokens * routes * out_dim);
        assert_eq!(in_dim % 32, 0);
        assert_eq!(packed.len(), n_experts * out_dim * in_dim);
        assert_eq!(scales.len(), n_experts * out_dim * (in_dim / 32));

        let pso = self.gpu.pso(
            "k_mxfp8_gather_gemv",
            &[
                (0, FcValue::U32(tokens as u32)),
                (1, FcValue::U32(routes as u32)),
                (2, FcValue::U32(out_dim as u32)),
                (3, FcValue::U32(in_dim as u32)),
                (4, FcValue::U32(n_experts as u32)),
                (9, FcValue::U32(if per_route_x { 1 } else { 0 })),
            ],
        )?;
        let b_p = self.gpu.buffer_shared(packed.len())?;
        let b_s = self.gpu.buffer_shared(scales.len())?;
        let b_i = self.gpu.buffer_shared(indices.len() * 4)?;
        let b_x = self.gpu.buffer_shared(x.len() * 4)?;
        let b_y = self.gpu.buffer_shared(y.len() * 4)?;
        write_buffer(&b_p, 0, packed);
        write_buffer(&b_s, 0, scales);
        write_buffer(&b_i, 0, indices);
        write_buffer(&b_x, 0, x);

        let tg = MTLSize {
            width: 4,
            height: 2,
            depth: 16,
        };
        let groups = MTLSize {
            width: tokens.div_ceil(tg.width),
            height: routes.div_ceil(tg.height),
            depth: out_dim.div_ceil(tg.depth),
        };
        let cb = self.gpu.command_buffer()?;
        let enc = cb.computeCommandEncoder().context("encoder")?;
        enc.setComputePipelineState(&pso);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(&b_p), 0, 0);
            enc.setBuffer_offset_atIndex(Some(&b_s), 0, 1);
            enc.setBuffer_offset_atIndex(Some(&b_i), 0, 2);
            enc.setBuffer_offset_atIndex(Some(&b_x), 0, 3);
            enc.setBuffer_offset_atIndex(Some(&b_y), 0, 4);
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(groups, tg);
        enc.endEncoding();
        cb.commit();
        cb.waitUntilCompleted();
        read_buffer(&b_y, 0, y);
        Ok(())
    }

    pub fn softcapped_swiglu(
        &mut self,
        up: &[f32],
        gate: &[f32],
        tokens: usize,
        routes: usize,
        inter: usize,
        gate_cap: f32,
        up_cap: f32,
        hidden: &mut [f32],
    ) -> Result<()> {
        assert_eq!(up.len(), tokens * routes * inter);
        assert_eq!(gate.len(), hidden.len());
        let pso = self.gpu.pso(
            "k_softcapped_swiglu",
            &[
                (0, FcValue::U32(tokens as u32)),
                (1, FcValue::U32(routes as u32)),
                (5, FcValue::U32(inter as u32)),
                (6, FcValue::F32(gate_cap)),
                (7, FcValue::F32(up_cap)),
            ],
        )?;
        let n = up.len();
        let b_u = self.gpu.buffer_shared(n * 4)?;
        let b_g = self.gpu.buffer_shared(n * 4)?;
        let b_h = self.gpu.buffer_shared(n * 4)?;
        write_buffer(&b_u, 0, up);
        write_buffer(&b_g, 0, gate);
        let tg = 256usize;
        let cb = self.gpu.command_buffer()?;
        let enc = cb.computeCommandEncoder().context("encoder")?;
        enc.setComputePipelineState(&pso);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(&b_u), 0, 0);
            enc.setBuffer_offset_atIndex(Some(&b_g), 0, 1);
            enc.setBuffer_offset_atIndex(Some(&b_h), 0, 2);
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: n.div_ceil(tg),
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: tg,
                height: 1,
                depth: 1,
            },
        );
        enc.endEncoding();
        cb.commit();
        cb.waitUntilCompleted();
        read_buffer(&b_h, 0, hidden);
        Ok(())
    }

    pub fn combine(
        &mut self,
        routes_out: &[f32],
        weights: &[f32],
        tokens: usize,
        routes: usize,
        hidden: usize,
        out: &mut [f32],
    ) -> Result<()> {
        assert_eq!(routes_out.len(), tokens * routes * hidden);
        assert_eq!(weights.len(), tokens * routes);
        assert_eq!(out.len(), tokens * hidden);
        let pso = self.gpu.pso(
            "k_moe_combine",
            &[
                (0, FcValue::U32(tokens as u32)),
                (1, FcValue::U32(routes as u32)),
                (8, FcValue::U32(hidden as u32)),
            ],
        )?;
        let b_r = self.gpu.buffer_shared(routes_out.len() * 4)?;
        let b_w = self.gpu.buffer_shared(weights.len() * 4)?;
        let b_o = self.gpu.buffer_shared(out.len() * 4)?;
        write_buffer(&b_r, 0, routes_out);
        write_buffer(&b_w, 0, weights);
        let tg = MTLSize {
            width: 8,
            height: 32,
            depth: 1,
        };
        let groups = MTLSize {
            width: tokens.div_ceil(tg.width),
            height: hidden.div_ceil(tg.height),
            depth: 1,
        };
        let cb = self.gpu.command_buffer()?;
        let enc = cb.computeCommandEncoder().context("encoder")?;
        enc.setComputePipelineState(&pso);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(&b_r), 0, 0);
            enc.setBuffer_offset_atIndex(Some(&b_w), 0, 1);
            enc.setBuffer_offset_atIndex(Some(&b_o), 0, 2);
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(groups, tg);
        enc.endEncoding();
        cb.commit();
        cb.waitUntilCompleted();
        read_buffer(&b_o, 0, out);
        Ok(())
    }

    /// Full MoE on NVFP4 experts: gate/up + SwiGLU gather, then the down gather with
    /// the weighted combine, reading the layer's resident expert buffers.
    #[allow(clippy::too_many_arguments)]
    pub fn sparse_moe(
        &mut self,
        gate_w: &[u8],
        gate_s: &[u8],
        up_w: &[u8],
        up_s: &[u8],
        down_w: &[u8],
        down_s: &[u8],
        gate_up_amax: &[f32],
        down_amax: &[f32],
        n_experts: usize,
        hidden: usize,
        inter: usize,
        indices: &[i32],
        weights: &[f32],
        tokens: usize,
        routes: usize,
        x: &[f32],
        out: &mut [f32],
        layer: Option<usize>,
    ) -> Result<()> {
        assert_eq!(out.len(), tokens * hidden);
        let b_out = self.sparse_moe_dispatch_nvfp4(
            gate_w, gate_s, up_w, up_s, down_w, down_s, gate_up_amax, down_amax,
            n_experts, hidden, inter, indices, weights, tokens, routes, x, layer,
        )?;
        read_buffer(&b_out, 0, out);
        Ok(())
    }

    /// NVFP4 MoE leaving the result buffer on GPU (no host readback).
    #[allow(clippy::too_many_arguments)]
    pub fn sparse_moe_keep_gpu(
        &mut self,
        gate_w: &[u8],
        gate_s: &[u8],
        up_w: &[u8],
        up_s: &[u8],
        down_w: &[u8],
        down_s: &[u8],
        gate_up_amax: &[f32],
        down_amax: &[f32],
        n_experts: usize,
        hidden: usize,
        inter: usize,
        indices: &[i32],
        weights: &[f32],
        tokens: usize,
        routes: usize,
        x: &[f32],
        layer: Option<usize>,
    ) -> Result<Buffer> {
        self.sparse_moe_dispatch_nvfp4(
            gate_w, gate_s, up_w, up_s, down_w, down_s, gate_up_amax, down_amax,
            n_experts, hidden, inter, indices, weights, tokens, routes, x, layer,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn sparse_moe_dispatch_nvfp4(
        &mut self,
        gate_w: &[u8],
        gate_s: &[u8],
        up_w: &[u8],
        up_s: &[u8],
        down_w: &[u8],
        down_s: &[u8],
        gate_up_amax: &[f32],
        down_amax: &[f32],
        n_experts: usize,
        hidden: usize,
        inter: usize,
        indices: &[i32],
        weights: &[f32],
        tokens: usize,
        routes: usize,
        x: &[f32],
        layer: Option<usize>,
    ) -> Result<Buffer> {
        assert!(self.x_override.is_some() || x.len() == tokens * hidden);
        assert_eq!(indices.len(), tokens * routes);
        assert_eq!(weights.len(), tokens * routes);
        let use_layer = layer.and_then(|l| self.resident_nv_layer.contains_key(&l).then_some(l));
        if use_layer.is_none() {
            assert_eq!(gate_up_amax.len(), n_experts);
            assert_eq!(down_amax.len(), n_experts);
        }

        let profile_moe = crate::flags::profile_moe();

        let (b_gw, ogw, b_gs, ogs, b_uw, ouw, b_us, ous, b_dw, odw, b_ds, ods, b_gua, ogua, b_da, oda, n_exp, pack_gu_ms, pack_dn_ms) =
            {
                let t_up = profile_moe.then(std::time::Instant::now);
                let (b_gw, ogw, b_gs, ogs, b_uw, ouw, b_us, ous, b_dw, odw, b_ds, ods, b_gua, ogua, b_da, oda) =
                    if let Some(l) = use_layer {
                        let r = self.resident_nv_layer.get(&l).unwrap();
                        (
                            r.gate_w.clone(), r.gate_w_off,
                            r.gate_s.clone(), r.gate_s_off,
                            r.up_w.clone(), r.up_w_off,
                            r.up_s.clone(), r.up_s_off,
                            r.down_w.clone(), r.down_w_off,
                            r.down_s.clone(), r.down_s_off,
                            r.gate_up_amax.clone(), r.gate_up_amax_off,
                            r.down_amax.clone(), r.down_amax_off,
                        )
                    } else {
                    let r = self.ensure_resident_nvfp4(
                        gate_w, gate_s, up_w, up_s, down_w, down_s, gate_up_amax, down_amax,
                    )?;
                    (
                        r.gate_w.clone(), r.gate_w_off,
                        r.gate_s.clone(), r.gate_s_off,
                        r.up_w.clone(), r.up_w_off,
                        r.up_s.clone(), r.up_s_off,
                        r.down_w.clone(), r.down_w_off,
                        r.down_s.clone(), r.down_s_off,
                        r.gate_up_amax.clone(), r.gate_up_amax_off,
                        r.down_amax.clone(), r.down_amax_off,
                    )
                    };
                let upload_ms = t_up
                    .map(|t| t.elapsed().as_secs_f64() * 1000.0)
                    .unwrap_or(0.0);
                (
                    b_gw, ogw, b_gs, ogs, b_uw, ouw, b_us, ous, b_dw, odw, b_ds, ods,
                    b_gua, ogua, b_da, oda, n_experts, upload_ms, 0.0,
                )
            };

        // Pairs grouped by expert, so each expert's weights are decoded once per chunk
        // of up to 32 tokens (see the `*_mma` kernels in moe.metal).
        let (table, spans) = build_chunks(indices, n_exp)?;
        let b_chunks = self.scratch_buf(16, table.len() * 4)?;
        let b_wt = self.scratch_buf(9, weights.len() * 4)?;
        let b_x = match self.x_override.take() {
            Some(b) => b,
            None => {
                let b = self.scratch_buf(10, x.len() * 4)?;
                write_buffer(&b, 0, x);
                b
            }
        };
        let b_hid = self.scratch_buf(13, tokens * routes * inter * 4)?;
        let b_dn = self.scratch_buf(14, tokens * routes * hidden * 4)?;
        let b_out = self.scratch_buf(15, tokens * hidden * 4)?;
        let b_lut = self.e4m3_half_lut()?;
        write_buffer(&b_chunks, 0, &table);
        write_buffer(&b_wt, 0, weights);
        let simdgroup = MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        };

        let t_gpu = profile_moe.then(std::time::Instant::now);
        let cb = self.gpu.command_buffer()?;
        {
            let enc = cb.computeCommandEncoder().context("moe")?;
            for (i, &(first, n)) in spans.iter().enumerate() {
                if n == 0 {
                    continue;
                }
                let pso = self.gpu.pso(
                    "k_nvfp4_dual_mma",
                    &[
                        (1, FcValue::U32(routes as u32)),
                        (2, FcValue::U32(inter as u32)),
                        (3, FcValue::U32(hidden as u32)),
                        (6, FcValue::F32(4.0)),
                        (7, FcValue::F32(25.0)),
                        (11, FcValue::U32(i as u32 + 1)),
                    ],
                )?;
                enc.setComputePipelineState(&pso);
                unsafe {
                    enc.setBuffer_offset_atIndex(Some(&b_gw), ogw, 0);
                    enc.setBuffer_offset_atIndex(Some(&b_gs), ogs, 1);
                    enc.setBuffer_offset_atIndex(Some(&b_uw), ouw, 2);
                    enc.setBuffer_offset_atIndex(Some(&b_us), ous, 3);
                    enc.setBuffer_offset_atIndex(Some(&b_gua), ogua, 4);
                    enc.setBuffer_offset_atIndex(Some(&b_chunks), first * 33 * 4, 5);
                    enc.setBuffer_offset_atIndex(Some(&b_x), 0, 6);
                    enc.setBuffer_offset_atIndex(Some(&b_hid), 0, 7);
                    enc.setBuffer_offset_atIndex(Some(&b_lut), 0, 8);
                }
                enc.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize {
                        width: n,
                        height: inter / 8,
                        depth: 1,
                    },
                    simdgroup,
                );
            }
            enc.memoryBarrierWithScope(objc2_metal::MTLBarrierScope::Buffers);
            for (i, &(first, n)) in spans.iter().enumerate() {
                if n == 0 {
                    continue;
                }
                let pso = self.gpu.pso(
                    "k_nvfp4_down_mma",
                    &[
                        (1, FcValue::U32(routes as u32)),
                        (2, FcValue::U32(hidden as u32)),
                        (3, FcValue::U32(inter as u32)),
                        (11, FcValue::U32(i as u32 + 1)),
                    ],
                )?;
                enc.setComputePipelineState(&pso);
                unsafe {
                    enc.setBuffer_offset_atIndex(Some(&b_dw), odw, 0);
                    enc.setBuffer_offset_atIndex(Some(&b_ds), ods, 1);
                    enc.setBuffer_offset_atIndex(Some(&b_da), oda, 2);
                    enc.setBuffer_offset_atIndex(Some(&b_chunks), first * 33 * 4, 3);
                    enc.setBuffer_offset_atIndex(Some(&b_hid), 0, 4);
                    enc.setBuffer_offset_atIndex(Some(&b_wt), 0, 5);
                    enc.setBuffer_offset_atIndex(Some(&b_dn), 0, 6);
                    enc.setBuffer_offset_atIndex(Some(&b_lut), 0, 7);
                }
                enc.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize {
                        width: n,
                        height: hidden / 8,
                        depth: 1,
                    },
                    simdgroup,
                );
            }
            enc.memoryBarrierWithScope(objc2_metal::MTLBarrierScope::Buffers);
            let pso = self.gpu.pso(
                "k_moe_sum_routes",
                &[
                    (1, FcValue::U32(routes as u32)),
                    (2, FcValue::U32(hidden as u32)),
                ],
            )?;
            enc.setComputePipelineState(&pso);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(&b_dn), 0, 0);
                enc.setBuffer_offset_atIndex(Some(&b_out), 0, 1);
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: hidden / 256,
                    height: tokens,
                    depth: 1,
                },
                MTLSize {
                    width: 256,
                    height: 1,
                    depth: 1,
                },
            );
            enc.endEncoding();
        }
        cb.commit();
        cb.waitUntilCompleted();
        let gpu_ms = t_gpu
            .map(|t| t.elapsed().as_secs_f64() * 1000.0)
            .unwrap_or(0.0);
        if profile_moe {
            eprintln!(
                "profile moe n_exp={n_exp} upload_gu={pack_gu_ms:.2} ms upload_dn={pack_dn_ms:.2} ms gpu={gpu_ms:.2} ms"
            );
        }
        Ok(b_out)
    }

    /// The experts of a block of at most eight tokens, recorded on `enc` with nothing
    /// for the host to do.  `b_i` and `b_wt` hold each token's routes as the GPU
    /// picked them, `[tokens][routes]`.  A kernel groups the pairs by expert, and the
    /// `*_mma` kernels then decode each expert once for all the tokens that chose it.
    /// `b_x` and `b_out` are `[tokens][hidden]`.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_block_mma(
        &mut self,
        enc: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputeCommandEncoder>,
        pack: &crate::moe::ExpertPack<'_>,
        hidden: usize,
        inter: usize,
        routes: usize,
        tokens: usize,
        b_x: &Buffer,
        b_i: &Buffer,
        b_wt: &Buffer,
        b_out: &Buffer,
    ) -> Result<()> {
        const SLOTS: usize = BLOCK_PAIRS;
        let pairs = tokens * routes;
        anyhow::ensure!(pairs <= SLOTS, "a block has at most {SLOTS} routed pairs");
        let b_chunks = self.scratch_buf(20, SLOTS * 33 * 4)?;
        let b_meta = self.scratch_buf(21, 16)?;
        let b_hid = self.scratch_buf(22, SLOTS * inter * 4)?;
        let b_dn = self.scratch_buf(23, SLOTS * hidden * 4)?;
        write_buffer(&b_meta, 0, &[pairs as u32]);
        let simdgroup = MTLSize { width: 32, height: 1, depth: 1 };
        let one_block = (11, FcValue::U32(1));

        let pso = self.gpu.pso("k_block_chunks", &[])?;
        enc.setComputePipelineState(&pso);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(b_i), 0, 0);
            enc.setBuffer_offset_atIndex(Some(&b_chunks), 0, 1);
            enc.setBuffer_offset_atIndex(Some(&b_meta), 0, 2);
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize { width: 1, height: 1, depth: 1 },
            MTLSize { width: SLOTS, height: 1, depth: 1 },
        );

        let dual_grid = MTLSize { width: pairs, height: inter / 8, depth: 1 };
        let down_grid = MTLSize { width: pairs, height: hidden / 8, depth: 1 };
        let fc_dual = [
            (1, FcValue::U32(routes as u32)),
            (2, FcValue::U32(inter as u32)),
            (3, FcValue::U32(hidden as u32)),
            (6, FcValue::F32(4.0)),
            (7, FcValue::F32(25.0)),
            one_block,
        ];
        let fc_down = [
            (1, FcValue::U32(routes as u32)),
            (2, FcValue::U32(hidden as u32)),
            (3, FcValue::U32(inter as u32)),
            one_block,
        ];
        if pack.mxfp8 {
            let layer = pack.layer.filter(|l| self.resident_mx_layer.contains_key(l));
            let r = match layer {
                Some(l) => self.resident_mx_layer.get(&l).unwrap(),
                None => self.ensure_resident_mxfp8(
                    pack.gate_w, pack.gate_s, pack.up_w, pack.up_s, pack.down_w, pack.down_s,
                )?,
            };
            let dual = [
                (r.gate_w.clone(), r.gate_w_off),
                (r.gate_s.clone(), r.gate_s_off),
                (r.up_w.clone(), r.up_w_off),
                (r.up_s.clone(), r.up_s_off),
                (b_chunks.clone(), 0),
                (b_x.clone(), 0),
                (b_hid.clone(), 0),
            ];
            let down = [
                (r.down_w.clone(), r.down_w_off),
                (r.down_s.clone(), r.down_s_off),
                (b_chunks.clone(), 0),
                (b_hid.clone(), 0),
                (b_wt.clone(), 0),
                (b_dn.clone(), 0),
            ];
            let pso = self.gpu.pso("k_mxfp8_dual_mma", &fc_dual)?;
            enc.setComputePipelineState(&pso);
            for (i, (b, off)) in dual.iter().enumerate() {
                unsafe { enc.setBuffer_offset_atIndex(Some(b), *off, i) };
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(dual_grid, simdgroup);
            let pso = self.gpu.pso("k_mxfp8_down_mma", &fc_down)?;
            enc.setComputePipelineState(&pso);
            for (i, (b, off)) in down.iter().enumerate() {
                unsafe { enc.setBuffer_offset_atIndex(Some(b), *off, i) };
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(down_grid, simdgroup);
        } else {
            let b_lut = self.e4m3_half_lut()?;
            let layer = pack.layer.filter(|l| self.resident_nv_layer.contains_key(l));
            let r = match layer {
                Some(l) => self.resident_nv_layer.get(&l).unwrap(),
                None => self.ensure_resident_nvfp4(
                    pack.gate_w, pack.gate_s, pack.up_w, pack.up_s, pack.down_w, pack.down_s,
                    pack.gate_up_amax, pack.down_amax,
                )?,
            };
            let dual = [
                (r.gate_w.clone(), r.gate_w_off),
                (r.gate_s.clone(), r.gate_s_off),
                (r.up_w.clone(), r.up_w_off),
                (r.up_s.clone(), r.up_s_off),
                (r.gate_up_amax.clone(), r.gate_up_amax_off),
                (b_chunks.clone(), 0),
                (b_x.clone(), 0),
                (b_hid.clone(), 0),
                (b_lut.clone(), 0),
            ];
            let down = [
                (r.down_w.clone(), r.down_w_off),
                (r.down_s.clone(), r.down_s_off),
                (r.down_amax.clone(), r.down_amax_off),
                (b_chunks.clone(), 0),
                (b_hid.clone(), 0),
                (b_wt.clone(), 0),
                (b_dn.clone(), 0),
                (b_lut.clone(), 0),
            ];
            let pso = self.gpu.pso("k_nvfp4_dual_mma", &fc_dual)?;
            enc.setComputePipelineState(&pso);
            for (i, (b, off)) in dual.iter().enumerate() {
                unsafe { enc.setBuffer_offset_atIndex(Some(b), *off, i) };
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(dual_grid, simdgroup);
            let pso = self.gpu.pso("k_nvfp4_down_mma", &fc_down)?;
            enc.setComputePipelineState(&pso);
            for (i, (b, off)) in down.iter().enumerate() {
                unsafe { enc.setBuffer_offset_atIndex(Some(b), *off, i) };
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(down_grid, simdgroup);
        }

        let pso = self.gpu.pso(
            "k_moe_sum_routes",
            &[(1, FcValue::U32(routes as u32)), (2, FcValue::U32(hidden as u32))],
        )?;
        enc.setComputePipelineState(&pso);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(&b_dn), 0, 0);
            enc.setBuffer_offset_atIndex(Some(b_out), 0, 1);
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize { width: hidden / 256, height: tokens, depth: 1 },
            MTLSize { width: 256, height: 1, depth: 1 },
        );
        Ok(())
    }

    /// Encode resident NVFP4 T=1 MoE into an existing compute encoder (no commit/wait).
    ///
    /// `b_x` / `b_i` / `b_wt` / `b_out` are device buffers already filled (or written by
    /// prior dispatches in this encoder). Used to fuse MoE onto the attn CB.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_nvfp4_resident_t1(
        &mut self,
        enc: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputeCommandEncoder>,
        gate_w: &[u8],
        gate_s: &[u8],
        up_w: &[u8],
        up_s: &[u8],
        down_w: &[u8],
        down_s: &[u8],
        gate_up_amax: &[f32],
        down_amax: &[f32],
        n_experts: usize,
        hidden: usize,
        inter: usize,
        routes: usize,
        b_x: &Buffer,
        b_i: &Buffer,
        b_wt: &Buffer,
        b_out: &Buffer,
        // Distinct MoE temp buffers when several visits share one CB.
        moe_slot: u8,
        layer: Option<usize>,
    ) -> Result<()> {
        let use_layer = layer.and_then(|l| self.resident_nv_layer.contains_key(&l).then_some(l));
        let (
            rgw, ogw, rgs, ogs, ruw, ouw, rus, ous, rdw, odw, rds, ods, rgua, ogua, rda, oda,
        ) = if let Some(l) = use_layer {
            let r = self.resident_nv_layer.get(&l).unwrap();
            (
                r.gate_w.clone(), r.gate_w_off,
                r.gate_s.clone(), r.gate_s_off,
                r.up_w.clone(), r.up_w_off,
                r.up_s.clone(), r.up_s_off,
                r.down_w.clone(), r.down_w_off,
                r.down_s.clone(), r.down_s_off,
                r.gate_up_amax.clone(), r.gate_up_amax_off,
                r.down_amax.clone(), r.down_amax_off,
            )
        } else {
            let r = self.ensure_resident_nvfp4(
                gate_w, gate_s, up_w, up_s, down_w, down_s, gate_up_amax, down_amax,
            )?;
            (
                r.gate_w.clone(), r.gate_w_off,
                r.gate_s.clone(), r.gate_s_off,
                r.up_w.clone(), r.up_w_off,
                r.up_s.clone(), r.up_s_off,
                r.down_w.clone(), r.down_w_off,
                r.down_s.clone(), r.down_s_off,
                r.gate_up_amax.clone(), r.gate_up_amax_off,
                r.down_amax.clone(), r.down_amax_off,
            )
        };
        let n_exp = n_experts;
        let tokens = 1usize;
        const SIMD: usize = 32;
        let fc_dual = [
            (0, FcValue::U32(tokens as u32)),
            (1, FcValue::U32(routes as u32)),
            (2, FcValue::U32(inter as u32)),
            (3, FcValue::U32(hidden as u32)),
            (4, FcValue::U32(n_exp as u32)),
            (6, FcValue::F32(4.0)),
            (7, FcValue::F32(25.0)),
        ];
        // These kernels decode in registers, with no threadgroup copy of x, and take
        // several output rows per simdgroup on the down-projection.
        let down_rows = crate::flags::moe_down_rows();
        let pso_dual = self.gpu.pso("k_nvfp4_dual_swiglu_dec", &fc_dual)?;
        let pso_dn = self.gpu.pso(
            "k_nvfp4_down_combine_dec",
            &[
                (1, FcValue::U32(routes as u32)),
                (2, FcValue::U32(hidden as u32)),
                (3, FcValue::U32(inter as u32)),
                (4, FcValue::U32(n_exp as u32)),
                (10, FcValue::U32(down_rows as u32)),
            ],
        )?;
        assert!(moe_slot < 48);
        let b_lut = self.e4m3_half_lut()?;
        let b_hid = self.scratch_buf(40 + moe_slot, tokens * routes * inter * 4)?;
        enc.setComputePipelineState(&pso_dual);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(&rgw), ogw, 0);
            enc.setBuffer_offset_atIndex(Some(&rgs), ogs, 1);
            enc.setBuffer_offset_atIndex(Some(&ruw), ouw, 2);
            enc.setBuffer_offset_atIndex(Some(&rus), ous, 3);
            enc.setBuffer_offset_atIndex(Some(&rgua), ogua, 4);
            enc.setBuffer_offset_atIndex(Some(b_i), 0, 5);
            enc.setBuffer_offset_atIndex(Some(b_x), 0, 6);
            enc.setBuffer_offset_atIndex(Some(&b_hid), 0, 7);
            enc.setBuffer_offset_atIndex(Some(&b_lut), 0, 8);
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize { width: 1, height: routes, depth: inter.div_ceil(8) },
            MTLSize { width: SIMD, height: 8, depth: 1 },
        );
        enc.memoryBarrierWithScope(objc2_metal::MTLBarrierScope::Buffers);
        enc.setComputePipelineState(&pso_dn);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(&rdw), odw, 0);
            enc.setBuffer_offset_atIndex(Some(&rds), ods, 1);
            enc.setBuffer_offset_atIndex(Some(&rda), oda, 2);
            enc.setBuffer_offset_atIndex(Some(b_i), 0, 3);
            enc.setBuffer_offset_atIndex(Some(&b_hid), 0, 4);
            enc.setBuffer_offset_atIndex(Some(b_wt), 0, 5);
            enc.setBuffer_offset_atIndex(Some(b_out), 0, 6);
            enc.setBuffer_offset_atIndex(Some(&b_lut), 0, 7);
            enc.setThreadgroupMemoryLength_atIndex(routes * down_rows * 4, 0);
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize { width: hidden.div_ceil(down_rows), height: 1, depth: 1 },
            MTLSize { width: SIMD, height: routes, depth: 1 },
        );
        Ok(())
    
    }

    /// Encode resident MXFP8 T=1 MoE into an existing compute encoder (no commit/wait).
    #[allow(clippy::too_many_arguments)]
    pub fn encode_mxfp8_resident_t1(
        &mut self,
        enc: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputeCommandEncoder>,
        gate_w: &[u8],
        gate_s: &[u8],
        up_w: &[u8],
        up_s: &[u8],
        down_w: &[u8],
        down_s: &[u8],
        n_experts: usize,
        hidden: usize,
        inter: usize,
        routes: usize,
        b_x: &Buffer,
        b_i: &Buffer,
        b_wt: &Buffer,
        b_out: &Buffer,
        moe_slot: u8,
        layer: Option<usize>,
    ) -> Result<()> {
        let use_layer = layer.and_then(|l| self.resident_mx_layer.contains_key(&l).then_some(l));
        let (rgw, ogw, rgs, ogs, ruw, ouw, rus, ous, rdw, odw, rds, ods) =
            if let Some(l) = use_layer {
                let r = self.resident_mx_layer.get(&l).unwrap();
                (
                    r.gate_w.clone(), r.gate_w_off,
                    r.gate_s.clone(), r.gate_s_off,
                    r.up_w.clone(), r.up_w_off,
                    r.up_s.clone(), r.up_s_off,
                    r.down_w.clone(), r.down_w_off,
                    r.down_s.clone(), r.down_s_off,
                )
            } else {
                let r = self.ensure_resident_mxfp8(gate_w, gate_s, up_w, up_s, down_w, down_s)?;
                (
                    r.gate_w.clone(), r.gate_w_off,
                    r.gate_s.clone(), r.gate_s_off,
                    r.up_w.clone(), r.up_w_off,
                    r.up_s.clone(), r.up_s_off,
                    r.down_w.clone(), r.down_w_off,
                    r.down_s.clone(), r.down_s_off,
                )
            };
        let n_exp = n_experts;
        let tokens = 1usize;
        const SIMD: usize = 32;
        let fc_dual = [
            (0, FcValue::U32(tokens as u32)),
            (1, FcValue::U32(routes as u32)),
            (2, FcValue::U32(inter as u32)),
            (3, FcValue::U32(hidden as u32)),
            (4, FcValue::U32(n_exp as u32)),
            (6, FcValue::F32(4.0)),
            (7, FcValue::F32(25.0)),
        ];
        let down_rows = crate::flags::moe_down_rows();
        let pso_dual = self.gpu.pso("k_mxfp8_dual_swiglu_dec", &fc_dual)?;
        let pso_dn = self.gpu.pso(
            "k_mxfp8_down_combine_dec",
            &[
                (1, FcValue::U32(routes as u32)),
                (2, FcValue::U32(hidden as u32)),
                (3, FcValue::U32(inter as u32)),
                (4, FcValue::U32(n_exp as u32)),
                (10, FcValue::U32(down_rows as u32)),
            ],
        )?;
        assert!(moe_slot < 48);
        let b_hid = self.scratch_buf(40 + moe_slot, tokens * routes * inter * 4)?;
        enc.setComputePipelineState(&pso_dual);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(&rgw), ogw, 0);
            enc.setBuffer_offset_atIndex(Some(&rgs), ogs, 1);
            enc.setBuffer_offset_atIndex(Some(&ruw), ouw, 2);
            enc.setBuffer_offset_atIndex(Some(&rus), ous, 3);
            enc.setBuffer_offset_atIndex(Some(b_i), 0, 4);
            enc.setBuffer_offset_atIndex(Some(b_x), 0, 5);
            enc.setBuffer_offset_atIndex(Some(&b_hid), 0, 6);
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize { width: 1, height: routes, depth: inter.div_ceil(8) },
            MTLSize { width: SIMD, height: 8, depth: 1 },
        );
        enc.memoryBarrierWithScope(objc2_metal::MTLBarrierScope::Buffers);
        enc.setComputePipelineState(&pso_dn);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(&rdw), odw, 0);
            enc.setBuffer_offset_atIndex(Some(&rds), ods, 1);
            enc.setBuffer_offset_atIndex(Some(b_i), 0, 2);
            enc.setBuffer_offset_atIndex(Some(&b_hid), 0, 3);
            enc.setBuffer_offset_atIndex(Some(b_wt), 0, 4);
            enc.setBuffer_offset_atIndex(Some(b_out), 0, 5);
            enc.setThreadgroupMemoryLength_atIndex(routes * down_rows * 4, 0);
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize { width: hidden.div_ceil(down_rows), height: 1, depth: 1 },
            MTLSize { width: SIMD, height: routes, depth: 1 },
        );
        Ok(())
    
    }

    /// Full MoE on MXFP8 experts (layers 20-23).
    ///
    /// Reads the layer's resident expert buffers (global indices, n_experts=128).
    #[allow(clippy::too_many_arguments)]
    pub fn sparse_moe_mxfp8(
        &mut self,
        gate_w: &[u8],
        gate_s: &[u8],
        up_w: &[u8],
        up_s: &[u8],
        down_w: &[u8],
        down_s: &[u8],
        n_experts: usize,
        hidden: usize,
        inter: usize,
        indices: &[i32],
        weights: &[f32],
        tokens: usize,
        routes: usize,
        x: &[f32],
        out: &mut [f32],
        layer: Option<usize>,
    ) -> Result<()> {
        assert_eq!(out.len(), tokens * hidden);
        let b_out = self.sparse_moe_dispatch_mxfp8(
            gate_w, gate_s, up_w, up_s, down_w, down_s, n_experts, hidden, inter,
            indices, weights, tokens, routes, x, layer,
        )?;
        read_buffer(&b_out, 0, out);
        Ok(())
    }

    /// MXFP8 MoE leaving the result buffer on GPU (no host readback).
    #[allow(clippy::too_many_arguments)]
    pub fn sparse_moe_mxfp8_keep_gpu(
        &mut self,
        gate_w: &[u8],
        gate_s: &[u8],
        up_w: &[u8],
        up_s: &[u8],
        down_w: &[u8],
        down_s: &[u8],
        n_experts: usize,
        hidden: usize,
        inter: usize,
        indices: &[i32],
        weights: &[f32],
        tokens: usize,
        routes: usize,
        x: &[f32],
        layer: Option<usize>,
    ) -> Result<Buffer> {
        self.sparse_moe_dispatch_mxfp8(
            gate_w, gate_s, up_w, up_s, down_w, down_s, n_experts, hidden, inter,
            indices, weights, tokens, routes, x, layer,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn sparse_moe_dispatch_mxfp8(
        &mut self,
        gate_w: &[u8],
        gate_s: &[u8],
        up_w: &[u8],
        up_s: &[u8],
        down_w: &[u8],
        down_s: &[u8],
        n_experts: usize,
        hidden: usize,
        inter: usize,
        indices: &[i32],
        weights: &[f32],
        tokens: usize,
        routes: usize,
        x: &[f32],
        layer: Option<usize>,
    ) -> Result<Buffer> {
        assert!(self.x_override.is_some() || x.len() == tokens * hidden);
        assert_eq!(indices.len(), tokens * routes);
        assert_eq!(weights.len(), tokens * routes);
        assert_eq!(hidden % 32, 0);
        assert_eq!(inter % 32, 0);

        let use_layer = layer.and_then(|l| self.resident_mx_layer.contains_key(&l).then_some(l));
        let (b_gw, ogw, b_gs, ogs, b_uw, ouw, b_us, ous, b_dw, odw, b_ds, ods, n_exp) =
            {
                let (b_gw, ogw, b_gs, ogs, b_uw, ouw, b_us, ous, b_dw, odw, b_ds, ods) =
                    if let Some(l) = use_layer {
                        let r = self.resident_mx_layer.get(&l).unwrap();
                        (
                            r.gate_w.clone(), r.gate_w_off,
                            r.gate_s.clone(), r.gate_s_off,
                            r.up_w.clone(), r.up_w_off,
                            r.up_s.clone(), r.up_s_off,
                            r.down_w.clone(), r.down_w_off,
                            r.down_s.clone(), r.down_s_off,
                        )
                    } else {
                        let r = self.ensure_resident_mxfp8(
                            gate_w, gate_s, up_w, up_s, down_w, down_s,
                        )?;
                        (
                            r.gate_w.clone(), r.gate_w_off,
                            r.gate_s.clone(), r.gate_s_off,
                            r.up_w.clone(), r.up_w_off,
                            r.up_s.clone(), r.up_s_off,
                            r.down_w.clone(), r.down_w_off,
                            r.down_s.clone(), r.down_s_off,
                        )
                    };
                (b_gw, ogw, b_gs, ogs, b_uw, ouw, b_us, ous, b_dw, odw, b_ds, ods, n_experts)
            };

        // Pairs grouped by expert, so each expert's weights are decoded once per chunk
        // of up to 32 tokens (see the `*_mma` kernels in moe.metal).
        let (table, spans) = build_chunks(indices, n_exp)?;
        let b_chunks = self.scratch_buf(16, table.len() * 4)?;
        let b_wt = self.scratch_buf(9, weights.len() * 4)?;
        let b_x = match self.x_override.take() {
            Some(b) => b,
            None => {
                let b = self.scratch_buf(10, x.len() * 4)?;
                write_buffer(&b, 0, x);
                b
            }
        };
        let b_hid = self.scratch_buf(13, tokens * routes * inter * 4)?;
        let b_dn = self.scratch_buf(14, tokens * routes * hidden * 4)?;
        let b_out = self.scratch_buf(15, tokens * hidden * 4)?;
        write_buffer(&b_chunks, 0, &table);
        write_buffer(&b_wt, 0, weights);
        let simdgroup = MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        };

        let cb = self.gpu.command_buffer()?;
        {
            let enc = cb.computeCommandEncoder().context("moe")?;
            for (i, &(first, n)) in spans.iter().enumerate() {
                if n == 0 {
                    continue;
                }
                let pso = self.gpu.pso(
                    "k_mxfp8_dual_mma",
                    &[
                        (1, FcValue::U32(routes as u32)),
                        (2, FcValue::U32(inter as u32)),
                        (3, FcValue::U32(hidden as u32)),
                        (6, FcValue::F32(4.0)),
                        (7, FcValue::F32(25.0)),
                        (11, FcValue::U32(i as u32 + 1)),
                    ],
                )?;
                enc.setComputePipelineState(&pso);
                unsafe {
                    enc.setBuffer_offset_atIndex(Some(&b_gw), ogw, 0);
                    enc.setBuffer_offset_atIndex(Some(&b_gs), ogs, 1);
                    enc.setBuffer_offset_atIndex(Some(&b_uw), ouw, 2);
                    enc.setBuffer_offset_atIndex(Some(&b_us), ous, 3);
                    enc.setBuffer_offset_atIndex(Some(&b_chunks), first * 33 * 4, 4);
                    enc.setBuffer_offset_atIndex(Some(&b_x), 0, 5);
                    enc.setBuffer_offset_atIndex(Some(&b_hid), 0, 6);
                }
                enc.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize {
                        width: n,
                        height: inter / 8,
                        depth: 1,
                    },
                    simdgroup,
                );
            }
            enc.memoryBarrierWithScope(objc2_metal::MTLBarrierScope::Buffers);
            for (i, &(first, n)) in spans.iter().enumerate() {
                if n == 0 {
                    continue;
                }
                let pso = self.gpu.pso(
                    "k_mxfp8_down_mma",
                    &[
                        (1, FcValue::U32(routes as u32)),
                        (2, FcValue::U32(hidden as u32)),
                        (3, FcValue::U32(inter as u32)),
                        (11, FcValue::U32(i as u32 + 1)),
                    ],
                )?;
                enc.setComputePipelineState(&pso);
                unsafe {
                    enc.setBuffer_offset_atIndex(Some(&b_dw), odw, 0);
                    enc.setBuffer_offset_atIndex(Some(&b_ds), ods, 1);
                    enc.setBuffer_offset_atIndex(Some(&b_chunks), first * 33 * 4, 2);
                    enc.setBuffer_offset_atIndex(Some(&b_hid), 0, 3);
                    enc.setBuffer_offset_atIndex(Some(&b_wt), 0, 4);
                    enc.setBuffer_offset_atIndex(Some(&b_dn), 0, 5);
                }
                enc.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize {
                        width: n,
                        height: hidden / 8,
                        depth: 1,
                    },
                    simdgroup,
                );
            }
            enc.memoryBarrierWithScope(objc2_metal::MTLBarrierScope::Buffers);
            let pso = self.gpu.pso(
                "k_moe_sum_routes",
                &[
                    (1, FcValue::U32(routes as u32)),
                    (2, FcValue::U32(hidden as u32)),
                ],
            )?;
            enc.setComputePipelineState(&pso);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(&b_dn), 0, 0);
                enc.setBuffer_offset_atIndex(Some(&b_out), 0, 1);
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: hidden / 256,
                    height: tokens,
                    depth: 1,
                },
                MTLSize {
                    width: 256,
                    height: 1,
                    depth: 1,
                },
            );
            enc.endEncoding();
        }
        cb.commit();
        cb.waitUntilCompleted();
        Ok(b_out)
    }
}
