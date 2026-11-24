//! Checkpoint weight loading helpers for decoder layers.

use anyhow::Result;
use half::bf16;
use safetensors::tensor::SafeTensors;

use crate::attn::{FullAttnWeights, SlidingAttnWeights};
use crate::decoder::SlidingLayerWeights;
use crate::layer5::FullLayerWeights;
use crate::loop_msg::MessageControllerWeights;
use crate::moe::ExpertPack;
use crate::ops::HIDDEN;

pub fn bf16_f32(st: &SafeTensors, name: &str) -> Result<Vec<f32>> {
    let t = st.tensor(name)?;
    let data = t.data();
    let n = data.len() / 2;
    // Large attn/proj matrices dominate cold preload; fan out the convert.
    const PARALLEL_AT: usize = 64 * 1024;
    if n < PARALLEL_AT {
        return Ok(data
            .chunks_exact(2)
            .map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
            .collect());
    }
    let mut out = vec![0f32; n];
    // Cap at 2: preload already fans out across layers.
    let n_threads = 2usize.min(n / 16_384).max(1);
    let chunk = n.div_ceil(n_threads);
    let out_ptr = out.as_mut_ptr() as usize;
    let data_ptr = data.as_ptr() as usize;
    std::thread::scope(|scope| {
        let mut start = 0usize;
        while start < n {
            let end = (start + chunk).min(n);
            scope.spawn(move || {
                let data =
                    unsafe { std::slice::from_raw_parts(data_ptr as *const u8, n * 2) };
                let out =
                    unsafe { std::slice::from_raw_parts_mut(out_ptr as *mut f32, n) };
                for i in start..end {
                    let c = &data[i * 2..i * 2 + 2];
                    out[i] = bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32();
                }
            });
            start = end;
        }
    });
    Ok(out)
}

pub fn f32s(st: &SafeTensors, name: &str) -> Result<Vec<f32>> {
    let t = st.tensor(name)?;
    let data = t.data();
    assert_eq!(data.len() % 4, 0);
    let n = data.len() / 4;
    let mut out = vec![0f32; n];
    // Checkpoint is little-endian f32; on LE hosts this is a straight memcpy.
    #[cfg(target_endian = "little")]
    {
        out.copy_from_slice(unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const f32, n)
        });
    }
    #[cfg(target_endian = "big")]
    {
        for (o, c) in out.iter_mut().zip(data.chunks_exact(4)) {
            *o = f32::from_le_bytes(c.try_into().unwrap());
        }
    }
    Ok(out)
}

pub fn u8s(st: &SafeTensors, name: &str) -> Result<Vec<u8>> {
    Ok(st.tensor(name)?.data().to_vec())
}

pub fn taps(w: &[f32]) -> Vec<f32> {
    w.to_vec()
}

pub fn split_gate_up(fused: &[u8], n: usize, rows: usize, stride: usize) -> (Vec<u8>, Vec<u8>) {
    let half = rows / 2;
    let row_bytes = half * stride;
    let mut gate = vec![0u8; n * row_bytes];
    let mut up = vec![0u8; n * row_bytes];
    // Expert rows are independent; parallelize across experts (warm path hotspot).
    let n_threads = std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(4)
        .min(n)
        .max(1);
    let chunk = n.div_ceil(n_threads);
    std::thread::scope(|scope| {
        let mut e0 = 0usize;
        for (gc, uc) in gate
            .chunks_mut(chunk * row_bytes)
            .zip(up.chunks_mut(chunk * row_bytes))
        {
            let n_e = gc.len() / row_bytes;
            let start = e0;
            e0 += n_e;
            scope.spawn(move || {
                for e in 0..n_e {
                    let ge = start + e;
                    for i in 0..half {
                        let sg = ((ge * rows) + i * 2) * stride;
                        let su = ((ge * rows) + i * 2 + 1) * stride;
                        let d = (e * half + i) * stride;
                        gc[d..d + stride].copy_from_slice(&fused[sg..sg + stride]);
                        uc[d..d + stride].copy_from_slice(&fused[su..su + stride]);
                    }
                }
            });
        }
    });
    (gate, up)
}

fn load_ctrl(st: &SafeTensors, layer: usize, name: &str) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    let p = format!("model.llm.layers.{layer}.controllers.{name}.");
    Ok((
        bf16_f32(st, &format!("{p}down_weight"))?,
        bf16_f32(st, &format!("{p}forget_weight"))?,
        bf16_f32(st, &format!("{p}gain_weight"))?,
    ))
}

/// Owned buffers for one sliding layer (optionally with controllers).
pub struct SlidingLayerBufs {
    pub attn_norm: Vec<f32>,
    pub mlp_norm: Vec<f32>,
    pub attn_sconv: Vec<f32>,
    pub mlp_sconv: Vec<f32>,
    pub attn_query: Option<Vec<f32>>,
    pub mlp_query: Vec<f32>,
    pub router_w: Vec<f32>,
    pub router_b: Vec<f32>,
    pub wq: Vec<f32>,
    pub wk: Vec<f32>,
    pub wv: Vec<f32>,
    pub wr: Vec<f32>,
    pub wo: Vec<f32>,
    pub gate: Vec<f32>,
    pub sink: Vec<f32>,
    pub q_scale: Vec<f32>,
    pub rel_scale: Vec<f32>,
    pub rel_proj: Vec<f32>,
    pub k_sconv: Vec<f32>,
    pub v_sconv: Vec<f32>,
    pub gate_w: Vec<u8>,
    pub gate_s: Vec<u8>,
    pub up_w: Vec<u8>,
    pub up_s: Vec<u8>,
    pub down_w: Vec<u8>,
    pub down_s: Vec<u8>,
    pub gate_up_amax: Vec<f32>,
    pub down_amax: Vec<f32>,
    pub mxfp8: bool,
    pub boundary: bool,
    /// Set when experts live in a `.gpk` Metal view (host expert vecs empty).
    pub gpk_layer: Option<usize>,
    pub ctrl_attn: Option<(Vec<f32>, Vec<f32>, Vec<f32>)>,
    pub ctrl_k: Option<(Vec<f32>, Vec<f32>, Vec<f32>)>,
    pub ctrl_v: Option<(Vec<f32>, Vec<f32>, Vec<f32>)>,
    pub ctrl_mlp: Option<(Vec<f32>, Vec<f32>, Vec<f32>)>,
}

impl SlidingLayerBufs {
    pub fn load(st: &SafeTensors, i: usize, with_controllers: bool) -> Result<Self> {
        let p = format!("model.llm.layers.{i}.");
        let ep = format!("{p}mlp.experts.");
        let mxfp8 = is_mxfp8(i);
        let w13 = u8s(st, &format!("{ep}w13_weight"))?;
        let s13 = u8s(st, &format!("{ep}w13_weight_scale"))?;
        let (w_stride, s_stride) = if mxfp8 {
            (HIDDEN, HIDDEN / 32)
        } else {
            (HIDDEN / 2, HIDDEN / 16)
        };
        let (gate_w, up_w) = split_gate_up(&w13, 128, 2304, w_stride);
        let (gate_s, up_s) = split_gate_up(&s13, 128, 2304, s_stride);
        let (gate_up_amax, down_amax) = if mxfp8 {
            (Vec::new(), Vec::new())
        } else {
            (
                f32s(st, &format!("{ep}w13_weight_amax"))?,
                f32s(st, &format!("{ep}w2_weight_amax"))?,
            )
        };
        let attn_query = if i == 0 {
            None
        } else {
            Some(f32s(
                st,
                &format!("model.llm.attn_res.attn_queries.{}", i - 1),
            )?)
        };
        let (ctrl_attn, ctrl_k, ctrl_v, ctrl_mlp) = if with_controllers {
            (
                Some(load_ctrl(st, i, "attn")?),
                Some(load_ctrl(st, i, "k")?),
                Some(load_ctrl(st, i, "v")?),
                Some(load_ctrl(st, i, "mlp")?),
            )
        } else {
            (None, None, None, None)
        };
        Ok(Self {
            attn_norm: bf16_f32(st, &format!("{p}attn_norm.weight"))?,
            mlp_norm: bf16_f32(st, &format!("{p}mlp_norm.weight"))?,
            attn_sconv: taps(&bf16_f32(st, &format!("{p}attn_sconv.weight"))?),
            mlp_sconv: taps(&bf16_f32(st, &format!("{p}mlp_sconv.weight"))?),
            attn_query,
            mlp_query: f32s(st, &format!("model.llm.attn_res.mlp_queries.{i}"))?,
            router_w: bf16_f32(st, &format!("{p}mlp.gate.weight"))?,
            router_b: f32s(st, &format!("{p}mlp.gate.bias"))?,
            wq: bf16_f32(st, &format!("{p}attn.wq_du.weight"))?,
            wk: bf16_f32(st, &format!("{p}attn.wk_dv.weight"))?,
            wv: bf16_f32(st, &format!("{p}attn.wv_dv.weight"))?,
            wr: bf16_f32(st, &format!("{p}attn.wr_du.weight"))?,
            wo: bf16_f32(st, &format!("{p}attn.wo_ud.weight"))?,
            gate: bf16_f32(st, &format!("{p}attn.gate_attn.weight"))?,
            sink: bf16_f32(st, &format!("{p}attn.attn_sink_proj.weight"))?,
            q_scale: bf16_f32(st, &format!("{p}attn.q_l2_clip.scale"))?,
            rel_scale: bf16_f32(st, &format!("{p}attn.rel_l2_clip.scale"))?,
            rel_proj: bf16_f32(st, &format!("{p}attn.rel_logits_proj.proj"))?,
            k_sconv: taps(&bf16_f32(st, &format!("{p}attn.k_sconv.weight"))?),
            v_sconv: taps(&bf16_f32(st, &format!("{p}attn.v_sconv.weight"))?),
            gate_w,
            gate_s,
            up_w,
            up_s,
            down_w: u8s(st, &format!("{ep}w2_weight"))?,
            down_s: u8s(st, &format!("{ep}w2_weight_scale"))?,
            gate_up_amax,
            down_amax,
            mxfp8,
            boundary: i.is_multiple_of(6),
            gpk_layer: None,
            ctrl_attn,
            ctrl_k,
            ctrl_v,
            ctrl_mlp,
        })
    }

    pub fn as_weights(&self) -> SlidingLayerWeights<'_> {
        SlidingLayerWeights {
            attn_norm: &self.attn_norm,
            mlp_norm: &self.mlp_norm,
            attn_sconv: &self.attn_sconv,
            mlp_sconv: &self.mlp_sconv,
            attn_query: self.attn_query.as_deref(),
            mlp_query: &self.mlp_query,
            router_w: &self.router_w,
            router_b: &self.router_b,
            attn: SlidingAttnWeights {
                layer: self.gpk_layer,
                wq: &self.wq,
                wk: &self.wk,
                wv: &self.wv,
                wr: &self.wr,
                wo: &self.wo,
                gate: &self.gate,
                sink: &self.sink,
                q_scale: &self.q_scale,
                rel_scale: &self.rel_scale,
                rel_proj: &self.rel_proj,
                k_sconv: &self.k_sconv,
                v_sconv: &self.v_sconv,
            },
            experts: ExpertPack {
                gate_w: &self.gate_w,
                gate_s: &self.gate_s,
                up_w: &self.up_w,
                up_s: &self.up_s,
                down_w: &self.down_w,
                down_s: &self.down_s,
                gate_up_amax: &self.gate_up_amax,
                down_amax: &self.down_amax,
                n_experts: 128,
                mxfp8: self.mxfp8,
                layer: self.gpk_layer,
            },
            is_block_boundary: self.boundary,
            ctrl_attn: self.ctrl_attn.as_ref().map(|(d, f, g)| MessageControllerWeights {
                down: d,
                forget: f,
                gain: g,
            }),
            ctrl_k: self.ctrl_k.as_ref().map(|(d, f, g)| MessageControllerWeights {
                down: d,
                forget: f,
                gain: g,
            }),
            ctrl_v: self.ctrl_v.as_ref().map(|(d, f, g)| MessageControllerWeights {
                down: d,
                forget: f,
                gain: g,
            }),
            ctrl_mlp: self.ctrl_mlp.as_ref().map(|(d, f, g)| MessageControllerWeights {
                down: d,
                forget: f,
                gain: g,
            }),
        }
    }
}

/// Owned buffers for one full-attention layer.
pub struct FullLayerBufs {
    pub q_norm: Vec<f32>,
    pub k_norm: Vec<f32>,
    pub v_norm: Vec<f32>,
    pub mlp_norm: Vec<f32>,
    pub attn_sconv: Vec<f32>,
    pub mlp_sconv: Vec<f32>,
    pub attn_queries: Vec<f32>,
    pub mlp_query: Vec<f32>,
    pub router_w: Vec<f32>,
    pub router_b: Vec<f32>,
    pub wq: Vec<f32>,
    pub wk: Vec<f32>,
    pub wv: Vec<f32>,
    pub wr: Vec<f32>,
    pub wo: Vec<f32>,
    pub gate: Vec<f32>,
    pub sink: Vec<f32>,
    pub q_scale: Vec<f32>,
    pub rel_scale: Vec<f32>,
    pub rel_proj: Vec<f32>,
    pub k_static: Vec<f32>,
    pub k_wproj: Vec<f32>,
    pub v_static: Vec<f32>,
    pub v_wproj: Vec<f32>,
    pub gate_w: Vec<u8>,
    pub gate_s: Vec<u8>,
    pub up_w: Vec<u8>,
    pub up_s: Vec<u8>,
    pub down_w: Vec<u8>,
    pub down_s: Vec<u8>,
    pub gate_up_amax: Vec<f32>,
    pub down_amax: Vec<f32>,
    pub mxfp8: bool,
    pub boundary: bool,
    pub gpk_layer: Option<usize>,
    pub ctrl_attn: Option<(Vec<f32>, Vec<f32>, Vec<f32>)>,
    pub ctrl_k: Option<(Vec<f32>, Vec<f32>, Vec<f32>)>,
    pub ctrl_v: Option<(Vec<f32>, Vec<f32>, Vec<f32>)>,
    pub ctrl_mlp: Option<(Vec<f32>, Vec<f32>, Vec<f32>)>,
}

impl FullLayerBufs {
    pub fn load(st: &SafeTensors, i: usize) -> Result<Self> {
        Self::load_with_controllers(st, i, false)
    }

    pub fn load_with_controllers(st: &SafeTensors, i: usize, with_controllers: bool) -> Result<Self> {
        let p = format!("model.llm.layers.{i}.");
        let ep = format!("{p}mlp.experts.");
        let mxfp8 = is_mxfp8(i);
        let w13 = u8s(st, &format!("{ep}w13_weight"))?;
        let s13 = u8s(st, &format!("{ep}w13_weight_scale"))?;
        let (w_stride, s_stride) = if mxfp8 {
            (HIDDEN, HIDDEN / 32)
        } else {
            (HIDDEN / 2, HIDDEN / 16)
        };
        let (gate_w, up_w) = split_gate_up(&w13, 128, 2304, w_stride);
        let (gate_s, up_s) = split_gate_up(&s13, 128, 2304, s_stride);
        let (gate_up_amax, down_amax) = if mxfp8 {
            (Vec::new(), Vec::new())
        } else {
            (
                f32s(st, &format!("{ep}w13_weight_amax"))?,
                f32s(st, &format!("{ep}w2_weight_amax"))?,
            )
        };
        let (ctrl_attn, ctrl_k, ctrl_v, ctrl_mlp) = if with_controllers {
            (
                Some(load_ctrl(st, i, "attn")?),
                Some(load_ctrl(st, i, "k")?),
                Some(load_ctrl(st, i, "v")?),
                Some(load_ctrl(st, i, "mlp")?),
            )
        } else {
            (None, None, None, None)
        };
        Ok(Self {
            q_norm: bf16_f32(st, &format!("{p}q_input_norm.weight"))?,
            k_norm: bf16_f32(st, &format!("{p}k_input_norm.weight"))?,
            v_norm: bf16_f32(st, &format!("{p}v_input_norm.weight"))?,
            mlp_norm: bf16_f32(st, &format!("{p}mlp_norm.weight"))?,
            attn_sconv: taps(&bf16_f32(st, &format!("{p}attn_sconv.weight"))?),
            mlp_sconv: taps(&bf16_f32(st, &format!("{p}mlp_sconv.weight"))?),
            attn_queries: f32s(st, &format!("model.llm.attn_res.attn_queries.{}", i - 1))?,
            mlp_query: f32s(st, &format!("model.llm.attn_res.mlp_queries.{i}"))?,
            router_w: bf16_f32(st, &format!("{p}mlp.gate.weight"))?,
            router_b: f32s(st, &format!("{p}mlp.gate.bias"))?,
            wq: bf16_f32(st, &format!("{p}attn.wq_du.weight"))?,
            wk: bf16_f32(st, &format!("{p}attn.wk_dv.weight"))?,
            wv: bf16_f32(st, &format!("{p}attn.wv_dv.weight"))?,
            wr: bf16_f32(st, &format!("{p}attn.wr_du.weight"))?,
            wo: bf16_f32(st, &format!("{p}attn.wo_ud.weight"))?,
            gate: bf16_f32(st, &format!("{p}attn.gate_attn.weight"))?,
            sink: bf16_f32(st, &format!("{p}attn.attn_sink_proj.weight"))?,
            q_scale: bf16_f32(st, &format!("{p}attn.q_l2_clip.scale"))?,
            rel_scale: bf16_f32(st, &format!("{p}attn.rel_l2_clip.scale"))?,
            rel_proj: bf16_f32(st, &format!("{p}attn.rel_logits_proj.proj"))?,
            k_static: bf16_f32(st, &format!("{p}attn.k_sconv.static_w"))?,
            k_wproj: bf16_f32(st, &format!("{p}attn.k_sconv.weight_proj"))?,
            v_static: bf16_f32(st, &format!("{p}attn.v_sconv.static_w"))?,
            v_wproj: bf16_f32(st, &format!("{p}attn.v_sconv.weight_proj"))?,
            gate_w,
            gate_s,
            up_w,
            up_s,
            down_w: u8s(st, &format!("{ep}w2_weight"))?,
            down_s: u8s(st, &format!("{ep}w2_weight_scale"))?,
            gate_up_amax,
            down_amax,
            mxfp8,
            boundary: i.is_multiple_of(6),
            gpk_layer: None,
            ctrl_attn,
            ctrl_k,
            ctrl_v,
            ctrl_mlp,
        })
    }

    pub fn as_weights(&self) -> FullLayerWeights<'_> {
        FullLayerWeights {
            q_norm: &self.q_norm,
            k_norm: &self.k_norm,
            v_norm: &self.v_norm,
            mlp_norm: &self.mlp_norm,
            attn_sconv: &self.attn_sconv,
            mlp_sconv: &self.mlp_sconv,
            attn_queries: &self.attn_queries,
            mlp_query: &self.mlp_query,
            router_w: &self.router_w,
            router_b: &self.router_b,
            attn: FullAttnWeights {
                layer: self.gpk_layer,
                wq: &self.wq,
                wk: &self.wk,
                wv: &self.wv,
                wr: &self.wr,
                wo: &self.wo,
                gate: &self.gate,
                sink: &self.sink,
                q_scale: &self.q_scale,
                rel_scale: &self.rel_scale,
                rel_proj: &self.rel_proj,
                k_static: &self.k_static,
                k_wproj: &self.k_wproj,
                v_static: &self.v_static,
                v_wproj: &self.v_wproj,
            },
            experts: ExpertPack {
                gate_w: &self.gate_w,
                gate_s: &self.gate_s,
                up_w: &self.up_w,
                up_s: &self.up_s,
                down_w: &self.down_w,
                down_s: &self.down_s,
                gate_up_amax: &self.gate_up_amax,
                down_amax: &self.down_amax,
                n_experts: 128,
                mxfp8: self.mxfp8,
                layer: self.gpk_layer,
            },
            is_block_boundary: self.boundary,
            ctrl_attn: self.ctrl_attn.as_ref().map(|(d, f, g)| MessageControllerWeights {
                down: d,
                forget: f,
                gain: g,
            }),
            ctrl_k: self.ctrl_k.as_ref().map(|(d, f, g)| MessageControllerWeights {
                down: d,
                forget: f,
                gain: g,
            }),
            ctrl_v: self.ctrl_v.as_ref().map(|(d, f, g)| MessageControllerWeights {
                down: d,
                forget: f,
                gain: g,
            }),
            ctrl_mlp: self.ctrl_mlp.as_ref().map(|(d, f, g)| MessageControllerWeights {
                down: d,
                forget: f,
                gain: g,
            }),
        }
    }
}

pub fn is_full_attn(layer: usize) -> bool {
    matches!(layer, 5 | 11 | 17 | 23)
}

/// Experts layers 20-23 ship as MXFP8 (E4M3 × E8M0, group 32); earlier layers are NVFP4.
pub fn is_mxfp8(layer: usize) -> bool {
    (20..=23).contains(&layer)
}

#[cfg(test)]
mod tests {
    use super::split_gate_up;

    #[test]
    fn split_gate_up_interleaves_rows() {
        let n = 4usize;
        let rows = 8usize;
        let stride = 3usize;
        let mut fused = vec![0u8; n * rows * stride];
        for e in 0..n {
            for r in 0..rows {
                for s in 0..stride {
                    fused[(e * rows + r) * stride + s] = (e * 16 + r * 2 + s) as u8;
                }
            }
        }
        let (gate, up) = split_gate_up(&fused, n, rows, stride);
        let half = rows / 2;
        assert_eq!(gate.len(), n * half * stride);
        assert_eq!(up.len(), n * half * stride);
        for e in 0..n {
            for i in 0..half {
                let d = (e * half + i) * stride;
                let sg = ((e * rows) + i * 2) * stride;
                let su = ((e * rows) + i * 2 + 1) * stride;
                assert_eq!(&gate[d..d + stride], &fused[sg..sg + stride]);
                assert_eq!(&up[d..d + stride], &fused[su..su + stride]);
            }
        }
    }
}
