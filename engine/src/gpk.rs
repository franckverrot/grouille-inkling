//! Packed Inkling weights (`.gpk` v1): mmap + one shared Metal buffer.
//!
//! Experts stay quantized; large matrices are IEEE f16; small tensors f32.
//! Metal binds tensor byte offsets into the single no-copy buffer, so experts and
//! attention projections get no second host copy. The head is the exception:
//! `HeadWeights` still materializes the embedding and the unembed on the host.

use anyhow::{anyhow, bail, Context, Result};
use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

use crate::gpu::{Buffer, Gpu};
use crate::ops::HIDDEN;
use crate::weights::{taps, FullLayerBufs, SlidingLayerBufs};

pub const MAGIC: &[u8; 4] = b"GPK1";
pub const VERSION: u32 = 1;
pub const DTYPE_U8: u32 = 1;
pub const DTYPE_F16: u32 = 2;
pub const DTYPE_F32: u32 = 3;

#[derive(Clone, Debug)]
pub struct TensorMeta {
    pub name: String,
    pub dtype: u32,
    pub offset: u64,
    pub nbytes: u64,
    pub shape: Vec<usize>,
}

/// Open `.gpk`: file-backed mmap + Metal `bytesNoCopy` over the whole file.
pub struct GpkFile {
    pub mmap: Mmap,
    pub metal: Buffer,
    pub toc: HashMap<String, TensorMeta>,
    pub file_size: u64,
}

impl GpkFile {
    pub fn open(path: &Path, gpu: &Gpu) -> Result<Self> {
        let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
        let mmap = unsafe { Mmap::map(&file)? };
        if mmap.len() < 32 {
            bail!("gpk too small");
        }
        if &mmap[0..4] != MAGIC {
            bail!("bad gpk magic");
        }
        let version = u32::from_le_bytes(mmap[4..8].try_into()?);
        if version != VERSION {
            bail!("gpk version {version} != {VERSION}");
        }
        let n = u32::from_le_bytes(mmap[8..12].try_into()?) as usize;
        let header_size = u32::from_le_bytes(mmap[12..16].try_into()?) as usize;
        let file_size = u64::from_le_bytes(mmap[16..24].try_into()?);
        if mmap.len() < file_size as usize {
            bail!("gpk truncated: map {} < file_size {file_size}", mmap.len());
        }

        let mut toc = HashMap::with_capacity(n);
        let mut pos = 32usize;
        for _ in 0..n {
            let nlen = u16::from_le_bytes(mmap[pos..pos + 2].try_into()?) as usize;
            pos += 2;
            let name = std::str::from_utf8(&mmap[pos..pos + nlen])
                .context("gpk name utf8")?
                .to_string();
            pos += nlen;
            let dtype = u32::from_le_bytes(mmap[pos..pos + 4].try_into()?);
            pos += 4;
            let ndim = u32::from_le_bytes(mmap[pos..pos + 4].try_into()?) as usize;
            pos += 4;
            let mut shape = Vec::with_capacity(ndim);
            for i in 0..4 {
                let d = u32::from_le_bytes(mmap[pos..pos + 4].try_into()?) as usize;
                pos += 4;
                if i < ndim {
                    shape.push(d);
                }
            }
            let offset = u64::from_le_bytes(mmap[pos..pos + 8].try_into()?);
            pos += 8;
            let nbytes = u64::from_le_bytes(mmap[pos..pos + 8].try_into()?);
            pos += 8;
            toc.insert(
                name.clone(),
                TensorMeta {
                    name,
                    dtype,
                    offset,
                    nbytes,
                    shape,
                },
            );
        }
        if pos > header_size {
            bail!("toc overran header_size");
        }

        // Page-align length for bytesNoCopy (Gpu rounds up to 16 KiB).
        let metal = unsafe {
            gpu.buffer_no_copy(
                mmap.as_ptr() as *mut std::ffi::c_void,
                file_size as usize,
            )?
        };

        Ok(Self {
            mmap,
            metal,
            toc,
            file_size,
        })
    }

    pub fn meta(&self, name: &str) -> Result<&TensorMeta> {
        self.toc
            .get(name)
            .ok_or_else(|| anyhow!("gpk missing tensor {name}"))
    }

    pub fn bytes(&self, name: &str) -> Result<&[u8]> {
        let m = self.meta(name)?;
        let start = m.offset as usize;
        let end = start + m.nbytes as usize;
        Ok(&self.mmap[start..end])
    }

    pub fn f32_vec(&self, name: &str) -> Result<Vec<f32>> {
        let m = self.meta(name)?;
        if m.dtype != DTYPE_F32 {
            bail!("{name} dtype {} != f32", m.dtype);
        }
        let b = self.bytes(name)?;
        if b.is_empty() {
            return Ok(Vec::new());
        }
        let n = b.len() / 4;
        let mut out = vec![0f32; n];
        out.copy_from_slice(unsafe { std::slice::from_raw_parts(b.as_ptr() as *const f32, n) });
        Ok(out)
    }

    pub fn f16_as_f32(&self, name: &str) -> Result<Vec<f32>> {
        let m = self.meta(name)?;
        if m.dtype != DTYPE_F16 {
            bail!("{name} dtype {} != f16", m.dtype);
        }
        let b = self.bytes(name)?;
        let n = b.len() / 2;
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let bits = u16::from_le_bytes([b[i * 2], b[i * 2 + 1]]);
            out.push(half::f16::from_bits(bits).to_f32());
        }
        Ok(out)
    }

    /// Byte offset into the shared Metal buffer for `setBuffer(..., offset, ...)`.
    pub fn metal_offset(&self, name: &str) -> Result<usize> {
        Ok(self.meta(name)?.offset as usize)
    }
}

/// Head weights from `.gpk`: embed kept as f16 (row lookup), unembed GPU-only.
pub struct GpkHead {
    /// Vocab×H IEEE f16 bytes (mmap view owned via parent GpkFile lifetime; copied for simplicity).
    pub embed_f16: Vec<u16>,
    pub embed_norm: Vec<f32>,
    pub output_query: Vec<f32>,
    pub final_norm: Vec<f32>,
    /// Host f32 unembed only if needed for logits path; prefer Metal f16 view.
    pub unembed_f32: Vec<f32>,
}

impl GpkHead {
    pub fn load(gpk: &GpkFile) -> Result<Self> {
        let emb = gpk.bytes("head.embed")?;
        assert_eq!(emb.len() % 2, 0);
        let mut embed_f16 = vec![0u16; emb.len() / 2];
        for (i, slot) in embed_f16.iter_mut().enumerate() {
            *slot = u16::from_le_bytes([emb[i * 2], emb[i * 2 + 1]]);
        }
        // Materialize unpadded unembed as f32 for any host path; Metal preload uses f16 bytes.
        let un_live = crate::head::UNPADDED_VOCAB * HIDDEN;
        let un_bytes = gpk.bytes("head.unembed")?;
        let mut unembed_f32 = Vec::with_capacity(un_live);
        for i in 0..un_live {
            let bits = u16::from_le_bytes([un_bytes[i * 2], un_bytes[i * 2 + 1]]);
            unembed_f32.push(half::f16::from_bits(bits).to_f32());
        }
        // Pad to full VOCAB rows with zeros for HeadBufs shape compatibility.
        unembed_f32.resize(crate::head::VOCAB * HIDDEN, 0.0);
        Ok(Self {
            embed_f16,
            embed_norm: gpk.f32_vec("head.embed_norm")?,
            output_query: gpk.f32_vec("head.output_query")?,
            final_norm: gpk.f32_vec("head.final_norm")?,
            unembed_f32,
        })
    }

    pub fn to_head_bufs(&self) -> crate::prefill::HeadBufs {
        // Embed: expand f16→f32 once (still ~1.85GB). Prefer embed_tokens_f16 at call sites.
        let mut embed = Vec::with_capacity(self.embed_f16.len());
        for &bits in &self.embed_f16 {
            embed.push(half::f16::from_bits(bits).to_f32());
        }
        crate::prefill::HeadBufs {
            embed,
            embed_norm: self.embed_norm.clone(),
            output_query: self.output_query.clone(),
            final_norm: self.final_norm.clone(),
            unembed: self.unembed_f32.clone(),
        }
    }
}

/// Embed from IEEE f16 table (no host f32 vocab matrix).
pub fn embed_tokens_f16(ids: &[i32], weight_f16: &[u16], norm: &[f32], out: &mut [f32]) {
    use crate::head::VOCAB;
    use crate::ops::rms_norm;
    let tokens = ids.len();
    assert_eq!(weight_f16.len(), VOCAB * HIDDEN);
    assert_eq!(norm.len(), HIDDEN);
    assert_eq!(out.len(), tokens * HIDDEN);
    let mut row = [0f32; HIDDEN];
    for (t, &id) in ids.iter().enumerate() {
        let id = id as usize;
        assert!(id < VOCAB);
        let base = id * HIDDEN;
        for d in 0..HIDDEN {
            row[d] = half::f16::from_bits(weight_f16[base + d]).to_f32();
        }
        rms_norm(&row, Some(norm), &mut out[t * HIDDEN..(t + 1) * HIDDEN]);
    }
}

fn load_ctrl_gpk(
    gpk: &GpkFile,
    pref: &str,
    cname: &str,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    Ok((
        gpk.f32_vec(&format!("{pref}.ctrl_{cname}_down"))?,
        gpk.f32_vec(&format!("{pref}.ctrl_{cname}_forget"))?,
        gpk.f32_vec(&format!("{pref}.ctrl_{cname}_gain"))?,
    ))
}

impl SlidingLayerBufs {
    pub fn load_gpk(gpk: &GpkFile, i: usize, with_controllers: bool) -> Result<Self> {
        let pref = format!("L{i:02}");
        let mxfp8 = crate::weights::is_mxfp8(i);
        let (ctrl_attn, ctrl_k, ctrl_v, ctrl_mlp) = if with_controllers {
            (
                Some(load_ctrl_gpk(gpk, &pref, "attn")?),
                Some(load_ctrl_gpk(gpk, &pref, "k")?),
                Some(load_ctrl_gpk(gpk, &pref, "v")?),
                Some(load_ctrl_gpk(gpk, &pref, "mlp")?),
            )
        } else {
            (None, None, None, None)
        };
        let attn_query = if i == 0 {
            None
        } else {
            Some(gpk.f32_vec(&format!("{pref}.attn_query"))?)
        };
        Ok(Self {
            attn_norm: gpk.f32_vec(&format!("{pref}.attn_norm"))?,
            mlp_norm: gpk.f32_vec(&format!("{pref}.mlp_norm"))?,
            attn_sconv: taps(&gpk.f32_vec(&format!("{pref}.attn_sconv"))?),
            mlp_sconv: taps(&gpk.f32_vec(&format!("{pref}.mlp_sconv"))?),
            attn_query,
            mlp_query: gpk.f32_vec(&format!("{pref}.mlp_query"))?,
            router_w: Vec::new(), // f16 lives in MetalOps (install_attn_views)
            router_b: gpk.f32_vec(&format!("{pref}.router_b"))?,
            wq: Vec::new(), // f16 lives in MetalOps (install_attn_views)
            wk: Vec::new(), // f16 lives in MetalOps (install_attn_views)
            wv: Vec::new(), // f16 lives in MetalOps (install_attn_views)
            wr: Vec::new(), // f16 lives in MetalOps (install_attn_views)
            wo: Vec::new(), // f16 lives in MetalOps (install_attn_views)
            gate: Vec::new(), // f16 lives in MetalOps (install_attn_views)
            sink: Vec::new(), // f16 lives in MetalOps (install_attn_views)
            q_scale: gpk.f16_as_f32(&format!("{pref}.q_scale"))?,
            rel_scale: gpk.f16_as_f32(&format!("{pref}.rel_scale"))?,
            rel_proj: gpk.f16_as_f32(&format!("{pref}.rel_proj"))?,
            k_sconv: taps(&gpk.f32_vec(&format!("{pref}.k_sconv"))?),
            v_sconv: taps(&gpk.f32_vec(&format!("{pref}.v_sconv"))?),
            // Experts stay in the shared Metal view, with no host duplicate.
            gate_w: Vec::new(),
            gate_s: Vec::new(),
            up_w: Vec::new(),
            up_s: Vec::new(),
            down_w: Vec::new(),
            down_s: Vec::new(),
            gate_up_amax: Vec::new(),
            down_amax: Vec::new(),
            mxfp8,
            boundary: i.is_multiple_of(6),
            gpk_layer: Some(i),
            ctrl_attn,
            ctrl_k,
            ctrl_v,
            ctrl_mlp,
        })
    }
}

impl FullLayerBufs {
    pub fn load_gpk(gpk: &GpkFile, i: usize, with_controllers: bool) -> Result<Self> {
        let pref = format!("L{i:02}");
        let mxfp8 = crate::weights::is_mxfp8(i);
        let (ctrl_attn, ctrl_k, ctrl_v, ctrl_mlp) = if with_controllers {
            (
                Some(load_ctrl_gpk(gpk, &pref, "attn")?),
                Some(load_ctrl_gpk(gpk, &pref, "k")?),
                Some(load_ctrl_gpk(gpk, &pref, "v")?),
                Some(load_ctrl_gpk(gpk, &pref, "mlp")?),
            )
        } else {
            (None, None, None, None)
        };
        Ok(Self {
            q_norm: gpk.f32_vec(&format!("{pref}.q_norm"))?,
            k_norm: gpk.f32_vec(&format!("{pref}.k_norm"))?,
            v_norm: gpk.f32_vec(&format!("{pref}.v_norm"))?,
            mlp_norm: gpk.f32_vec(&format!("{pref}.mlp_norm"))?,
            attn_sconv: taps(&gpk.f32_vec(&format!("{pref}.attn_sconv"))?),
            mlp_sconv: taps(&gpk.f32_vec(&format!("{pref}.mlp_sconv"))?),
            attn_queries: gpk.f32_vec(&format!("{pref}.attn_query"))?,
            mlp_query: gpk.f32_vec(&format!("{pref}.mlp_query"))?,
            router_w: Vec::new(), // f16 lives in MetalOps (install_attn_views)
            router_b: gpk.f32_vec(&format!("{pref}.router_b"))?,
            wq: Vec::new(), // f16 lives in MetalOps (install_attn_views)
            wk: Vec::new(), // f16 lives in MetalOps (install_attn_views)
            wv: Vec::new(), // f16 lives in MetalOps (install_attn_views)
            wr: Vec::new(), // f16 lives in MetalOps (install_attn_views)
            wo: Vec::new(), // f16 lives in MetalOps (install_attn_views)
            gate: Vec::new(), // f16 lives in MetalOps (install_attn_views)
            sink: Vec::new(), // f16 lives in MetalOps (install_attn_views)
            q_scale: gpk.f16_as_f32(&format!("{pref}.q_scale"))?,
            rel_scale: gpk.f16_as_f32(&format!("{pref}.rel_scale"))?,
            rel_proj: gpk.f16_as_f32(&format!("{pref}.rel_proj"))?,
            k_static: gpk.f32_vec(&format!("{pref}.k_static"))?,
            k_wproj: gpk.f32_vec(&format!("{pref}.k_wproj"))?,
            v_static: gpk.f32_vec(&format!("{pref}.v_static"))?,
            v_wproj: gpk.f32_vec(&format!("{pref}.v_wproj"))?,
            gate_w: Vec::new(),
            gate_s: Vec::new(),
            up_w: Vec::new(),
            up_s: Vec::new(),
            down_w: Vec::new(),
            down_s: Vec::new(),
            gate_up_amax: Vec::new(),
            down_amax: Vec::new(),
            mxfp8,
            boundary: i.is_multiple_of(6),
            gpk_layer: Some(i),
            ctrl_attn,
            ctrl_k,
            ctrl_v,
            ctrl_mlp,
        })
    }
}

/// Install expert tensors from gpk into MetalMoe as views of the shared buffer (no memcpy).
pub fn install_moe_views(
    gpk: &GpkFile,
    moe: &mut crate::metal_moe::MetalMoe,
) -> Result<()> {
    for i in 0..24usize {
        let pref = format!("L{i:02}");
        if crate::weights::is_mxfp8(i) {
            moe.register_resident_mxfp8_view(
                i,
                gpk.metal.clone(),
                gpk.metal_offset(&format!("{pref}.gate_w"))?,
                gpk.metal_offset(&format!("{pref}.gate_s"))?,
                gpk.metal_offset(&format!("{pref}.up_w"))?,
                gpk.metal_offset(&format!("{pref}.up_s"))?,
                gpk.metal_offset(&format!("{pref}.down_w"))?,
                gpk.metal_offset(&format!("{pref}.down_s"))?,
            )?;
        } else {
            // amax still needs a small buffer: upload once from host slice.
            let gua = gpk.f32_vec(&format!("{pref}.gate_up_amax"))?;
            let da = gpk.f32_vec(&format!("{pref}.down_amax"))?;
            moe.register_resident_nvfp4_view(
                i,
                gpk.metal.clone(),
                gpk.metal_offset(&format!("{pref}.gate_w"))?,
                gpk.metal_offset(&format!("{pref}.gate_s"))?,
                gpk.metal_offset(&format!("{pref}.up_w"))?,
                gpk.metal_offset(&format!("{pref}.up_s"))?,
                gpk.metal_offset(&format!("{pref}.down_w"))?,
                gpk.metal_offset(&format!("{pref}.down_s"))?,
                &gua,
                &da,
            )?;
        }
    }
    Ok(())
}

/// Register the pack's f16 attention and router tensors with the Metal kernels, one
/// copy each.
pub fn install_attn_views(gpk: &GpkFile, ops: &mut crate::metal_ops::MetalOps) -> Result<()> {
    use crate::metal_ops::{
        ATTN_GATE, ATTN_QSCALE, ATTN_RELPROJ, ATTN_RELSCALE, ATTN_ROUTER, ATTN_SINK, ATTN_WK,
        ATTN_WO, ATTN_WQ, ATTN_WR, ATTN_WV,
    };
    let tensors: [(&str, u8); 11] = [
        ("wq", ATTN_WQ),
        ("wk", ATTN_WK),
        ("wv", ATTN_WV),
        ("wr", ATTN_WR),
        ("wo", ATTN_WO),
        ("gate", ATTN_GATE),
        ("sink", ATTN_SINK),
        ("router_w", ATTN_ROUTER),
        ("q_scale", ATTN_QSCALE),
        ("rel_scale", ATTN_RELSCALE),
        ("rel_proj", ATTN_RELPROJ),
    ];
    for i in 0..24usize {
        let pref = format!("L{i:02}");
        for (name, id) in tensors {
            let key = format!("{pref}.{name}");
            let m = gpk.meta(&key)?;
            if m.dtype != DTYPE_F16 {
                bail!("{key} dtype {} != f16", m.dtype);
            }
            let bytes = gpk.bytes(&key)?;
            let buf = ops.attn_f16_buffer_from_bytes(bytes)?;
            ops.register_attn_f16_buffer(i, id, buf);
        }
    }
    Ok(())
}
