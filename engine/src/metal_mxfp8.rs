//! Metal MXFP8 GEMV host path.

use anyhow::{Context, Result};
use objc2_metal::{MTLCommandBuffer, MTLCommandEncoder};

use crate::gpu::{encode, read_buffer, write_buffer, Dispatch, FcValue, Gpu};

const MSL: &str = include_str!("../kernels/mxfp8.metal");

pub struct Mxfp8Gemv {
    gpu: Gpu,
}

impl Mxfp8Gemv {
    pub fn new() -> Result<Self> {
        Ok(Self {
            gpu: Gpu::new(MSL).context("compile mxfp8.metal")?,
        })
    }

    pub fn device_name(&self) -> String {
        self.gpu.name()
    }

    /// `packed`: [out, in] u8; `scales`: [out, in/32] u8; `x`: [in] f32 → `y`: [out] f32.
    pub fn gemv(
        &mut self,
        packed: &[u8],
        scales: &[u8],
        out_dim: usize,
        in_dim: usize,
        x: &[f32],
        y: &mut [f32],
    ) -> Result<()> {
        assert_eq!(x.len(), in_dim);
        assert_eq!(y.len(), out_dim);
        assert_eq!(in_dim % 32, 0);
        assert_eq!(packed.len(), out_dim * in_dim);
        assert_eq!(scales.len(), out_dim * (in_dim / 32));

        let pso = self.gpu.pso(
            "k_mxfp8_gemv",
            &[
                (0, FcValue::U32(out_dim as u32)),
                (1, FcValue::U32(in_dim as u32)),
            ],
        )?;
        let b_packed = self.gpu.buffer_shared(packed.len())?;
        let b_scales = self.gpu.buffer_shared(scales.len())?;
        let b_x = self.gpu.buffer_shared(in_dim * 4)?;
        let b_y = self.gpu.buffer_shared(out_dim * 4)?;
        write_buffer(&b_packed, 0, packed);
        write_buffer(&b_scales, 0, scales);
        write_buffer(&b_x, 0, x);

        let tg = 256usize;
        let n_tg = out_dim.div_ceil(tg);
        let cb = self.gpu.command_buffer()?;
        let enc = cb.computeCommandEncoder().context("encoder")?;
        encode(
            &enc,
            &Dispatch {
                pso: &pso,
                bindings: &[
                    (0, &b_packed, 0),
                    (1, &b_scales, 0),
                    (2, &b_x, 0),
                    (3, &b_y, 0),
                ],
                tgs: (n_tg, 1),
                tg_size: tg,
                tg_mem: &[],
            },
        );
        enc.endEncoding();
        cb.commit();
        cb.waitUntilCompleted();
        read_buffer(&b_y, 0, y);
        Ok(())
    }
}
