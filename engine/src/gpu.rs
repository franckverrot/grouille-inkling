//! Thin Metal host layer over objc2-metal.
//!
//! Everything the engine needs: device/queue, runtime MSL compilation with
//! function constants, PSO cache, shared/private buffers, zero-copy mmap
//! buffers, command-buffer encoding helpers, and dispatch counting.

use anyhow::{anyhow, bail, Context, Result};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandQueue, MTLCompileOptions, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice, MTLFunctionConstantValues,
    MTLLibrary, MTLMathMode, MTLResourceOptions, MTLSize,
};
use std::collections::HashMap;
use std::ffi::c_void;
use std::ptr::NonNull;

pub type Device = Retained<ProtocolObject<dyn MTLDevice>>;
pub type Queue = Retained<ProtocolObject<dyn MTLCommandQueue>>;
pub type Library = Retained<ProtocolObject<dyn MTLLibrary>>;
pub type Pso = Retained<ProtocolObject<dyn MTLComputePipelineState>>;
pub type Buffer = Retained<ProtocolObject<dyn MTLBuffer>>;
pub type CommandBuffer = Retained<ProtocolObject<dyn MTLCommandBuffer>>;

pub struct Gpu {
    pub device: Device,
    pub queue: Queue,
    pub library: Library,
    psos: HashMap<String, Pso>,
}

/// Value for a Metal function constant.
#[derive(Clone, Copy, Debug)]
pub enum FcValue {
    U32(u32),
    F32(f32),
    Bool(bool),
}

impl Gpu {
    pub fn new(msl_source: &str) -> Result<Self> {
        Self::new_with_math(msl_source, true)
    }

    /// `fast_math=false` compiles IEEE-strict kernels. The bf16-emulation
    /// parity gate uses it so results don't drift with compiler reassociation
    /// whenever a kernel is restructured; production keeps fast math.
    pub fn new_with_math(msl_source: &str, fast_math: bool) -> Result<Self> {
        let device = MTLCreateSystemDefaultDevice().ok_or_else(|| anyhow!("no Metal device"))?;
        // the prefill GEMM needs simdgroup_matrix: Apple7+ (A14 / M1 or newer)
        if !device.supportsFamily(objc2_metal::MTLGPUFamily::Apple7) {
            bail!(
                "GPU '{}' lacks Apple7 family support (A14/M1 or newer required)",
                device.name()
            );
        }
        let queue = device
            .newCommandQueue()
            .ok_or_else(|| anyhow!("no command queue"))?;
        let opts = MTLCompileOptions::new();
        opts.setMathMode(if fast_math {
            MTLMathMode::Fast
        } else {
            MTLMathMode::Safe
        });
        let src = NSString::from_str(msl_source);
        let library = device
            .newLibraryWithSource_options_error(&src, Some(&opts))
            .map_err(|e| anyhow!("MSL compile failed: {e}"))?;
        Ok(Self {
            device,
            queue,
            library,
            psos: HashMap::new(),
        })
    }

    pub fn name(&self) -> String {
        self.device.name().to_string()
    }

    /// Build (and cache) a compute pipeline for `function` specialized with
    /// the given function-constant values (index, value).
    pub fn pso(&mut self, function: &str, constants: &[(usize, FcValue)]) -> Result<Pso> {
        let key = format!("{function}:{constants:?}");
        if let Some(p) = self.psos.get(&key) {
            return Ok(p.clone());
        }
        let fcv = MTLFunctionConstantValues::new();
        for (idx, v) in constants {
            unsafe {
                match v {
                    FcValue::U32(x) => fcv.setConstantValue_type_atIndex(
                        NonNull::from(x).cast::<c_void>(),
                        objc2_metal::MTLDataType::UInt,
                        *idx,
                    ),
                    FcValue::F32(x) => fcv.setConstantValue_type_atIndex(
                        NonNull::from(x).cast::<c_void>(),
                        objc2_metal::MTLDataType::Float,
                        *idx,
                    ),
                    FcValue::Bool(x) => fcv.setConstantValue_type_atIndex(
                        NonNull::from(x).cast::<c_void>(),
                        objc2_metal::MTLDataType::Bool,
                        *idx,
                    ),
                }
            }
        }
        let name = NSString::from_str(function);
        let f = self
            .library
            .newFunctionWithName_constantValues_error(&name, &fcv)
            .map_err(|e| anyhow!("function {function}: {e}"))?;
        let pso = self
            .device
            .newComputePipelineStateWithFunction_error(&f)
            .map_err(|e| anyhow!("pso {function}: {e}"))?;
        self.psos.insert(key, pso.clone());
        Ok(pso)
    }

    /// Shared scratch / resident x / blocks / KV rings. **Hazard-untracked by default**
    /// (fast): safe only because the fused decode segment keeps ONE compute encoder
    /// open (see `MetalOps::pending_enc`), so dispatches are ordered within it.
    /// Code that spreads dependent dispatches over several encoders in one command
    /// buffer must use [`Self::buffer_shared_tracked`] instead.
    ///
    /// `INKLING_TRACKED=1` forces hazard-tracked buffers everywhere (debugging; slower).
    pub fn buffer_shared(&self, len: usize) -> Result<Buffer> {
        if crate::flags::tracked_scratch() {
            self.buffer_shared_tracked(len)
        } else {
            self.device
                .newBufferWithLength_options(
                    len.max(16),
                    MTLResourceOptions::StorageModeShared
                        | MTLResourceOptions::HazardTrackingModeUntracked,
                )
                .ok_or_else(|| anyhow!("buffer alloc {len}"))
        }
    }

    /// Shared, hazard-**tracked**: for scratch that consecutive encoders in one command
    /// buffer write then read (untracked would let Metal overlap them and race).
    pub fn buffer_shared_tracked(&self, len: usize) -> Result<Buffer> {
        self.device
            .newBufferWithLength_options(
                len.max(16),
                MTLResourceOptions::StorageModeShared | MTLResourceOptions::HazardTrackingModeTracked,
            )
            .ok_or_else(|| anyhow!("buffer alloc {len}"))
    }

    /// Shared, untracked, host-written once and only ever read by the GPU
    /// (weights, taps, constants): no hazards possible, skip tracking cost.
    pub fn buffer_shared_readonly(&self, len: usize) -> Result<Buffer> {
        self.device
            .newBufferWithLength_options(
                len.max(16),
                MTLResourceOptions::StorageModeShared
                    | MTLResourceOptions::HazardTrackingModeUntracked,
            )
            .ok_or_else(|| anyhow!("buffer alloc {len}"))
    }

    /// Shared + write-combined: host writes once (resident weight upload), GPU reads.
    pub fn buffer_shared_write_combined(&self, len: usize) -> Result<Buffer> {
        self.device
            .newBufferWithLength_options(
                len.max(16),
                MTLResourceOptions::StorageModeShared
                    | MTLResourceOptions::HazardTrackingModeUntracked
                    | MTLResourceOptions::CPUCacheModeWriteCombined,
            )
            .ok_or_else(|| anyhow!("buffer alloc {len}"))
    }

    /// Zero-copy wrap of a page-aligned mmap region (unified memory).
    ///
    /// # Safety
    /// `ptr` must be page-aligned, valid for `len_rounded` bytes, and outlive
    /// the returned buffer.
    pub unsafe fn buffer_no_copy(&self, ptr: *mut c_void, len: usize) -> Result<Buffer> {
        let page = 16384usize;
        let rounded = len.div_ceil(page) * page;
        self.device
            .newBufferWithBytesNoCopy_length_options_deallocator(
                NonNull::new(ptr).context("null ptr")?,
                rounded,
                MTLResourceOptions::StorageModeShared,
                None,
            )
            .ok_or_else(|| anyhow!("bytesNoCopy failed (ptr align? len {len})"))
    }

    pub fn command_buffer(&self) -> Result<CommandBuffer> {
        self.queue
            .commandBuffer()
            .ok_or_else(|| anyhow!("command buffer"))
    }
}

pub fn write_buffer<T: Copy>(buf: &Buffer, offset_bytes: usize, data: &[T]) {
    unsafe {
        let dst = (buf.contents().as_ptr() as *mut u8).add(offset_bytes) as *mut T;
        std::ptr::copy_nonoverlapping(data.as_ptr(), dst, data.len());
    }
}

pub fn read_buffer<T: Copy>(buf: &Buffer, offset_bytes: usize, out: &mut [T]) {
    unsafe {
        let src = (buf.contents().as_ptr() as *const u8).add(offset_bytes) as *const T;
        std::ptr::copy_nonoverlapping(src, out.as_mut_ptr(), out.len());
    }
}

/// Copy `bytes` between two shared buffers.  The GPU must not be using either.
pub fn copy_buffer(src: &Buffer, src_off: usize, dst: &Buffer, dst_off: usize, bytes: usize) {
    unsafe {
        let s = (src.contents().as_ptr() as *const u8).add(src_off);
        let d = (dst.contents().as_ptr() as *mut u8).add(dst_off);
        std::ptr::copy_nonoverlapping(s, d, bytes);
    }
}

/// Write `data` into a shared buffer as f16, starting at `offset_bytes`.
pub fn write_buffer_f16(buf: &Buffer, offset_bytes: usize, data: &[f32]) {
    use half::slice::HalfFloatSliceExt;
    unsafe {
        let dst = (buf.contents().as_ptr() as *mut u8).add(offset_bytes) as *mut half::f16;
        std::slice::from_raw_parts_mut(dst, data.len()).convert_from_f32_slice(data);
    }
}

/// Read f32 values from a shared buffer and write them to another as f16.
pub fn copy_buffer_f32_to_f16(src: &Buffer, src_off: usize, dst: &Buffer, dst_off: usize, n: usize) {
    use half::slice::HalfFloatSliceExt;
    unsafe {
        let s = (src.contents().as_ptr() as *const u8).add(src_off) as *const f32;
        let d = (dst.contents().as_ptr() as *mut u8).add(dst_off) as *mut half::f16;
        std::slice::from_raw_parts_mut(d, n).convert_from_f32_slice(std::slice::from_raw_parts(s, n));
    }
}

/// One dispatch: (pso, bindings, 2-D threadgroup grid, threads_per_tg,
/// threadgroup memory lengths). Binding = (index, buffer, byte offset).
pub struct Dispatch<'a> {
    pub pso: &'a Pso,
    pub bindings: &'a [(usize, &'a Buffer, usize)],
    pub tgs: (usize, usize),
    pub tg_size: usize,
    pub tg_mem: &'a [(usize, usize)],
}

pub fn encode<'a>(enc: &ProtocolObject<dyn MTLComputeCommandEncoder>, d: &Dispatch<'a>) {
    enc.setComputePipelineState(d.pso);
    for (idx, buf, off) in d.bindings {
        unsafe { enc.setBuffer_offset_atIndex(Some(buf), *off, *idx) };
    }
    for (idx, len) in d.tg_mem {
        unsafe { enc.setThreadgroupMemoryLength_atIndex(*len, *idx) };
    }
    enc.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: d.tgs.0,
            height: d.tgs.1,
            depth: 1,
        },
        MTLSize {
            width: d.tg_size,
            height: 1,
            depth: 1,
        },
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use objc2_metal::MTLCommandEncoder;

    #[test]
    fn smoke_add() {
        let src = r#"
            #include <metal_stdlib>
            using namespace metal;
            constant uint N [[function_constant(0)]];
            kernel void k_add(device const float* a [[buffer(0)]],
                              device float* b [[buffer(1)]],
                              uint gid [[thread_position_in_grid]]) {
                if (gid < N) b[gid] = a[gid] + 1.0f;
            }
        "#;
        let mut gpu = Gpu::new(src).unwrap();
        let pso = gpu.pso("k_add", &[(0, FcValue::U32(64))]).unwrap();
        let a = gpu.buffer_shared(64 * 4).unwrap();
        let b = gpu.buffer_shared(64 * 4).unwrap();
        let data: Vec<f32> = (0..64).map(|i| i as f32).collect();
        write_buffer(&a, 0, &data);
        let cb = gpu.command_buffer().unwrap();
        let enc = cb.computeCommandEncoder().unwrap();
        encode(
            &enc,
            &Dispatch {
                pso: &pso,
                bindings: &[(0, &a, 0), (1, &b, 0)],
                tgs: (1, 1),
                tg_size: 64,
                tg_mem: &[],
            },
        );
        enc.endEncoding();
        cb.commit();
        cb.waitUntilCompleted();
        let mut out = vec![0f32; 64];
        read_buffer(&b, 0, &mut out);
        assert_eq!(out[10], 11.0);
        assert_eq!(out[63], 64.0);
        println!("device: {}", gpu.name());
    }
}
