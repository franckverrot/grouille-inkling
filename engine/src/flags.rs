//! Cached env feature flags (avoid getenv on every MoE/visit call).

use std::sync::OnceLock;

fn profile_raw() -> Option<&'static str> {
    static V: OnceLock<Option<String>> = OnceLock::new();
    V.get_or_init(|| std::env::var("INKLING_PROFILE").ok())
        .as_deref()
}

#[inline]
pub fn profile_moe() -> bool {
    matches!(profile_raw(), Some("moe"))
}

#[inline]
pub fn profile_visits() -> bool {
    matches!(profile_raw(), Some("visits") | Some("2"))
}

/// Wall-time split of prompt processing by phase, summed over visits.
#[inline]
pub fn profile_prefill() -> bool {
    matches!(profile_raw(), Some("prefill"))
}

/// Phases of a prefill visit for [`profile_prefill`], in milliseconds.
/// Stages of a prompt visit: everything before the router logits are read, picking
/// the experts on the host, the experts, and the output conv with the bookkeeping.
pub const PREFILL_PHASES: [&str; 4] = ["to_router", "route", "moe", "tail"];

thread_local! {
    static PREFILL: std::cell::RefCell<[f64; 4]> = const { std::cell::RefCell::new([0.0; 4]) };
}

/// Adds the time since `*t` to phase `i` and restarts the clock. No-op unless profiling.
#[inline]
pub fn prefill_lap(t: &mut Option<std::time::Instant>, i: usize) {
    if let Some(t0) = t {
        let now = std::time::Instant::now();
        PREFILL.with(|c| c.borrow_mut()[i] += (now - *t0).as_secs_f64() * 1000.0);
        *t0 = now;
    }
}

pub fn prefill_take() -> [f64; 4] {
    PREFILL.with(|c| std::mem::replace(&mut *c.borrow_mut(), [0.0; 4]))
}

/// Per-decode-token wall split: CPU encode time and the single GPU wait.
#[inline]
pub fn profile_step() -> bool {
    matches!(profile_raw(), Some("step"))
}

/// Split each fused visit into attn CB + MoE CB; record Metal GPU timestamps.
/// Slower wall tok/s (extra submits); use for MoE-vs-attn ratios, not absolute speed.
#[inline]
pub fn profile_gpu() -> bool {
    matches!(profile_raw(), Some("gpu"))
}

#[inline]
pub fn profile_any() -> bool {
    profile_raw().is_some()
}

/// Wall-clock buckets for [`profile_step`] (one decode token).
#[derive(Clone, Copy, Debug)]
pub struct StepAccum {
    /// CPU time recording the visits onto the open command buffer.
    pub encode_ms: f64,
    /// The single wait at the end of the token (GPU time for the whole step).
    pub end_sync_ms: f64,
    pub fused_visits: u32,
}

impl StepAccum {
    pub const ZERO: Self = Self {
        encode_ms: 0.0,
        end_sync_ms: 0.0,
        fused_visits: 0,
    };

    pub fn take_and_reset() -> Self {
        STEP.with(|c| {
            let mut g = c.borrow_mut();
            let out = *g;
            *g = Self::ZERO;
            out
        })
    }

    pub fn with_mut<R>(f: impl FnOnce(&mut Self) -> R) -> R {
        STEP.with(|c| f(&mut c.borrow_mut()))
    }
}

thread_local! {
    static STEP: std::cell::RefCell<StepAccum> = const { std::cell::RefCell::new(StepAccum::ZERO) };
}

/// Per-visit GPU split for [`profile_gpu`]:
/// proj (pre→GEMV→norm) / sdpa (+wo) / premoe (sconv→mix→route) / moe (+tail).
#[derive(Clone, Copy, Debug)]
pub struct GpuSplitAccum {
    pub proj_gpu_ms: f64,
    pub sdpa_gpu_ms: f64,
    pub premoe_gpu_ms: f64,
    pub moe_gpu_ms: f64,
    pub proj_wall_ms: f64,
    pub sdpa_wall_ms: f64,
    pub premoe_wall_ms: f64,
    pub moe_wall_ms: f64,
    pub seg_gpu_ms: f64,
    pub visits: u32,
    pub segs: u32,
}

impl GpuSplitAccum {
    pub const ZERO: Self = Self {
        proj_gpu_ms: 0.0,
        sdpa_gpu_ms: 0.0,
        premoe_gpu_ms: 0.0,
        moe_gpu_ms: 0.0,
        proj_wall_ms: 0.0,
        sdpa_wall_ms: 0.0,
        premoe_wall_ms: 0.0,
        moe_wall_ms: 0.0,
        seg_gpu_ms: 0.0,
        visits: 0,
        segs: 0,
    };

    pub fn take_and_reset() -> Self {
        GPU_SPLIT.with(|c| {
            let mut g = c.borrow_mut();
            let out = *g;
            *g = Self::ZERO;
            out
        })
    }

    pub fn with_mut<R>(f: impl FnOnce(&mut Self) -> R) -> R {
        GPU_SPLIT.with(|c| f(&mut c.borrow_mut()))
    }

    #[inline]
    pub fn add_proj(&mut self, wall_ms: f64, gpu_ms: f64) {
        self.proj_wall_ms += wall_ms;
        self.proj_gpu_ms += gpu_ms;
        self.visits += 1;
    }

    #[inline]
    pub fn add_sdpa(&mut self, wall_ms: f64, gpu_ms: f64) {
        self.sdpa_wall_ms += wall_ms;
        self.sdpa_gpu_ms += gpu_ms;
    }

    #[inline]
    pub fn add_premoe(&mut self, wall_ms: f64, gpu_ms: f64) {
        self.premoe_wall_ms += wall_ms;
        self.premoe_gpu_ms += gpu_ms;
    }

    #[inline]
    pub fn add_moe(&mut self, wall_ms: f64, gpu_ms: f64) {
        self.moe_wall_ms += wall_ms;
        self.moe_gpu_ms += gpu_ms;
    }

    #[inline]
    pub fn add_seg_gpu(&mut self, gpu_ms: f64) {
        self.seg_gpu_ms += gpu_ms;
        self.segs += 1;
    }

    #[inline]
    pub fn attn_gpu_ms(&self) -> f64 {
        self.proj_gpu_ms + self.sdpa_gpu_ms + self.premoe_gpu_ms
    }
}

thread_local! {
    static GPU_SPLIT: std::cell::RefCell<GpuSplitAccum> =
        const { std::cell::RefCell::new(GpuSplitAccum::ZERO) };
}

#[inline]
pub fn progress() -> bool {
    static F: OnceLock<bool> = OnceLock::new();
    *F.get_or_init(|| std::env::var_os("INKLING_PROGRESS").is_some())
}

/// Scratch/x/blocks/KV buffers are hazard-**untracked** by default (fast). That is safe
/// because the fused decode segment keeps one compute encoder open across all visits,
/// GPU bank reentry and finalize (`MetalOps::pending_enc`), so dispatches are ordered.
/// `INKLING_TRACKED=1` forces hazard-tracked buffers instead (debug / A-B correctness).
#[inline]
pub fn tracked_scratch() -> bool {
    static F: OnceLock<bool> = OnceLock::new();
    *F.get_or_init(|| std::env::var_os("INKLING_TRACKED").is_some())
}

/// Segment compute encoders default to `MTLDispatchType::Concurrent` (with explicit
/// buffer barriers on every producer→consumer edge). `INKLING_SERIAL_ENC=1` forces
/// `Serial` encoders (and skips the barriers) for A/B correctness and perf checks.
#[inline]
pub fn serial_enc() -> bool {
    static F: OnceLock<bool> = OnceLock::new();
    *F.get_or_init(|| std::env::var_os("INKLING_SERIAL_ENC").is_some())
}

/// Debug: `INKLING_DUMP_X=dir` appends the residual stream after every visit to
/// `dir/prefill.bin` ([tokens * HIDDEN] f32 per visit, per prompt chunk) and `dir/decode.bin`
/// ([HIDDEN] f32 per visit per decoded token). Forces a GPU sync after each decode
/// visit, so it is for parity work only (see spike/ref_dump.py).
pub fn dump_dir() -> Option<&'static std::path::Path> {
    static V: OnceLock<Option<std::path::PathBuf>> = OnceLock::new();
    V.get_or_init(|| std::env::var_os("INKLING_DUMP_X").map(std::path::PathBuf::from))
        .as_deref()
}

/// `INKLING_PROMPT_CHUNK=n`: process the prompt at most `n` tokens at a time (up to
/// the ring size).  A small value exercises chunk continuation on a short prompt.
pub fn prompt_chunk() -> usize {
    static V: OnceLock<usize> = OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("INKLING_PROMPT_CHUNK")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&n| n > 0)
            .map_or(crate::cache::RING, |n| n.min(crate::cache::RING))
    })
}

pub fn dump_f32(dir: &std::path::Path, name: &str, data: &[f32]) {
    use std::io::Write;
    let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
    if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(dir.join(name)) {
        let _ = f.write_all(bytes);
    }
}

/// Output rows per simdgroup in the `*_down_combine_dec` kernels (1, 2, 4, 8 or 16; default 2).
#[inline]
pub fn moe_down_rows() -> usize {
    static F: OnceLock<usize> = OnceLock::new();
    *F.get_or_init(|| {
        std::env::var("INKLING_MOE_DOWN_ROWS")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|v| [1usize, 2, 4, 8, 16].contains(v))
            .unwrap_or(2)
    })
}

/// Opt-in lossy 8-bit attention projections + unembed (see `metal_q8.rs`).
#[inline]
pub fn attn_q8() -> bool {
    static F: OnceLock<bool> = OnceLock::new();
    *F.get_or_init(|| std::env::var_os("INKLING_ATTN_Q8").is_some())
}

/// Opt-in lossy f16 storage for the K/V rows a full-attention visit keeps past the
/// ring.  Halves what long context costs in memory.
#[inline]
pub fn kv_f16() -> bool {
    static F: OnceLock<bool> = OnceLock::new();
    *F.get_or_init(|| std::env::var_os("INKLING_KV_F16").is_some())
}

/// `INKLING_BLOCK_BY_CHUNK=1`: check a drafted block with the prompt path's three
/// command buffers per visit instead of one for the whole block.  For telling the two
/// apart when they disagree.
pub fn block_by_chunk() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| std::env::var_os("INKLING_BLOCK_BY_CHUNK").is_some())
}
