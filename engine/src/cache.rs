//! Per-visit KV + short-conv caches for decode (48 visits, not 24 layers).

use anyhow::Result;

use crate::attn::{DH, N_KV, SWA};
use crate::gpu::{copy_buffer, copy_buffer_f32_to_f16, Buffer, Gpu};
use crate::ops::{CONV_K, HIDDEN};

/// Rows in a visit's K/V ring.  Equal to the sliding window and to the extent of
/// the relative-position logits, so everything a sliding visit can attend to, and
/// everything that gets a relative logit, is in the ring.
pub const RING: usize = 1024;
const KV_CH: usize = N_KV * DH;
/// Bytes per archive element: 4 for f32, or 2 for f16 with `INKLING_KV_F16`.  The
/// kernels get the matching element type through the `ARCH_F16` macro.
#[inline]
pub fn arch_bytes() -> usize {
    if crate::flags::kv_f16() {
        2
    } else {
        4
    }
}

/// Archive rows per segment.  A multiple of the block and chunk sizes the kernels
/// walk the archive in.
pub const ARCH_SEG: usize = 8192;

/// One visit's K/V history on the GPU.
///
/// The newest [`RING`] rows are f32 in a ring, with position `p` at slot `p % RING`.
/// A full-attention visit also keeps every older row in an archive (f32, or f16
/// with `INKLING_KV_F16`), where row `i` is position `i`.  The archive is a list of
/// segments of [`ARCH_SEG`] rows, so it grows without ever being copied.
pub struct GpuKv {
    pub k_ring: Buffer,
    pub v_ring: Buffer,
    pub k_arch: Vec<Buffer>,
    pub v_arch: Vec<Buffer>,
    /// Tokens this visit has seen.
    pub len: usize,
    pub full: bool,
}

impl GpuKv {
    pub fn new(gpu: &Gpu, full: bool) -> Result<Self> {
        Ok(Self {
            k_ring: gpu.buffer_shared(RING * KV_CH * 4)?,
            v_ring: gpu.buffer_shared(RING * KV_CH * 4)?,
            k_arch: Vec::new(),
            v_arch: Vec::new(),
            len: 0,
            full,
        })
    }

    /// Rows currently valid in the ring.
    pub fn ring_rows(&self) -> usize {
        self.len.min(RING)
    }

    /// Rows currently in the archive.
    pub fn archive_rows(&self) -> usize {
        if self.full {
            self.len.saturating_sub(RING)
        } else {
            0
        }
    }

    /// Forget the history.  The ring buffers are kept, the archive is released.
    pub fn clear(&mut self) {
        self.len = 0;
        self.k_arch.clear();
        self.v_arch.clear();
    }

    /// Make room for `rows` archive rows.
    pub fn reserve_archive(&mut self, gpu: &Gpu, rows: usize) -> Result<()> {
        while self.k_arch.len() * ARCH_SEG < rows {
            self.k_arch.push(gpu.buffer_shared(ARCH_SEG * KV_CH * arch_bytes())?);
            self.v_arch.push(gpu.buffer_shared(ARCH_SEG * KV_CH * arch_bytes())?);
        }
        Ok(())
    }

    /// Segment and byte offset of archive row `row`.
    pub fn archive_slot(row: usize) -> (usize, usize) {
        (row / ARCH_SEG, (row % ARCH_SEG) * KV_CH * arch_bytes())
    }

    /// The segments that hold the first `rows` archive rows, as (K, V, rows in it).
    pub fn archive_segments(&self, rows: usize) -> Vec<(Buffer, Buffer, usize)> {
        (0..rows.div_ceil(ARCH_SEG))
            .map(|s| {
                let n = ARCH_SEG.min(rows - s * ARCH_SEG);
                (self.k_arch[s].clone(), self.v_arch[s].clone(), n)
            })
            .collect()
    }

    /// Append `tokens` rows the caller has on the host (`k`, `v`: `[tokens, N_KV * DH]`).
    /// Rows that fall out of the ring move to the archive on a full-attention
    /// visit and are dropped on a sliding one.  Call only while the GPU is idle.
    pub fn append(&mut self, gpu: &Gpu, k: &[f32], v: &[f32], tokens: usize) -> Result<()> {
        use crate::gpu::{write_buffer, write_buffer_f16};
        assert_eq!(k.len(), tokens * KV_CH);
        assert_eq!(v.len(), tokens * KV_CH);
        let end = self.len + tokens;
        if self.full && end > RING {
            self.reserve_archive(gpu, end - RING)?;
            // Positions that leave the ring: some sit in the ring now, the rest are
            // early rows of this very batch.
            for p in self.len.saturating_sub(RING)..end - RING {
                let (seg, dst) = Self::archive_slot(p);
                let (ka, va) = (&self.k_arch[seg], &self.v_arch[seg]);
                if p < self.len {
                    let src = (p % RING) * KV_CH * 4;
                    if arch_bytes() == 2 {
                        copy_buffer_f32_to_f16(&self.k_ring, src, ka, dst, KV_CH);
                        copy_buffer_f32_to_f16(&self.v_ring, src, va, dst, KV_CH);
                    } else {
                        copy_buffer(&self.k_ring, src, ka, dst, KV_CH * 4);
                        copy_buffer(&self.v_ring, src, va, dst, KV_CH * 4);
                    }
                } else {
                    let i = (p - self.len) * KV_CH;
                    if arch_bytes() == 2 {
                        write_buffer_f16(ka, dst, &k[i..i + KV_CH]);
                        write_buffer_f16(va, dst, &v[i..i + KV_CH]);
                    } else {
                        write_buffer(ka, dst, &k[i..i + KV_CH]);
                        write_buffer(va, dst, &v[i..i + KV_CH]);
                    }
                }
            }
        }
        // Only the last RING rows of the batch can still be in the ring afterwards.
        for i in tokens.saturating_sub(RING)..tokens {
            let slot = (self.len + i) % RING;
            write_buffer(&self.k_ring, slot * KV_CH * 4, &k[i * KV_CH..(i + 1) * KV_CH]);
            write_buffer(&self.v_ring, slot * KV_CH * 4, &v[i * KV_CH..(i + 1) * KV_CH]);
        }
        self.len = end;
        Ok(())
    }

    /// [`Self::append`] for rows that sit in f32 GPU buffers, starting at row `row0`.
    pub fn append_buffers(
        &mut self,
        gpu: &Gpu,
        k_src: &Buffer,
        v_src: &Buffer,
        row0: usize,
        tokens: usize,
    ) -> Result<()> {
        let row = KV_CH * 4;
        let end = self.len + tokens;
        if self.full && end > RING {
            self.reserve_archive(gpu, end - RING)?;
            for p in self.len.saturating_sub(RING)..end - RING {
                let (seg, dst) = Self::archive_slot(p);
                let (ka, va) = (&self.k_arch[seg], &self.v_arch[seg]);
                let (ks, vs, src) = if p < self.len {
                    (&self.k_ring, &self.v_ring, (p % RING) * row)
                } else {
                    (k_src, v_src, (row0 + p - self.len) * row)
                };
                if arch_bytes() == 2 {
                    copy_buffer_f32_to_f16(ks, src, ka, dst, KV_CH);
                    copy_buffer_f32_to_f16(vs, src, va, dst, KV_CH);
                } else {
                    copy_buffer(ks, src, ka, dst, row);
                    copy_buffer(vs, src, va, dst, row);
                }
            }
        }
        for i in tokens.saturating_sub(RING)..tokens {
            let slot = (self.len + i) % RING;
            copy_buffer(k_src, (row0 + i) * row, &self.k_ring, slot * row, row);
            copy_buffer(v_src, (row0 + i) * row, &self.v_ring, slot * row, row);
        }
        self.len = end;
        Ok(())
    }

    /// Copy the ring rows, oldest first, to the start of `k_dst` / `v_dst` (f32).
    /// Returns the number of rows.  Call only while the GPU is idle.
    pub fn unroll_ring(&self, k_dst: &Buffer, v_dst: &Buffer) -> usize {
        let rows = self.ring_rows();
        let row = KV_CH * 4;
        // Oldest position in the ring and the slot it lives in.
        let first = (self.len - rows) % RING;
        let head = (RING - first).min(rows);
        for (src, dst) in [(&self.k_ring, k_dst), (&self.v_ring, v_dst)] {
            copy_buffer(src, first * row, dst, 0, head * row);
            copy_buffer(src, 0, dst, head * row, (rows - head) * row);
        }
        rows
    }
}

/// Rolling short-conv history: `[channels * (CONV_K - 1)]`, oldest first.
pub type ConvState = Vec<f32>;

pub fn empty_conv_state(channels: usize) -> ConvState {
    vec![0f32; channels * (CONV_K - 1)]
}

/// Ring KV cache for one attention visit (sliding or full).
///
/// Rows are stored in a circular buffer. Logical index `0` is always the
/// oldest token; [`Self::row`] maps that to a physical slot so sliding-window
/// decode does not memmove the whole window on every push.
pub struct KvCache {
    pub k: Vec<f32>, // [cap * N_KV * DH]
    pub v: Vec<f32>,
    pub len: usize,
    pub cap: usize,
    /// Physical index of the oldest row when the ring is full; unused while `len < cap`.
    start: usize,
}

impl KvCache {
    pub fn new(cap: usize) -> Self {
        Self {
            k: Vec::new(),
            v: Vec::new(),
            len: 0,
            cap,
            start: 0,
        }
    }

    pub fn sliding() -> Self {
        Self::new(SWA.max(1))
    }

    /// Physical row index for logical position `j` (0 = oldest).
    #[inline]
    pub fn row(&self, j: usize) -> usize {
        debug_assert!(j < self.len);
        if self.len < self.cap {
            j
        } else {
            (self.start + j) % self.cap
        }
    }

    pub fn push(&mut self, k_t: &[f32], v_t: &[f32]) {
        assert_eq!(k_t.len(), N_KV * DH);
        assert_eq!(v_t.len(), N_KV * DH);
        if self.k.is_empty() {
            // Allocated on first use: the Metal path keeps its rows on the GPU.
            self.k = vec![0f32; self.cap * N_KV * DH];
            self.v = vec![0f32; self.cap * N_KV * DH];
        }
        let slot = self.advance_slot();
        let off = slot * N_KV * DH;
        self.k[off..off + N_KV * DH].copy_from_slice(k_t);
        self.v[off..off + N_KV * DH].copy_from_slice(v_t);
    }

    fn advance_slot(&mut self) -> usize {
        if self.len < self.cap {
            let s = self.len;
            self.len += 1;
            s
        } else {
            let s = self.start;
            self.start = (self.start + 1) % self.cap;
            s
        }
    }

    /// Grow capacity without disturbing logical order (full-attn decode).
    pub fn ensure_cap(&mut self, need: usize) {
        if need <= self.cap {
            return;
        }
        let new_cap = need.next_power_of_two().max(self.cap.saturating_mul(2)).max(need);
        if self.k.is_empty() {
            self.cap = new_cap;
            return;
        }
        let row = N_KV * DH;
        let mut k = vec![0f32; new_cap * row];
        let mut v = vec![0f32; new_cap * row];
        if self.start == 0 || self.len < self.cap {
            // Contiguous layout: single memcpy.
            let keep = self.len.min(self.cap) * row;
            k[..keep].copy_from_slice(&self.k[..keep]);
            v[..keep].copy_from_slice(&self.v[..keep]);
        } else {
            // Wrapped ring: copy in logical order so the new buffer is linear.
            for j in 0..self.len {
                let src = self.row(j) * row;
                let dst = j * row;
                k[dst..dst + row].copy_from_slice(&self.k[src..src + row]);
                v[dst..dst + row].copy_from_slice(&self.v[src..src + row]);
            }
        }
        self.k = k;
        self.v = v;
        self.cap = new_cap;
        self.start = 0;
    }
}

/// Conv + KV state for one layer visit in the 48-visit schedule.
pub struct VisitCache {
    pub attn_sconv: ConvState,
    pub mlp_sconv: ConvState,
    pub k_sconv: ConvState,
    pub v_sconv: ConvState,
    /// Host K/V rows.  Only the CPU reference path fills them.
    pub kv: KvCache,
    /// K/V rows on the GPU, created by the first Metal visit.
    pub gpu: Option<GpuKv>,
    pub is_full: bool,
    /// While set, a prompt visit leaves this cache as it is and puts its rows in
    /// `held`: the block is being checked and may be kept only in part.
    pub hold: bool,
    pub held: Option<HeldRows>,
}

/// The rows a visit produced for a block that has not been committed.
pub struct HeldRows {
    pub tokens: usize,
    /// Finished K and V rows, `[tokens][N_KV * DH]`.
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    /// Conv inputs: raw K and V projections, attention output, expert output.
    pub k_in: Vec<f32>,
    pub v_in: Vec<f32>,
    pub attn_in: Vec<f32>,
    pub mlp_in: Vec<f32>,
}

impl VisitCache {
    pub fn sliding() -> Self {
        Self {
            attn_sconv: empty_conv_state(HIDDEN),
            mlp_sconv: empty_conv_state(HIDDEN),
            k_sconv: empty_conv_state(N_KV * DH),
            v_sconv: empty_conv_state(N_KV * DH),
            kv: KvCache::sliding(),
            gpu: None,
            is_full: false,
            hold: false,
            held: None,
        }
    }

    pub fn full(cap: usize) -> Self {
        Self {
            attn_sconv: empty_conv_state(HIDDEN),
            mlp_sconv: empty_conv_state(HIDDEN),
            k_sconv: empty_conv_state(N_KV * DH),
            v_sconv: empty_conv_state(N_KV * DH),
            kv: KvCache::new(cap),
            gpu: None,
            is_full: true,
            hold: false,
            held: None,
        }
    }

    /// Zero conv histories and reset KV length (keeps buffer capacity).
    pub fn clear(&mut self) {
        clear_f32(&mut self.attn_sconv);
        clear_f32(&mut self.mlp_sconv);
        clear_f32(&mut self.k_sconv);
        clear_f32(&mut self.v_sconv);
        self.kv.len = 0;
        self.kv.start = 0;
        self.held = None;
        if let Some(g) = self.gpu.as_mut() {
            g.clear();
        }
    }

    /// Keep the first `keep` tokens of the held block: their K and V rows join the
    /// store and the conv states move on by that many inputs.  The rest is dropped.
    pub fn commit_held(&mut self, gpu: &Gpu, keep: usize) -> Result<()> {
        let Some(h) = self.held.take() else { return Ok(()) };
        assert!(keep >= 1 && keep <= h.tokens);
        let continuing = self.tokens_seen() > 0;
        let tail = (CONV_K - 1).min(keep);
        for (rows, channels, state) in [
            (&h.k_in, KV_CH, &mut self.k_sconv),
            (&h.v_in, KV_CH, &mut self.v_sconv),
            (&h.attn_in, HIDDEN, &mut self.attn_sconv),
            (&h.mlp_in, HIDDEN, &mut self.mlp_sconv),
        ] {
            let last = &rows[(keep - tail) * channels..keep * channels];
            crate::ops::conv_state_roll(last, tail, channels, state, continuing);
        }
        self.gpu_kv(gpu)?.append(gpu, &h.k[..keep * KV_CH], &h.v[..keep * KV_CH], keep)
    }

    /// The visit's GPU K/V store, created on first use.
    pub fn gpu_kv(&mut self, gpu: &Gpu) -> Result<&mut GpuKv> {
        if self.gpu.is_none() {
            self.gpu = Some(GpuKv::new(gpu, self.is_full)?);
        }
        Ok(self.gpu.as_mut().unwrap())
    }

    /// Tokens this visit has seen, wherever its rows are kept.
    pub fn tokens_seen(&self) -> usize {
        self.gpu.as_ref().map_or(self.kv.len, |g| g.len.max(self.kv.len))
    }
}

#[inline]
fn clear_f32(x: &mut [f32]) {
    if x.is_empty() {
        return;
    }
    #[cfg(target_os = "macos")]
    {
        #[link(name = "Accelerate", kind = "framework")]
        extern "C" {
            fn vDSP_vclr(c: *mut f32, ic: i32, n: u32);
        }
        unsafe {
            vDSP_vclr(x.as_mut_ptr(), 1, x.len() as u32);
        }
    }
    #[cfg(not(target_os = "macos"))]
    {
        x.fill(0.0);
    }
}

/// All visit caches for a decode session (one entry per `layer_indices()` slot).
pub struct DecodeCaches {
    pub visits: Vec<VisitCache>,
    /// When set, prompt processing and decode append the target states the drafter
    /// reads for every token they process: `[tokens][4 * HIDDEN]`.
    pub taps: Option<Vec<f32>>,
}

impl DecodeCaches {
    pub fn for_schedule(layer_indices: &[usize], full_cap: usize) -> Self {
        use crate::weights::is_full_attn;
        let visits = layer_indices
            .iter()
            .map(|&layer| {
                if is_full_attn(layer) {
                    VisitCache::full(full_cap)
                } else {
                    VisitCache::sliding()
                }
            })
            .collect();
        Self { visits, taps: None }
    }

    /// Have the next prompt chunk hold its rows instead of committing them.
    pub fn hold_block(&mut self, on: bool) {
        for v in &mut self.visits {
            v.hold = on;
        }
    }

    /// Keep the first `keep` tokens of the block the visits hold.
    pub fn commit_block(&mut self, gpu: &Gpu, keep: usize) -> Result<()> {
        for v in &mut self.visits {
            v.commit_held(gpu, keep)?;
        }
        Ok(())
    }

    /// Ensure full-attention visit KVs can hold `need` tokens (no-op for sliding).
    pub fn ensure_full_cap(&mut self, need: usize) {
        use crate::loop_sched::layer_indices;
        use crate::weights::is_full_attn;
        let indices = layer_indices();
        for (visit, cache) in self.visits.iter_mut().enumerate() {
            if visit < indices.len() && is_full_attn(indices[visit]) {
                cache.kv.ensure_cap(need);
            }
        }
    }

    /// Clear all visit state and grow full-attn caps for a fresh warm.
    pub fn reset_for_warm(&mut self, full_cap: usize) {
        for v in &mut self.visits {
            v.clear();
        }
        if let Some(t) = self.taps.as_mut() {
            t.clear();
        }
        self.ensure_full_cap(full_cap);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kv_cache_ensure_cap_preserves_rows() {
        let mut kv = KvCache::new(2);
        let z = vec![0f32; N_KV * DH];
        let mut a = z.clone();
        a[0] = 1.0;
        let mut b = z.clone();
        b[0] = 2.0;
        kv.push(&a, &z);
        kv.push(&b, &z);
        kv.ensure_cap(5);
        assert!(kv.cap >= 5);
        assert_eq!(kv.len, 2);
        assert_eq!(kv.k[kv.row(0) * N_KV * DH], 1.0);
        assert_eq!(kv.k[kv.row(1) * N_KV * DH], 2.0);
        let mut c = z.clone();
        c[0] = 3.0;
        kv.push(&c, &z);
        assert_eq!(kv.len, 3);
        assert_eq!(kv.k[kv.row(2) * N_KV * DH], 3.0);
    }

    #[test]
    fn kv_cache_rings_at_capacity() {
        let mut kv = KvCache::new(3);
        let z = vec![0f32; N_KV * DH];
        let mut a = z.clone();
        a[0] = 1.0;
        let mut b = z.clone();
        b[0] = 2.0;
        let mut c = z.clone();
        c[0] = 3.0;
        let mut d = z.clone();
        d[0] = 4.0;
        kv.push(&a, &z);
        kv.push(&b, &z);
        kv.push(&c, &z);
        assert_eq!(kv.len, 3);
        assert_eq!(kv.k[kv.row(0) * N_KV * DH], 1.0);
        kv.push(&d, &z);
        assert_eq!(kv.len, 3);
        assert_eq!(kv.k[kv.row(0) * N_KV * DH], 2.0);
        assert_eq!(kv.k[kv.row(1) * N_KV * DH], 3.0);
        assert_eq!(kv.k[kv.row(2) * N_KV * DH], 4.0);
        // No memmove: physical slot 0 holds the newest after one wrap.
        assert_eq!(kv.k[0], 4.0);
    }
}
