//! Walk `layer_indices` with prelude reset + message bank.

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::Result;
use safetensors::tensor::SafeTensors;

use crate::cache::DecodeCaches;
use crate::decoder::sliding_layer_prefill_into;
use crate::layer5::full_layer_prefill_into;
use crate::loop_msg::{gather_messages, MSG_SLOTS};
use crate::loop_sched::{
    layer_indices, rebuild_message_bank_into, unpack_message_queries, CORE_END, CORE_START,
};
use crate::metal_moe::MetalMoe;
use crate::metal_ops::MetalOps;
use crate::metal_prompt::{PromptGpu, PromptLayer};
use crate::ops::HIDDEN;
use crate::weights::{is_full_attn, FullLayerBufs, SlidingLayerBufs};

/// Rebuild per-token message banks (RMS-normed core blocks + partial).
fn rebuild_banks(
    blocks: &[Vec<f32>],
    x: &[f32],
    prelude_n: usize,
    tokens: usize,
    banks: &mut [Vec<f32>],
) {
    assert_eq!(blocks.len(), tokens);
    assert_eq!(banks.len(), tokens);
    // Short chats still have enough RMS work per token to beat spawn overhead at 8.
    const PARALLEL_AT: usize = 8;
    if tokens < PARALLEL_AT {
        for t in 0..tokens {
            rebuild_message_bank_into(
                &blocks[t],
                prelude_n,
                &x[t * HIDDEN..(t + 1) * HIDDEN],
                &mut banks[t],
            );
        }
        return;
    }
    let n_threads = std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(4)
        .min(tokens)
        .max(1);
    let chunk = tokens.div_ceil(n_threads);
    // Split banks into disjoint mutable chunks for scoped threads.
    let mut rest = banks;
    let mut start = 0usize;
    std::thread::scope(|scope| {
        while start < tokens {
            let end = (start + chunk).min(tokens);
            let (mine, next) = rest.split_at_mut(end - start);
            let blocks = &blocks[start..end];
            let x = &x[start * HIDDEN..end * HIDDEN];
            let n = end - start;
            scope.spawn(move || {
                for i in 0..n {
                    rebuild_message_bank_into(
                        &blocks[i],
                        prelude_n,
                        &x[i * HIDDEN..(i + 1) * HIDDEN],
                        &mut mine[i],
                    );
                }
            });
            rest = next;
            start = end;
        }
    });
}

/// Gather per-token messages into `msgs` `[tokens * MSG_SLOTS * H]`.
///
/// Parallel across tokens when the prompt is long enough that spawn pays off.
fn fill_messages(banks: &[Vec<f32>], q: &[f32], tokens: usize, msgs: &mut [f32]) {
    assert_eq!(banks.len(), tokens);
    assert_eq!(msgs.len(), tokens * MSG_SLOTS * HIDDEN);
    const PARALLEL_AT: usize = 8;
    if tokens < PARALLEL_AT {
        for t in 0..tokens {
            let n = banks[t].len() / HIDDEN;
            let mut out = [0f32; MSG_SLOTS * HIDDEN];
            gather_messages(&banks[t], n, q, &mut out);
            msgs[t * MSG_SLOTS * HIDDEN..(t + 1) * MSG_SLOTS * HIDDEN].copy_from_slice(&out);
        }
        return;
    }
    let n_threads = std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(4)
        .min(tokens)
        .max(1);
    let chunk = tokens.div_ceil(n_threads);
    let msgs_ptr = msgs.as_mut_ptr() as usize;
    std::thread::scope(|scope| {
        let mut start = 0usize;
        while start < tokens {
            let end = (start + chunk).min(tokens);
            let banks = &banks[start..end];
            let base = start;
            scope.spawn(move || {
                for (i, bank) in banks.iter().enumerate() {
                    let t = base + i;
                    let n = bank.len() / HIDDEN;
                    let mut out = [0f32; MSG_SLOTS * HIDDEN];
                    gather_messages(bank, n, q, &mut out);
                    let dst = unsafe {
                        std::slice::from_raw_parts_mut(
                            (msgs_ptr as *mut f32).add(t * MSG_SLOTS * HIDDEN),
                            MSG_SLOTS * HIDDEN,
                        )
                    };
                    dst.copy_from_slice(&out);
                }
            });
            start = end;
        }
    });
}

/// What the Metal paths run on: the kernels for everything but the experts, and the
/// experts.
pub struct MetalAccel {
    pub ops: MetalOps,
    pub moe: MetalMoe,
}

impl MetalAccel {
    pub fn new() -> Result<Self> {
        Ok(Self {
            ops: MetalOps::new()?,
            moe: MetalMoe::new()?,
        })
    }
}

pub struct VisitState {
    pub x: Vec<f32>,
    /// Ping-pong partner for warm prefill writes (avoids per-visit `x` alloc).
    x_alt: Vec<f32>,
    pub blocks: Vec<Vec<f32>>,
    pub tokens: usize,
    /// Saved at first `CORE_START` visit; used on reentry.
    pub prelude_x: Option<Vec<f32>>,
    pub prelude_blocks: Option<Vec<Vec<f32>>>,
    /// Per-token message banks `[n_bank * H]`, set on core reentry.
    pub bank: Option<Vec<Vec<f32>>>,
    /// Reused gather_messages output `[tokens * MSG_SLOTS * H]` across core visits.
    msg_scratch: Vec<f32>,
}

impl VisitState {
    pub fn new(x: Vec<f32>) -> Self {
        let tokens = x.len() / HIDDEN;
        let need = tokens * HIDDEN;
        Self {
            x,
            x_alt: Vec::with_capacity(need),
            blocks: (0..tokens).map(|_| Vec::new()).collect(),
            tokens,
            prelude_x: None,
            prelude_blocks: None,
            bank: None,
            msg_scratch: Vec::new(),
        }
    }

    /// Prefill writes into `x_alt`, then swap so `x` holds the new residual.
    fn swap_prefill_out(&mut self) {
        std::mem::swap(&mut self.x, &mut self.x_alt);
    }
}

/// Caches decoded layer weights across the 48-visit schedule.
///
/// Controllers are keyed separately so pass-1 (no messages) and pass-2/3
/// (with controllers) do not thrash each other.
#[derive(Default)]
pub struct WeightCache {
    /// Keyed by layer index only. Core layers always load message controllers on
    /// first touch so pass-2/3 reentry does not re-decode ~0.5 GB expert tensors.
    sliding: HashMap<usize, SlidingLayerBufs>,
    full: HashMap<usize, FullLayerBufs>,
    /// Unpacked middle-cycle message queries (`[12 * 4 * H]`), loaded once.
    msg_queries: Option<Arc<[f32]>>,
    /// True after [`Self::preload_attn_gemv`] has uploaded every layer's proj.
    attn_gemv_preloaded: bool,
}

impl WeightCache {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.sliding.len() + self.full.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// True after attn GEMV weights have been uploaded for all 24 layers.
    pub fn attn_gemv_preloaded(&self) -> bool {
        self.attn_gemv_preloaded
    }

    /// Cross-pass message queries used on core reentry; cached after first load.
    ///
    /// Returns an `Arc` so visit loops can hold queries while mutating layer caches.
    pub fn msg_queries(&mut self, st: &SafeTensors) -> Result<Arc<[f32]>> {
        if self.msg_queries.is_none() {
            let cross = crate::weights::f32s(st, "model.llm.middle_cycle_cross_pass_queries")?;
            let kv = crate::weights::f32s(st, "model.llm.middle_cycle_cross_pass_kv_queries")?;
            self.msg_queries = Some(Arc::from(unpack_message_queries(&cross, &kv)));
        }
        Ok(Arc::clone(self.msg_queries.as_ref().unwrap()))
    }

    pub(crate) fn sliding<'a>(
        &'a mut self,
        st: &SafeTensors,
        index: usize,
        _with_ctrl: bool,
    ) -> Result<&'a SlidingLayerBufs> {
        use std::collections::hash_map::Entry;
        match self.sliding.entry(index) {
            Entry::Occupied(e) => Ok(e.into_mut()),
            Entry::Vacant(e) => {
                // Core layers need controllers on later passes; load once up front.
                let need_ctrl = (CORE_START..CORE_END).contains(&index);
                let buf = SlidingLayerBufs::load(st, index, need_ctrl)?;
                Ok(e.insert(buf))
            }
        }
    }

    pub(crate) fn full<'a>(
        &'a mut self,
        st: &SafeTensors,
        index: usize,
        _with_ctrl: bool,
    ) -> Result<&'a FullLayerBufs> {
        use std::collections::hash_map::Entry;
        match self.full.entry(index) {
            Entry::Occupied(e) => Ok(e.into_mut()),
            Entry::Vacant(e) => {
                let need_ctrl = (CORE_START..CORE_END).contains(&index);
                let buf = FullLayerBufs::load_with_controllers(st, index, need_ctrl)?;
                Ok(e.insert(buf))
            }
        }
    }

    /// Upload the attention projections of every loaded layer as f16, so the first
    /// token does not pay for the conversion.  Empty host slices are skipped: the pack
    /// registers its own f16 buffers.
    pub fn preload_attn_gemv(&mut self, accel: &mut MetalAccel) -> Result<()> {
        if self.attn_gemv_preloaded && self.len() >= 24 {
            return Ok(());
        }
        let sliding = self
            .sliding
            .values()
            .map(|b| [&b.wq, &b.wk, &b.wv, &b.wr, &b.wo, &b.gate, &b.sink]);
        let full = self
            .full
            .values()
            .map(|b| [&b.wq, &b.wk, &b.wv, &b.wr, &b.wo, &b.gate, &b.sink]);
        for w in sliding.chain(full).flatten() {
            if !w.is_empty() {
                accel.ops.preload_weight_f16(w)?;
            }
        }
        if self.len() >= 24 {
            self.attn_gemv_preloaded = true;
        }
        Ok(())
    }

    /// Load all 24 layers from a `.gpk` (experts stay on the shared Metal view).
    pub fn preload_all_from_gpk(&mut self, gpk: &crate::gpk::GpkFile) -> Result<()> {
        const N_LAYERS: usize = 24;
        for index in 0..N_LAYERS {
            let need_ctrl = (CORE_START..CORE_END).contains(&index);
            if is_full_attn(index) {
                if let std::collections::hash_map::Entry::Vacant(e) = self.full.entry(index) {
                    e.insert(FullLayerBufs::load_gpk(gpk, index, need_ctrl)?);
                }
            } else if let std::collections::hash_map::Entry::Vacant(e) = self.sliding.entry(index) {
                e.insert(SlidingLayerBufs::load_gpk(gpk, index, need_ctrl)?);
            }
        }
        // Message queries from gpk.
        if self.msg_queries.is_none() {
            let cross = gpk.f32_vec("msg.cross")?;
            let kv = gpk.f32_vec("msg.kv")?;
            self.msg_queries = Some(Arc::from(unpack_message_queries(&cross, &kv)));
        }
        Ok(())
    }

    /// Decode all 24 layer weight packs in parallel (batched) before the visit loop.
    ///
    /// Overlaps host `split_gate_up` / bf16 work across layers so warm is not
    /// serialized behind one layer at a time.
    pub fn preload_all_layers(&mut self, st: &SafeTensors) -> Result<()> {
        const N_LAYERS: usize = 24;
        if self.len() >= N_LAYERS {
            return Ok(());
        }
        // 12-wide batches beat both 8-wide and a single 24-wide wave on this host.
        const BATCH: usize = 12;
        for start in (0..N_LAYERS).step_by(BATCH) {
            let end = (start + BATCH).min(N_LAYERS);
            let mut sliding_batch = Vec::new();
            let mut full_batch = Vec::new();
            std::thread::scope(|scope| {
                let mut slid_h = Vec::new();
                let mut full_h = Vec::new();
                for index in start..end {
                    if self.sliding.contains_key(&index) || self.full.contains_key(&index) {
                        continue;
                    }
                    let need_ctrl = (CORE_START..CORE_END).contains(&index);
                    if is_full_attn(index) {
                        full_h.push(scope.spawn(move || {
                            FullLayerBufs::load_with_controllers(st, index, need_ctrl)
                                .map(|b| (index, b))
                        }));
                    } else {
                        slid_h.push(scope.spawn(move || {
                            SlidingLayerBufs::load(st, index, need_ctrl).map(|b| (index, b))
                        }));
                    }
                }
                for h in slid_h {
                    sliding_batch.push(h.join().unwrap()?);
                }
                for h in full_h {
                    full_batch.push(h.join().unwrap()?);
                }
                Ok::<(), anyhow::Error>(())
            })?;
            for (index, buf) in sliding_batch {
                self.sliding.insert(index, buf);
            }
            for (index, buf) in full_batch {
                self.full.insert(index, buf);
            }
        }
        Ok(())
    }
}

/// Run visits in `layer_indices()[start_visit..end_visit)` on the host.
///
/// This is the reference the GPU prompt path is checked against.  When `end_visit`
/// is past the first core reentry, message controllers are used for core layers on
/// pass 2 and 3.
pub fn run_visits(
    st: &SafeTensors,
    state: &mut VisitState,
    start_visit: usize,
    end_visit: usize,
) -> Result<()> {
    let mut cache = WeightCache::new();
    run_visits_cached(st, state, start_visit, end_visit, &mut cache)
}

/// Same as [`run_visits`] but reuses a caller-owned [`WeightCache`].
pub fn run_visits_cached(
    st: &SafeTensors,
    state: &mut VisitState,
    start_visit: usize,
    end_visit: usize,
    cache: &mut WeightCache,
) -> Result<()> {
    run_visits_host(st, state, start_visit, end_visit, cache, None)
}

/// [`run_visits_cached`] that also fills `decode_caches` for host decode steps.
pub fn run_visits_into_caches(
    st: &SafeTensors,
    state: &mut VisitState,
    start_visit: usize,
    end_visit: usize,
    cache: &mut WeightCache,
    decode_caches: &mut DecodeCaches,
) -> Result<()> {
    run_visits_host(st, state, start_visit, end_visit, cache, Some(decode_caches))
}

fn run_visits_host(
    st: &SafeTensors,
    state: &mut VisitState,
    start_visit: usize,
    end_visit: usize,
    cache: &mut WeightCache,
    mut decode_caches: Option<&mut DecodeCaches>,
) -> Result<()> {
    let indices = layer_indices();
    assert!(end_visit <= indices.len());
    let msg_queries = cache.msg_queries(st)?;
    for visit in start_visit..end_visit {
        let index = indices[visit];
        if index == CORE_START {
            if visit == index {
                match state.prelude_x.as_mut() {
                    Some(px) => px.clone_from(&state.x),
                    None => state.prelude_x = Some(state.x.clone()),
                }
                match state.prelude_blocks.as_mut() {
                    Some(pb) => pb.clone_from(&state.blocks),
                    None => state.prelude_blocks = Some(state.blocks.clone()),
                }
            } else {
                let pb = state.prelude_blocks.as_ref().expect("prelude");
                let prelude_n = pb[0].len() / HIDDEN;
                let mut banks = state.bank.take().unwrap_or_default();
                if banks.len() < state.tokens {
                    banks.resize_with(state.tokens, Vec::new);
                }
                banks.truncate(state.tokens);
                rebuild_banks(
                    &state.blocks,
                    &state.x,
                    prelude_n,
                    state.tokens,
                    &mut banks,
                );
                state.bank = Some(banks);
                let px = state.prelude_x.as_ref().unwrap();
                state.x.clear();
                state.x.extend_from_slice(px);
                let mut old_blocks = std::mem::take(&mut state.blocks);
                if old_blocks.len() < pb.len() {
                    old_blocks.resize_with(pb.len(), Vec::new);
                }
                old_blocks.truncate(pb.len());
                for (dst, src) in old_blocks.iter_mut().zip(pb.iter()) {
                    dst.clear();
                    dst.extend_from_slice(src);
                }
                state.blocks = old_blocks;
            }
        }

        let with_ctrl = state.bank.is_some() && index < CORE_END;
        if with_ctrl {
            let q = &msg_queries[(index - CORE_START) * MSG_SLOTS * HIDDEN
                ..(index - CORE_START + 1) * MSG_SLOTS * HIDDEN];
            let need = state.tokens * MSG_SLOTS * HIDDEN;
            let mut msgs = std::mem::take(&mut state.msg_scratch);
            msgs.resize(need, 0.0);
            let banks = state.bank.as_ref().unwrap();
            fill_messages(banks, q, state.tokens, &mut msgs);
            state.msg_scratch = msgs;
        }

        let msg_ref = with_ctrl.then_some(state.msg_scratch.as_slice());
        let visit_cache = decode_caches.as_mut().map(|c| &mut c.visits[visit]);
        if is_full_attn(index) {
            let buf = cache.full(st, index, with_ctrl)?;
            full_layer_prefill_into(
                &state.x,
                state.tokens,
                &mut state.blocks,
                &buf.as_weights(),
                msg_ref,
                visit_cache,
                &mut state.x_alt,
            )?;
        } else {
            let buf = cache.sliding(st, index, with_ctrl)?;
            sliding_layer_prefill_into(
                &state.x,
                state.tokens,
                &mut state.blocks,
                &buf.as_weights(),
                msg_ref,
                visit_cache,
                &mut state.x_alt,
            )?;
        }
        state.swap_prefill_out();
    }
    Ok(())
}

/// Visits `[start_visit, end_visit)` over a host state on the GPU prompt path.  The
/// state goes up, the visits run exactly as they do for a prompt, and the state
/// comes back.  The parity tests compare this with [`run_visits`].
pub fn run_visits_metal(
    st: &SafeTensors,
    state: &mut VisitState,
    start_visit: usize,
    end_visit: usize,
    accel: &mut MetalAccel,
) -> Result<()> {
    let mut cache = WeightCache::new();
    run_visits_metal_cached(st, state, start_visit, end_visit, &mut cache, accel)
}

/// Same as [`run_visits_metal`] with a shared [`WeightCache`].
pub fn run_visits_metal_cached(
    st: &SafeTensors,
    state: &mut VisitState,
    start_visit: usize,
    end_visit: usize,
    cache: &mut WeightCache,
    accel: &mut MetalAccel,
) -> Result<()> {
    // Nothing is kept for decode: every visit gets a throwaway history.
    let mut caches = DecodeCaches::for_schedule(layer_indices(), 0);
    run_visits_gpu(st, state, start_visit, end_visit, cache, &mut caches, accel)
}

/// [`run_visits_metal_cached`] that also fills `decode_caches` for Metal decode.
pub fn run_visits_into_caches_metal(
    st: &SafeTensors,
    state: &mut VisitState,
    start_visit: usize,
    end_visit: usize,
    cache: &mut WeightCache,
    decode_caches: &mut DecodeCaches,
    accel: &mut MetalAccel,
) -> Result<()> {
    run_visits_gpu(st, state, start_visit, end_visit, cache, decode_caches, accel)
}

fn run_visits_gpu(
    st: &SafeTensors,
    state: &mut VisitState,
    start_visit: usize,
    end_visit: usize,
    cache: &mut WeightCache,
    decode_caches: &mut DecodeCaches,
    accel: &mut MetalAccel,
) -> Result<()> {
    let mut chunk = accel.ops.prompt_begin(&state.x, state.tokens)?;
    if state.blocks.first().is_some_and(|b| !b.is_empty()) {
        accel.ops.prompt_load_stack(&mut chunk, &state.blocks)?;
    }
    if let (Some(px), Some(pb)) = (&state.prelude_x, &state.prelude_blocks) {
        if !px.is_empty() {
            accel.ops.prompt_load_prelude(&mut chunk, px, pb);
        }
    }
    if let Some(bank) = &state.bank {
        accel.ops.prompt_load_bank(&mut chunk, bank)?;
    }
    prompt_visits(st, &mut chunk, start_visit, end_visit, cache, decode_caches, accel, None)?;
    let ops = &accel.ops;
    ops.prompt_read_x(&chunk, &mut state.x);
    state.blocks = ops.prompt_read_stack(&chunk);
    if let Some((px, pb)) = ops.prompt_read_prelude(&chunk) {
        state.prelude_x = Some(px);
        state.prelude_blocks = Some(pb);
    }
    if chunk.has_bank() {
        state.bank = Some(ops.prompt_read_bank(&chunk));
    }
    Ok(())
}

/// One chunk of prompt tokens through the 48 visits with the stream on the GPU.
/// `x` is the embedded chunk.  Returns the last token's stream and block stack.
pub fn run_prompt_chunk_gpu(
    st: &SafeTensors,
    x: &[f32],
    tokens: usize,
    cache: &mut WeightCache,
    decode_caches: &mut DecodeCaches,
    accel: &mut MetalAccel,
) -> Result<(Vec<f32>, Vec<f32>)> {
    let mut chunk = accel.ops.prompt_begin(x, tokens)?;
    prompt_visits(st, &mut chunk, 0, layer_indices().len(), cache, decode_caches, accel, None)?;
    Ok(accel.ops.prompt_last_token(&chunk))
}

/// Every row a chunk ends with, for callers that need more than its last token.
pub struct ChunkRows {
    pub tokens: usize,
    /// The stream after the last visit, `[tokens][H]`.
    pub x: Vec<f32>,
    /// Each token's block stack.
    pub stacks: Vec<Vec<f32>>,
    /// The states the drafter reads, `[tokens][FEAT]`: the stream entering the
    /// tapped visits.  The fourth state, the final norm, is the caller's to fill.
    pub taps: Vec<f32>,
}

/// [`run_prompt_chunk_gpu`] that returns every token's rows and tapped states.
pub fn run_prompt_chunk_rows(
    st: &SafeTensors,
    x: &[f32],
    tokens: usize,
    cache: &mut WeightCache,
    decode_caches: &mut DecodeCaches,
    accel: &mut MetalAccel,
) -> Result<ChunkRows> {
    let mut chunk = accel.ops.prompt_begin(x, tokens)?;
    let mut taps = vec![0f32; tokens * crate::drafter::FEAT];
    let visits = layer_indices().len();
    prompt_visits(st, &mut chunk, 0, visits, cache, decode_caches, accel, Some(&mut taps))?;
    let mut xs = vec![0f32; tokens * HIDDEN];
    accel.ops.prompt_read_x(&chunk, &mut xs);
    Ok(ChunkRows { tokens, x: xs, stacks: accel.ops.prompt_read_stack(&chunk), taps })
}

/// What a block leaves for its caller.
pub struct BlockOut {
    /// The model's pick after each token.
    pub picks: Vec<i32>,
    /// The states the drafter reads, `[tokens][FEAT]`.
    pub feats: Vec<f32>,
}

/// A block of at most eight tokens through every visit in one command buffer, with
/// nothing committed: each visit's rows end up in its cache's `held`.  The host does
/// nothing between the first visit and the last, which is what makes a short block
/// cheap.
pub fn run_prompt_block_gpu(
    st: &SafeTensors,
    x: &[f32],
    tokens: usize,
    head: &crate::prefill::HeadBufs,
    cache: &mut WeightCache,
    decode_caches: &mut DecodeCaches,
    accel: &mut MetalAccel,
) -> Result<BlockOut> {
    use crate::drafter::{FEAT, TAP_VISITS};
    use anyhow::Context;
    use objc2_metal::{MTLCommandBuffer, MTLCommandEncoder};
    let indices = layer_indices();
    assert_eq!(decode_caches.visits.len(), indices.len());
    let msg_queries = cache.msg_queries(st)?;
    let ops = &mut accel.ops;
    let moe = &mut accel.moe;
    let mut chunk = ops.prompt_begin(x, tokens)?;
    ops.prompt_meta_n = 0;
    let cb = ops.gemv.command_buffer()?;
    let mut rows = Vec::with_capacity(indices.len());
    let mut tap_bufs = Vec::with_capacity(TAP_VISITS.len());
    let finish;
    {
        let enc = cb.computeCommandEncoder().context("encoder")?;
        for (visit, &index) in indices.iter().enumerate() {
            if index == CORE_START {
                if visit == index {
                    ops.block_stash_prelude(&enc, &mut chunk)?;
                } else {
                    ops.block_reenter_core(&enc, &mut chunk)?;
                }
            }
            let with_ctrl = chunk.has_bank() && index < CORE_END;
            let queries = with_ctrl.then(|| {
                let o = (index - CORE_START) * MSG_SLOTS * HIDDEN;
                &msg_queries[o..o + MSG_SLOTS * HIDDEN]
            });
            if let Some(j) = TAP_VISITS.iter().position(|&v| v == visit) {
                tap_bufs.push(ops.block_tap(&enc, &chunk, j)?);
            }
            let visit_cache = &mut decode_caches.visits[visit];
            rows.push(if is_full_attn(index) {
                let buf = cache.full(st, index, with_ctrl)?;
                let w = buf.as_weights();
                ops.block_visit(&enc, moe, &mut chunk, &PromptLayer::full(&w), visit_cache, queries, visit)?
            } else {
                let buf = cache.sliding(st, index, with_ctrl)?;
                let w = buf.as_weights();
                ops.block_visit(&enc, moe, &mut chunk, &PromptLayer::sliding(&w), visit_cache, queries, visit)?
            });
        }
        finish = ops.block_finish(
            &enc,
            &chunk,
            &head.output_query,
            &head.final_norm,
            &head.unembed[..crate::head::UNPADDED_VOCAB * HIDDEN],
        )?;
        enc.endEncoding();
    }
    cb.commit();
    cb.waitUntilCompleted();

    for (visit_cache, r) in decode_caches.visits.iter_mut().zip(&rows) {
        visit_cache.held = Some(crate::metal_ops::MetalOps::read_block_rows(r, tokens));
    }
    let mut taps = vec![0f32; tokens * FEAT];
    let mut xs = vec![0f32; tokens * HIDDEN];
    for (j, b) in tap_bufs.iter().enumerate() {
        crate::gpu::read_buffer(b, 0, &mut xs);
        for (t, row) in xs.chunks_exact(HIDDEN).enumerate() {
            taps[t * FEAT + j * HIDDEN..][..HIDDEN].copy_from_slice(row);
        }
    }
    // The final state is the fourth state the drafter reads.
    let (b_hidden, b_part) = finish;
    crate::gpu::read_buffer(&b_hidden, 0, &mut xs);
    for (t, row) in xs.chunks_exact(HIDDEN).enumerate() {
        taps[t * FEAT + 3 * HIDDEN..][..HIDDEN].copy_from_slice(row);
    }
    // The GPU picked among the vocabulary rows it covers; the last few are done here.
    use crate::head::UNPADDED_VOCAB;
    use crate::metal_draft::HEAD_ROWS;
    let mut picks = Vec::with_capacity(tokens);
    let found = crate::metal_ops::MetalOps::read_block_picks(&b_part, tokens);
    for ((mut best, mut top), h) in found.into_iter().zip(xs.chunks_exact(HIDDEN)) {
        for r in HEAD_ROWS..UNPADDED_VOCAB {
            let w = &head.unembed[r * HIDDEN..(r + 1) * HIDDEN];
            let v: f32 = w.iter().zip(h).map(|(a, b)| a * b).sum();
            if v > top {
                top = v;
                best = r;
            }
        }
        picks.push(best as i32);
    }
    Ok(BlockOut { picks, feats: taps })
}

/// Visits `[start_visit, end_visit)` over a chunk that is already on the GPU.
/// `taps`, `[tokens][FEAT]`, receives the stream entering each tapped visit.
#[allow(clippy::too_many_arguments)]
fn prompt_visits(
    st: &SafeTensors,
    chunk: &mut PromptGpu,
    start_visit: usize,
    end_visit: usize,
    cache: &mut WeightCache,
    decode_caches: &mut DecodeCaches,
    accel: &mut MetalAccel,
    mut taps: Option<&mut [f32]>,
) -> Result<()> {
    let indices = layer_indices();
    assert!(end_visit <= indices.len());
    assert_eq!(decode_caches.visits.len(), indices.len());
    let msg_queries = cache.msg_queries(st)?;
    let ops = &mut accel.ops;
    let moe = &mut accel.moe;
    let profile_visits = crate::flags::profile_visits();
    let mut dump = crate::flags::dump_dir().map(|d| (d, vec![0f32; chunk.tokens * HIDDEN]));
    for visit in start_visit..end_visit {
        let t_visit = profile_visits.then(std::time::Instant::now);
        let index = indices[visit];
        if index == CORE_START {
            if visit == index {
                ops.prompt_stash_prelude(chunk);
            } else {
                ops.prompt_reenter_core(chunk)?;
            }
        }
        // Core passes two and three: each layer reads four messages from the bank.
        let with_ctrl = chunk.has_bank() && index < CORE_END;
        let queries = with_ctrl.then(|| {
            let o = (index - CORE_START) * MSG_SLOTS * HIDDEN;
            &msg_queries[o..o + MSG_SLOTS * HIDDEN]
        });
        if let (Some(taps), Some(j)) =
            (taps.as_deref_mut(), crate::drafter::TAP_VISITS.iter().position(|&v| v == visit))
        {
            let mut xs = vec![0f32; chunk.tokens * HIDDEN];
            ops.prompt_read_x(chunk, &mut xs);
            for (t, row) in xs.chunks_exact(HIDDEN).enumerate() {
                let o = t * crate::drafter::FEAT + j * HIDDEN;
                taps[o..o + HIDDEN].copy_from_slice(row);
            }
        }
        let visit_cache = &mut decode_caches.visits[visit];
        if is_full_attn(index) {
            let buf = cache.full(st, index, with_ctrl)?;
            let w = buf.as_weights();
            ops.prompt_visit(moe, chunk, &PromptLayer::full(&w), visit_cache, queries)?;
        } else {
            let buf = cache.sliding(st, index, with_ctrl)?;
            let w = buf.as_weights();
            ops.prompt_visit(moe, chunk, &PromptLayer::sliding(&w), visit_cache, queries)?;
        }
        if let Some((dir, host)) = dump.as_mut() {
            ops.prompt_read_x(chunk, host);
            crate::flags::dump_f32(dir, "prefill.bin", host);
        }
        if let Some(t0) = t_visit {
            eprintln!(
                "profile visit={visit} layer={index} {:.1} ms",
                t0.elapsed().as_secs_f64() * 1000.0
            );
        }
    }
    Ok(())
}
