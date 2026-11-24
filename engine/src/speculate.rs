//! Greedy generation with the drafter proposing and the target checking.
//!
//! A round starts from the anchor, the newest token the target has chosen but not
//! processed.  The drafter proposes the tokens after it, the target runs the anchor
//! and some of the proposals as one block, and its pick after every position says
//! how many proposals were right.  Those are kept together with the target's own
//! pick after the last of them, which becomes the next anchor.  Every token that
//! comes out is therefore one the target picked: the drafter changes how many arrive
//! per pass, not which.
//!
//! A block costs several plain decode steps, so it only pays when enough proposals
//! are kept.  The drafter says how sure it is of each proposal, and the round checks
//! as many as make the expected tokens per millisecond highest.  When that is no
//! better than a plain step, the round is a plain step, and the drafter is left
//! alone for a while: longer each time it fails to help.

use anyhow::{Context, Result};
use safetensors::tensor::SafeTensors;

use crate::cache::DecodeCaches;
use crate::decode_visit::{
    commit_block_metal, decode_next_id_metal, verify_block_metal, warm_prompt_taps_metal,
};
use crate::drafter::{Drafter, BLOCK, FEAT};
use crate::head::{UNPADDED_VOCAB, VOCAB};
use crate::prefill::HeadBufs;
use crate::visit::{MetalAccel, WeightCache};

/// What the rounds of one generation added up to.
#[derive(Default, Debug, Clone)]
pub struct SpecStats {
    /// Rounds that checked a block, and plain decode steps.
    pub rounds: usize,
    pub plain_steps: usize,
    /// Drafts made, including those no block followed.
    pub drafts: usize,
    /// Proposals sent to the target, and how many it agreed with.
    pub proposed: usize,
    pub accepted: usize,
    /// Rounds by tokens gained: `gained[n]` rounds produced `n` tokens.
    pub gained: [usize; BLOCK + 1],
    pub draft_ms: f64,
    pub verify_ms: f64,
    pub plain_ms: f64,
    /// Checked proposals by the drafter's probability, in tenths: (seen, kept).
    /// A proposal after the first rejected one is not counted.
    pub by_confidence: [(usize, usize); 10],
}

impl SpecStats {
    /// Tokens the blocks produced.
    pub fn block_tokens(&self) -> usize {
        self.gained.iter().enumerate().map(|(n, c)| n * c).sum()
    }
}

/// Running costs, in milliseconds, that decide what a round does.  They start from
/// rough figures and follow what the rounds measure.
#[derive(Debug, Clone)]
pub struct Costs {
    pub plain: f64,
    pub draft: f64,
    /// A block of `n` tokens, by `n`.
    pub verify: [f64; BLOCK + 1],
}

impl Default for Costs {
    fn default() -> Self {
        let mut verify = [0f64; BLOCK + 1];
        for (n, v) in verify.iter_mut().enumerate() {
            *v = 50.0 + 5.5 * n as f64;
        }
        Self { plain: 27.0, draft: 13.0, verify }
    }
}

fn follow(est: &mut f64, seen: f64) {
    *est += 0.2 * (seen - *est);
}

/// How the rounds are run.
#[derive(Debug, Clone)]
pub struct SpecOptions {
    /// Check every block in full and never fall back to plain steps.
    pub always_full: bool,
    /// Plain steps after the first round that did not pay.  It doubles with each
    /// further one, up to `max_pause`.
    pub pause: usize,
    pub max_pause: usize,
}

impl Default for SpecOptions {
    fn default() -> Self {
        Self { always_full: false, pause: 4, max_pause: 32 }
    }
}

fn argmax(logits: &[f32]) -> i32 {
    let mut best = 0;
    for (i, v) in logits.iter().enumerate() {
        if *v > logits[best] {
            best = i;
        }
    }
    best as i32
}

/// How many of the proposals to check: the count that gives the most expected
/// tokens per millisecond, or 0 when a plain step does better.  A proposal counts
/// only if every one before it is kept, so the expected gain of checking `k` is
/// one (the target's own pick) plus the running products of the probabilities.
fn choose(probs: &[f32], costs: &Costs) -> usize {
    // A block has to promise a tenth more than a plain step: its estimate is the
    // less certain of the two.
    let mut best = (0usize, 1.1 / costs.plain);
    let (mut gain, mut all) = (1.0f64, 1.0f64);
    for (i, p) in probs.iter().enumerate() {
        all *= *p as f64;
        gain += all;
        let k = i + 1;
        let rate = gain / (costs.draft + costs.verify[k + 1]);
        if rate > best.1 {
            best = (k, rate);
        }
    }
    best.0
}

/// The state that carries from one round to the next.
pub struct Speculator {
    pub opts: SpecOptions,
    pub costs: Costs,
    pub stats: SpecStats,
    /// Plain steps left before the drafter is tried again, and rounds in a row that
    /// did not pay.
    paused: usize,
    misses: u32,
}

impl Speculator {
    pub fn new(opts: SpecOptions) -> Self {
        Self { opts, costs: Costs::default(), stats: SpecStats::default(), paused: 0, misses: 0 }
    }

    /// A new prompt: the counts start over, the measured costs stay.
    pub fn reset(&mut self) {
        self.stats = SpecStats::default();
        self.paused = 0;
        self.misses = 0;
    }

    fn pause(&mut self) {
        self.misses += 1;
        self.paused = (self.opts.pause << (self.misses - 1).min(8)).min(self.opts.max_pause);
    }

    /// The tokens after `anchor`, the newest token the target has picked but not
    /// processed: between one and `room` of them.  All but the last have been
    /// processed when this returns, and the last is the next anchor.  A token in
    /// `stop_ids` is only ever returned last, so the caller can stop there with the
    /// caches holding nothing after it.  `caches.taps` has to be on: it is how the
    /// drafter sees what the target has processed.
    #[allow(clippy::too_many_arguments)]
    pub fn step(
        &mut self,
        st: &SafeTensors,
        anchor: i32,
        room: usize,
        stop_ids: &[i32],
        head: &HeadBufs,
        drafter: &mut Drafter,
        caches: &mut DecodeCaches,
        accel: &mut MetalAccel,
        cache: &mut WeightCache,
    ) -> Result<Vec<i32>> {
        anyhow::ensure!(room >= 1, "no room for a token");
        let taps = caches.taps.as_mut().context("drafting needs the caches to record taps")?;

        let (stats, costs) = (&mut self.stats, &mut self.costs);
        let mut k = 0;
        let mut draft = None;
        if self.paused == 0 && room > 1 {
            let t = std::time::Instant::now();
            // The drafter catches up on everything processed since its last draft,
            // plain steps included, only now that it is asked for one.
            let feats = std::mem::take(taps);
            drafter.push_context(&feats, feats.len() / FEAT, &mut accel.ops)?;
            drop(feats);
            let d = drafter.draft(anchor, (BLOCK - 1).min(room - 1), head, &mut accel.ops)?;
            let ms = t.elapsed().as_secs_f64() * 1e3;
            stats.draft_ms += ms;
            stats.drafts += 1;
            follow(&mut costs.draft, ms);
            k = if self.opts.always_full { d.tokens.len() } else { choose(&d.probs, costs) };
            draft = Some(d);
        }

        if k == 0 {
            // A plain step.  Its states reach the drafter at the next call.
            let t = std::time::Instant::now();
            let next = decode_next_id_metal(st, anchor, head, caches, accel, cache)?;
            let ms = t.elapsed().as_secs_f64() * 1e3;
            stats.plain_ms += ms;
            stats.plain_steps += 1;
            follow(&mut costs.plain, ms);
            if self.paused > 0 {
                self.paused -= 1;
            } else if draft.is_some() {
                // The draft was not worth checking.
                self.pause();
            }
            return Ok(vec![next]);
        }

        let draft = draft.unwrap();
        let mut block = Vec::with_capacity(k + 1);
        block.push(anchor);
        block.extend_from_slice(&draft.tokens[..k]);
        let t = std::time::Instant::now();
        let pass = verify_block_metal(st, &block, head, caches, accel, cache)?;

        // Proposals the target agrees with, up to the first it does not.  A stop
        // token ends the block as well: nothing after it is wanted.
        let mut kept = 0;
        while kept < k && draft.tokens[kept] == pass.picks[kept] && !stop_ids.contains(&draft.tokens[kept]) {
            kept += 1;
        }
        commit_block_metal(caches, accel, kept + 1)?;
        let ms = t.elapsed().as_secs_f64() * 1e3;
        stats.verify_ms += ms;
        follow(&mut costs.verify[k + 1], ms);
        caches.taps.as_mut().unwrap().extend_from_slice(&pass.feats[..(kept + 1) * FEAT]);

        for i in 0..(kept + 1).min(k) {
            let bin = ((draft.probs[i] * 10.0) as usize).min(9);
            stats.by_confidence[bin].0 += 1;
            stats.by_confidence[bin].1 += (i < kept) as usize;
        }
        // The target's own pick after the last kept proposal comes for free.
        let mut out = draft.tokens[..kept].to_vec();
        out.push(pass.picks[kept]);
        stats.rounds += 1;
        stats.proposed += k;
        stats.accepted += kept;
        stats.gained[out.len()] += 1;
        // A block that produced fewer tokens than the same time in plain steps counts
        // against the drafter.
        if !self.opts.always_full && (out.len() as f64) * self.costs.plain < ms + self.costs.draft {
            if kept == 0 {
                self.pause();
            } else {
                self.misses += 1;
            }
        } else {
            self.misses = 0;
        }
        Ok(out)
    }
}

/// Generate up to `n_new` tokens after `prompt`, greedily, stopping after a token
/// in `stop_ids`.  Returns the new tokens.
#[allow(clippy::too_many_arguments)]
pub fn generate_drafted_metal(
    st: &SafeTensors,
    prompt: &[i32],
    n_new: usize,
    stop_ids: &[i32],
    head: &HeadBufs,
    drafter: &mut Drafter,
    accel: &mut MetalAccel,
    cache: &mut WeightCache,
    opts: &SpecOptions,
) -> Result<(Vec<i32>, SpecStats)> {
    let mut out = Vec::with_capacity(n_new);
    let mut spec = Speculator::new(opts.clone());
    if n_new == 0 {
        return Ok((out, spec.stats));
    }
    let mut warm = warm_prompt_taps_metal(st, prompt, head, accel, cache, prompt.len() + n_new, true)?;
    drafter.reset();
    let mut anchor = argmax(&warm.logits[..UNPADDED_VOCAB.min(VOCAB)]);
    out.push(anchor);
    while out.len() < n_new && !stop_ids.contains(&anchor) {
        let room = n_new - out.len();
        let more = spec.step(st, anchor, room, stop_ids, head, drafter, &mut warm.caches, accel, cache)?;
        anchor = *more.last().unwrap();
        out.extend_from_slice(&more);
    }
    crate::decode_visit::recycle_decode_caches(warm.caches);
    Ok((out, spec.stats))
}
