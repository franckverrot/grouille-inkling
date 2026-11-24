//! Shared MoE router (softcap sigmoid + top-k) for decode/prefill layers.

use crate::moe::TOP_K;
use crate::ops::{sigmoid_from_neg_logits_inplace, vadd, vneg};

const N_EXP: usize = 128;
const ROUTE_SCALE: f32 = 8.0;

/// Softcap-sigmoid scores, add `b`, take top-`TOP_K`, renormalize by `ROUTE_SCALE`.
pub fn route_from_logits(
    logits: &[f32],
    tokens: usize,
    b: &[f32],
    indices: &mut [i32],
    weights: &mut [f32],
) {
    assert_eq!(b.len(), N_EXP);
    assert_eq!(indices.len(), tokens * TOP_K);
    assert_eq!(weights.len(), tokens * TOP_K);
    let mut scores = [0f32; N_EXP];
    let mut keyed = [0f32; N_EXP];
    let mut order = [0usize; N_EXP];
    for t in 0..tokens {
        vneg(&logits[t * N_EXP..(t + 1) * N_EXP], &mut scores);
        sigmoid_from_neg_logits_inplace(&mut scores);
        vadd(&scores, b, &mut keyed);
        for e in 0..N_EXP {
            order[e] = e;
        }
        // Only need the top-8; partial select beats a full 128-sort per visit.
        order.select_nth_unstable_by(TOP_K - 1, |&i, &j| {
            keyed[j]
                .partial_cmp(&keyed[i])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        order[..TOP_K].sort_unstable_by(|&i, &j| {
            keyed[j]
                .partial_cmp(&keyed[i])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut sum = 0f32;
        for r in 0..TOP_K {
            let e = order[r];
            indices[t * TOP_K + r] = e as i32;
            weights[t * TOP_K + r] = scores[e];
            sum += scores[e];
        }
        let inv = ROUTE_SCALE / (sum + 1e-20);
        for r in 0..TOP_K {
            weights[t * TOP_K + r] *= inv;
        }
    }
}
