//! Inkling visit schedule: prelude layers then multi-pass core.

use crate::ops::HIDDEN;

/// Mini config: core [6,18), 3 passes; layer_indices from the checkpoint.
pub const CORE_START: usize = 6;
pub const CORE_END: usize = 18;
pub const NUM_PASSES: usize = 3;

/// Canonical visit list: prelude 0..5, then core 6..17 × `NUM_PASSES`, postlude 18..23.
pub const N_VISITS: usize = CORE_START + (CORE_END - CORE_START) * NUM_PASSES + (24 - CORE_END);

/// Fixed schedule (48 visits). Prefer this over allocating a Vec every token.
pub const LAYER_INDICES: [usize; N_VISITS] = [
    0, 1, 2, 3, 4, 5, // prelude
    6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, // core pass 1
    6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, // core pass 2
    6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, // core pass 3
    18, 19, 20, 21, 22, 23, // postlude
];

/// Canonical visit list (see [`LAYER_INDICES`]).
#[inline]
pub fn layer_indices() -> &'static [usize] {
    &LAYER_INDICES
}

/// When re-entering `CORE_START`, rebuild the message bank from completed core
/// block states (skipping the prelude partial that sits at `prelude_n_blocks`).
///
/// `block_states` is flat `[n_blocks * H]` for one token; `partial` is `[H]`.
/// Returns rms-normalized bank rows `[n_bank * H]`.
pub fn rebuild_message_bank(
    block_states: &[f32],
    prelude_n_blocks: usize,
    partial: &[f32],
) -> Vec<f32> {
    let mut out = Vec::new();
    rebuild_message_bank_into(block_states, prelude_n_blocks, partial, &mut out);
    out
}

/// Like [`rebuild_message_bank`] writing into a reused buffer.
pub fn rebuild_message_bank_into(
    block_states: &[f32],
    prelude_n_blocks: usize,
    partial: &[f32],
    out: &mut Vec<f32>,
) {
    assert_eq!(partial.len(), HIDDEN);
    let n_blocks = block_states.len() / HIDDEN;
    // Skip prelude blocks and the first core block (prelude partial appended
    // at the first core boundary). Remaining completed core blocks + partial.
    let skip = prelude_n_blocks + 1;
    let n_core = n_blocks.saturating_sub(skip);
    let n = n_core + 1;
    out.resize(n * HIDDEN, 0.0);
    for i in 0..n_core {
        let src = &block_states[(skip + i) * HIDDEN..(skip + i + 1) * HIDDEN];
        crate::ops::rms_norm(src, None, &mut out[i * HIDDEN..(i + 1) * HIDDEN]);
    }
    crate::ops::rms_norm(partial, None, &mut out[n_core * HIDDEN..]);
}

/// Unpack HF `middle_cycle_cross_pass_{,kv_}queries` into `[12*4*H]`.
///
/// Checkpoint layout: each tensor is `[24, H]` = 12 layers × 2 slots.
/// Message slots are `(attn, k, v, mlp) = (cross[0], kv[0], kv[1], cross[1])`.
pub fn unpack_message_queries(cross: &[f32], kv: &[f32]) -> Vec<f32> {
    const N_CORE: usize = CORE_END - CORE_START; // 12
    assert_eq!(cross.len(), N_CORE * 2 * HIDDEN);
    assert_eq!(kv.len(), N_CORE * 2 * HIDDEN);
    let mut out = vec![0f32; N_CORE * 4 * HIDDEN];
    for i in 0..N_CORE {
        let c0 = &cross[(i * 2) * HIDDEN..(i * 2 + 1) * HIDDEN];
        let c1 = &cross[(i * 2 + 1) * HIDDEN..(i * 2 + 2) * HIDDEN];
        let k0 = &kv[(i * 2) * HIDDEN..(i * 2 + 1) * HIDDEN];
        let k1 = &kv[(i * 2 + 1) * HIDDEN..(i * 2 + 2) * HIDDEN];
        let dst = i * 4 * HIDDEN;
        out[dst..dst + HIDDEN].copy_from_slice(c0); // attn
        out[dst + HIDDEN..dst + 2 * HIDDEN].copy_from_slice(k0); // k
        out[dst + 2 * HIDDEN..dst + 3 * HIDDEN].copy_from_slice(k1); // v
        out[dst + 3 * HIDDEN..dst + 4 * HIDDEN].copy_from_slice(c1); // mlp
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn visit_list_matches_inkling_mini() {
        let idx = layer_indices();
        assert_eq!(idx.len(), 6 + 12 * 3 + 6);
        assert_eq!(&idx[..6], &[0, 1, 2, 3, 4, 5]);
        assert_eq!(idx[6], 6);
        assert_eq!(idx[17], 17);
        assert_eq!(idx[18], 6); // second core pass
        assert_eq!(idx[idx.len() - 1], 23);
    }

    #[test]
    fn rebuild_skips_prelude_and_first_core_block() {
        // prelude_n=1, blocks=[p0, core0, core1], partial=p
        // bank rows = rms(core1), rms(partial)
        let mut blocks = vec![1.0f32; 3 * HIDDEN];
        for i in 0..HIDDEN {
            blocks[HIDDEN + i] = 2.0;
            blocks[2 * HIDDEN + i] = 3.0;
        }
        let partial = vec![4.0f32; HIDDEN];
        let bank = rebuild_message_bank(&blocks, 1, &partial);
        assert_eq!(bank.len(), 2 * HIDDEN);
        let mut expect = [0f32; HIDDEN];
        crate::ops::rms_norm(&blocks[2 * HIDDEN..3 * HIDDEN], None, &mut expect);
        for i in 0..HIDDEN {
            assert!((bank[i] - expect[i]).abs() < 1e-6);
        }
        crate::ops::rms_norm(&partial, None, &mut expect);
        for i in 0..HIDDEN {
            assert!((bank[HIDDEN + i] - expect[i]).abs() < 1e-6);
        }
    }
}
