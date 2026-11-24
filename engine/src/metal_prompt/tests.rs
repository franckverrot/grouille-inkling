//! The per-token prompt kernels on synthetic data, against the host functions they
//! stand in for.  Needs a GPU and no model weights.

use super::*;
use crate::loop_msg::{gather_messages, message_controller};
use crate::ops::{
    attention_residual_token_extras, dynamic_short_conv_step, l2_clip, rms_norm,
    short_conv_with_state,
};

const T: usize = 40; // not a multiple of the 32-token tile
const TOL: f32 = 2e-5;

struct Rng(u64);

impl Rng {
    fn uniform(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }

    fn vec(&mut self, n: usize, scale: f64) -> Vec<f32> {
        (0..n)
            .map(|_| {
                let r = (-2.0 * self.uniform().ln()).sqrt();
                (scale * r * (std::f64::consts::TAU * self.uniform()).cos()) as f32
            })
            .collect()
    }

    /// Values that survive the trip through f16, for weights the GPU keeps in f16.
    fn vec_f16(&mut self, n: usize, scale: f64) -> Vec<f32> {
        self.vec(n, scale).into_iter().map(|v| half::f16::from_f32(v).to_f32()).collect()
    }
}

fn worst(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

/// Encode with `f` on a serial encoder and wait.
fn run(ops: &mut MetalOps, f: impl FnOnce(&mut MetalOps, &Enc)) {
    ops.prompt_meta_n = 0;
    let cb = ops.gemv.command_buffer().unwrap();
    let enc = cb.computeCommandEncoder().unwrap();
    f(ops, &enc);
    enc.endEncoding();
    cb.commit();
    cb.waitUntilCompleted();
}

fn buf_with(ops: &MetalOps, data: &[f32]) -> Buffer {
    let b = ops.gemv.buffer_shared(data.len().next_multiple_of(4) * 4 + 64).unwrap();
    write_buffer(&b, 0, data);
    b
}

fn read(b: &Buffer, n: usize) -> Vec<f32> {
    let mut out = vec![0f32; n];
    read_buffer(b, 0, &mut out);
    out
}

/// A chunk with `n` random block states per token.
fn chunk(ops: &mut MetalOps, rng: &mut Rng, n: usize) -> (PromptGpu, Vec<f32>, Vec<Vec<f32>>) {
    let x = rng.vec(T * HIDDEN, 1.0);
    let blocks: Vec<Vec<f32>> = (0..T).map(|_| rng.vec(n * HIDDEN, 1.0)).collect();
    let mut st = ops.prompt_begin(&x, T).unwrap();
    ops.prompt_load_stack(&mut st, &blocks).unwrap();
    (st, x, blocks)
}

#[test]
fn mix_matches_host() {
    let mut ops = MetalOps::new().unwrap();
    let mut rng = Rng(11);
    let (st, x, blocks) = chunk(&mut ops, &mut rng, 3);
    let extra = rng.vec(T * HIDDEN, 1.0);
    let query = rng.vec(HIDDEN, 1.0);
    let norm = rng.vec(HIDDEN, 1.0);
    let b_extra = buf_with(&ops, &extra);
    let b_norm = buf_with(&ops, &norm);
    let out = ops.gemv.buffer_shared(st.tp * HIDDEN * 4).unwrap();
    // The stack and two more vectors, as the MLP mix of a block-opening layer, normed.
    run(&mut ops, |ops, enc| {
        ops.enc_mix(enc, &st, (&st.x, Some(&b_extra)), Some(&query), Some(&b_norm), &out).unwrap()
    });
    let got = read(&out, T * HIDDEN);
    let mut want = vec![0f32; T * HIDDEN];
    for t in 0..T {
        let row = t * HIDDEN..(t + 1) * HIDDEN;
        let mut mixed = [0f32; HIDDEN];
        attention_residual_token_extras(
            &blocks[t],
            &[&x[row.clone()], &extra[row.clone()]],
            3,
            &query,
            true,
            &mut mixed,
        );
        rms_norm(&mixed, Some(&norm), &mut want[row]);
    }
    assert!(worst(&got, &want) < TOL, "mix off by {}", worst(&got, &want));

    // No query: the stream passes through and is only normed (layer 0).
    run(&mut ops, |ops, enc| {
        ops.enc_mix(enc, &st, (&st.x, None), None, Some(&b_norm), &out).unwrap()
    });
    let got = read(&out, T * HIDDEN);
    for t in 0..T {
        let row = t * HIDDEN..(t + 1) * HIDDEN;
        rms_norm(&x[row.clone()], Some(&norm), &mut want[row]);
    }
    assert!(worst(&got, &want) < TOL, "pass-through off by {}", worst(&got, &want));
}

#[test]
fn controller_matches_host() {
    let mut ops = MetalOps::new().unwrap();
    let mut rng = Rng(12);
    let (st, x, _) = chunk(&mut ops, &mut rng, 1);
    let msgs = rng.vec(T * MSG_SLOTS * HIDDEN, 1.0);
    write_buffer(&st.msgs, 0, &msgs);
    let down = rng.vec_f16(GATE_RANK * 2 * HIDDEN, 0.02);
    let forget = rng.vec_f16(HIDDEN * GATE_RANK, 0.1);
    let gain = rng.vec_f16(HIDDEN * GATE_RANK, 0.1);
    let norm = rng.vec(HIDDEN, 1.0);
    let c = MessageControllerWeights { down: &down, forget: &forget, gain: &gain };
    let b_norm = buf_with(&ops, &norm);
    let out = ops.gemv.buffer_shared(st.tp * HIDDEN * 4).unwrap();
    for slot in [0, 3] {
        run(&mut ops, |ops, enc| ops.enc_ctrl(enc, &st, &st.x, slot, &c, &b_norm, &out).unwrap());
        let got = read(&out, T * HIDDEN);
        let mut want = vec![0f32; T * HIDDEN];
        for t in 0..T {
            let row = t * HIDDEN..(t + 1) * HIDDEN;
            let m = &msgs[(t * MSG_SLOTS + slot) * HIDDEN..][..HIDDEN];
            let mut blended = [0f32; HIDDEN];
            message_controller(&x[row.clone()], m, &c, &mut blended);
            rms_norm(&blended, Some(&norm), &mut want[row]);
        }
        assert!(worst(&got, &want) < TOL, "controller slot {slot} off by {}", worst(&got, &want));
    }
}

#[test]
fn gather_matches_host() {
    let mut ops = MetalOps::new().unwrap();
    let mut rng = Rng(13);
    for n in [1usize, 3] {
        let (mut st, _, _) = chunk(&mut ops, &mut rng, 0);
        let banks: Vec<Vec<f32>> = (0..T).map(|_| rng.vec(n * HIDDEN, 1.0)).collect();
        ops.prompt_load_bank(&mut st, &banks).unwrap();
        let queries = rng.vec(MSG_SLOTS * HIDDEN, 1.0);
        let b_queries = buf_with(&ops, &queries);
        run(&mut ops, |ops, enc| {
            let meta = ops.pmeta(&[n as u32, st.tp as u32, HIDDEN as u32]).unwrap();
            ops.enc_tok(
                enc,
                "k_tok_gather",
                &[],
                &[(&st.bank, 0), (&b_queries, 0), (&st.msgs, 0), (&meta, 0)],
                T,
                1,
            )
            .unwrap()
        });
        let got = read(&st.msgs, T * MSG_SLOTS * HIDDEN);
        let mut want = vec![0f32; T * MSG_SLOTS * HIDDEN];
        for t in 0..T {
            gather_messages(&banks[t], n, &queries, &mut want[t * MSG_SLOTS * HIDDEN..][..MSG_SLOTS * HIDDEN]);
        }
        assert!(worst(&got, &want) < TOL, "gather over {n} off by {}", worst(&got, &want));
    }
}

/// K (clip) and V (norm) rows through `k_tok_kv_conv`, with a non-zero state.
fn kv_conv(ops: &mut MetalOps, x: &[f32], taps: &[f32], state: &[f32], dyn_taps: Option<&[f32]>, mode: u32) -> Vec<f32> {
    let (b_x, b_taps, b_state) = (buf_with(ops, x), buf_with(ops, taps), buf_with(ops, state));
    let b_dyn = buf_with(ops, dyn_taps.unwrap_or(&[0.0; 4]));
    let out = ops.gemv.buffer_shared(T * KV_CH * 4).unwrap();
    let geom = [(5, FcValue::U32(N_KV as u32)), (6, FcValue::U32(DH as u32))];
    run(ops, |ops, enc| {
        let meta = ops.pmeta(&[mode, dyn_taps.is_some() as u32]).unwrap();
        ops.enc_tok(
            enc,
            "k_tok_kv_conv",
            &geom,
            &[(&b_x, 0), (&b_taps, 0), (&b_state, 0), (&b_dyn, 0), (&out, 0), (&meta, 0)],
            T,
            N_KV,
        )
        .unwrap()
    });
    read(&out, T * KV_CH)
}

/// What the host does to a conv output row by row: clip K, norm V.
fn finish_kv(rows: &mut [f32], mode: u32) {
    for head in rows.chunks_mut(DH) {
        if mode == 0 {
            l2_clip(head, QK_CLIP, None);
        } else {
            let mut n = [0f32; DH];
            rms_norm(head, None, &mut n);
            head.copy_from_slice(&n);
        }
    }
}

#[test]
fn kv_conv_carries_state() {
    let mut ops = MetalOps::new().unwrap();
    let mut rng = Rng(14);
    // Large enough that the K clip is active on some heads.
    let x = rng.vec(T * KV_CH, 3.0);
    let state = rng.vec(KV_CH * (CONV_K - 1), 3.0);

    // Sliding layer: static taps, [channel][4].
    let taps = rng.vec(KV_CH * CONV_K, 0.5);
    let mut conv = vec![0f32; T * KV_CH];
    short_conv_with_state(&x, &taps, T, KV_CH, Some(&state), &mut conv, None);
    for mode in [0u32, 1] {
        let got = kv_conv(&mut ops, &x, &taps, &state, None, mode);
        let mut want = conv.clone();
        finish_kv(&mut want, mode);
        assert!(worst(&got, &want) < TOL, "static conv mode {mode} off by {}", worst(&got, &want));
    }

    // Full-attention layer: static taps [lag][channel] plus a tap per (token, head, lag).
    const DYN_IN: usize = 8;
    let static_w = rng.vec(CONV_K * KV_CH, 0.5);
    let wproj = rng.vec(N_KV * CONV_K * DYN_IN, 0.3);
    let dyn_in = rng.vec(T * DYN_IN, 1.0);
    let dyn_taps: Vec<f32> = (0..T * N_KV * CONV_K)
        .map(|i| {
            let (t, r) = (i / (N_KV * CONV_K), i % (N_KV * CONV_K));
            (0..DYN_IN).map(|j| wproj[r * DYN_IN + j] * dyn_in[t * DYN_IN + j]).sum()
        })
        .collect();
    let mut conv = vec![0f32; T * KV_CH];
    let mut cur = state.clone();
    let mut next = vec![0f32; cur.len()];
    for t in 0..T {
        dynamic_short_conv_step(
            &x[t * KV_CH..(t + 1) * KV_CH],
            &dyn_in[t * DYN_IN..(t + 1) * DYN_IN],
            &static_w,
            &wproj,
            KV_CH,
            DYN_IN,
            N_KV,
            CONV_K,
            &cur,
            &mut conv[t * KV_CH..(t + 1) * KV_CH],
            &mut next,
        );
        std::mem::swap(&mut cur, &mut next);
    }
    for mode in [0u32, 1] {
        let got = kv_conv(&mut ops, &x, &static_w, &state, Some(&dyn_taps), mode);
        let mut want = conv.clone();
        finish_kv(&mut want, mode);
        assert!(worst(&got, &want) < TOL, "dynamic conv mode {mode} off by {}", worst(&got, &want));
    }
}

#[test]
fn hidden_conv_carries_state() {
    let mut ops = MetalOps::new().unwrap();
    let mut rng = Rng(15);
    let x = rng.vec(T * HIDDEN, 1.0);
    let add = rng.vec(T * HIDDEN, 1.0);
    let taps = rng.vec(HIDDEN * CONV_K, 0.5);
    let state = rng.vec(HIDDEN * (CONV_K - 1), 1.0);
    let (b_x, b_add, b_taps, b_state) =
        (buf_with(&ops, &x), buf_with(&ops, &add), buf_with(&ops, &taps), buf_with(&ops, &state));
    let out = ops.gemv.buffer_shared(T * HIDDEN * 4).unwrap();
    let mut conv = vec![0f32; T * HIDDEN];
    short_conv_with_state(&x, &taps, T, HIDDEN, Some(&state), &mut conv, None);
    for has_add in [0u32, 1] {
        run(&mut ops, |ops, enc| {
            let meta = ops.pmeta(&[HIDDEN as u32, has_add]).unwrap();
            ops.enc_tok(
                enc,
                "k_tok_sconv_add",
                &[],
                &[(&b_x, 0), (&b_taps, 0), (&b_state, 0), (&b_add, 0), (&out, 0), (&meta, 0)],
                T,
                1,
            )
            .unwrap()
        });
        let got = read(&out, T * HIDDEN);
        let want: Vec<f32> =
            conv.iter().zip(&add).map(|(c, a)| if has_add == 1 { c + a } else { *c }).collect();
        assert!(worst(&got, &want) < TOL, "hidden conv add={has_add} off by {}", worst(&got, &want));
    }
}

#[test]
fn clip_matches_host() {
    let mut ops = MetalOps::new().unwrap();
    let mut rng = Rng(16);
    for (n, thr) in [(DH, QK_CLIP), (D_REL, REL_CLIP)] {
        // Scaled so that some heads are over the threshold and some under.
        let x = rng.vec(T * N_HEADS * n, 2.5);
        let scale = rng.vec(N_HEADS * n, 1.0);
        let (b_x, b_scale, b_thr) = (buf_with(&ops, &x), buf_with(&ops, &scale), buf_with(&ops, &[thr]));
        run(&mut ops, |ops, enc| {
            let meta = ops.pmeta(&[N_HEADS as u32, n as u32]).unwrap();
            ops.enc_tok(
                enc,
                "k_tok_l2_clip",
                &[],
                &[(&b_x, 0), (&b_scale, 0), (&meta, 0), (&b_thr, 0)],
                T,
                N_HEADS,
            )
            .unwrap()
        });
        let got = read(&b_x, T * N_HEADS * n);
        let mut want = x.clone();
        for (i, head) in want.chunks_mut(n).enumerate() {
            let h = i % N_HEADS;
            l2_clip(head, thr, Some(&scale[h * n..(h + 1) * n]));
        }
        let clipped = want.chunks(n).zip(x.chunks(n).enumerate()).filter(|(w, (i, x))| {
            let h = i % N_HEADS;
            (w[0] - x[0] * scale[h * n]).abs() > 1e-6
        });
        assert!(clipped.count() > 0, "no head of width {n} was clipped");
        assert!(worst(&got, &want) < TOL, "clip width {n} off by {}", worst(&got, &want));
    }
}
