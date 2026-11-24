//! Cross-pass message bank + MessageController (Inkling loop).

use crate::ops::{axpy, dot, gemv, gemv_lda, rms_norm, softmax_inplace, HIDDEN};

pub const GATE_RANK: usize = 64;
pub const MSG_SLOTS: usize = 4; // attn, k, v, mlp

pub struct MessageControllerWeights<'a> {
    pub down: &'a [f32],   // [RANK, 2*H]
    pub forget: &'a [f32], // [H, RANK]
    pub gain: &'a [f32],   // [H, RANK]
}

/// `y = (1 - forget) * x + gain * message`
pub fn message_controller(
    x: &[f32],
    message: &[f32],
    w: &MessageControllerWeights<'_>,
    out: &mut [f32],
) {
    assert_eq!(x.len(), HIDDEN);
    assert_eq!(message.len(), HIDDEN);
    assert_eq!(out.len(), HIDDEN);
    let mut xn = [0f32; HIDDEN];
    let mut mn = [0f32; HIDDEN];
    rms_norm(x, None, &mut xn);
    rms_norm(message, None, &mut mn);

    // down is [RANK, 2H]: hidden = W[:, :H] @ xn + W[:, H:] @ mn (no cat copy).
    assert_eq!(w.down.len(), GATE_RANK * 2 * HIDDEN);
    let mut hidden = [0f32; GATE_RANK];
    gemv_lda(
        w.down,
        GATE_RANK,
        HIDDEN,
        2 * HIDDEN,
        0,
        &xn,
        0.0,
        &mut hidden,
    );
    gemv_lda(
        w.down,
        GATE_RANK,
        HIDDEN,
        2 * HIDDEN,
        HIDDEN,
        &mn,
        1.0,
        &mut hidden,
    );

    let mut f_acc = [0f32; HIDDEN];
    let mut g_acc = [0f32; HIDDEN];
    gemv(w.forget, HIDDEN, &hidden, &mut f_acc);
    gemv(w.gain, HIDDEN, &hidden, &mut g_acc);
    #[cfg(target_os = "macos")]
    {
        extern "C" {
            fn vvtanhf(y: *mut f32, x: *const f32, n: *const i32);
        }
        let n_i = HIDDEN as i32;
        unsafe {
            vvtanhf(f_acc.as_mut_ptr(), f_acc.as_ptr(), &n_i);
            vvtanhf(g_acc.as_mut_ptr(), g_acc.as_ptr(), &n_i);
        }
        // out = (1 - forget) * x + gain * message
        #[link(name = "Accelerate", kind = "framework")]
        extern "C" {
            fn vDSP_vsub(
                b: *const f32,
                ib: i32,
                a: *const f32,
                ia: i32,
                c: *mut f32,
                ic: i32,
                n: u32,
            );
            fn vDSP_vmul(
                a: *const f32,
                ia: i32,
                b: *const f32,
                ib: i32,
                c: *mut f32,
                ic: i32,
                n: u32,
            );
            fn vDSP_vadd(
                a: *const f32,
                ia: i32,
                b: *const f32,
                ib: i32,
                c: *mut f32,
                ic: i32,
                n: u32,
            );
        }
        let n = HIDDEN as u32;
        let mut scratch = [0f32; HIDDEN];
        unsafe {
            // out = forget * x; then out = x - out = (1 - forget) * x
            vDSP_vmul(f_acc.as_ptr(), 1, x.as_ptr(), 1, out.as_mut_ptr(), 1, n);
            vDSP_vsub(out.as_ptr(), 1, x.as_ptr(), 1, out.as_mut_ptr(), 1, n);
            vDSP_vmul(
                g_acc.as_ptr(),
                1,
                message.as_ptr(),
                1,
                scratch.as_mut_ptr(),
                1,
                n,
            );
            vDSP_vadd(
                out.as_ptr(),
                1,
                scratch.as_ptr(),
                1,
                out.as_mut_ptr(),
                1,
                n,
            );
        }
    }
    #[cfg(not(target_os = "macos"))]
    {
        for i in 0..HIDDEN {
            let forget = f_acc[i].tanh();
            let gain = g_acc[i].tanh();
            out[i] = (1.0 - forget) * x[i] + gain * message[i];
        }
    }
}

/// Build messages `[4, H]` for one token from bank `[n_states, H]` and queries `[4, H]`.
pub fn gather_messages(
    bank: &[f32],
    n_states: usize,
    queries: &[f32], // [4, H]
    out: &mut [f32], // [4, H]
) {
    assert_eq!(bank.len(), n_states * HIDDEN);
    assert_eq!(queries.len(), MSG_SLOTS * HIDDEN);
    assert_eq!(out.len(), MSG_SLOTS * HIDDEN);
    let scale = (HIDDEN as f32).sqrt().recip();
    if n_states == 1 {
        // Softmax over a single logit is 1; each message is the sole bank row.
        for q in 0..MSG_SLOTS {
            out[q * HIDDEN..(q + 1) * HIDDEN].copy_from_slice(&bank[..HIDDEN]);
        }
        return;
    }
    if (2..=4).contains(&n_states) {
        // Common early-core bank sizes: fixed score arrays, no fill+STACK setup.
        let mut scores = [0f32; 4];
        for q in 0..MSG_SLOTS {
            let query = &queries[q * HIDDEN..(q + 1) * HIDDEN];
            for s in 0..n_states {
                let row = &bank[s * HIDDEN..(s + 1) * HIDDEN];
                scores[s] = dot(query, row) * scale;
            }
            softmax_inplace(&mut scores[..n_states]);
            let dest = &mut out[q * HIDDEN..(q + 1) * HIDDEN];
            dest.copy_from_slice(&bank[..HIDDEN]);
            #[cfg(target_os = "macos")]
            {
                #[link(name = "Accelerate", kind = "framework")]
                extern "C" {
                    fn vDSP_vsmul(
                        a: *const f32,
                        ia: i32,
                        b: *const f32,
                        c: *mut f32,
                        ic: i32,
                        n: u32,
                    );
                }
                unsafe {
                    vDSP_vsmul(
                        dest.as_ptr(),
                        1,
                        &scores[0],
                        dest.as_mut_ptr(),
                        1,
                        HIDDEN as u32,
                    );
                }
            }
            #[cfg(not(target_os = "macos"))]
            {
                let w0 = scores[0];
                for v in dest.iter_mut() {
                    *v *= w0;
                }
            }
            for s in 1..n_states {
                let row = &bank[s * HIDDEN..(s + 1) * HIDDEN];
                axpy(scores[s], row, dest);
            }
        }
        return;
    }
    // Decode banks stay tiny (a few core blocks); avoid per-call heap.
    const STACK: usize = 16;
    if n_states <= STACK {
        let mut scores = [0f32; STACK];
        for q in 0..MSG_SLOTS {
            let query = &queries[q * HIDDEN..(q + 1) * HIDDEN];
            for s in 0..n_states {
                let row = &bank[s * HIDDEN..(s + 1) * HIDDEN];
                scores[s] = dot(query, row) * scale;
            }
            softmax_inplace(&mut scores[..n_states]);
            let dest = &mut out[q * HIDDEN..(q + 1) * HIDDEN];
            dest.fill(0.0);
            for s in 0..n_states {
                let row = &bank[s * HIDDEN..(s + 1) * HIDDEN];
                axpy(scores[s], row, dest);
            }
        }
        return;
    }
    let mut scores = vec![0f32; n_states];
    for q in 0..MSG_SLOTS {
        let query = &queries[q * HIDDEN..(q + 1) * HIDDEN];
        for s in 0..n_states {
            let row = &bank[s * HIDDEN..(s + 1) * HIDDEN];
            scores[s] = dot(query, row) * scale;
        }
        softmax_inplace(&mut scores);
        let dest = &mut out[q * HIDDEN..(q + 1) * HIDDEN];
        dest.fill(0.0);
        for s in 0..n_states {
            let row = &bank[s * HIDDEN..(s + 1) * HIDDEN];
            axpy(scores[s], row, dest);
        }
    }
}
