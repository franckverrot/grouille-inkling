//! Shared CPU ops for Inkling decode (f32 reference path).

pub const HIDDEN: usize = 2304;
pub const EPS: f32 = 1e-6;
pub const CONV_K: usize = 4;

/// `out[i] = -x[i]` (Accelerate `vDSP_vneg` on macOS).
#[inline]
pub fn vneg(x: &[f32], out: &mut [f32]) {
    assert_eq!(x.len(), out.len());
    #[cfg(target_os = "macos")]
    {
        #[link(name = "Accelerate", kind = "framework")]
        extern "C" {
            fn vDSP_vneg(a: *const f32, ia: i32, c: *mut f32, ic: i32, n: u32);
        }
        unsafe {
            vDSP_vneg(x.as_ptr(), 1, out.as_mut_ptr(), 1, x.len() as u32);
        }
    }
    #[cfg(not(target_os = "macos"))]
    {
        for (o, &v) in out.iter_mut().zip(x.iter()) {
            *o = -v;
        }
    }
}

/// `out[i] = a[i] + b[i]` (Accelerate `vDSP_vadd` on macOS).
#[inline]
pub fn vadd(a: &[f32], b: &[f32], out: &mut [f32]) {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), out.len());
    #[cfg(target_os = "macos")]
    {
        #[link(name = "Accelerate", kind = "framework")]
        extern "C" {
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
        unsafe {
            vDSP_vadd(
                a.as_ptr(),
                1,
                b.as_ptr(),
                1,
                out.as_mut_ptr(),
                1,
                a.len() as u32,
            );
        }
    }
    #[cfg(not(target_os = "macos"))]
    {
        for i in 0..a.len() {
            out[i] = a[i] + b[i];
        }
    }
}

/// In-place `1 / (1 + exp(x[i]))` where `x` already holds `-logit`.
#[inline]
pub fn sigmoid_from_neg_logits_inplace(x: &mut [f32]) {
    if x.is_empty() {
        return;
    }
    #[cfg(target_os = "macos")]
    {
        #[link(name = "Accelerate", kind = "framework")]
        extern "C" {
            fn vvexpf(y: *mut f32, x: *const f32, n: *const i32);
            fn vDSP_vsadd(
                a: *const f32,
                ia: i32,
                b: *const f32,
                c: *mut f32,
                ic: i32,
                n: u32,
            );
            fn vDSP_svdiv(a: *const f32, b: *const f32, ib: i32, c: *mut f32, ic: i32, n: u32);
        }
        let n_i = x.len() as i32;
        let n_u = x.len() as u32;
        let one = 1.0f32;
        unsafe {
            // x = exp(x); then x = 1 / (1 + x)
            vvexpf(x.as_mut_ptr(), x.as_ptr(), &n_i);
            vDSP_vsadd(x.as_ptr(), 1, &one, x.as_mut_ptr(), 1, n_u);
            vDSP_svdiv(&one, x.as_ptr(), 1, x.as_mut_ptr(), 1, n_u);
        }
    }
    #[cfg(not(target_os = "macos"))]
    {
        for v in x.iter_mut() {
            *v = 1.0 / (1.0 + (*v).exp());
        }
    }
}

#[inline]
pub fn rms_norm(x: &[f32], weight: Option<&[f32]>, out: &mut [f32]) {
    let n = x.len();
    assert_eq!(out.len(), n);
    if let Some(w) = weight {
        assert_eq!(w.len(), n);
    }
    #[cfg(target_os = "macos")]
    {
        extern "C" {
            fn vDSP_svesq(a: *const f32, ia: i32, c: *mut f32, n: u32);
            fn vDSP_vsmul(a: *const f32, ia: i32, b: *const f32, c: *mut f32, ic: i32, n: u32);
            fn vDSP_vmul(
                a: *const f32,
                ia: i32,
                b: *const f32,
                ib: i32,
                c: *mut f32,
                ic: i32,
                n: u32,
            );
        }
        let mut sumsq = 0f32;
        unsafe {
            vDSP_svesq(x.as_ptr(), 1, &mut sumsq, n as u32);
        }
        let inv = (sumsq / n as f32 + EPS).sqrt().recip();
        match weight {
            Some(w) => unsafe {
                vDSP_vmul(x.as_ptr(), 1, w.as_ptr(), 1, out.as_mut_ptr(), 1, n as u32);
                vDSP_vsmul(out.as_ptr(), 1, &inv, out.as_mut_ptr(), 1, n as u32);
            },
            None => unsafe {
                vDSP_vsmul(x.as_ptr(), 1, &inv, out.as_mut_ptr(), 1, n as u32);
            },
        }
    }
    #[cfg(not(target_os = "macos"))]
    {
        let mut var = 0f32;
        for &v in x {
            var += v * v;
        }
        var /= n as f32;
        let inv = (var + EPS).sqrt().recip();
        if let Some(w) = weight {
            for i in 0..n {
                out[i] = x[i] * inv * w[i];
            }
        } else {
            for i in 0..n {
                out[i] = x[i] * inv;
            }
        }
    }
}

/// Depthwise causal conv L=4 + residual, `/sqrt(5)`.
/// `taps` is `[C * K]` row-major `(c, k)`; `x`/`out` are `[T * C]`.
pub fn short_conv(x: &[f32], taps: &[f32], tokens: usize, channels: usize, out: &mut [f32]) {
    short_conv_with_state(x, taps, tokens, channels, None, out, None);
}

/// T=1 decode short-conv that rotates `state` in place (no hist temp).
///
/// `state` is `[channels * (CONV_K - 1)]` oldest-first; `x`/`out` are `[channels]`.
pub fn short_conv_decode_inplace(
    x: &[f32],
    taps: &[f32],
    channels: usize,
    state: &mut [f32],
    out: &mut [f32],
) {
    debug_assert_eq!(CONV_K, 4);
    let hist = CONV_K - 1;
    assert_eq!(x.len(), channels);
    assert_eq!(out.len(), channels);
    assert_eq!(taps.len(), channels * CONV_K);
    assert_eq!(state.len(), channels * hist);
    let scale = (CONV_K as f32 + 1.0).sqrt().recip();
    // Manual 4-way unroll: decode fires this 2×/visit ×48 visits/token.
    let mut c = 0usize;
    while c + 4 <= channels {
        for d in 0..4 {
            let cc = c + d;
            let tbase = cc * CONV_K;
            let sbase = cc * hist;
            let xv = x[cc];
            let s0 = state[sbase];
            let s1 = state[sbase + 1];
            let s2 = state[sbase + 2];
            let acc = taps[tbase] * s0
                + taps[tbase + 1] * s1
                + taps[tbase + 2] * s2
                + taps[tbase + 3] * xv;
            out[cc] = (acc + xv) * scale;
            state[sbase] = s1;
            state[sbase + 1] = s2;
            state[sbase + 2] = xv;
        }
        c += 4;
    }
    while c < channels {
        let tbase = c * CONV_K;
        let sbase = c * hist;
        let xv = x[c];
        let s0 = state[sbase];
        let s1 = state[sbase + 1];
        let s2 = state[sbase + 2];
        let acc = taps[tbase] * s0
            + taps[tbase + 1] * s1
            + taps[tbase + 2] * s2
            + taps[tbase + 3] * xv;
        out[c] = (acc + xv) * scale;
        state[sbase] = s1;
        state[sbase + 1] = s2;
        state[sbase + 2] = xv;
        c += 1;
    }
}

/// Prefill/decode short-conv with optional rolling state.
///
/// `state_in` / `state_out` are `[channels * (CONV_K - 1)]` holding the last
/// `CONV_K - 1` inputs per channel (oldest first). When `state_in` is `None`,
/// missing taps are treated as zero (prefill).
pub fn short_conv_with_state(
    x: &[f32],
    taps: &[f32],
    tokens: usize,
    channels: usize,
    state_in: Option<&[f32]>,
    out: &mut [f32],
    mut state_out: Option<&mut [f32]>,
) {
    assert_eq!(x.len(), tokens * channels);
    assert_eq!(taps.len(), channels * CONV_K);
    assert_eq!(out.len(), tokens * channels);
    let hist = CONV_K - 1;
    if let Some(s) = state_in {
        assert_eq!(s.len(), channels * hist);
    }
    if let Some(ref s) = state_out {
        assert_eq!(s.len(), channels * hist);
    }
    let scale = (CONV_K as f32 + 1.0).sqrt().recip();
    // Decode hot path: T=1 with rolling state: unrolled taps, no per-k branch.
    if tokens == 1 && hist == 3 && CONV_K == 4 {
        if let Some(s) = state_in {
            for c in 0..channels {
                let tbase = c * CONV_K;
                let sbase = c * hist;
                let xv = x[c];
                let acc = taps[tbase] * s[sbase]
                    + taps[tbase + 1] * s[sbase + 1]
                    + taps[tbase + 2] * s[sbase + 2]
                    + taps[tbase + 3] * xv;
                out[c] = (acc + xv) * scale;
                if let Some(ref mut sout) = state_out {
                    sout[sbase] = s[sbase + 1];
                    sout[sbase + 1] = s[sbase + 2];
                    sout[sbase + 2] = xv;
                }
            }
            return;
        }
    }
    for t in 0..tokens {
        for c in 0..channels {
            let mut acc = 0f32;
            for k in 0..CONV_K {
                let src = t as isize + k as isize - hist as isize;
                let v = if src < 0 {
                    match state_in {
                        Some(s) => {
                            // state layout: per-channel [oldest .. newest] of length hist
                            let age = (src + hist as isize) as usize;
                            s[c * hist + age]
                        }
                        None => 0f32,
                    }
                } else {
                    x[src as usize * channels + c]
                };
                acc += taps[c * CONV_K + k] * v;
            }
            let xv = x[t * channels + c];
            out[t * channels + c] = (acc + xv) * scale;
        }
    }
    if let Some(sout) = state_out {
        // Last hist inputs across the sequence (or padded with state_in).
        for c in 0..channels {
            for age in 0..hist {
                let src = tokens as isize - hist as isize + age as isize;
                sout[c * hist + age] = if src < 0 {
                    match state_in {
                        Some(s) => s[c * hist + (src + hist as isize) as usize],
                        None => 0f32,
                    }
                } else {
                    x[src as usize * channels + c]
                };
            }
        }
    }
}

/// Roll a short-conv state forward over a chunk of inputs.  `continuing` keeps what
/// the previous chunk left when this one is shorter than the history.
pub fn conv_state_roll(x: &[f32], tokens: usize, channels: usize, state: &mut [f32], continuing: bool) {
    let hist = CONV_K - 1;
    if tokens >= hist || !continuing {
        short_conv_tail_state(x, tokens, channels, state);
        return;
    }
    for c in 0..channels {
        let base = c * hist;
        for i in 0..hist - tokens {
            state[base + i] = state[base + i + tokens];
        }
        for t in 0..tokens {
            state[base + hist - tokens + t] = x[t * channels + c];
        }
    }
}

pub fn softmax_inplace(logits: &mut [f32]) {
    if logits.is_empty() {
        return;
    }
    #[cfg(target_os = "macos")]
    {
        #[link(name = "Accelerate", kind = "framework")]
        extern "C" {
            fn vDSP_maxv(a: *const f32, ia: i32, c: *mut f32, n: u32);
            fn vDSP_vsadd(
                a: *const f32,
                ia: i32,
                b: *const f32,
                c: *mut f32,
                ic: i32,
                n: u32,
            );
            fn vvexpf(y: *mut f32, x: *const f32, n: *const i32);
            fn vDSP_sve(a: *const f32, ia: i32, c: *mut f32, n: u32);
            fn vDSP_vsmul(a: *const f32, ia: i32, b: *const f32, c: *mut f32, ic: i32, n: u32);
        }
        let n = logits.len();
        let n_u = n as u32;
        let mut max = 0f32;
        unsafe {
            vDSP_maxv(logits.as_ptr(), 1, &mut max, n_u);
            let neg_max = -max;
            vDSP_vsadd(logits.as_ptr(), 1, &neg_max, logits.as_mut_ptr(), 1, n_u);
            let ni = n as i32;
            vvexpf(logits.as_mut_ptr(), logits.as_ptr(), &ni);
            let mut sum = 0f32;
            vDSP_sve(logits.as_ptr(), 1, &mut sum, n_u);
            let inv = sum.recip();
            vDSP_vsmul(logits.as_ptr(), 1, &inv, logits.as_mut_ptr(), 1, n_u);
        }
    }
    #[cfg(not(target_os = "macos"))]
    {
        let mut max = f32::NEG_INFINITY;
        for &v in logits.iter() {
            max = max.max(v);
        }
        let mut sum = 0f32;
        for v in logits.iter_mut() {
            *v = (*v - max).exp();
            sum += *v;
        }
        let inv = sum.recip();
        for v in logits.iter_mut() {
            *v *= inv;
        }
    }
}

/// Attention-residual mixer over stacked states `[n_states, H]` for one token.
pub fn attention_residual_token(
    states: &[f32], // [n, H] row-major
    n_states: usize,
    query: &[f32], // [H]
    normalize_values: bool,
    out: &mut [f32],
) {
    attention_residual_token_ext(states, None, n_states, query, normalize_values, out);
}

/// Like [`attention_residual_token`] but optionally appends `extra` (one HIDDEN
/// vector) without cloning `states` into a temporary.
pub fn attention_residual_token_ext(
    states: &[f32],
    extra: Option<&[f32]>,
    n_states: usize,
    query: &[f32],
    normalize_values: bool,
    out: &mut [f32],
) {
    match extra {
        None => attention_residual_token_extras(
            states, &[], n_states, query, normalize_values, out,
        ),
        Some(e) => attention_residual_token_extras(
            states, &[e], n_states, query, normalize_values, out,
        ),
    }
}

/// Like [`attention_residual_token_ext`] with zero or more trailing HIDDEN
/// vectors (block-boundary mlp mixes pass `x_in` + residual without a concat).
pub fn attention_residual_token_extras(
    states: &[f32],
    extras: &[&[f32]],
    n_states: usize,
    query: &[f32],
    normalize_values: bool,
    out: &mut [f32],
) {
    let n_extra = extras.len();
    let n = n_states + n_extra;
    assert_eq!(states.len(), n_states * HIDDEN);
    assert_eq!(query.len(), HIDDEN);
    for e in extras {
        assert_eq!(e.len(), HIDDEN);
    }
    if n == 0 {
        out.fill(0.0);
        return;
    }
    // Decode block stacks stay small; keep keys/scores off the heap.
    const MAX_STACK: usize = 16;
    let scale = (HIDDEN as f32).sqrt().recip();
    if n <= MAX_STACK {
        let mut keys = [0f32; MAX_STACK * HIDDEN];
        let mut scores = [0f32; MAX_STACK];
        for i in 0..n_states {
            let s = &states[i * HIDDEN..(i + 1) * HIDDEN];
            rms_norm(s, None, &mut keys[i * HIDDEN..(i + 1) * HIDDEN]);
        }
        for (j, e) in extras.iter().enumerate() {
            let base = (n_states + j) * HIDDEN;
            rms_norm(e, None, &mut keys[base..base + HIDDEN]);
        }
        for i in 0..n {
            let k = &keys[i * HIDDEN..(i + 1) * HIDDEN];
            scores[i] = dot(k, query) * scale;
        }
        softmax_inplace(&mut scores[..n]);
        // First source: copy+scale; remaining axpy (skip zero-fill).
        {
            let src0 = residual_mix_source(
                0, n_states, normalize_values, states, extras, &keys,
            );
            out.copy_from_slice(src0);
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
                        out.as_ptr(),
                        1,
                        &scores[0],
                        out.as_mut_ptr(),
                        1,
                        HIDDEN as u32,
                    );
                }
            }
            #[cfg(not(target_os = "macos"))]
            {
                let w0 = scores[0];
                for v in out.iter_mut() {
                    *v *= w0;
                }
            }
        }
        for i in 1..n {
            let src = residual_mix_source(
                i, n_states, normalize_values, states, extras, &keys,
            );
            axpy(scores[i], src, out);
        }
        return;
    }
    out.fill(0.0);
    let mut keys = vec![0f32; n * HIDDEN];
    let mut scores = vec![0f32; n];
    for i in 0..n_states {
        let s = &states[i * HIDDEN..(i + 1) * HIDDEN];
        rms_norm(s, None, &mut keys[i * HIDDEN..(i + 1) * HIDDEN]);
    }
    for (j, e) in extras.iter().enumerate() {
        let base = (n_states + j) * HIDDEN;
        rms_norm(e, None, &mut keys[base..base + HIDDEN]);
    }
    for i in 0..n {
        let k = &keys[i * HIDDEN..(i + 1) * HIDDEN];
        scores[i] = dot(k, query) * scale;
    }
    softmax_inplace(&mut scores);
    for i in 0..n {
        let src = residual_mix_source(
            i, n_states, normalize_values, states, extras, &keys,
        );
        axpy(scores[i], src, out);
    }
}

#[inline]
fn residual_mix_source<'a>(
    i: usize,
    n_states: usize,
    normalize_values: bool,
    states: &'a [f32],
    extras: &'a [&'a [f32]],
    keys: &'a [f32],
) -> &'a [f32] {
    if normalize_values {
        &keys[i * HIDDEN..(i + 1) * HIDDEN]
    } else if i < n_states {
        &states[i * HIDDEN..(i + 1) * HIDDEN]
    } else {
        extras[i - n_states]
    }
}

pub fn l2_clip(x: &mut [f32], threshold: f32, scale: Option<&[f32]>) {
    let n = x.len();
    if n == 0 {
        return;
    }
    #[cfg(target_os = "macos")]
    {
        extern "C" {
            fn vDSP_vmul(
                a: *const f32,
                ia: i32,
                b: *const f32,
                ib: i32,
                c: *mut f32,
                ic: i32,
                n: u32,
            );
            fn vDSP_svesq(a: *const f32, ia: i32, c: *mut f32, n: u32);
            fn vDSP_vsmul(a: *const f32, ia: i32, b: *const f32, c: *mut f32, ic: i32, n: u32);
        }
        if let Some(s) = scale {
            assert_eq!(s.len(), n);
            unsafe {
                vDSP_vmul(x.as_ptr(), 1, s.as_ptr(), 1, x.as_mut_ptr(), 1, n as u32);
            }
        }
        let mut norm_sq = 0f32;
        unsafe {
            vDSP_svesq(x.as_ptr(), 1, &mut norm_sq, n as u32);
        }
        let factor = (threshold / (norm_sq.sqrt() + 1e-6)).min(1.0);
        if factor < 1.0 {
            unsafe {
                vDSP_vsmul(x.as_ptr(), 1, &factor, x.as_mut_ptr(), 1, n as u32);
            }
        }
    }
    #[cfg(not(target_os = "macos"))]
    {
        if let Some(s) = scale {
            for i in 0..n {
                x[i] *= s[i];
            }
        }
        let mut norm_sq = 0f32;
        for &v in x.iter() {
            norm_sq += v * v;
        }
        let factor = (threshold / (norm_sq.sqrt() + 1e-6)).min(1.0);
        for v in x.iter_mut() {
            *v *= factor;
        }
    }
}

/// Dynamic short-conv (full-attn K/V): static taps + per-token weight_proj.
/// `x`: `[T*C]`, `dyn_in`: `[T*H]`, `static_w`: `[K*C]` row-major `(k, c)`,
/// `weight_proj`: `[n_heads*K, H]`, scale `/sqrt(2K+1)`.
/// Lag 0 multiplies the current token (matches MLX `DynamicShortConvolution`).
pub fn dynamic_short_conv(
    x: &[f32],
    dyn_in: &[f32],
    static_w: &[f32],
    weight_proj: &[f32],
    tokens: usize,
    channels: usize,
    hidden: usize,
    n_heads: usize,
    k: usize,
    out: &mut [f32],
) {
    assert_eq!(x.len(), tokens * channels);
    assert_eq!(dyn_in.len(), tokens * hidden);
    assert_eq!(static_w.len(), k * channels);
    assert_eq!(weight_proj.len(), n_heads * k * hidden);
    assert_eq!(out.len(), tokens * channels);
    assert_eq!(channels % n_heads, 0);
    let head_ch = channels / n_heads;
    let scale = (2.0 * k as f32 + 1.0).sqrt().recip();
    let n_dyn = n_heads * k;
    let mut dyn_w = vec![0f32; tokens * n_dyn];
    for t in 0..tokens {
        let din = &dyn_in[t * hidden..(t + 1) * hidden];
        gemv(
            weight_proj,
            n_dyn,
            din,
            &mut dyn_w[t * n_dyn..(t + 1) * n_dyn],
        );
    }
    let k_i = k as isize;
    for t in 0..tokens {
        for h in 0..n_heads {
            for c in 0..head_ch {
                let mut acc = 0f32;
                let ch = h * head_ch + c;
                for lag in 0..k {
                    // ctx = [zeros(K-1); x]; values[t] = ctx[(K-1-lag)+t]
                    let j = (k_i - 1 - lag as isize) + t as isize;
                    let vv = if j < (k_i - 1) {
                        0f32
                    } else {
                        x[(j - (k_i - 1)) as usize * channels + ch]
                    };
                    let static_tap = static_w[lag * channels + ch];
                    let dyn_tap = dyn_w[t * n_heads * k + h * k + lag];
                    acc += vv * (static_tap + dyn_tap);
                }
                let xv = x[t * channels + ch];
                out[t * channels + ch] = (acc + xv) * scale;
            }
        }
    }
}

/// Final short-conv rolling state after a `state_in=None` prefill of `x`.
///
/// Layout matches [`short_conv_with_state`]: per-channel `[oldest .. newest]`
/// of length `CONV_K - 1` taken from the end of `x` (zeros if `tokens < hist`).
pub fn short_conv_tail_state(x: &[f32], tokens: usize, channels: usize, state_out: &mut [f32]) {
    let hist = CONV_K - 1;
    assert_eq!(x.len(), tokens * channels);
    assert_eq!(state_out.len(), channels * hist);
    // Warm path: tokens ≥ 3, hist = 3: copy the last three rows without branches.
    if hist == 3 && tokens >= 3 {
        let t0 = (tokens - 3) * channels;
        let t1 = (tokens - 2) * channels;
        let t2 = (tokens - 1) * channels;
        for c in 0..channels {
            let s = c * 3;
            state_out[s] = x[t0 + c];
            state_out[s + 1] = x[t1 + c];
            state_out[s + 2] = x[t2 + c];
        }
        return;
    }
    for c in 0..channels {
        for age in 0..hist {
            let src = tokens as isize - hist as isize + age as isize;
            state_out[c * hist + age] = if src < 0 {
                0.0
            } else {
                x[(src as usize) * channels + c]
            };
        }
    }
}

/// Single-token dynamic short-conv with rolling `state` `[channels * (k-1)]`.
/// T=1 dyn short-conv that rotates `state` in place (no hist temp).
pub fn dynamic_short_conv_decode_inplace(
    x: &[f32],
    dyn_in: &[f32],
    static_w: &[f32],
    weight_proj: &[f32],
    channels: usize,
    hidden: usize,
    n_heads: usize,
    state: &mut [f32],
    out: &mut [f32],
) {
    debug_assert_eq!(CONV_K, 4);
    let k = CONV_K;
    let hist = k - 1;
    assert_eq!(x.len(), channels);
    assert_eq!(dyn_in.len(), hidden);
    assert_eq!(static_w.len(), k * channels);
    assert_eq!(weight_proj.len(), n_heads * k * hidden);
    assert_eq!(out.len(), channels);
    assert_eq!(state.len(), channels * hist);
    assert_eq!(channels % n_heads, 0);
    let head_ch = channels / n_heads;
    let scale = (2.0 * k as f32 + 1.0).sqrt().recip();
    const DYN_STACK: usize = 64;
    let n_dyn = n_heads * k;
    let mut dyn_stack = [0f32; DYN_STACK];
    debug_assert!(n_dyn <= DYN_STACK);
    let dyn_w = &mut dyn_stack[..n_dyn];
    gemv(weight_proj, n_dyn, dyn_in, dyn_w);
    for h in 0..n_heads {
        let d0 = dyn_w[h * 4];
        let d1 = dyn_w[h * 4 + 1];
        let d2 = dyn_w[h * 4 + 2];
        let d3 = dyn_w[h * 4 + 3];
        for c in 0..head_ch {
            let ch = h * head_ch + c;
            let base = ch * 3;
            let v0 = state[base];
            let v1 = state[base + 1];
            let v2 = state[base + 2];
            let v3 = x[ch];
            let acc = v3 * (static_w[ch] + d0)
                + v2 * (static_w[channels + ch] + d1)
                + v1 * (static_w[2 * channels + ch] + d2)
                + v0 * (static_w[3 * channels + ch] + d3);
            out[ch] = (acc + v3) * scale;
            state[base] = v1;
            state[base + 1] = v2;
            state[base + 2] = v3;
        }
    }
}

pub fn dynamic_short_conv_step(
    x: &[f32],
    dyn_in: &[f32],
    static_w: &[f32],
    weight_proj: &[f32],
    channels: usize,
    hidden: usize,
    n_heads: usize,
    k: usize,
    state_in: &[f32],
    out: &mut [f32],
    state_out: &mut [f32],
) {
    assert_eq!(x.len(), channels);
    assert_eq!(dyn_in.len(), hidden);
    assert_eq!(static_w.len(), k * channels);
    assert_eq!(weight_proj.len(), n_heads * k * hidden);
    assert_eq!(out.len(), channels);
    let hist = k - 1;
    assert_eq!(state_in.len(), channels * hist);
    assert_eq!(state_out.len(), channels * hist);
    assert_eq!(channels % n_heads, 0);
    let head_ch = channels / n_heads;
    let scale = (2.0 * k as f32 + 1.0).sqrt().recip();
    // K/V dyn sconv uses N_KV(=8)×CONV_K(=4)=32 rows; keep off-heap.
    const DYN_STACK: usize = 64;
    let n_dyn = n_heads * k;
    let mut dyn_stack = [0f32; DYN_STACK];
    let mut dyn_heap = if n_dyn > DYN_STACK {
        vec![0f32; n_dyn]
    } else {
        Vec::new()
    };
    let dyn_w: &mut [f32] = if n_dyn <= DYN_STACK {
        &mut dyn_stack[..n_dyn]
    } else {
        dyn_heap.as_mut_slice()
    };
    gemv(weight_proj, n_dyn, dyn_in, dyn_w);
    // Decode path always uses CONV_K=4; unroll the lag loop and state shift.
    if k == 4 && hist == 3 {
        for h in 0..n_heads {
            let d0 = dyn_w[h * 4];
            let d1 = dyn_w[h * 4 + 1];
            let d2 = dyn_w[h * 4 + 2];
            let d3 = dyn_w[h * 4 + 3];
            for c in 0..head_ch {
                let ch = h * head_ch + c;
                let base = ch * 3;
                let v0 = state_in[base];
                let v1 = state_in[base + 1];
                let v2 = state_in[base + 2];
                let v3 = x[ch];
                // ctx_i for lag 0..3 → 3,2,1,0 → x, state[2], state[1], state[0]
                let acc = v3 * (static_w[ch] + d0)
                    + v2 * (static_w[channels + ch] + d1)
                    + v1 * (static_w[2 * channels + ch] + d2)
                    + v0 * (static_w[3 * channels + ch] + d3);
                out[ch] = (acc + v3) * scale;
                state_out[base] = v1;
                state_out[base + 1] = v2;
                state_out[base + 2] = v3;
            }
        }
        return;
    }
    for h in 0..n_heads {
        for c in 0..head_ch {
            let mut acc = 0f32;
            let ch = h * head_ch + c;
            for lag in 0..k {
                // ctx = state (hist) || x(1); values[t] = ctx[(K-1-lag)+t] with t=0 → ctx[K-1-lag]
                let ctx_i = k - 1 - lag;
                let vv = if ctx_i < hist {
                    state_in[ch * hist + ctx_i]
                } else {
                    x[ch]
                };
                let static_tap = static_w[lag * channels + ch];
                let dyn_tap = dyn_w[h * k + lag];
                acc += vv * (static_tap + dyn_tap);
            }
            out[ch] = (acc + x[ch]) * scale;
        }
    }
    for ch in 0..channels {
        for age in 0..hist {
            let src = 1isize - hist as isize + age as isize;
            state_out[ch * hist + age] = if src < 0 {
                state_in[ch * hist + (src + hist as isize) as usize]
            } else {
                x[ch]
            };
        }
    }
}

/// Multi-query attention residual. `queries` is `[Q*H]`; `out` is `[Q*H]`.
pub fn attention_residual_multi(
    states: &[f32],
    n_states: usize,
    queries: &[f32],
    n_queries: usize,
    normalize_values: bool,
    out: &mut [f32],
) {
    attention_residual_multi_ext(
        states,
        None,
        n_states,
        queries,
        n_queries,
        normalize_values,
        out,
    );
}

/// Like [`attention_residual_multi`] but optionally appends `extra` (one HIDDEN
/// vector) without cloning `states` into a temporary.
pub fn attention_residual_multi_ext(
    states: &[f32],
    extra: Option<&[f32]>,
    n_states: usize,
    queries: &[f32],
    n_queries: usize,
    normalize_values: bool,
    out: &mut [f32],
) {
    let n_extra = usize::from(extra.is_some());
    let n = n_states + n_extra;
    assert_eq!(states.len(), n_states * HIDDEN);
    assert_eq!(queries.len(), n_queries * HIDDEN);
    assert_eq!(out.len(), n_queries * HIDDEN);
    if let Some(e) = extra {
        assert_eq!(e.len(), HIDDEN);
    }
    // Decode / small block stacks stay off-heap; warm can grow past this.
    const MAX_STACK: usize = 16;
    const MAX_Q: usize = 4;
    let scale = (HIDDEN as f32).sqrt().recip();
    if n <= MAX_STACK && n_queries <= MAX_Q {
        let mut keys = [0f32; MAX_STACK * HIDDEN];
        let mut scores = [0f32; MAX_STACK * MAX_Q];
        let mut col = [0f32; MAX_STACK];
        for i in 0..n_states {
            rms_norm(
                &states[i * HIDDEN..(i + 1) * HIDDEN],
                None,
                &mut keys[i * HIDDEN..(i + 1) * HIDDEN],
            );
        }
        if let Some(e) = extra {
            rms_norm(
                e,
                None,
                &mut keys[n_states * HIDDEN..(n_states + 1) * HIDDEN],
            );
        }
        for i in 0..n {
            let k = &keys[i * HIDDEN..(i + 1) * HIDDEN];
            for q in 0..n_queries {
                let query = &queries[q * HIDDEN..(q + 1) * HIDDEN];
                scores[i * n_queries + q] = dot(k, query) * scale;
            }
        }
        for q in 0..n_queries {
            for i in 0..n {
                col[i] = scores[i * n_queries + q];
            }
            softmax_inplace(&mut col[..n]);
            let dst = &mut out[q * HIDDEN..(q + 1) * HIDDEN];
            dst.fill(0.0);
            for i in 0..n {
                let src = if normalize_values {
                    &keys[i * HIDDEN..(i + 1) * HIDDEN]
                } else if i < n_states {
                    &states[i * HIDDEN..(i + 1) * HIDDEN]
                } else {
                    extra.unwrap()
                };
                axpy(col[i], src, dst);
            }
        }
        return;
    }
    let mut keys = vec![0f32; n * HIDDEN];
    for i in 0..n_states {
        rms_norm(
            &states[i * HIDDEN..(i + 1) * HIDDEN],
            None,
            &mut keys[i * HIDDEN..(i + 1) * HIDDEN],
        );
    }
    if let Some(e) = extra {
        rms_norm(
            e,
            None,
            &mut keys[n_states * HIDDEN..(n_states + 1) * HIDDEN],
        );
    }
    let mut scores = vec![0f32; n * n_queries];
    for i in 0..n {
        let k = &keys[i * HIDDEN..(i + 1) * HIDDEN];
        for q in 0..n_queries {
            let query = &queries[q * HIDDEN..(q + 1) * HIDDEN];
            scores[i * n_queries + q] = dot(k, query) * scale;
        }
    }
    let mut col = vec![0f32; n];
    for q in 0..n_queries {
        for i in 0..n {
            col[i] = scores[i * n_queries + q];
        }
        softmax_inplace(&mut col);
        let dst = &mut out[q * HIDDEN..(q + 1) * HIDDEN];
        dst.fill(0.0);
        for i in 0..n {
            let src = if normalize_values {
                &keys[i * HIDDEN..(i + 1) * HIDDEN]
            } else if i < n_states {
                &states[i * HIDDEN..(i + 1) * HIDDEN]
            } else {
                extra.unwrap()
            };
            axpy(col[i], src, dst);
        }
    }
}

/// In-place `x[i] *= sigmoid(gate[i])` with Accelerate `vvexpf` on macOS.
pub fn sigmoid_gate_mul(gate: &[f32], x: &mut [f32]) {
    assert_eq!(gate.len(), x.len());
    let n = gate.len();
    #[cfg(target_os = "macos")]
    {
        #[link(name = "Accelerate", kind = "framework")]
        extern "C" {
            fn vvexpf(y: *mut f32, x: *const f32, n: *const i32);
            fn vDSP_vneg(a: *const f32, ia: i32, c: *mut f32, ic: i32, n: u32);
            fn vDSP_vsadd(
                a: *const f32,
                ia: i32,
                b: *const f32,
                c: *mut f32,
                ic: i32,
                n: u32,
            );
            fn vDSP_vdiv(
                a: *const f32,
                ia: i32,
                b: *const f32,
                ib: i32,
                c: *mut f32,
                ic: i32,
                n: u32,
            );
        }
        // Attn gate is N_HEADS*DH (=3072); keep that path stack-only.
        const STACK: usize = 3072;
        let n_i = n as i32;
        let n_u = n as u32;
        let one = 1.0f32;
        if n <= STACK {
            let mut neg = [0f32; STACK];
            let mut den = [0f32; STACK];
            unsafe {
                vDSP_vneg(gate.as_ptr(), 1, neg.as_mut_ptr(), 1, n_u);
                vvexpf(neg.as_mut_ptr(), neg.as_ptr(), &n_i);
                vDSP_vsadd(neg.as_ptr(), 1, &one, den.as_mut_ptr(), 1, n_u);
                // x *= 1/den  (vDSP_vdiv: c = b/a with a=den, b=x)
                vDSP_vdiv(den.as_ptr(), 1, x.as_ptr(), 1, x.as_mut_ptr(), 1, n_u);
            }
        } else {
            let mut neg = vec![0f32; n];
            let mut den = vec![0f32; n];
            unsafe {
                vDSP_vneg(gate.as_ptr(), 1, neg.as_mut_ptr(), 1, n_u);
                vvexpf(neg.as_mut_ptr(), neg.as_ptr(), &n_i);
                vDSP_vsadd(neg.as_ptr(), 1, &one, den.as_mut_ptr(), 1, n_u);
                vDSP_vdiv(den.as_ptr(), 1, x.as_ptr(), 1, x.as_mut_ptr(), 1, n_u);
            }
        }
    }
    #[cfg(not(target_os = "macos"))]
    {
        for i in 0..n {
            x[i] *= 1.0 / (1.0 + (-gate[i]).exp());
        }
    }
}

/// Dot product; uses Accelerate on macOS.
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    dot_n_strided(a.len(), a, 1, b, 1)
}

/// `sum_{i=0}^{n-1} a[i*inca] * b[i*incb]` (Accelerate `cblas_sdot` on macOS).
#[inline]
pub fn dot_n_strided(n: usize, a: &[f32], inca: usize, b: &[f32], incb: usize) -> f32 {
    debug_assert!(inca >= 1 && incb >= 1);
    debug_assert!(n == 0 || a.len() > (n - 1) * inca);
    debug_assert!(n == 0 || b.len() > (n - 1) * incb);
    #[cfg(target_os = "macos")]
    {
        #[link(name = "Accelerate", kind = "framework")]
        extern "C" {
            fn cblas_sdot(n: i32, x: *const f32, incx: i32, y: *const f32, incy: i32) -> f32;
        }
        unsafe {
            cblas_sdot(n as i32, a.as_ptr(), inca as i32, b.as_ptr(), incb as i32)
        }
    }
    #[cfg(not(target_os = "macos"))]
    {
        let mut acc = 0f32;
        for i in 0..n {
            acc += a[i * inca] * b[i * incb];
        }
        acc
    }
}

/// `y += alpha * x` (Accelerate saxpy on macOS).
#[inline]
pub fn axpy(alpha: f32, x: &[f32], y: &mut [f32]) {
    assert_eq!(x.len(), y.len());
    #[cfg(target_os = "macos")]
    {
        #[link(name = "Accelerate", kind = "framework")]
        extern "C" {
            fn cblas_saxpy(
                n: i32,
                alpha: f32,
                x: *const f32,
                incx: i32,
                y: *mut f32,
                incy: i32,
            );
        }
        unsafe {
            cblas_saxpy(x.len() as i32, alpha, x.as_ptr(), 1, y.as_mut_ptr(), 1);
        }
    }
    #[cfg(not(target_os = "macos"))]
    {
        for i in 0..x.len() {
            y[i] += alpha * x[i];
        }
    }
}

pub fn gemv(weight: &[f32], out_dim: usize, x: &[f32], out: &mut [f32]) {
    let in_dim = x.len();
    assert_eq!(weight.len(), out_dim * in_dim);
    assert_eq!(out.len(), out_dim);
    gemv_lda(weight, out_dim, in_dim, in_dim, 0, x, 0.0, out);
}

/// Row-major GEMV over a column window: `out = beta * out + A[:, col0..col0+in_dim] @ x`.
///
/// `weight` is `[out_dim, lda]` packed row-major; `lda >= col0 + in_dim`.
pub fn gemv_lda(
    weight: &[f32],
    out_dim: usize,
    in_dim: usize,
    lda: usize,
    col0: usize,
    x: &[f32],
    beta: f32,
    out: &mut [f32],
) {
    assert_eq!(x.len(), in_dim);
    assert_eq!(out.len(), out_dim);
    assert!(lda >= col0 + in_dim);
    assert_eq!(weight.len(), out_dim * lda);
    #[cfg(target_os = "macos")]
    {
        #[link(name = "Accelerate", kind = "framework")]
        extern "C" {
            fn cblas_sgemv(
                order: i32,
                trans: i32,
                m: i32,
                n: i32,
                alpha: f32,
                a: *const f32,
                lda: i32,
                x: *const f32,
                incx: i32,
                beta: f32,
                y: *mut f32,
                incy: i32,
            );
        }
        const ROW_MAJOR: i32 = 101;
        const NO_TRANS: i32 = 111;
        unsafe {
            cblas_sgemv(
                ROW_MAJOR,
                NO_TRANS,
                out_dim as i32,
                in_dim as i32,
                1.0,
                weight.as_ptr().add(col0),
                lda as i32,
                x.as_ptr(),
                1,
                beta,
                out.as_mut_ptr(),
                1,
            );
        }
    }
    #[cfg(not(target_os = "macos"))]
    {
        for o in 0..out_dim {
            let row = &weight[o * lda + col0..o * lda + col0 + in_dim];
            let mut acc = 0f32;
            for i in 0..in_dim {
                acc += row[i] * x[i];
            }
            out[o] = beta * out[o] + acc;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rms_norm_matches_scalar() {
        let n = HIDDEN;
        let mut x = vec![0f32; n];
        let mut w = vec![0f32; n];
        for i in 0..n {
            x[i] = ((i % 23) as f32) * 0.01 - 0.1;
            w[i] = 0.5 + ((i % 7) as f32) * 0.1;
        }
        let mut out = vec![0f32; n];
        rms_norm(&x, Some(&w), &mut out);
        let mut var = 0f32;
        for &v in &x {
            var += v * v;
        }
        let inv = (var / n as f32 + EPS).sqrt().recip();
        for i in 0..n {
            let expect = x[i] * inv * w[i];
            assert!((out[i] - expect).abs() < 1e-5, "i={i} {} vs {expect}", out[i]);
        }
    }

    #[test]
    fn sigmoid_from_neg_logits_matches_scalar() {
        let mut x = [0.0f32, 1.0, -0.5, 2.5, -3.0];
        let expect: Vec<f32> = x.iter().map(|&v| 1.0 / (1.0 + v.exp())).collect();
        sigmoid_from_neg_logits_inplace(&mut x);
        for (i, (&got, &exp)) in x.iter().zip(expect.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-5,
                "i={i} got={got} expect={exp}"
            );
        }
    }

    #[test]
    fn softmax_sums_to_one() {
        let mut x = [1.0f32, 2.0, 3.0, 0.5];
        softmax_inplace(&mut x);
        let s: f32 = x.iter().sum();
        assert!((s - 1.0).abs() < 1e-5, "sum={s} x={x:?}");
        assert!(x[2] > x[1] && x[1] > x[0]);
    }

    #[test]
    fn dot_and_axpy_match_naive() {
        let a = [1.0f32, 2.0, 3.0, 4.0];
        let b = [0.5f32, -1.0, 2.0, 0.25];
        let d = dot(&a, &b);
        assert!((d - (0.5 - 2.0 + 6.0 + 1.0)).abs() < 1e-5);
        let mut y = [10.0f32, 20.0, 30.0, 40.0];
        axpy(2.0, &a, &mut y);
        assert_eq!(y, [12.0, 24.0, 36.0, 48.0]);
    }

    #[test]
    fn attention_residual_multi_ext_matches_clone() {
        let n_states = 3usize;
        let n_queries = 3usize;
        let mut states = vec![0f32; n_states * HIDDEN];
        let mut extra = vec![0f32; HIDDEN];
        let mut queries = vec![0f32; n_queries * HIDDEN];
        for i in 0..states.len() {
            states[i] = ((i % 17) as f32) * 0.01 - 0.05;
        }
        for i in 0..HIDDEN {
            extra[i] = ((i % 13) as f32) * 0.02 - 0.1;
        }
        for i in 0..queries.len() {
            queries[i] = ((i % 11) as f32) * 0.015;
        }
        let mut joined = states.clone();
        joined.extend_from_slice(&extra);
        let mut a = vec![0f32; n_queries * HIDDEN];
        let mut b = vec![0f32; n_queries * HIDDEN];
        attention_residual_multi(&joined, n_states + 1, &queries, n_queries, true, &mut a);
        attention_residual_multi_ext(
            &states,
            Some(extra.as_slice()),
            n_states,
            &queries,
            n_queries,
            true,
            &mut b,
        );
        for i in 0..a.len() {
            assert!(
                (a[i] - b[i]).abs() < 1e-5,
                "i={i} {} vs {}",
                a[i],
                b[i]
            );
        }
    }

    #[test]
    fn attention_residual_token_extras_matches_concat() {
        let n_states = 4usize;
        let mut states = vec![0f32; n_states * HIDDEN];
        let mut e0 = vec![0f32; HIDDEN];
        let mut e1 = vec![0f32; HIDDEN];
        let mut query = vec![0f32; HIDDEN];
        for i in 0..states.len() {
            states[i] = ((i % 19) as f32) * 0.01 - 0.04;
        }
        for i in 0..HIDDEN {
            e0[i] = ((i % 13) as f32) * 0.02 - 0.1;
            e1[i] = ((i % 11) as f32) * 0.015;
            query[i] = ((i % 7) as f32) * 0.03 - 0.05;
        }
        let mut joined = states.clone();
        joined.extend_from_slice(&e0);
        joined.extend_from_slice(&e1);
        let mut a = [0f32; HIDDEN];
        let mut b = [0f32; HIDDEN];
        attention_residual_token(&joined, n_states + 2, &query, true, &mut a);
        attention_residual_token_extras(
            &states,
            &[e0.as_slice(), e1.as_slice()],
            n_states,
            &query,
            true,
            &mut b,
        );
        for i in 0..HIDDEN {
            assert!((a[i] - b[i]).abs() < 1e-5, "i={i} {} vs {}", a[i], b[i]);
        }
    }

    #[test]
    fn gemv_matches_naive() {
        let out_dim = 5usize;
        let in_dim = 7usize;
        let mut w = vec![0f32; out_dim * in_dim];
        let mut x = vec![0f32; in_dim];
        for (i, v) in w.iter_mut().enumerate() {
            *v = (i as f32) * 0.01 - 0.1;
        }
        for (i, v) in x.iter_mut().enumerate() {
            *v = (i as f32) * 0.1;
        }
        let mut out = vec![0f32; out_dim];
        gemv(&w, out_dim, &x, &mut out);
        for o in 0..out_dim {
            let mut acc = 0f32;
            for i in 0..in_dim {
                acc += w[o * in_dim + i] * x[i];
            }
            assert!(
                (out[o] - acc).abs() < 1e-5,
                "o={o} out={} acc={acc}",
                out[o]
            );
        }
    }

    #[test]
    fn gemv_lda_split_matches_concat() {
        let out_dim = 4usize;
        let half = 6usize;
        let lda = 2 * half;
        let mut w = vec![0f32; out_dim * lda];
        let mut x0 = vec![0f32; half];
        let mut x1 = vec![0f32; half];
        for (i, v) in w.iter_mut().enumerate() {
            *v = (i as f32) * 0.02 - 0.3;
        }
        for i in 0..half {
            x0[i] = (i as f32) * 0.1;
            x1[i] = (i as f32) * -0.05 + 0.2;
        }
        let mut cat = vec![0f32; lda];
        cat[..half].copy_from_slice(&x0);
        cat[half..].copy_from_slice(&x1);
        let mut expect = vec![0f32; out_dim];
        gemv(&w, out_dim, &cat, &mut expect);
        let mut got = vec![0f32; out_dim];
        gemv_lda(&w, out_dim, half, lda, 0, &x0, 0.0, &mut got);
        gemv_lda(&w, out_dim, half, lda, half, &x1, 1.0, &mut got);
        for o in 0..out_dim {
            assert!(
                (got[o] - expect[o]).abs() < 1e-5,
                "o={o} got={} expect={}",
                got[o],
                expect[o]
            );
        }
    }

    #[test]
    fn short_conv_impulse() {
        let channels = 2;
        let tokens = 3;
        // taps all 1 on last tap only
        let mut taps = vec![0f32; channels * CONV_K];
        for c in 0..channels {
            taps[c * CONV_K + (CONV_K - 1)] = 1.0;
        }
        let mut x = vec![0f32; tokens * channels];
        x[0] = 1.0;
        x[1] = 2.0;
        let mut out = vec![0f32; tokens * channels];
        short_conv(&x, &taps, tokens, channels, &mut out);
        let s = (5f32).sqrt().recip();
        // y = (x_t + x_t)/sqrt5 when only last tap is 1 and no history
        assert!((out[0] - (1.0 + 1.0) * s).abs() < 1e-5);
    }

    #[test]
    fn short_conv_tail_state_matches_with_state() {
        let tokens = 5usize;
        let channels = 8usize;
        let mut x = vec![0f32; tokens * channels];
        let mut taps = vec![0f32; channels * CONV_K];
        for i in 0..x.len() {
            x[i] = ((i % 11) as f32) * 0.1;
        }
        for i in 0..taps.len() {
            taps[i] = ((i % 5) as f32) * 0.05;
        }
        let mut out = vec![0f32; tokens * channels];
        let mut hist = vec![0f32; channels * (CONV_K - 1)];
        short_conv_with_state(&x, &taps, tokens, channels, None, &mut out, Some(&mut hist));
        let mut hist2 = vec![0f32; channels * (CONV_K - 1)];
        short_conv_tail_state(&x, tokens, channels, &mut hist2);
        for i in 0..hist.len() {
            assert!(
                (hist[i] - hist2[i]).abs() < 1e-6,
                "i={i} {} vs {}",
                hist[i],
                hist2[i]
            );
        }
    }

    #[test]
    fn short_conv_stepwise_matches_prefill() {
        let channels = 4;
        let tokens = 8;
        let hist = CONV_K - 1;
        let mut taps = vec![0f32; channels * CONV_K];
        for (i, v) in taps.iter_mut().enumerate() {
            *v = ((i % 7) as f32) * 0.1 - 0.3;
        }
        let mut x = vec![0f32; tokens * channels];
        for (i, v) in x.iter_mut().enumerate() {
            *v = ((i % 11) as f32) * 0.05 - 0.2;
        }
        let mut pref = vec![0f32; tokens * channels];
        short_conv(&x, &taps, tokens, channels, &mut pref);

        let mut state = vec![0f32; channels * hist];
        let mut step = vec![0f32; tokens * channels];
        let mut y = vec![0f32; channels];
        for t in 0..tokens {
            let xt = &x[t * channels..(t + 1) * channels];
            let mut next = vec![0f32; channels * hist];
            short_conv_with_state(xt, &taps, 1, channels, Some(&state), &mut y, Some(&mut next));
            step[t * channels..(t + 1) * channels].copy_from_slice(&y);
            state = next;
        }
        let mut maxabs = 0f32;
        for (a, b) in pref.iter().zip(step.iter()) {
            maxabs = maxabs.max((a - b).abs());
        }
        assert!(maxabs < 1e-6, "stepwise vs prefill maxabs={maxabs}");
    }

    #[test]
    fn short_conv_decode_inplace_matches_with_state() {
        let channels = 8;
        let hist = CONV_K - 1;
        let mut taps = vec![0f32; channels * CONV_K];
        for (i, v) in taps.iter_mut().enumerate() {
            *v = ((i % 5) as f32) * 0.07 - 0.1;
        }
        let mut x = vec![0f32; channels];
        for (i, v) in x.iter_mut().enumerate() {
            *v = ((i % 3) as f32) * 0.2;
        }
        let state_a = vec![0.1f32; channels * hist];
        let mut state_b = state_a.clone();
        let mut ya = vec![0f32; channels];
        let mut yb = vec![0f32; channels];
        let mut next = vec![0f32; channels * hist];
        short_conv_with_state(&x, &taps, 1, channels, Some(&state_a), &mut ya, Some(&mut next));
        short_conv_decode_inplace(&x, &taps, channels, &mut state_b, &mut yb);
        for i in 0..channels {
            assert!((ya[i] - yb[i]).abs() < 1e-6, "y i={i}");
        }
        for i in 0..state_b.len() {
            assert!((next[i] - state_b[i]).abs() < 1e-6, "state i={i}");
        }
    }

    #[test]
    fn short_conv_stepwise_hidden() {
        let channels = HIDDEN;
        let tokens = 4;
        let hist = CONV_K - 1;
        let mut taps = vec![0f32; channels * CONV_K];
        for (i, v) in taps.iter_mut().enumerate() {
            *v = ((i % 7) as f32) * 0.1 - 0.3;
        }
        let mut x = vec![0f32; tokens * channels];
        for (i, v) in x.iter_mut().enumerate() {
            *v = ((i % 11) as f32) * 0.05 - 0.2;
        }
        let mut pref = vec![0f32; tokens * channels];
        short_conv(&x, &taps, tokens, channels, &mut pref);
        let mut state = vec![0f32; channels * hist];
        let mut step = vec![0f32; tokens * channels];
        let mut y = vec![0f32; channels];
        for t in 0..tokens {
            let xt = &x[t * channels..(t + 1) * channels];
            let mut next = vec![0f32; channels * hist];
            short_conv_with_state(xt, &taps, 1, channels, Some(&state), &mut y, Some(&mut next));
            step[t * channels..(t + 1) * channels].copy_from_slice(&y);
            state = next;
        }
        let mut maxabs = 0f32;
        for (a, b) in pref.iter().zip(step.iter()) {
            maxabs = maxabs.max((a - b).abs());
        }
        assert!(maxabs < 1e-5, "hidden stepwise maxabs={maxabs}");
        }

    #[test]
    fn dynamic_short_conv_step_matches_prefill() {
        let n_heads = 2usize;
        let head_ch = 4usize;
        let channels = n_heads * head_ch;
        let hidden = 8usize;
        let k = CONV_K;
        let tokens = 5usize;
        let hist = k - 1;
        let mut static_w = vec![0f32; k * channels];
        let mut weight_proj = vec![0f32; n_heads * k * hidden];
        let mut x = vec![0f32; tokens * channels];
        let mut dyn_in = vec![0f32; tokens * hidden];
        for (i, v) in static_w.iter_mut().enumerate() {
            *v = ((i % 5) as f32) * 0.1;
        }
        for (i, v) in weight_proj.iter_mut().enumerate() {
            *v = ((i % 3) as f32) * 0.05 - 0.02;
        }
        for (i, v) in x.iter_mut().enumerate() {
            *v = ((i % 7) as f32) * 0.02;
        }
        for (i, v) in dyn_in.iter_mut().enumerate() {
            *v = ((i % 4) as f32) * 0.03;
        }
        let mut pref = vec![0f32; tokens * channels];
        dynamic_short_conv(
            &x, &dyn_in, &static_w, &weight_proj, tokens, channels, hidden, n_heads, k, &mut pref,
        );
        let mut state = vec![0f32; channels * hist];
        let mut step = vec![0f32; tokens * channels];
        let mut y = vec![0f32; channels];
        for t in 0..tokens {
            let mut next = vec![0f32; channels * hist];
            dynamic_short_conv_step(
                &x[t * channels..(t + 1) * channels],
                &dyn_in[t * hidden..(t + 1) * hidden],
                &static_w,
                &weight_proj,
                channels,
                hidden,
                n_heads,
                k,
                &state,
                &mut y,
                &mut next,
            );
            step[t * channels..(t + 1) * channels].copy_from_slice(&y);
            state = next;
        }
        let mut maxabs = 0f32;
        for (a, b) in pref.iter().zip(step.iter()) {
            maxabs = maxabs.max((a - b).abs());
        }
        assert!(maxabs < 1e-5, "dyn sconv step maxabs={maxabs}");
    }

    #[test]
    fn dynamic_short_conv_decode_inplace_matches_step() {
        let n_heads = 2usize;
        let head_ch = 4usize;
        let channels = n_heads * head_ch;
        let hidden = 8usize;
        let hist = CONV_K - 1;
        let mut static_w = vec![0f32; CONV_K * channels];
        let mut weight_proj = vec![0f32; n_heads * CONV_K * hidden];
        let mut x = vec![0f32; channels];
        let mut dyn_in = vec![0f32; hidden];
        for (i, v) in static_w.iter_mut().enumerate() {
            *v = ((i % 5) as f32) * 0.1;
        }
        for (i, v) in weight_proj.iter_mut().enumerate() {
            *v = ((i % 3) as f32) * 0.05 - 0.02;
        }
        for (i, v) in x.iter_mut().enumerate() {
            *v = ((i % 7) as f32) * 0.02;
        }
        for (i, v) in dyn_in.iter_mut().enumerate() {
            *v = ((i % 4) as f32) * 0.03;
        }
        let state_a = vec![0.05f32; channels * hist];
        let mut state_b = state_a.clone();
        let mut ya = vec![0f32; channels];
        let mut yb = vec![0f32; channels];
        let mut next = vec![0f32; channels * hist];
        dynamic_short_conv_step(
            &x,
            &dyn_in,
            &static_w,
            &weight_proj,
            channels,
            hidden,
            n_heads,
            CONV_K,
            &state_a,
            &mut ya,
            &mut next,
        );
        dynamic_short_conv_decode_inplace(
            &x,
            &dyn_in,
            &static_w,
            &weight_proj,
            channels,
            hidden,
            n_heads,
            &mut state_b,
            &mut yb,
        );
        for i in 0..channels {
            assert!((ya[i] - yb[i]).abs() < 1e-5, "y i={i}");
        }
        for i in 0..state_b.len() {
            assert!((next[i] - state_b[i]).abs() < 1e-5, "state i={i}");
        }
    }
}
