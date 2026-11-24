// Batched dense GEMV: y[t, :] = W @ x[t, :] for t in 0..TOKENS.
// W is row-major [OUT, IN].

#include <metal_stdlib>
using namespace metal;

constant uint TOKENS [[function_constant(0)]];
constant uint OUT_DIM [[function_constant(1)]];
constant uint IN_DIM [[function_constant(2)]];
// 0 => plain GEMV; else apply softcap * tanh(acc / softcap) on write.
constant float SOFTCAP [[function_constant(3)]];

kernel void k_dense_gemv(
    device const float* x [[buffer(0)]], // [T, IN]
    device const float* w [[buffer(1)]], // [OUT, IN]
    device float* y [[buffer(2)]],       // [T, OUT]
    uint2 gid [[thread_position_in_grid]]) {
    uint t = gid.x;
    uint o = gid.y;
    if (t >= TOKENS || o >= OUT_DIM) return;

    device const float* xt = x + t * IN_DIM;
    device const float* row = w + o * IN_DIM;
    float acc = 0.f;
    // float4 when IN_DIM % 4 == 0 (all Inkling GEMVs); cuts loop trip count 4×.
    uint n4 = IN_DIM / 4u;
    for (uint i = 0; i < n4; ++i) {
        uint j = i * 4u;
        float4 xv = float4(xt[j], xt[j + 1u], xt[j + 2u], xt[j + 3u]);
        float4 wv = float4(row[j], row[j + 1u], row[j + 2u], row[j + 3u]);
        acc += dot(xv, wv);
    }
    if (SOFTCAP > 0.f) {
        acc = SOFTCAP * metal::tanh(acc / SOFTCAP);
    }
    y[t * OUT_DIM + o] = acc;
}

// Softcap helper shared by reduce GEMVs.
inline float apply_softcap(float v) {
    if (SOFTCAP > 0.f) return SOFTCAP * metal::tanh(v / SOFTCAP);
    return v;
}

// Single-row T=1 GEMV (any OUT_DIM). Used for unembed where OUT % 4 != 0.
// Launch threadgroups=(OUT_DIM,1), threads=(TG_N,1); tg mem ≥ (TG_N/32)*4.
kernel void k_dense_gemv_reduce_row(
    device const float* x [[buffer(0)]],
    device const float* w [[buffer(1)]],
    device float* y [[buffer(2)]],
    uint lid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tptg [[threads_per_threadgroup]],
    threadgroup float* partials [[threadgroup(0)]]) {
    uint o = lid;
    if (o >= OUT_DIM) return;
    device const float4* row4 = (device const float4*)(w + o * IN_DIM);
    device const float4* x4 = (device const float4*)x;
    uint n4 = IN_DIM / 4u;
    float acc = 0.f;
    for (uint i = tid; i < n4; i += tptg) {
        acc += dot(x4[i], row4[i]);
    }
    acc = simd_sum(acc);
    uint simd_lane = tid & 31u;
    uint simd_id = tid >> 5u;
    uint n_simd = tptg >> 5u;
    if (simd_lane == 0u) partials[simd_id] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        float v = 0.f;
        for (uint s = 0u; s < n_simd; ++s) v += partials[s];
        y[o] = apply_softcap(v);
    }
}

// T=1 f32w GEMV with simdgroup reduction: one TG per output row.
// Launch threadsPerTG=(TG_N,1), threadgroups=(OUT_DIM,1);
// threadgroup(0) ≥ (TG_N/32)*4 bytes.
kernel void k_dense_gemv_reduce(
    device const float* x [[buffer(0)]],
    device const float* w [[buffer(1)]],
    device float* y [[buffer(2)]],
    uint lid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tptg [[threads_per_threadgroup]],
    threadgroup float* partials [[threadgroup(0)]]) {
    uint o = lid;
    if (o >= OUT_DIM) return;
    device const float4* row4 = (device const float4*)(w + o * IN_DIM);
    device const float4* x4 = (device const float4*)x;
    uint n4 = IN_DIM / 4u;
    float acc = 0.f;
    for (uint i = tid; i < n4; i += tptg) {
        acc += dot(x4[i], row4[i]);
    }
    acc = simd_sum(acc);
    uint simd_lane = tid & 31u;
    uint simd_id = tid >> 5u;
    uint n_simd = tptg >> 5u;
    if (simd_lane == 0u) partials[simd_id] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        float v = 0.f;
        for (uint s = 0u; s < n_simd; ++s) v += partials[s];
        y[o] = apply_softcap(v);
    }
}

// Same as k_dense_gemv_reduce but W is float16 (≈2× less bandwidth).
kernel void k_dense_gemv_reduce_f16w(
    device const float* x [[buffer(0)]],
    device const half* w [[buffer(1)]],
    device float* y [[buffer(2)]],
    uint lid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tptg [[threads_per_threadgroup]],
    threadgroup float* partials [[threadgroup(0)]]) {
    uint o = lid;
    if (o >= OUT_DIM) return;
    device const half4* row4 = (device const half4*)(w + o * IN_DIM);
    device const float4* x4 = (device const float4*)x;
    uint n4 = IN_DIM / 4u;
    float acc = 0.f;
    for (uint i = tid; i < n4; i += tptg) {
        acc += dot(x4[i], float4(row4[i]));
    }
    acc = simd_sum(acc);
    uint simd_lane = tid & 31u;
    uint simd_id = tid >> 5u;
    uint n_simd = tptg >> 5u;
    if (simd_lane == 0u) partials[simd_id] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        float v = 0.f;
        for (uint s = 0u; s < n_simd; ++s) v += partials[s];
        y[o] = apply_softcap(v);
    }
}

// Dual T=1 f16w GEMV from the same x (e.g. Q+gate).
// threadgroup(0) ≥ 2*(TG_N/32)*4 bytes.
kernel void k_dense_gemv_reduce_dual_f16w(
    device const float* x [[buffer(0)]],
    device const half* w0 [[buffer(1)]],
    device const half* w1 [[buffer(2)]],
    device float* y0 [[buffer(3)]],
    device float* y1 [[buffer(4)]],
    uint lid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tptg [[threads_per_threadgroup]],
    threadgroup float* partials [[threadgroup(0)]]) {
    uint o = lid;
    if (o >= OUT_DIM) return;
    device const float4* x4 = (device const float4*)x;
    device const half4* r0 = (device const half4*)(w0 + o * IN_DIM);
    device const half4* r1 = (device const half4*)(w1 + o * IN_DIM);
    uint n4 = IN_DIM / 4u;
    float a0 = 0.f;
    float a1 = 0.f;
    for (uint i = tid; i < n4; i += tptg) {
        float4 xv = x4[i];
        a0 += dot(xv, float4(r0[i]));
        a1 += dot(xv, float4(r1[i]));
    }
    a0 = simd_sum(a0);
    a1 = simd_sum(a1);
    uint simd_lane = tid & 31u;
    uint simd_id = tid >> 5u;
    uint n_simd = tptg >> 5u;
    if (simd_lane == 0u) {
        partials[simd_id] = a0;
        partials[n_simd + simd_id] = a1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        float v0 = 0.f;
        float v1 = 0.f;
        for (uint s = 0u; s < n_simd; ++s) {
            v0 += partials[s];
            v1 += partials[n_simd + s];
        }
        y0[o] = apply_softcap(v0);
        y1[o] = apply_softcap(v1);
    }
}

// T=1 f16w GEMV: 8 output rows per TG, shared x cache.
// Launch: threadgroups=ceil(OUT/8), threads=(32,8). threadgroup(0) ≥ IN_DIM*4.
// Each SIMD (tid.y) owns one output row; simd_sum only (no cross-SIMD reduce).
kernel void k_dense_gemv_reduce_f16w_tile8(
    device const float* x [[buffer(0)]],
    device const half* w [[buffer(1)]],
    device float* y [[buffer(2)]],
    uint3 tgid3 [[threadgroup_position_in_grid]],
    uint3 tid3 [[thread_position_in_threadgroup]],
    uint3 tptg3 [[threads_per_threadgroup]],
    threadgroup float* x_tg [[threadgroup(0)]]) {
    constexpr uint ROWS = 8u;
    // Launch is 2-D (32, 8). A scalar `uint` attribute only yields .x, which made
    // every simdgroup compute row 0 of its tile; read uint3 and flatten.
    uint tgid = tgid3.x;
    uint tid = tid3.y * 32u + tid3.x;
    uint tptg = tptg3.x * tptg3.y;
    // Launch threads=(32,8) → flat tid = lane + row*32.
    uint lane = tid & 31u;
    uint row = tid >> 5u;
    for (uint i = tid; i < IN_DIM; i += tptg) {
        x_tg[i] = x[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint o = tgid * ROWS + row;
    if (o >= OUT_DIM) return;
    device const half4* row4 = (device const half4*)(w + o * IN_DIM);
    threadgroup float4* x4 = (threadgroup float4*)x_tg;
    uint n4 = IN_DIM / 4u;
    float acc = 0.f;
    for (uint i = lane; i < n4; i += 32u) {
        acc += dot(x4[i], float4(row4[i]));
    }
    acc = simd_sum(acc);
    if (lane == 0u) y[o] = apply_softcap(acc);
}

// Dual T=1 f16w tile8 (shared x). Same launch; threadgroup(0) ≥ IN_DIM*4.
kernel void k_dense_gemv_reduce_dual_f16w_tile8(
    device const float* x [[buffer(0)]],
    device const half* w0 [[buffer(1)]],
    device const half* w1 [[buffer(2)]],
    device float* y0 [[buffer(3)]],
    device float* y1 [[buffer(4)]],
    uint3 tgid3 [[threadgroup_position_in_grid]],
    uint3 tid3 [[thread_position_in_threadgroup]],
    uint3 tptg3 [[threads_per_threadgroup]],
    threadgroup float* x_tg [[threadgroup(0)]]) {
    constexpr uint ROWS = 8u;
    // Launch is 2-D (32, 8). A scalar `uint` attribute only yields .x, which made
    // every simdgroup compute row 0 of its tile; read uint3 and flatten.
    uint tgid = tgid3.x;
    uint tid = tid3.y * 32u + tid3.x;
    uint tptg = tptg3.x * tptg3.y;
    uint lane = tid & 31u;
    uint row = tid >> 5u;
    for (uint i = tid; i < IN_DIM; i += tptg) {
        x_tg[i] = x[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint o = tgid * ROWS + row;
    if (o >= OUT_DIM) return;
    device const half4* r0 = (device const half4*)(w0 + o * IN_DIM);
    device const half4* r1 = (device const half4*)(w1 + o * IN_DIM);
    threadgroup float4* x4 = (threadgroup float4*)x_tg;
    uint n4 = IN_DIM / 4u;
    float a0 = 0.f, a1 = 0.f;
    for (uint i = lane; i < n4; i += 32u) {
        float4 xv = x4[i];
        a0 += dot(xv, float4(r0[i]));
        a1 += dot(xv, float4(r1[i]));
    }
    a0 = simd_sum(a0);
    a1 = simd_sum(a1);
    if (lane == 0u) {
        y0[o] = apply_softcap(a0);
        y1[o] = apply_softcap(a1);
    }
}

// T=1 f16w GEMV with x cached in threadgroup. Same launch as k_dense_gemv_reduce
// (one TG per output row, TG_N=256) but x is loaded once into TG memory so the
// W stream hits local x. threadgroup(0) ≥ IN_DIM*4 + (TG_N/32)*4 bytes:
// [0..IN_DIM) = x, [IN_DIM .. IN_DIM+n_simd) = simd partials.
kernel void k_dense_gemv_reduce_f16w_tgx(
    device const float* x [[buffer(0)]],
    device const half* w [[buffer(1)]],
    device float* y [[buffer(2)]],
    uint lid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tptg [[threads_per_threadgroup]],
    threadgroup float* tg [[threadgroup(0)]]) {
    uint o = lid;
    threadgroup float* x_tg = tg;
    for (uint i = tid; i < IN_DIM; i += tptg) {
        x_tg[i] = x[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (o >= OUT_DIM) return;
    device const half4* row4 = (device const half4*)(w + o * IN_DIM);
    threadgroup float4* x4 = (threadgroup float4*)x_tg;
    uint n4 = IN_DIM / 4u;
    float acc = 0.f;
    for (uint i = tid; i < n4; i += tptg) {
        acc += dot(x4[i], float4(row4[i]));
    }
    acc = simd_sum(acc);
    uint simd_lane = tid & 31u;
    uint simd_id = tid >> 5u;
    uint n_simd = tptg >> 5u;
    threadgroup float* partials = tg + IN_DIM;
    if (simd_lane == 0u) partials[simd_id] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        float v = 0.f;
        for (uint s = 0u; s < n_simd; ++s) v += partials[s];
        y[o] = apply_softcap(v);
    }
}

// Dual T=1 f16w; x cached once. threadgroup(0) ≥ IN_DIM*4 + 2*(TG_N/32)*4.
kernel void k_dense_gemv_reduce_dual_f16w_tgx(
    device const float* x [[buffer(0)]],
    device const half* w0 [[buffer(1)]],
    device const half* w1 [[buffer(2)]],
    device float* y0 [[buffer(3)]],
    device float* y1 [[buffer(4)]],
    uint lid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tptg [[threads_per_threadgroup]],
    threadgroup float* tg [[threadgroup(0)]]) {
    uint o = lid;
    threadgroup float* x_tg = tg;
    for (uint i = tid; i < IN_DIM; i += tptg) {
        x_tg[i] = x[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (o >= OUT_DIM) return;
    threadgroup float4* x4 = (threadgroup float4*)x_tg;
    device const half4* r0 = (device const half4*)(w0 + o * IN_DIM);
    device const half4* r1 = (device const half4*)(w1 + o * IN_DIM);
    uint n4 = IN_DIM / 4u;
    float a0 = 0.f;
    float a1 = 0.f;
    for (uint i = tid; i < n4; i += tptg) {
        float4 xv = x4[i];
        a0 += dot(xv, float4(r0[i]));
        a1 += dot(xv, float4(r1[i]));
    }
    a0 = simd_sum(a0);
    a1 = simd_sum(a1);
    uint simd_lane = tid & 31u;
    uint simd_id = tid >> 5u;
    uint n_simd = tptg >> 5u;
    threadgroup float* partials = tg + IN_DIM;
    if (simd_lane == 0u) {
        partials[simd_id] = a0;
        partials[n_simd + simd_id] = a1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        float v0 = 0.f;
        float v1 = 0.f;
        for (uint s = 0u; s < n_simd; ++s) {
            v0 += partials[s];
            v1 += partials[n_simd + s];
        }
        y0[o] = apply_softcap(v0);
        y1[o] = apply_softcap(v1);
    }
}

// Contiguous float copy (append KV row, etc.).
kernel void k_copy_f32(
    device const float* src [[buffer(0)]],
    device float* dst [[buffer(1)]],
    device const uint* n_buf [[buffer(2)]],
    uint i [[thread_position_in_grid]]) {
    uint n = n_buf[0];
    if (i < n) dst[i] = src[i];
}

// Geometry of the decode attention kernel (k_decode_sdpa_gate).
constant uint N_HEADS_D [[function_constant(4)]];
constant uint N_KV_D [[function_constant(5)]];
constant uint DH_D [[function_constant(6)]];
constant uint D_REL_D [[function_constant(7)]];
constant uint REL_EXTENT_D [[function_constant(8)]];
constant uint SWA_D [[function_constant(9)]];

// Tree-reduce argmax over Y[0..N). Launch threadgroups=ceil(N/TG), TG threads.
// Each TG writes its local winner to partial_idx/partial_val[tg]; host does a
// second tiny reduce, or dispatch again with N'=n_tg.
kernel void k_argmax_partial(
    device const float* y [[buffer(0)]],
    device int* partial_idx [[buffer(1)]],
    device float* partial_val [[buffer(2)]],
    device const uint* n_buf [[buffer(3)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tptg [[threads_per_threadgroup]],
    threadgroup float* vals [[threadgroup(0)]],
    threadgroup uint* idxs [[threadgroup(1)]]) {
    uint n = n_buf[0];
    uint i = tgid * tptg + tid;
    float v = (i < n) ? y[i] : -1e30f;
    uint idx = (i < n) ? i : 0u;
    vals[tid] = v;
    idxs[tid] = idx;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tptg / 2u; stride > 0u; stride /= 2u) {
        if (tid < stride) {
            if (vals[tid + stride] > vals[tid]) {
                vals[tid] = vals[tid + stride];
                idxs[tid] = idxs[tid + stride];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) {
        partial_idx[tgid] = int(idxs[0]);
        partial_val[tgid] = vals[0];
    }
}

// RMSNorm with affine weight: out = rms(x) * weight. weight may be null if has_w=0.
kernel void k_rms_norm_affine(
    device const float* x [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device float* out [[buffer(2)]],
    device const uint* meta [[buffer(3)]], // [n, has_w]
    uint tid [[thread_position_in_threadgroup]],
    uint tptg [[threads_per_threadgroup]],
    threadgroup float* partials [[threadgroup(0)]]) {
    uint n = meta[0];
    uint has_w = meta[1];
    float acc = 0.f;
    for (uint i = tid; i < n; i += tptg) {
        float v = x[i];
        acc += v * v;
    }
    acc = simd_sum(acc);
    uint simd_lane = tid & 31u;
    uint simd_id = tid >> 5u;
    uint n_simd = tptg >> 5u;
    if (simd_lane == 0u) partials[simd_id] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        float v = 0.f;
        for (uint s = 0u; s < n_simd; ++s) v += partials[s];
        partials[0] = metal::rsqrt(v / float(n) + 1e-6f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv = partials[0];
    for (uint i = tid; i < n; i += tptg) {
        float v = x[i] * inv;
        out[i] = (has_w != 0u) ? v * weight[i] : v;
    }
}

// T=1 residual mix: softmax over rms-normed (states + n_extra extras), weighted sum.
// meta: [n_states, hidden, normalize_values, n_extra] with n_extra in {1,2}.
// extra1 used only when n_extra==2 (block-boundary mlp: x_in + residual).
// Launch TG=256.
kernel void k_residual_mix_t1(
    device const float* states [[buffer(0)]],
    device const float* extra [[buffer(1)]],
    device const float* query [[buffer(2)]],
    device float* out [[buffer(3)]],
    device const uint* meta [[buffer(4)]],
    device const float* extra1 [[buffer(5)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tptg [[threads_per_threadgroup]],
    threadgroup float* tg [[threadgroup(0)]]) {
    uint n_states = meta[0];
    uint hidden = meta[1];
    uint normalize = meta[2];
    uint n_extra = meta[3];
    if (n_extra < 1u) n_extra = 1u;
    if (n_extra > 2u) n_extra = 2u;
    uint n = n_states + n_extra;
    if (n == 0u || n > 16u) return;
    threadgroup float* scores = tg;       // [16]
    threadgroup float* inv_rms = tg + 16; // [16]
    threadgroup float* partial = tg + 32; // ≥ n_simd floats (TG=256 → 8)
    uint simd_lane = tid & 31u;
    uint simd_id = tid >> 5u;
    uint n_simd = tptg >> 5u;
    for (uint s = 0; s < n; ++s) {
        device const float* src;
        if (s < n_states) src = states + s * hidden;
        else if (s == n_states) src = extra;
        else src = extra1;
        float acc = 0.f;
        for (uint i = tid; i < hidden; i += tptg) {
            float v = src[i];
            acc += v * v;
        }
        acc = simd_sum(acc);
        if (simd_lane == 0u) partial[simd_id] = acc;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0u) {
            float v = 0.f;
            for (uint i = 0u; i < n_simd; ++i) v += partial[i];
            inv_rms[s] = metal::rsqrt(v / float(hidden) + 1e-6f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float scale = metal::rsqrt(float(hidden));
    for (uint s = 0; s < n; ++s) {
        device const float* src;
        if (s < n_states) src = states + s * hidden;
        else if (s == n_states) src = extra;
        else src = extra1;
        float inv = inv_rms[s];
        float acc = 0.f;
        for (uint i = tid; i < hidden; i += tptg) acc += (src[i] * inv) * query[i];
        acc = simd_sum(acc);
        if (simd_lane == 0u) partial[simd_id] = acc;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0u) {
            float v = 0.f;
            for (uint i = 0u; i < n_simd; ++i) v += partial[i];
            scores[s] = v * scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) {
        float maxv = -1e30f;
        for (uint s = 0; s < n; ++s) maxv = metal::max(maxv, scores[s]);
        float sum = 0.f;
        for (uint s = 0; s < n; ++s) {
            float e = metal::exp(scores[s] - maxv);
            scores[s] = e;
            sum += e;
        }
        float inv = 1.f / sum;
        for (uint s = 0; s < n; ++s) scores[s] *= inv;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < hidden; i += tptg) {
        float acc = 0.f;
        for (uint s = 0; s < n; ++s) {
            device const float* src;
            if (s < n_states) src = states + s * hidden;
            else if (s == n_states) src = extra;
            else src = extra1;
            float v = src[i];
            if (normalize != 0u) v *= inv_rms[s];
            acc += scores[s] * v;
        }
        out[i] = acc;
    }
}


// Copy HIDDEN floats: dst[off..] = src[0..H). Launch ceil(H/256) groups × 256 threads.
kernel void k_copy_hidden(
    device const float* src [[buffer(0)]],
    device float* dst [[buffer(1)]],
    device const uint* meta [[buffer(2)]], // [H, dst_off_elements]
    uint gid [[thread_position_in_grid]]) {
    uint H = meta[0];
    uint off = meta[1];
    if (gid < H) dst[off + gid] = src[gid];
}

// Router top-8, one threadgroup per token: logits[T][128], bias[128] →
// indices[T][8], weights[T][8].
// Softcap-sigmoid scores, add bias for ranking, renormalize by 8/sum.
kernel void k_route_topk_t1(
    device const float* logits [[buffer(0)]],
    device const float* bias [[buffer(1)]],
    device int* indices [[buffer(2)]],
    device float* weights [[buffer(3)]],
    uint tid [[thread_position_in_threadgroup]],
    uint row [[threadgroup_position_in_grid]],
    threadgroup float* scores [[threadgroup(0)]],
    threadgroup float* keyed [[threadgroup(1)]],
    threadgroup uint* order [[threadgroup(2)]]) {
    constexpr uint N = 128u;
    constexpr uint K = 8u;
    constexpr float ROUTE_SCALE = 8.f;
    // One threadgroup per token: decode launches one, a block one per row.
    logits += row * N;
    indices += row * K;
    weights += row * K;
    // Compute sigmoid(logit) scores.
    for (uint e = tid; e < N; e += 32u) {
        float z = logits[e];
        float s = 1.f / (1.f + metal::exp(-z));
        scores[e] = s;
        keyed[e] = s + bias[e];
        order[e] = e;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // SIMD-parallel top-K: each of 32 lanes scans N/32 keyed values per round.
    // Tie-break: lower expert index wins (matches serial select).
    for (uint i = 0u; i < K; ++i) {
        float best_v = keyed[tid];
        uint best_e = tid;
        for (uint j = tid + 32u; j < N; j += 32u) {
            float v = keyed[j];
            if (v > best_v || (v == best_v && j < best_e)) {
                best_v = v;
                best_e = j;
            }
        }
        float m = simd_max(best_v);
        uint cand = (best_v == m) ? best_e : 0xffffffffu;
        uint win = simd_min(cand);
        if (tid == 0u) {
            order[i] = win;
            keyed[win] = -1e30f; // mask out for next round
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) {
        // Sort the K winners descending by original keyed (= scores+bias).
        // keyed[win] was masked; re-rank via scores[e]+bias[e].
        for (uint i = 0u; i < K; ++i) {
            for (uint j = i + 1u; j < K; ++j) {
                float ki = scores[order[i]] + bias[order[i]];
                float kj = scores[order[j]] + bias[order[j]];
                if (kj > ki || (kj == ki && order[j] < order[i])) {
                    uint tmp = order[i];
                    order[i] = order[j];
                    order[j] = tmp;
                }
            }
        }
        float sum = 0.f;
        for (uint r = 0; r < K; ++r) sum += scores[order[r]];
        float inv = ROUTE_SCALE / (sum + 1e-20f);
        for (uint r = 0; r < K; ++r) {
            uint e = order[r];
            indices[r] = int(e);
            weights[r] = scores[e] * inv;
        }
    }
}

// Final reduce over k_argmax_partial outputs. One TG (TG=256). Ties → lower index.
// n_buf[0] = number of partials. Writes the winning row index to out_idx[0].
kernel void k_argmax_final(
    device const int* partial_idx [[buffer(0)]],
    device const float* partial_val [[buffer(1)]],
    device const uint* n_buf [[buffer(2)]],
    device int* out_idx [[buffer(3)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tptg [[threads_per_threadgroup]],
    threadgroup float* vals [[threadgroup(0)]],
    threadgroup int* idxs [[threadgroup(1)]]) {
    uint n = n_buf[0];
    float bv = -1e30f;
    int bi = 0;
    bool have = false;
    for (uint i = tid; i < n; i += tptg) {
        float v = partial_val[i];
        if (!have || v > bv) {
            bv = v;
            bi = partial_idx[i];
            have = true;
        }
    }
    vals[tid] = bv;
    idxs[tid] = bi;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tptg / 2u; stride > 0u; stride /= 2u) {
        if (tid < stride) {
            float ov = vals[tid + stride];
            int oi = idxs[tid + stride];
            if (ov > vals[tid] || (ov == vals[tid] && oi < idxs[tid])) {
                vals[tid] = ov;
                idxs[tid] = oi;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) out_idx[0] = idxs[0];
}

// Message bank rebuild at core reentry (GPU port of `rebuild_message_bank_into`).
// bank[r] = rms_norm(blocks[(skip + r)]) for r < n_core, bank[n_core] = rms_norm(x).
// meta: [n_core, hidden, skip]. Launch n_core+1 TGs × 256 threads;
// threadgroup(0) = 256 floats.
kernel void k_build_bank_t1(
    device const float* blocks [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* bank [[buffer(2)]],
    device const uint* meta [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tptg [[threads_per_threadgroup]],
    threadgroup float* partial [[threadgroup(0)]]) {
    uint n_core = meta[0];
    uint hidden = meta[1];
    uint skip = meta[2];
    device const float* src = (row < n_core) ? (blocks + (skip + row) * hidden) : x;
    device float* dst = bank + row * hidden;
    float acc = 0.f;
    for (uint i = tid; i < hidden; i += tptg) {
        float v = src[i];
        acc += v * v;
    }
    partial[tid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tptg / 2u; s > 0u; s /= 2u) {
        if (tid < s) partial[tid] += partial[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv = metal::rsqrt(partial[0] / float(hidden) + 1e-6f);
    for (uint i = tid; i < hidden; i += tptg) dst[i] = src[i] * inv;
}

// GPU `gather_messages`: for each query q (one TG each), softmax-attend over the bank.
// meta: [n_bank, hidden, n_q]; n_bank <= 32. Launch n_q TGs × 256 threads;
// threadgroup(0) = (256 + 32) floats.
kernel void k_gather_messages_t1(
    device const float* bank [[buffer(0)]],
    device const float* queries [[buffer(1)]],
    device float* out [[buffer(2)]],
    device const uint* meta [[buffer(3)]],
    uint q [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tptg [[threads_per_threadgroup]],
    threadgroup float* tg [[threadgroup(0)]]) {
    uint n = meta[0];
    uint hidden = meta[1];
    if (n == 0u || n > 32u) return;
    threadgroup float* partial = tg;       // [256]
    threadgroup float* scores = tg + 256;  // [32]
    device const float* query = queries + q * hidden;
    device float* dst = out + q * hidden;
    float scale = metal::rsqrt(float(hidden));
    for (uint s = 0; s < n; ++s) {
        device const float* row = bank + s * hidden;
        float acc = 0.f;
        for (uint i = tid; i < hidden; i += tptg) acc += query[i] * row[i];
        partial[tid] = acc;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = tptg / 2u; stride > 0u; stride /= 2u) {
            if (tid < stride) partial[tid] += partial[tid + stride];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (tid == 0u) scores[s] = partial[0] * scale;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) {
        float maxv = -1e30f;
        for (uint s = 0; s < n; ++s) maxv = metal::max(maxv, scores[s]);
        float sum = 0.f;
        for (uint s = 0; s < n; ++s) {
            float e = metal::exp(scores[s] - maxv);
            scores[s] = e;
            sum += e;
        }
        float inv = 1.f / sum;
        for (uint s = 0; s < n; ++s) scores[s] *= inv;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < hidden; i += tptg) {
        float acc = 0.f;
        for (uint s = 0; s < n; ++s) acc += scores[s] * bank[s * hidden + i];
        dst[i] = acc;
    }
}

// ============================================================================
// Staged glue for decode T=1 (driven from engine/src/metal_glue.rs).
//
// A kernel that runs in one threadgroup runs on one GPU core. These split the
// mix, message-controller and norm math into a few stages that are each spread
// over many threadgroups, and several jobs (q/k/v) share a stage.
// ============================================================================

// Residual mix, one stage. Sources are states[s] for s < n_states, then `extra`,
// then `extra1`. For each query j < nq:
//   w_j = softmax_s(src_s.q_j * inv_rms[s] / sqrt(H))
//   out_j[i] = sum_s w_j[s] * src_s[i] * (normalize ? inv_rms[s] : 1)
// Every threadgroup reduces the sources itself (simdgroup g takes sources g and
// g + 8), so the weights need no stage of their own. Also writes per-simdgroup sums
// of out_j^2 to partials[j * H/32 + ...] so a later stage can RMS-normalize without
// another reduction pass.
// meta = [n_states, n_extra, nq, normalize], at most 16 sources.
// Launch threadgroups=(H/256), threads=256. threadgroup(0) >= 64 floats.
kernel void k_mix(
    device const float* states [[buffer(0)]],
    device const float* extra [[buffer(1)]],
    device const float* extra1 [[buffer(2)]],
    device const float* q0 [[buffer(3)]],
    device const float* q1 [[buffer(4)]],
    device const float* q2 [[buffer(5)]],
    device const uint* meta [[buffer(6)]],
    device float* out0 [[buffer(7)]],
    device float* out1 [[buffer(8)]],
    device float* out2 [[buffer(9)]],
    device float* partials [[buffer(10)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    threadgroup float* tg [[threadgroup(0)]]) {
    constexpr uint H = 2304u;
    constexpr uint NP = H / 32u;
    uint n_states = meta[0];
    uint n = n_states + meta[1];
    uint nq = meta[2];
    bool normalize = meta[3] != 0u;
    uint lane = tid & 31u;
    // tg[0..16) = inv_rms; tg[16 + 16 j + s] = score, then weight, of source s for query j.
    for (uint s = tid >> 5u; s < n; s += 8u) {
        device const float4* src = (device const float4*)(
            s < n_states ? states + s * H : (s == n_states ? extra : extra1));
        device const float4* a = (device const float4*)q0;
        device const float4* b = (device const float4*)q1;
        device const float4* c = (device const float4*)q2;
        float ss = 0.f, d0 = 0.f, d1 = 0.f, d2 = 0.f;
        for (uint i = lane; i < H / 4u; i += 32u) {
            float4 v = src[i];
            ss += dot(v, v);
            d0 += dot(v, a[i]);
            d1 += dot(v, b[i]);
            d2 += dot(v, c[i]);
        }
        ss = simd_sum(ss);
        d0 = simd_sum(d0);
        d1 = simd_sum(d1);
        d2 = simd_sum(d2);
        if (lane == 0u) {
            float inv = metal::rsqrt(ss / float(H) + 1e-6f);
            float sc = inv * metal::rsqrt(float(H));
            tg[s] = inv;
            tg[16u + s] = d0 * sc;
            tg[32u + s] = d1 * sc;
            tg[48u + s] = d2 * sc;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 3u) {
        threadgroup float* w = tg + 16u + 16u * tid;
        float maxv = -1e30f;
        for (uint s = 0u; s < n; ++s) maxv = metal::max(maxv, w[s]);
        float sum = 0.f;
        for (uint s = 0u; s < n; ++s) {
            float e = metal::exp(w[s] - maxv);
            w[s] = e;
            sum += e;
        }
        float inv = 1.f / sum;
        for (uint s = 0u; s < n; ++s) w[s] *= normalize ? inv * tg[s] : inv;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint i = tgid * 256u + tid;
    float o0 = 0.f, o1 = 0.f, o2 = 0.f;
    if (i < H) {
        for (uint s = 0u; s < n; ++s) {
            device const float* src =
                s < n_states ? states + s * H : (s == n_states ? extra : extra1);
            float v = src[i];
            o0 += tg[16u + s] * v;
            o1 += tg[32u + s] * v;
            o2 += tg[48u + s] * v;
        }
        out0[i] = o0;
        if (nq > 1u) {
            out1[i] = o1;
            out2[i] = o2;
        }
    }
    uint p = tgid * 8u + (tid >> 5u);
    float s0 = simd_sum(o0 * o0);
    float s1 = simd_sum(o1 * o1);
    float s2 = simd_sum(o2 * o2);
    if ((tid & 31u) == 0u) {
        partials[p] = s0;
        if (nq > 1u) {
            partials[NP + p] = s1;
            partials[2u * NP + p] = s2;
        }
    }
}

// Message controller, stage 1: one simdgroup per row.
// rows 0..63: dn[r] = x.down[r, :H], dn[64 + r] = msg.down[r, H:]
// row 64:     dn[128] = |x|^2,       dn[129] = |msg|^2
// (h = dn[r] * inv_rms(x) + dn[64 + r] * inv_rms(msg) is formed in stage 2.)
// Launch threadgroups=65, threads=32.
kernel void k_ctrl_down(
    device const float* x [[buffer(0)]],
    device const float* msg [[buffer(1)]],
    device const float* down [[buffer(2)]], // [64, 2H]
    device float* dn [[buffer(3)]],         // [130]
    uint r [[threadgroup_position_in_grid]],
    uint lane [[thread_position_in_threadgroup]]) {
    constexpr uint H = 2304u;
    constexpr uint RANK = 64u;
    device const float4* x4 = (device const float4*)x;
    device const float4* m4 = (device const float4*)msg;
    float a = 0.f, b = 0.f;
    if (r < RANK) {
        device const float4* dx = (device const float4*)(down + r * 2u * H);
        device const float4* dm = dx + H / 4u;
        for (uint i = lane; i < H / 4u; i += 32u) {
            a += dot(x4[i], dx[i]);
            b += dot(m4[i], dm[i]);
        }
    } else {
        for (uint i = lane; i < H / 4u; i += 32u) {
            float4 xv = x4[i];
            float4 mv = m4[i];
            a += dot(xv, xv);
            b += dot(mv, mv);
        }
    }
    a = simd_sum(a);
    b = simd_sum(b);
    if (lane == 0u) {
        if (r < RANK) {
            dn[r] = a;
            dn[RANK + r] = b;
        } else {
            dn[2u * RANK] = a;
            dn[2u * RANK + 1u] = b;
        }
    }
}

// Message controller, stage 2: blend[i] = (1 - tanh(Wf[i].h)) x[i] + tanh(Wg[i].h) msg[i].
// One thread per element; h is rebuilt once per threadgroup from stage 1.
// Also writes per-simdgroup sums of blend^2 to partials[H/32].
// Launch threadgroups=(H/256), threads=256. threadgroup(0) >= 64 floats.
kernel void k_ctrl_apply(
    device const float* x [[buffer(0)]],
    device const float* msg [[buffer(1)]],
    device const float* w_forget [[buffer(2)]], // [H, 64]
    device const float* w_gain [[buffer(3)]],   // [H, 64]
    device const float* dn [[buffer(4)]],
    device float* blend [[buffer(5)]],
    device float* partials [[buffer(6)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    threadgroup float* h_tg [[threadgroup(0)]]) {
    constexpr uint H = 2304u;
    constexpr uint RANK = 64u;
    if (tid < RANK) {
        float inv_x = metal::rsqrt(dn[2u * RANK] / float(H) + 1e-6f);
        float inv_m = metal::rsqrt(dn[2u * RANK + 1u] / float(H) + 1e-6f);
        h_tg[tid] = dn[tid] * inv_x + dn[RANK + tid] * inv_m;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint i = tgid * 256u + tid;
    float v = 0.f;
    if (i < H) {
        device const float4* fr = (device const float4*)(w_forget + i * RANK);
        device const float4* gr = (device const float4*)(w_gain + i * RANK);
        threadgroup float4* h4 = (threadgroup float4*)h_tg;
        float fd = 0.f, gd = 0.f;
        for (uint r = 0u; r < RANK / 4u; ++r) {
            float4 hv = h4[r];
            fd += dot(fr[r], hv);
            gd += dot(gr[r], hv);
        }
        v = (1.f - metal::tanh(fd)) * x[i] + metal::tanh(gd) * msg[i];
        blend[i] = v;
    }
    float s = simd_sum(v * v);
    if ((tid & 31u) == 0u) partials[tgid * 8u + (tid >> 5u)] = s;
}

// RMS norm with affine weight, using sums of v^2 left by a previous stage:
// out[i] = v[i] * rsqrt(sum(partials) / H + eps) * weight[i].
// Launch threadgroups=(H/256), threads=256.
kernel void k_norm_from_partials(
    device const float* v [[buffer(0)]],
    device const float* partials [[buffer(1)]], // [H/32]
    device const float* weight [[buffer(2)]],
    device float* out [[buffer(3)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]) {
    constexpr uint H = 2304u;
    constexpr uint NP = H / 32u;
    uint lane = tid & 31u;
    float acc = 0.f;
    for (uint p = lane; p < NP; p += 32u) acc += partials[p];
    float inv = metal::rsqrt(simd_sum(acc) / float(H) + 1e-6f);
    if (gid < H) out[gid] = v[gid] * inv * weight[gid];
}

// K or V after projection: short-conv into the KV ring row, then per-head
// normalization, in one stage. One threadgroup per KV head, 128 threads (one channel
// each). meta = [mode, dynamic]: mode 0 = L2 clip to thr_buf[0] (K), mode 1 = RMS
// norm without gain (V). dynamic 0: taps = [C, 4], oldest tap first (sliding layers).
// dynamic 1: taps = static [4, C], newest first, plus per-token dyn = [n_heads * 4]
// (full-attention layers). state = [C, 3], oldest to newest.
// Launch threadgroups=8, threads=128. threadgroup(0) >= 4 floats.
kernel void k_kv_sconv_head(
    device const float* x [[buffer(0)]],
    device const float* taps [[buffer(1)]],
    device float* state [[buffer(2)]],
    device float* out [[buffer(3)]],
    device const uint* meta [[buffer(4)]],
    device const float* thr_buf [[buffer(5)]],
    device const float* dyn [[buffer(6)]],
    uint h [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    threadgroup float* part [[threadgroup(0)]]) {
    constexpr uint DH = 128u;
    constexpr uint C = 8u * DH;
    constexpr uint K = 4u;
    uint c = h * DH + tid;
    uint sbase = c * 3u;
    float xv = x[c];
    float s0 = state[sbase];
    float s1 = state[sbase + 1u];
    float s2 = state[sbase + 2u];
    float v;
    if (meta[1] == 0u) {
        uint tb = c * 4u;
        float acc = taps[tb] * s0 + taps[tb + 1u] * s1 + taps[tb + 2u] * s2 + taps[tb + 3u] * xv;
        v = (acc + xv) * metal::rsqrt(5.f);
    } else {
        float acc = xv * (taps[c] + dyn[h * K])
            + s2 * (taps[C + c] + dyn[h * K + 1u])
            + s1 * (taps[2u * C + c] + dyn[h * K + 2u])
            + s0 * (taps[3u * C + c] + dyn[h * K + 3u]);
        v = (acc + xv) * metal::rsqrt(2.f * float(K) + 1.f);
    }
    state[sbase] = s1;
    state[sbase + 1u] = s2;
    state[sbase + 2u] = xv;
    float ss = simd_sum(v * v);
    if ((tid & 31u) == 0u) part[tid >> 5u] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float n2 = part[0] + part[1] + part[2] + part[3];
    float f = (meta[0] == 0u)
        ? metal::min(thr_buf[0] / (metal::sqrt(n2) + 1e-6f), 1.f)
        : metal::rsqrt(n2 / float(DH) + 1e-6f);
    out[c] = v * f;
}

// In-place scaled L2 clip, one threadgroup per head.
// meta = [n_rows, dh, has_scale, _].
// Launch threadgroups=n_rows, threads=dh rounded up to 32. threadgroup(0) >= 4 floats.
kernel void k_l2_clip_head(
    device float* x [[buffer(0)]],
    device const half* scale [[buffer(1)]],
    device const uint* meta [[buffer(2)]],
    device const float* thr_buf [[buffer(3)]],
    uint h [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tptg [[threads_per_threadgroup]],
    threadgroup float* part [[threadgroup(0)]]) {
    uint dh = meta[1];
    bool live = tid < dh;
    uint i = h * dh + tid;
    float v = 0.f;
    if (live) {
        v = x[i];
        if (meta[2] != 0u) v *= float(scale[i]);
    }
    float ss = simd_sum(v * v);
    if ((tid & 31u) == 0u) part[tid >> 5u] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float n2 = 0.f;
    for (uint s = 0u; s < (tptg >> 5u); ++s) n2 += part[s];
    float f = metal::min(thr_buf[0] / (metal::sqrt(n2) + 1e-6f), 1.f);
    if (live) x[i] = v * f;
}

// Short-conv with rolling state, then an optional add:
// out[c] = short_conv(x)[c] + (meta[1] ? add[c] : 0). meta = [channels, has_add].
// taps = [C, 4], state = [C, 3]. Launch threadgroups=ceil(C/64), threads=64.
kernel void k_short_conv_add(
    device const float* x [[buffer(0)]],
    device const float* taps [[buffer(1)]],
    device float* state [[buffer(2)]],
    device float* out [[buffer(3)]],
    device const uint* meta [[buffer(4)]],
    device const float* add [[buffer(5)]],
    uint c [[thread_position_in_grid]]) {
    if (c >= meta[0]) return;
    uint tbase = c * 4u;
    uint sbase = c * 3u;
    float xv = x[c];
    float s0 = state[sbase];
    float s1 = state[sbase + 1u];
    float s2 = state[sbase + 2u];
    float acc = taps[tbase] * s0 + taps[tbase + 1u] * s1 + taps[tbase + 2u] * s2
        + taps[tbase + 3u] * xv;
    float v = (acc + xv) * metal::rsqrt(5.f);
    out[c] = meta[1] != 0u ? v + add[c] : v;
    state[sbase] = s1;
    state[sbase + 1u] = s2;
    state[sbase + 2u] = xv;
}

// ---- Opt-in 8-bit projections (INKLING_ATTN_Q8=1; lossy, see metal_q8.rs) ----
// int8 weights [OUT, IN] with one f32 scale per output row. One simdgroup owns 4
// output rows and reads x once per step for all of them; 16-byte weight loads.
// Launch threadgroups=ceil(OUT/32), threads=(32, 8). OUT % 4 == 0, IN % 16 == 0.
kernel void k_dense_gemv_q8(
    device const float* x [[buffer(0)]],
    device const char* w [[buffer(1)]],
    device const float* scale [[buffer(2)]],
    device float* y [[buffer(3)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 tptg [[threads_per_threadgroup]]) {
    constexpr uint R = 4u;
    uint lane = tid.x;
    uint row0 = (tg.x * tptg.y + tid.y) * R;
    if (row0 >= OUT_DIM) return;
    device const float4* x4 = (device const float4*)x;
    uint n16 = IN_DIM / 16u;
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    for (uint i = lane; i < n16; i += 32u) {
        float4 x0 = x4[4u * i], x1 = x4[4u * i + 1u], x2 = x4[4u * i + 2u], x3 = x4[4u * i + 3u];
        for (uint r = 0u; r < R; ++r) {
            device const uint4* row = (device const uint4*)(w + (row0 + r) * IN_DIM);
            uint4 u = row[i];
            acc[r] += dot(x0, float4(as_type<char4>(u.x))) + dot(x1, float4(as_type<char4>(u.y)))
                    + dot(x2, float4(as_type<char4>(u.z))) + dot(x3, float4(as_type<char4>(u.w)));
        }
    }
    for (uint r = 0u; r < R; ++r) {
        float s = simd_sum(acc[r]);
        if (lane == 0u) y[row0 + r] = apply_softcap(s * scale[row0 + r]);
    }
}

// Two 8-bit GEMVs over the same x (q + gate). Same launch as k_dense_gemv_q8.
kernel void k_dense_gemv_dual_q8(
    device const float* x [[buffer(0)]],
    device const char* w0 [[buffer(1)]],
    device const float* scale0 [[buffer(2)]],
    device const char* w1 [[buffer(3)]],
    device const float* scale1 [[buffer(4)]],
    device float* y0 [[buffer(5)]],
    device float* y1 [[buffer(6)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 tptg [[threads_per_threadgroup]]) {
    constexpr uint R = 4u;
    uint lane = tid.x;
    uint row0 = (tg.x * tptg.y + tid.y) * R;
    if (row0 >= OUT_DIM) return;
    device const float4* x4 = (device const float4*)x;
    uint n16 = IN_DIM / 16u;
    float a0[4] = {0.f, 0.f, 0.f, 0.f};
    float a1[4] = {0.f, 0.f, 0.f, 0.f};
    for (uint i = lane; i < n16; i += 32u) {
        float4 x0 = x4[4u * i], x1 = x4[4u * i + 1u], x2 = x4[4u * i + 2u], x3 = x4[4u * i + 3u];
        for (uint r = 0u; r < R; ++r) {
            uint4 u = ((device const uint4*)(w0 + (row0 + r) * IN_DIM))[i];
            uint4 v = ((device const uint4*)(w1 + (row0 + r) * IN_DIM))[i];
            a0[r] += dot(x0, float4(as_type<char4>(u.x))) + dot(x1, float4(as_type<char4>(u.y)))
                   + dot(x2, float4(as_type<char4>(u.z))) + dot(x3, float4(as_type<char4>(u.w)));
            a1[r] += dot(x0, float4(as_type<char4>(v.x))) + dot(x1, float4(as_type<char4>(v.y)))
                   + dot(x2, float4(as_type<char4>(v.z))) + dot(x3, float4(as_type<char4>(v.w)));
        }
    }
    for (uint r = 0u; r < R; ++r) {
        float s0 = simd_sum(a0[r]);
        float s1 = simd_sum(a1[r]);
        if (lane == 0u) {
            y0[row0 + r] = apply_softcap(s0 * scale0[row0 + r]);
            y1[row0 + r] = apply_softcap(s1 * scale1[row0 + r]);
        }
    }
}

// Element type of the archive rows: f32, or f16 when the host defines ARCH_F16.
#ifdef ARCH_F16
typedef half arch_t;
typedef half4 arch4_t;
#else
typedef float arch_t;
typedef float4 arch4_t;
#endif

// Floats per (far chunk, head) partial: [max, sum, 0, 0, ctx[DH]].
constant uint FAR_STRIDE_D = 132u;
// Archive rows one threadgroup of k_decode_far_part covers.
constant uint FAR_CHUNK_D = 512u;

// Decode T=1 attention over a short history, in one stage: one threadgroup per query
// head covers logits (dot product plus relative logits), the sink, the softmax, the
// weighted sum of V and the output gate.  Each of its eight simdgroups owns a
// contiguous run of keys, and a lane owns four dims of a row.
//
// It reads every K and V row once per query head, three times in all, and its 24
// threadgroups are all the parallelism it gets.  That is the cheaper way while the
// history is short, because it needs no second stage.  Past RING_SPLIT rows the
// engine switches to k_decode_ring_part and k_decode_merge.
// meta = [n, ..]: n valid ring rows, no wrap yet, the newest one last.
// Launch threadgroups=N_HEADS, threads=256.
// threadgroup(0) >= (MAX_T + 4 + 16 + 8 * DH) floats.
kernel void k_decode_attn_small(
    device const float* q [[buffer(0)]],
    device const float* k [[buffer(1)]],
    device const float* v [[buffer(2)]],
    device const float* r [[buffer(3)]],
    device const half* rel_proj [[buffer(4)]],
    device const float* sinks [[buffer(5)]],
    device const float* gate [[buffer(6)]],
    device float* out [[buffer(7)]],
    device const uint* meta [[buffer(8)]],
    uint h [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tptg [[threads_per_threadgroup]],
    threadgroup float* tg [[threadgroup(0)]]) {
    constexpr uint MAX_T = 1024u;
    constexpr uint N_SG = 8u;
    uint tlen = meta[0];
    if (h >= N_HEADS_D || tlen == 0u || tlen > MAX_T || tptg != N_SG * 32u) return;
    uint lane = tid & 31u;
    uint sg = tid >> 5u;
    threadgroup float* w = tg;                    // [MAX_T + 4]: logits, then weights
    threadgroup float* pmax = tg + (MAX_T + 4u);  // [8]
    threadgroup float* psum = pmax + 8u;          // [8]
    threadgroup float4* pacc = (threadgroup float4*)(psum + 8u); // [8 * 32]

    uint kv_ch = N_KV_D * DH_D;
    uint kv_off = (h / (N_HEADS_D / N_KV_D)) * DH_D;
    uint t = tlen - 1u;
    float scale = metal::rsqrt(float(DH_D));
    uint chunk = (tlen + N_SG - 1u) / N_SG;
    uint j0 = metal::min(sg * chunk, tlen);
    uint j1 = metal::min(j0 + chunk, tlen);

    // 1. Logits of this simdgroup's keys.
    float4 qv = ((device const float4*)(q + h * DH_D))[lane];
    bool has_rel = lane < D_REL_D;
    float rl = has_rel ? r[h * D_REL_D + lane] : 0.f;
    device const half* rel_row = rel_proj + (has_rel ? lane : 0u) * REL_EXTENT_D;
    float mx = -1e30f;
    for (uint j = j0; j < j1; ++j) {
        float4 kv = ((device const float4*)(k + j * kv_ch + kv_off))[lane];
        float p = scale * dot(qv, kv);
        if (has_rel) p += rl * float(rel_row[t - j]);
        float logit = simd_sum(p);
        if (lane == 0u) w[j] = logit;
        mx = metal::max(mx, logit);
    }
    if (tid == 0u) {
        float s = sinks[h];
        w[tlen] = s;
        mx = metal::max(mx, s);
    }
    if (lane == 0u) pmax[sg] = mx;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 2. Softmax weights (unnormalized) and their sum.
    float maxv = pmax[0];
    for (uint s = 1u; s < N_SG; ++s) maxv = metal::max(maxv, pmax[s]);
    float sum = 0.f;
    for (uint j = j0 + lane; j < j1; j += 32u) {
        float e = metal::exp(w[j] - maxv);
        w[j] = e;
        sum += e;
    }
    if (tid == 0u) sum += metal::exp(w[tlen] - maxv); // the sink only enters the sum
    sum = simd_sum(sum);
    if (lane == 0u) psum[sg] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 3. Weighted sum of V over this simdgroup's keys.
    float4 acc = float4(0.f);
    for (uint j = j0; j < j1; ++j) {
        float4 vv = ((device const float4*)(v + j * kv_ch + kv_off))[lane];
        acc += w[j] * vv;
    }
    pacc[sg * 32u + lane] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg == 0u) {
        float total = 0.f;
        float4 ctx = float4(0.f);
        for (uint s = 0u; s < N_SG; ++s) {
            total += psum[s];
            ctx += pacc[s * 32u + lane];
        }
        float4 g = ((device const float4*)(gate + h * DH_D))[lane];
        float4 sgm = 1.f / (1.f + metal::exp(-g));
        ((device float4*)(out + h * DH_D))[lane] = ctx * (1.f / total) * sgm;
    }
}

// Ring rows one simdgroup of k_decode_ring_part covers.
constant uint RING_CHUNK_D [[function_constant(10)]];

// Decode T=1 attention over a longer history, first half.  One simdgroup takes one chunk
// of ring rows and one KV head and serves the query heads that share it, so every K
// and V row is read once.  A lane owns four dims of a row.
//
// meta = [n, slot, ..]: n valid ring rows, the newest one at `slot`.  Row j is
// (slot - j) mod MAX_T tokens back, which is plain slot - j until the ring wraps.
// logit = q . k / sqrt(DH) + r . rel_proj[:, distance].
//
// It keeps a running max per query head and rescales as it goes, so one pass does
// it and nothing leaves the registers.  It writes, per query head, the chunk's max
// logit, the sum of exp(logit - max) and the V sum weighted the same way, in the
// layout k_decode_far_part uses.  k_decode_merge puts the chunks together.
// Launch threadgroups=(ceil(n / RING_CHUNK), N_KV), threads=32.
kernel void k_decode_ring_part(
    device const float* q [[buffer(0)]],
    device const float* k [[buffer(1)]],
    device const float* v [[buffer(2)]],
    device const float* r [[buffer(3)]],
    device const half* rel_proj [[buffer(4)]],
    device float* parts [[buffer(5)]],
    device const uint* meta [[buffer(6)]],
    uint3 tgp [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint MAX_T = 1024u;
    constexpr uint REP = 3u;
    uint n = meta[0];
    uint t = meta[1];
    uint c = tgp.x;
    uint g = tgp.y;
    uint j0 = c * RING_CHUNK_D;
    uint j1 = metal::min(j0 + RING_CHUNK_D, n);
    uint kv_ch = N_KV_D * DH_D;
    float scale = metal::rsqrt(float(DH_D));
    device const float4* kr = (device const float4*)(k + g * DH_D);
    device const float4* vr = (device const float4*)(v + g * DH_D);

    uint h0 = g * REP;
    float4 q0 = ((device const float4*)(q + h0 * DH_D))[lane];
    float4 q1 = ((device const float4*)(q + (h0 + 1u) * DH_D))[lane];
    float4 q2 = ((device const float4*)(q + (h0 + 2u) * DH_D))[lane];
    bool has_rel = lane < D_REL_D;
    uint rl_i = has_rel ? lane : 0u;
    float3 rl = has_rel
        ? float3(r[h0 * D_REL_D + rl_i], r[(h0 + 1u) * D_REL_D + rl_i], r[(h0 + 2u) * D_REL_D + rl_i])
        : float3(0.f);
    device const half* rel_row = rel_proj + rl_i * REL_EXTENT_D;

    float3 m = float3(-1e30f);
    float3 sum = float3(0.f);
    float4 a0 = float4(0.f);
    float4 a1 = float4(0.f);
    float4 a2 = float4(0.f);
    for (uint j = j0; j < j1; ++j) {
        float4 kv = kr[j * (kv_ch / 4u) + lane];
        uint dist = (t + MAX_T - j) & (MAX_T - 1u);
        float3 p = float3(dot(q0, kv), dot(q1, kv), dot(q2, kv)) * scale;
        if (has_rel) p += rl * float(rel_row[dist]);
        float3 l = float3(simd_sum(p.x), simd_sum(p.y), simd_sum(p.z));
        float3 m_new = metal::max(m, l);
        float3 f = metal::exp(m - m_new);
        float3 e = metal::exp(l - m_new);
        float4 vv = vr[j * (kv_ch / 4u) + lane];
        sum = sum * f + e;
        a0 = a0 * f.x + e.x * vv;
        a1 = a1 * f.y + e.y * vv;
        a2 = a2 * f.z + e.z * vv;
        m = m_new;
    }
    for (uint i = 0u; i < REP; ++i) {
        device float* part = parts + (c * N_HEADS_D + h0 + i) * FAR_STRIDE_D;
        if (lane == 0u) {
            part[0] = m[i];
            part[1] = sum[i];
        }
        ((device float4*)(part + 4u))[lane] = i == 0u ? a0 : (i == 1u ? a1 : a2);
    }
}

// Decode T=1 attention, second half: one simdgroup per query head folds the ring
// chunks, the archive chunks and the sink into one softmax, then applies the output
// gate.  meta = [n, slot, far_chunks, ..]; the ring has ceil(n / RING_CHUNK) chunks.
// Launch threadgroups=N_HEADS, threads=32.
kernel void k_decode_merge(
    device const float* ring_parts [[buffer(0)]],
    device const float* far_parts [[buffer(1)]],
    device const float* sinks [[buffer(2)]],
    device const float* gate [[buffer(3)]],
    device float* out [[buffer(4)]],
    device const uint* meta [[buffer(5)]],
    uint3 tgp [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    uint h = tgp.x;
    uint n_ring = (meta[0] + RING_CHUNK_D - 1u) / RING_CHUNK_D;
    uint n_far = meta[2];
    uint step = N_HEADS_D * FAR_STRIDE_D;
    device const float* ring_h = ring_parts + h * FAR_STRIDE_D;
    device const float* far_h = far_parts + h * FAR_STRIDE_D;
    float sink = sinks[h];
    float maxv = sink;
    for (uint c = 0u; c < n_ring; ++c) maxv = metal::max(maxv, ring_h[c * step]);
    for (uint c = 0u; c < n_far; ++c) maxv = metal::max(maxv, far_h[c * step]);
    float total = metal::exp(sink - maxv); // the sink only enters the sum
    float4 ctx = float4(0.f);
    for (uint c = 0u; c < n_ring; ++c) {
        device const float* part = ring_h + c * step;
        float f = metal::exp(part[0] - maxv);
        total += part[1] * f;
        ctx += ((device const float4*)(part + 4u))[lane] * f;
    }
    for (uint c = 0u; c < n_far; ++c) {
        device const float* part = far_h + c * step;
        float f = metal::exp(part[0] - maxv);
        total += part[1] * f;
        ctx += ((device const float4*)(part + 4u))[lane] * f;
    }
    float4 g = ((device const float4*)(gate + h * DH_D))[lane];
    float4 sgm = 1.f / (1.f + metal::exp(-g));
    ((device float4*)(out + h * DH_D))[lane] = ctx * (1.f / total) * sgm;
}

// Decode T=1 attention over the archive of a full-attention visit.  Archive rows
// are more than REL_EXTENT tokens back, so a logit is the plain scaled dot product.
//
// One threadgroup takes one chunk of FAR_CHUNK rows and one KV head, and serves the
// query heads that share it, so every K and V row is read once.  It leaves, per
// query head, the chunk's max logit, the sum of exp(logit - max) and the V sum
// weighted the same way.  k_decode_merge rescales and adds them.
// The archive comes in segments, one dispatch each: `k`, `v` are a segment, `rows`
// its row count, and `far` is bound at the segment's first chunk.
// Launch threadgroups=(chunks in the segment, N_KV), threads=256.
// threadgroup(0) >= 3 * (FAR_CHUNK + 16 + 8 * DH) floats.
kernel void k_decode_far_part(
    device const float* q [[buffer(0)]],
    device const arch_t* k [[buffer(1)]],
    device const arch_t* v [[buffer(2)]],
    device float* far [[buffer(3)]],
    device const uint* seg_rows [[buffer(4)]],
    uint3 tgp [[threadgroup_position_in_grid]],
    uint3 tid3 [[thread_position_in_threadgroup]],
    threadgroup float* tg [[threadgroup(0)]]) {
    constexpr uint N_SG = 8u;
    constexpr uint REP = 3u;
    uint tid = tid3.x;
    uint lane = tid & 31u;
    uint sg = tid >> 5u;
    uint c = tgp.x;
    uint g = tgp.y;
    uint rows = seg_rows[0];
    uint row0 = c * FAR_CHUNK_D;
    uint n = metal::min(FAR_CHUNK_D, rows - row0);
    threadgroup float* w = tg;                              // [REP][FAR_CHUNK]
    threadgroup float* pmax = tg + REP * FAR_CHUNK_D;       // [REP][8]
    threadgroup float* psum = pmax + REP * 8u;              // [REP][8]
    threadgroup float4* pacc = (threadgroup float4*)(psum + REP * 8u); // [REP][8 * 32]

    uint kv_ch = N_KV_D * DH_D;
    float scale = metal::rsqrt(float(DH_D));
    uint chunk = (n + N_SG - 1u) / N_SG;
    uint j0 = metal::min(sg * chunk, n);
    uint j1 = metal::min(j0 + chunk, n);
    device const arch_t* kr = k + row0 * kv_ch + g * DH_D;
    device const arch_t* vr = v + row0 * kv_ch + g * DH_D;

    float4 q0 = ((device const float4*)(q + (g * REP) * DH_D))[lane];
    float4 q1 = ((device const float4*)(q + (g * REP + 1u) * DH_D))[lane];
    float4 q2 = ((device const float4*)(q + (g * REP + 2u) * DH_D))[lane];
    float3 mx = float3(-1e30f);
    for (uint j = j0; j < j1; ++j) {
        float4 kv = float4(((device const arch4_t*)(kr + j * kv_ch))[lane]);
        float3 l = float3(simd_sum(dot(q0, kv)), simd_sum(dot(q1, kv)), simd_sum(dot(q2, kv))) * scale;
        if (lane == 0u) {
            w[j] = l.x;
            w[FAR_CHUNK_D + j] = l.y;
            w[2u * FAR_CHUNK_D + j] = l.z;
        }
        mx = metal::max(mx, l);
    }
    if (lane == 0u) {
        pmax[sg] = mx.x;
        pmax[8u + sg] = mx.y;
        pmax[16u + sg] = mx.z;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float3 maxv = float3(pmax[0], pmax[8], pmax[16]);
    for (uint s = 1u; s < N_SG; ++s) {
        maxv = metal::max(maxv, float3(pmax[s], pmax[8u + s], pmax[16u + s]));
    }
    float3 sum = float3(0.f);
    for (uint j = j0 + lane; j < j1; j += 32u) {
        float3 e = metal::exp(float3(w[j], w[FAR_CHUNK_D + j], w[2u * FAR_CHUNK_D + j]) - maxv);
        w[j] = e.x;
        w[FAR_CHUNK_D + j] = e.y;
        w[2u * FAR_CHUNK_D + j] = e.z;
        sum += e;
    }
    sum = float3(simd_sum(sum.x), simd_sum(sum.y), simd_sum(sum.z));
    if (lane == 0u) {
        psum[sg] = sum.x;
        psum[8u + sg] = sum.y;
        psum[16u + sg] = sum.z;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float4 a0 = float4(0.f);
    float4 a1 = float4(0.f);
    float4 a2 = float4(0.f);
    for (uint j = j0; j < j1; ++j) {
        float4 vv = float4(((device const arch4_t*)(vr + j * kv_ch))[lane]);
        a0 += w[j] * vv;
        a1 += w[FAR_CHUNK_D + j] * vv;
        a2 += w[2u * FAR_CHUNK_D + j] * vv;
    }
    pacc[sg * 32u + lane] = a0;
    pacc[256u + sg * 32u + lane] = a1;
    pacc[512u + sg * 32u + lane] = a2;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg == 0u) {
        for (uint i = 0u; i < REP; ++i) {
            float total = 0.f;
            float4 ctx = float4(0.f);
            for (uint s = 0u; s < N_SG; ++s) {
                total += psum[i * 8u + s];
                ctx += pacc[i * 256u + s * 32u + lane];
            }
            device float* part = far + (c * N_HEADS_D + g * REP + i) * FAR_STRIDE_D;
            if (lane == 0u) {
                part[0] = maxv[i];
                part[1] = total;
            }
            ((device float4*)(part + 4u))[lane] = ctx;
        }
    }
}

// Copies one K row and one V row from the ring to the archive.  The four
// buffers are bound at the row.  Launch threadgroups=1, threads=256.
kernel void k_archive_row(
    device const float* k_ring [[buffer(0)]],
    device const float* v_ring [[buffer(1)]],
    device arch_t* k_arch [[buffer(2)]],
    device arch_t* v_arch [[buffer(3)]],
    uint tid [[thread_position_in_threadgroup]]) {
    ((device arch4_t*)k_arch)[tid] = arch4_t(((device const float4*)k_ring)[tid]);
    ((device arch4_t*)v_arch)[tid] = arch4_t(((device const float4*)v_ring)[tid]);
}

// Multi-token GEMM on simdgroup matrices: Y[T, OUT] = X[T, IN] * W[OUT, IN]^T, W in
// f16. One simdgroup computes 8 output rows for 32 tokens:
//   C^T[n, m] += W[n, k] * X^T[k, m],  8 columns of k at a time.
// A lane holds two adjacent k of one output row (one 4-byte load of W); X is loaded
// transposed straight from the token rows.
// X and Y need room for T rounded up to a multiple of 32 rows; the extra rows are
// computed and ignored. OUT and IN are multiples of 8.
// Launch threadgroups=(OUT/8, ceil(T/32)), threads=32.
kernel void k_gemm_f16w_mma(
    device const float* x [[buffer(0)]],
    device const half* w [[buffer(1)]],
    device float* y [[buffer(2)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    uint n0 = tg.x * 8u;
    uint m0 = tg.y * 32u;
    uint fm = ((lane / 4u) & 4u) + ((lane / 2u) % 4u);
    uint fn = ((lane / 4u) & 2u) * 2u + (lane % 2u) * 2u;
    device const half2* wr = (device const half2*)(w + (n0 + fm) * IN_DIM + fn);
    simdgroup_float8x8 C0 = make_filled_simdgroup_matrix<float, 8, 8>(0.f);
    simdgroup_float8x8 C1 = make_filled_simdgroup_matrix<float, 8, 8>(0.f);
    simdgroup_float8x8 C2 = make_filled_simdgroup_matrix<float, 8, 8>(0.f);
    simdgroup_float8x8 C3 = make_filled_simdgroup_matrix<float, 8, 8>(0.f);
    simdgroup_float8x8 W = make_filled_simdgroup_matrix<float, 8, 8>(0.f);
    simdgroup_float8x8 Xt;
    device const float* xr = x + m0 * IN_DIM;
    for (uint k0 = 0u; k0 < IN_DIM; k0 += 8u) {
        float2 wv = float2(wr[k0 / 2u]);
        W.thread_elements()[0] = wv.x;
        W.thread_elements()[1] = wv.y;
        simdgroup_load(Xt, xr + k0, IN_DIM, ulong2(0, 0), true);
        simdgroup_multiply_accumulate(C0, W, Xt, C0);
        simdgroup_load(Xt, xr + 8u * IN_DIM + k0, IN_DIM, ulong2(0, 0), true);
        simdgroup_multiply_accumulate(C1, W, Xt, C1);
        simdgroup_load(Xt, xr + 16u * IN_DIM + k0, IN_DIM, ulong2(0, 0), true);
        simdgroup_multiply_accumulate(C2, W, Xt, C2);
        simdgroup_load(Xt, xr + 24u * IN_DIM + k0, IN_DIM, ulong2(0, 0), true);
        simdgroup_multiply_accumulate(C3, W, Xt, C3);
    }
    device float* yr = y + m0 * OUT_DIM + n0;
    simdgroup_store(C0, yr, OUT_DIM, ulong2(0, 0), true);
    simdgroup_store(C1, yr + 8u * OUT_DIM, OUT_DIM, ulong2(0, 0), true);
    simdgroup_store(C2, yr + 16u * OUT_DIM, OUT_DIM, ulong2(0, 0), true);
    simdgroup_store(C3, yr + 24u * OUT_DIM, OUT_DIM, ulong2(0, 0), true);
}

// The same product for a block of at most 8 tokens per tile: one accumulator, so a
// small block does a quarter of the arithmetic of a 32-token tile. X and Y need room
// for T rounded up to a multiple of 8 rows.
// Launch threadgroups=(OUT/8, ceil(T/8)), threads=32.
kernel void k_gemm_f16w_mma8(
    device const float* x [[buffer(0)]],
    device const half* w [[buffer(1)]],
    device float* y [[buffer(2)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    uint n0 = tg.x * 8u;
    uint m0 = tg.y * 8u;
    uint fm = ((lane / 4u) & 4u) + ((lane / 2u) % 4u);
    uint fn = ((lane / 4u) & 2u) * 2u + (lane % 2u) * 2u;
    device const half2* wr = (device const half2*)(w + (n0 + fm) * IN_DIM + fn);
    simdgroup_float8x8 C0 = make_filled_simdgroup_matrix<float, 8, 8>(0.f);
    simdgroup_float8x8 W = make_filled_simdgroup_matrix<float, 8, 8>(0.f);
    simdgroup_float8x8 Xt;
    device const float* xr = x + m0 * IN_DIM;
    for (uint k0 = 0u; k0 < IN_DIM; k0 += 8u) {
        float2 wv = float2(wr[k0 / 2u]);
        W.thread_elements()[0] = wv.x;
        W.thread_elements()[1] = wv.y;
        simdgroup_load(Xt, xr + k0, IN_DIM, ulong2(0, 0), true);
        simdgroup_multiply_accumulate(C0, W, Xt, C0);
    }
    simdgroup_store(C0, y + m0 * OUT_DIM + n0, OUT_DIM, ulong2(0, 0), true);
}

// ---- The drafter ----

// g[i] = silu(g[i]) * u[i].  meta = [n].
// Launch threadgroups=ceil(n/1024), threads=1024.
kernel void k_silu_mul(
    device float* g [[buffer(0)]],
    device const float* u [[buffer(1)]],
    device const uint* meta [[buffer(2)]],
    uint i [[thread_position_in_grid]]) {
    if (i >= meta[0]) return;
    float v = g[i];
    g[i] = v / (1.f + metal::exp(-v)) * u[i];
}

// The drafter's head for one position:
//   logit[r] = base[r] + bias[r] + fold[r, :] . latent,  capped like the target's.
// `fold` is f16, meta[1] wide (a multiple of 4).  One thread takes one row, and a
// threadgroup leaves [best capped logit, its row, sum of exp(capped), 0] in
// part[4 * tg].  The lowest row wins a tie.  meta = [rows, width].
// Launch threadgroups=ceil(rows/256), threads=256. threadgroup(0) >= 32 floats.
kernel void k_draft_pick(
    device const float* base [[buffer(0)]],
    device const float* bias [[buffer(1)]],
    device const half* fold [[buffer(2)]],
    device const float* latent [[buffer(3)]],
    device float* part [[buffer(4)]],
    device const uint* meta [[buffer(5)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    threadgroup float* tg [[threadgroup(0)]]) {
    constexpr float CAP = 30.f;
    uint rows = meta[0];
    uint width = meta[1];
    uint r = tgid * 256u + tid;
    bool live = r < rows;
    float c = -1e30f;
    if (live) {
        device const half4* w = (device const half4*)(fold + r * width);
        device const float4* x = (device const float4*)latent;
        float4 acc = float4(0.f);
        for (uint k = 0u; k < width / 4u; ++k) acc += float4(w[k]) * x[k];
        float l = base[r] + bias[r] + acc.x + acc.y + acc.z + acc.w;
        c = CAP * metal::precise::tanh(l / CAP);
    }
    float m = simd_max(c);
    float s = simd_sum(live ? metal::exp(c) : 0.f);
    uint best = simd_min((live && c == m) ? r : 0xffffffffu);
    uint sg = tid >> 5u;
    if ((tid & 31u) == 0u) {
        tg[sg * 4u] = m;
        tg[sg * 4u + 1u] = float(best);
        tg[sg * 4u + 2u] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        float bm = tg[0], br = tg[1], bs = tg[2];
        for (uint i = 1u; i < 8u; ++i) {
            if (tg[i * 4u] > bm) {
                bm = tg[i * 4u];
                br = tg[i * 4u + 1u];
            }
            bs += tg[i * 4u + 2u];
        }
        part[tgid * 4u] = bm;
        part[tgid * 4u + 1u] = br;
        part[tgid * 4u + 2u] = bs;
        part[tgid * 4u + 3u] = 0.f;
    }
}

// The best logit of each token of a block.  logits is [T][rows]; a threadgroup
// takes 256 rows of one token and leaves [best, its row] in
// part[(t * groups + tg) * 2].  The lowest row wins a tie.  meta = [rows, groups].
// Launch threadgroups=(groups, T) with groups = ceil(rows/256), threads=256.
// threadgroup(0) >= 16 floats.
kernel void k_pick_rows(
    device const float* logits [[buffer(0)]],
    device float* part [[buffer(1)]],
    device const uint* meta [[buffer(2)]],
    uint3 tgp [[threadgroup_position_in_grid]],
    uint3 tidp [[thread_position_in_threadgroup]],
    threadgroup float* tg [[threadgroup(0)]]) {
    uint rows = meta[0];
    uint groups = meta[1];
    uint tid = tidp.x;
    uint r = tgp.x * 256u + tid;
    bool live = r < rows;
    float v = live ? logits[tgp.y * rows + r] : -1e30f;
    float m = simd_max(v);
    uint best = simd_min((live && v == m) ? r : 0xffffffffu);
    uint sg = tid >> 5u;
    if ((tid & 31u) == 0u) {
        tg[sg * 2u] = m;
        tg[sg * 2u + 1u] = float(best);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        float bm = tg[0], br = tg[1];
        for (uint i = 1u; i < 8u; ++i) {
            if (tg[i * 2u] > bm) {
                bm = tg[i * 2u];
                br = tg[i * 2u + 1u];
            }
        }
        device float* out = part + (tgp.y * groups + tgp.x) * 2u;
        out[0] = bm;
        out[1] = br;
    }
}

// ---- Multi-token attention on simdgroup matrices (prompt) ----
//
// A chunk of queries attends to its keys one block at a time.  The first block is
// the visit's ring rows followed by the chunk's own rows, in f32.  A full-attention
// visit then adds blocks of archive rows.  A running softmax ties the blocks
// together: per (query, head) the state holds the max logit so far, the sum of
// exp(logit - max), the factor that rescales what earlier blocks accumulated, and
// 1 / sum.
//
// Buffers are token-major.  Queries are padded to nqp (a multiple of 32) and the
// keys of a block to nkp (a multiple of 8).
// meta = [nq, nqp, nk, nkp, position of query 0, position of key 0, flags, 0],
// flags bit 0: first block, bit 1: last block.
// Each block runs three stages: scores (GEMM), row softmax, context (GEMM).

// Stage 1: S[h][t][j] = q[t][h] . k[j][g(h)], unscaled.
// One simdgroup: head h, 8 keys, 32 queries.  C^T[j, t] += K[j, d] * Q^T[d, t].
// Launch threadgroups=(nkp/8, nqp/32, N_HEADS), threads=32.
kernel void k_attn_scores_mma(
    device const float* q [[buffer(0)]],
    device const float* k [[buffer(1)]],
    device float* s [[buffer(2)]],
    device const uint* meta [[buffer(3)]],
    uint3 tg [[threadgroup_position_in_grid]]) {
    uint nqp = meta[1];
    uint nkp = meta[3];
    uint j0 = tg.x * 8u;
    uint t0 = tg.y * 32u;
    uint h = tg.z;
    uint q_lo = meta[4] + t0;
    uint k_lo = meta[5] + j0;
    if (k_lo > q_lo + 31u) return; // every key in this tile is after every query
    if (SWA_D > 0u && q_lo >= k_lo + 7u + SWA_D) return; // or outside every window
    uint q_ch = N_HEADS_D * DH_D;
    uint kv_ch = N_KV_D * DH_D;
    device const float* kr = k + j0 * kv_ch + (h / (N_HEADS_D / N_KV_D)) * DH_D;
    device const float* qr = q + t0 * q_ch + h * DH_D;
    simdgroup_float8x8 C0 = make_filled_simdgroup_matrix<float, 8, 8>(0.f);
    simdgroup_float8x8 C1 = make_filled_simdgroup_matrix<float, 8, 8>(0.f);
    simdgroup_float8x8 C2 = make_filled_simdgroup_matrix<float, 8, 8>(0.f);
    simdgroup_float8x8 C3 = make_filled_simdgroup_matrix<float, 8, 8>(0.f);
    simdgroup_float8x8 W;
    simdgroup_float8x8 Xt;
    for (uint d0 = 0u; d0 < DH_D; d0 += 8u) {
        simdgroup_load(W, kr + d0, kv_ch);
        simdgroup_load(Xt, qr + d0, q_ch, ulong2(0, 0), true);
        simdgroup_multiply_accumulate(C0, W, Xt, C0);
        simdgroup_load(Xt, qr + 8u * q_ch + d0, q_ch, ulong2(0, 0), true);
        simdgroup_multiply_accumulate(C1, W, Xt, C1);
        simdgroup_load(Xt, qr + 16u * q_ch + d0, q_ch, ulong2(0, 0), true);
        simdgroup_multiply_accumulate(C2, W, Xt, C2);
        simdgroup_load(Xt, qr + 24u * q_ch + d0, q_ch, ulong2(0, 0), true);
        simdgroup_multiply_accumulate(C3, W, Xt, C3);
    }
    device float* sr = s + (h * nqp + t0) * nkp + j0;
    simdgroup_store(C0, sr, nkp, ulong2(0, 0), true);
    simdgroup_store(C1, sr + 8u * nkp, nkp, ulong2(0, 0), true);
    simdgroup_store(C2, sr + 16u * nkp, nkp, ulong2(0, 0), true);
    simdgroup_store(C3, sr + 24u * nkp, nkp, ulong2(0, 0), true);
}

// Stage 1 for a block of archive rows.  A lane holds two adjacent dims of one key.
kernel void k_attn_scores_far_mma(
    device const float* q [[buffer(0)]],
    device const arch_t* k [[buffer(1)]],
    device float* s [[buffer(2)]],
    device const uint* meta [[buffer(3)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    uint nqp = meta[1];
    uint nkp = meta[3];
    uint j0 = tg.x * 8u;
    uint t0 = tg.y * 32u;
    uint h = tg.z;
    uint q_ch = N_HEADS_D * DH_D;
    uint kv_ch = N_KV_D * DH_D;
    uint fm = ((lane / 4u) & 4u) + ((lane / 2u) % 4u);
    uint fn = ((lane / 4u) & 2u) * 2u + (lane % 2u) * 2u;
    device const arch_t* kr = k + (j0 + fm) * kv_ch + (h / (N_HEADS_D / N_KV_D)) * DH_D + fn;
    device const float* qr = q + t0 * q_ch + h * DH_D;
    simdgroup_float8x8 C0 = make_filled_simdgroup_matrix<float, 8, 8>(0.f);
    simdgroup_float8x8 C1 = make_filled_simdgroup_matrix<float, 8, 8>(0.f);
    simdgroup_float8x8 C2 = make_filled_simdgroup_matrix<float, 8, 8>(0.f);
    simdgroup_float8x8 C3 = make_filled_simdgroup_matrix<float, 8, 8>(0.f);
    simdgroup_float8x8 W = make_filled_simdgroup_matrix<float, 8, 8>(0.f);
    simdgroup_float8x8 Xt;
    for (uint d0 = 0u; d0 < DH_D; d0 += 8u) {
        W.thread_elements()[0] = float(kr[d0]);
        W.thread_elements()[1] = float(kr[d0 + 1u]);
        simdgroup_load(Xt, qr + d0, q_ch, ulong2(0, 0), true);
        simdgroup_multiply_accumulate(C0, W, Xt, C0);
        simdgroup_load(Xt, qr + 8u * q_ch + d0, q_ch, ulong2(0, 0), true);
        simdgroup_multiply_accumulate(C1, W, Xt, C1);
        simdgroup_load(Xt, qr + 16u * q_ch + d0, q_ch, ulong2(0, 0), true);
        simdgroup_multiply_accumulate(C2, W, Xt, C2);
        simdgroup_load(Xt, qr + 24u * q_ch + d0, q_ch, ulong2(0, 0), true);
        simdgroup_multiply_accumulate(C3, W, Xt, C3);
    }
    device float* sr = s + (h * nqp + t0) * nkp + j0;
    simdgroup_store(C0, sr, nkp, ulong2(0, 0), true);
    simdgroup_store(C1, sr + 8u * nkp, nkp, ulong2(0, 0), true);
    simdgroup_store(C2, sr + 16u * nkp, nkp, ulong2(0, 0), true);
    simdgroup_store(C3, sr + 24u * nkp, nkp, ulong2(0, 0), true);
}

// Stage 2, one simdgroup per (query t, head h): turns row S[h][t] into softmax
// weights relative to the running max and updates the running state.  A key after
// the query, or outside a sliding window, gets weight zero.
// logit = S / sqrt(DH) + r[t][h] . rel_proj[:, distance] while distance < REL_EXTENT.
// state = [N_HEADS][nqp][4]: max, sum, rescale factor, 1 / sum.  The sink seeds it.
// Launch threadgroups=(nq, N_HEADS), threads=32.
kernel void k_attn_softmax_rows(
    device float* s [[buffer(0)]],
    device const float* r [[buffer(1)]],
    device const half* rel_proj [[buffer(2)]],
    device const float* sinks [[buffer(3)]],
    device float* state [[buffer(4)]],
    device const uint* meta [[buffer(5)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    uint nqp = meta[1];
    uint nk = meta[2];
    uint nkp = meta[3];
    uint t = tg.x;
    uint h = tg.y;
    uint q_pos = meta[4] + t;
    uint k_pos0 = meta[5];
    bool first = (meta[6] & 1u) != 0u;
    device float* row = s + (h * nqp + t) * nkp;
    device float* st = state + (h * nqp + t) * 4u;
    device const float* rh = r + t * N_HEADS_D * D_REL_D + h * D_REL_D;
    float scale = metal::rsqrt(float(DH_D));
    float m_old = first ? sinks[t * N_HEADS_D + h] : st[0];
    float s_old = first ? 1.f : st[1];
    float mx = -1e30f;
    for (uint j = lane; j < nkp; j += 32u) {
        float l = -1e30f;
        uint k_pos = k_pos0 + j;
        if (j < nk && k_pos <= q_pos) {
            uint dist = q_pos - k_pos;
            if (dist < REL_EXTENT_D) {
                float rel = 0.f;
                for (uint c = 0u; c < D_REL_D; ++c) rel += rh[c] * float(rel_proj[c * REL_EXTENT_D + dist]);
                l = scale * row[j] + rel;
            } else if (SWA_D == 0u) {
                // Past the relative extent a key gets no relative logit.
                l = scale * row[j];
            }
        }
        row[j] = l;
        mx = metal::max(mx, l);
    }
    mx = metal::max(simd_max(mx), m_old);
    float sum = 0.f;
    for (uint j = lane; j < nkp; j += 32u) {
        float l = row[j];
        float e = l > -1e29f ? metal::exp(l - mx) : 0.f;
        row[j] = e;
        sum += e;
    }
    float f = metal::exp(m_old - mx);
    sum = simd_sum(sum) + s_old * f;
    if (lane == 0u) {
        st[0] = mx;
        st[1] = sum;
        st[2] = f;
        st[3] = 1.f / sum;
    }
}

// Tiles of keys a context simdgroup has to visit for queries t0 .. t0 + 31.
static inline uint2 ctx_key_range(device const uint* meta, uint t0) {
    uint nkp = meta[3];
    uint q_lo = meta[4] + t0;
    uint k_pos0 = meta[5];
    if (k_pos0 > q_lo + 31u) return uint2(0u, 0u);
    uint hi = metal::min(nkp, ((q_lo + 31u - k_pos0) / 8u) * 8u + 8u);
    uint lo = 0u;
    if (SWA_D > 0u && q_lo + 1u > k_pos0 + SWA_D) lo = ((q_lo + 1u - SWA_D - k_pos0) / 8u) * 8u;
    return uint2(metal::min(lo, hi), hi);
}

// What a context simdgroup does with its 8 dims x 32 queries of sum_j P * V: add what
// earlier blocks left (rescaled), and on the last block normalize and apply the gate.
static inline void ctx_store(
    simdgroup_float8x8 C0, simdgroup_float8x8 C1, simdgroup_float8x8 C2, simdgroup_float8x8 C3,
    device const float* state, device const float* gate, device float* ctx,
    device const uint* meta, uint d0, uint t0, uint h, uint lane) {
    uint nq = meta[0];
    uint nqp = meta[1];
    bool first = (meta[6] & 1u) != 0u;
    bool last = (meta[6] & 2u) != 0u;
    uint q_ch = N_HEADS_D * DH_D;
    uint d = d0 + ((lane / 4u) & 4u) + ((lane / 2u) % 4u);
    uint fn = ((lane / 4u) & 2u) * 2u + (lane % 2u) * 2u;
    for (uint i = 0u; i < 4u; ++i) {
        for (uint el = 0u; el < 2u; ++el) {
            uint t = t0 + 8u * i + fn + el;
            if (t >= nq) continue;
            float c = i == 0u ? C0.thread_elements()[el]
                    : i == 1u ? C1.thread_elements()[el]
                    : i == 2u ? C2.thread_elements()[el]
                              : C3.thread_elements()[el];
            device const float* st = state + (h * nqp + t) * 4u;
            device float* o = ctx + t * q_ch + h * DH_D + d;
            if (!first) c += o[0] * st[2];
            if (last) {
                float g = gate[t * q_ch + h * DH_D + d];
                c = c * st[3] / (1.f + metal::exp(-g));
            }
            o[0] = c;
        }
    }
}

// Stage 3: ctx[t][h][d] accumulates sum_j P[h][t][j] * v[j][g(h)][d] over the blocks;
// the last block turns it into sigmoid(gate) * ctx / sum.
// One simdgroup: head h, 8 dims, 32 queries.  C^T[d, t] += V^T[d, j] * P^T[j, t].
// Launch threadgroups=(DH/8, nqp/32, N_HEADS), threads=32.
kernel void k_attn_ctx_mma(
    device const float* s [[buffer(0)]],
    device const float* v [[buffer(1)]],
    device const float* state [[buffer(2)]],
    device const float* gate [[buffer(3)]],
    device float* ctx [[buffer(4)]],
    device const uint* meta [[buffer(5)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    uint nqp = meta[1];
    uint nkp = meta[3];
    uint d0 = tg.x * 8u;
    uint t0 = tg.y * 32u;
    uint h = tg.z;
    uint kv_ch = N_KV_D * DH_D;
    device const float* vr = v + (h / (N_HEADS_D / N_KV_D)) * DH_D + d0;
    device const float* pr = s + (h * nqp + t0) * nkp;
    simdgroup_float8x8 C0 = make_filled_simdgroup_matrix<float, 8, 8>(0.f);
    simdgroup_float8x8 C1 = make_filled_simdgroup_matrix<float, 8, 8>(0.f);
    simdgroup_float8x8 C2 = make_filled_simdgroup_matrix<float, 8, 8>(0.f);
    simdgroup_float8x8 C3 = make_filled_simdgroup_matrix<float, 8, 8>(0.f);
    simdgroup_float8x8 W;
    simdgroup_float8x8 Xt;
    uint2 jr = ctx_key_range(meta, t0);
    for (uint j0 = jr.x; j0 < jr.y; j0 += 8u) {
        simdgroup_load(W, vr + j0 * kv_ch, kv_ch, ulong2(0, 0), true);
        simdgroup_load(Xt, pr + j0, nkp, ulong2(0, 0), true);
        simdgroup_multiply_accumulate(C0, W, Xt, C0);
        simdgroup_load(Xt, pr + 8u * nkp + j0, nkp, ulong2(0, 0), true);
        simdgroup_multiply_accumulate(C1, W, Xt, C1);
        simdgroup_load(Xt, pr + 16u * nkp + j0, nkp, ulong2(0, 0), true);
        simdgroup_multiply_accumulate(C2, W, Xt, C2);
        simdgroup_load(Xt, pr + 24u * nkp + j0, nkp, ulong2(0, 0), true);
        simdgroup_multiply_accumulate(C3, W, Xt, C3);
    }
    ctx_store(C0, C1, C2, C3, state, gate, ctx, meta, d0, t0, h, lane);
}

// Stage 3 for a block of archive rows.  A lane holds one dim of two adjacent keys.
kernel void k_attn_ctx_far_mma(
    device const float* s [[buffer(0)]],
    device const arch_t* v [[buffer(1)]],
    device const float* state [[buffer(2)]],
    device const float* gate [[buffer(3)]],
    device float* ctx [[buffer(4)]],
    device const uint* meta [[buffer(5)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    uint nqp = meta[1];
    uint nkp = meta[3];
    uint d0 = tg.x * 8u;
    uint t0 = tg.y * 32u;
    uint h = tg.z;
    uint kv_ch = N_KV_D * DH_D;
    uint fm = ((lane / 4u) & 4u) + ((lane / 2u) % 4u);
    uint fn = ((lane / 4u) & 2u) * 2u + (lane % 2u) * 2u;
    device const arch_t* vr = v + fn * kv_ch + (h / (N_HEADS_D / N_KV_D)) * DH_D + d0 + fm;
    device const float* pr = s + (h * nqp + t0) * nkp;
    simdgroup_float8x8 C0 = make_filled_simdgroup_matrix<float, 8, 8>(0.f);
    simdgroup_float8x8 C1 = make_filled_simdgroup_matrix<float, 8, 8>(0.f);
    simdgroup_float8x8 C2 = make_filled_simdgroup_matrix<float, 8, 8>(0.f);
    simdgroup_float8x8 C3 = make_filled_simdgroup_matrix<float, 8, 8>(0.f);
    simdgroup_float8x8 W = make_filled_simdgroup_matrix<float, 8, 8>(0.f);
    simdgroup_float8x8 Xt;
    uint2 jr = ctx_key_range(meta, t0);
    for (uint j0 = jr.x; j0 < jr.y; j0 += 8u) {
        W.thread_elements()[0] = float(vr[j0 * kv_ch]);
        W.thread_elements()[1] = float(vr[(j0 + 1u) * kv_ch]);
        simdgroup_load(Xt, pr + j0, nkp, ulong2(0, 0), true);
        simdgroup_multiply_accumulate(C0, W, Xt, C0);
        simdgroup_load(Xt, pr + 8u * nkp + j0, nkp, ulong2(0, 0), true);
        simdgroup_multiply_accumulate(C1, W, Xt, C1);
        simdgroup_load(Xt, pr + 16u * nkp + j0, nkp, ulong2(0, 0), true);
        simdgroup_multiply_accumulate(C2, W, Xt, C2);
        simdgroup_load(Xt, pr + 24u * nkp + j0, nkp, ulong2(0, 0), true);
        simdgroup_multiply_accumulate(C3, W, Xt, C3);
    }
    ctx_store(C0, C1, C2, C3, state, gate, ctx, meta, d0, t0, h, lane);
}

// ---- Prompt glue: one simdgroup per token ----
//
// Hidden vectors are token-major, [Tp][H], and a lane owns every 32nd element.
// The block stack and the message bank are state-major: state s of token t sits at
// (s * Tp + t) * H.  Launch threads=32 everywhere.

constant float TOK_EPS = 1e-6f;

// Softmax mix over the block stack and up to two more vectors, as the residual
// queries do: key_i = rms(source_i), w = softmax(key_i . query / sqrt(H)),
// out = sum_i w_i key_i.  With norm set, out = rms(out) * norm_w.  With raw set the
// single extra vector is taken as it is (layer 0 has no attention query).
// meta = [n_states, n_extra, Tp, H, norm, raw].  Launch threadgroups=T.
kernel void k_tok_mix(
    device const float* stack [[buffer(0)]],
    device const float* e0 [[buffer(1)]],
    device const float* e1 [[buffer(2)]],
    device const float* query [[buffer(3)]],
    device const float* norm_w [[buffer(4)]],
    device float* out [[buffer(5)]],
    device const uint* meta [[buffer(6)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    uint ns = meta[0];
    uint n = ns + meta[1];
    uint Tp = meta[2];
    uint H = meta[3];
    uint t = tg.x;
    if (n == 0u || n > 8u) return;
    float wgt[8]; // weight on each raw source: softmax weight times 1 / rms
    if (meta[5] != 0u) {
        wgt[0] = 1.f;
    } else {
        float inv[8];
        float scale = metal::rsqrt(float(H));
        float mx = -1e30f;
        for (uint s = 0u; s < n; ++s) {
            device const float* src = s < ns ? stack + (s * Tp + t) * H : (s == ns ? e0 : e1) + t * H;
            float ss = 0.f;
            float d = 0.f;
            for (uint i = lane; i < H; i += 32u) {
                float v = src[i];
                ss += v * v;
                d += v * query[i];
            }
            inv[s] = metal::rsqrt(simd_sum(ss) / float(H) + TOK_EPS);
            wgt[s] = simd_sum(d) * inv[s] * scale;
            mx = metal::max(mx, wgt[s]);
        }
        float tot = 0.f;
        for (uint s = 0u; s < n; ++s) {
            wgt[s] = metal::exp(wgt[s] - mx);
            tot += wgt[s];
        }
        for (uint s = 0u; s < n; ++s) wgt[s] = wgt[s] / tot * inv[s];
    }
    device float* o = out + t * H;
    float ss = 0.f;
    for (uint i = lane; i < H; i += 32u) {
        float acc = 0.f;
        for (uint s = 0u; s < n; ++s) {
            device const float* src = s < ns ? stack + (s * Tp + t) * H : (s == ns ? e0 : e1) + t * H;
            acc += wgt[s] * src[i];
        }
        o[i] = acc;
        ss += acc * acc;
    }
    if (meta[4] != 0u) {
        float r = metal::rsqrt(simd_sum(ss) / float(H) + TOK_EPS);
        for (uint i = lane; i < H; i += 32u) o[i] = o[i] * norm_w[i] * r;
    }
}

// k_tok_mix for a block of a few tokens: the same mix with a token's work spread
// over eight simdgroups, so it takes an eighth of the time to come back.  Thread i of
// a threadgroup owns every 256th element; sums meet in threadgroup memory.
// Launch threadgroups=T, threads=256. threadgroup(0) >= 8 * 20 floats.
kernel void k_tok_mix_wide(
    device const float* stack [[buffer(0)]],
    device const float* e0 [[buffer(1)]],
    device const float* e1 [[buffer(2)]],
    device const float* query [[buffer(3)]],
    device const float* norm_w [[buffer(4)]],
    device float* out [[buffer(5)]],
    device const uint* meta [[buffer(6)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    threadgroup float* sums [[threadgroup(0)]]) {
    uint ns = meta[0];
    uint n = ns + meta[1];
    uint Tp = meta[2];
    uint H = meta[3];
    uint t = tg.x;
    uint sg = tid >> 5u;
    bool lead = (tid & 31u) == 0u;
    if (n == 0u || n > 8u) return;
    float wgt[8]; // weight on each raw source: softmax weight times 1 / rms
    if (meta[5] != 0u) {
        wgt[0] = 1.f;
    } else {
        // sums[sg * 20 + 2 s] = |source s|^2, [.. + 1] = source s . query, per simdgroup.
        for (uint s = 0u; s < n; ++s) {
            device const float* src = s < ns ? stack + (s * Tp + t) * H : (s == ns ? e0 : e1) + t * H;
            float ss = 0.f;
            float d = 0.f;
            for (uint i = tid; i < H; i += 256u) {
                float v = src[i];
                ss += v * v;
                d += v * query[i];
            }
            ss = simd_sum(ss);
            d = simd_sum(d);
            if (lead) {
                sums[sg * 20u + 2u * s] = ss;
                sums[sg * 20u + 2u * s + 1u] = d;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float inv[8];
        float scale = metal::rsqrt(float(H));
        float mx = -1e30f;
        for (uint s = 0u; s < n; ++s) {
            float ss = 0.f;
            float d = 0.f;
            for (uint g = 0u; g < 8u; ++g) {
                ss += sums[g * 20u + 2u * s];
                d += sums[g * 20u + 2u * s + 1u];
            }
            inv[s] = metal::rsqrt(ss / float(H) + TOK_EPS);
            wgt[s] = d * inv[s] * scale;
            mx = metal::max(mx, wgt[s]);
        }
        float tot = 0.f;
        for (uint s = 0u; s < n; ++s) {
            wgt[s] = metal::exp(wgt[s] - mx);
            tot += wgt[s];
        }
        for (uint s = 0u; s < n; ++s) wgt[s] = wgt[s] / tot * inv[s];
    }
    device float* o = out + t * H;
    float ss = 0.f;
    for (uint i = tid; i < H; i += 256u) {
        float acc = 0.f;
        for (uint s = 0u; s < n; ++s) {
            device const float* src = s < ns ? stack + (s * Tp + t) * H : (s == ns ? e0 : e1) + t * H;
            acc += wgt[s] * src[i];
        }
        o[i] = acc;
        ss += acc * acc;
    }
    if (meta[4] != 0u) {
        // The squared sums go in their own slots: the barrier above may not have run.
        ss = simd_sum(ss);
        if (lead) sums[sg * 20u + 16u] = ss;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float total = 0.f;
        for (uint g = 0u; g < 8u; ++g) total += sums[g * 20u + 16u];
        float r = metal::rsqrt(total / float(H) + TOK_EPS);
        for (uint i = tid; i < H; i += 256u) o[i] = o[i] * norm_w[i] * r;
    }
}

// Input of a message controller's down projection: [rms(x) | rms(msg)], [Tp][2H].
// msg is bound at the controller's slot.  meta = [msg stride, H].
// Launch threadgroups=T.
kernel void k_tok_ctrl_prep(
    device const float* x [[buffer(0)]],
    device const float* msg [[buffer(1)]],
    device float* out [[buffer(2)]],
    device const uint* meta [[buffer(3)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    uint H = meta[1];
    uint t = tg.x;
    device const float* xr = x + t * H;
    device const float* mr = msg + t * meta[0];
    float sx = 0.f;
    float sm = 0.f;
    for (uint i = lane; i < H; i += 32u) {
        sx += xr[i] * xr[i];
        sm += mr[i] * mr[i];
    }
    float ix = metal::rsqrt(simd_sum(sx) / float(H) + TOK_EPS);
    float im = metal::rsqrt(simd_sum(sm) / float(H) + TOK_EPS);
    device float* o = out + t * 2u * H;
    for (uint i = lane; i < H; i += 32u) {
        o[i] = xr[i] * ix;
        o[H + i] = mr[i] * im;
    }
}

// The rest of a message controller and the norm after it:
// y = (1 - tanh(f)) * x + tanh(g) * msg, out = rms(y) * norm_w.
// f and g are the forget and gain projections, [Tp][H].  meta = [msg stride, H].
// Launch threadgroups=T.
kernel void k_tok_ctrl_apply(
    device const float* x [[buffer(0)]],
    device const float* msg [[buffer(1)]],
    device const float* f [[buffer(2)]],
    device const float* g [[buffer(3)]],
    device const float* norm_w [[buffer(4)]],
    device float* out [[buffer(5)]],
    device const uint* meta [[buffer(6)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    uint H = meta[1];
    uint t = tg.x;
    device const float* xr = x + t * H;
    device const float* mr = msg + t * meta[0];
    device float* o = out + t * H;
    float ss = 0.f;
    for (uint i = lane; i < H; i += 32u) {
        float y = (1.f - metal::tanh(f[t * H + i])) * xr[i] + metal::tanh(g[t * H + i]) * mr[i];
        o[i] = y;
        ss += y * y;
    }
    float r = metal::rsqrt(simd_sum(ss) / float(H) + TOK_EPS);
    for (uint i = lane; i < H; i += 32u) o[i] = o[i] * norm_w[i] * r;
}

// K or V rows of a prompt chunk: the short conv over the projection, then the L2
// clip for K or the RMS norm for V.  One simdgroup per (token, KV head), a lane owns
// four dims.  The three tokens before the chunk come from `state` ([channel][3],
// oldest first).  A sliding layer has static taps [channel][4], newest last.  A
// full-attention layer has static taps [lag][channel], newest first, plus one
// dynamic tap per (token, head, lag) in `dyn`.
// `out` is bound at the chunk's first row of the key buffer.
// meta = [mode (0 K, 1 V), dynamic].  Launch threadgroups=(T, N_KV).
kernel void k_tok_kv_conv(
    device const float* x [[buffer(0)]],
    device const float* taps [[buffer(1)]],
    device const float* state [[buffer(2)]],
    device const float* dyn [[buffer(3)]],
    device float* out [[buffer(4)]],
    device const uint* meta [[buffer(5)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    constexpr float QK_CLIP = 22.627417f;
    uint t = tg.x;
    uint g = tg.y;
    uint kv_ch = N_KV_D * DH_D;
    uint c0 = g * DH_D + lane * 4u;
    float4 v[4]; // oldest first, v[3] is this token
    for (uint a = 0u; a < 4u; ++a) {
        if (t + a >= 3u) {
            v[a] = ((device const float4*)(x + (t + a - 3u) * kv_ch + c0))[0];
        } else {
            uint age = t + a;
            v[a] = float4(state[c0 * 3u + age], state[(c0 + 1u) * 3u + age],
                          state[(c0 + 2u) * 3u + age], state[(c0 + 3u) * 3u + age]);
        }
    }
    float4 acc = float4(0.f);
    float4 y;
    if (meta[1] != 0u) {
        device const float* d = dyn + t * (N_KV_D * 4u) + g * 4u;
        for (uint lag = 0u; lag < 4u; ++lag) {
            float4 w = ((device const float4*)(taps + lag * kv_ch + c0))[0] + d[lag];
            acc += v[3u - lag] * w;
        }
        y = (acc + v[3]) * metal::rsqrt(9.f);
    } else {
        for (uint a = 0u; a < 4u; ++a) {
            float4 w = float4(taps[c0 * 4u + a], taps[(c0 + 1u) * 4u + a],
                              taps[(c0 + 2u) * 4u + a], taps[(c0 + 3u) * 4u + a]);
            acc += w * v[a];
        }
        y = (acc + v[3]) * metal::rsqrt(5.f);
    }
    float ss = simd_sum(dot(y, y));
    if (meta[0] == 0u) {
        y *= metal::min(QK_CLIP / (metal::sqrt(ss) + 1e-6f), 1.f);
    } else {
        y *= metal::rsqrt(ss / float(DH_D) + TOK_EPS);
    }
    ((device float4*)(out + t * kv_ch + c0))[0] = y;
}

// In place, per (token, head): x *= scale, then clip the L2 norm to thr.
// x: [Tp][heads * n], scale: [heads * n].  meta = [heads, n].
// Launch threadgroups=(T, heads).
kernel void k_tok_l2_clip(
    device float* x [[buffer(0)]],
    device const float* scale [[buffer(1)]],
    device const uint* meta [[buffer(2)]],
    device const float* thr [[buffer(3)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    uint n = meta[1];
    uint h = tg.y;
    device float* xr = x + (tg.x * meta[0] + h) * n;
    device const float* sr = scale + h * n;
    float ss = 0.f;
    for (uint i = lane; i < n; i += 32u) {
        float v = xr[i] * sr[i];
        xr[i] = v;
        ss += v * v;
    }
    float f = metal::min(thr[0] / (metal::sqrt(simd_sum(ss)) + 1e-6f), 1.f);
    if (f < 1.f) {
        for (uint i = lane; i < n; i += 32u) xr[i] *= f;
    }
}

// out[t] = short_conv(x)[t] (+ add[t]) over hidden vectors.  The three tokens before
// the chunk come from `state` ([H][3], oldest first); taps are [H][4], newest last.
// out must not be x.  meta = [H, has_add].  Launch threadgroups=T.
kernel void k_tok_sconv_add(
    device const float* x [[buffer(0)]],
    device const float* taps [[buffer(1)]],
    device const float* state [[buffer(2)]],
    device const float* add [[buffer(3)]],
    device float* out [[buffer(4)]],
    device const uint* meta [[buffer(5)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    uint H = meta[0];
    uint t = tg.x;
    float scale = metal::rsqrt(5.f);
    for (uint c = lane; c < H; c += 32u) {
        float acc = 0.f;
        for (uint a = 0u; a < 4u; ++a) {
            float v = t + a >= 3u ? x[(t + a - 3u) * H + c] : state[c * 3u + t + a];
            acc += taps[c * 4u + a] * v;
        }
        float y = (acc + x[t * H + c]) * scale;
        out[t * H + c] = meta[1] != 0u ? y + add[t * H + c] : y;
    }
}

// dst[t] = rms(src[t]).  meta = [H].  Launch threadgroups=T.
kernel void k_tok_rms(
    device const float* src [[buffer(0)]],
    device float* dst [[buffer(1)]],
    device const uint* meta [[buffer(2)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    uint H = meta[0];
    device const float* s = src + tg.x * H;
    device float* d = dst + tg.x * H;
    float ss = 0.f;
    for (uint i = lane; i < H; i += 32u) ss += s[i] * s[i];
    float r = metal::rsqrt(simd_sum(ss) / float(H) + TOK_EPS);
    for (uint i = lane; i < H; i += 32u) d[i] = s[i] * r;
}

// The four messages of one core layer: for each query a softmax mix over the
// token's bank states.  bank: [n][Tp][H], queries: [4][H], out: [Tp][4][H].
// meta = [n, Tp, H].  Launch threadgroups=T.
kernel void k_tok_gather(
    device const float* bank [[buffer(0)]],
    device const float* queries [[buffer(1)]],
    device float* out [[buffer(2)]],
    device const uint* meta [[buffer(3)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    uint n = meta[0];
    uint Tp = meta[1];
    uint H = meta[2];
    uint t = tg.x;
    if (n == 0u || n > 8u) return;
    float scale = metal::rsqrt(float(H));
    for (uint q = 0u; q < 4u; ++q) {
        device const float* qr = queries + q * H;
        float w[8];
        float mx = -1e30f;
        for (uint s = 0u; s < n; ++s) {
            device const float* b = bank + (s * Tp + t) * H;
            float d = 0.f;
            for (uint i = lane; i < H; i += 32u) d += qr[i] * b[i];
            w[s] = simd_sum(d) * scale;
            mx = metal::max(mx, w[s]);
        }
        float tot = 0.f;
        for (uint s = 0u; s < n; ++s) {
            w[s] = metal::exp(w[s] - mx);
            tot += w[s];
        }
        device float* o = out + (t * 4u + q) * H;
        for (uint i = lane; i < H; i += 32u) {
            float acc = 0.f;
            for (uint s = 0u; s < n; ++s) acc += (w[s] / tot) * bank[(s * Tp + t) * H + i];
            o[i] = acc;
        }
    }
}
