// Gathered NVFP4 GEMV for MoE: one threadgroup per (token, route) output row
// is overkill; this kernel maps one thread per (token, route, out_row).
//
// packed_gate: [E, OUT, IN/2], scales_gate: [E, OUT, IN/16], amax: [E]
// indices: [T, R], x: [T, IN], y: [T, R, OUT]

#include <metal_stdlib>
using namespace metal;

constant uint TOKENS [[function_constant(0)]];
constant uint ROUTES [[function_constant(1)]];
constant uint OUT_DIM [[function_constant(2)]];
constant uint IN_DIM [[function_constant(3)]];
constant uint N_EXPERTS [[function_constant(4)]];

static inline float e2m1(uint nibble) {
    constexpr float LUT[16] = {
        0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f,
        -0.f, -0.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};
    return LUT[nibble & 15u];
}

static inline float e4m3_from_bits(uint b) {
    uint sign = (b >> 7) & 1u;
    int exp = int((b >> 3) & 0xFu);
    float mant = float(b & 7u);
    float mag = (exp != 0)
                    ? (1.f + mant / 8.f) * metal::exp2(float(exp - 7))
                    : (mant / 8.f) * metal::exp2(float(1 - 7));
    return sign ? -mag : mag;
}

// PER_ROUTE_X: 0 => x is [T, IN]; 1 => x is [T, R, IN]
constant uint PER_ROUTE_X [[function_constant(9)]];

kernel void k_nvfp4_gather_gemv(
    device const uchar* packed [[buffer(0)]],
    device const uchar* scales [[buffer(1)]],
    device const float* amax [[buffer(2)]],
    device const int* indices [[buffer(3)]],
    device const float* x [[buffer(4)]],
    device float* y [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]]) {
    uint t = gid.x;
    uint r = gid.y;
    uint o = gid.z;
    if (t >= TOKENS || r >= ROUTES || o >= OUT_DIM) return;
    int e = indices[t * ROUTES + r];
    if (e < 0 || uint(e) >= N_EXPERTS) {
        y[(t * ROUTES + r) * OUT_DIM + o] = 0.f;
        return;
    }
    constexpr float NVFP4_MAX = 2688.f;
    float scale_factor = amax[uint(e)] / NVFP4_MAX;
    uint packed_k = IN_DIM / 2u;
    uint scale_k = IN_DIM / 16u;
    uint expert_pack_stride = OUT_DIM * packed_k;
    uint expert_scale_stride = OUT_DIM * scale_k;
    device const uchar* prow =
        packed + uint(e) * expert_pack_stride + o * packed_k;
    device const uchar* srow =
        scales + uint(e) * expert_scale_stride + o * scale_k;
    device const float* xt = PER_ROUTE_X
                                 ? (x + (t * ROUTES + r) * IN_DIM)
                                 : (x + t * IN_DIM);
    float acc = 0.f;
    for (uint g = 0; g < scale_k; ++g) {
        float sc = e4m3_from_bits(uint(srow[g])) * scale_factor;
        uint base = g * 8u;
        for (uint j = 0; j < 8u; ++j) {
            uchar byte = prow[base + j];
            uint i = g * 16u + j * 2u;
            acc += e2m1(byte & 0xFu) * sc * xt[i];
            acc += e2m1(byte >> 4) * sc * xt[i + 1u];
        }
    }
    y[(t * ROUTES + r) * OUT_DIM + o] = acc;
}

// Dual NVFP4 gather + softcapped SwiGLU: one thread writes hid[t,r,o] directly.
constant float GATE_CAP [[function_constant(6)]];
constant float UP_CAP [[function_constant(7)]];

// Softcapped SwiGLU over gathered gate/up: h[t,r,d] = sc_swiglu(up, gate)
constant uint INTER [[function_constant(5)]];

kernel void k_softcapped_swiglu(
    device const float* up [[buffer(0)]],   // [T, R, INTER]
    device const float* gate [[buffer(1)]], // [T, R, INTER]
    device float* hidden [[buffer(2)]],     // [T, R, INTER]
    uint gid [[thread_position_in_grid]]) {
    uint n = TOKENS * ROUTES * INTER;
    if (gid >= n) return;
    float u = up[gid];
    float g = gate[gid];
    float sg = metal::tanh(g / GATE_CAP) * GATE_CAP;
    float silu = sg / (1.f + metal::exp(-g));
    float su = metal::tanh(u / UP_CAP) * UP_CAP;
    hidden[gid] = silu * su;
}

// Weighted sum of route outputs: out[t,d] = sum_r weights[t,r] * y[t,r,d]
constant uint HIDDEN [[function_constant(8)]];

kernel void k_moe_combine(
    device const float* routes_out [[buffer(0)]], // [T, R, H]
    device const float* weights [[buffer(1)]],    // [T, R]
    device float* out [[buffer(2)]],              // [T, H]
    uint2 gid [[thread_position_in_grid]]) {
    uint t = gid.x;
    uint d = gid.y;
    if (t >= TOKENS || d >= HIDDEN) return;
    float acc = 0.f;
    // Unroll common R=8 decode path.
    if (ROUTES == 8u) {
        uint base = t * 8u * HIDDEN + d;
        acc += weights[t * 8u + 0u] * routes_out[base + 0u * HIDDEN];
        acc += weights[t * 8u + 1u] * routes_out[base + 1u * HIDDEN];
        acc += weights[t * 8u + 2u] * routes_out[base + 2u * HIDDEN];
        acc += weights[t * 8u + 3u] * routes_out[base + 3u * HIDDEN];
        acc += weights[t * 8u + 4u] * routes_out[base + 4u * HIDDEN];
        acc += weights[t * 8u + 5u] * routes_out[base + 5u * HIDDEN];
        acc += weights[t * 8u + 6u] * routes_out[base + 6u * HIDDEN];
        acc += weights[t * 8u + 7u] * routes_out[base + 7u * HIDDEN];
    } else {
        for (uint r = 0; r < ROUTES; ++r) {
            acc += weights[t * ROUTES + r] * routes_out[(t * ROUTES + r) * HIDDEN + d];
        }
    }
    out[t * HIDDEN + d] = acc;
}

// --- MXFP8 gathered GEMV (layers 20-23) ---

static inline float mx_e4m3(uint b) {
    uint v = b & 127u;
    uint sign = ((b >> 7) & 1u) << 15;
    ushort u = ushort((v << 7) | ((((v + 1u) >> 7) << 14) | sign));
    return float(as_type<half>(u)) * 256.f;
}

static inline float mx_e8m0(uint b) {
    if (b == 0u) {
        return as_type<float>(uint(0x00400000u));
    }
    return as_type<float>(b << 23);
}

// packed: [E, OUT, IN], scales: [E, OUT, IN/32]
// indices: [T, R], x: [T, IN] or [T, R, IN], y: [T, R, OUT]
kernel void k_mxfp8_gather_gemv(
    device const uchar* packed [[buffer(0)]],
    device const uchar* scales [[buffer(1)]],
    device const int* indices [[buffer(2)]],
    device const float* x [[buffer(3)]],
    device float* y [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]]) {
    uint t = gid.x;
    uint r = gid.y;
    uint o = gid.z;
    if (t >= TOKENS || r >= ROUTES || o >= OUT_DIM) return;
    int e = indices[t * ROUTES + r];
    if (e < 0 || uint(e) >= N_EXPERTS) {
        y[(t * ROUTES + r) * OUT_DIM + o] = 0.f;
        return;
    }
    uint scale_k = IN_DIM / 32u;
    uint expert_pack_stride = OUT_DIM * IN_DIM;
    uint expert_scale_stride = OUT_DIM * scale_k;
    device const uchar* prow =
        packed + uint(e) * expert_pack_stride + o * IN_DIM;
    device const uchar* srow =
        scales + uint(e) * expert_scale_stride + o * scale_k;
    device const float* xt = PER_ROUTE_X
                                 ? (x + (t * ROUTES + r) * IN_DIM)
                                 : (x + t * IN_DIM);
    float acc = 0.f;
    for (uint g = 0; g < scale_k; ++g) {
        float sc = mx_e8m0(uint(srow[g]));
        uint base = g * 32u;
        for (uint j = 0; j < 32u; ++j) {
            acc += mx_e4m3(uint(prow[base + j])) * sc * xt[base + j];
        }
    }
    y[(t * ROUTES + r) * OUT_DIM + o] = acc;
}

// ============================================================================
// T=1 decode kernels. (Multi-token prefill uses the *_mma kernels further down.)
//
// Measured on M2 Max these are bound by decode arithmetic, not memory: a
// read-only probe over the same bytes runs nearly twice as fast. What paid for
// itself in isolation:
//   * E2M1 decode from a packed constant in a register instead of a LUT load;
//   * 4-byte weight loads instead of one load per byte;
//   * per-group scale applied once to the group sum (E4M3 from a 256-entry table);
//   * x read straight from device memory (no per-threadgroup copy and barrier);
//   * down-projection: one simdgroup owns DOWN_ROWS output rows of a route, so
//     threads are longer and there are DOWN_ROWS x fewer of them.
// ============================================================================

constant uint DOWN_ROWS [[function_constant(10)]];

// 2 * |e2m1| for magnitude codes 0..7, four bits each: {0,1,2,3,4,6,8,12}.
static inline float e2m1_x2(uint n) {
    uint mag = (0xC8643210u >> ((n & 7u) << 2u)) & 15u;
    return (n & 8u) ? -float(mag) : float(mag);
}

// 2 * sum_j e2m1(nibble j) * x[j] over one 16-weight group (lo, hi = 8 packed bytes).
static inline float nv_group_x2(uint lo, uint hi, device const float* xg) {
    float s = 0.f;
    for (uint j = 0u; j < 8u; ++j) {
        s += e2m1_x2(lo >> (4u * j)) * xg[j] + e2m1_x2(hi >> (4u * j)) * xg[8u + j];
    }
    return s;
}

// NVFP4 gate+up+SwiGLU. Launch threadgroups=(1, ROUTES, ceil(OUT/8)), threads=(32, 8).
// e4m3_lut: 256 floats, already multiplied by 0.5 (see nv_group_x2).
kernel void k_nvfp4_dual_swiglu_dec(
    device const uchar* gate_packed [[buffer(0)]],
    device const uchar* gate_scales [[buffer(1)]],
    device const uchar* up_packed [[buffer(2)]],
    device const uchar* up_scales [[buffer(3)]],
    device const float* amax [[buffer(4)]],
    device const int* indices [[buffer(5)]],
    device const float* x [[buffer(6)]],
    device float* hid [[buffer(7)]],
    device const float* e4m3_lut [[buffer(8)]],
    uint3 lid [[threadgroup_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 tptg [[threads_per_threadgroup]]) {
    uint r = lid.y;
    uint lane = tid.x;
    uint o = lid.z * tptg.y + tid.y;
    if (r >= ROUTES || o >= OUT_DIM) return;
    int ei = indices[r];
    uint out_i = r * OUT_DIM + o;
    if (ei < 0 || uint(ei) >= N_EXPERTS) {
        if (lane == 0u) hid[out_i] = 0.f;
        return;
    }
    uint e = uint(ei);
    float sf = amax[e] / 2688.f;
    uint packed_k = IN_DIM / 2u;
    uint scale_k = IN_DIM / 16u;
    device const uint* gp =
        (device const uint*)(gate_packed + e * OUT_DIM * packed_k + o * packed_k);
    device const uint* upp =
        (device const uint*)(up_packed + e * OUT_DIM * packed_k + o * packed_k);
    device const uchar* gs = gate_scales + e * OUT_DIM * scale_k + o * scale_k;
    device const uchar* us = up_scales + e * OUT_DIM * scale_k + o * scale_k;
    float ga = 0.f, ua = 0.f;
    for (uint g = lane; g < scale_k; g += 32u) {
        device const float* xg = x + 16u * g;
        ga += e4m3_lut[gs[g]] * nv_group_x2(gp[2u * g], gp[2u * g + 1u], xg);
        ua += e4m3_lut[us[g]] * nv_group_x2(upp[2u * g], upp[2u * g + 1u], xg);
    }
    float gacc = simd_sum(ga) * sf;
    float uacc = simd_sum(ua) * sf;
    if (lane == 0u) {
        float sg = metal::tanh(gacc / GATE_CAP) * GATE_CAP;
        float silu = sg / (1.f + metal::exp(-gacc));
        float su = metal::tanh(uacc / UP_CAP) * UP_CAP;
        hid[out_i] = silu * su;
    }
}

// NVFP4 down-projection + route-weighted combine.
// Launch threadgroups=(OUT/DOWN_ROWS), threads=(32, ROUTES).
// threadgroup(0) >= ROUTES * DOWN_ROWS floats. hid: [ROUTES, IN], y: [OUT].
kernel void k_nvfp4_down_combine_dec(
    device const uchar* packed [[buffer(0)]],
    device const uchar* scales [[buffer(1)]],
    device const float* amax [[buffer(2)]],
    device const int* indices [[buffer(3)]],
    device const float* hid [[buffer(4)]],
    device const float* weights [[buffer(5)]],
    device float* y [[buffer(6)]],
    device const float* e4m3_lut [[buffer(7)]],
    uint3 lid [[threadgroup_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]],
    threadgroup float* part [[threadgroup(0)]]) {
    uint lane = tid.x;
    uint r = tid.y;
    uint row0 = lid.x * DOWN_ROWS;
    int ei = indices[r];
    bool ok = ei >= 0 && uint(ei) < N_EXPERTS;
    uint e = ok ? uint(ei) : 0u;
    float sf = ok ? amax[e] / 2688.f * weights[r] : 0.f;
    uint packed_k = IN_DIM / 2u;
    uint scale_k = IN_DIM / 16u;
    device const uchar* p_e = packed + e * OUT_DIM * packed_k;
    device const uchar* s_e = scales + e * OUT_DIM * scale_k;
    device const float* xr = hid + r * IN_DIM;
    float acc[16];
    for (uint k = 0u; k < DOWN_ROWS; ++k) acc[k] = 0.f;
    for (uint g = lane; g < scale_k; g += 32u) {
        device const float* xg = xr + 16u * g;
        for (uint k = 0u; k < DOWN_ROWS; ++k) {
            uint o = row0 + k;
            device const uint* pp = (device const uint*)(p_e + o * packed_k);
            acc[k] += e4m3_lut[s_e[o * scale_k + g]] * nv_group_x2(pp[2u * g], pp[2u * g + 1u], xg);
        }
    }
    for (uint k = 0u; k < DOWN_ROWS; ++k) {
        float v = simd_sum(acc[k]) * sf;
        if (lane == 0u) part[r * DOWN_ROWS + k] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (r == 0u && lane < DOWN_ROWS && row0 + lane < OUT_DIM) {
        float v = 0.f;
        for (uint q = 0u; q < ROUTES; ++q) v += part[q * DOWN_ROWS + lane];
        y[row0 + lane] = v;
    }
}

// E4M3 byte -> value / 256 (same half-float bit trick as mx_e4m3, low 8 bits of b).
static inline float mx_dec(uint b) {
    uint v = b & 127u;
    uint bits = (v << 7u) | (((v + 1u) >> 7u) << 14u) | ((b & 128u) << 8u);
    return float(as_type<half>(ushort(bits)));
}

// sum_j e4m3(byte j) * x[j] / 256 over one 32-weight group (8 words).
static inline float mx_group(device const uint* wp, device const float* xg) {
    float s = 0.f;
    for (uint q = 0u; q < 8u; ++q) {
        uint w = wp[q];
        s += mx_dec(w) * xg[4u * q] + mx_dec(w >> 8u) * xg[4u * q + 1u]
           + mx_dec(w >> 16u) * xg[4u * q + 2u] + mx_dec(w >> 24u) * xg[4u * q + 3u];
    }
    return s;
}

// MXFP8 gate+up+SwiGLU. Launch as k_nvfp4_dual_swiglu_dec.
kernel void k_mxfp8_dual_swiglu_dec(
    device const uchar* gate_packed [[buffer(0)]],
    device const uchar* gate_scales [[buffer(1)]],
    device const uchar* up_packed [[buffer(2)]],
    device const uchar* up_scales [[buffer(3)]],
    device const int* indices [[buffer(4)]],
    device const float* x [[buffer(5)]],
    device float* hid [[buffer(6)]],
    uint3 lid [[threadgroup_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 tptg [[threads_per_threadgroup]]) {
    uint r = lid.y;
    uint lane = tid.x;
    uint o = lid.z * tptg.y + tid.y;
    if (r >= ROUTES || o >= OUT_DIM) return;
    int ei = indices[r];
    uint out_i = r * OUT_DIM + o;
    if (ei < 0 || uint(ei) >= N_EXPERTS) {
        if (lane == 0u) hid[out_i] = 0.f;
        return;
    }
    uint e = uint(ei);
    uint scale_k = IN_DIM / 32u;
    device const uint* gp = (device const uint*)(gate_packed + e * OUT_DIM * IN_DIM + o * IN_DIM);
    device const uint* upp = (device const uint*)(up_packed + e * OUT_DIM * IN_DIM + o * IN_DIM);
    device const uchar* gs = gate_scales + e * OUT_DIM * scale_k + o * scale_k;
    device const uchar* us = up_scales + e * OUT_DIM * scale_k + o * scale_k;
    float ga = 0.f, ua = 0.f;
    for (uint g = lane; g < scale_k; g += 32u) {
        device const float* xg = x + 32u * g;
        ga += mx_e8m0(uint(gs[g])) * mx_group(gp + 8u * g, xg);
        ua += mx_e8m0(uint(us[g])) * mx_group(upp + 8u * g, xg);
    }
    float gacc = simd_sum(ga) * 256.f;
    float uacc = simd_sum(ua) * 256.f;
    if (lane == 0u) {
        float sg = metal::tanh(gacc / GATE_CAP) * GATE_CAP;
        float silu = sg / (1.f + metal::exp(-gacc));
        float su = metal::tanh(uacc / UP_CAP) * UP_CAP;
        hid[out_i] = silu * su;
    }
}

// MXFP8 down-projection + combine. Launch as k_nvfp4_down_combine_dec.
kernel void k_mxfp8_down_combine_dec(
    device const uchar* packed [[buffer(0)]],
    device const uchar* scales [[buffer(1)]],
    device const int* indices [[buffer(2)]],
    device const float* hid [[buffer(3)]],
    device const float* weights [[buffer(4)]],
    device float* y [[buffer(5)]],
    uint3 lid [[threadgroup_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]],
    threadgroup float* part [[threadgroup(0)]]) {
    uint lane = tid.x;
    uint r = tid.y;
    uint row0 = lid.x * DOWN_ROWS;
    int ei = indices[r];
    bool ok = ei >= 0 && uint(ei) < N_EXPERTS;
    uint e = ok ? uint(ei) : 0u;
    float sf = ok ? 256.f * weights[r] : 0.f;
    uint scale_k = IN_DIM / 32u;
    device const uchar* p_e = packed + e * OUT_DIM * IN_DIM;
    device const uchar* s_e = scales + e * OUT_DIM * scale_k;
    device const float* xr = hid + r * IN_DIM;
    float acc[16];
    for (uint k = 0u; k < DOWN_ROWS; ++k) acc[k] = 0.f;
    for (uint g = lane; g < scale_k; g += 32u) {
        device const float* xg = xr + 32u * g;
        for (uint k = 0u; k < DOWN_ROWS; ++k) {
            uint o = row0 + k;
            device const uint* pp = (device const uint*)(p_e + o * IN_DIM);
            acc[k] += mx_e8m0(uint(s_e[o * scale_k + g])) * mx_group(pp + 8u * g, xg);
        }
    }
    for (uint k = 0u; k < DOWN_ROWS; ++k) {
        float v = simd_sum(acc[k]) * sf;
        if (lane == 0u) part[r * DOWN_ROWS + k] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (r == 0u && lane < DOWN_ROWS && row0 + lane < OUT_DIM) {
        float v = 0.f;
        for (uint q = 0u; q < ROUTES; ++q) v += part[q * DOWN_ROWS + lane];
        y[row0 + lane] = v;
    }
}

// ============================================================================
// Multi-token (prefill) expert kernels on simdgroup matrices.
//
// The (token, route) pairs of a visit are grouped by expert on the CPU into chunks
// of up to 32 pairs. One simdgroup computes 8 output rows of one expert for MM_TM
// blocks of 8 pairs:  C^T[n, m] += W[n, k] * X^T[k, m],  8 columns of k at a time.
// A lane of an 8x8 simdgroup matrix holds two adjacent columns of one row, which
// for W is two adjacent k of one output row: the two nibbles of one NVFP4 byte, or
// two MXFP8 bytes. So the weights are decoded straight into the matrix, once per
// chunk instead of once per token.
//
// MM_TM is a compile-time constant. With a runtime block count the accumulators
// fall out of registers and the kernel is about five times slower, so the chunk
// table is sectioned by block count and each section gets its own pipeline.
//
// chunks: [n_chunks][33] ints = expert, then 32 pair ids (token * ROUTES + route),
// -1 for an empty slot. Launch threadgroups=(n_chunks, OUT/8), threads=32.
// ============================================================================

constant uint MM_TM [[function_constant(11)]];

// Coordinates of the two elements a lane holds: row sg_row, columns sg_col, sg_col + 1.
static inline uint sg_row(uint lane) { return ((lane / 4u) & 4u) + ((lane / 2u) % 4u); }
static inline uint sg_col(uint lane) { return ((lane / 4u) & 2u) * 2u + (lane % 2u) * 2u; }

// NVFP4 gate + up + SwiGLU. x: [T, IN], hid: [T * ROUTES, OUT].
kernel void k_nvfp4_dual_mma(
    device const uchar* gate_packed [[buffer(0)]],
    device const uchar* gate_scales [[buffer(1)]],
    device const uchar* up_packed [[buffer(2)]],
    device const uchar* up_scales [[buffer(3)]],
    device const float* amax [[buffer(4)]],
    device const int* chunks [[buffer(5)]],
    device const float* x [[buffer(6)]],
    device float* hid [[buffer(7)]],
    device const float* e4m3_lut [[buffer(8)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    device const int* c = chunks + tg.x * 33u;
    if (c[0] < 0) return; // a chunk slot nobody owns
    uint e = uint(c[0]);
    uint fm = sg_row(lane);
    uint fn = sg_col(lane);
    uint packed_k = IN_DIM / 2u;
    uint scale_k = IN_DIM / 16u;
    uint row = tg.y * 8u + fm;
    device const uchar* gp = gate_packed + (e * OUT_DIM + row) * packed_k + fn / 2u;
    device const uchar* upp = up_packed + (e * OUT_DIM + row) * packed_k + fn / 2u;
    device const uchar* gs = gate_scales + (e * OUT_DIM + row) * scale_k;
    device const uchar* us = up_scales + (e * OUT_DIM + row) * scale_k;
    int pid0[4];
    int pid1[4];
    device const float* x0[4];
    device const float* x1[4];
    simdgroup_float8x8 Cg[4];
    simdgroup_float8x8 Cu[4];
    for (uint i = 0u; i < MM_TM; ++i) {
        pid0[i] = c[1u + 8u * i + fn];
        pid1[i] = c[1u + 8u * i + fn + 1u];
        x0[i] = x + (pid0[i] < 0 ? 0u : uint(pid0[i]) / ROUTES) * IN_DIM + fm;
        x1[i] = x + (pid1[i] < 0 ? 0u : uint(pid1[i]) / ROUTES) * IN_DIM + fm;
        Cg[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.f);
        Cu[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.f);
    }
    simdgroup_float8x8 Wg = make_filled_simdgroup_matrix<float, 8, 8>(0.f);
    simdgroup_float8x8 Wu = make_filled_simdgroup_matrix<float, 8, 8>(0.f);
    simdgroup_float8x8 Xt = make_filled_simdgroup_matrix<float, 8, 8>(0.f);
    for (uint g = 0u; g < scale_k; ++g) {
        float gsc = e4m3_lut[gs[g]];
        float usc = e4m3_lut[us[g]];
        for (uint hb = 0u; hb < 2u; ++hb) {
            uint k0 = g * 16u + hb * 8u;
            uint gb = gp[k0 / 2u];
            uint ub = upp[k0 / 2u];
            Wg.thread_elements()[0] = e2m1_x2(gb) * gsc;
            Wg.thread_elements()[1] = e2m1_x2(gb >> 4u) * gsc;
            Wu.thread_elements()[0] = e2m1_x2(ub) * usc;
            Wu.thread_elements()[1] = e2m1_x2(ub >> 4u) * usc;
            for (uint i = 0u; i < MM_TM; ++i) {
                Xt.thread_elements()[0] = x0[i][k0];
                Xt.thread_elements()[1] = x1[i][k0];
                simdgroup_multiply_accumulate(Cg[i], Wg, Xt, Cg[i]);
                simdgroup_multiply_accumulate(Cu[i], Wu, Xt, Cu[i]);
            }
        }
    }
    float sf = amax[e] / 2688.f;
    for (uint i = 0u; i < MM_TM; ++i) {
        for (uint el = 0u; el < 2u; ++el) {
            int pid = el == 0u ? pid0[i] : pid1[i];
            if (pid < 0) continue;
            float gacc = Cg[i].thread_elements()[el] * sf;
            float uacc = Cu[i].thread_elements()[el] * sf;
            float sg = metal::tanh(gacc / GATE_CAP) * GATE_CAP;
            float silu = sg / (1.f + metal::exp(-gacc));
            float su = metal::tanh(uacc / UP_CAP) * UP_CAP;
            hid[uint(pid) * OUT_DIM + row] = silu * su;
        }
    }
}

// NVFP4 down-projection, scaled by the pair's route weight.
// hid: [T * ROUTES, IN], weights: [T * ROUTES], dn: [T * ROUTES, OUT].
kernel void k_nvfp4_down_mma(
    device const uchar* packed [[buffer(0)]],
    device const uchar* scales [[buffer(1)]],
    device const float* amax [[buffer(2)]],
    device const int* chunks [[buffer(3)]],
    device const float* hid [[buffer(4)]],
    device const float* weights [[buffer(5)]],
    device float* dn [[buffer(6)]],
    device const float* e4m3_lut [[buffer(7)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    device const int* c = chunks + tg.x * 33u;
    if (c[0] < 0) return; // a chunk slot nobody owns
    uint e = uint(c[0]);
    uint fm = sg_row(lane);
    uint fn = sg_col(lane);
    uint packed_k = IN_DIM / 2u;
    uint scale_k = IN_DIM / 16u;
    uint row = tg.y * 8u + fm;
    device const uchar* wp = packed + (e * OUT_DIM + row) * packed_k + fn / 2u;
    device const uchar* ws = scales + (e * OUT_DIM + row) * scale_k;
    int pid0[4];
    int pid1[4];
    device const float* x0[4];
    device const float* x1[4];
    simdgroup_float8x8 C[4];
    for (uint i = 0u; i < MM_TM; ++i) {
        pid0[i] = c[1u + 8u * i + fn];
        pid1[i] = c[1u + 8u * i + fn + 1u];
        x0[i] = hid + (pid0[i] < 0 ? 0u : uint(pid0[i])) * IN_DIM + fm;
        x1[i] = hid + (pid1[i] < 0 ? 0u : uint(pid1[i])) * IN_DIM + fm;
        C[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.f);
    }
    simdgroup_float8x8 W = make_filled_simdgroup_matrix<float, 8, 8>(0.f);
    simdgroup_float8x8 Xt = make_filled_simdgroup_matrix<float, 8, 8>(0.f);
    for (uint g = 0u; g < scale_k; ++g) {
        float sc = e4m3_lut[ws[g]];
        for (uint hb = 0u; hb < 2u; ++hb) {
            uint k0 = g * 16u + hb * 8u;
            uint b = wp[k0 / 2u];
            W.thread_elements()[0] = e2m1_x2(b) * sc;
            W.thread_elements()[1] = e2m1_x2(b >> 4u) * sc;
            for (uint i = 0u; i < MM_TM; ++i) {
                Xt.thread_elements()[0] = x0[i][k0];
                Xt.thread_elements()[1] = x1[i][k0];
                simdgroup_multiply_accumulate(C[i], W, Xt, C[i]);
            }
        }
    }
    float sf = amax[e] / 2688.f;
    for (uint i = 0u; i < MM_TM; ++i) {
        for (uint el = 0u; el < 2u; ++el) {
            int pid = el == 0u ? pid0[i] : pid1[i];
            if (pid < 0) continue;
            dn[uint(pid) * OUT_DIM + row] = C[i].thread_elements()[el] * sf * weights[pid];
        }
    }
}

// MXFP8 gate + up + SwiGLU (layers 20-23). A lane's two elements are two bytes.
kernel void k_mxfp8_dual_mma(
    device const uchar* gate_packed [[buffer(0)]],
    device const uchar* gate_scales [[buffer(1)]],
    device const uchar* up_packed [[buffer(2)]],
    device const uchar* up_scales [[buffer(3)]],
    device const int* chunks [[buffer(4)]],
    device const float* x [[buffer(5)]],
    device float* hid [[buffer(6)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    device const int* c = chunks + tg.x * 33u;
    if (c[0] < 0) return; // a chunk slot nobody owns
    uint e = uint(c[0]);
    uint fm = sg_row(lane);
    uint fn = sg_col(lane);
    uint scale_k = IN_DIM / 32u;
    uint row = tg.y * 8u + fm;
    device const ushort* gp = (device const ushort*)(gate_packed + (e * OUT_DIM + row) * IN_DIM + fn);
    device const ushort* upp = (device const ushort*)(up_packed + (e * OUT_DIM + row) * IN_DIM + fn);
    device const uchar* gs = gate_scales + (e * OUT_DIM + row) * scale_k;
    device const uchar* us = up_scales + (e * OUT_DIM + row) * scale_k;
    int pid0[4];
    int pid1[4];
    device const float* x0[4];
    device const float* x1[4];
    simdgroup_float8x8 Cg[4];
    simdgroup_float8x8 Cu[4];
    for (uint i = 0u; i < MM_TM; ++i) {
        pid0[i] = c[1u + 8u * i + fn];
        pid1[i] = c[1u + 8u * i + fn + 1u];
        x0[i] = x + (pid0[i] < 0 ? 0u : uint(pid0[i]) / ROUTES) * IN_DIM + fm;
        x1[i] = x + (pid1[i] < 0 ? 0u : uint(pid1[i]) / ROUTES) * IN_DIM + fm;
        Cg[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.f);
        Cu[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.f);
    }
    simdgroup_float8x8 Wg = make_filled_simdgroup_matrix<float, 8, 8>(0.f);
    simdgroup_float8x8 Wu = make_filled_simdgroup_matrix<float, 8, 8>(0.f);
    simdgroup_float8x8 Xt = make_filled_simdgroup_matrix<float, 8, 8>(0.f);
    for (uint g = 0u; g < scale_k; ++g) {
        float gsc = mx_e8m0(uint(gs[g]));
        float usc = mx_e8m0(uint(us[g]));
        for (uint hb = 0u; hb < 4u; ++hb) {
            uint k0 = g * 32u + hb * 8u;
            uint gw = gp[k0 / 2u];
            uint uw = upp[k0 / 2u];
            Wg.thread_elements()[0] = mx_dec(gw) * gsc;
            Wg.thread_elements()[1] = mx_dec(gw >> 8u) * gsc;
            Wu.thread_elements()[0] = mx_dec(uw) * usc;
            Wu.thread_elements()[1] = mx_dec(uw >> 8u) * usc;
            for (uint i = 0u; i < MM_TM; ++i) {
                Xt.thread_elements()[0] = x0[i][k0];
                Xt.thread_elements()[1] = x1[i][k0];
                simdgroup_multiply_accumulate(Cg[i], Wg, Xt, Cg[i]);
                simdgroup_multiply_accumulate(Cu[i], Wu, Xt, Cu[i]);
            }
        }
    }
    for (uint i = 0u; i < MM_TM; ++i) {
        for (uint el = 0u; el < 2u; ++el) {
            int pid = el == 0u ? pid0[i] : pid1[i];
            if (pid < 0) continue;
            float gacc = Cg[i].thread_elements()[el] * 256.f;
            float uacc = Cu[i].thread_elements()[el] * 256.f;
            float sg = metal::tanh(gacc / GATE_CAP) * GATE_CAP;
            float silu = sg / (1.f + metal::exp(-gacc));
            float su = metal::tanh(uacc / UP_CAP) * UP_CAP;
            hid[uint(pid) * OUT_DIM + row] = silu * su;
        }
    }
}

// MXFP8 down-projection, scaled by the pair's route weight.
kernel void k_mxfp8_down_mma(
    device const uchar* packed [[buffer(0)]],
    device const uchar* scales [[buffer(1)]],
    device const int* chunks [[buffer(2)]],
    device const float* hid [[buffer(3)]],
    device const float* weights [[buffer(4)]],
    device float* dn [[buffer(5)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    device const int* c = chunks + tg.x * 33u;
    if (c[0] < 0) return; // a chunk slot nobody owns
    uint e = uint(c[0]);
    uint fm = sg_row(lane);
    uint fn = sg_col(lane);
    uint scale_k = IN_DIM / 32u;
    uint row = tg.y * 8u + fm;
    device const ushort* wp = (device const ushort*)(packed + (e * OUT_DIM + row) * IN_DIM + fn);
    device const uchar* ws = scales + (e * OUT_DIM + row) * scale_k;
    int pid0[4];
    int pid1[4];
    device const float* x0[4];
    device const float* x1[4];
    simdgroup_float8x8 C[4];
    for (uint i = 0u; i < MM_TM; ++i) {
        pid0[i] = c[1u + 8u * i + fn];
        pid1[i] = c[1u + 8u * i + fn + 1u];
        x0[i] = hid + (pid0[i] < 0 ? 0u : uint(pid0[i])) * IN_DIM + fm;
        x1[i] = hid + (pid1[i] < 0 ? 0u : uint(pid1[i])) * IN_DIM + fm;
        C[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.f);
    }
    simdgroup_float8x8 W = make_filled_simdgroup_matrix<float, 8, 8>(0.f);
    simdgroup_float8x8 Xt = make_filled_simdgroup_matrix<float, 8, 8>(0.f);
    for (uint g = 0u; g < scale_k; ++g) {
        float sc = mx_e8m0(uint(ws[g]));
        for (uint hb = 0u; hb < 4u; ++hb) {
            uint k0 = g * 32u + hb * 8u;
            uint w = wp[k0 / 2u];
            W.thread_elements()[0] = mx_dec(w) * sc;
            W.thread_elements()[1] = mx_dec(w >> 8u) * sc;
            for (uint i = 0u; i < MM_TM; ++i) {
                Xt.thread_elements()[0] = x0[i][k0];
                Xt.thread_elements()[1] = x1[i][k0];
                simdgroup_multiply_accumulate(C[i], W, Xt, C[i]);
            }
        }
    }
    for (uint i = 0u; i < MM_TM; ++i) {
        for (uint el = 0u; el < 2u; ++el) {
            int pid = el == 0u ? pid0[i] : pid1[i];
            if (pid < 0) continue;
            dn[uint(pid) * OUT_DIM + row] = C[i].thread_elements()[el] * 256.f * weights[pid];
        }
    }
}

// Groups the (token, route) pairs of a short block by expert, on the GPU, into the
// chunk table the `*_mma` kernels read.  Chunk p belongs to pair p: it lists every
// pair with that pair's expert if no earlier pair has the same expert, and is empty
// (expert -1) otherwise.  A token names an expert once, so a chunk holds at most as
// many pairs as the block has tokens.  meta = [pairs].
// Launch threadgroups=1, threads=64 (pairs <= 64).
kernel void k_block_chunks(
    device const int* indices [[buffer(0)]],
    device int* chunks [[buffer(1)]],
    device const uint* meta [[buffer(2)]],
    uint p [[thread_position_in_threadgroup]]) {
    uint n = meta[0];
    if (p >= n) return;
    device int* c = chunks + p * 33u;
    int e = indices[p];
    bool owner = true;
    for (uint q = 0u; q < p; ++q) owner = owner && indices[q] != e;
    c[0] = owner ? e : -1;
    uint k = 0u;
    for (uint q = 0u; q < n; ++q) {
        if (owner && indices[q] == e && k < 32u) {
            c[1u + k] = int(q);
            k += 1u;
        }
    }
    for (; k < 32u; ++k) c[1u + k] = -1;
}

// out[t, o] = sum over the token's routes of dn[t * ROUTES + r, o].
// Launch threadgroups=(OUT/256, T), threads=256.
kernel void k_moe_sum_routes(
    device const float* dn [[buffer(0)]],
    device float* out [[buffer(1)]],
    uint3 gid [[thread_position_in_grid]]) {
    uint o = gid.x;
    uint t = gid.y;
    if (o >= OUT_DIM) return;
    float acc = 0.f;
    for (uint r = 0u; r < ROUTES; ++r) acc += dn[(t * ROUTES + r) * OUT_DIM + o];
    out[t * OUT_DIM + o] = acc;
}
