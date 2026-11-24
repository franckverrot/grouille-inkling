// MXFP8 expert GEMV: y[out] = W_mxfp8[out, in] @ x[in]
// Packing: one E4M3 byte per weight, E8M0 scales every 32 elements (MLX mode=mxfp8).

#include <metal_stdlib>
using namespace metal;

constant uint OUT_DIM [[function_constant(0)]];
constant uint IN_DIM [[function_constant(1)]]; // must be divisible by 32

// Matches MLX fp8_e4m3 → half trick × 256.
static inline float e4m3_from_bits(uint b) {
    uint v = b & 127u;
    uint sign = ((b >> 7) & 1u) << 15;
    ushort u = ushort((v << 7) | ((((v + 1u) >> 7) << 14) | sign));
    return float(as_type<half>(u)) * 256.f;
}

// Matches MLX fp8_e8m0.
static inline float e8m0_from_bits(uint b) {
    if (b == 0u) {
        return as_type<float>(uint(0x00400000u)); // 2^-127
    }
    return as_type<float>(b << 23);
}

kernel void k_mxfp8_gemv(
    device const uchar* packed [[buffer(0)]], // [OUT, IN]
    device const uchar* scales [[buffer(1)]], // [OUT, IN/32]
    device const float* x [[buffer(2)]],      // [IN]
    device float* y [[buffer(3)]],            // [OUT]
    uint gid [[thread_position_in_grid]]) {
    if (gid >= OUT_DIM) return;
    uint scale_k = IN_DIM / 32u;
    device const uchar* prow = packed + gid * IN_DIM;
    device const uchar* srow = scales + gid * scale_k;
    float acc = 0.f;
    for (uint g = 0; g < scale_k; ++g) {
        float sc = e8m0_from_bits(uint(srow[g]));
        uint base = g * 32u;
        for (uint j = 0; j < 32u; ++j) {
            acc += e4m3_from_bits(uint(prow[base + j])) * sc * x[base + j];
        }
    }
    y[gid] = acc;
}
