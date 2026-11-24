// NVFP4 expert GEMV: y[out] = W_nvfp4[out, in] @ x[in]
// Packing: uint8 lo-nibble first, E2M1 × E4M3 groups of 16, scale *= amax/2688.

#include <metal_stdlib>
using namespace metal;

constant uint OUT_DIM [[function_constant(0)]];
constant uint IN_DIM [[function_constant(1)]]; // must be divisible by 16

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

kernel void k_nvfp4_gemv(
    device const uchar* packed [[buffer(0)]], // [OUT, IN/2]
    device const uchar* scales [[buffer(1)]], // [OUT, IN/16]
    device const float* x [[buffer(2)]],      // [IN]
    device float* y [[buffer(3)]],            // [OUT]
    constant float& amax [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= OUT_DIM) return;
    constexpr float NVFP4_MAX = 2688.f;
    float scale_factor = amax / NVFP4_MAX;
    uint packed_k = IN_DIM / 2u;
    uint scale_k = IN_DIM / 16u;
    device const uchar* prow = packed + gid * packed_k;
    device const uchar* srow = scales + gid * scale_k;
    float acc = 0.f;
    for (uint g = 0; g < scale_k; ++g) {
        float sc = e4m3_from_bits(uint(srow[g])) * scale_factor;
        uint base = g * 8u;
        for (uint j = 0; j < 8u; ++j) {
            uchar byte = prow[base + j];
            uint i = g * 16u + j * 2u;
            acc += e2m1(byte & 0xFu) * sc * x[i];
            acc += e2m1(byte >> 4) * sc * x[i + 1u];
        }
    }
    y[gid] = acc;
}
