//! Native MXFP8 decode (E4M3 × E8M0, group 32).
//!
//! Matches MLX `mx.dequantize(..., mode="mxfp8", group_size=32, bits=8)`.

pub const MXFP8_GROUP: usize = 32;

/// OCP E4M3FN → f32 (same finite values as MLX's half-trick path).
#[inline]
pub fn e4m3_from_bits(b: u8) -> f32 {
    let v = (b & 127) as u16;
    let sign = ((b as u16) >> 7) << 15;
    let u = (v << 7) | ((((v + 1) >> 7) << 14) | sign);
    half::f16::from_bits(u).to_f32() * 256.0
}

/// UE8M0 power-of-two scale → f32 (MLX `fp8_e8m0`).
#[inline]
pub fn e8m0_from_bits(b: u8) -> f32 {
    if b == 0 {
        f32::from_bits(0x0040_0000) // 2^-127
    } else {
        f32::from_bits((b as u32) << 23)
    }
}

/// `y[out] = W_mxfp8[out, in] @ x[in]`.
///
/// * `packed`: `[out, in]` uint8 E4M3
/// * `scales`: `[out, in/32]` uint8 E8M0
pub fn mxfp8_gemv(packed: &[u8], scales: &[u8], out_dim: usize, x: &[f32]) -> Vec<f32> {
    let in_dim = x.len();
    assert_eq!(packed.len(), out_dim * in_dim);
    assert_eq!(in_dim % MXFP8_GROUP, 0);
    assert_eq!(scales.len(), out_dim * (in_dim / MXFP8_GROUP));
    let mut y = vec![0f32; out_dim];
    let groups = in_dim / MXFP8_GROUP;
    for r in 0..out_dim {
        let prow = &packed[r * in_dim..(r + 1) * in_dim];
        let srow = &scales[r * groups..(r + 1) * groups];
        let mut acc = 0f32;
        for g in 0..groups {
            let sc = e8m0_from_bits(srow[g]);
            let base = g * MXFP8_GROUP;
            for j in 0..MXFP8_GROUP {
                acc += e4m3_from_bits(prow[base + j]) * sc * x[base + j];
            }
        }
        y[r] = acc;
    }
    y
}

/// Dequantize packed MXFP8 rows to f32 `[out, in]`.
pub fn dequant_mxfp8_rows(packed: &[u8], scales: &[u8], out_dim: usize) -> Vec<f32> {
    let in_dim = packed.len() / out_dim;
    assert_eq!(in_dim % MXFP8_GROUP, 0);
    assert_eq!(scales.len(), out_dim * (in_dim / MXFP8_GROUP));
    let groups = in_dim / MXFP8_GROUP;
    let mut out = vec![0f32; out_dim * in_dim];
    for r in 0..out_dim {
        let prow = &packed[r * in_dim..(r + 1) * in_dim];
        let srow = &scales[r * groups..(r + 1) * groups];
        for g in 0..groups {
            let sc = e8m0_from_bits(srow[g]);
            let base = g * MXFP8_GROUP;
            for j in 0..MXFP8_GROUP {
                out[r * in_dim + base + j] = e4m3_from_bits(prow[base + j]) * sc;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn e8m0_powers() {
        assert!((e8m0_from_bits(127) - 1.0).abs() < 1e-6);
        assert!((e8m0_from_bits(128) - 2.0).abs() < 1e-6);
    }

    #[test]
    fn e4m3_one() {
        // 0b00111000 = exp=7, mant=0 → 1.0
        assert!((e4m3_from_bits(0b0011_1000) - 1.0).abs() < 1e-5);
    }
}
