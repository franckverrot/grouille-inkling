//! Native NVFP4 decode (E2M1 × E4M3, group 16, lo-nibble first).
//!
//! Matches MLX `mx.dequantize(..., mode="nvfp4")` then scales by `amax / 2688`.

/// Max |E2M1| × max finite E4M3 = 6 × 448.
pub const NVFP4_MAX: f32 = 2688.0;

const E2M1: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0,
    -6.0,
];

#[inline]
fn e4m3_from_bits(b: u8) -> f32 {
    let sign = (b >> 7) & 1;
    let exp = ((b >> 3) & 0xF) as i32;
    let mant = (b & 0x7) as f32;
    let mag = if exp != 0 {
        (1.0 + mant / 8.0) * 2f32.powi(exp - 7)
    } else {
        (mant / 8.0) * 2f32.powi(1 - 7)
    };
    if sign != 0 {
        -mag
    } else {
        mag
    }
}

/// Dequantize packed NVFP4 rows.
///
/// * `packed`: `[out, in/2]` uint8, two E2M1 values per byte (lo nibble first)
/// * `scales`: `[out, in/16]` raw E4M3 bit patterns as u8
/// * `amax`: per-expert outer scale (ModelOpt); applied as `amax / 2688`
pub fn dequant_nvfp4_rows(packed: &[u8], scales: &[u8], out_dim: usize, amax: f32) -> Vec<f32> {
    let packed_k = packed.len() / out_dim;
    let in_dim = packed_k * 2;
    assert_eq!(scales.len(), out_dim * (in_dim / 16));
    let scale_factor = amax / NVFP4_MAX;
    let mut out = vec![0f32; out_dim * in_dim];
    for r in 0..out_dim {
        let prow = &packed[r * packed_k..(r + 1) * packed_k];
        let srow = &scales[r * (in_dim / 16)..(r + 1) * (in_dim / 16)];
        for (g, &sb) in srow.iter().enumerate() {
            let sc = e4m3_from_bits(sb) * scale_factor;
            let base = g * 8; // 8 bytes → 16 nibbles
            for j in 0..8 {
                let byte = prow[base + j];
                let o = r * in_dim + g * 16 + j * 2;
                out[o] = E2M1[(byte & 0x0F) as usize] * sc;
                out[o + 1] = E2M1[(byte >> 4) as usize] * sc;
            }
        }
    }
    out
}

/// `y[out] = W[out, in] @ x[in]` with W stored NVFP4-packed.
pub fn nvfp4_gemv(
    packed: &[u8],
    scales: &[u8],
    out_dim: usize,
    amax: f32,
    x: &[f32],
) -> Vec<f32> {
    let packed_k = packed.len() / out_dim;
    let in_dim = packed_k * 2;
    assert_eq!(x.len(), in_dim);
    assert_eq!(scales.len(), out_dim * (in_dim / 16));
    let scale_factor = amax / NVFP4_MAX;
    let mut y = vec![0f32; out_dim];
    for r in 0..out_dim {
        let prow = &packed[r * packed_k..(r + 1) * packed_k];
        let srow = &scales[r * (in_dim / 16)..(r + 1) * (in_dim / 16)];
        let mut acc = 0f32;
        for (g, &sb) in srow.iter().enumerate() {
            let sc = e4m3_from_bits(sb) * scale_factor;
            let base = g * 8;
            for j in 0..8 {
                let byte = prow[base + j];
                let i = g * 16 + j * 2;
                acc += E2M1[(byte & 0x0F) as usize] * sc * x[i];
                acc += E2M1[(byte >> 4) as usize] * sc * x[i + 1];
            }
        }
        y[r] = acc;
    }
    y
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn e2m1_table_endpoints() {
        assert_eq!(E2M1[0], 0.0);
        assert_eq!(E2M1[7], 6.0);
        assert_eq!(E2M1[15], -6.0);
    }

    #[test]
    fn e4m3_one() {
        // 0b00111000 = exp=7, mant=0 → 1.0
        assert!((e4m3_from_bits(0b0011_1000) - 1.0).abs() < 1e-6);
    }
}
