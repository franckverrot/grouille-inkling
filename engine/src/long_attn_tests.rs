//! Attention over a long history, on synthetic data, against a plain host reference.
//! Needs a GPU and no model weights.

use objc2_metal::{MTLCommandBuffer, MTLCommandEncoder};

use crate::cache::{GpuKv, RING};
use crate::gpu::{read_buffer, write_buffer, Buffer};
use crate::metal_glue::{decode_attn_meta, DecodeAttn, FAR_STRIDE};
use crate::metal_ops::MetalOps;

const N_HEADS: usize = 24;
const N_KV: usize = 8;
const DH: usize = 128;
const D_REL: usize = 16;
const Q_CH: usize = N_HEADS * DH;
const KV_CH: usize = N_KV * DH;

/// f32 kernels against an f64 reference land near 1e-7.  With the f16 archive
/// (`INKLING_KV_F16`) the stored rows themselves are rounded.
fn tol() -> f64 {
    if crate::flags::kv_f16() {
        5e-3
    } else {
        1e-5
    }
}

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
}

/// Inputs of one query token.
struct Query {
    q: Vec<f32>,
    r: Vec<f32>,
    sinks: Vec<f32>,
    gate: Vec<f32>,
}

impl Query {
    fn random(rng: &mut Rng, tokens: usize) -> Self {
        Self {
            q: rng.vec(tokens * Q_CH, 1.0),
            r: rng.vec(tokens * N_HEADS * D_REL, 0.5),
            sinks: rng.vec(tokens * N_HEADS, 1.0),
            gate: rng.vec(tokens * Q_CH, 1.0),
        }
    }
}

/// The definition: token `t` of `x`, sitting at position `pos`, attends to rows
/// `0..=pos` of `k` / `v`.  A sliding visit sees the last `RING` of them.  A key
/// gets a relative logit while it is fewer than `RING` tokens back.
fn host_attn(x: &Query, t: usize, pos: usize, k: &[f32], v: &[f32], rel: &[f32], full: bool) -> Vec<f64> {
    let scale = 1.0 / (DH as f64).sqrt();
    let mut out = vec![0f64; Q_CH];
    for h in 0..N_HEADS {
        let g = h / (N_HEADS / N_KV);
        let q = &x.q[t * Q_CH + h * DH..][..DH];
        let r = &x.r[(t * N_HEADS + h) * D_REL..][..D_REL];
        let first = if full { 0 } else { (pos + 1).saturating_sub(RING) };
        let logits: Vec<f64> = (first..=pos)
            .map(|j| {
                let kr = &k[j * KV_CH + g * DH..][..DH];
                let mut l = scale * q.iter().zip(kr).map(|(a, b)| *a as f64 * *b as f64).sum::<f64>();
                let dist = pos - j;
                if dist < RING {
                    l += (0..D_REL).map(|c| r[c] as f64 * rel[c * RING + dist] as f64).sum::<f64>();
                }
                l
            })
            .collect();
        let sink = x.sinks[t * N_HEADS + h] as f64;
        let max = logits.iter().copied().fold(sink, f64::max);
        let total = logits.iter().map(|l| (l - max).exp()).sum::<f64>() + (sink - max).exp();
        for (i, l) in logits.iter().enumerate() {
            let w = (l - max).exp() / total;
            let vr = &v[(first + i) * KV_CH + g * DH..][..DH];
            for d in 0..DH {
                out[h * DH + d] += w * vr[d] as f64;
            }
        }
        for d in 0..DH {
            out[h * DH + d] /= 1.0 + (-(x.gate[t * Q_CH + h * DH + d] as f64)).exp();
        }
    }
    out
}

fn worst(got: &[f32], want: &[f64]) -> f64 {
    got.iter().zip(want).map(|(a, b)| (*a as f64 - b).abs()).fold(0.0, f64::max)
}

/// Relative-logit table: f16 on the GPU, the same values as f32 for the host.
fn rel_table(ops: &MetalOps, rng: &mut Rng) -> (Buffer, Vec<f32>) {
    let halfs: Vec<half::f16> = rng.vec(D_REL * RING, 0.3).into_iter().map(half::f16::from_f32).collect();
    let b = ops.gemv.buffer_shared(halfs.len() * 2).unwrap();
    write_buffer(&b, 0, &halfs);
    (b, halfs.iter().map(|h| h.to_f32()).collect())
}

/// Feed `total` tokens one at a time, as decode does, and check the attention of the
/// tokens at `probes`.  `prefilled` rows go in through the host path first.
fn decode_case(full: bool, prefilled: usize, total: usize, probes: &[usize]) {
    let mut ops = MetalOps::new().unwrap();
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15 ^ total as u64);
    let (b_rel, rel) = rel_table(&ops, &mut rng);
    let k = rng.vec(total * KV_CH, 1.0);
    let v = rng.vec(total * KV_CH, 1.0);
    let mut kv = GpuKv::new(&ops.gemv, full).unwrap();
    kv.append(&ops.gemv, &k[..prefilled * KV_CH], &v[..prefilled * KV_CH], prefilled).unwrap();
    let buf = |n: usize| ops.gemv.buffer_shared(n * 4).unwrap();
    let (b_q, b_r, b_sink, b_gate, b_ctx, b_meta) =
        (buf(Q_CH), buf(N_HEADS * D_REL), buf(N_HEADS), buf(Q_CH), buf(Q_CH), buf(64));
    let b_far = buf((total / 512 + 2) * N_HEADS * FAR_STRIDE);
    for pos in prefilled..total {
        let meta = decode_attn_meta(pos, full);
        let far_rows = meta[3] as usize;
        let slot = pos % RING;
        if far_rows > 0 {
            kv.reserve_archive(&ops.gemv, far_rows).unwrap();
            let cb = ops.gemv.command_buffer().unwrap();
            let enc = cb.computeCommandEncoder().unwrap();
            ops.encode_archive_row(&enc, &kv, slot, far_rows - 1).unwrap();
            enc.endEncoding();
            cb.commit();
            cb.waitUntilCompleted();
        }
        // The fused visit writes the new row on the GPU; here the host does.
        write_buffer(&kv.k_ring, slot * KV_CH * 4, &k[pos * KV_CH..][..KV_CH]);
        write_buffer(&kv.v_ring, slot * KV_CH * 4, &v[pos * KV_CH..][..KV_CH]);
        if probes.contains(&pos) {
            let x = Query::random(&mut rng, 1);
            write_buffer(&b_q, 0, &x.q);
            write_buffer(&b_r, 0, &x.r);
            write_buffer(&b_sink, 0, &x.sinks);
            write_buffer(&b_gate, 0, &x.gate);
            write_buffer(&b_meta, 0, &meta);
            let cb = ops.gemv.command_buffer().unwrap();
            let enc = cb.computeCommandEncoder().unwrap();
            let far = kv.archive_segments(far_rows);
            ops.encode_decode_attn(
                &enc,
                &DecodeAttn {
                    q: &b_q,
                    k_ring: &kv.k_ring,
                    v_ring: &kv.v_ring,
                    r: &b_r,
                    rel_proj: &b_rel,
                    sinks: &b_sink,
                    gate: &b_gate,
                    ctx: &b_ctx,
                    meta: &b_meta,
                    far_parts: &b_far,
                    far: &far,
                },
                pos,
            )
            .unwrap();
            enc.endEncoding();
            cb.commit();
            cb.waitUntilCompleted();
            let mut got = vec![0f32; Q_CH];
            read_buffer(&b_ctx, 0, &mut got);
            let err = worst(&got, &host_attn(&x, 0, pos, &k, &v, &rel, full));
            if std::env::var_os("ATTN_TEST_VERBOSE").is_some() {
                eprintln!("decode full={full} position {pos}: off by {err:.2e}");
            }
            assert!(err < tol(), "decode full={full} position {pos}: off by {err}");
        }
        kv.len += 1;
    }
}

/// Feed `n_chunks` chunks of `chunk` tokens, as prompt processing does, and check
/// some tokens of every chunk.
fn prompt_case(full: bool, chunk: usize, n_chunks: usize) {
    let mut ops = MetalOps::new().unwrap();
    let mut rng = Rng(0xD1B5_4A32_D192_ED03 ^ (chunk * n_chunks) as u64);
    let (b_rel, rel) = rel_table(&ops, &mut rng);
    let total = chunk * n_chunks;
    let k = rng.vec(total * KV_CH, 1.0);
    let v = rng.vec(total * KV_CH, 1.0);
    let mut kv = GpuKv::new(&ops.gemv, full).unwrap();
    for c in 0..n_chunks {
        let pos0 = c * chunk;
        let x = Query::random(&mut rng, chunk);
        let rows = pos0 * KV_CH..(pos0 + chunk) * KV_CH;
        let mut ctx = vec![0f32; chunk * Q_CH];
        ops.attn_chunk(
            &x.q, &k[rows.clone()], &v[rows.clone()], &x.r, &b_rel, &x.sinks, &x.gate, chunk,
            Some(&kv), full, &mut ctx,
        )
        .unwrap();
        for t in [0, 1, 7, 31, 32, chunk / 2, chunk - 2, chunk - 1] {
            let want = host_attn(&x, t, pos0 + t, &k, &v, &rel, full);
            let err = worst(&ctx[t * Q_CH..(t + 1) * Q_CH], &want);
            if std::env::var_os("ATTN_TEST_VERBOSE").is_some() {
                eprintln!("prompt full={full} chunk {c} token {t}: off by {err:.2e}");
            }
            assert!(err < tol(), "prompt full={full} chunk {c} token {t}: off by {err}");
        }
        kv.append(&ops.gemv, &k[rows.clone()], &v[rows], chunk).unwrap();
    }
}

#[test]
fn decode_sliding_wraps() {
    decode_case(false, 0, 2300, &[0, 5, 700, 1023, 1024, 1025, 1500, 2047, 2048, 2299]);
}

#[test]
fn decode_full_uses_archive() {
    decode_case(true, 0, 2700, &[0, 700, 1023, 1024, 1025, 1300, 1535, 1536, 1537, 2047, 2048, 2049, 2699]);
}

#[test]
fn decode_full_after_long_prompt() {
    // An archive segment is 8192 rows, so the second one starts at position 9216.
    decode_case(true, 9000, 9500, &[9000, 9001, 9215, 9216, 9217, 9218, 9499]);
}

#[test]
fn prompt_sliding_chunks() {
    prompt_case(false, 862, 4);
    prompt_case(false, 1024, 3);
}

#[test]
fn prompt_full_chunks() {
    prompt_case(true, 862, 5);
    // Eleven chunks reach into the second archive segment.
    prompt_case(true, 1024, 11);
}
