//! Executable CPU kernels. These run — they are not stubs.
//!
//! Matmul uses a blocked layout plus Rayon over the outer dimension and
//! `wide::f32x8` SIMD on the inner accumulation when the K dimension
//! allows it. Activations and reductions are straightforward and
//! numerically match the reference (same algorithm, different schedule).

use rayon::prelude::*;
use wide::f32x8;

use catnn_core::tensor::Tensor;
use catnn_core::{CatError, CatResult};

pub fn relu(x: &Tensor) -> Tensor {
    let mut y = x.clone();
    y.data.iter_mut().for_each(|v| *v = v.max(0.0));
    y
}

pub fn gelu(x: &Tensor) -> Tensor {
    // tanh approximation of GELU, used by both CPU and reference.
    let mut y = x.clone();
    y.data.iter_mut().for_each(|v| {
        let z = *v;
        let inner = (2.0f32 / std::f32::consts::PI).sqrt() * (z + 0.044715 * z * z * z);
        *v = 0.5 * z * (1.0 + inner.tanh());
    });
    y
}

pub fn softmax_last(x: &Tensor) -> CatResult<Tensor> {
    let dims = &x.ty.shape.dims;
    if dims.is_empty() {
        return Ok(x.clone());
    }
    let last = *dims.last().unwrap();
    let rows = x.data.len() / last;
    let mut y = x.clone();
    y.data.par_chunks_mut(last).zip(x.data.par_chunks(last)).take(rows).for_each(
        |(out, inp)| {
            let max = inp.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for (o, i) in out.iter_mut().zip(inp.iter()) {
                *o = (i - max).exp();
                sum += *o;
            }
            let inv = 1.0 / sum.max(1e-30);
            for o in out.iter_mut() {
                *o *= inv;
            }
        },
    );
    Ok(y)
}

pub fn layer_norm(x: &Tensor, gamma: &Tensor, beta: &Tensor, eps: f32) -> CatResult<Tensor> {
    let last = *x.ty.shape.dims.last().ok_or_else(|| CatError::ShapeMismatch("ln scalar".into()))?;
    if gamma.data.len() != last || beta.data.len() != last {
        return Err(CatError::ShapeMismatch("ln gamma/beta length".into()));
    }
    let mut y = x.clone();
    y.data.par_chunks_mut(last).zip(x.data.par_chunks(last)).for_each(|(out, inp)| {
        let n = last as f32;
        let mean = inp.iter().sum::<f32>() / n;
        let var = inp.iter().map(|v| {
            let d = v - mean;
            d * d
        }).sum::<f32>()
            / n;
        let inv = 1.0 / (var + eps).sqrt();
        for i in 0..last {
            out[i] = (inp[i] - mean) * inv * gamma.data[i] + beta.data[i];
        }
    });
    Ok(y)
}

pub fn add(a: &Tensor, b: &Tensor) -> CatResult<Tensor> {
    if a.data.len() == b.data.len() {
        let mut y = a.clone();
        for (o, v) in y.data.iter_mut().zip(b.data.iter()) {
            *o += *v;
        }
        return Ok(y);
    }
    // Broadcast b over the last dim (bias).
    let last = *a.ty.shape.dims.last().unwrap_or(&a.data.len());
    if b.data.len() == last {
        let mut y = a.clone();
        for (i, o) in y.data.iter_mut().enumerate() {
            *o += b.data[i % last];
        }
        return Ok(y);
    }
    Err(CatError::ShapeMismatch(format!("add {} vs {}", a.ty, b.ty)))
}

pub fn mul(a: &Tensor, b: &Tensor) -> CatResult<Tensor> {
    if a.data.len() != b.data.len() {
        return Err(CatError::ShapeMismatch("mul length".into()));
    }
    let mut y = a.clone();
    for (o, v) in y.data.iter_mut().zip(b.data.iter()) {
        *o *= *v;
    }
    Ok(y)
}

/// `y = x @ w + b?` with `x: [M, K]`, `w: [K, N]`.
pub fn linear(x: &Tensor, w: &Tensor, bias: Option<&Tensor>) -> CatResult<Tensor> {
    let xd = &x.ty.shape.dims;
    let wd = &w.ty.shape.dims;
    if xd.len() < 2 || wd.len() != 2 {
        return Err(CatError::ShapeMismatch(format!("linear {} @ {}", x.ty, w.ty)));
    }
    let k = xd[xd.len() - 1];
    let n = wd[1];
    if wd[0] != k {
        return Err(CatError::ShapeMismatch(format!("linear K {k} vs {}", wd[0])));
    }
    let m = x.data.len() / k;
    let mut out_dims = xd.clone();
    *out_dims.last_mut().unwrap() = n;
    let mut y = Tensor::zeros(catnn_core::tensor::TensorType::f32(out_dims));
    matmul_mk_kn(&x.data, &w.data, &mut y.data, m, k, n);
    if let Some(b) = bias {
        if b.data.len() != n {
            return Err(CatError::ShapeMismatch("bias length".into()));
        }
        for row in 0..m {
            for j in 0..n {
                y.data[row * n + j] += b.data[j];
            }
        }
    }
    Ok(y)
}

pub fn matmul_mk_kn(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    const BN: usize = 32;
    c.par_chunks_mut(n).enumerate().take(m).for_each(|(i, row)| {
        let ai = &a[i * k..(i + 1) * k];
        for j0 in (0..n).step_by(BN) {
            let j1 = (j0 + BN).min(n);
            for j in j0..j1 {
                row[j] = dot_row(ai, b, k, n, j);
            }
        }
    });
}

fn dot_row(a: &[f32], b: &[f32], k: usize, n: usize, j: usize) -> f32 {
    let mut acc = f32x8::ZERO;
    let mut p = 0;
    while p + 8 <= k {
        let av = f32x8::from([
            a[p],
            a[p + 1],
            a[p + 2],
            a[p + 3],
            a[p + 4],
            a[p + 5],
            a[p + 6],
            a[p + 7],
        ]);
        let bv = f32x8::from([
            b[p * n + j],
            b[(p + 1) * n + j],
            b[(p + 2) * n + j],
            b[(p + 3) * n + j],
            b[(p + 4) * n + j],
            b[(p + 5) * n + j],
            b[(p + 6) * n + j],
            b[(p + 7) * n + j],
        ]);
        acc += av * bv;
        p += 8;
    }
    let mut s = acc.reduce_add();
    while p < k {
        s += a[p] * b[p * n + j];
        p += 1;
    }
    s
}

/// Multi-head self-attention on `x: [B, S, D]`.
pub fn attention(
    x: &Tensor,
    wq: &Tensor,
    wk: &Tensor,
    wv: &Tensor,
    wo: &Tensor,
    n_heads: usize,
    d_head: usize,
) -> CatResult<Tensor> {
    let dims = &x.ty.shape.dims;
    if dims.len() != 3 {
        return Err(CatError::ShapeMismatch("attention expects [B,S,D]".into()));
    }
    let (b, s, d) = (dims[0], dims[1], dims[2]);
    let q = linear3(x, wq, b, s, d)?;
    let k = linear3(x, wk, b, s, d)?;
    let v = linear3(x, wv, b, s, d)?;
    let scale = 1.0 / (d_head as f32).sqrt();
    let mut ctx = vec![0.0f32; b * s * d];
    for bi in 0..b {
        for h in 0..n_heads {
            // scores [S, S]
            let mut scores = vec![0.0f32; s * s];
            for i in 0..s {
                for j in 0..s {
                    let mut dot = 0.0;
                    for t in 0..d_head {
                        let qi = q.data[bi * s * d + i * d + h * d_head + t];
                        let kj = k.data[bi * s * d + j * d + h * d_head + t];
                        dot += qi * kj;
                    }
                    scores[i * s + j] = dot * scale;
                }
            }
            // softmax rows
            for i in 0..s {
                let row = &mut scores[i * s..(i + 1) * s];
                let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0;
                for v in row.iter_mut() {
                    *v = (*v - max).exp();
                    sum += *v;
                }
                let inv = 1.0 / sum.max(1e-30);
                for v in row.iter_mut() {
                    *v *= inv;
                }
            }
            for i in 0..s {
                for t in 0..d_head {
                    let mut acc = 0.0;
                    for j in 0..s {
                        let att = scores[i * s + j];
                        let vv = v.data[bi * s * d + j * d + h * d_head + t];
                        acc += att * vv;
                    }
                    ctx[bi * s * d + i * d + h * d_head + t] = acc;
                }
            }
        }
    }
    let ctx_t = Tensor::from_vec(x.ty.clone(), ctx)?;
    linear3(&ctx_t, wo, b, s, d)
}

fn linear3(x: &Tensor, w: &Tensor, b: usize, s: usize, d: usize) -> CatResult<Tensor> {
    let flat = Tensor::from_vec(
        catnn_core::tensor::TensorType::f32([b * s, d]),
        x.data.clone(),
    )?;
    let y = linear(&flat, w, None)?;
    Tensor::from_vec(catnn_core::tensor::TensorType::f32([b, s, d]), y.data)
}

pub fn apply_fused_linear_relu(x: &Tensor, w: &Tensor, bias: Option<&Tensor>) -> CatResult<Tensor> {
    let mut y = linear(x, w, bias)?;
    y.data.iter_mut().for_each(|v| *v = v.max(0.0));
    Ok(y)
}

#[cfg(test)]
mod tests {
    use super::*;
    use catnn_core::tensor::TensorType;

    #[test]
    fn relu_zeros_negatives() {
        let t = Tensor::from_vec(TensorType::f32([4]), vec![-1.0, 0.0, 2.0, -0.5]).unwrap();
        assert_eq!(relu(&t).data, vec![0.0, 0.0, 2.0, 0.0]);
    }

    #[test]
    fn linear_matches_tiny_hand() {
        let x = Tensor::from_vec(TensorType::f32([1, 2]), vec![1.0, 2.0]).unwrap();
        let w = Tensor::from_vec(TensorType::f32([2, 2]), vec![1.0, 0.0, 0.0, 1.0]).unwrap();
        let y = linear(&x, &w, None).unwrap();
        assert_eq!(y.data, vec![1.0, 2.0]);
    }
}
