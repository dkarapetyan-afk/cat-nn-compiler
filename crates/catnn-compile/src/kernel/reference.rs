//! Naive sequential reference implementation. Used as the left-hand side
//! of every 2-cell witness: optimized kernels must match this within
//! [`catnn_core::category::EquivWitness::abs_tol`].

use catnn_core::tensor::{Tensor, TensorType};
use catnn_core::{CatError, CatResult};

pub fn relu(x: &Tensor) -> Tensor {
    let mut y = x.clone();
    for v in &mut y.data {
        *v = v.max(0.0);
    }
    y
}

pub fn gelu(x: &Tensor) -> Tensor {
    let mut y = x.clone();
    for v in &mut y.data {
        let z = *v;
        let inner = (2.0f32 / std::f32::consts::PI).sqrt() * (z + 0.044715 * z * z * z);
        *v = 0.5 * z * (1.0 + inner.tanh());
    }
    y
}

pub fn softmax_last(x: &Tensor) -> CatResult<Tensor> {
    let last = *x.ty.shape.dims.last().unwrap_or(&x.data.len());
    let mut y = x.clone();
    for row in y.data.chunks_mut(last) {
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
    Ok(y)
}

pub fn layer_norm(x: &Tensor, gamma: &Tensor, beta: &Tensor, eps: f32) -> CatResult<Tensor> {
    let last = *x.ty.shape.dims.last().ok_or_else(|| CatError::ShapeMismatch("ln".into()))?;
    let mut y = x.clone();
    for row in y.data.chunks_mut(last) {
        let n = last as f32;
        let mean = row.iter().sum::<f32>() / n;
        let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
        let inv = 1.0 / (var + eps).sqrt();
        for (i, val) in row.iter_mut().enumerate() {
            *val = (*val - mean) * inv * gamma.data[i] + beta.data[i];
        }
    }
    Ok(y)
}

pub fn add(a: &Tensor, b: &Tensor) -> CatResult<Tensor> {
    crate::kernel::cpu::add(a, b)
}

pub fn linear(x: &Tensor, w: &Tensor, bias: Option<&Tensor>) -> CatResult<Tensor> {
    let xd = &x.ty.shape.dims;
    let wd = &w.ty.shape.dims;
    let k = *xd.last().unwrap();
    let n = wd[1];
    let m = x.data.len() / k;
    let mut out_dims = xd.clone();
    *out_dims.last_mut().unwrap() = n;
    let mut data = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut s = 0.0;
            for p in 0..k {
                s += x.data[i * k + p] * w.data[p * n + j];
            }
            if let Some(b) = bias {
                s += b.data[j];
            }
            data[i * n + j] = s;
        }
    }
    Tensor::from_vec(TensorType::f32(out_dims), data)
}

pub fn attention(
    x: &Tensor,
    wq: &Tensor,
    wk: &Tensor,
    wv: &Tensor,
    wo: &Tensor,
    n_heads: usize,
    d_head: usize,
) -> CatResult<Tensor> {
    crate::kernel::cpu::attention(x, wq, wk, wv, wo, n_heads, d_head)
}
