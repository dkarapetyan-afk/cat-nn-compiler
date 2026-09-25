//! Reverse-mode rules: `put` of each generator of `Para(Lens)`.
//!
//! Composition of these puts is the chain rule (see
//! [`catnn_core::lens::para`]). The Mixtral router uses a **straight-through
//! / stop-gradient** on the discrete top-k index: gradients flow through
//! the softmax weights of the *selected* experts only.

use catnn_core::tensor::{Tensor, TensorType};
use catnn_core::{CatError, CatResult};

use crate::kernel::cpu;
use crate::kernel::moe;

fn last_dim(t: &Tensor) -> CatResult<usize> {
    t.ty.shape.last_dim()
}

pub fn add_tensors(a: &Tensor, b: &Tensor) -> CatResult<Tensor> {
    cpu::add(a, b)
}

pub fn scale(t: &Tensor, s: f32) -> Tensor {
    let mut y = t.clone();
    for v in &mut y.data {
        *v *= s;
    }
    y
}

/// `y = x @ w + b?` with `x: [M,K]`, `w: [K,N]`.
pub fn linear_backward(
    x: &Tensor,
    w: &Tensor,
    dy: &Tensor,
) -> CatResult<(Tensor, Tensor, Option<Tensor>)> {
    let xd = &x.ty.shape.dims;
    let k = *xd.last().unwrap();
    let m = x.data.len() / k;
    let n = w.ty.shape.dims[1];
    let mut dw = vec![0.0f32; k * n];
    let mut dx = vec![0.0f32; m * k];
    let mut db = vec![0.0f32; n];
    for i in 0..m {
        for j in 0..n {
            let g = dy.data[i * n + j];
            db[j] += g;
            for p in 0..k {
                dw[p * n + j] += x.data[i * k + p] * g;
                dx[i * k + p] += g * w.data[p * n + j];
            }
        }
    }
    let dx_t = Tensor::from_vec(x.ty.clone(), dx)?;
    let dw_t = Tensor::from_vec(w.ty.clone(), dw)?;
    let db_t = Tensor::from_vec(TensorType::f32([n]), db)?;
    Ok((dx_t, dw_t, Some(db_t)))
}

pub fn relu_backward(x: &Tensor, dy: &Tensor) -> CatResult<Tensor> {
    if x.data.len() != dy.data.len() {
        return Err(CatError::ShapeMismatch("relu grad".into()));
    }
    let mut dx = dy.clone();
    for (o, xi) in dx.data.iter_mut().zip(x.data.iter()) {
        if *xi <= 0.0 {
            *o = 0.0;
        }
    }
    Ok(dx)
}

pub fn gelu_backward(x: &Tensor, dy: &Tensor) -> CatResult<Tensor> {
    // Derivative of the tanh GELU used in the forward kernel.
    let mut dx = dy.clone();
    const C: f32 = 0.044715;
    let k = (2.0f32 / std::f32::consts::PI).sqrt();
    for (o, &z) in dx.data.iter_mut().zip(x.data.iter()) {
        let u = k * (z + C * z * z * z);
        let t = u.tanh();
        let sech2 = 1.0 - t * t;
        let du = k * (1.0 + 3.0 * C * z * z);
        let dgelu = 0.5 * (1.0 + t) + 0.5 * z * sech2 * du;
        *o *= dgelu;
    }
    Ok(dx)
}

pub fn softmax_backward(y: &Tensor, dy: &Tensor) -> CatResult<Tensor> {
    let last = last_dim(y)?;
    let mut dx = dy.clone();
    for (row_y, (row_dy, row_dx)) in y
        .data
        .chunks(last)
        .zip(dy.data.chunks(last).zip(dx.data.chunks_mut(last)))
    {
        let dot: f32 = row_y.iter().zip(row_dy.iter()).map(|(s, g)| s * g).sum();
        for i in 0..last {
            row_dx[i] = row_y[i] * (row_dy[i] - dot);
        }
    }
    Ok(dx)
}

pub fn mse_loss(pred: &Tensor, target: &Tensor) -> CatResult<(f32, Tensor)> {
    if pred.data.len() != target.data.len() {
        return Err(CatError::ShapeMismatch("mse".into()));
    }
    let n = pred.data.len() as f32;
    let mut loss = 0.0f32;
    let mut d = pred.clone();
    for i in 0..pred.data.len() {
        let e = pred.data[i] - target.data[i];
        loss += e * e;
        d.data[i] = 2.0 * e / n;
    }
    Ok((loss / n, d))
}

/// Mean softmax cross-entropy on the last dim. Targets are integer ids
/// stored as `f32` (same convention as the embedding input).
pub fn cross_entropy(logits: &Tensor, targets: &[u32]) -> CatResult<(f32, Tensor)> {
    let last = last_dim(logits)?;
    let rows = logits.data.len() / last;
    if targets.len() != rows {
        return Err(CatError::ShapeMismatch("ce targets".into()));
    }
    let probs = cpu::softmax_last(logits)?;
    let mut loss = 0.0f32;
    let mut d = probs.clone();
    for (r, &tgt) in targets.iter().enumerate() {
        let t = tgt as usize;
        if t >= last {
            return Err(CatError::ShapeMismatch(format!("target {t} >= {last}")));
        }
        let p = probs.data[r * last + t].max(1e-30);
        loss -= p.ln();
        for j in 0..last {
            d.data[r * last + j] /= rows as f32;
        }
        d.data[r * last + t] -= 1.0 / rows as f32;
    }
    Ok((loss / rows as f32, d))
}

#[allow(clippy::needless_range_loop)]
pub fn rmsnorm_backward(
    x: &Tensor,
    w: &Tensor,
    dy: &Tensor,
    eps: f32,
) -> CatResult<(Tensor, Tensor)> {
    let last = last_dim(x)?;
    if w.data.len() != last {
        return Err(CatError::ShapeMismatch("rmsnorm w".into()));
    }
    let mut dx = Tensor::zeros(x.ty.clone());
    let mut dw = vec![0.0f32; last];
    let n = last as f32;
    for (row, (drow, yrow)) in
        x.data.chunks(last).zip(dx.data.chunks_mut(last).zip(dy.data.chunks(last)))
    {
        let ms = row.iter().map(|v| v * v).sum::<f32>() / n;
        let inv = 1.0 / (ms + eps).sqrt();
        let mut dinv = 0.0f32;
        for ((dwi, yi), (xi, wi)) in
            dw.iter_mut().zip(yrow.iter()).zip(row.iter().zip(w.data.iter()))
        {
            *dwi += *yi * *xi * inv;
            dinv += *yi * *xi * *wi;
        }
        let dms = dinv * (-0.5) * (ms + eps).powf(-1.5);
        for (i, slot) in drow.iter_mut().enumerate() {
            yrow_dx(yrow, row, w, inv, dms, n, i, slot);
        }
    }
    Ok((dx, Tensor::from_vec(w.ty.clone(), dw)?))
}

#[allow(clippy::too_many_arguments)]
fn yrow_dx(yrow: &[f32], row: &[f32], w: &Tensor, inv: f32, dms: f32, n: f32, i: usize, out: &mut f32) {
    *out = yrow[i] * inv * w.data[i] + dms * (2.0 * row[i] / n);
}

pub fn embedding_backward(
    ids: &[u32],
    table: &Tensor,
    dy: &Tensor,
) -> CatResult<Tensor> {
    let hidden = table.ty.shape.dims[1];
    let mut dw = Tensor::zeros(table.ty.clone());
    for (t, &id) in ids.iter().enumerate() {
        let off = id as usize * hidden;
        let go = t * hidden;
        for i in 0..hidden {
            dw.data[off + i] += dy.data[go + i];
        }
    }
    Ok(dw)
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

fn silu_grad(x: f32) -> f32 {
    let s = 1.0 / (1.0 + (-x).exp());
    s + x * s * (1.0 - s)
}

pub fn swiglu_backward(
    x: &Tensor,
    w1: &Tensor,
    w3: &Tensor,
    w2: &Tensor,
    dy: &Tensor,
) -> CatResult<(Tensor, Tensor, Tensor, Tensor)> {
    let hidden = *x.ty.shape.dims.last().unwrap();
    let tokens = x.data.len() / hidden;
    let mid = w1.ty.shape.dims[1];
    let xf = Tensor::from_vec(TensorType::f32([tokens, hidden]), x.data.clone())?;
    let a = cpu::linear_seq(&xf, w1, None)?;
    let c = cpu::linear_seq(&xf, w3, None)?;
    let mut h = a.clone();
    for i in 0..h.data.len() {
        h.data[i] = silu(a.data[i]) * c.data[i];
    }
    h.ty = TensorType::f32([tokens, mid]);
    let dyf = Tensor::from_vec(TensorType::f32([tokens, hidden]), dy.data.clone())?;
    let (dh, dw2, _) = linear_backward(&h, w2, &dyf)?;
    let mut da = Tensor::zeros(a.ty.clone());
    let mut dc = Tensor::zeros(c.ty.clone());
    for i in 0..h.data.len() {
        da.data[i] = dh.data[i] * c.data[i] * silu_grad(a.data[i]);
        dc.data[i] = dh.data[i] * silu(a.data[i]);
    }
    let (dx1, dw1, _) = linear_backward(&xf, w1, &da)?;
    let (dx3, dw3, _) = linear_backward(&xf, w3, &dc)?;
    let mut dx = dx1;
    for (o, v) in dx.data.iter_mut().zip(dx3.data.iter()) {
        *o += *v;
    }
    dx.ty = x.ty.clone();
    Ok((dx, dw1, dw3, dw2))
}

/// Router adjoint. Discrete top-k indices are **stop-gradient**; the
/// softmax weights of selected experts carry the gradient (STE).
pub fn router_backward_ste(
    x: &Tensor,
    gate: &Tensor,
    top_k: usize,
    d_weights: &[f32],
) -> CatResult<(Tensor, Tensor)> {
    let hidden = *x.ty.shape.dims.last().unwrap();
    let n_exp = gate.ty.shape.dims[1];
    let tokens = x.data.len() / hidden;
    let xf = Tensor::from_vec(TensorType::f32([tokens, hidden]), x.data.clone())?;
    let logits = cpu::linear_seq(&xf, gate, None)?;
    let mut dlogits = Tensor::zeros(logits.ty.clone());
    for t in 0..tokens {
        let row = &logits.data[t * n_exp..(t + 1) * n_exp];
        let (idx, wts) = catnn_core::lens::moe::topk_softmax(row, top_k);
        // Softmax Jacobian on the selected subset, treating unselected as 0.
        let mut dsel = vec![0.0f32; top_k];
        for k in 0..top_k {
            dsel[k] = d_weights[t * top_k + k];
        }
        let dot: f32 = wts.iter().zip(dsel.iter()).map(|(w, g)| w * g).sum();
        for k in 0..top_k {
            dlogits.data[t * n_exp + idx[k]] += wts[k] * (dsel[k] - dot);
        }
    }
    let (dx, dg, _) = linear_backward(&xf, gate, &dlogits)?;
    let mut dxi = dx;
    dxi.ty = x.ty.clone();
    Ok((dxi, dg))
}

#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
pub fn gqa_backward(
    x: &Tensor,
    wq: &Tensor,
    wk: &Tensor,
    wv: &Tensor,
    wo: &Tensor,
    n_heads: usize,
    n_kv: usize,
    head_dim: usize,
    rope_theta: f64,
    sliding_window: Option<usize>,
    dy: &Tensor,
) -> CatResult<(Tensor, Tensor, Tensor, Tensor, Tensor)> {
    // Recompute the forward intermediates, then walk the adjoint.
    let dims = &x.ty.shape.dims;
    let (b, s, d) = (dims[0], dims[1], dims[2]);
    let q = linear3(x, wq, b, s, d, n_heads * head_dim)?;
    let k = linear3(x, wk, b, s, d, n_kv * head_dim)?;
    let v = linear3(x, wv, b, s, d, n_kv * head_dim)?;
    let mut q = q.data;
    let mut k = k.data;
    let repeats = n_heads / n_kv.max(1);
    for bi in 0..b {
        for t in 0..s {
            for h in 0..n_heads {
                let off = ((bi * s + t) * n_heads + h) * head_dim;
                apply_rope(&mut q[off..off + head_dim], head_dim, t, rope_theta);
            }
            for h in 0..n_kv {
                let off = ((bi * s + t) * n_kv + h) * head_dim;
                apply_rope(&mut k[off..off + head_dim], head_dim, t, rope_theta);
            }
        }
    }
    let scale = 1.0 / (head_dim as f32).sqrt();
    // ctx, scores
    let mut ctx = vec![0.0f32; b * s * n_heads * head_dim];
    let mut attn = vec![0.0f32; b * n_heads * s * s];
    for bi in 0..b {
        for h in 0..n_heads {
            let kv_h = h / repeats.max(1);
            for i in 0..s {
                let start = sliding_window.map(|w| i.saturating_sub(w - 1)).unwrap_or(0);
                let q_off = ((bi * s + i) * n_heads + h) * head_dim;
                let mut scores = vec![f32::NEG_INFINITY; s];
                for (j, score) in scores.iter_mut().enumerate().take(i + 1).skip(start) {
                    let k_off = ((bi * s + j) * n_kv + kv_h) * head_dim;
                    let mut dot = 0.0;
                    for t in 0..head_dim {
                        dot += q[q_off + t] * k[k_off + t];
                    }
                    *score = dot * scale;
                }
                let max = scores[start..=i].iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0;
                for score in scores.iter_mut().take(i + 1).skip(start) {
                    *score = (*score - max).exp();
                    sum += *score;
                }
                let inv = 1.0 / sum.max(1e-30);
                for (j, score) in scores.iter_mut().enumerate().take(i + 1).skip(start) {
                    *score *= inv;
                    attn[((bi * n_heads + h) * s + i) * s + j] = *score;
                }
                let o_off = ((bi * s + i) * n_heads + h) * head_dim;
                for (j, score) in scores.iter().enumerate().take(i + 1).skip(start) {
                    let v_off = ((bi * s + j) * n_kv + kv_h) * head_dim;
                    for t in 0..head_dim {
                        ctx[o_off + t] += *score * v.data[v_off + t];
                    }
                }
            }
        }
    }
    let ctx_t = Tensor::from_vec(TensorType::f32([b, s, n_heads * head_dim]), ctx.clone())?;
    let dyf = Tensor::from_vec(TensorType::f32([b * s, d]), dy.data.clone())?;
    let ctx_flat = Tensor::from_vec(TensorType::f32([b * s, n_heads * head_dim]), ctx)?;
    let (dctx, dwo, _) = linear_backward(&ctx_flat, wo, &dyf)?;

    let mut dq = vec![0.0f32; q.len()];
    let mut dk = vec![0.0f32; k.len()];
    let mut dv = vec![0.0f32; v.data.len()];
    for bi in 0..b {
        for h in 0..n_heads {
            let kv_h = h / repeats.max(1);
            for i in 0..s {
                let start = sliding_window.map(|w| i.saturating_sub(w - 1)).unwrap_or(0);
                let o_off = ((bi * s + i) * n_heads + h) * head_dim;
                let mut dscore = vec![0.0f32; s];
                for j in start..=i {
                    let v_off = ((bi * s + j) * n_kv + kv_h) * head_dim;
                    let mut dot = 0.0;
                    for t in 0..head_dim {
                        dot += dctx.data[o_off + t] * v.data[v_off + t];
                    }
                    dscore[j] = dot;
                }
                // softmax backward on the causal window
                let mut dot = 0.0;
                for j in start..=i {
                    let a = attn[((bi * n_heads + h) * s + i) * s + j];
                    dot += a * dscore[j];
                }
                for j in start..=i {
                    let a = attn[((bi * n_heads + h) * s + i) * s + j];
                    dscore[j] = a * (dscore[j] - dot);
                }
                let q_off = ((bi * s + i) * n_heads + h) * head_dim;
                for j in start..=i {
                    let k_off = ((bi * s + j) * n_kv + kv_h) * head_dim;
                    let v_off = ((bi * s + j) * n_kv + kv_h) * head_dim;
                    let a = attn[((bi * n_heads + h) * s + i) * s + j];
                    for t in 0..head_dim {
                        dv[v_off + t] += a * dctx.data[o_off + t];
                        dq[q_off + t] += dscore[j] * scale * k[k_off + t];
                        dk[k_off + t] += dscore[j] * scale * q[q_off + t];
                    }
                }
            }
        }
    }
    // RoPE adjoint = rotate the incoming cotangent by −θ.
    for bi in 0..b {
        for tpos in 0..s {
            for h in 0..n_heads {
                let off = ((bi * s + tpos) * n_heads + h) * head_dim;
                apply_rope(&mut dq[off..off + head_dim], head_dim, tpos, -rope_theta);
            }
            for h in 0..n_kv {
                let off = ((bi * s + tpos) * n_kv + h) * head_dim;
                apply_rope(&mut dk[off..off + head_dim], head_dim, tpos, -rope_theta);
            }
        }
    }

    let xf = Tensor::from_vec(TensorType::f32([b * s, d]), x.data.clone())?;
    let dq_t = Tensor::from_vec(TensorType::f32([b * s, n_heads * head_dim]), dq)?;
    let dk_t = Tensor::from_vec(TensorType::f32([b * s, n_kv * head_dim]), dk)?;
    let dv_t = Tensor::from_vec(TensorType::f32([b * s, n_kv * head_dim]), dv)?;
    let (dxq, dwq, _) = linear_backward(&xf, wq, &dq_t)?;
    let (dxk, dwk, _) = linear_backward(&xf, wk, &dk_t)?;
    let (dxv, dwv, _) = linear_backward(&xf, wv, &dv_t)?;
    let mut dx = dxq;
    for (o, v) in dx.data.iter_mut().zip(dxk.data.iter()) {
        *o += *v;
    }
    for (o, v) in dx.data.iter_mut().zip(dxv.data.iter()) {
        *o += *v;
    }
    dx.ty = x.ty.clone();
    let _ = ctx_t;
    Ok((dx, dwq, dwk, dwv, dwo))
}

fn linear3(
    x: &Tensor,
    w: &Tensor,
    b: usize,
    s: usize,
    din: usize,
    dout: usize,
) -> CatResult<Tensor> {
    let flat = Tensor::from_vec(TensorType::f32([b * s, din]), x.data.clone())?;
    let y = cpu::linear_seq(&flat, w, None)?;
    Tensor::from_vec(TensorType::f32([b, s, dout]), y.data)
}

fn apply_rope(q: &mut [f32], head_dim: usize, pos: usize, theta: f64) {
    for i in 0..head_dim / 2 {
        let freq = 1.0 / theta.abs().powf(2.0 * i as f64 / head_dim as f64);
        let ang = pos as f64 * freq;
        let (s, c) = ang.sin_cos();
        let (s, c) = if theta < 0.0 { (-s as f32, c as f32) } else { (s as f32, c as f32) };
        let i0 = 2 * i;
        let i1 = i0 + 1;
        let (a, b) = (q[i0], q[i1]);
        q[i0] = a * c - b * s;
        q[i1] = a * s + b * c;
    }
}

/// Expert coproduct adjoint: each selected SwiGLU gets `w_k · dy`;
/// `dx` sums `w_k · dx_e` plus the router-weight cotangent.
#[allow(clippy::type_complexity, clippy::needless_range_loop)]
pub fn moe_backward(
    x: &Tensor,
    gate: &Tensor,
    experts: &[(Tensor, Tensor, Tensor)],
    top_k: usize,
    dy: &Tensor,
) -> CatResult<(Tensor, Tensor, Vec<(Tensor, Tensor, Tensor)>)> { // expert (w1,w3,w2) grads
    let hidden = *x.ty.shape.dims.last().unwrap();
    let tokens = x.data.len() / hidden;
    let (idx, wts, _) = moe::route(x, gate, top_k)?;
    let mut dx = Tensor::zeros(x.ty.clone());
    let mut d_wts = vec![0.0f32; tokens * top_k];
    let mut dex: Vec<(Tensor, Tensor, Tensor)> = experts
        .iter()
        .map(|(a, b, c)| {
            (Tensor::zeros(a.ty.clone()), Tensor::zeros(b.ty.clone()), Tensor::zeros(c.ty.clone()))
        })
        .collect();
    for t in 0..tokens {
        let xt = Tensor::from_vec(
            TensorType::f32([1, hidden]),
            x.data[t * hidden..(t + 1) * hidden].to_vec(),
        )?;
        let dyt = Tensor::from_vec(
            TensorType::f32([1, hidden]),
            dy.data[t * hidden..(t + 1) * hidden].to_vec(),
        )?;
        for k in 0..top_k {
            let e = idx[t * top_k + k];
            let wt = wts[t * top_k + k];
            let (w1, w3, w2) = &experts[e];
            let y = moe::swiglu(&xt, w1, w3, w2)?;
            d_wts[t * top_k + k] = y.data.iter().zip(dyt.data.iter()).map(|(a, b)| a * b).sum();
            let dye = scale(&dyt, wt);
            let (dxe, dw1, dw3, dw2) = swiglu_backward(&xt, w1, w3, w2, &dye)?;
            for i in 0..hidden {
                dx.data[t * hidden + i] += dxe.data[i];
            }
            for (o, v) in dex[e].0.data.iter_mut().zip(dw1.data.iter()) {
                *o += *v;
            }
            for (o, v) in dex[e].1.data.iter_mut().zip(dw3.data.iter()) {
                *o += *v;
            }
            for (o, v) in dex[e].2.data.iter_mut().zip(dw2.data.iter()) {
                *o += *v;
            }
        }
    }
    let (dx_gate, dgate) = router_backward_ste(x, gate, top_k, &d_wts)?;
    for (o, v) in dx.data.iter_mut().zip(dx_gate.data.iter()) {
        *o += *v;
    }
    Ok((dx, dgate, dex))
}

/// Finite-difference check of a scalar loss `f` against an analytic grad.
pub fn finite_diff_report(
    p: &Tensor,
    analytic: &Tensor,
    mut f: impl FnMut(&Tensor) -> CatResult<f32>,
    eps: f32,
    n_probe: usize,
) -> CatResult<f32> {
    let mut max_rel = 0.0f32;
    let step = (p.data.len() / n_probe.max(1)).max(1);
    for i in (0..p.data.len()).step_by(step).take(n_probe) {
        let mut plus = p.clone();
        let mut minus = p.clone();
        plus.data[i] += eps;
        minus.data[i] -= eps;
        let num = (f(&plus)? - f(&minus)?) / (2.0 * eps);
        let den = analytic.data[i].abs().max(1e-4);
        max_rel = max_rel.max((num - analytic.data[i]).abs() / den);
    }
    Ok(max_rel)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_matches_finite_diff() {
        let x = Tensor::lcg(TensorType::f32([2, 3]), 1);
        let w = Tensor::lcg(TensorType::f32([3, 2]), 2);
        let y = cpu::linear(&x, &w, None).unwrap();
        let dy = Tensor::ones(y.ty.clone());
        let (_dx, dw, _) = linear_backward(&x, &w, &dy).unwrap();
        let rel = finite_diff_report(&w, &dw, |wp| {
            let yp = cpu::linear(&x, wp, None)?;
            Ok(yp.data.iter().sum())
        }, 1e-3, 4)
        .unwrap();
        assert!(rel < 0.05, "rel {rel}");
    }

    #[test]
    fn relu_zeros_negatives() {
        let x = Tensor::from_vec(TensorType::f32([3]), vec![-1.0, 2.0, 0.0]).unwrap();
        let dy = Tensor::ones(x.ty.clone());
        let dx = relu_backward(&x, &dy).unwrap();
        assert_eq!(dx.data, vec![0.0, 1.0, 0.0]);
    }

    #[test]
    fn rmsnorm_matches_finite_diff() {
        let x = Tensor::lcg(TensorType::f32([2, 4]), 3);
        let w = Tensor::ones(TensorType::f32([4]));
        let y = crate::kernel::moe::rmsnorm(&x, &w, 1e-5).unwrap();
        let dy = Tensor::ones(y.ty.clone());
        let (_dx, dw) = rmsnorm_backward(&x, &w, &dy, 1e-5).unwrap();
        let rel = finite_diff_report(
            &w,
            &dw,
            |wp| {
                let yp = crate::kernel::moe::rmsnorm(&x, wp, 1e-5)?;
                Ok(yp.data.iter().sum())
            },
            1e-3,
            4,
        )
        .unwrap();
        assert!(rel < 0.08, "rel {rel}");
    }

    #[test]
    fn softmax_row_sums_to_zero_grad() {
        let z = Tensor::from_vec(TensorType::f32([1, 3]), vec![1.0, 2.0, 3.0]).unwrap();
        let y = cpu::softmax_last(&z).unwrap();
        let dy = Tensor::ones(y.ty.clone());
        let dx = softmax_backward(&y, &dy).unwrap();
        let s: f32 = dx.data.iter().sum();
        assert!(s.abs() < 1e-5, "{s}");
    }
}
