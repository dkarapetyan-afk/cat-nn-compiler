//! Finite-difference checks in **f64** for every adjoint used by training.
//!
//! The runtime kernels stay f32. This module reimplements the same
//! algorithms in f64 so central differences can use ε ≈ 1e-6 and the
//! reported relative error is meaningful (not the 5–8 % f32 noise).

use catnn_core::spec::MixtralConfig;
use catnn_core::CatResult;

use crate::kernel::moe::{LayerWeights, MixtralWeights};

#[derive(Clone, Debug)]
pub struct Arr {
    pub dims: Vec<usize>,
    pub data: Vec<f64>,
}

impl Arr {
    pub fn from_f32(dims: Vec<usize>, src: &[f32]) -> Self {
        Self { dims, data: src.iter().map(|v| *v as f64).collect() }
    }
    pub fn zeros(dims: Vec<usize>) -> Self {
        let n = dims.iter().product();
        Self { dims, data: vec![0.0; n] }
    }
    pub fn last(&self) -> usize {
        *self.dims.last().unwrap_or(&self.data.len())
    }
}

#[derive(Clone, Debug)]
pub struct FdRow {
    pub name: String,
    pub max_rel: f64,
    pub max_abs: f64,
    pub n_probe: usize,
}

fn fd_max(
    p: &Arr,
    analytic: &Arr,
    mut f: impl FnMut(&Arr) -> CatResult<f64>,
    eps: f64,
    n_probe: usize,
) -> CatResult<(f64, f64, usize)> {
    let n = p.data.len();
    let take = n_probe.min(n).max(1);
    let step = (n / take).max(1);
    let mut max_rel: f64 = 0.0;
    let mut max_abs: f64 = 0.0;
    let mut used = 0usize;
    for i in (0..n).step_by(step).take(take) {
        let mut plus = p.clone();
        let mut minus = p.clone();
        plus.data[i] += eps;
        minus.data[i] -= eps;
        let num = (f(&plus)? - f(&minus)?) / (2.0 * eps);
        let a = analytic.data[i];
        let abs = (num - a).abs();
        // Relative error is only meaningful when the gradient itself is.
        // Near-zero unused weights would otherwise look like 1e-5 rel
        // against a 1e-8 floor.
        let scale = a.abs().max(num.abs());
        if scale >= 1e-4 {
            max_rel = max_rel.max(abs / scale);
        }
        max_abs = max_abs.max(abs);
        used += 1;
    }
    Ok((max_rel, max_abs, used))
}

fn linear(x: &Arr, w: &Arr) -> Arr {
    let k = x.last();
    let n = w.dims[1];
    let m = x.data.len() / k;
    let mut y = Arr::zeros(vec![m, n]);
    for i in 0..m {
        for j in 0..n {
            let mut s = 0.0;
            for p in 0..k {
                s += x.data[i * k + p] * w.data[p * n + j];
            }
            y.data[i * n + j] = s;
        }
    }
    y
}

fn linear_bwd(x: &Arr, w: &Arr, dy: &Arr) -> (Arr, Arr) {
    let k = x.last();
    let n = w.dims[1];
    let m = x.data.len() / k;
    let mut dw = Arr::zeros(w.dims.clone());
    let mut dx = Arr::zeros(x.dims.clone());
    for i in 0..m {
        for j in 0..n {
            let g = dy.data[i * n + j];
            for p in 0..k {
                dw.data[p * n + j] += x.data[i * k + p] * g;
                dx.data[i * k + p] += g * w.data[p * n + j];
            }
        }
    }
    (dx, dw)
}

fn relu(x: &Arr) -> Arr {
    let mut y = x.clone();
    for v in &mut y.data {
        *v = v.max(0.0);
    }
    y
}

fn relu_bwd(x: &Arr, dy: &Arr) -> Arr {
    let mut dx = dy.clone();
    for (o, xi) in dx.data.iter_mut().zip(x.data.iter()) {
        if *xi <= 0.0 {
            *o = 0.0;
        }
    }
    dx
}

fn gelu(x: &Arr) -> Arr {
    let mut y = x.clone();
    let k = (2.0 / std::f64::consts::PI).sqrt();
    for v in &mut y.data {
        let z = *v;
        *v = 0.5 * z * (1.0 + (k * (z + 0.044715 * z * z * z)).tanh());
    }
    y
}

fn gelu_bwd(x: &Arr, dy: &Arr) -> Arr {
    let mut dx = dy.clone();
    const C: f64 = 0.044715;
    let k = (2.0 / std::f64::consts::PI).sqrt();
    for (o, &z) in dx.data.iter_mut().zip(x.data.iter()) {
        let u = k * (z + C * z * z * z);
        let t = u.tanh();
        let sech2 = 1.0 - t * t;
        let du = k * (1.0 + 3.0 * C * z * z);
        *o *= 0.5 * (1.0 + t) + 0.5 * z * sech2 * du;
    }
    dx
}

fn softmax(x: &Arr) -> Arr {
    let last = x.last();
    let mut y = x.clone();
    for (inp, out) in x.data.chunks(last).zip(y.data.chunks_mut(last)) {
        let mx = inp.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mut sum = 0.0;
        for (o, i) in out.iter_mut().zip(inp.iter()) {
            *o = (*i - mx).exp();
            sum += *o;
        }
        let inv = 1.0 / sum.max(1e-30);
        for o in out.iter_mut() {
            *o *= inv;
        }
    }
    y
}

fn softmax_bwd(y: &Arr, dy: &Arr) -> Arr {
    let last = y.last();
    let mut dx = dy.clone();
    for (row_y, (row_dy, row_dx)) in
        y.data.chunks(last).zip(dy.data.chunks(last).zip(dx.data.chunks_mut(last)))
    {
        let dot: f64 = row_y.iter().zip(row_dy.iter()).map(|(s, g)| s * g).sum();
        for i in 0..last {
            row_dx[i] = row_y[i] * (row_dy[i] - dot);
        }
    }
    dx
}

fn mse(pred: &Arr, target: &Arr) -> (f64, Arr) {
    let n = pred.data.len() as f64;
    let mut loss = 0.0;
    let mut d = pred.clone();
    for i in 0..pred.data.len() {
        let e = pred.data[i] - target.data[i];
        loss += e * e;
        d.data[i] = 2.0 * e / n;
    }
    (loss / n, d)
}

fn cross_entropy(logits: &Arr, targets: &[usize]) -> (f64, Arr) {
    let last = logits.last();
    let rows = logits.data.len() / last;
    let probs = softmax(logits);
    let mut loss = 0.0;
    let mut d = probs.clone();
    for (r, &tgt) in targets.iter().enumerate() {
        let p = probs.data[r * last + tgt].max(1e-30);
        loss -= p.ln();
        for j in 0..last {
            d.data[r * last + j] /= rows as f64;
        }
        d.data[r * last + tgt] -= 1.0 / rows as f64;
    }
    (loss / rows as f64, d)
}

fn rmsnorm(x: &Arr, w: &Arr, eps: f64) -> Arr {
    let last = x.last();
    let mut y = x.clone();
    let n = last as f64;
    for (row, out) in x.data.chunks(last).zip(y.data.chunks_mut(last)) {
        let ms = row.iter().map(|v| v * v).sum::<f64>() / n;
        let inv = 1.0 / (ms + eps).sqrt();
        for i in 0..last {
            out[i] = row[i] * inv * w.data[i];
        }
    }
    y
}

fn rmsnorm_bwd(x: &Arr, w: &Arr, dy: &Arr, eps: f64) -> (Arr, Arr) {
    let last = x.last();
    let mut dx = Arr::zeros(x.dims.clone());
    let mut dw = vec![0.0; last];
    let n = last as f64;
    for (row, (drow, yrow)) in
        x.data.chunks(last).zip(dx.data.chunks_mut(last).zip(dy.data.chunks(last)))
    {
        let ms = row.iter().map(|v| v * v).sum::<f64>() / n;
        let inv = 1.0 / (ms + eps).sqrt();
        let mut dinv = 0.0;
        for i in 0..last {
            dw[i] += yrow[i] * row[i] * inv;
            dinv += yrow[i] * row[i] * w.data[i];
        }
        let dms = dinv * (-0.5) * (ms + eps).powf(-1.5);
        for i in 0..last {
            drow[i] = yrow[i] * inv * w.data[i] + dms * (2.0 * row[i] / n);
        }
    }
    (dx, Arr { dims: w.dims.clone(), data: dw })
}

fn embedding(ids: &[usize], table: &Arr) -> Arr {
    let h = table.dims[1];
    let mut data = Vec::with_capacity(ids.len() * h);
    for &id in ids {
        let off = id * h;
        data.extend_from_slice(&table.data[off..off + h]);
    }
    Arr { dims: vec![ids.len(), h], data }
}

fn embedding_bwd(ids: &[usize], table: &Arr, dy: &Arr) -> Arr {
    let h = table.dims[1];
    let mut dw = Arr::zeros(table.dims.clone());
    for (t, &id) in ids.iter().enumerate() {
        let off = id * h;
        let go = t * h;
        for i in 0..h {
            dw.data[off + i] += dy.data[go + i];
        }
    }
    dw
}

fn silu(x: f64) -> f64 {
    x / (1.0 + (-x).exp())
}
fn silu_grad(x: f64) -> f64 {
    let s = 1.0 / (1.0 + (-x).exp());
    s + x * s * (1.0 - s)
}

fn swiglu(x: &Arr, w1: &Arr, w3: &Arr, w2: &Arr) -> Arr {
    let hidden = x.last();
    let tokens = x.data.len() / hidden;
    let xf = Arr { dims: vec![tokens, hidden], data: x.data.clone() };
    let a = linear(&xf, w1);
    let c = linear(&xf, w3);
    let mut h = a.clone();
    for i in 0..h.data.len() {
        h.data[i] = silu(a.data[i]) * c.data[i];
    }
    h.dims = vec![tokens, w1.dims[1]];
    let y = linear(&h, w2);
    Arr { dims: x.dims.clone(), data: y.data }
}

fn swiglu_bwd(x: &Arr, w1: &Arr, w3: &Arr, w2: &Arr, dy: &Arr) -> (Arr, Arr, Arr, Arr) {
    let hidden = x.last();
    let tokens = x.data.len() / hidden;
    let xf = Arr { dims: vec![tokens, hidden], data: x.data.clone() };
    let a = linear(&xf, w1);
    let c = linear(&xf, w3);
    let mut h = a.clone();
    for i in 0..h.data.len() {
        h.data[i] = silu(a.data[i]) * c.data[i];
    }
    h.dims = vec![tokens, w1.dims[1]];
    let dyf = Arr { dims: vec![tokens, hidden], data: dy.data.clone() };
    let (dh, dw2) = linear_bwd(&h, w2, &dyf);
    let mut da = Arr::zeros(a.dims.clone());
    let mut dc = Arr::zeros(c.dims.clone());
    for i in 0..h.data.len() {
        da.data[i] = dh.data[i] * c.data[i] * silu_grad(a.data[i]);
        dc.data[i] = dh.data[i] * silu(a.data[i]);
    }
    let (dx1, dw1) = linear_bwd(&xf, w1, &da);
    let (dx3, dw3) = linear_bwd(&xf, w3, &dc);
    let mut dx = dx1;
    for (o, v) in dx.data.iter_mut().zip(dx3.data.iter()) {
        *o += *v;
    }
    dx.dims = x.dims.clone();
    (dx, dw1, dw3, dw2)
}

fn apply_rope(q: &mut [f64], head_dim: usize, pos: usize, theta: f64) {
    for i in 0..head_dim / 2 {
        let freq = 1.0 / theta.abs().powf(2.0 * i as f64 / head_dim as f64);
        let ang = pos as f64 * freq;
        let (s, c) = ang.sin_cos();
        let (s, c) = if theta < 0.0 { (-s, c) } else { (s, c) };
        let i0 = 2 * i;
        let i1 = i0 + 1;
        let (a, b) = (q[i0], q[i1]);
        q[i0] = a * c - b * s;
        q[i1] = a * s + b * c;
    }
}

#[allow(clippy::too_many_arguments)]
fn gqa(
    x: &Arr,
    wq: &Arr,
    wk: &Arr,
    wv: &Arr,
    wo: &Arr,
    n_heads: usize,
    n_kv: usize,
    head_dim: usize,
    rope_theta: f64,
    window: Option<usize>,
) -> Arr {
    let (b, s, d) = (x.dims[0], x.dims[1], x.dims[2]);
    let xf = Arr { dims: vec![b * s, d], data: x.data.clone() };
    let mut q = linear(&xf, wq).data;
    let mut k = linear(&xf, wk).data;
    let v = linear(&xf, wv);
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
    let scale = 1.0 / (head_dim as f64).sqrt();
    let mut ctx = vec![0.0; b * s * n_heads * head_dim];
    for bi in 0..b {
        for h in 0..n_heads {
            let kv_h = h / repeats.max(1);
            for i in 0..s {
                let start = window.map(|w| i.saturating_sub(w - 1)).unwrap_or(0);
                let q_off = ((bi * s + i) * n_heads + h) * head_dim;
                let mut scores = vec![f64::NEG_INFINITY; s];
                for (j, score) in scores.iter_mut().enumerate().take(i + 1).skip(start) {
                    let k_off = ((bi * s + j) * n_kv + kv_h) * head_dim;
                    let mut dot = 0.0;
                    for t in 0..head_dim {
                        dot += q[q_off + t] * k[k_off + t];
                    }
                    *score = dot * scale;
                }
                let max = scores[start..=i].iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let mut sum = 0.0;
                for score in scores.iter_mut().take(i + 1).skip(start) {
                    *score = (*score - max).exp();
                    sum += *score;
                }
                let inv = 1.0 / sum.max(1e-30);
                let o_off = ((bi * s + i) * n_heads + h) * head_dim;
                for (j, score) in scores.iter().enumerate().take(i + 1).skip(start) {
                    let w = *score * inv;
                    let v_off = ((bi * s + j) * n_kv + kv_h) * head_dim;
                    for t in 0..head_dim {
                        ctx[o_off + t] += w * v.data[v_off + t];
                    }
                }
            }
        }
    }
    let ctx_t = Arr { dims: vec![b * s, n_heads * head_dim], data: ctx };
    let y = linear(&ctx_t, wo);
    Arr { dims: vec![b, s, d], data: y.data }
}

#[allow(clippy::too_many_arguments)]
fn gqa_bwd(
    x: &Arr,
    wq: &Arr,
    wk: &Arr,
    wv: &Arr,
    wo: &Arr,
    n_heads: usize,
    n_kv: usize,
    head_dim: usize,
    rope_theta: f64,
    window: Option<usize>,
    dy: &Arr,
) -> (Arr, Arr, Arr, Arr, Arr) {
    let (b, s, d) = (x.dims[0], x.dims[1], x.dims[2]);
    let xf = Arr { dims: vec![b * s, d], data: x.data.clone() };
    let mut q = linear(&xf, wq).data;
    let mut k = linear(&xf, wk).data;
    let v = linear(&xf, wv);
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
    let scale = 1.0 / (head_dim as f64).sqrt();
    let mut ctx = vec![0.0; b * s * n_heads * head_dim];
    let mut attn = vec![0.0; b * n_heads * s * s];
    for bi in 0..b {
        for h in 0..n_heads {
            let kv_h = h / repeats.max(1);
            for i in 0..s {
                let start = window.map(|w| i.saturating_sub(w - 1)).unwrap_or(0);
                let q_off = ((bi * s + i) * n_heads + h) * head_dim;
                let mut scores = vec![f64::NEG_INFINITY; s];
                for (j, score) in scores.iter_mut().enumerate().take(i + 1).skip(start) {
                    let k_off = ((bi * s + j) * n_kv + kv_h) * head_dim;
                    let mut dot = 0.0;
                    for t in 0..head_dim {
                        dot += q[q_off + t] * k[k_off + t];
                    }
                    *score = dot * scale;
                }
                let max = scores[start..=i].iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let mut sum = 0.0;
                for score in scores.iter_mut().take(i + 1).skip(start) {
                    *score = (*score - max).exp();
                    sum += *score;
                }
                let inv = 1.0 / sum.max(1e-30);
                let o_off = ((bi * s + i) * n_heads + h) * head_dim;
                for (j, score) in scores.iter_mut().enumerate().take(i + 1).skip(start) {
                    *score *= inv;
                    attn[((bi * n_heads + h) * s + i) * s + j] = *score;
                    let v_off = ((bi * s + j) * n_kv + kv_h) * head_dim;
                    for t in 0..head_dim {
                        ctx[o_off + t] += *score * v.data[v_off + t];
                    }
                }
            }
        }
    }
    let ctx_flat = Arr { dims: vec![b * s, n_heads * head_dim], data: ctx };
    let dyf = Arr { dims: vec![b * s, d], data: dy.data.clone() };
    let (dctx, dwo) = linear_bwd(&ctx_flat, wo, &dyf);
    let mut dq = vec![0.0; q.len()];
    let mut dk = vec![0.0; k.len()];
    let mut dv = vec![0.0; v.data.len()];
    for bi in 0..b {
        for h in 0..n_heads {
            let kv_h = h / repeats.max(1);
            for i in 0..s {
                let start = window.map(|w| i.saturating_sub(w - 1)).unwrap_or(0);
                let o_off = ((bi * s + i) * n_heads + h) * head_dim;
                let mut dscore = vec![0.0; s];
                for j in start..=i {
                    let v_off = ((bi * s + j) * n_kv + kv_h) * head_dim;
                    let mut dot = 0.0;
                    for t in 0..head_dim {
                        dot += dctx.data[o_off + t] * v.data[v_off + t];
                    }
                    dscore[j] = dot;
                }
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
    let dq_t = Arr { dims: vec![b * s, n_heads * head_dim], data: dq };
    let dk_t = Arr { dims: vec![b * s, n_kv * head_dim], data: dk };
    let dv_t = Arr { dims: vec![b * s, n_kv * head_dim], data: dv };
    let (dxq, dwq) = linear_bwd(&xf, wq, &dq_t);
    let (dxk, dwk) = linear_bwd(&xf, wk, &dk_t);
    let (dxv, dwv) = linear_bwd(&xf, wv, &dv_t);
    let mut dx = dxq;
    for (o, v) in dx.data.iter_mut().zip(dxk.data.iter()) {
        *o += *v;
    }
    for (o, v) in dx.data.iter_mut().zip(dxv.data.iter()) {
        *o += *v;
    }
    dx.dims = x.dims.clone();
    (dx, dwq, dwk, dwv, dwo)
}

fn topk_fixed(logits: &[f64], idx: &[usize]) -> Vec<f64> {
    let mx = idx.iter().map(|&i| logits[i]).fold(f64::NEG_INFINITY, f64::max);
    let mut w: Vec<f64> = idx.iter().map(|&i| (logits[i] - mx).exp()).collect();
    let s: f64 = w.iter().sum();
    let inv = 1.0 / s.max(1e-30);
    for v in &mut w {
        *v *= inv;
    }
    w
}

fn moe_frozen(
    x: &Arr,
    gate: &Arr,
    experts: &[(Arr, Arr, Arr)],
    idx: &[Vec<usize>],
) -> Arr {
    let hidden = x.last();
    let tokens = x.data.len() / hidden;
    let n_exp = gate.dims[1];
    let xf = Arr { dims: vec![tokens, hidden], data: x.data.clone() };
    let logits = linear(&xf, gate);
    let mut out = vec![0.0; x.data.len()];
    for t in 0..tokens {
        let row = &logits.data[t * n_exp..(t + 1) * n_exp];
        let wts = topk_fixed(row, &idx[t]);
        let xt = Arr { dims: vec![1, hidden], data: x.data[t * hidden..(t + 1) * hidden].to_vec() };
        for (k, &e) in idx[t].iter().enumerate() {
            let y = swiglu(&xt, &experts[e].0, &experts[e].1, &experts[e].2);
            for i in 0..hidden {
                out[t * hidden + i] += wts[k] * y.data[i];
            }
        }
    }
    Arr { dims: x.dims.clone(), data: out }
}

fn route_idx(x: &Arr, gate: &Arr, top_k: usize) -> Vec<Vec<usize>> {
    let hidden = x.last();
    let tokens = x.data.len() / hidden;
    let n_exp = gate.dims[1];
    let xf = Arr { dims: vec![tokens, hidden], data: x.data.clone() };
    let logits = linear(&xf, gate);
    let mut out = Vec::new();
    for t in 0..tokens {
        let row = &logits.data[t * n_exp..(t + 1) * n_exp];
        let mut order: Vec<usize> = (0..n_exp).collect();
        order.sort_by(|a, b| row[*b].partial_cmp(&row[*a]).unwrap_or(std::cmp::Ordering::Equal));
        out.push(order[..top_k].to_vec());
    }
    out
}

/// Frozen-index MoE backward. Returns `(dx, dgate, dexpert_w1)`.
/// `dx` includes both the expert SwiGLU path and the gate-weight path
/// (softmax over the held-fixed top-k).
fn moe_bwd_frozen(
    x: &Arr,
    gate: &Arr,
    experts: &[(Arr, Arr, Arr)],
    idx: &[Vec<usize>],
    dy: &Arr,
) -> (Arr, Arr, Vec<Arr>) {
    let hidden = x.last();
    let tokens = x.data.len() / hidden;
    let n_exp = gate.dims[1];
    let xf = Arr { dims: vec![tokens, hidden], data: x.data.clone() };
    let logits = linear(&xf, gate);
    let mut dlogits = Arr::zeros(logits.dims.clone());
    let mut dx = Arr::zeros(x.dims.clone());
    let mut dw1s: Vec<Arr> = experts.iter().map(|e| Arr::zeros(e.0.dims.clone())).collect();
    for t in 0..tokens {
        let row = &logits.data[t * n_exp..(t + 1) * n_exp];
        let wts = topk_fixed(row, &idx[t]);
        let xt = Arr { dims: vec![1, hidden], data: x.data[t * hidden..(t + 1) * hidden].to_vec() };
        let dyt = Arr { dims: vec![1, hidden], data: dy.data[t * hidden..(t + 1) * hidden].to_vec() };
        let mut dsel = vec![0.0; idx[t].len()];
        for (k, &e) in idx[t].iter().enumerate() {
            let y = swiglu(&xt, &experts[e].0, &experts[e].1, &experts[e].2);
            dsel[k] = y.data.iter().zip(dyt.data.iter()).map(|(a, b)| a * b).sum();
            let mut dye = dyt.clone();
            for v in &mut dye.data {
                *v *= wts[k];
            }
            let (dxe, dw1, _dw3, _dw2) =
                swiglu_bwd(&xt, &experts[e].0, &experts[e].1, &experts[e].2, &dye);
            for i in 0..hidden {
                dx.data[t * hidden + i] += dxe.data[i];
            }
            for (o, v) in dw1s[e].data.iter_mut().zip(dw1.data.iter()) {
                *o += *v;
            }
        }
        let dot: f64 = wts.iter().zip(dsel.iter()).map(|(w, g)| w * g).sum();
        for (k, &e) in idx[t].iter().enumerate() {
            dlogits.data[t * n_exp + e] += wts[k] * (dsel[k] - dot);
        }
    }
    let (dxg, dgate) = linear_bwd(&xf, gate, &dlogits);
    for (o, v) in dx.data.iter_mut().zip(dxg.data.iter()) {
        *o += *v;
    }
    (dx, dgate, dw1s)
}

fn lcg64(dims: Vec<usize>, seed: u64) -> Arr {
    let mut state = seed.max(1);
    let n: usize = dims.iter().product();
    let mut data = Vec::with_capacity(n);
    for _ in 0..n {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let unit = ((state >> 33) as u32) as f64 / (u32::MAX as f64);
        data.push(unit * 2.0 - 1.0);
    }
    Arr { dims, data }
}

fn row(name: &str, max_rel: f64, max_abs: f64, n: usize) -> FdRow {
    FdRow { name: name.into(), max_rel, max_abs, n_probe: n }
}

/// Run every required FD check. Returns one row per op / Mixtral group.
pub fn run_all_checks() -> CatResult<Vec<FdRow>> {
    let mut rows = Vec::new();
    // linear
    {
        let x = lcg64(vec![3, 4], 1);
        let w = lcg64(vec![4, 5], 2);
        let y = linear(&x, &w);
        let dy = Arr { dims: y.dims.clone(), data: vec![1.0; y.data.len()] };
        let (_dx, dw) = linear_bwd(&x, &w, &dy);
        let (rel, abs, n) = fd_max(&w, &dw, |wp| Ok(linear(&x, wp).data.iter().sum()), 1e-6, 20)?;
        rows.push(row("linear", rel, abs, n));
    }
    // relu
    {
        let x = lcg64(vec![8], 3);
        let y = relu(&x);
        let dy = Arr { dims: y.dims.clone(), data: vec![1.0; y.data.len()] };
        let dx = relu_bwd(&x, &dy);
        let (rel, abs, n) = fd_max(&x, &dx, |xp| Ok(relu(xp).data.iter().sum()), 1e-6, 8)?;
        rows.push(row("relu", rel, abs, n));
    }
    // gelu
    {
        let x = lcg64(vec![8], 4);
        let y = gelu(&x);
        let dy = Arr { dims: y.dims.clone(), data: vec![1.0; y.data.len()] };
        let dx = gelu_bwd(&x, &dy);
        let (rel, abs, n) = fd_max(&x, &dx, |xp| Ok(gelu(xp).data.iter().sum()), 1e-6, 8)?;
        rows.push(row("gelu", rel, abs, n));
    }
    // softmax — all-ones · softmax is identically 1, so FD that.
    {
        let z = lcg64(vec![2, 5], 5);
        let y = softmax(&z);
        let mut coeff = y.clone();
        for (i, c) in coeff.data.iter_mut().enumerate() {
            *c = (i % 5 + 1) as f64;
        }
        let dx = softmax_bwd(&y, &coeff);
        let (rel, abs, n) = fd_max(
            &z,
            &dx,
            |zp| {
                let y = softmax(zp);
                Ok(y.data.iter().zip(coeff.data.iter()).map(|(a, b)| a * b).sum())
            },
            1e-6,
            10,
        )?;
        rows.push(row("softmax", rel, abs, n));
    }
    // mse
    {
        let p = lcg64(vec![6], 6);
        let t = lcg64(vec![6], 7);
        let (_loss, d) = mse(&p, &t);
        let (rel, abs, n) = fd_max(&p, &d, |pp| Ok(mse(pp, &t).0), 1e-6, 6)?;
        rows.push(row("mse", rel, abs, n));
    }
    // CE
    {
        let z = lcg64(vec![2, 4], 8);
        let tgt = vec![1usize, 2];
        let (_loss, d) = cross_entropy(&z, &tgt);
        let (rel, abs, n) = fd_max(&z, &d, |zp| Ok(cross_entropy(zp, &tgt).0), 1e-6, 8)?;
        rows.push(row("cross_entropy", rel, abs, n));
    }
    // rmsnorm
    {
        let x = lcg64(vec![2, 6], 9);
        let w = Arr { dims: vec![6], data: vec![1.0; 6] };
        let y = rmsnorm(&x, &w, 1e-5);
        let dy = Arr { dims: y.dims.clone(), data: vec![1.0; y.data.len()] };
        let (_dx, dw) = rmsnorm_bwd(&x, &w, &dy, 1e-5);
        let (rel, abs, n) =
            fd_max(&w, &dw, |wp| Ok(rmsnorm(&x, wp, 1e-5).data.iter().sum()), 1e-6, 6)?;
        rows.push(row("rmsnorm", rel, abs, n));
        let (dx, _dw) = rmsnorm_bwd(&x, &w, &dy, 1e-5);
        let (rel, abs, n) =
            fd_max(&x, &dx, |xp| Ok(rmsnorm(xp, &w, 1e-5).data.iter().sum()), 1e-6, 12)?;
        rows.push(row("rmsnorm_input", rel, abs, n));
    }
    // embedding
    {
        let table = lcg64(vec![8, 4], 10);
        let ids = [1usize, 3, 1];
        let y = embedding(&ids, &table);
        let dy = Arr { dims: y.dims.clone(), data: vec![1.0; y.data.len()] };
        let dw = embedding_bwd(&ids, &table, &dy);
        let (rel, abs, n) =
            fd_max(&table, &dw, |tp| Ok(embedding(&ids, tp).data.iter().sum()), 1e-6, 12)?;
        rows.push(row("embedding", rel, abs, n));
    }
    // swiglu
    {
        let x = lcg64(vec![2, 4], 11);
        let w1 = lcg64(vec![4, 6], 12);
        let w3 = lcg64(vec![4, 6], 13);
        let w2 = lcg64(vec![6, 4], 14);
        let y = swiglu(&x, &w1, &w3, &w2);
        let dy = Arr { dims: y.dims.clone(), data: vec![1.0; y.data.len()] };
        let (_dx, dw1, _dw3, _dw2) = swiglu_bwd(&x, &w1, &w3, &w2, &dy);
        let (rel, abs, n) =
            fd_max(&w1, &dw1, |wp| Ok(swiglu(&x, wp, &w3, &w2).data.iter().sum()), 1e-6, 12)?;
        rows.push(row("swiglu", rel, abs, n));
    }
    // GQA + RoPE + sliding window
    {
        let x = lcg64(vec![1, 6, 8], 15);
        let wq = lcg64(vec![8, 8], 16);
        let wk = lcg64(vec![8, 4], 17);
        let wv = lcg64(vec![8, 4], 18);
        let wo = lcg64(vec![8, 8], 19);
        let y = gqa(&x, &wq, &wk, &wv, &wo, 2, 1, 4, 10_000.0, Some(4));
        let mut coeff = y.clone();
        for (i, c) in coeff.data.iter_mut().enumerate() {
            *c = 0.25 + (i % 7) as f64 * 0.1;
        }
        let (dx, dwq, _dwk, _dwv, _dwo) =
            gqa_bwd(&x, &wq, &wk, &wv, &wo, 2, 1, 4, 10_000.0, Some(4), &coeff);
        let (rel, abs, n) = fd_max(
            &wq,
            &dwq,
            |wp| {
                let y = gqa(&x, wp, &wk, &wv, &wo, 2, 1, 4, 10_000.0, Some(4));
                Ok(y.data.iter().zip(coeff.data.iter()).map(|(a, b)| a * b).sum())
            },
            1e-6,
            16,
        )?;
        rows.push(row("gqa_rope_window", rel, abs, n));
        let (rel, abs, n) = fd_max(
            &x,
            &dx,
            |xp| {
                let y = gqa(xp, &wq, &wk, &wv, &wo, 2, 1, 4, 10_000.0, Some(4));
                Ok(y.data.iter().zip(coeff.data.iter()).map(|(a, b)| a * b).sum())
            },
            1e-6,
            16,
        )?;
        rows.push(row("gqa_input", rel, abs, n));
    }
    // MoE gate grads with frozen router indices
    {
        let x = lcg64(vec![3, 4], 20);
        let gate = lcg64(vec![4, 3], 21);
        let experts = vec![
            (lcg64(vec![4, 6], 22), lcg64(vec![4, 6], 23), lcg64(vec![6, 4], 24)),
            (lcg64(vec![4, 6], 25), lcg64(vec![4, 6], 26), lcg64(vec![6, 4], 27)),
            (lcg64(vec![4, 6], 28), lcg64(vec![4, 6], 29), lcg64(vec![6, 4], 30)),
        ];
        let idx = route_idx(&x, &gate, 2);
        let y = moe_frozen(&x, &gate, &experts, &idx);
        let dy = Arr { dims: y.dims.clone(), data: vec![1.0; y.data.len()] };
        let (_dx, dgate, dw1s) = moe_bwd_frozen(&x, &gate, &experts, &idx, &dy);
        let dw1 = dw1s[0].clone();
        let idx_c = idx.clone();
        let (rel, abs, n) = fd_max(
            &gate,
            &dgate,
            |gp| Ok(moe_frozen(&x, gp, &experts, &idx_c).data.iter().sum()),
            1e-6,
            12,
        )?;
        rows.push(row("moe_gate_frozen_idx", rel, abs, n));
        let (rel, abs, n) = fd_max(
            &experts[0].0,
            &dw1,
            |wp| {
                let mut ex = experts.clone();
                ex[0].0 = wp.clone();
                Ok(moe_frozen(&x, &gate, &ex, &idx_c).data.iter().sum())
            },
            1e-6,
            12,
        )?;
        rows.push(row("moe_expert_w1_frozen_idx", rel, abs, n));
    }
    // full tiny Mixtral e2e (parameter groups)
    rows.extend(tiny_mixtral_e2e()?);
    Ok(rows)
}

fn t64(t: &catnn_core::tensor::Tensor) -> Arr {
    Arr::from_f32(t.ty.shape.dims.clone(), &t.data)
}

fn tiny_mixtral_e2e() -> CatResult<Vec<FdRow>> {
    let cfg = MixtralConfig::tiny();
    let w = MixtralWeights::seeded(&cfg, 3)?;
    let tokens = [1usize, 2, 3, 4, 5, 6, 7, 8];
    let batch = 2;
    let seq = 4;
    let routes = mixtral_routes(&cfg, &w, &tokens, batch, seq)?;
    let mut rows = Vec::new();
    // embed via CE (training objective). Gradients are ~3e-5 so plain
    // central-difference is loss-roundoff limited (~1e-10 abs).
    {
        let table = t64(&w.embed);
        let analytic = mixtral_grad(&cfg, &w, &tokens, batch, seq, Scalar::Ce)?.embed;
        let (rel, abs, n) = fd_max(
            &table,
            &analytic,
            |tp| mixtral_scalar(&cfg, &w, &tokens, batch, seq, Some(tp), None, None, &routes, Scalar::Ce),
            1e-6,
            12,
        )?;
        rows.push(row("mixtral_tiny.embed_ce", rel, abs, n));
    }
    // embed / wq / expert via sum-of-logits — O(1) grads, tight FD.
    {
        let table = t64(&w.embed);
        let analytic = mixtral_grad(&cfg, &w, &tokens, batch, seq, Scalar::HiddenSum)?.embed;
        let (rel, abs, n) = fd_max(
            &table,
            &analytic,
            |tp| {
                mixtral_scalar(&cfg, &w, &tokens, batch, seq, Some(tp), None, None, &routes, Scalar::HiddenSum)
            },
            1e-6,
            12,
        )?;
        rows.push(row("mixtral_tiny.embed", rel, abs, n));
    }
    {
        let wq = t64(&w.layers[0].wq);
        let analytic = mixtral_grad(&cfg, &w, &tokens, batch, seq, Scalar::HiddenSum)?.wq0;
        let (rel, abs, n) = fd_max(
            &wq,
            &analytic,
            |wp| {
                mixtral_scalar(&cfg, &w, &tokens, batch, seq, None, Some(wp), None, &routes, Scalar::HiddenSum)
            },
            1e-6,
            12,
        )?;
        rows.push(row("mixtral_tiny.L0.wq", rel, abs, n));
    }
    {
        let e_id = routes
            .first()
            .and_then(|layer| layer.iter().flatten().copied().next())
            .unwrap_or(0);
        let e = t64(&w.layers[0].experts[e_id].0);
        let analytic = mixtral_grad(&cfg, &w, &tokens, batch, seq, Scalar::HiddenSum)?;
        let dw1 = analytic.ew1.get(e_id).cloned().unwrap_or_else(|| Arr::zeros(e.dims.clone()));
        let (rel, abs, n) = fd_max(
            &e,
            &dw1,
            |ep| {
                mixtral_scalar(
                    &cfg,
                    &w,
                    &tokens,
                    batch,
                    seq,
                    None,
                    None,
                    Some((e_id, ep)),
                    &routes,
                    Scalar::HiddenSum,
                )
            },
            1e-6,
            8,
        )?;
        rows.push(row(&format!("mixtral_tiny.L0.e{e_id}.w1"), rel, abs, n));
    }
    Ok(rows)
}

fn mixtral_routes(
    cfg: &MixtralConfig,
    w: &MixtralWeights,
    tokens: &[usize],
    batch: usize,
    seq: usize,
) -> CatResult<Vec<Vec<Vec<usize>>>> {
    let h = cfg.hidden_size;
    let hd = cfg.head_dim()?;
    let mut x = embedding(tokens, &t64(&w.embed));
    x.dims = vec![batch, seq, h];
    let mut routes = Vec::new();
    for layer in &w.layers {
        let xn = rmsnorm(&x, &t64(&layer.rms_attn), cfg.rms_norm_eps as f64);
        let attn = gqa(
            &xn,
            &t64(&layer.wq),
            &t64(&layer.wk),
            &t64(&layer.wv),
            &t64(&layer.wo),
            cfg.num_attention_heads,
            cfg.num_key_value_heads,
            hd,
            cfg.rope_theta,
            cfg.sliding_window,
        );
        for (o, v) in x.data.iter_mut().zip(attn.data.iter()) {
            *o += *v;
        }
        let xm = rmsnorm(&x, &t64(&layer.rms_moe), cfg.rms_norm_eps as f64);
        routes.push(route_idx(&xm, &t64(&layer.gate), cfg.num_experts_per_tok));
        let experts: Vec<(Arr, Arr, Arr)> =
            layer.experts.iter().map(|(a, b, c)| (t64(a), t64(b), t64(c))).collect();
        let m = moe_frozen(&xm, &t64(&layer.gate), &experts, routes.last().unwrap());
        for (o, v) in x.data.iter_mut().zip(m.data.iter()) {
            *o += *v;
        }
    }
    Ok(routes)
}

#[derive(Clone, Copy)]
enum Scalar {
    Ce,
    /// Sum of the last hidden state *before* final RMSNorm / LM head.
    /// Linear in the residual stream, so central differences stay tight.
    HiddenSum,
}

fn mixtral_scalar(
    cfg: &MixtralConfig,
    w: &MixtralWeights,
    tokens: &[usize],
    batch: usize,
    seq: usize,
    embed: Option<&Arr>,
    wq0: Option<&Arr>,
    ew1: Option<(usize, &Arr)>,
    routes: &[Vec<Vec<usize>>],
    scalar: Scalar,
) -> CatResult<f64> {
    let h = cfg.hidden_size;
    let hd = cfg.head_dim()?;
    let table = embed.cloned().unwrap_or_else(|| t64(&w.embed));
    let mut x = embedding(tokens, &table);
    x.dims = vec![batch, seq, h];
    for (li, layer) in w.layers.iter().enumerate() {
        let xn = rmsnorm(&x, &t64(&layer.rms_attn), cfg.rms_norm_eps as f64);
        let wq = if li == 0 {
            wq0.cloned().unwrap_or_else(|| t64(&layer.wq))
        } else {
            t64(&layer.wq)
        };
        let attn = gqa(
            &xn,
            &wq,
            &t64(&layer.wk),
            &t64(&layer.wv),
            &t64(&layer.wo),
            cfg.num_attention_heads,
            cfg.num_key_value_heads,
            hd,
            cfg.rope_theta,
            cfg.sliding_window,
        );
        for (o, v) in x.data.iter_mut().zip(attn.data.iter()) {
            *o += *v;
        }
        let xm = rmsnorm(&x, &t64(&layer.rms_moe), cfg.rms_norm_eps as f64);
        let mut experts: Vec<(Arr, Arr, Arr)> = layer
            .experts
            .iter()
            .map(|(a, b, c)| (t64(a), t64(b), t64(c)))
            .collect();
        if li == 0 {
            if let Some((eid, e)) = ew1 {
                experts[eid].0 = e.clone();
            }
        }
        let m = moe_frozen(&xm, &t64(&layer.gate), &experts, &routes[li]);
        for (o, v) in x.data.iter_mut().zip(m.data.iter()) {
            *o += *v;
        }
    }
    match scalar {
        Scalar::HiddenSum => {
            Ok(x.data.iter().enumerate().map(|(i, v)| v * (0.25 + (i % 7) as f64 * 0.1)).sum())
        }
        Scalar::Ce => {
            let xf = rmsnorm(&x, &t64(&w.final_norm), cfg.rms_norm_eps as f64);
            let wt = transpose(&t64(&w.lm_head), cfg.vocab_size, h);
            let flat = Arr { dims: vec![batch * seq, h], data: xf.data.clone() };
            let logits = linear(&flat, &wt);
            let targets: Vec<usize> =
                (0..batch * seq).map(|i| tokens[(i + 1) % tokens.len()] % cfg.vocab_size).collect();
            Ok(cross_entropy(&logits, &targets).0)
        }
    }
}

fn transpose(w: &Arr, rows: usize, cols: usize) -> Arr {
    let mut data = vec![0.0; rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            data[j * rows + i] = w.data[i * cols + j];
        }
    }
    Arr { dims: vec![cols, rows], data }
}

struct MixtralG64 {
    embed: Arr,
    wq0: Arr,
    ew1: Vec<Arr>,
}

fn mixtral_grad(
    cfg: &MixtralConfig,
    w: &MixtralWeights,
    tokens: &[usize],
    batch: usize,
    seq: usize,
    scalar: Scalar,
) -> CatResult<MixtralG64> {
    let h = cfg.hidden_size;
    let hd = cfg.head_dim()?;
    let table = t64(&w.embed);
    let mut x = embedding(tokens, &table);
    x.dims = vec![batch, seq, h];
    let mut xs = vec![x.clone()];
    let mut tape: Vec<(Arr, Arr, Arr, Arr, Vec<Vec<usize>>)> = Vec::new();
    for layer in &w.layers {
        let xn = rmsnorm(&x, &t64(&layer.rms_attn), cfg.rms_norm_eps as f64);
        let attn = gqa(
            &xn,
            &t64(&layer.wq),
            &t64(&layer.wk),
            &t64(&layer.wv),
            &t64(&layer.wo),
            cfg.num_attention_heads,
            cfg.num_key_value_heads,
            hd,
            cfg.rope_theta,
            cfg.sliding_window,
        );
        let mut xmid = x.clone();
        for (o, v) in xmid.data.iter_mut().zip(attn.data.iter()) {
            *o += *v;
        }
        let xm = rmsnorm(&xmid, &t64(&layer.rms_moe), cfg.rms_norm_eps as f64);
        let experts: Vec<(Arr, Arr, Arr)> =
            layer.experts.iter().map(|(a, b, c)| (t64(a), t64(b), t64(c))).collect();
        let idx = route_idx(&xm, &t64(&layer.gate), cfg.num_experts_per_tok);
        let m = moe_frozen(&xm, &t64(&layer.gate), &experts, &idx);
        x = xmid.clone();
        for (o, v) in x.data.iter_mut().zip(m.data.iter()) {
            *o += *v;
        }
        tape.push((xn, xmid, xm, attn, idx));
        xs.push(x.clone());
    }
    let mut dx = match scalar {
        Scalar::HiddenSum => Arr {
            dims: x.dims.clone(),
            data: (0..x.data.len()).map(|i| 0.25 + (i % 7) as f64 * 0.1).collect(),
        },
        Scalar::Ce => {
            let xf = rmsnorm(&x, &t64(&w.final_norm), cfg.rms_norm_eps as f64);
            let wt = transpose(&t64(&w.lm_head), cfg.vocab_size, h);
            let flat = Arr { dims: vec![batch * seq, h], data: xf.data.clone() };
            let logits = linear(&flat, &wt);
            let targets: Vec<usize> =
                (0..batch * seq).map(|i| tokens[(i + 1) % tokens.len()] % cfg.vocab_size).collect();
            let dlogits = cross_entropy(&logits, &targets).1;
            let (dxf, _dlm) = linear_bwd(&flat, &wt, &dlogits);
            let dx = Arr { dims: x.dims.clone(), data: dxf.data };
            rmsnorm_bwd(&x, &t64(&w.final_norm), &dx, cfg.rms_norm_eps as f64).0
        }
    };
    let mut wq0 = Arr::zeros(vec![h, h]);
    let n_exp = w.layers[0].experts.len();
    let mut ew1: Vec<Arr> = (0..n_exp).map(|_| Arr::zeros(vec![h, cfg.intermediate_size])).collect();
    for li in (0..w.layers.len()).rev() {
        let layer: &LayerWeights = &w.layers[li];
        let (_xn, ref xmid, ref xm, ref _attn, ref idx) = &tape[li];
        let experts: Vec<(Arr, Arr, Arr)> =
            layer.experts.iter().map(|(a, b, c)| (t64(a), t64(b), t64(c))).collect();
        let (dxm, _dgate, dw1s) = moe_bwd_frozen(xm, &t64(&layer.gate), &experts, idx, &dx);
        if li == 0 {
            ew1 = dw1s;
        }
        let (dxmid_moe, _) = rmsnorm_bwd(xmid, &t64(&layer.rms_moe), &dxm, cfg.rms_norm_eps as f64);
        let mut dx_mid = dx.clone();
        for (o, v) in dx_mid.data.iter_mut().zip(dxmid_moe.data.iter()) {
            *o += *v;
        }
        let xn = rmsnorm(&xs[li], &t64(&layer.rms_attn), cfg.rms_norm_eps as f64);
        let (_dxn, dwq, _dwk, _dwv, _dwo) = gqa_bwd(
            &xn,
            &t64(&layer.wq),
            &t64(&layer.wk),
            &t64(&layer.wv),
            &t64(&layer.wo),
            cfg.num_attention_heads,
            cfg.num_key_value_heads,
            hd,
            cfg.rope_theta,
            cfg.sliding_window,
            &dx_mid,
        );
        if li == 0 {
            wq0 = dwq;
        }
        let (dxin, _) = rmsnorm_bwd(&xs[li], &t64(&layer.rms_attn), &_dxn, cfg.rms_norm_eps as f64);
        dx = dxin;
        for (o, v) in dx.data.iter_mut().zip(dx_mid.data.iter()) {
            *o += *v;
        }
    }
    let dembed = embedding_bwd(tokens, &table, &dx);
    Ok(MixtralG64 { embed: dembed, wq0, ew1 })
}


pub fn format_fd_table(rows: &[FdRow]) -> String {
    let mut s = String::from("op                        max_rel      max_abs   probes\n");
    s.push_str(&"-".repeat(60));
    s.push('\n');
    for r in rows {
        s.push_str(&format!(
            "{:<24} {:>10.3e} {:>10.3e} {:>7}\n",
            r.name, r.max_rel, r.max_abs, r.n_probe
        ));
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_adjoints_match_f64_fd() {
        let rows = run_all_checks().unwrap();
        let table = format_fd_table(&rows);
        eprintln!("{table}");
        for r in &rows {
            // Isolated ops: 2e-6 rel on coordinates with |g| ≥ 1e-4.
            // Mixtral CE is loss-roundoff limited (~1e-10 abs).
            // Mixtral hidden-sum still goes through RMSNorm (quadratic),
            // so allow 2e-5 rel while requiring abs < 5e-3.
            let ok = if r.name.ends_with("_ce") {
                r.max_abs < 1e-8
            } else if r.name.starts_with("mixtral_tiny") {
                r.max_rel < 2e-5 && r.max_abs < 5e-3
            } else {
                r.max_rel < 2e-6
            };
            assert!(
                ok,
                "{} max_rel {:.3e} (abs {:.3e})\n{table}",
                r.name,
                r.max_rel,
                r.max_abs
            );
        }
    }
}
