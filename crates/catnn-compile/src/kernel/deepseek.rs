//! DeepSeek-V3 generators: MLA, sigmoid MoE with a shared expert, and a scalar mix.
//!
//! These are the same kernels the diagram lowers to. A mixture is not a
//! second interpreter: both branches are ordinary IR nodes, and `mix` is one more.

use catnn_core::tensor::{Tensor, TensorType};
use catnn_core::{CatError, CatResult};

use crate::kernel::cpu::linear_seq;
use crate::kernel::moe::swiglu;

fn flat(x: &Tensor) -> CatResult<(usize, usize, Tensor)> {
    let dims = &x.ty.shape.dims;
    if dims.len() != 3 {
        return Err(CatError::ShapeMismatch("expected [B,S,H]".into()));
    }
    let (b, s, h) = (dims[0], dims[1], dims[2]);
    let t = Tensor::from_vec(TensorType::f32([b * s, h]), x.data.clone())?;
    Ok((b, s, t))
}

fn rms_vec(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len() as f32;
    let ms = x.iter().map(|v| v * v).sum::<f32>() / n;
    let inv = 1.0 / (ms + eps).sqrt();
    x.iter().zip(w.iter()).map(|(a, g)| a * inv * g).collect()
}

fn rope_inplace(q: &mut [f32], pos: usize, theta: f64) {
    let head = q.len();
    for i in 0..head / 2 {
        let freq = 1.0 / theta.powf(2.0 * i as f64 / head as f64);
        let ang = pos as f64 * freq;
        let (s, c) = ang.sin_cos();
        let (s, c) = (s as f32, c as f32);
        let (a, b) = (q[2 * i], q[2 * i + 1]);
        q[2 * i] = a * c - b * s;
        q[2 * i + 1] = a * s + b * c;
    }
}

#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
pub fn mla(
    x: &Tensor,
    w_dq: &Tensor,
    q_norm: &Tensor,
    w_uq: &Tensor,
    w_dkv: &Tensor,
    kv_norm: &Tensor,
    w_uk: &Tensor,
    w_uv: &Tensor,
    w_kr: &Tensor,
    w_o: &Tensor,
    n_heads: usize,
    qk_nope: usize,
    qk_rope: usize,
    v_dim: usize,
    rope_theta: f64,
) -> CatResult<Tensor> {
    let (b, s, h) = {
        let d = &x.ty.shape.dims;
        if d.len() != 3 {
            return Err(CatError::ShapeMismatch("mla expects [B,S,H]".into()));
        }
        (d[0], d[1], d[2])
    };
    let (_, _, xf) = flat(x)?;
    let q_lat = linear_seq(&xf, w_dq, None)?;
    let kv_lat = linear_seq(&xf, w_dkv, None)?;
    let q_up_w = w_uq;
    let mut q_heads = vec![0.0f32; b * s * n_heads * (qk_nope + qk_rope)];
    let mut k_nope = vec![0.0f32; b * s * n_heads * qk_nope];
    let mut k_rope = vec![0.0f32; b * s * qk_rope];
    let mut v = vec![0.0f32; b * s * n_heads * v_dim];
    let q_head = qk_nope + qk_rope;
    for t in 0..b * s {
        let pos = t % s;
        let ql = rms_vec(
            &q_lat.data[t * w_dq.ty.shape.dims[1]..(t + 1) * w_dq.ty.shape.dims[1]],
            &q_norm.data,
            1e-6,
        );
        let ql_t = Tensor::from_vec(TensorType::f32([1, ql.len()]), ql)?;
        let qu = linear_seq(&ql_t, q_up_w, None)?;
        for head in 0..n_heads {
            let mut slice = qu.data[head * q_head..(head + 1) * q_head].to_vec();
            rope_inplace(&mut slice[qk_nope..], pos, rope_theta);
            q_heads[t * n_heads * q_head + head * q_head
                ..(t * n_heads * q_head + (head + 1) * q_head)]
                .copy_from_slice(&slice);
        }
        let kl = rms_vec(
            &kv_lat.data[t * w_dkv.ty.shape.dims[1]..(t + 1) * w_dkv.ty.shape.dims[1]],
            &kv_norm.data,
            1e-6,
        );
        let kl_t = Tensor::from_vec(TensorType::f32([1, kl.len()]), kl)?;
        let kn = linear_seq(&kl_t, w_uk, None)?;
        let vv = linear_seq(&kl_t, w_uv, None)?;
        k_nope[t * n_heads * qk_nope..(t + 1) * n_heads * qk_nope].copy_from_slice(&kn.data);
        v[t * n_heads * v_dim..(t + 1) * n_heads * v_dim].copy_from_slice(&vv.data);
        let xt = Tensor::from_vec(TensorType::f32([1, h]), xf.data[t * h..(t + 1) * h].to_vec())?;
        let mut kr = linear_seq(&xt, w_kr, None)?.data;
        rope_inplace(&mut kr, pos, rope_theta);
        k_rope[t * qk_rope..(t + 1) * qk_rope].copy_from_slice(&kr);
    }
    let scale = 1.0 / (q_head as f32).sqrt();
    let mut ctx = vec![0.0f32; b * s * n_heads * v_dim];
    for bi in 0..b {
        for head in 0..n_heads {
            for i in 0..s {
                let q_off = ((bi * s + i) * n_heads + head) * q_head;
                let mut scores = vec![f32::NEG_INFINITY; s];
                for j in 0..=i {
                    let kn_off = ((bi * s + j) * n_heads + head) * qk_nope;
                    let mut dot = 0.0f32;
                    for p in 0..qk_nope {
                        dot += q_heads[q_off + p] * k_nope[kn_off + p];
                    }
                    let kr_off = (bi * s + j) * qk_rope;
                    for p in 0..qk_rope {
                        dot += q_heads[q_off + qk_nope + p] * k_rope[kr_off + p];
                    }
                    scores[j] = dot * scale;
                }
                let max = scores[..=i].iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0f32;
                for sc in scores.iter_mut().take(i + 1) {
                    *sc = (*sc - max).exp();
                    sum += *sc;
                }
                let inv = 1.0 / sum.max(1e-30);
                let o_off = ((bi * s + i) * n_heads + head) * v_dim;
                for j in 0..=i {
                    let w = scores[j] * inv;
                    let v_off = ((bi * s + j) * n_heads + head) * v_dim;
                    for p in 0..v_dim {
                        ctx[o_off + p] += w * v[v_off + p];
                    }
                }
            }
        }
    }
    let ctx_t = Tensor::from_vec(TensorType::f32([b * s, n_heads * v_dim]), ctx)?;
    let y = linear_seq(&ctx_t, w_o, None)?;
    Tensor::from_vec(TensorType::f32([b, s, h]), y.data)
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Group-limited sigmoid top-k. Returns `(expert_index, weight)` per token, length `top_k`.
fn route_sigmoid(
    logits: &[f32],
    n_group: usize,
    topk_group: usize,
    top_k: usize,
    scale: f32,
) -> Vec<(usize, f32)> {
    let n = logits.len();
    let per = n / n_group.max(1);
    let scores: Vec<f32> = logits.iter().copied().map(sigmoid).collect();
    let mut group_score = vec![0.0f32; n_group];
    for g in 0..n_group {
        group_score[g] = scores[g * per..(g + 1) * per].iter().copied().fold(0.0, f32::max);
    }
    let mut groups: Vec<usize> = (0..n_group).collect();
    groups.sort_by(|&a, &b| {
        group_score[b].partial_cmp(&group_score[a]).unwrap_or(std::cmp::Ordering::Equal)
    });
    let keep = &groups[..topk_group.min(n_group)];
    let mut cand: Vec<usize> = Vec::new();
    for &g in keep {
        cand.extend(g * per..(g + 1) * per);
    }
    cand.sort_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap_or(std::cmp::Ordering::Equal));
    cand.truncate(top_k.min(cand.len()));
    let mut picked: Vec<(usize, f32)> = cand.into_iter().map(|i| (i, scores[i])).collect();
    let sum: f32 = picked.iter().map(|(_, w)| *w).sum();
    let inv = scale / sum.max(1e-30);
    for (_, w) in &mut picked {
        *w *= inv;
    }
    picked
}

#[allow(clippy::too_many_arguments)]
pub fn deepseek_moe(
    x: &Tensor,
    gate: &Tensor,
    shared: &[(Tensor, Tensor, Tensor)],
    routed: &[(Tensor, Tensor, Tensor)],
    top_k: usize,
    n_group: usize,
    topk_group: usize,
    scale: f32,
) -> CatResult<Tensor> {
    let hidden = *x.ty.shape.dims.last().unwrap();
    let tokens = x.data.len() / hidden;
    let n_exp = gate.ty.shape.dims[1];
    let xf = Tensor::from_vec(TensorType::f32([tokens, hidden]), x.data.clone())?;
    let logits = linear_seq(&xf, gate, None)?;
    let mut out = vec![0.0f32; x.data.len()];
    for (w1, w3, w2) in shared {
        let y = swiglu(&xf, w1, w3, w2)?;
        for (o, v) in out.iter_mut().zip(y.data.iter()) {
            *o += *v;
        }
    }
    for t in 0..tokens {
        let row = &logits.data[t * n_exp..(t + 1) * n_exp];
        let picks = route_sigmoid(row, n_group, topk_group, top_k, scale);
        let xt = Tensor::from_vec(
            TensorType::f32([1, hidden]),
            x.data[t * hidden..(t + 1) * hidden].to_vec(),
        )?;
        for (e, w) in picks {
            let (w1, w3, w2) = &routed[e];
            let y = swiglu(&xt, w1, w3, w2)?;
            for i in 0..hidden {
                out[t * hidden + i] += w * y.data[i];
            }
        }
    }
    Tensor::from_vec(x.ty.clone(), out)
}

pub fn mix(a: &Tensor, b: &Tensor, gate: f32) -> CatResult<Tensor> {
    if a.data.len() != b.data.len() {
        return Err(CatError::ShapeMismatch("mix operands differ".into()));
    }
    let s = sigmoid(gate);
    let mut y = a.clone();
    for (o, r) in y.data.iter_mut().zip(b.data.iter()) {
        *o = s * *o + (1.0 - s) * *r;
    }
    Ok(y)
}

pub fn mix_backward(
    a: &Tensor,
    b: &Tensor,
    gate: f32,
    dy: &Tensor,
) -> CatResult<(Tensor, Tensor, f32)> {
    let s = sigmoid(gate);
    let mut da = dy.clone();
    let mut db = dy.clone();
    let mut dg = 0.0f32;
    for i in 0..dy.data.len() {
        da.data[i] *= s;
        db.data[i] *= 1.0 - s;
        dg += (a.data[i] - b.data[i]) * dy.data[i];
    }
    dg *= s * (1.0 - s);
    Ok((da, db, dg))
}
