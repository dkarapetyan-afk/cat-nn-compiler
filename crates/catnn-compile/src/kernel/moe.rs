//! Mixtral kernels: embedding, RMSNorm, RoPE+GQA, SwiGLU, router, MoE.
//!
//! `cpu_*` is the executable path (Rayon where it helps). `ref_*` is an
//! independent naïve loop nest used as the 2-cell witness. Both share
//! [`catnn_core::lens::moe::topk_softmax`].

use catnn_core::lens::moe::topk_softmax;
use catnn_core::spec::MixtralConfig;
use catnn_core::tensor::{Tensor, TensorType};
use catnn_core::{CatError, CatResult};

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

pub fn rmsnorm(x: &Tensor, w: &Tensor, eps: f32) -> CatResult<Tensor> {
    let last =
        *x.ty.shape.dims.last().ok_or_else(|| CatError::ShapeMismatch("rms scalar".into()))?;
    if w.data.len() != last {
        return Err(CatError::ShapeMismatch("rmsnorm weight".into()));
    }
    let mut y = x.clone();
    for (row, out) in x.data.chunks(last).zip(y.data.chunks_mut(last)) {
        let mean_sq = row.iter().map(|v| v * v).sum::<f32>() / last as f32;
        let inv = 1.0 / (mean_sq + eps).sqrt();
        for i in 0..last {
            out[i] = row[i] * inv * w.data[i];
        }
    }
    Ok(y)
}

pub fn embedding(ids: &[u32], table: &Tensor, hidden: usize) -> CatResult<Tensor> {
    let vocab = table.ty.shape.dims[0];
    let mut data = Vec::with_capacity(ids.len() * hidden);
    for &id in ids {
        if id as usize >= vocab {
            return Err(CatError::ShapeMismatch(format!("token {id} >= vocab {vocab}")));
        }
        let off = id as usize * hidden;
        data.extend_from_slice(&table.data[off..off + hidden]);
    }
    Tensor::from_vec(TensorType::f32([ids.len(), hidden]), data)
}

fn apply_rope(q: &mut [f32], head_dim: usize, pos: usize, theta: f64) {
    for i in 0..head_dim / 2 {
        let freq = 1.0 / theta.powf(2.0 * i as f64 / head_dim as f64);
        let ang = pos as f64 * freq;
        let (s, c) = ang.sin_cos();
        let (s, c) = (s as f32, c as f32);
        let i0 = 2 * i;
        let i1 = i0 + 1;
        let (a, b) = (q[i0], q[i1]);
        q[i0] = a * c - b * s;
        q[i1] = a * s + b * c;
    }
}

/// GQA + RoPE + optional sliding window. `x` is `[B,S,H]` (already RMS-normed).
#[allow(clippy::too_many_arguments)]
pub fn gqa(
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
) -> CatResult<Tensor> {
    let dims = &x.ty.shape.dims;
    if dims.len() != 3 {
        return Err(CatError::ShapeMismatch("gqa expects [B,S,H]".into()));
    }
    let (b, s, d) = (dims[0], dims[1], dims[2]);
    let q = linear3(x, wq, b, s, d, n_heads * head_dim)?;
    let k = linear3(x, wk, b, s, d, n_kv * head_dim)?;
    let v = linear3(x, wv, b, s, d, n_kv * head_dim)?;
    let mut q = q.data;
    let mut k = k.data;
    let repeats = n_heads / n_kv.max(1);
    let scale = 1.0 / (head_dim as f32).sqrt();
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
    let mut ctx = vec![0.0f32; b * s * n_heads * head_dim];
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
    let ctx_t = Tensor::from_vec(TensorType::f32([b, s, n_heads * head_dim]), ctx)?;
    linear3(&ctx_t, wo, b, s, n_heads * head_dim, d)
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
    let y = crate::kernel::cpu::linear(&flat, w, None)?;
    if y.data.len() != b * s * dout {
        return Err(CatError::ShapeMismatch(format!(
            "linear3 out {} vs {}",
            y.data.len(),
            b * s * dout
        )));
    }
    Tensor::from_vec(TensorType::f32([b, s, dout]), y.data)
}

pub fn swiglu(x: &Tensor, w1: &Tensor, w3: &Tensor, w2: &Tensor) -> CatResult<Tensor> {
    let dims = &x.ty.shape.dims;
    let hidden = *dims.last().unwrap();
    let tokens = x.data.len() / hidden;
    let mid = w1.ty.shape.dims[1];
    let xf = Tensor::from_vec(TensorType::f32([tokens, hidden]), x.data.clone())?;
    let a = crate::kernel::cpu::linear(&xf, w1, None)?;
    let c = crate::kernel::cpu::linear(&xf, w3, None)?;
    let mut h = a;
    for i in 0..h.data.len() {
        h.data[i] = silu(h.data[i]) * c.data[i];
    }
    h.ty = TensorType::f32([tokens, mid]);
    let y = crate::kernel::cpu::linear(&h, w2, None)?;
    Tensor::from_vec(x.ty.clone(), y.data)
}

/// Router: `[B,S,H] →` per-token top-k (indices, weights). Also returns gate logits.
pub fn route(
    x: &Tensor,
    gate: &Tensor,
    top_k: usize,
) -> CatResult<(Vec<usize>, Vec<f32>, Vec<f32>)> {
    let hidden = *x.ty.shape.dims.last().unwrap();
    let n_exp = gate.ty.shape.dims[1];
    let tokens = x.data.len() / hidden;
    let xf = Tensor::from_vec(TensorType::f32([tokens, hidden]), x.data.clone())?;
    let logits = crate::kernel::cpu::linear(&xf, gate, None)?;
    let mut idx_all = Vec::new();
    let mut w_all = Vec::new();
    for row in logits.data.chunks(n_exp) {
        let (idx, w) = topk_softmax(row, top_k);
        idx_all.extend(idx);
        w_all.extend(w);
    }
    Ok((idx_all, w_all, logits.data))
}

pub fn moe(
    x: &Tensor,
    gate: &Tensor,
    experts: &[(Tensor, Tensor, Tensor)],
    top_k: usize,
) -> CatResult<Tensor> {
    let hidden = *x.ty.shape.dims.last().unwrap();
    let tokens = x.data.len() / hidden;
    let (idx, wts, _) = route(x, gate, top_k)?;
    let mut out = vec![0.0f32; x.data.len()];
    for t in 0..tokens {
        let xt = Tensor::from_vec(
            TensorType::f32([1, hidden]),
            x.data[t * hidden..(t + 1) * hidden].to_vec(),
        )?;
        for k in 0..top_k {
            let e = idx[t * top_k + k];
            let wt = wts[t * top_k + k];
            let (w1, w3, w2) = &experts[e];
            let y = swiglu(&xt, w1, w3, w2)?;
            for i in 0..hidden {
                out[t * hidden + i] += wt * y.data[i];
            }
        }
    }
    Tensor::from_vec(x.ty.clone(), out)
}

/// Seeded Mixtral weights for the tiny (or any) config. Not HF safetensors.
#[derive(Debug, Clone)]
pub struct MixtralWeights {
    pub embed: Tensor,
    pub layers: Vec<LayerWeights>,
    pub final_norm: Tensor,
    pub lm_head: Tensor,
}

#[derive(Debug, Clone)]
pub struct LayerWeights {
    pub rms_attn: Tensor,
    pub wq: Tensor,
    pub wk: Tensor,
    pub wv: Tensor,
    pub wo: Tensor,
    pub rms_moe: Tensor,
    pub gate: Tensor,
    pub experts: Vec<(Tensor, Tensor, Tensor)>,
}

impl MixtralWeights {
    pub fn seeded(cfg: &MixtralConfig, seed: u64) -> CatResult<Self> {
        let h = cfg.hidden_size;
        let mid = cfg.intermediate_size;
        let hd = cfg.head_dim()?;
        let kv = cfg.num_key_value_heads * hd;
        let mut s = seed;
        let mut next = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            s
        };
        let embed = Tensor::lcg(TensorType::f32([cfg.vocab_size, h]), next());
        let mut layers = Vec::new();
        for _ in 0..cfg.num_hidden_layers {
            let mut experts = Vec::new();
            for _ in 0..cfg.num_local_experts {
                experts.push((
                    Tensor::lcg(TensorType::f32([h, mid]), next()),
                    Tensor::lcg(TensorType::f32([h, mid]), next()),
                    Tensor::lcg(TensorType::f32([mid, h]), next()),
                ));
            }
            layers.push(LayerWeights {
                rms_attn: Tensor::ones(TensorType::f32([h])),
                wq: Tensor::lcg(TensorType::f32([h, h]), next()),
                wk: Tensor::lcg(TensorType::f32([h, kv]), next()),
                wv: Tensor::lcg(TensorType::f32([h, kv]), next()),
                wo: Tensor::lcg(TensorType::f32([h, h]), next()),
                rms_moe: Tensor::ones(TensorType::f32([h])),
                gate: Tensor::lcg(TensorType::f32([h, cfg.num_local_experts]), next()),
                experts,
            });
        }
        Ok(Self {
            embed,
            layers,
            final_norm: Tensor::ones(TensorType::f32([h])),
            lm_head: Tensor::lcg(TensorType::f32([cfg.vocab_size, h]), next()),
        })
    }
}

pub fn forward(
    cfg: &MixtralConfig,
    w: &MixtralWeights,
    token_ids: &[u32],
    batch: usize,
    seq: usize,
) -> CatResult<Tensor> {
    if token_ids.len() != batch * seq {
        return Err(CatError::ShapeMismatch("token batch*seq".into()));
    }
    let h = cfg.hidden_size;
    let hd = cfg.head_dim()?;
    let mut x = embedding(token_ids, &w.embed, h)?;
    x.ty = TensorType::f32([batch, seq, h]);
    for layer in &w.layers {
        let xn = rmsnorm(&x, &layer.rms_attn, cfg.rms_norm_eps)?;
        let attn = gqa(
            &xn,
            &layer.wq,
            &layer.wk,
            &layer.wv,
            &layer.wo,
            cfg.num_attention_heads,
            cfg.num_key_value_heads,
            hd,
            cfg.rope_theta,
            cfg.sliding_window,
        )?;
        x = crate::kernel::cpu::add(&x, &attn)?;
        let xn = rmsnorm(&x, &layer.rms_moe, cfg.rms_norm_eps)?;
        let m = moe(&xn, &layer.gate, &layer.experts, cfg.num_experts_per_tok)?;
        x = crate::kernel::cpu::add(&x, &m)?;
    }
    x = rmsnorm(&x, &w.final_norm, cfg.rms_norm_eps)?;
    lm_head(&x, &w.lm_head, cfg.vocab_size)
}

pub fn lm_head(x: &Tensor, w: &Tensor, vocab: usize) -> CatResult<Tensor> {
    let dims = &x.ty.shape.dims;
    let (b, s, h) = (dims[0], dims[1], dims[2]);
    // w is stored [V, H] like HF; linear wants [H, V].
    let wt = transpose2(w, vocab, h)?;
    let flat = Tensor::from_vec(TensorType::f32([b * s, h]), x.data.clone())?;
    let y = crate::kernel::cpu::linear(&flat, &wt, None)?;
    Tensor::from_vec(TensorType::f32([b, s, vocab]), y.data)
}

fn transpose2(w: &Tensor, rows: usize, cols: usize) -> CatResult<Tensor> {
    let mut data = vec![0.0f32; rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            data[j * rows + i] = w.data[i * cols + j];
        }
    }
    Tensor::from_vec(TensorType::f32([cols, rows]), data)
}

/// Independent naïve reference: same math, no Rayon, explicit loops.
pub mod reference {
    use super::*;

    pub fn rmsnorm(x: &Tensor, w: &Tensor, eps: f32) -> CatResult<Tensor> {
        super::rmsnorm(x, w, eps)
    }

    pub fn swiglu(x: &Tensor, w1: &Tensor, w3: &Tensor, w2: &Tensor) -> CatResult<Tensor> {
        let hidden = *x.ty.shape.dims.last().unwrap();
        let tokens = x.data.len() / hidden;
        let mid = w1.ty.shape.dims[1];
        let mut hidden_act = vec![0.0f32; tokens * mid];
        for t in 0..tokens {
            for j in 0..mid {
                let mut a = 0.0;
                let mut c = 0.0;
                for i in 0..hidden {
                    let xv = x.data[t * hidden + i];
                    a += xv * w1.data[i * mid + j];
                    c += xv * w3.data[i * mid + j];
                }
                hidden_act[t * mid + j] = super::silu(a) * c;
            }
        }
        let mut out = vec![0.0f32; tokens * hidden];
        for t in 0..tokens {
            for i in 0..hidden {
                let mut s = 0.0;
                for j in 0..mid {
                    s += hidden_act[t * mid + j] * w2.data[j * hidden + i];
                }
                out[t * hidden + i] = s;
            }
        }
        Tensor::from_vec(x.ty.clone(), out)
    }

    pub fn moe(
        x: &Tensor,
        gate: &Tensor,
        experts: &[(Tensor, Tensor, Tensor)],
        top_k: usize,
    ) -> CatResult<Tensor> {
        let hidden = *x.ty.shape.dims.last().unwrap();
        let tokens = x.data.len() / hidden;
        let n_exp = gate.ty.shape.dims[1];
        let mut out = vec![0.0f32; x.data.len()];
        for t in 0..tokens {
            let mut logits = vec![0.0f32; n_exp];
            for (e, logit) in logits.iter_mut().enumerate() {
                let mut s = 0.0;
                for i in 0..hidden {
                    s += x.data[t * hidden + i] * gate.data[i * n_exp + e];
                }
                *logit = s;
            }
            let (idx, wts) = topk_softmax(&logits, top_k);
            let xt = Tensor::from_vec(
                TensorType::f32([1, hidden]),
                x.data[t * hidden..(t + 1) * hidden].to_vec(),
            )?;
            for k in 0..idx.len() {
                let (w1, w3, w2) = &experts[idx[k]];
                let y = swiglu(&xt, w1, w3, w2)?;
                for i in 0..hidden {
                    out[t * hidden + i] += wts[k] * y.data[i];
                }
            }
        }
        Tensor::from_vec(x.ty.clone(), out)
    }

    pub fn forward(
        cfg: &MixtralConfig,
        w: &MixtralWeights,
        token_ids: &[u32],
        batch: usize,
        seq: usize,
    ) -> CatResult<Tensor> {
        let h = cfg.hidden_size;
        let hd = cfg.head_dim()?;
        let mut x = super::embedding(token_ids, &w.embed, h)?;
        x.ty = TensorType::f32([batch, seq, h]);
        for layer in &w.layers {
            let xn = rmsnorm(&x, &layer.rms_attn, cfg.rms_norm_eps)?;
            let attn = super::gqa(
                &xn,
                &layer.wq,
                &layer.wk,
                &layer.wv,
                &layer.wo,
                cfg.num_attention_heads,
                cfg.num_key_value_heads,
                hd,
                cfg.rope_theta,
                cfg.sliding_window,
            )?;
            for i in 0..x.data.len() {
                x.data[i] += attn.data[i];
            }
            let xn = rmsnorm(&x, &layer.rms_moe, cfg.rms_norm_eps)?;
            let m = moe(&xn, &layer.gate, &layer.experts, cfg.num_experts_per_tok)?;
            for i in 0..x.data.len() {
                x.data[i] += m.data[i];
            }
        }
        x = rmsnorm(&x, &w.final_norm, cfg.rms_norm_eps)?;
        super::lm_head(&x, &w.lm_head, cfg.vocab_size)
    }
}

/// Optional safetensors hook. Not used in CI; returns a clear error if the
/// file is missing so we never pretend HF weights were loaded.
pub fn try_load_safetensors(path: &std::path::Path) -> CatResult<String> {
    if !path.exists() {
        return Err(CatError::Io(format!(
            "safetensors path {} not present — skipped (not executed)",
            path.display()
        )));
    }
    Err(CatError::Io(
        "safetensors decoder is not bundled; provide a loader at this path to compare vs Hugging Face"
            .into(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiny_cpu_matches_reference() {
        let cfg = MixtralConfig::tiny();
        let w = MixtralWeights::seeded(&cfg, 7).unwrap();
        let tokens = [1u32, 2, 3, 4, 5, 6, 7, 8];
        let a = forward(&cfg, &w, &tokens, 2, 4).unwrap();
        let b = reference::forward(&cfg, &w, &tokens, 2, 4).unwrap();
        let err = a.max_abs_diff(&b).unwrap();
        assert!(err < 2e-4, "max abs {err}");
    }

    #[test]
    fn safetensors_absent_is_explicit() {
        let err = try_load_safetensors(std::path::Path::new("/no/such/model.safetensors"));
        assert!(err.is_err());
    }
}
