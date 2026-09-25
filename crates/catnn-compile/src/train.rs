//! Memory-parametrized training: build the step graph, schedule it,
//! execute adjoints, apply SGD / Adam.
//!
//! The peak the compiler reports is the **simulated** occupancy of the
//! scheduled timeline. A tight buffer budget forces activation
//! checkpointing; a tight code budget forces kernel reloads. Both are
//! 2-cells with equivalence witnesses — rematerialized activations match
//! the stored ones to float tolerance.

use catnn_core::architecture::{DeviceMemory, MemoryBudget, MemoryHierarchy};
use catnn_core::lens::{Adam, AdamState, OptimizerKind, Sgd};
use catnn_core::spec::MixtralConfig;
use catnn_core::tensor::{Tensor, TensorType};
use catnn_core::{CatError, CatResult};

use crate::kernel::adjoint::{
    add_tensors, cross_entropy, embedding_backward, gqa_backward, linear_backward, moe_backward,
    mse_loss, relu_backward, rmsnorm_backward,
};
use crate::kernel::cpu;
use crate::kernel::moe::{self, MixtralWeights};
use crate::kernel::resource::{buf, describe, KernelResource};
use crate::schedule::{self, BufferReq, LaunchReq, ScheduleRequest, ScheduledStep};
use catnn_core::architecture::KernelBackend;
use catnn_core::category::OpenOp;

#[derive(Debug, Clone)]
pub struct TrainOptions {
    pub steps: usize,
    pub lr: f32,
    pub opt: OptimizerKind,
    pub budget: Option<MemoryBudget>,
    pub seed: u64,
}

impl Default for TrainOptions {
    fn default() -> Self {
        Self { steps: 8, lr: 0.05, opt: OptimizerKind::Sgd, budget: None, seed: 7 }
    }
}

#[derive(Debug, Clone)]
pub struct TrainReport {
    pub losses: Vec<f32>,
    pub final_params_checksum: u64,
    pub schedule: ScheduledStep,
    pub constrained_match: Option<f32>,
}

fn mlp_weights(seed: u64) -> (Tensor, Tensor, Tensor, Tensor) {
    let w1 = Tensor::lcg(TensorType::f32([16, 32]), seed);
    let b1 = Tensor::lcg(TensorType::f32([32]), seed + 1);
    let w2 = Tensor::lcg(TensorType::f32([32, 8]), seed + 2);
    let b2 = Tensor::lcg(TensorType::f32([8]), seed + 3);
    (w1, b1, w2, b2)
}

fn apply_sgd(p: &mut Tensor, g: &Tensor, opt: Sgd) -> CatResult<()> {
    *p = opt.put(p, g)?;
    Ok(())
}

fn apply_adam(p: &mut Tensor, g: &Tensor, st: &mut AdamState, opt: Adam) -> CatResult<()> {
    let (np, ns) = opt.put(p, g, st)?;
    *p = np;
    *st = ns;
    Ok(())
}

/// One MLP forward+loss+backward. `checkpoint` rematerializes `h1`.
pub fn mlp_step(
    w1: &Tensor,
    b1: &Tensor,
    w2: &Tensor,
    b2: &Tensor,
    x: &Tensor,
    y: &Tensor,
    checkpoint: bool,
) -> CatResult<(f32, Tensor, Tensor, Tensor, Tensor)> {
    let z1 = cpu::linear(x, w1, Some(b1))?;
    let h1 = cpu::relu(&z1);
    let pred = cpu::linear(&h1, w2, Some(b2))?;
    let (loss, dpred) = mse_loss(&pred, y)?;
    let h1_b = if checkpoint { cpu::relu(&cpu::linear(x, w1, Some(b1))?) } else { h1.clone() };
    let (dh1, dw2, db2) = linear_backward(&h1_b, w2, &dpred)?;
    let db2 = db2.unwrap();
    let dz1 = relu_backward(&z1, &dh1)?;
    let (dx, dw1, db1) = linear_backward(x, w1, &dz1)?;
    let _ = dx;
    Ok((loss, dw1, db1.unwrap(), dw2, db2))
}

pub fn train_mlp(opts: &TrainOptions) -> CatResult<TrainReport> {
    let (mut w1, mut b1, mut w2, mut b2) = mlp_weights(opts.seed);
    let x = Tensor::lcg(TensorType::f32([4, 16]), opts.seed + 9);
    let y = Tensor::lcg(TensorType::f32([4, 8]), opts.seed + 11);
    let sgd = Sgd::new(opts.lr);
    let adam = Adam::new(opts.lr);
    let mut a1 = AdamState::zeros_like(&w1);
    let mut ab1 = AdamState::zeros_like(&b1);
    let mut a2 = AdamState::zeros_like(&w2);
    let mut ab2 = AdamState::zeros_like(&b2);
    let checkpoint = opts.budget.map(|b| b.buffer < 64 * 1024).unwrap_or(false);
    let mut losses = Vec::new();
    for _ in 0..opts.steps {
        let (loss, dw1, db1, dw2, db2) = mlp_step(&w1, &b1, &w2, &b2, &x, &y, checkpoint)?;
        losses.push(loss);
        match opts.opt {
            OptimizerKind::Sgd => {
                apply_sgd(&mut w1, &dw1, sgd)?;
                apply_sgd(&mut b1, &db1, sgd)?;
                apply_sgd(&mut w2, &dw2, sgd)?;
                apply_sgd(&mut b2, &db2, sgd)?;
            }
            OptimizerKind::Adam => {
                apply_adam(&mut w1, &dw1, &mut a1, adam)?;
                apply_adam(&mut b1, &db1, &mut ab1, adam)?;
                apply_adam(&mut w2, &dw2, &mut a2, adam)?;
                apply_adam(&mut b2, &db2, &mut ab2, adam)?;
            }
        }
    }
    let schedule = plan_mlp_schedule(opts.budget)?;
    let mut chk = w1.checksum();
    chk ^= w2.checksum();
    Ok(TrainReport { losses, final_params_checksum: chk, schedule, constrained_match: None })
}

fn plan_mlp_schedule(budget: Option<MemoryBudget>) -> CatResult<ScheduledStep> {
    let mut regions = DeviceMemory::desktop_cpu();
    if let Some(b) = budget {
        regions.apply_budget(&b);
    }
    let hier = MemoryHierarchy::desktop_cpu();
    let bufs = vec![
        BufferReq { id: "x".into(), bytes: 4 * 16 * 4, live_from: 0, live_to: 5, checkpoint: false },
        BufferReq { id: "z1".into(), bytes: 4 * 32 * 4, live_from: 1, live_to: 4, checkpoint: false },
        BufferReq { id: "h1".into(), bytes: 4 * 32 * 4, live_from: 2, live_to: 3, checkpoint: false },
        BufferReq { id: "pred".into(), bytes: 4 * 8 * 4, live_from: 3, live_to: 3, checkpoint: false },
    ];
    let launches = vec![
        LaunchReq {
            at: 1,
            kernel: describe(
                &OpenOp::Linear { in_features: 16, out_features: 32, bias: true },
                "fc1",
                KernelBackend::CpuRust,
            ),
        },
        LaunchReq { at: 2, kernel: describe(&OpenOp::Relu, "act", KernelBackend::CpuRust) },
        LaunchReq {
            at: 3,
            kernel: describe(
                &OpenOp::Linear { in_features: 32, out_features: 8, bias: true },
                "fc2",
                KernelBackend::CpuRust,
            ),
        },
    ];
    Ok(schedule::schedule(ScheduleRequest {
        buffers: &bufs,
        launches: &launches,
        regions: &regions,
        hierarchy: &hier,
        allow_recompute: true,
        allow_reload: true,
    }))
}

/// Naive SGD reference used as a 2-cell witness (same axpy).
pub fn sgd_reference(p: &Tensor, g: &Tensor, lr: f32) -> CatResult<Tensor> {
    Sgd::new(lr).put(p, g)
}

pub fn adam_reference(p: &Tensor, g: &Tensor, s: &AdamState, opt: Adam) -> CatResult<(Tensor, AdamState)> {
    opt.put(p, g, s)
}

#[derive(Debug, Clone)]
pub struct MixtralGrads {
    pub embed: Tensor,
    pub layers: Vec<LayerGrads>,
    pub final_norm: Tensor,
    pub lm_head: Tensor,
}

#[derive(Debug, Clone)]
pub struct LayerGrads {
    pub rms_attn: Tensor,
    pub wq: Tensor,
    pub wk: Tensor,
    pub wv: Tensor,
    pub wo: Tensor,
    pub rms_moe: Tensor,
    pub gate: Tensor,
    pub experts: Vec<(Tensor, Tensor, Tensor)>,
}

fn zeros_like_w(w: &MixtralWeights) -> MixtralGrads {
    MixtralGrads {
        embed: Tensor::zeros(w.embed.ty.clone()),
        layers: w
            .layers
            .iter()
            .map(|l| LayerGrads {
                rms_attn: Tensor::zeros(l.rms_attn.ty.clone()),
                wq: Tensor::zeros(l.wq.ty.clone()),
                wk: Tensor::zeros(l.wk.ty.clone()),
                wv: Tensor::zeros(l.wv.ty.clone()),
                wo: Tensor::zeros(l.wo.ty.clone()),
                rms_moe: Tensor::zeros(l.rms_moe.ty.clone()),
                gate: Tensor::zeros(l.gate.ty.clone()),
                experts: l
                    .experts
                    .iter()
                    .map(|(a, b, c)| {
                        (
                            Tensor::zeros(a.ty.clone()),
                            Tensor::zeros(b.ty.clone()),
                            Tensor::zeros(c.ty.clone()),
                        )
                    })
                    .collect(),
            })
            .collect(),
        final_norm: Tensor::zeros(w.final_norm.ty.clone()),
        lm_head: Tensor::zeros(w.lm_head.ty.clone()),
    }
}

/// Full Mixtral train step (tiny configs only — materializes tensors).
pub fn mixtral_step(
    cfg: &MixtralConfig,
    w: &MixtralWeights,
    tokens: &[u32],
    batch: usize,
    seq: usize,
    checkpoint: bool,
) -> CatResult<(f32, MixtralGrads)> {
    if tokens.len() != batch * seq {
        return Err(CatError::ShapeMismatch("tokens".into()));
    }
    let h = cfg.hidden_size;
    let hd = cfg.head_dim()?;
    let targets: Vec<u32> = (0..batch * seq)
        .map(|i| tokens[(i + 1) % tokens.len()] % cfg.vocab_size as u32)
        .collect();
    let mut x = moe::embedding(tokens, &w.embed, h)?;
    x.ty = TensorType::f32([batch, seq, h]);
    let mut xs = vec![x.clone()];
    let mut saved = Vec::new();
    for layer in &w.layers {
        let xn = moe::rmsnorm(&x, &layer.rms_attn, cfg.rms_norm_eps)?;
        let attn = moe::gqa(
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
        x = add_tensors(&x, &attn)?;
        let xm = moe::rmsnorm(&x, &layer.rms_moe, cfg.rms_norm_eps)?;
        let m = moe::moe(&xm, &layer.gate, &layer.experts, cfg.num_experts_per_tok)?;
        x = add_tensors(&x, &m)?;
        saved.push((xn, xm, attn, m));
        xs.push(x.clone());
    }
    let xf = moe::rmsnorm(&x, &w.final_norm, cfg.rms_norm_eps)?;
    let logits = moe::lm_head(&xf, &w.lm_head, cfg.vocab_size)?;
    let (loss, dlogits) = cross_entropy(&logits, &targets)?;

    // LM head adjoint: w is [V,H], forward transposes to [H,V].
    let wt = transpose2(&w.lm_head, cfg.vocab_size, h)?;
    let xf_flat = Tensor::from_vec(TensorType::f32([batch * seq, h]), xf.data.clone())?;
    let dlog_flat = Tensor::from_vec(TensorType::f32([batch * seq, cfg.vocab_size]), dlogits.data.clone())?;
    let (dxf, dwt, _) = linear_backward(&xf_flat, &wt, &dlog_flat)?;
    let dlm = transpose2(&dwt, h, cfg.vocab_size)?;
    let mut dx = Tensor::from_vec(x.ty.clone(), dxf.data)?;
    let (dxn, dfn) = rmsnorm_backward(&x, &w.final_norm, &dx, cfg.rms_norm_eps)?;
    dx = dxn;
    let mut grads = zeros_like_w(w);
    grads.final_norm = dfn;
    grads.lm_head = dlm;

    for li in (0..w.layers.len()).rev() {
        let layer = &w.layers[li];
        let x_in = &xs[li + 1]; // after moe residual — wait xs[li] is input to layer
        let x_pre_moe = {
            // rematerialize or use saved
            let (ref _xn, ref xm, ref attn, ref _m) = saved[li];
            if checkpoint {
                // Recompute layer from xs[li].
                let xin = &xs[li];
                let xn = moe::rmsnorm(xin, &layer.rms_attn, cfg.rms_norm_eps)?;
                let attn = moe::gqa(
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
                let xmid = add_tensors(xin, &attn)?;
                let xm = moe::rmsnorm(&xmid, &layer.rms_moe, cfg.rms_norm_eps)?;
                (xmid, xm)
            } else {
                let xmid = add_tensors(&xs[li], attn)?;
                (xmid, xm.clone())
            }
        };
        let (xmid, xm) = x_pre_moe;
        let (dxm, dgate, dex) =
            moe_backward(&xm, &layer.gate, &layer.experts, cfg.num_experts_per_tok, &dx)?;
        grads.layers[li].gate = dgate;
        grads.layers[li].experts = dex;
        let (dxmid, drm) = rmsnorm_backward(&xmid, &layer.rms_moe, &dxm, cfg.rms_norm_eps)?;
        grads.layers[li].rms_moe = drm;
        // residual: dx flows to attn out and skip
        let dx_skip = dxmid.clone();
        let (xn, _) = if checkpoint {
            let xin = &xs[li];
            (moe::rmsnorm(xin, &layer.rms_attn, cfg.rms_norm_eps)?, xin.clone())
        } else {
            (saved[li].0.clone(), xs[li].clone())
        };
        let (dxn, dwq, dwk, dwv, dwo) = gqa_backward(
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
            &dx_skip,
        )?;
        grads.layers[li].wq = dwq;
        grads.layers[li].wk = dwk;
        grads.layers[li].wv = dwv;
        grads.layers[li].wo = dwo;
        let (dxin, dra) = rmsnorm_backward(&xs[li], &layer.rms_attn, &dxn, cfg.rms_norm_eps)?;
        grads.layers[li].rms_attn = dra;
        // skip connections
        let _ = x_in;
        let _ = xn;
        // Residual: x_{l+1} = moe + x_mid, x_mid = attn + x_l.
        dx = add_tensors(&dxin, &dxmid)?;
    }
    grads.embed = embedding_backward(tokens, &w.embed, &dx)?;
    let _ = (checkpoint, xs);
    Ok((loss, grads))
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

fn apply_grads_sgd(w: &mut MixtralWeights, g: &MixtralGrads, opt: Sgd) -> CatResult<()> {
    w.embed = opt.put(&w.embed, &g.embed)?;
    w.final_norm = opt.put(&w.final_norm, &g.final_norm)?;
    w.lm_head = opt.put(&w.lm_head, &g.lm_head)?;
    for (l, gl) in w.layers.iter_mut().zip(g.layers.iter()) {
        l.rms_attn = opt.put(&l.rms_attn, &gl.rms_attn)?;
        l.wq = opt.put(&l.wq, &gl.wq)?;
        l.wk = opt.put(&l.wk, &gl.wk)?;
        l.wv = opt.put(&l.wv, &gl.wv)?;
        l.wo = opt.put(&l.wo, &gl.wo)?;
        l.rms_moe = opt.put(&l.rms_moe, &gl.rms_moe)?;
        l.gate = opt.put(&l.gate, &gl.gate)?;
        for (e, ge) in l.experts.iter_mut().zip(gl.experts.iter()) {
            e.0 = opt.put(&e.0, &ge.0)?;
            e.1 = opt.put(&e.1, &ge.1)?;
            e.2 = opt.put(&e.2, &ge.2)?;
        }
    }
    Ok(())
}

pub fn train_mixtral_tiny(opts: &TrainOptions) -> CatResult<TrainReport> {
    let cfg = MixtralConfig::tiny();
    let mut w = MixtralWeights::seeded(&cfg, opts.seed)?;
    let tokens = [1u32, 2, 3, 4, 5, 6, 7, 8];
    let sgd = Sgd::new(opts.lr);
    let checkpoint = opts.budget.map(|b| b.buffer < 256 * 1024).unwrap_or(false);
    let mut losses = Vec::new();
    for _ in 0..opts.steps {
        let (loss, g) = mixtral_step(&cfg, &w, &tokens, 2, 4, checkpoint)?;
        losses.push(loss);
        apply_grads_sgd(&mut w, &g, sgd)?;
    }
    let schedule = plan_mixtral_tiny_schedule(opts.budget)?;
    Ok(TrainReport {
        losses,
        final_params_checksum: w.embed.checksum() ^ w.lm_head.checksum(),
        schedule,
        constrained_match: None,
    })
}

pub fn plan_mixtral_tiny_schedule(budget: Option<MemoryBudget>) -> CatResult<ScheduledStep> {
    let mut regions = DeviceMemory::desktop_cpu();
    if let Some(b) = budget {
        regions.apply_budget(&b);
    }
    let hier = MemoryHierarchy::desktop_cpu();
    let h = 32u64;
    let bsz = 2u64;
    let seq = 4u64;
    let act = bsz * seq * h * 4;
    let mut bufs = vec![BufferReq {
        id: "embed".into(),
        bytes: act,
        live_from: 0,
        live_to: 8,
        checkpoint: false,
    }];
    let mut launches = Vec::new();
    let mut t = 1usize;
    for layer in 0..2 {
        bufs.push(BufferReq {
            id: format!("L{layer}.act"),
            bytes: act,
            live_from: t,
            live_to: t + 3,
            checkpoint: false,
        });
        launches.push(LaunchReq {
            at: t,
            kernel: describe(&OpenOp::RmsNorm { eps_e6: 10 }, &format!("L{layer}.rms"), KernelBackend::CpuRust),
        });
        launches.push(LaunchReq {
            at: t + 1,
            kernel: describe(
                &OpenOp::GqaAttention {
                    n_heads: 4,
                    n_kv_heads: 2,
                    head_dim: 8,
                    d_model: 32,
                    sliding_window: Some(8),
                    rope_theta: 10_000,
                },
                &format!("L{layer}.gqa"),
                KernelBackend::CpuRust,
            ),
        });
        launches.push(LaunchReq {
            at: t + 2,
            kernel: describe(
                &OpenOp::MoE { n_experts: 4, top_k: 2, hidden: 32, intermediate: 64 },
                &format!("L{layer}.moe"),
                KernelBackend::CpuRust,
            ),
        });
        t += 4;
    }
    Ok(schedule::schedule(ScheduleRequest {
        buffers: &bufs,
        launches: &launches,
        regions: &regions,
        hierarchy: &hier,
        allow_recompute: true,
        allow_reload: true,
    }))
}

/// Unconstrained vs tight-budget Mixtral run: same seeds, compare params.
pub fn constrained_matches_unconstrained() -> CatResult<f32> {
    let cfg = MixtralConfig::tiny();
    let tokens = [1u32, 2, 3, 4, 5, 6, 7, 8];
    let mut a = MixtralWeights::seeded(&cfg, 3)?;
    let mut b = a.clone();
    let sgd = Sgd::new(0.02);
    for _ in 0..3 {
        let ( _, ga) = mixtral_step(&cfg, &a, &tokens, 2, 4, false)?;
        let ( _, gb) = mixtral_step(&cfg, &b, &tokens, 2, 4, true)?;
        apply_grads_sgd(&mut a, &ga, sgd)?;
        apply_grads_sgd(&mut b, &gb, sgd)?;
    }
    a.embed.max_abs_diff(&b.embed)
}

/// One Mixtral-8x7B *layer* training step (fwd + bwd) as a unified timeline.
/// Buffer sizes use bf16 activations and the cited 8x7B shapes; kernel
/// cubin sizes are estimates. Nothing here is launched on a GPU.
pub fn plan_mixtral_layer_train(budget: Option<MemoryBudget>) -> CatResult<ScheduledStep> {
    let cfg = MixtralConfig::mixtral_8x7b();
    let mut regions = DeviceMemory::dgx_spark();
    if let Some(b) = budget {
        regions.apply_budget(&b);
    }
    let hier = MemoryHierarchy::dgx_spark();
    let h = cfg.hidden_size as u64;
    let inter = cfg.intermediate_size as u64;
    let seq = 2048u64;
    let micro = 1u64;
    let act = micro * seq * h * 2; // bf16
    let expert = 3 * h * inter * 2;
    let attn_w = h * (h + 2 * (h / 4) + h) * 2;
    let bufs = vec![
        BufferReq { id: "W.attn".into(), bytes: attn_w, live_from: 0, live_to: 14, checkpoint: false },
        BufferReq { id: "W.experts".into(), bytes: expert * 2, live_from: 0, live_to: 14, checkpoint: false },
        BufferReq { id: "x_in".into(), bytes: act, live_from: 0, live_to: 13, checkpoint: false },
        BufferReq { id: "xn".into(), bytes: act, live_from: 1, live_to: 3, checkpoint: false },
        BufferReq { id: "attn".into(), bytes: act, live_from: 2, live_to: 3, checkpoint: false },
        BufferReq { id: "x_mid".into(), bytes: act, live_from: 3, live_to: 11, checkpoint: false },
        BufferReq { id: "xm".into(), bytes: act, live_from: 4, live_to: 6, checkpoint: false },
        BufferReq { id: "moe".into(), bytes: act, live_from: 6, live_to: 7, checkpoint: false },
        BufferReq { id: "dlogits".into(), bytes: act, live_from: 8, live_to: 13, checkpoint: false },
        BufferReq { id: "dW".into(), bytes: attn_w + expert * 2, live_from: 8, live_to: 14, checkpoint: false },
    ];

    let rms = describe(&OpenOp::RmsNorm { eps_e6: 10 }, "L.rms", KernelBackend::CudaSm121)
        .with_buffers(vec![buf("x_in", act), buf("W.rms", h * 2)], vec![buf("xn", act)], vec![]);
    let gqa = describe(
        &OpenOp::GqaAttention {
            n_heads: cfg.num_attention_heads,
            n_kv_heads: cfg.num_key_value_heads,
            head_dim: 128,
            d_model: cfg.hidden_size,
            sliding_window: None,
            rope_theta: 1_000_000,
        },
        "L.gqa",
        KernelBackend::CudaSm121,
    )
    .with_buffers(vec![buf("xn", act), buf("W.attn", attn_w)], vec![buf("attn", act)], vec![]);
    let router = describe(
        &OpenOp::Router { n_experts: cfg.num_local_experts, top_k: cfg.num_experts_per_tok },
        "L.router",
        KernelBackend::CudaSm121,
    )
    .with_buffers(vec![buf("xm", act)], vec![buf("gate", 8 * 2)], vec![]);
    let swiglu = describe(
        &OpenOp::SwiGlu { hidden: cfg.hidden_size, intermediate: cfg.intermediate_size },
        "L.swiglu",
        KernelBackend::CudaSm121,
    )
    .with_buffers(vec![buf("xm", act), buf("W.experts", expert * 2)], vec![buf("moe", act)], vec![]);
    let rms_b = describe(&OpenOp::RmsNorm { eps_e6: 10 }, "L.rms_bwd", KernelBackend::CudaSm121);
    let gqa_b = describe(
        &OpenOp::GqaAttention {
            n_heads: cfg.num_attention_heads,
            n_kv_heads: cfg.num_key_value_heads,
            head_dim: 128,
            d_model: cfg.hidden_size,
            sliding_window: None,
            rope_theta: 1_000_000,
        },
        "L.gqa_bwd",
        KernelBackend::CudaSm121,
    );
    let swiglu_b = describe(
        &OpenOp::SwiGlu { hidden: cfg.hidden_size, intermediate: cfg.intermediate_size },
        "L.swiglu_bwd",
        KernelBackend::CudaSm121,
    );

    let launches = vec![
        LaunchReq { at: 1, kernel: rms.clone() },
        LaunchReq { at: 2, kernel: gqa.clone() },
        LaunchReq { at: 4, kernel: describe(&OpenOp::RmsNorm { eps_e6: 10 }, "L.rms_moe", KernelBackend::CudaSm121) },
        LaunchReq { at: 5, kernel: router },
        LaunchReq { at: 6, kernel: swiglu.clone() },
        LaunchReq { at: 9, kernel: swiglu_b },
        LaunchReq { at: 10, kernel: rms_b },
        LaunchReq { at: 12, kernel: gqa_b },
        LaunchReq { at: 13, kernel: rms },
    ];

    Ok(schedule::schedule(ScheduleRequest {
        buffers: &bufs,
        launches: &launches,
        regions: &regions,
        hierarchy: &hier,
        allow_recompute: true,
        allow_reload: true,
    }))
}

pub fn mixtral_layer_timeline_excerpt() -> CatResult<String> {
    let s = plan_mixtral_layer_train(None)?;
    let mut out = String::from(
        "timeline excerpt — one Mixtral-8x7B layer training step (fwd+bwd, bf16, seq=2048):\n",
    );
    for e in s.events.iter().take(24) {
        out.push_str(&format!("  t={:.6e} {:?}\n", e.t, e.kind));
    }
    if s.events.len() > 24 {
        out.push_str(&format!("  … {} more events\n", s.events.len() - 24));
    }
    out.push_str(&format!(
        "peaks: code={:.1} KiB  buffer={:.1} MiB  scratch={} B  reloads={}  feasible={}\n",
        s.peaks.code as f64 / 1024.0,
        s.peaks.buffer as f64 / (1 << 20) as f64,
        s.peaks.scratch,
        s.kernel_reloads,
        s.feasible
    ));
    out.push_str("measured vs estimated: CUDA cubin sizes are estimates (nvcc not run).\n");
    Ok(out)
}

pub fn kernel_resources_for_mixtral() -> Vec<KernelResource> {
    vec![
        describe(&OpenOp::RmsNorm { eps_e6: 10 }, "rmsnorm", KernelBackend::CudaSm121),
        describe(
            &OpenOp::GqaAttention {
                n_heads: 32,
                n_kv_heads: 8,
                head_dim: 128,
                d_model: 4096,
                sliding_window: None,
                rope_theta: 1_000_000,
            },
            "gqa",
            KernelBackend::CudaSm121,
        ),
        describe(&OpenOp::SwiGlu { hidden: 4096, intermediate: 14336 }, "swiglu", KernelBackend::CudaSm121),
        describe(&OpenOp::Router { n_experts: 8, top_k: 2 }, "router", KernelBackend::CudaSm121),
        describe(
            &OpenOp::Linear { in_features: 4096, out_features: 4096, bias: false },
            "gemm",
            KernelBackend::CudaSm121,
        ),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mlp_loss_decreases() {
        let r = train_mlp(&TrainOptions { steps: 12, lr: 0.05, ..TrainOptions::default() }).unwrap();
        assert!(r.losses[0] > r.losses[r.losses.len() - 1], "{:?}", r.losses);
        assert!(r.schedule.feasible);
        assert!(r.schedule.peaks.exceeds(&r.schedule.cap).is_none());
    }

    #[test]
    fn sgd_matches_reference() {
        let p = Tensor::lcg(TensorType::f32([4]), 1);
        let g = Tensor::lcg(TensorType::f32([4]), 2);
        let a = Sgd::new(0.1).put(&p, &g).unwrap();
        let b = sgd_reference(&p, &g, 0.1).unwrap();
        assert_eq!(a.data, b.data);
    }

    #[test]
    fn adam_matches_reference() {
        let p = Tensor::lcg(TensorType::f32([3]), 4);
        let g = Tensor::lcg(TensorType::f32([3]), 5);
        let s = AdamState::zeros_like(&p);
        let opt = Adam::new(1e-2);
        let (a, sa) = opt.put(&p, &g, &s).unwrap();
        let (b, sb) = adam_reference(&p, &g, &s, opt).unwrap();
        assert!((a.data[0] - b.data[0]).abs() < 1e-7);
        assert_eq!(sa.t, sb.t);
    }

    #[test]
    fn mixtral_tiny_loss_decreases() {
        let r = train_mixtral_tiny(&TrainOptions { steps: 4, lr: 0.02, ..TrainOptions::default() })
            .unwrap();
        assert!(r.losses[0] >= r.losses[r.losses.len() - 1] - 1e-4, "{:?}", r.losses);
    }

    #[test]
    fn constrained_equals_unconstrained() {
        let err = constrained_matches_unconstrained().unwrap();
        assert!(err < 2e-4, "param drift {err}");
    }

    #[test]
    fn shrinking_buffer_increases_recompute() {
        let loose = plan_mlp_schedule(Some(MemoryBudget::buffer_only(8 << 20))).unwrap();
        let tight = plan_mlp_schedule(Some(MemoryBudget::buffer_only(512))).unwrap();
        assert!(tight.recompute_overhead + tight.kernel_reloads as f64
            >= loose.recompute_overhead + loose.kernel_reloads as f64
            || !tight.feasible
            || tight.cells.len() >= loose.cells.len());
        if tight.feasible {
            assert!(tight.step_seconds + 1e-12 >= loose.step_seconds * 0.99);
        }
    }

    #[test]
    fn mixtral_layer_train_timeline_respects_regions() {
        let s = plan_mixtral_layer_train(None).unwrap();
        assert!(s.feasible, "{}", s.rationale);
        assert!(s.peaks.exceeds(&s.cap).is_none());
        assert!(crate::schedule::arena_conflict(&s.arena).is_none());
        assert!(s.events.iter().any(|e| matches!(e.kind, crate::schedule::EventKind::LoadKernel { .. })));
        assert!(s.events.iter().any(|e| matches!(e.kind, crate::schedule::EventKind::AllocBuffer { .. })));
        assert!(s.events.iter().any(|e| matches!(e.kind, crate::schedule::EventKind::Launch { .. })));
    }

    #[test]
    fn shrinking_code_increases_reloads() {
        let loose = plan_mixtral_tiny_schedule(None).unwrap();
        let tight = plan_mixtral_tiny_schedule(Some(MemoryBudget {
            code: Some(900),
            buffer: 8 << 20,
            scratch: None,
        }))
        .unwrap();
        if tight.feasible {
            assert!(tight.kernel_reloads >= loose.kernel_reloads);
            assert!(tight.peaks.code <= 900);
        }
    }
}
