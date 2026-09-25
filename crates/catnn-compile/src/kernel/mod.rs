//! Kernel backends: executable CPU, generated+validated WGSL, naive reference.

pub mod adjoint;
pub mod cpu;
pub mod cuda;
pub mod fd64;
pub mod measure;
pub mod moe;
pub mod reference;
pub mod resource;
pub mod wgsl;

use catnn_core::category::OpenOp;
use catnn_core::tensor::Tensor;
use catnn_core::CatResult;

use crate::ir::IrNode;

/// Dispatch a node on the CPU backend (always executable).
pub fn eval_cpu(node: &IrNode, inputs: &[Tensor]) -> CatResult<Tensor> {
    if inputs.is_empty() && matches!(node.op, OpenOp::Identity) {
        return Ok(Tensor::zeros(node.output_ty.clone()));
    }
    let x = &inputs[0];
    match &node.op {
        OpenOp::Identity | OpenOp::Reshape { .. } | OpenOp::Transpose { .. } => Ok(x.clone()),
        OpenOp::Relu => Ok(cpu::relu(x)),
        OpenOp::Gelu => Ok(cpu::gelu(x)),
        OpenOp::Softmax { .. } => cpu::softmax_last(x),
        OpenOp::Add | OpenOp::ResidualAdd => {
            if inputs.len() > 1 {
                cpu::add(x, &inputs[1])
            } else {
                Ok(x.clone())
            }
        }
        OpenOp::Mul => {
            if inputs.len() > 1 {
                cpu::mul(x, &inputs[1])
            } else {
                Ok(x.clone())
            }
        }
        OpenOp::Linear { .. } | OpenOp::MatMul => {
            let w = node
                .params
                .get("w")
                .or_else(|| node.params.values().next())
                .ok_or_else(|| catnn_core::CatError::UnknownComponent(node.name.clone()))?;
            let bias = node.params.get("b");
            cpu::linear(x, w, bias)
        }
        OpenOp::LayerNorm { eps_e6 } => {
            let gamma =
                node.params.get("gamma").ok_or_else(|| catnn_core::CatError::UnknownPort {
                    interface: node.name.clone(),
                    port: "gamma".into(),
                })?;
            let beta =
                node.params.get("beta").ok_or_else(|| catnn_core::CatError::UnknownPort {
                    interface: node.name.clone(),
                    port: "beta".into(),
                })?;
            cpu::layer_norm(x, gamma, beta, *eps_e6 as f32 * 1e-6)
        }
        OpenOp::Attention { n_heads, d_head, .. } => {
            let wq = node.params.get("wq").unwrap();
            let wk = node.params.get("wk").unwrap();
            let wv = node.params.get("wv").unwrap();
            let wo = node.params.get("wo").unwrap();
            cpu::attention(x, wq, wk, wv, wo, *n_heads, *d_head)
        }
        OpenOp::RmsNorm { eps_e6 } => {
            let w = node.params.get("w").ok_or_else(|| catnn_core::CatError::UnknownPort {
                interface: node.name.clone(),
                port: "w".into(),
            })?;
            moe::rmsnorm(x, w, *eps_e6 as f32 * 1e-6)
        }
        OpenOp::Embedding { hidden, .. } => {
            let table = node.params.get("w").ok_or_else(|| catnn_core::CatError::UnknownPort {
                interface: node.name.clone(),
                port: "w".into(),
            })?;
            let ids: Vec<u32> =
                x.data.iter().map(|v| v.round().clamp(0.0, u32::MAX as f32) as u32).collect();
            let mut y = moe::embedding(&ids, table, *hidden)?;
            if x.ty.shape.dims.len() == 2 {
                y.ty = catnn_core::tensor::TensorType::f32([
                    x.ty.shape.dims[0],
                    x.ty.shape.dims[1],
                    *hidden,
                ]);
            }
            Ok(y)
        }
        OpenOp::SwiGlu { .. } => {
            let w1 = node.params.get("w1").unwrap();
            let w3 = node.params.get("w3").unwrap();
            let w2 = node.params.get("w2").unwrap();
            moe::swiglu(x, w1, w3, w2)
        }
        OpenOp::MoE { top_k, n_experts, .. } => {
            let gate = node.params.get("gate").unwrap();
            let mut experts = Vec::new();
            for e in 0..*n_experts {
                experts.push((
                    node.params.get(&format!("e{e}.w1")).unwrap().clone(),
                    node.params.get(&format!("e{e}.w3")).unwrap().clone(),
                    node.params.get(&format!("e{e}.w2")).unwrap().clone(),
                ));
            }
            moe::moe(x, gate, &experts, *top_k)
        }
        OpenOp::GqaAttention {
            n_heads, n_kv_heads, head_dim, sliding_window, rope_theta, ..
        } => moe::gqa(
            x,
            node.params.get("wq").unwrap(),
            node.params.get("wk").unwrap(),
            node.params.get("wv").unwrap(),
            node.params.get("wo").unwrap(),
            *n_heads,
            *n_kv_heads,
            *head_dim,
            *rope_theta as f64,
            *sliding_window,
        ),
        OpenOp::LmHead { vocab, .. } => moe::lm_head(x, node.params.get("w").unwrap(), *vocab),
        OpenOp::Router { .. } => Ok(x.clone()),
    }
}

/// Reference evaluation used by 2-cell witnesses.
pub fn eval_reference(node: &IrNode, inputs: &[Tensor]) -> CatResult<Tensor> {
    if inputs.is_empty() && matches!(node.op, OpenOp::Identity) {
        return Ok(Tensor::zeros(node.output_ty.clone()));
    }
    let x = &inputs[0];
    match &node.op {
        OpenOp::Identity | OpenOp::Reshape { .. } | OpenOp::Transpose { .. } => Ok(x.clone()),
        OpenOp::Relu => Ok(reference::relu(x)),
        OpenOp::Gelu => Ok(reference::gelu(x)),
        OpenOp::Softmax { .. } => reference::softmax_last(x),
        OpenOp::Add | OpenOp::ResidualAdd => {
            if inputs.len() > 1 {
                reference::add(x, &inputs[1])
            } else {
                Ok(x.clone())
            }
        }
        OpenOp::Mul => {
            if inputs.len() > 1 {
                crate::kernel::cpu::mul(x, &inputs[1])
            } else {
                Ok(x.clone())
            }
        }
        OpenOp::Linear { .. } | OpenOp::MatMul => {
            let w = node.params.values().next().unwrap();
            let bias = node.params.get("b");
            // Prefer the "w" key when present.
            let w = node.params.get("w").unwrap_or(w);
            reference::linear(x, w, bias)
        }
        OpenOp::LayerNorm { eps_e6 } => {
            let gamma = node.params.get("gamma").unwrap();
            let beta = node.params.get("beta").unwrap();
            reference::layer_norm(x, gamma, beta, *eps_e6 as f32 * 1e-6)
        }
        OpenOp::Attention { n_heads, d_head, .. } => {
            let wq = node.params.get("wq").unwrap();
            let wk = node.params.get("wk").unwrap();
            let wv = node.params.get("wv").unwrap();
            let wo = node.params.get("wo").unwrap();
            reference::attention(x, wq, wk, wv, wo, *n_heads, *d_head)
        }
        OpenOp::RmsNorm { .. }
        | OpenOp::Embedding { .. }
        | OpenOp::SwiGlu { .. }
        | OpenOp::MoE { .. }
        | OpenOp::GqaAttention { .. }
        | OpenOp::LmHead { .. }
        | OpenOp::Router { .. } => eval_cpu(node, inputs),
    }
}
