//! Kernel backends: executable CPU, generated+validated WGSL, naive reference.

pub mod cpu;
pub mod reference;
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
            let gamma = node.params.get("gamma").ok_or_else(|| {
                catnn_core::CatError::UnknownPort {
                    interface: node.name.clone(),
                    port: "gamma".into(),
                }
            })?;
            let beta = node.params.get("beta").ok_or_else(|| {
                catnn_core::CatError::UnknownPort { interface: node.name.clone(), port: "beta".into() }
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
    }
}
