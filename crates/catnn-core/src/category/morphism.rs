//! 1-morphisms of AISys: open components and structural maps of `Tens`.
//!
//! A [`Hom`] is a typed arrow `dom → cod`. The body is either a structural
//! map of the SMC (id, associator, unitor, braid), an [`OpenComponent`],
//! a sequential/parallel composite, or a structured cospan.

use serde::{Deserialize, Serialize};

use crate::category::interface::Interface;
use crate::error::{CatError, CatResult};
use crate::lens::open::OpenComponent;
use crate::lens::wiring::StructuredCospan;
use crate::tensor::{DType, Shape, TensorType};

/// Atomic neural / structural operations. These are the generators of the
/// hom-sets that the compiler knows how to lower.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OpenOp {
    Identity,
    Linear {
        in_features: usize,
        out_features: usize,
        bias: bool,
    },
    Relu,
    Gelu,
    Softmax {
        axis: i32,
    },
    LayerNorm {
        eps_e6: u32,
    },
    Add,
    Mul,
    MatMul,
    Transpose {
        dim0: usize,
        dim1: usize,
    },
    Reshape {
        out: Vec<usize>,
    },
    Attention {
        n_heads: usize,
        d_model: usize,
        d_head: usize,
    },
    ResidualAdd,
    /// Token embedding table `V × H`. Input is integer token ids `[B,S]`.
    Embedding {
        vocab: usize,
        hidden: usize,
    },
    /// RMSNorm (Mixtral / Llama): `x * rsqrt(mean(x²) + eps) * w`. No bias.
    RmsNorm {
        eps_e6: u32,
    },
    /// Grouped-query attention + RoPE, optional sliding window.
    GqaAttention {
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        d_model: usize,
        /// `None` matches official HF `sliding_window: null`.
        sliding_window: Option<usize>,
        /// RoPE θ as an integer (HF `rope_theta`, e.g. 1_000_000).
        rope_theta: u64,
    },
    /// SwiGLU expert: `w2(silu(w1 x) ⊙ (w3 x))`.
    SwiGlu {
        hidden: usize,
        intermediate: usize,
    },
    /// Softmax router producing top-k of n experts. An *optic*, not a lens.
    Router {
        n_experts: usize,
        top_k: usize,
    },
    /// Indexed family / coproduct of SwiGLU experts + router + combine.
    MoE {
        n_experts: usize,
        top_k: usize,
        hidden: usize,
        intermediate: usize,
    },
    /// Untied LM head: `H → V`.
    LmHead {
        hidden: usize,
        vocab: usize,
    },
}

impl OpenOp {
    pub fn name(&self) -> &'static str {
        match self {
            OpenOp::Identity => "id",
            OpenOp::Linear { .. } => "linear",
            OpenOp::Relu => "relu",
            OpenOp::Gelu => "gelu",
            OpenOp::Softmax { .. } => "softmax",
            OpenOp::LayerNorm { .. } => "layernorm",
            OpenOp::Add => "add",
            OpenOp::Mul => "mul",
            OpenOp::MatMul => "matmul",
            OpenOp::Transpose { .. } => "transpose",
            OpenOp::Reshape { .. } => "reshape",
            OpenOp::Attention { .. } => "attention",
            OpenOp::ResidualAdd => "residual_add",
            OpenOp::Embedding { .. } => "embedding",
            OpenOp::RmsNorm { .. } => "rmsnorm",
            OpenOp::GqaAttention { .. } => "gqa",
            OpenOp::SwiGlu { .. } => "swiglu",
            OpenOp::Router { .. } => "router",
            OpenOp::MoE { .. } => "moe",
            OpenOp::LmHead { .. } => "lm_head",
        }
    }

    /// Infer output tensor type from the primary input (and, for binary ops,
    /// the secondary input). Used by the DSL and the textual parser.
    pub fn infer_output(
        &self,
        input: &TensorType,
        aux: Option<&TensorType>,
    ) -> CatResult<TensorType> {
        match self {
            OpenOp::Identity
            | OpenOp::Relu
            | OpenOp::Gelu
            | OpenOp::Softmax { .. }
            | OpenOp::LayerNorm { .. } => Ok(input.clone()),
            OpenOp::Linear { out_features, .. } => {
                let mut dims = input.shape.dims.clone();
                if dims.is_empty() {
                    return Err(CatError::ShapeMismatch("linear on scalar".into()));
                }
                *dims.last_mut().unwrap() = *out_features;
                Ok(TensorType { shape: Shape { dims }, dtype: input.dtype })
            }
            OpenOp::MatMul => {
                let rhs = aux.ok_or_else(|| CatError::ShapeMismatch("matmul needs rhs".into()))?;
                if input.dtype != rhs.dtype {
                    return Err(CatError::DTypeMismatch {
                        left: input.dtype.to_string(),
                        right: rhs.dtype.to_string(),
                    });
                }
                Ok(TensorType { shape: input.shape.matmul(&rhs.shape)?, dtype: input.dtype })
            }
            OpenOp::Add | OpenOp::Mul | OpenOp::ResidualAdd => {
                if let Some(rhs) = aux {
                    Ok(TensorType {
                        shape: input.shape.broadcast_add(&rhs.shape)?,
                        dtype: input.dtype,
                    })
                } else {
                    Ok(input.clone())
                }
            }
            OpenOp::Transpose { dim0, dim1 } => {
                let mut dims = input.shape.dims.clone();
                if *dim0 >= dims.len() || *dim1 >= dims.len() {
                    return Err(CatError::ShapeMismatch(format!(
                        "transpose dims {dim0},{dim1} on rank {}",
                        dims.len()
                    )));
                }
                dims.swap(*dim0, *dim1);
                Ok(TensorType { shape: Shape { dims }, dtype: input.dtype })
            }
            OpenOp::Reshape { out } => {
                let want: usize = out.iter().product();
                if want != input.numel() {
                    return Err(CatError::ShapeMismatch(format!(
                        "reshape {} -> {out:?} changes numel",
                        input.shape
                    )));
                }
                Ok(TensorType { shape: Shape::new(out.clone()), dtype: input.dtype })
            }
            OpenOp::Attention { d_model, .. } | OpenOp::GqaAttention { d_model, .. } => {
                let mut dims = input.shape.dims.clone();
                if dims.len() < 2 {
                    return Err(CatError::ShapeMismatch("attention needs [B,S,D]".into()));
                }
                *dims.last_mut().unwrap() = *d_model;
                Ok(TensorType { shape: Shape { dims }, dtype: input.dtype })
            }
            OpenOp::Embedding { hidden, .. } => {
                let mut dims = input.shape.dims.clone();
                dims.push(*hidden);
                Ok(TensorType { shape: Shape { dims }, dtype: DType::F32 })
            }
            OpenOp::RmsNorm { .. } | OpenOp::SwiGlu { .. } | OpenOp::MoE { .. } => {
                Ok(input.clone())
            }
            OpenOp::Router { n_experts, .. } => {
                let mut dims = input.shape.dims.clone();
                if dims.is_empty() {
                    return Err(CatError::ShapeMismatch("router on scalar".into()));
                }
                *dims.last_mut().unwrap() = *n_experts;
                Ok(TensorType { shape: Shape { dims }, dtype: input.dtype })
            }
            OpenOp::LmHead { vocab, .. } => {
                let mut dims = input.shape.dims.clone();
                if dims.is_empty() {
                    return Err(CatError::ShapeMismatch("lm_head on scalar".into()));
                }
                *dims.last_mut().unwrap() = *vocab;
                Ok(TensorType { shape: Shape { dims }, dtype: input.dtype })
            }
        }
    }

    pub fn state_interface(&self, name: &str, in_features: Option<usize>) -> Interface {
        match self {
            OpenOp::Linear { in_features, out_features, bias } => {
                let mut iface = Interface::new(format!("{name}.S"))
                    .port("w", TensorType::f32([*in_features, *out_features]));
                if *bias {
                    iface = iface.port("b", TensorType::f32([*out_features]));
                }
                iface
            }
            OpenOp::LayerNorm { .. } => {
                let n = in_features.unwrap_or(1);
                Interface::new(format!("{name}.S"))
                    .port("gamma", TensorType::f32([n]))
                    .port("beta", TensorType::f32([n]))
            }
            OpenOp::Attention { d_model, .. } => {
                let d = *d_model;
                Interface::new(format!("{name}.S"))
                    .port("wq", TensorType::f32([d, d]))
                    .port("wk", TensorType::f32([d, d]))
                    .port("wv", TensorType::f32([d, d]))
                    .port("wo", TensorType::f32([d, d]))
            }
            OpenOp::Embedding { vocab, hidden } | OpenOp::LmHead { vocab, hidden } => {
                Interface::new(format!("{name}.S")).port("w", TensorType::f32([*vocab, *hidden]))
            }
            OpenOp::RmsNorm { .. } => {
                let n = in_features.unwrap_or(1);
                Interface::new(format!("{name}.S")).port("w", TensorType::f32([n]))
            }
            OpenOp::GqaAttention { n_kv_heads, head_dim, d_model, .. } => {
                let d = *d_model;
                let kv = *n_kv_heads * *head_dim;
                Interface::new(format!("{name}.S"))
                    .port("wq", TensorType::f32([d, d]))
                    .port("wk", TensorType::f32([d, kv]))
                    .port("wv", TensorType::f32([d, kv]))
                    .port("wo", TensorType::f32([d, d]))
            }
            OpenOp::SwiGlu { hidden, intermediate } => Interface::new(format!("{name}.S"))
                .port("w1", TensorType::f32([*hidden, *intermediate]))
                .port("w3", TensorType::f32([*hidden, *intermediate]))
                .port("w2", TensorType::f32([*intermediate, *hidden])),
            OpenOp::Router { n_experts, .. } => {
                let h = in_features.unwrap_or(1);
                Interface::new(format!("{name}.S")).port("gate", TensorType::f32([h, *n_experts]))
            }
            OpenOp::MoE { n_experts, hidden, intermediate, .. } => {
                let mut iface = Interface::new(format!("{name}.S"))
                    .port("gate", TensorType::f32([*hidden, *n_experts]));
                for e in 0..*n_experts {
                    iface = iface
                        .port(format!("e{e}.w1"), TensorType::f32([*hidden, *intermediate]))
                        .port(format!("e{e}.w3"), TensorType::f32([*hidden, *intermediate]))
                        .port(format!("e{e}.w2"), TensorType::f32([*intermediate, *hidden]));
                }
                iface
            }
            _ => Interface::new(format!("{name}.S")),
        }
    }
}

/// Body of a 1-morphism. Recursive composites keep the categorical
/// derivation tree; the compiler flattens it to an IR graph.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum HomBody {
    Id,
    Associator,
    LeftUnitor,
    RightUnitor,
    Braid,
    Component(OpenComponent),
    Seq(Box<Hom>, Box<Hom>),
    Par(Box<Hom>, Box<Hom>),
    Cospan(StructuredCospan),
}

/// A typed 1-morphism of `Tens` / AISys.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Hom {
    pub name: String,
    pub dom: Interface,
    pub cod: Interface,
    pub body: HomBody,
}

impl Hom {
    pub fn component(c: OpenComponent) -> Self {
        Hom {
            name: c.name.clone(),
            dom: c.input.clone(),
            cod: c.output.clone(),
            body: HomBody::Component(c),
        }
    }

    /// Walk the derivation tree collecting atomic components (not structural maps).
    pub fn components(&self) -> Vec<&OpenComponent> {
        match &self.body {
            HomBody::Component(c) => vec![c],
            HomBody::Seq(f, g) | HomBody::Par(f, g) => {
                let mut v = f.components();
                v.extend(g.components());
                v
            }
            HomBody::Cospan(c) => c.components(),
            _ => vec![],
        }
    }

    pub fn is_structural(&self) -> bool {
        matches!(
            self.body,
            HomBody::Id
                | HomBody::Associator
                | HomBody::LeftUnitor
                | HomBody::RightUnitor
                | HomBody::Braid
        )
    }
}

/// Convenience: a binary interface `{lhs, rhs}` of the same dtype.
pub fn binary_interface(name: &str, lhs: TensorType, rhs: TensorType) -> Interface {
    Interface::new(name).port("lhs", lhs).port("rhs", rhs)
}

pub fn f32_activation(name: &str, shape: impl Into<Vec<usize>>) -> (Interface, Interface) {
    let ty = TensorType::new(shape, DType::F32);
    (
        Interface::single(format!("{name}.in"), ty.clone()),
        Interface::single(format!("{name}.out"), ty),
    )
}
