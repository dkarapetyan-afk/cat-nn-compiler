//! Shared AST for the Rust DSL and the textual/TOML/JSON formats.
//!
//! All three surface syntaxes parse (or build) into [`NetworkSpec`], which
//! is then assembled into an [`crate::architecture::Architecture`].

use serde::{Deserialize, Serialize};

use crate::architecture::Architecture;
use crate::category::interface::Interface;
use crate::category::morphism::OpenOp;
use crate::error::{CatError, CatResult};
use crate::lens::open::OpenComponent;
use crate::tensor::{DType, TensorType};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NetworkSpec {
    pub name: String,
    #[serde(default)]
    pub batch: usize,
    #[serde(default)]
    pub layers: Vec<LayerSpec>,
    #[serde(default)]
    pub compose: ComposeStyle,
}

impl Default for NetworkSpec {
    fn default() -> Self {
        Self {
            name: "net".into(),
            batch: 1,
            layers: vec![],
            compose: ComposeStyle::Sequential,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ComposeStyle {
    #[default]
    Sequential,
    Parallel,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum LayerSpec {
    Linear {
        name: String,
        in_features: usize,
        out_features: usize,
        #[serde(default = "default_true")]
        bias: bool,
    },
    Relu {
        name: String,
    },
    Gelu {
        name: String,
    },
    Softmax {
        name: String,
        #[serde(default = "neg_one")]
        axis: i32,
    },
    LayerNorm {
        name: String,
        features: usize,
    },
    Attention {
        name: String,
        seq: usize,
        d_model: usize,
        n_heads: usize,
    },
    ResidualAdd {
        name: String,
        #[serde(default)]
        shape: Vec<usize>,
    },
}

fn default_true() -> bool {
    true
}
fn neg_one() -> i32 {
    -1
}

impl LayerSpec {
    pub fn name(&self) -> &str {
        match self {
            LayerSpec::Linear { name, .. }
            | LayerSpec::Relu { name }
            | LayerSpec::Gelu { name }
            | LayerSpec::Softmax { name, .. }
            | LayerSpec::LayerNorm { name, .. }
            | LayerSpec::Attention { name, .. }
            | LayerSpec::ResidualAdd { name, .. } => name,
        }
    }
}

impl NetworkSpec {
    pub fn to_architecture(&self) -> CatResult<Architecture> {
        let batch = if self.batch == 0 { 1 } else { self.batch };
        let mut parts: Vec<OpenComponent> = Vec::new();
        let mut current_out: Option<Interface> = None;

        for layer in &self.layers {
            let c = match layer {
                LayerSpec::Linear { name, in_features, out_features, bias } => {
                    match &current_out {
                        Some(prev) => OpenComponent::linear_on(
                            name,
                            prev.clone(),
                            *in_features,
                            *out_features,
                            *bias,
                        ),
                        None => {
                            OpenComponent::linear(name, batch, *in_features, *out_features, *bias)
                        }
                    }
                }
                LayerSpec::Relu { name } => {
                    let input = current_out.clone().ok_or_else(|| {
                        CatError::Parse(format!("relu {name} has no preceding interface"))
                    })?;
                    OpenComponent::relu(name, input)
                }
                LayerSpec::Gelu { name } => {
                    let input = current_out.clone().ok_or_else(|| {
                        CatError::Parse(format!("gelu {name} has no preceding interface"))
                    })?;
                    OpenComponent::gelu(name, input)
                }
                LayerSpec::Softmax { name, axis } => {
                    let input = current_out.clone().ok_or_else(|| {
                        CatError::Parse(format!("softmax {name} has no preceding interface"))
                    })?;
                    OpenComponent::softmax(name, input, *axis)
                }
                LayerSpec::LayerNorm { name, features } => {
                    let input = current_out.clone().ok_or_else(|| {
                        CatError::Parse(format!("layernorm {name} has no preceding interface"))
                    })?;
                    OpenComponent::layer_norm(name, input, *features)
                }
                LayerSpec::Attention { name, seq, d_model, n_heads } => {
                    OpenComponent::attention(name, batch, *seq, *d_model, *n_heads)?
                }
                LayerSpec::ResidualAdd { name, shape } => {
                    let ty = if shape.is_empty() {
                        current_out
                            .as_ref()
                            .and_then(|i| i.ports.values().next().cloned())
                            .unwrap_or_else(|| TensorType::f32([batch, 1]))
                    } else {
                        TensorType::new(shape.clone(), DType::F32)
                    };
                    OpenComponent::residual_add(name, ty)
                }
            };
            current_out = Some(c.output.clone());
            parts.push(c);
        }

        if parts.is_empty() {
            return Err(CatError::Parse("network has no layers".into()));
        }
        let _ = OpenOp::Identity;
        Architecture::from_components(&self.name, parts)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spec_to_arch_mlp() {
        let spec = NetworkSpec {
            name: "mlp".into(),
            batch: 4,
            layers: vec![
                LayerSpec::Linear {
                    name: "fc1".into(),
                    in_features: 16,
                    out_features: 32,
                    bias: true,
                },
                LayerSpec::Relu { name: "act".into() },
                LayerSpec::Linear {
                    name: "fc2".into(),
                    in_features: 32,
                    out_features: 8,
                    bias: true,
                },
            ],
            compose: ComposeStyle::Sequential,
        };
        let arch = spec.to_architecture().unwrap();
        assert_eq!(arch.components().len(), 3);
    }
}
