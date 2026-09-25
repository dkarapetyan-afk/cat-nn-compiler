//! Open dynamical systems as lenses (spec §4).
//!
//! A component is `(S, U, Y, update: U⊗S → S, readout)`.
//! For neural layers we use the *Mealy* form where readout also sees `U`
//! (otherwise a feedforward layer cannot depend on its input). Inference
//! update is the identity on `S` (weights are constant); the training
//! update would be a gradient step and is not implemented here.
//!
//! Sequential composition of Moore/Mealy machines is categorical
//! composition in the category of open systems, not string concatenation.

use serde::{Deserialize, Serialize};

use crate::category::interface::Interface;
use crate::category::morphism::{Hom, HomBody, OpenOp};
use crate::error::{CatError, CatResult};
use crate::tensor::TensorType;

/// An open component: a 1-morphism of AISys with explicit state.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OpenComponent {
    pub name: String,
    /// Internal state `S` (weights, caches).
    pub state: Interface,
    /// Input `U`.
    pub input: Interface,
    /// Output `Y`.
    pub output: Interface,
    pub op: OpenOp,
    /// `true` = Moore (readout `S → Y`); `false` = Mealy (`U⊗S → Y`).
    pub moore: bool,
}

impl OpenComponent {
    pub fn new(
        name: impl Into<String>,
        state: Interface,
        input: Interface,
        output: Interface,
        op: OpenOp,
    ) -> Self {
        Self { name: name.into(), state, input, output, op, moore: false }
    }

    pub fn identity(name: impl Into<String>, iface: Interface) -> Self {
        let name = name.into();
        Self {
            name: name.clone(),
            state: Interface::new(format!("{name}.S")),
            input: iface.clone(),
            output: iface,
            op: OpenOp::Identity,
            moore: false,
        }
    }

    pub fn linear(
        name: impl Into<String>,
        batch: usize,
        in_features: usize,
        out_features: usize,
        bias: bool,
    ) -> Self {
        let name = name.into();
        let input = Interface::single(format!("{name}.U"), TensorType::f32([batch, in_features]));
        Self::linear_on(name, input, in_features, out_features, bias)
            .expect("constructed input matches in_features")
    }

    /// Linear map on the last axis of an existing interface. Leading
    /// dimensions (batch, sequence, …) are preserved — this is what makes
    /// a transformer block type-check after attention.
    pub fn linear_on(
        name: impl Into<String>,
        input: Interface,
        in_features: usize,
        out_features: usize,
        bias: bool,
    ) -> CatResult<Self> {
        let name = name.into();
        let in_ty =
            input.ports.values().next().cloned().unwrap_or_else(|| TensorType::f32([in_features]));
        let last = in_ty.shape.dims.last().copied().unwrap_or(0);
        if last != in_features {
            return Err(CatError::ShapeMismatch(format!(
                "linear {name}: last dim {last} != in_features {in_features}"
            )));
        }
        let mut out_dims = in_ty.shape.dims.clone();
        if out_dims.is_empty() {
            out_dims.push(out_features);
        } else {
            *out_dims.last_mut().unwrap() = out_features;
        }
        let output = Interface::single(format!("{name}.Y"), TensorType::f32(out_dims));
        let op = OpenOp::Linear { in_features, out_features, bias };
        let state = op.state_interface(&name, Some(in_features));
        Ok(Self { name, state, input, output, op, moore: false })
    }

    pub fn relu(name: impl Into<String>, input: Interface) -> Self {
        let name = name.into();
        Self {
            name: name.clone(),
            state: Interface::new(format!("{name}.S")),
            output: {
                let mut o = input.clone();
                o.name = format!("{name}.Y");
                o
            },
            input,
            op: OpenOp::Relu,
            moore: false,
        }
    }

    pub fn gelu(name: impl Into<String>, input: Interface) -> Self {
        let mut c = Self::relu(name, input);
        c.op = OpenOp::Gelu;
        c
    }

    pub fn softmax(name: impl Into<String>, input: Interface, axis: i32) -> Self {
        let mut c = Self::relu(name, input);
        c.op = OpenOp::Softmax { axis };
        c
    }

    pub fn layer_norm(name: impl Into<String>, input: Interface, features: usize) -> Self {
        let name = name.into();
        let op = OpenOp::LayerNorm { eps_e6: 1 };
        let state = op.state_interface(&name, Some(features));
        Self {
            name: name.clone(),
            state,
            output: {
                let mut o = input.clone();
                o.name = format!("{name}.Y");
                o
            },
            input,
            op,
            moore: false,
        }
    }

    pub fn attention(
        name: impl Into<String>,
        batch: usize,
        seq: usize,
        d_model: usize,
        n_heads: usize,
    ) -> CatResult<Self> {
        if n_heads == 0 || d_model % n_heads != 0 {
            return Err(CatError::ShapeMismatch(format!(
                "d_model {d_model} not divisible by n_heads {n_heads}"
            )));
        }
        let name = name.into();
        let d_head = d_model / n_heads;
        let input = Interface::single(format!("{name}.U"), TensorType::f32([batch, seq, d_model]));
        let output = Interface::single(format!("{name}.Y"), TensorType::f32([batch, seq, d_model]));
        let op = OpenOp::Attention { n_heads, d_model, d_head };
        let state = op.state_interface(&name, Some(d_model));
        Ok(Self { name, state, input, output, op, moore: false })
    }

    pub fn residual_add(name: impl Into<String>, ty: TensorType) -> Self {
        let name = name.into();
        let input =
            Interface::new(format!("{name}.U")).port("skip", ty.clone()).port("branch", ty.clone());
        let output = Interface::single(format!("{name}.Y"), ty);
        Self {
            name: name.clone(),
            state: Interface::new(format!("{name}.S")),
            input,
            output,
            op: OpenOp::ResidualAdd,
            moore: false,
        }
    }

    pub fn as_hom(&self) -> Hom {
        Hom {
            name: self.name.clone(),
            dom: self.input.clone(),
            cod: self.output.clone(),
            body: HomBody::Component(self.clone()),
        }
    }

    /// Sequential composition of open systems: state is the monoidal product,
    /// input of `self`, output of `next`. Requires `self.output ≅ next.input`.
    pub fn then(&self, next: &OpenComponent) -> CatResult<CompositeSystem> {
        self.output.same_as(&next.input).map_err(|_| CatError::InterfaceMismatch {
            expected: self.output.to_string(),
            actual: next.input.to_string(),
        })?;
        Ok(CompositeSystem {
            name: format!("{}⨾{}", self.name, next.name),
            state: self.state.tensor(&next.state),
            input: self.input.clone(),
            output: next.output.clone(),
            parts: vec![self.clone(), next.clone()],
            parallel: false,
        })
    }
}

/// A composite of open systems, still an open system (closure of the category).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompositeSystem {
    pub name: String,
    pub state: Interface,
    pub input: Interface,
    pub output: Interface,
    pub parts: Vec<OpenComponent>,
    pub parallel: bool,
}

impl CompositeSystem {
    pub fn then(&self, next: &OpenComponent) -> CatResult<CompositeSystem> {
        self.output.same_as(&next.input).map_err(|_| CatError::InterfaceMismatch {
            expected: self.output.to_string(),
            actual: next.input.to_string(),
        })?;
        let mut parts = self.parts.clone();
        parts.push(next.clone());
        Ok(CompositeSystem {
            name: format!("{}⨾{}", self.name, next.name),
            state: self.state.tensor(&next.state),
            input: self.input.clone(),
            output: next.output.clone(),
            parts,
            parallel: false,
        })
    }

    pub fn then_composite(&self, next: &CompositeSystem) -> CatResult<CompositeSystem> {
        self.output.same_as(&next.input).map_err(|_| CatError::InterfaceMismatch {
            expected: self.output.to_string(),
            actual: next.input.to_string(),
        })?;
        let mut parts = self.parts.clone();
        parts.extend(next.parts.iter().cloned());
        Ok(CompositeSystem {
            name: format!("{}⨾{}", self.name, next.name),
            state: self.state.tensor(&next.state),
            input: self.input.clone(),
            output: next.output.clone(),
            parts,
            parallel: false,
        })
    }

    /// Monoidal / parallel product of two open systems (independent wires).
    pub fn tensor(&self, other: &CompositeSystem) -> CompositeSystem {
        let mut parts = self.parts.clone();
        parts.extend(other.parts.iter().cloned());
        CompositeSystem {
            name: format!("{}⊗{}", self.name, other.name),
            state: self.state.tensor(&other.state),
            input: self.input.tensor(&other.input),
            output: self.output.tensor(&other.output),
            parts,
            parallel: true,
        }
    }

    pub fn as_hom(&self) -> Hom {
        let homs: Vec<Hom> = self.parts.iter().map(|c| c.as_hom()).collect();
        if self.parallel {
            homs.iter().cloned().reduce(|a, b| crate::category::smc::Tens::tensor(&a, &b)).unwrap()
        } else {
            crate::category::smc::Tens::compose_all(&homs).expect("parts already type-checked")
        }
    }

    pub fn singleton(c: OpenComponent) -> Self {
        CompositeSystem {
            name: c.name.clone(),
            state: c.state.clone(),
            input: c.input.clone(),
            output: c.output.clone(),
            parts: vec![c],
            parallel: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sequential_mlp_types() {
        let fc1 = OpenComponent::linear("fc1", 4, 16, 32, true);
        let act = OpenComponent::relu("act", fc1.output.clone());
        let fc2 = OpenComponent::linear("fc2", 4, 32, 8, true);
        let net = fc1.then(&act).unwrap().then(&fc2).unwrap();
        assert_eq!(net.parts.len(), 3);
        assert_eq!(net.output.ports["data"].shape.dims, vec![4, 8]);
    }

    #[test]
    fn sequential_rejects_bad_wiring() {
        let fc1 = OpenComponent::linear("fc1", 4, 16, 32, true);
        let fc2 = OpenComponent::linear("fc2", 4, 8, 8, true);
        assert!(fc1.then(&fc2).is_err());
    }
}
