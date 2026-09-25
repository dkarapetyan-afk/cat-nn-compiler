//! Parametric lenses `Para(Lens)` — reverse-mode AD as a 1-morphism
//! (Cruttwell, Gavranović, Ghani, Wilson, Zanasi, *Categorical Foundations
//! of Gradient-Based Learning*).
//!
//! A parametric map `f: P × X → Y` is a lens whose **get** is the forward
//! pass and whose **put** sends an output cotangent to an input cotangent
//! *and* a parameter cotangent. Sequential composition of these lenses *is*
//! the chain rule: we do not bolt on a separate autograd engine.

use serde::{Deserialize, Serialize};

use crate::category::morphism::OpenOp;
use crate::error::CatResult;
use crate::tensor::Tensor;

/// A parametric 1-morphism `P × X → Y` with an adjoint `put`.
///
/// The body is a derivation tree. Execution (get/put on tensors) lives in
/// `catnn-compile::kernel::adjoint`; this module is the categorical skeleton.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParaMorphism {
    pub name: String,
    pub body: ParaBody,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ParaBody {
    /// Generator: one [`OpenOp`] (plus loss, treated as a 1-morphism).
    Generator(ParaGen),
    /// Sequential composite `g ∘ f`. Parameters are the product `P_f × P_g`.
    Seq(Box<ParaMorphism>, Box<ParaMorphism>),
}

/// Generators of `Para(Lens)` used by this compiler.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ParaGen {
    Op(OpenOp),
    /// Mean softmax cross-entropy: logits `[..., V]` and integer targets.
    CrossEntropy,
    /// Mean squared error against a same-shaped target.
    Mse,
}

impl ParaMorphism {
    pub fn op(name: impl Into<String>, op: OpenOp) -> Self {
        Self { name: name.into(), body: ParaBody::Generator(ParaGen::Op(op)) }
    }

    pub fn loss_ce() -> Self {
        Self { name: "cross_entropy".into(), body: ParaBody::Generator(ParaGen::CrossEntropy) }
    }

    pub fn loss_mse() -> Self {
        Self { name: "mse".into(), body: ParaBody::Generator(ParaGen::Mse) }
    }

    /// Sequential composition. `put` of the result is the chain rule.
    pub fn then(self, g: ParaMorphism) -> Self {
        Self {
            name: format!("({}∘{})", g.name, self.name),
            body: ParaBody::Seq(Box::new(self), Box::new(g)),
        }
    }

    pub fn generators(&self) -> Vec<&ParaGen> {
        match &self.body {
            ParaBody::Generator(g) => vec![g],
            ParaBody::Seq(f, g) => {
                let mut v = f.generators();
                v.extend(g.generators());
                v
            }
        }
    }
}

/// Cotangent pair produced by `put`: parameter grads then input grad.
#[derive(Debug, Clone, PartialEq)]
pub struct Cotangent {
    pub dparams: Vec<(String, Tensor)>,
    pub dinput: Tensor,
}

/// A single reverse step of a generator. The compiler supplies the
/// concrete `put` (this type is the *interface* of the law).
pub trait ReverseRule {
    fn get(&self, params: &[(&str, &Tensor)], input: &Tensor) -> CatResult<Tensor>;
    fn put(
        &self,
        params: &[(&str, &Tensor)],
        input: &Tensor,
        dout: &Tensor,
    ) -> CatResult<Cotangent>;
}

/// Witness that `put(g ∘ f)` equals `put_f ∘ put_g` (chain rule).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChainRuleWitness {
    pub left: String,
    pub right: String,
    pub max_abs: f32,
    pub verified: bool,
}

impl ChainRuleWitness {
    pub fn accept(left: impl Into<String>, right: impl Into<String>, max_abs: f32, tol: f32) -> Self {
        Self {
            left: left.into(),
            right: right.into(),
            max_abs,
            verified: max_abs <= tol,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::category::morphism::OpenOp;

    #[test]
    fn compose_is_a_seq_tree() {
        let f = ParaMorphism::op("fc", OpenOp::Linear { in_features: 4, out_features: 8, bias: true });
        let g = ParaMorphism::op("act", OpenOp::Relu);
        let h = f.then(g);
        assert!(matches!(h.body, ParaBody::Seq(_, _)));
        assert_eq!(h.generators().len(), 2);
    }
}
