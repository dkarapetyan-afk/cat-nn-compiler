//! Parameter-update lenses / optics (SGD, Adam).
//!
//! SGD is a lens `P × dP → P`. PutGet fails: the view is a cotangent, not a
//! reconstructible snapshot of `P` (same reason the architecture-level
//! optimizer is an optic). Adam is an [`Optic`] whose residual is the
//! moment state `(m, v, t)`.

use serde::{Deserialize, Serialize};

use crate::error::{CatError, CatResult};
use crate::lens::optic::{Optic, UnlawfulWitness};
use crate::tensor::Tensor;

/// SGD hyper-parameters. The lens `put` is `p ↦ p − lr · g`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Sgd {
    pub lr: f32,
}

impl Default for Sgd {
    fn default() -> Self {
        Self { lr: 0.01 }
    }
}

impl Sgd {
    pub fn new(lr: f32) -> Self {
        Self { lr }
    }

    /// `put(p, g) = p − lr · g`.
    pub fn put(&self, p: &Tensor, g: &Tensor) -> CatResult<Tensor> {
        if p.data.len() != g.data.len() {
            return Err(CatError::ShapeMismatch(format!("sgd put {} vs {}", p.ty, g.ty)));
        }
        let mut out = p.clone();
        let lr = self.lr;
        for (o, gi) in out.data.iter_mut().zip(g.data.iter()) {
            *o -= lr * *gi;
        }
        Ok(out)
    }

    /// Get is the identity on parameters (the “current weights” view).
    pub fn get<'a>(&self, p: &'a Tensor) -> &'a Tensor {
        p
    }

    pub fn unlawful() -> UnlawfulWitness {
        UnlawfulWitness {
            optic_name: "sgd_lens".into(),
            law_that_fails: "PutGet".into(),
            explanation: "get(put(p, g)) = p − lr·g ≠ g. The view of the \
                 update lens is a cotangent, not a section of P, so PutGet \
                 fails. GetPut holds only for the zero cotangent."
                .into(),
        }
    }
}

/// Adam residual: first/second moments and step counter.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AdamState {
    pub m: Tensor,
    pub v: Tensor,
    pub t: u64,
}

impl AdamState {
    pub fn zeros_like(p: &Tensor) -> Self {
        Self { m: Tensor::zeros(p.ty.clone()), v: Tensor::zeros(p.ty.clone()), t: 0 }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Adam {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
}

impl Default for Adam {
    fn default() -> Self {
        Self { lr: 1e-3, beta1: 0.9, beta2: 0.999, eps: 1e-8 }
    }
}

impl Adam {
    pub fn new(lr: f32) -> Self {
        Self { lr, ..Self::default() }
    }

    /// One Adam step. Matches the independent naïve reference in tests.
    pub fn put(&self, p: &Tensor, g: &Tensor, s: &AdamState) -> CatResult<(Tensor, AdamState)> {
        if p.data.len() != g.data.len() || p.data.len() != s.m.data.len() {
            return Err(CatError::ShapeMismatch("adam shapes".into()));
        }
        let t = s.t.saturating_add(1);
        let b1 = self.beta1;
        let b2 = self.beta2;
        let mut m = s.m.clone();
        let mut v = s.v.clone();
        let mut out = p.clone();
        let b1t = 1.0 - b1.powi(t as i32);
        let b2t = 1.0 - b2.powi(t as i32);
        for i in 0..p.data.len() {
            let gi = g.data[i];
            let mi = b1 * s.m.data[i] + (1.0 - b1) * gi;
            let vi = b2 * s.v.data[i] + (1.0 - b2) * gi * gi;
            m.data[i] = mi;
            v.data[i] = vi;
            let mhat = mi / b1t.max(1e-30);
            let vhat = vi / b2t.max(1e-30);
            out.data[i] = p.data[i] - self.lr * mhat / (vhat.sqrt() + self.eps);
        }
        Ok((out, AdamState { m, v, t }))
    }

    /// Adam as an optic `P ─(m,v,t)→ dP`. The view is unused (grad arrives
    /// as the `V` argument of [`Optic::step`]).
    pub fn as_optic(&self) -> Optic<Tensor, Tensor, AdamState> {
        // Function pointers cannot capture `self`; callers use [`Adam::put`].
        // The optic object records the *type* of the update for the fibration.
        Optic::new("adam", |p, _s| p.clone(), |p, _g, s| (p.clone(), s.clone()))
    }

    pub fn unlawful() -> UnlawfulWitness {
        UnlawfulWitness {
            optic_name: "adam_optic".into(),
            law_that_fails: "PutGet".into(),
            explanation: "Adam's residual (m, v, t) is why the update is an \
                 optic: observe(update(P, g)) is a new parameter, not g."
                .into(),
        }
    }
}

/// Which update morphism the training compiler realises.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OptimizerKind {
    Sgd,
    Adam,
}

impl OptimizerKind {
    pub fn parse(s: &str) -> CatResult<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "sgd" => Ok(Self::Sgd),
            "adam" => Ok(Self::Adam),
            other => Err(CatError::Parse(format!("unknown optimizer {other}"))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::TensorType;

    #[test]
    fn sgd_put_is_axpy() {
        let p = Tensor::from_vec(TensorType::f32([2]), vec![1.0, 2.0]).unwrap();
        let g = Tensor::from_vec(TensorType::f32([2]), vec![1.0, -1.0]).unwrap();
        let q = Sgd::new(0.5).put(&p, &g).unwrap();
        assert_eq!(q.data, vec![0.5, 2.5]);
    }

    #[test]
    fn sgd_zero_grad_is_getput() {
        let p = Tensor::from_vec(TensorType::f32([2]), vec![3.0, 4.0]).unwrap();
        let z = Tensor::zeros(p.ty.clone());
        let q = Sgd::new(0.1).put(&p, &z).unwrap();
        assert_eq!(q.data, p.data);
    }

    #[test]
    fn adam_first_step_matches_closed_form() {
        let p = Tensor::from_vec(TensorType::f32([1]), vec![0.0]).unwrap();
        let g = Tensor::from_vec(TensorType::f32([1]), vec![1.0]).unwrap();
        let s = AdamState::zeros_like(&p);
        let opt = Adam::new(0.1);
        let (q, s2) = opt.put(&p, &g, &s).unwrap();
        let m = 0.1;
        let v = 0.001;
        let mhat = m / (1.0 - 0.9);
        let vhat: f32 = v / (1.0 - 0.999);
        let expect = 0.0 - 0.1 * mhat / (vhat.sqrt() + 1e-8);
        assert!((q.data[0] - expect).abs() < 1e-6);
        assert_eq!(s2.t, 1);
    }

    #[test]
    fn witnesses_name_putget() {
        assert_eq!(Sgd::unlawful().law_that_fails, "PutGet");
        assert_eq!(Adam::unlawful().law_that_fails, "PutGet");
    }
}
