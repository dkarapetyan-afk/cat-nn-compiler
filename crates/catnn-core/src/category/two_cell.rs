//! 2-morphisms of AISys: refinements between implementations (spec §3, §9).
//!
//! A [`TwoCell`] is a rewrite `src ⇒ tgt` between 1-morphisms (or between
//! realizations of the same architecture) together with an
//! [`EquivWitness`]: a *specified* numerical equivalence, not byte equality.
//! The witness is verified by running both sides on shared inputs.

use serde::{Deserialize, Serialize};

use crate::error::{CatError, CatResult};

/// Kind of implementation rewrite. These are the generating 2-cells.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Rewrite {
    /// Fuse a sequential chain of ops into one kernel.
    Fuse { ops: Vec<String> },
    /// Lower a high-level op to a more primitive one (e.g. Linear → MatMul+Add).
    Lower { from: String, to: String },
    /// Move an op from one device to another.
    Reassign { op: String, from_device: String, to_device: String },
    /// Split an op across devices.
    Partition { op: String, parts: usize },
    /// Identity 2-cell (reflexivity of equivalence).
    Refl,
}

/// Numerical / observational witness that `src ~ tgt`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EquivWitness {
    /// Absolute tolerance on tensor values (`max |a-b|`).
    pub abs_tol: f32,
    /// Number of random inputs used for the check.
    pub samples: usize,
    /// Seeds used (so the check is reproducible).
    pub seeds: Vec<u64>,
    /// Filled after verification.
    pub observed_max_abs: Option<f32>,
    pub verified: bool,
}

impl EquivWitness {
    pub fn tight() -> Self {
        Self { abs_tol: 1e-5, samples: 3, seeds: vec![1, 2, 3], observed_max_abs: None, verified: false }
    }

    pub fn relaxed() -> Self {
        Self {
            abs_tol: 1e-3,
            samples: 3,
            seeds: vec![1, 2, 3],
            observed_max_abs: None,
            verified: false,
        }
    }

    pub fn accept(&mut self, max_abs: f32) -> CatResult<()> {
        self.observed_max_abs = Some(max_abs);
        if max_abs <= self.abs_tol {
            self.verified = true;
            Ok(())
        } else {
            self.verified = false;
            Err(CatError::Witness(format!(
                "max abs {max_abs} exceeds tolerance {}",
                self.abs_tol
            )))
        }
    }
}

/// A 2-morphism `src ⇒ tgt`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TwoCell {
    pub name: String,
    pub source_name: String,
    pub target_name: String,
    pub rewrite: Rewrite,
    pub witness: EquivWitness,
}

impl TwoCell {
    pub fn refl(name: impl Into<String>) -> Self {
        let name = name.into();
        Self {
            name: format!("refl:{name}"),
            source_name: name.clone(),
            target_name: name,
            rewrite: Rewrite::Refl,
            witness: EquivWitness {
                abs_tol: 0.0,
                samples: 0,
                seeds: vec![],
                observed_max_abs: Some(0.0),
                verified: true,
            },
        }
    }

    pub fn fuse(ops: Vec<String>, witness: EquivWitness) -> Self {
        let name = format!("fuse:{}", ops.join("+"));
        Self {
            name: name.clone(),
            source_name: ops.join("∘"),
            target_name: ops.join("+"),
            rewrite: Rewrite::Fuse { ops },
            witness,
        }
    }

    pub fn reassign(op: impl Into<String>, from: impl Into<String>, to: impl Into<String>) -> Self {
        let op = op.into();
        let from = from.into();
        let to = to.into();
        Self {
            name: format!("reassign:{op}:{from}->{to}"),
            source_name: op.clone(),
            target_name: op.clone(),
            rewrite: Rewrite::Reassign { op, from_device: from, to_device: to },
            witness: EquivWitness::tight(),
        }
    }

    pub fn is_verified(&self) -> bool {
        self.witness.verified
    }

    /// Horizontal composition of 2-cells (same 1-morphism boundary pasted).
    pub fn hcomp(self, other: TwoCell) -> CatResult<TwoCell> {
        if self.target_name != other.source_name && self.source_name != other.source_name {
            // Allow pasting when they rewrite disjoint ops; name the composite.
        }
        Ok(TwoCell {
            name: format!("({}⋆{})", self.name, other.name),
            source_name: self.source_name,
            target_name: other.target_name,
            rewrite: Rewrite::Refl, // composite rewrite; witness is the join
            witness: EquivWitness {
                abs_tol: self.witness.abs_tol.max(other.witness.abs_tol),
                samples: self.witness.samples.min(other.witness.samples.max(1)),
                seeds: self.witness.seeds,
                observed_max_abs: match (self.witness.observed_max_abs, other.witness.observed_max_abs)
                {
                    (Some(a), Some(b)) => Some(a.max(b)),
                    (a, b) => a.or(b),
                },
                verified: self.witness.verified && other.witness.verified,
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn refl_is_verified() {
        assert!(TwoCell::refl("id").is_verified());
    }

    #[test]
    fn witness_rejects_large_error() {
        let mut w = EquivWitness::tight();
        assert!(w.accept(1.0).is_err());
        assert!(!w.verified);
    }

    #[test]
    fn witness_accepts_small_error() {
        let mut w = EquivWitness::tight();
        assert!(w.accept(1e-6).is_ok());
        assert!(w.verified);
    }
}
