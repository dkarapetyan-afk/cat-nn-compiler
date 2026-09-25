//! Unlawful optics: residual / policy state (spec §7).
//!
//! A real optimizer does **not** satisfy the lens laws:
//! `observe(compile(update(A, O))) ≠ O` in general. We therefore refuse to
//! type the optimizer as [`crate::lens::lawful::LawfulLens`]. It is an
//! [`Optic`] with an explicit residual `P` (policy / adam moments / fusion
//! budget) that the laws are *not* claimed for.

use serde::{Deserialize, Serialize};

/// Residual policy carried by the adaptive optic (spec §7, §8, §10).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PolicyState {
    /// How many adaptation steps have been taken.
    pub step: u64,
    /// Prefer fusion when estimated speedup exceeds this ratio.
    pub fuse_threshold: f64,
    /// Prefer re-placement when estimated speedup exceeds this ratio.
    pub replace_threshold: f64,
    /// Last observed mean latency (milliseconds).
    pub last_latency_ms: f64,
    /// Last observed max numerical error vs reference.
    pub last_max_abs: f32,
    /// Residual that records *why* the last update is not a lens put.
    pub residual_note: String,
}

impl Default for PolicyState {
    fn default() -> Self {
        Self {
            step: 0,
            fuse_threshold: 1.05,
            replace_threshold: 1.10,
            last_latency_ms: f64::INFINITY,
            last_max_abs: 0.0,
            residual_note: "initial policy; no observation yet".into(),
        }
    }
}

/// Observation object `O` of the central square (spec §10).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Observation {
    pub latency_ms: f64,
    pub bytes_moved: u64,
    pub kernel_count: usize,
    pub device_occupancy: Vec<(String, f64)>,
    pub max_abs_error: f32,
    pub notes: Vec<String>,
}

impl Default for Observation {
    fn default() -> Self {
        Self {
            latency_ms: 0.0,
            bytes_moved: 0,
            kernel_count: 0,
            device_occupancy: vec![],
            max_abs_error: 0.0,
            notes: vec![],
        }
    }
}

/// An optic `S ─P→ V` with residual `P`. Forward and backward maps both
/// see and update the residual; there is no claim of PutGet/GetPut.
#[derive(Clone)]
pub struct Optic<S, V, P> {
    pub name: String,
    pub view: fn(&S, &P) -> V,
    pub update: fn(&S, &V, &P) -> (S, P),
}

impl<S, V, P> Optic<S, V, P> {
    pub fn new(
        name: impl Into<String>,
        view: fn(&S, &P) -> V,
        update: fn(&S, &V, &P) -> (S, P),
    ) -> Self {
        Self { name: name.into(), view, update }
    }

    pub fn observe(&self, s: &S, p: &P) -> V {
        (self.view)(s, p)
    }

    pub fn step(&self, s: &S, v: &V, p: &P) -> (S, P) {
        (self.update)(s, v, p)
    }
}

/// Witness that an optic is *not* a lawful lens, recorded as data.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UnlawfulWitness {
    pub optic_name: String,
    pub law_that_fails: String,
    pub explanation: String,
}

impl UnlawfulWitness {
    pub fn optimizer() -> Self {
        Self {
            optic_name: "adapt_optic".into(),
            law_that_fails: "PutGet".into(),
            explanation: "observe(compile(update(A,O))) is a *new* observation, not O. \
                 The residual policy state is why this is an optic, not a lens."
                .into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn optimizer_witness_names_putget() {
        assert_eq!(UnlawfulWitness::optimizer().law_that_fails, "PutGet");
    }
}
