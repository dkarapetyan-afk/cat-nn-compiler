//! Objects of `R(A)`: runtime realizations of an architecture (spec §2).
//!
//! A realization chooses kernels, device placement, partitioning, and
//! fusion groups. Morphisms in `R(A)` are 2-cells (implementation rewrites)
//! with equivalence witnesses.

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

use crate::architecture::hardware::Cluster;
use crate::category::two_cell::TwoCell;

/// Where a single IR node lives.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Placement {
    pub assign: IndexMap<String, String>,
}

impl Placement {
    pub fn new() -> Self {
        Self { assign: IndexMap::new() }
    }

    pub fn put(&mut self, op: impl Into<String>, device: impl Into<String>) {
        self.assign.insert(op.into(), device.into());
    }

    pub fn get(&self, op: &str) -> Option<&str> {
        self.assign.get(op).map(|s| s.as_str())
    }
}

impl Default for Placement {
    fn default() -> Self {
        Self::new()
    }
}

/// A group of ops fused into one kernel.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FusionGroup {
    pub name: String,
    pub ops: Vec<String>,
}

/// Kernel backend selected for an op or fusion group.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum KernelBackend {
    CpuRust,
    Wgsl,
    Reference,
    /// CUDA generated for GB10 / `sm_121`.
    CudaSm121,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KernelChoice {
    pub op: String,
    pub backend: KernelBackend,
}

/// An object of `R(A)`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Realization {
    pub id: String,
    pub architecture: String,
    pub cluster: String,
    pub placement: Placement,
    pub fusions: Vec<FusionGroup>,
    pub kernels: Vec<KernelChoice>,
    /// 1-morphisms of `R(A)` out of this object (rewrites already attached).
    pub cells: Vec<TwoCell>,
}

impl Realization {
    pub fn new(id: impl Into<String>, architecture: impl Into<String>, cluster: &Cluster) -> Self {
        Self {
            id: id.into(),
            architecture: architecture.into(),
            cluster: cluster.name.clone(),
            placement: Placement::new(),
            fusions: vec![],
            kernels: vec![],
            cells: vec![],
        }
    }

    pub fn device_of(&self, op: &str) -> Option<&str> {
        self.placement.get(op)
    }

    pub fn fused_with(&self, op: &str) -> Option<&FusionGroup> {
        self.fusions.iter().find(|g| g.ops.iter().any(|o| o == op))
    }

    pub fn backend_of(&self, op: &str) -> KernelBackend {
        self.kernels
            .iter()
            .find(|k| k.op == op)
            .map(|k| k.backend)
            .unwrap_or(KernelBackend::CpuRust)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::architecture::hardware::Cluster;

    #[test]
    fn realization_records_placement() {
        let c = Cluster::demo();
        let mut r = Realization::new("r0", "mlp", &c);
        r.placement.put("fc1", "node0.cpu");
        assert_eq!(r.device_of("fc1"), Some("node0.cpu"));
    }
}
