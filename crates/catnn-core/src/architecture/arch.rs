//! Category `A` of AI architectures (spec §2, §10).
//!
//! Objects are network specifications (composites of open systems).
//! 1-morphisms are *refinements*: fuse a chain, split a layer, change a
//! dtype, etc. Realization categories are obtained by the fibration
//! `R: A^op → Cat`.

use serde::{Deserialize, Serialize};

use crate::architecture::memory::MemoryRegionKind;
use crate::category::interface::Interface;
use crate::category::morphism::{Hom, OpenOp};
use crate::error::{CatError, CatResult};
use crate::lens::open::{CompositeSystem, OpenComponent};
use crate::lens::wiring::StructuredCospan;

/// An object of `A`: a named architecture.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Architecture {
    pub id: String,
    pub system: CompositeSystem,
    pub hom: Hom,
    pub cospan: Option<StructuredCospan>,
    /// When false, lowering records parameter *shapes* but does not fill
    /// tensors (full Mixtral is ~93 GB in bf16 — we refuse to allocate it).
    #[serde(default = "default_materialize")]
    pub materialize_params: bool,
    /// How components sit in the diagram. `Chain` is sequential composition.
    /// `Mixture` runs two branches on the same input and mixes their outputs.
    #[serde(default)]
    pub wiring: Wiring,
}

/// How an architecture's components are wired into the IR.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum Wiring {
    #[default]
    Chain,
    /// `parts[..left]` and `parts[left..left+right]` both read the diagram
    /// input. `parts[left+right]` is the mix morphism of the two outputs.
    Mixture { left: usize, right: usize },
}

fn default_materialize() -> bool {
    true
}

impl Architecture {
    pub fn from_system(id: impl Into<String>, system: CompositeSystem) -> CatResult<Self> {
        let id = id.into();
        let hom = system.as_hom();
        let cospan = StructuredCospan::from_chain(format!("{id}.apex"), &system.parts).ok();
        Ok(Self { id, system, hom, cospan, materialize_params: true, wiring: Wiring::Chain })
    }

    /// Sequential composition. `self.output` must match `next.input`.
    /// Component names are prefixed so the flattened diagram stays unambiguous.
    pub fn connect(&self, next: &Architecture) -> CatResult<Self> {
        if !matches!(self.wiring, Wiring::Chain) || !matches!(next.wiring, Wiring::Chain) {
            return Err(CatError::Wiring(
                "connect expects two chain diagrams; mix the result afterwards".into(),
            ));
        }
        let mut left = self.clone();
        prefix_components(&mut left, "a");
        let mut right = next.clone();
        prefix_components(&mut right, "b");
        let system = left.system.then_composite(&right.system)?;
        let hom = crate::compose(&left.hom, &right.hom)?;
        Ok(Self {
            id: format!("{}⨾{}", self.id, next.id),
            system,
            hom,
            cospan: None,
            materialize_params: self.materialize_params && next.materialize_params,
            wiring: Wiring::Chain,
        })
    }

    /// Parallel mixture: both diagrams read the same input and a scalar gate mixes their outputs.
    pub fn mixture(&self, other: &Architecture) -> CatResult<Self> {
        self.input().same_as(other.input()).map_err(|_| CatError::InterfaceMismatch {
            expected: self.input().to_string(),
            actual: other.input().to_string(),
        })?;
        self.output().same_as(other.output()).map_err(|_| CatError::InterfaceMismatch {
            expected: self.output().to_string(),
            actual: other.output().to_string(),
        })?;
        let mut left = self.clone();
        prefix_components(&mut left, "L");
        let mut right = other.clone();
        prefix_components(&mut right, "R");
        let n_left = left.components().len();
        let n_right = right.components().len();
        let mix = OpenComponent {
            name: "mix".into(),
            state: OpenOp::Mix.state_interface("mix", None),
            input: left.output().clone(),
            output: left.output().clone(),
            op: OpenOp::Mix,
            moore: false,
        };
        let mut parts = left.system.parts.clone();
        parts.extend(right.system.parts.iter().cloned());
        parts.push(mix.clone());
        let system = CompositeSystem {
            name: format!("μ({},{})", self.id, other.id),
            state: left.system.state.tensor(&right.system.state).tensor(&mix.state),
            input: left.system.input.clone(),
            output: mix.output.clone(),
            parts,
            parallel: true,
        };
        let hom = crate::tensor_hom(&left.hom, &right.hom);
        Ok(Self {
            id: format!("mixture({}⊗{})", self.id, other.id),
            system,
            hom,
            cospan: None,
            materialize_params: self.materialize_params && other.materialize_params,
            wiring: Wiring::Mixture { left: n_left, right: n_right },
        })
    }

    pub fn from_components(id: impl Into<String>, parts: Vec<OpenComponent>) -> CatResult<Self> {
        if parts.is_empty() {
            return Err(CatError::Wiring("architecture with no components".into()));
        }
        let mut sys = CompositeSystem::singleton(parts[0].clone());
        for c in parts.iter().skip(1) {
            sys = sys.then(c)?;
        }
        Self::from_system(id, sys)
    }

    pub fn input(&self) -> &Interface {
        &self.system.input
    }

    pub fn output(&self) -> &Interface {
        &self.system.output
    }

    pub fn components(&self) -> &[OpenComponent] {
        &self.system.parts
    }
}

fn prefix_components(arch: &mut Architecture, prefix: &str) {
    arch.id = format!("{prefix}.{}", arch.id);
    for c in &mut arch.system.parts {
        c.name = format!("{prefix}.{}", c.name);
    }
    arch.system.name = format!("{prefix}.{}", arch.system.name);
}

/// A 1-morphism of `A`: refinement `src → tgt`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ArchRefinementKind {
    /// Identity refinement.
    Id,
    /// Fuse a sequential chain into a single named op.
    Fuse { chain: Vec<String>, fused_name: String },
    /// Lower one op to a more primitive expansion (name-preserving).
    Lower { op: String },
    /// Re-partition / change intended placement hint.
    Repartition { op: String, parts: usize },
    /// Tighten one region of the memory object of `H`. The fibration is
    /// indexed by the structured spec: a smaller **code** object reindexes
    /// to kernel reload / fuse-code 2-cells; a smaller **buffer** object
    /// reindexes to rematerialize / offload; a smaller **scratch** object
    /// reindexes to occupancy-reducing fusion.
    TightenMemory { region: MemoryRegionKind, from_bytes: u64, to_bytes: u64 },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArchMorphism {
    pub name: String,
    pub src: String,
    pub tgt: String,
    pub kind: ArchRefinementKind,
}

impl ArchMorphism {
    pub fn id(arch: &Architecture) -> Self {
        Self {
            name: format!("id_{}", arch.id),
            src: arch.id.clone(),
            tgt: arch.id.clone(),
            kind: ArchRefinementKind::Id,
        }
    }

    pub fn fuse(
        src: &Architecture,
        tgt: &Architecture,
        chain: Vec<String>,
        fused_name: String,
    ) -> Self {
        Self {
            name: format!("fuse:{}→{}", src.id, tgt.id),
            src: src.id.clone(),
            tgt: tgt.id.clone(),
            kind: ArchRefinementKind::Fuse { chain, fused_name },
        }
    }

    /// Reindex `R` along a smaller memory object in one region.
    pub fn tighten_memory(
        arch: &Architecture,
        region: MemoryRegionKind,
        from_bytes: u64,
        to_bytes: u64,
    ) -> Self {
        Self {
            name: format!("tighten:{}:{}:{}→{}", arch.id, region.as_str(), from_bytes, to_bytes),
            src: arch.id.clone(),
            tgt: arch.id.clone(),
            kind: ArchRefinementKind::TightenMemory { region, from_bytes, to_bytes },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lens::open::OpenComponent;

    #[test]
    fn architecture_from_mlp() {
        let fc1 = OpenComponent::linear("fc1", 2, 4, 8, true);
        let act = OpenComponent::relu("act", fc1.output.clone());
        let fc2 = OpenComponent::linear("fc2", 2, 8, 2, true);
        let a = Architecture::from_components("mlp", vec![fc1, act, fc2]).unwrap();
        assert_eq!(a.components().len(), 3);
    }
}
