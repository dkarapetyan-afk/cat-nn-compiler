//! Category `A` of AI architectures (spec §2, §10).
//!
//! Objects are network specifications (composites of open systems).
//! 1-morphisms are *refinements*: fuse a chain, split a layer, change a
//! dtype, etc. Realization categories are obtained by the fibration
//! `R: A^op → Cat`.

use serde::{Deserialize, Serialize};

use crate::category::interface::Interface;
use crate::category::morphism::Hom;
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
}

impl Architecture {
    pub fn from_system(id: impl Into<String>, system: CompositeSystem) -> CatResult<Self> {
        let id = id.into();
        let hom = system.as_hom();
        let cospan = StructuredCospan::from_chain(format!("{id}.apex"), &system.parts).ok();
        Ok(Self { id, system, hom, cospan })
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

    pub fn fuse(src: &Architecture, tgt: &Architecture, chain: Vec<String>, fused_name: String) -> Self {
        Self {
            name: format!("fuse:{}→{}", src.id, tgt.id),
            src: src.id.clone(),
            tgt: tgt.id.clone(),
            kind: ArchRefinementKind::Fuse { chain, fused_name },
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
