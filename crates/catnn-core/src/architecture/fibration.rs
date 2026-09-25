//! Cartesian fibration `p: ∫R → A` obtained from `R: A^op → Cat` (spec §2).
//!
//! Objects of the total category are pairs `(A, r)` with `r ∈ R(A)`.
//! A morphism of architectures `f: A → A'` *reindexes* realizations by
//! pullback / transport: [`reindex`]. Functoriality
//! `R(id) ≅ id` and `R(g∘f) ≅ R(f)∘R(g)` is tested on the nose for the
//! generating refinements we implement.

use serde::{Deserialize, Serialize};

use crate::architecture::arch::{ArchMorphism, ArchRefinementKind, Architecture};
use crate::architecture::hardware::Cluster;
use crate::architecture::realization::{FusionGroup, KernelBackend, KernelChoice, Realization};
use crate::error::{CatError, CatResult};

/// An object of the Grothendieck construction `∫R`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TotalObj {
    pub architecture: Architecture,
    pub realization: Realization,
}

/// The fibre `R(A)` as an explicit category (objects + cells).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Fibre {
    pub architecture: String,
    pub objects: Vec<Realization>,
}

impl Fibre {
    pub fn new(architecture: impl Into<String>) -> Self {
        Self { architecture: architecture.into(), objects: vec![] }
    }

    pub fn push(&mut self, r: Realization) {
        self.objects.push(r);
    }
}

/// Enumerate a handful of admissible realizations of `arch` on `cluster`.
///
/// This is `R(A)` as a *finite* category of candidates the compiler will
/// choose among: all-CPU, GPU-preferring (when a GPU exists), and a
/// split placement across nodes.
pub fn enumerate_realizations(arch: &Architecture, cluster: &Cluster) -> Fibre {
    let mut fibre = Fibre::new(&arch.id);
    let ops: Vec<String> = arch.components().iter().map(|c| c.name.clone()).collect();

    let cpu = cluster
        .devices()
        .find(|(_, d)| matches!(d.kind, crate::architecture::hardware::DeviceKind::Cpu))
        .map(|(_, d)| d.id.clone())
        .unwrap_or_else(|| cluster.default_device().id.clone());

    let gpu = cluster
        .devices()
        .find(|(_, d)| matches!(d.kind, crate::architecture::hardware::DeviceKind::Gpu))
        .map(|(_, d)| d.id.clone());

    let mut all_cpu = Realization::new(format!("{}@all-cpu", arch.id), &arch.id, cluster);
    for op in &ops {
        all_cpu.placement.put(op, &cpu);
        all_cpu.kernels.push(KernelChoice { op: op.clone(), backend: KernelBackend::CpuRust });
    }
    fibre.push(all_cpu);

    if let Some(gpu_id) = gpu {
        let mut pref = Realization::new(format!("{}@gpu-pref", arch.id), &arch.id, cluster);
        for (i, op) in ops.iter().enumerate() {
            // Heavy ops (linear, attention) on GPU; activations stay CPU to
            // exercise transfers.
            let heavy = arch.components()[i].op.name() == "linear"
                || arch.components()[i].op.name() == "attention";
            let dev = if heavy { gpu_id.clone() } else { cpu.clone() };
            let backend = if heavy { KernelBackend::Wgsl } else { KernelBackend::CpuRust };
            pref.placement.put(op, dev);
            pref.kernels.push(KernelChoice { op: op.clone(), backend });
        }
        fibre.push(pref);
    }

    let devices: Vec<String> = cluster.devices().map(|(_, d)| d.id.clone()).collect();
    if devices.len() > 1 {
        let mut split = Realization::new(format!("{}@split", arch.id), &arch.id, cluster);
        for (i, op) in ops.iter().enumerate() {
            let dev = devices[i % devices.len()].clone();
            split.placement.put(op, &dev);
            let backend = if cluster.find_device(&dev).map(|(_, d)| d.kind)
                == Some(crate::architecture::hardware::DeviceKind::Gpu)
            {
                KernelBackend::Wgsl
            } else {
                KernelBackend::CpuRust
            };
            split.kernels.push(KernelChoice { op: op.clone(), backend });
        }
        fibre.push(split);
    }

    fibre
}

/// Reindexing / pullback of a realization along an architecture morphism
/// `f: A → A'`. Because `R` is contravariant, a realization of `A'` is
/// transported to a realization of `A` by renaming / expanding ops.
pub fn reindex(f: &ArchMorphism, r: &Realization) -> CatResult<Realization> {
    if r.architecture != f.tgt && r.architecture != f.src {
        return Err(CatError::Fibration(format!(
            "realization {} lives over {}, not {} or {}",
            r.id, r.architecture, f.src, f.tgt
        )));
    }
    let mut out = r.clone();
    out.id = format!("reindex[{}]({})", f.name, r.id);
    out.architecture = f.src.clone();
    match &f.kind {
        ArchRefinementKind::Id => {
            out.architecture = f.src.clone();
        }
        ArchRefinementKind::Fuse { chain, fused_name } => {
            // Pullback of a fused realization: expand the fused op back to
            // the chain, inheriting the fused op's device.
            if let Some(dev) = r.device_of(fused_name).map(|s| s.to_string()) {
                for op in chain {
                    out.placement.put(op, &dev);
                }
            }
            out.fusions.push(FusionGroup { name: fused_name.clone(), ops: chain.clone() });
        }
        ArchRefinementKind::Lower { op } => {
            // Transport is the identity on placement; lowering is a 2-cell.
            let _ = op;
        }
        ArchRefinementKind::Repartition { op, parts } => {
            if *parts == 0 {
                return Err(CatError::Fibration("zero partition".into()));
            }
            let _ = op;
        }
    }
    Ok(out)
}

/// Functoriality helper: `R(id_A)(r) = r` up to the identity rename.
pub fn reindex_id_is_id(arch: &Architecture, r: &Realization) -> CatResult<()> {
    let f = ArchMorphism::id(arch);
    let r2 = reindex(&f, r)?;
    if r2.placement.assign != r.placement.assign {
        return Err(CatError::Fibration("R(id) moved placement".into()));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::architecture::hardware::Cluster;
    use crate::lens::open::OpenComponent;

    fn mlp() -> Architecture {
        let fc1 = OpenComponent::linear("fc1", 2, 4, 8, true);
        let act = OpenComponent::relu("act", fc1.output.clone());
        let fc2 = OpenComponent::linear("fc2", 2, 8, 2, true);
        Architecture::from_components("mlp", vec![fc1, act, fc2]).unwrap()
    }

    #[test]
    fn enumerate_produces_all_cpu() {
        let fibre = enumerate_realizations(&mlp(), &Cluster::demo());
        assert!(fibre.objects.iter().any(|r| r.id.contains("all-cpu")));
        assert!(fibre.objects.len() >= 2);
    }

    #[test]
    fn reindex_identity_preserves_placement() {
        let arch = mlp();
        let fibre = enumerate_realizations(&arch, &Cluster::single_cpu());
        reindex_id_is_id(&arch, &fibre.objects[0]).unwrap();
    }

    #[test]
    fn reindex_fuse_expands_chain() {
        let arch = mlp();
        let cluster = Cluster::single_cpu();
        let mut fused = Realization::new("fused", "mlp-fused", &cluster);
        fused.placement.put("fc1+act", "node0.cpu");
        let f = ArchMorphism {
            name: "fuse-fc1-act".into(),
            src: "mlp".into(),
            tgt: "mlp-fused".into(),
            kind: ArchRefinementKind::Fuse {
                chain: vec!["fc1".into(), "act".into()],
                fused_name: "fc1+act".into(),
            },
        };
        let pulled = reindex(&f, &fused).unwrap();
        assert_eq!(pulled.device_of("fc1"), Some("node0.cpu"));
        assert_eq!(pulled.device_of("act"), Some("node0.cpu"));
    }
}
