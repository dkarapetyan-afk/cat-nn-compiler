//! Architecture category `A`, hardware category `H`, fibration `R` (spec §2, §6, §10).

pub mod arch;
pub mod fibration;
pub mod hardware;
pub mod memory;
pub mod realization;
pub mod spark;

pub use arch::{ArchMorphism, ArchRefinementKind, Architecture};
pub use fibration::{enumerate_realizations, reindex, reindex_id_is_id, Fibre, TotalObj};
pub use hardware::{Cluster, Device, DeviceKind, Link, Node};
pub use memory::{
    parse_bytes, DeviceMemory, MemoryBudget, MemoryHierarchy, MemoryKind, MemoryRegion,
    MemoryRegionKind, MemoryTier, RegionPeaks,
};
pub use realization::{FusionGroup, KernelBackend, KernelChoice, Placement, Realization};
pub use spark::{spark_cluster, DgxSparkSpec, Precision, SparkTopology};
