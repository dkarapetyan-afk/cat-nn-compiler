//! Architecture category `A`, hardware category `H`, fibration `R` (spec §2, §6, §10).

pub mod arch;
pub mod fibration;
pub mod hardware;
pub mod realization;

pub use arch::{ArchMorphism, ArchRefinementKind, Architecture};
pub use fibration::{enumerate_realizations, reindex, reindex_id_is_id, Fibre, TotalObj};
pub use hardware::{Cluster, Device, DeviceKind, Link, Node};
pub use realization::{FusionGroup, KernelBackend, KernelChoice, Placement, Realization};
