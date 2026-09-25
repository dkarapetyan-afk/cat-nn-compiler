//! `catnn-core`: load-bearing higher-categorical structures for NN compilation.
//!
//! See the crate-level README mapping and each module's docs for the
//! correspondence with `categorical-nn-spec.md`.

pub mod architecture;
pub mod category;
pub mod error;
pub mod lens;
pub mod spec;
pub mod tensor;

pub use architecture::{
    enumerate_realizations, parse_bytes, reindex, spark_cluster, ArchMorphism, Architecture,
    Cluster, Device, DeviceKind, DeviceMemory, DgxSparkSpec, Fibre, KernelBackend, MemoryBudget,
    MemoryHierarchy, MemoryKind, MemoryRegionKind, Placement, Precision, Realization, RegionPeaks,
    SparkTopology,
};
pub use category::{flatten_seq, EquivWitness, Hom, Interface, OpenOp, Rewrite, Tens, TwoCell};
pub use error::{CatError, CatResult};
pub use lens::{
    first_of_pair, Adam, AdamState, CompositeSystem, Cotangent, LawfulLens, Observation,
    OpenComponent, Optic, OptimizerKind, ParaMorphism, PolicyState, Sgd, StructuredCospan,
    UnlawfulWitness,
};
pub use spec::{
    deepseek_connected, deepseek_mixture, deepseek_trunk, deepseek_v3_architecture,
    mixtral_architecture, mlp_spec, transformer_block_spec, DeepSeekV3Config, MixtralConfig,
    NetworkBuilder, NetworkSpec,
};
pub use tensor::{DType, Shape, Tensor, TensorType};

/// Composition of two 1-morphisms in `Tens` — re-export for callers that
/// should not have to name the SMC.
pub fn compose(f: &Hom, g: &Hom) -> CatResult<Hom> {
    Tens::compose(f, g)
}

pub fn tensor_hom(f: &Hom, g: &Hom) -> Hom {
    Tens::tensor(f, g)
}
