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
    enumerate_realizations, reindex, spark_cluster, ArchMorphism, Architecture, Cluster, Device,
    DeviceKind, DgxSparkSpec, Fibre, KernelBackend, Placement, Precision, Realization,
    SparkTopology,
};
pub use category::{flatten_seq, EquivWitness, Hom, Interface, OpenOp, Rewrite, Tens, TwoCell};
pub use error::{CatError, CatResult};
pub use lens::{
    first_of_pair, CompositeSystem, LawfulLens, Observation, OpenComponent, Optic, PolicyState,
    StructuredCospan, UnlawfulWitness,
};
pub use spec::{
    mixtral_architecture, mlp_spec, transformer_block_spec, MixtralConfig, NetworkBuilder,
    NetworkSpec,
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
