//! Scheduler: execute a placed IR graph across device handlers.
//!
//! Cross-device tensors move along the cluster's link topology. The
//! consumer reads the delivered tensor. Per-device traces are glued as
//! a sheaf.

use catnn_compile::Compiled;
use catnn_core::lens::Observation;
use catnn_core::tensor::Tensor;
use catnn_core::CatResult;

use crate::coord;
use crate::sheaf::{glue, GlobalHistory, LocalSection};

#[derive(Debug, Clone)]
pub struct RunResult {
    pub output: Tensor,
    pub observation: Observation,
    pub sections: Vec<LocalSection>,
    pub history: GlobalHistory,
    pub used_gpu_fallback: bool,
}

pub fn run_compiled(compiled: &Compiled, input: &Tensor) -> CatResult<RunResult> {
    let fwd = coord::run_forward(compiled, input)?;
    let history = glue(&fwd.sections)?;
    let mut notes = Vec::new();
    if fwd.used_gpu_fallback {
        notes.push("GPU-placed ops executed via CPU fallback after WGSL validation".into());
    }
    if fwd.messages > 0 {
        notes.push(format!(
            "routed {} tensor messages ({} B) along cluster links",
            fwd.messages, fwd.bytes_moved
        ));
    }
    let observation = Observation {
        latency_ms: fwd.latency_ms,
        bytes_moved: fwd.bytes_moved,
        kernel_count: compiled.graph.nodes.len(),
        device_occupancy: fwd.occupancy,
        max_abs_error: 0.0,
        notes,
    };
    Ok(RunResult {
        output: fwd.output,
        observation,
        sections: fwd.sections,
        history,
        used_gpu_fallback: fwd.used_gpu_fallback,
    })
}

/// Run the same graph on the reference evaluator and attach the error
/// to the observation (fills a 2-cell witness).
pub fn run_with_reference(compiled: &Compiled, input: &Tensor) -> CatResult<RunResult> {
    let mut result = run_compiled(compiled, input)?;
    let reference = catnn_compile::eval_graph_reference(&compiled.graph, input)?;
    let err = result.output.max_abs_diff(&reference)?;
    result.observation.max_abs_error = err;
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use catnn_compile::compile;
    use catnn_core::architecture::Cluster;
    use catnn_core::spec::mlp_spec;
    use catnn_core::tensor::TensorType;

    #[test]
    fn scheduled_mlp_matches_reference() {
        let arch = mlp_spec(4, 16, 32, 8).unwrap();
        let compiled = compile(&arch, &Cluster::demo()).unwrap();
        let x = Tensor::lcg(TensorType::f32([4, 16]), 9);
        let r = run_with_reference(&compiled, &x).unwrap();
        assert!(r.observation.max_abs_error < 1e-4);
        assert!(r.history.is_coherent());
    }
}
