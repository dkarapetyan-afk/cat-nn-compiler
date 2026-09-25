//! Runtime: device handlers, cluster scheduler, temporal sheaf, adaptive optic.

pub mod adapt;
pub mod coord;
pub mod dist;
pub mod handler;
pub mod scheduler;
pub mod sheaf;
pub mod transfer;

pub use dist::{serve_member, train_attached, MemberEndpoint};

pub use adapt::{adapt_once, adapt_optic, AdaptStep};
pub use coord::{train_on_cluster, train_placed, ClusterTrainReport};
pub use scheduler::{run_compiled, run_with_reference, RunResult};
pub use sheaf::{compatible, glue, Event, EventKind, GlobalHistory, Interval, LocalSection};

use catnn_compile::Compiled;
use catnn_core::tensor::Tensor;
use catnn_core::CatResult;

/// End-to-end: run + glue + optional one adaptation step.
pub fn execute(compiled: &Compiled, input: &Tensor, adapt: bool) -> CatResult<ExecuteReport> {
    if adapt {
        let (next, step, run) =
            adapt_once(compiled, input, &catnn_core::lens::PolicyState::default())?;
        Ok(ExecuteReport {
            output: run.output,
            observation: step.after.clone(),
            history: run.history,
            sections: run.sections,
            adapt: Some(step),
            compiled: next,
        })
    } else {
        let run = run_with_reference(compiled, input)?;
        Ok(ExecuteReport {
            output: run.output,
            observation: run.observation,
            history: run.history,
            sections: run.sections,
            adapt: None,
            compiled: compiled.clone(),
        })
    }
}

pub struct ExecuteReport {
    pub output: Tensor,
    pub observation: catnn_core::lens::Observation,
    pub history: GlobalHistory,
    pub sections: Vec<LocalSection>,
    pub adapt: Option<AdaptStep>,
    pub compiled: Compiled,
}
