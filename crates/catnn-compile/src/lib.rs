//! Compiler: architecture → IR → placement → kernels.

pub mod ir;
pub mod kernel;
pub mod lower;
pub mod pipeline;
pub mod place;
pub mod schedule;
pub mod spark_place;
pub mod spark_train;
pub mod train;

use catnn_core::architecture::{Cluster, Realization};
use catnn_core::Architecture;
use catnn_core::CatResult;

use crate::ir::IrGraph;

/// A compiled programme: IR + chosen realization + generated WGSL sources.
#[derive(Debug, Clone)]
pub struct Compiled {
    pub graph: IrGraph,
    pub realization: Realization,
    pub wgsl: Vec<(String, String)>,
    pub cluster: Cluster,
}

pub fn compile(arch: &Architecture, cluster: &Cluster) -> CatResult<Compiled> {
    let graph = lower::lower_architecture(arch)?;
    let mut realization = place::realize(&arch.id, &graph, cluster, format!("{}@placed", arch.id));
    realization.fusions = place::propose_fusions(&graph, &realization.placement);
    let mut wgsl = Vec::new();
    for node in graph.nodes.values() {
        if matches!(node.op, catnn_core::category::OpenOp::Identity) {
            continue;
        }
        // Always emit + naga-validate accelerator source. GPU-placed ops
        // still execute via the CPU fallback in this environment; the
        // source is the real artifact.
        let k = kernel::wgsl::emit_and_validate(&node.op, &node.name)?;
        wgsl.push((node.name.clone(), k.source));
    }
    Ok(Compiled { graph, realization, wgsl, cluster: cluster.clone() })
}

/// Sequential CPU evaluation of a whole graph (used by tests and the
/// reference side of witnesses). `input` is bound to every graph input.
pub fn eval_graph_cpu(
    graph: &IrGraph,
    input: &catnn_core::tensor::Tensor,
) -> CatResult<catnn_core::tensor::Tensor> {
    use indexmap::IndexMap;
    let order = graph.topo()?;
    let mut vals: IndexMap<u32, catnn_core::tensor::Tensor> = IndexMap::new();
    for id in order {
        let node = graph.node(id).unwrap();
        if graph.inputs.contains(&id) {
            vals.insert(id.0, input.clone());
            continue;
        }
        let preds = graph.preds(id);
        let inputs: Vec<_> = preds.iter().map(|p| vals[&p.0].clone()).collect();
        let y = kernel::eval_cpu(node, &inputs)?;
        vals.insert(id.0, y);
    }
    let out_id = *graph.outputs.last().unwrap();
    Ok(vals[&out_id.0].clone())
}

pub fn eval_graph_reference(
    graph: &IrGraph,
    input: &catnn_core::tensor::Tensor,
) -> CatResult<catnn_core::tensor::Tensor> {
    use indexmap::IndexMap;
    let order = graph.topo()?;
    let mut vals: IndexMap<u32, catnn_core::tensor::Tensor> = IndexMap::new();
    for id in order {
        let node = graph.node(id).unwrap();
        if graph.inputs.contains(&id) {
            vals.insert(id.0, input.clone());
            continue;
        }
        let preds = graph.preds(id);
        let inputs: Vec<_> = preds.iter().map(|p| vals[&p.0].clone()).collect();
        let y = kernel::eval_reference(node, &inputs)?;
        vals.insert(id.0, y);
    }
    let out_id = *graph.outputs.last().unwrap();
    Ok(vals[&out_id.0].clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    use catnn_core::spec::mlp_spec;
    use catnn_core::tensor::{Tensor, TensorType};

    #[test]
    fn cpu_matches_reference_on_mlp() {
        let arch = mlp_spec(4, 16, 32, 8).unwrap();
        let compiled = compile(&arch, &Cluster::single_cpu()).unwrap();
        let x = Tensor::lcg(TensorType::f32([4, 16]), 42);
        let a = eval_graph_cpu(&compiled.graph, &x).unwrap();
        let b = eval_graph_reference(&compiled.graph, &x).unwrap();
        let err = a.max_abs_diff(&b).unwrap();
        assert!(err < 1e-4, "max abs {err}");
    }
}
