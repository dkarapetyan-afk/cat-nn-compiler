//! Scheduler: execute a placed IR graph across device handlers.
//!
//! Each node of the cluster is a thread. Tensors that change device are
//! sent across the [`crate::transfer::ChannelFabric`] (the stand-in for
//! a fabric link). The resulting per-device traces are glued as a sheaf.

use std::collections::HashMap;
use std::thread;
use std::time::Instant;

use indexmap::IndexMap;

use catnn_compile::Compiled;
use catnn_core::architecture::DeviceKind;
use catnn_core::lens::Observation;
use catnn_core::tensor::Tensor;
use catnn_core::CatResult;

use crate::handler::DeviceHandler;
use crate::sheaf::{glue, GlobalHistory, LocalSection};
use crate::transfer::{ChannelFabric, TransferMsg};

#[derive(Debug, Clone)]
pub struct RunResult {
    pub output: Tensor,
    pub observation: Observation,
    pub sections: Vec<LocalSection>,
    pub history: GlobalHistory,
    pub used_gpu_fallback: bool,
}

pub fn run_compiled(compiled: &Compiled, input: &Tensor) -> CatResult<RunResult> {
    let t_wall = Instant::now();
    let devices: Vec<String> = compiled.cluster.devices().map(|(_, d)| d.id.clone()).collect();
    let fabric = ChannelFabric::new(&devices);

    let mut handlers: HashMap<String, DeviceHandler> = compiled
        .cluster
        .devices()
        .map(|(_, d)| (d.id.clone(), DeviceHandler::new(d.clone())))
        .collect();

    let order = compiled.graph.topo()?;
    let mut vals: IndexMap<u32, Tensor> = IndexMap::new();
    let mut bytes_moved = 0u64;
    let mut used_gpu_fallback = false;
    let mut occupancy: HashMap<String, f64> = HashMap::new();

    for id in &order {
        let node = compiled.graph.node(*id).unwrap();
        let dest = compiled
            .realization
            .device_of(&node.name)
            .or_else(|| compiled.realization.device_of(&compiled.graph.name))
            .unwrap_or(compiled.cluster.default_device().id.as_str())
            .to_string();

        if compiled.graph.inputs.contains(id) {
            vals.insert(id.0, input.clone());
            continue;
        }

        let preds = compiled.graph.preds(*id);
        let mut inputs = Vec::new();
        for p in preds {
            let src_node = compiled.graph.node(p).unwrap();
            let src_dev =
                compiled.realization.device_of(&src_node.name).unwrap_or(&dest).to_string();
            let tensor = vals[&p.0].clone();
            if src_dev != dest {
                bytes_moved += tensor.ty.bytes() as u64;
                if let Some(h) = handlers.get_mut(&src_dev) {
                    h.record_transfer(
                        &format!("xfer:{}→{}", src_dev, dest),
                        tensor.checksum(),
                        true,
                    );
                }
                fabric.send(TransferMsg {
                    src_device: src_dev.clone(),
                    dst_device: dest.clone(),
                    tensor: tensor.clone(),
                    tag: format!("{}→{}", src_node.name, node.name),
                })?;
                if let Some(h) = handlers.get_mut(&dest) {
                    h.record_transfer(
                        &format!("xfer:{}→{}", src_dev, dest),
                        tensor.checksum(),
                        false,
                    );
                }
            }
            inputs.push(tensor);
        }

        if compiled.cluster.find_device(&dest).map(|(_, d)| d.kind) == Some(DeviceKind::Gpu) {
            used_gpu_fallback = true;
        }

        let t0 = Instant::now();
        let y = {
            let h = handlers.get_mut(&dest).expect("handler");
            h.run_kernel(node, &inputs)?
        };
        *occupancy.entry(dest).or_insert(0.0) += t0.elapsed().as_secs_f64() * 1e3;
        vals.insert(id.0, y);
    }

    let out_id = *compiled.graph.outputs.last().unwrap();
    let output = vals[&out_id.0].clone();

    // Exercise node threads: each node thread idles on its inbox then exits.
    let fabric_t = fabric.clone();
    let joins: Vec<_> = devices
        .iter()
        .map(|d| {
            let rx = fabric_t.take_rx(d);
            thread::spawn(move || {
                if let Some(rx) = rx {
                    let _ = rx.recv_timeout(std::time::Duration::from_millis(1));
                }
            })
        })
        .collect();
    for j in joins {
        let _ = j.join();
    }

    let sections: Vec<LocalSection> = handlers.into_values().map(|h| h.section).collect();
    let history = glue(&sections)?;
    let latency_ms = t_wall.elapsed().as_secs_f64() * 1e3;
    let observation = Observation {
        latency_ms,
        bytes_moved,
        kernel_count: compiled.graph.nodes.len(),
        device_occupancy: occupancy.into_iter().collect(),
        max_abs_error: 0.0,
        notes: if used_gpu_fallback {
            vec!["GPU-placed ops executed via CPU fallback after WGSL validation".into()]
        } else {
            vec![]
        },
    };
    Ok(RunResult { output, observation, sections, history, used_gpu_fallback })
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
