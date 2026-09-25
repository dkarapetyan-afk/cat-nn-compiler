//! Partitioning and placement on a diagram in `H`.
//!
//! Cost of a placement is estimated compute time plus transfer time along
//! links of the cluster. A greedy pass is followed by a local swap search.

use catnn_core::architecture::{
    Cluster, DeviceKind, KernelBackend, KernelChoice, Placement, Realization,
};
use catnn_core::category::OpenOp;

use crate::ir::IrGraph;

#[derive(Debug, Clone)]
pub struct Cost {
    pub compute_s: f64,
    pub transfer_s: f64,
}

impl Cost {
    pub fn total(&self) -> f64 {
        self.compute_s + self.transfer_s
    }
}

pub fn flops_of(op: &OpenOp, out_numel: usize) -> f64 {
    match op {
        OpenOp::Linear { in_features, out_features, .. } => {
            // batch inferred from numel / out
            let batch = (out_numel / (*out_features).max(1)).max(1);
            (2 * batch * in_features * out_features) as f64
        }
        OpenOp::MatMul => (2 * out_numel) as f64 * 8.0,
        OpenOp::Attention { d_model, n_heads, d_head } => {
            let seq_batch = out_numel / (*d_model).max(1);
            (4 * seq_batch * d_model * d_model + 2 * n_heads * seq_batch * seq_batch * d_head)
                as f64
        }
        OpenOp::Softmax { .. } | OpenOp::LayerNorm { .. } => (6 * out_numel) as f64,
        OpenOp::Relu | OpenOp::Gelu | OpenOp::Identity => out_numel as f64,
        _ => (2 * out_numel) as f64,
    }
}

pub fn estimate(graph: &IrGraph, placement: &Placement, cluster: &Cluster) -> Cost {
    let mut compute_s = 0.0;
    let mut transfer_s = 0.0;
    for node in graph.nodes.values() {
        let dev = placement.get(&node.name).unwrap_or("");
        let gflops = cluster.find_device(dev).map(|(_, d)| d.gflops).unwrap_or(10.0);
        let flops = flops_of(&node.op, node.output_ty.numel());
        compute_s += flops / (gflops * 1e9).max(1.0);
        for pred in graph.preds(node.id) {
            if let Some(p) = graph.node(pred) {
                let src_dev = placement.get(&p.name).unwrap_or(dev);
                transfer_s += cluster.transfer_seconds(src_dev, dev, graph.bytes_of(pred));
            }
        }
    }
    Cost { compute_s, transfer_s }
}

fn is_heavy(op: &OpenOp) -> bool {
    matches!(
        op,
        OpenOp::Linear { .. }
            | OpenOp::MatMul
            | OpenOp::Attention { .. }
            | OpenOp::GqaAttention { .. }
            | OpenOp::MoE { .. }
            | OpenOp::SwiGlu { .. }
    )
}

/// Greedy placement: put each op on the device that minimises incremental cost.
///
/// Heavy ops (linear / matmul / attention) are restricted to accelerators
/// when the cluster has one, so a mixed `H` actually produces a mixed
/// realization instead of collapsing to CPU on tiny demo tensors.
pub fn place_greedy(graph: &IrGraph, cluster: &Cluster) -> Placement {
    let mut placement = Placement::new();
    let devices: Vec<String> = cluster.devices().map(|(_, d)| d.id.clone()).collect();
    let accels: Vec<String> = cluster
        .devices()
        .filter(|(_, d)| matches!(d.kind, DeviceKind::Gpu | DeviceKind::Accelerator))
        .map(|(_, d)| d.id.clone())
        .collect();
    let order = graph.topo().unwrap_or_else(|_| graph.nodes.values().map(|n| n.id).collect());
    for id in order {
        let node = match graph.node(id) {
            Some(n) => n,
            None => continue,
        };
        if node.name.ends_with(".in") {
            placement.put(&node.name, cluster.default_device().id.clone());
            continue;
        }
        let candidates: &[String] =
            if is_heavy(&node.op) && !accels.is_empty() { &accels } else { &devices };
        let mut best_dev = candidates[0].clone();
        let mut best = f64::INFINITY;
        for dev in candidates {
            placement.put(&node.name, dev);
            let c = estimate(graph, &placement, cluster);
            if c.total() < best {
                best = c.total();
                best_dev = dev.clone();
            }
        }
        placement.put(&node.name, best_dev);
    }
    placement
}

/// Local search: try swapping each op onto every other device.
pub fn place_improve(graph: &IrGraph, cluster: &Cluster, mut placement: Placement) -> Placement {
    let devices: Vec<String> = cluster.devices().map(|(_, d)| d.id.clone()).collect();
    let mut best = estimate(graph, &placement, cluster).total();
    let names: Vec<String> = graph.nodes.values().map(|n| n.name.clone()).collect();
    for name in &names {
        if name.ends_with(".in") {
            continue;
        }
        // Do not move heavy ops off an accelerator during local search —
        // that would undo the mixed-cluster policy above.
        if let Some(node) = graph.nodes.values().find(|n| n.name == *name) {
            if is_heavy(&node.op)
                && cluster.find_device(placement.get(name).unwrap_or("")).map(|(_, d)| d.kind)
                    == Some(DeviceKind::Gpu)
            {
                continue;
            }
        }
        let current = placement.get(name).unwrap_or("").to_string();
        for dev in &devices {
            if *dev == current {
                continue;
            }
            placement.put(name, dev);
            let c = estimate(graph, &placement, cluster).total();
            if c + 1e-15 < best {
                best = c;
            } else {
                placement.put(name, &current);
            }
        }
    }
    placement
}

pub fn kernels_for(graph: &IrGraph, placement: &Placement, cluster: &Cluster) -> Vec<KernelChoice> {
    graph
        .nodes
        .values()
        .map(|n| {
            let dev = placement.get(&n.name).unwrap_or("");
            let kind = cluster.find_device(dev).map(|(_, d)| d.kind);
            let backend = match kind {
                Some(DeviceKind::Gpu) | Some(DeviceKind::Accelerator) => KernelBackend::Wgsl,
                _ => KernelBackend::CpuRust,
            };
            KernelChoice { op: n.name.clone(), backend }
        })
        .collect()
}

pub fn realize(
    arch_id: &str,
    graph: &IrGraph,
    cluster: &Cluster,
    id: impl Into<String>,
) -> Realization {
    let placement = place_improve(graph, cluster, place_greedy(graph, cluster));
    let kernels = kernels_for(graph, &placement, cluster);
    let mut r = Realization::new(id, arch_id, cluster);
    r.placement = placement;
    r.kernels = kernels;
    r
}

/// Propose fusion groups: consecutive elementwise ops on the same device
/// after a linear, or linear+relu.
pub fn propose_fusions(
    graph: &IrGraph,
    placement: &Placement,
) -> Vec<catnn_core::architecture::FusionGroup> {
    let order = graph.topo().unwrap_or_default();
    let mut groups = Vec::new();
    let mut i = 0;
    while i + 1 < order.len() {
        let a = graph.node(order[i]).unwrap();
        let b = graph.node(order[i + 1]).unwrap();
        let same = placement.get(&a.name) == placement.get(&b.name);
        let fuseable =
            matches!(a.op, OpenOp::Linear { .. }) && matches!(b.op, OpenOp::Relu | OpenOp::Gelu);
        if same && fuseable {
            groups.push(catnn_core::architecture::FusionGroup {
                name: format!("{}+{}", a.name, b.name),
                ops: vec![a.name.clone(), b.name.clone()],
            });
            i += 2;
        } else {
            i += 1;
        }
    }
    groups
}

#[cfg(test)]
mod tests {
    use super::*;
    use catnn_core::architecture::Cluster;
    use catnn_core::spec::mlp_spec;

    #[test]
    fn greedy_places_every_node() {
        let arch = mlp_spec(4, 16, 32, 8).unwrap();
        let g = crate::lower::lower_architecture(&arch).unwrap();
        let p = place_greedy(&g, &Cluster::demo());
        assert_eq!(p.assign.len(), g.nodes.len());
    }
}
