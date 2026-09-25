//! Stage a specification into a forward, adjoint, and parameter-update
//! pipeline, then place the messages that pipeline sends on the cluster.
//!
//! The cluster is an input: nodes, devices, and directed links. A tensor
//! that changes device is given a path in that link graph (intra-node hops
//! use the implicit device link). There is no path when the topology does
//! not connect the two nodes — the stage fails instead of inventing a hop.
//!
//! Each device is scheduled against its own code, buffer, and scratch caps.
//! The pipeline is feasible only when every device is. A tight buffer may
//! checkpoint one activation; a tight code region may reload kernel modules.
//! Execution is refused while any region still overflows.

use std::collections::{HashMap, VecDeque};

use catnn_core::architecture::{Cluster, DeviceKind, KernelBackend, MemoryRegionKind, Placement};
use catnn_core::category::{OpenOp, Rewrite};
use catnn_core::lens::{Adam, AdamState, OptimizerKind, Sgd};
use catnn_core::tensor::Tensor;
use catnn_core::{Architecture, CatError, CatResult};

use crate::ir::IrGraph;
use crate::kernel::adjoint::{adjoint_needs_output, adjoint_of, cross_entropy, mse_loss};
use crate::kernel::resource::{describe, KernelResource};
use crate::kernel::{self};
use crate::lower;
use crate::place;
use crate::schedule::{self, BufferReq, LaunchReq, ScheduleRequest, ScheduledStep};
use crate::train::TrainOptions;

/// Where a value enters the loss.
#[derive(Debug, Clone)]
pub enum TrainTarget {
    Mse(Tensor),
    Classes(Vec<u32>),
}

/// One routed value. `hops` includes the source device and the destination.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RouteRecord {
    pub tag: String,
    pub hops: Vec<String>,
    pub bytes: u64,
    pub from_node: u32,
    pub to_node: u32,
    pub backward: bool,
}

/// Per-device memory schedule for one training step.
#[derive(Debug, Clone)]
pub struct DeviceStage {
    pub device: String,
    pub schedule: ScheduledStep,
    /// Activation names this device must rematerialize in the adjoint.
    pub checkpointed: Vec<String>,
}

/// The compiled training step: placed forward graph, routes, and the
/// per-device region schedule. Execution is a separate step and runs
/// only when [`StagedTrain::feasible`] is set.
#[derive(Debug, Clone)]
pub struct StagedTrain {
    pub graph: IrGraph,
    pub placement: Placement,
    pub routes: Vec<RouteRecord>,
    pub devices: Vec<DeviceStage>,
    pub feasible: bool,
    pub overflow_device: Option<String>,
    pub overflow_region: Option<MemoryRegionKind>,
    pub rationale: String,
}

/// Shortest directed path from `src_dev` to `dst_dev`.
///
/// Same device is a trivial path. Distinct devices on one node use the
/// implicit intra-node link. Distinct nodes walk `cluster.links`. An
/// intermediate node is represented by its first device, which is the
/// relay that will forward the message.
pub fn route(cluster: &Cluster, src_dev: &str, dst_dev: &str) -> CatResult<Vec<String>> {
    if src_dev == dst_dev {
        return Ok(vec![src_dev.to_string()]);
    }
    let src_node = cluster
        .node_of(src_dev)
        .ok_or_else(|| CatError::Wiring(format!("unknown device {src_dev}")))?;
    let dst_node = cluster
        .node_of(dst_dev)
        .ok_or_else(|| CatError::Wiring(format!("unknown device {dst_dev}")))?;
    if src_node.id == dst_node.id {
        return Ok(vec![src_dev.to_string(), dst_dev.to_string()]);
    }
    let mut prev: HashMap<String, String> = HashMap::new();
    let mut q = VecDeque::new();
    prev.insert(src_node.id.clone(), String::new());
    q.push_back(src_node.id.clone());
    while let Some(n) = q.pop_front() {
        if n == dst_node.id {
            break;
        }
        for link in &cluster.links {
            if link.src_node == n && !prev.contains_key(&link.dst_node) {
                prev.insert(link.dst_node.clone(), n.clone());
                q.push_back(link.dst_node.clone());
            }
        }
    }
    if !prev.contains_key(&dst_node.id) {
        return Err(CatError::Wiring(format!(
            "no directed path from node {} to {} ({src_dev} -> {dst_dev})",
            src_node.id, dst_node.id
        )));
    }
    let mut nodes = vec![dst_node.id.clone()];
    while nodes.last().map(String::as_str) != Some(src_node.id.as_str()) {
        let cur = nodes.last().unwrap().clone();
        let p = prev.get(&cur).cloned().unwrap_or_default();
        if p.is_empty() {
            return Err(CatError::Wiring(format!("broken path at {cur}")));
        }
        nodes.push(p);
    }
    nodes.reverse();
    let mut hops = vec![src_dev.to_string()];
    for node_id in nodes.iter().skip(1).take(nodes.len().saturating_sub(2)) {
        let relay = cluster
            .nodes
            .iter()
            .find(|n| n.id == *node_id)
            .and_then(|n| n.devices.first())
            .map(|d| d.id.clone())
            .ok_or_else(|| CatError::Wiring(format!("relay node {node_id} has no device")))?;
        if hops.last().map(String::as_str) != Some(relay.as_str()) {
            hops.push(relay);
        }
    }
    if hops.last().map(String::as_str) != Some(dst_dev) {
        hops.push(dst_dev.to_string());
    }
    Ok(hops)
}

pub fn stage(arch: &Architecture, cluster: &Cluster, opt: OptimizerKind) -> CatResult<StagedTrain> {
    stage_with(arch, cluster, opt, None)
}

pub fn stage_with(
    arch: &Architecture,
    cluster: &Cluster,
    opt: OptimizerKind,
    placement: Option<Placement>,
) -> CatResult<StagedTrain> {
    let graph = lower::lower_architecture(arch)?;
    for node in graph.nodes.values() {
        if matches!(node.op, OpenOp::Router { .. }) {
            return Err(CatError::Unlawful(format!(
                "{}: router adjoint lives inside MoE",
                node.name
            )));
        }
        if needs_params(&node.op) && node.params.is_empty() {
            return Err(CatError::Fibration(format!(
                "refusing to execute {}: parameters are not materialized",
                node.name
            )));
        }
    }
    let mut placement = match placement {
        Some(p) => p,
        None => place::realize(&arch.id, &graph, cluster, format!("{}@train", arch.id)).placement,
    };
    let default = cluster.default_device().id.clone();
    for node in graph.nodes.values() {
        if placement.get(&node.name).is_none() {
            placement.put(&node.name, &default);
        }
        let dev = placement.get(&node.name).unwrap();
        if cluster.find_device(dev).is_none() {
            return Err(CatError::Wiring(format!(
                "placement of {} names unknown device {dev}",
                node.name
            )));
        }
    }

    let order = graph.topo()?;
    let mut routes = Vec::new();
    for id in &order {
        let node = graph.node(*id).unwrap();
        let src_dev = placement.get(&node.name).unwrap();
        for succ in graph.succs(*id) {
            let succ_node = graph.node(succ).unwrap();
            let dst_dev = placement.get(&succ_node.name).unwrap();
            if src_dev == dst_dev {
                continue;
            }
            let bytes = graph.bytes_of(*id);
            let fwd = route(cluster, src_dev, dst_dev)?;
            routes.push(RouteRecord {
                tag: format!("act:{}→{}", node.name, succ_node.name),
                hops: fwd,
                bytes,
                from_node: id.0,
                to_node: succ.0,
                backward: false,
            });
            let bwd = route(cluster, dst_dev, src_dev)?;
            routes.push(RouteRecord {
                tag: format!("grad:{}←{}", node.name, succ_node.name),
                hops: bwd,
                bytes,
                from_node: succ.0,
                to_node: id.0,
                backward: true,
            });
        }
    }

    let mut devices = Vec::new();
    for (_, dev) in cluster.devices() {
        devices.push(schedule_device(&dev.id, &graph, &placement, cluster, opt)?);
    }
    devices.sort_by(|a, b| a.device.cmp(&b.device));

    let mut overflow_device = None;
    let mut overflow_region = None;
    let mut notes = Vec::new();
    for d in &devices {
        if !d.schedule.feasible && overflow_device.is_none() {
            overflow_device = Some(d.device.clone());
            overflow_region = d.schedule.overflow_region;
        }
        notes.push(format!("{}: {}", d.device, d.schedule.rationale));
    }
    let feasible = overflow_device.is_none();
    let rationale = if feasible {
        format!("feasible on {} device(s). {}", devices.len(), notes.join(" | "))
    } else {
        format!(
            "infeasible on {} ({overflow_region:?}). {}",
            overflow_device.as_deref().unwrap_or("device"),
            notes.join(" | ")
        )
    };
    Ok(StagedTrain {
        graph,
        placement,
        routes,
        devices,
        feasible,
        overflow_device,
        overflow_region,
        rationale,
    })
}

fn needs_params(op: &OpenOp) -> bool {
    !matches!(
        op,
        OpenOp::Identity
            | OpenOp::Relu
            | OpenOp::Gelu
            | OpenOp::Softmax { .. }
            | OpenOp::Add
            | OpenOp::Mul
            | OpenOp::Transpose { .. }
            | OpenOp::Reshape { .. }
            | OpenOp::ResidualAdd
    )
}

fn schedule_device(
    device: &str,
    graph: &IrGraph,
    placement: &Placement,
    cluster: &Cluster,
    opt: OptimizerKind,
) -> CatResult<DeviceStage> {
    let order = graph.topo()?;
    let index: HashMap<u32, usize> = order.iter().enumerate().map(|(i, id)| (id.0, i)).collect();
    let n = order.len().max(1);
    let loss_tick = n;
    let bwd = |id: u32| n + 1 + (n - 1 - index[&id]);
    let upd = n + 1 + n;
    let dev_of = |name: &str| placement.get(name).unwrap_or(device);
    let backend = match cluster.find_device(device).map(|(_, d)| d.kind) {
        Some(DeviceKind::Gpu) | Some(DeviceKind::Accelerator) => KernelBackend::Wgsl,
        _ => KernelBackend::CpuRust,
    };
    let regions = cluster
        .find_device(device)
        .map(|(_, d)| d.regions.clone())
        .ok_or_else(|| CatError::Wiring(format!("unknown device {device}")))?;

    let mut buffers = Vec::new();
    let mut launches = Vec::new();

    for id in &order {
        let node = graph.node(*id).unwrap();
        let prod = dev_of(&node.name);
        let fwd = index[&id.0];
        let is_input = graph.inputs.contains(id);
        let is_output = graph.outputs.contains(id);
        let mut live_to = fwd;
        if is_output {
            live_to = live_to.max(loss_tick).max(bwd(id.0));
        }
        if adjoint_needs_output(&node.op) {
            live_to = live_to.max(bwd(id.0));
        }
        let mut local_consumer = false;
        for succ in graph.succs(*id) {
            let succ_node = graph.node(succ).unwrap();
            let sd = dev_of(&succ_node.name);
            if sd == prod {
                local_consumer = true;
                live_to = live_to.max(bwd(succ.0));
            } else if sd == device {
                buffers.push(BufferReq {
                    id: format!("act:{}#recv", node.name),
                    bytes: graph.bytes_of(*id).max(1),
                    live_from: index[&succ.0],
                    live_to: bwd(succ.0),
                    checkpoint: true,
                });
            }
        }
        if prod != device {
            continue;
        }
        if !local_consumer && !is_output && !adjoint_needs_output(&node.op) {
            live_to = fwd;
        }
        buffers.push(BufferReq {
            id: format!("act:{}", node.name),
            bytes: graph.bytes_of(*id).max(1),
            live_from: fwd,
            live_to,
            // The host-injected input cannot be rematerialized.
            checkpoint: is_input,
        });
        if !is_input {
            launches.push(LaunchReq { at: fwd, kernel: describe(&node.op, &node.name, backend) });
            launches.push(LaunchReq {
                at: bwd(id.0),
                kernel: describe(&node.op, &format!("{}.bwd", node.name), backend),
            });
        }
        if !node.params.is_empty() {
            launches.push(LaunchReq { at: upd, kernel: update_resource(&node.name, backend) });
        }
        for (port, tensor) in &node.params {
            let bytes = tensor.ty.bytes() as u64;
            buffers.push(BufferReq {
                id: format!("param:{}:{port}", node.name),
                bytes,
                live_from: 0,
                live_to: upd,
                checkpoint: true,
            });
            buffers.push(BufferReq {
                id: format!("grad:{}:{port}", node.name),
                bytes: bytes.max(1),
                live_from: bwd(id.0),
                live_to: bwd(id.0),
                checkpoint: true,
            });
            if opt == OptimizerKind::Adam {
                for moment in ["m", "v"] {
                    buffers.push(BufferReq {
                        id: format!("adam:{}:{port}:{moment}", node.name),
                        bytes,
                        live_from: 0,
                        live_to: upd,
                        checkpoint: true,
                    });
                }
            }
        }
    }

    let scheduled = schedule::schedule(ScheduleRequest {
        buffers: &buffers,
        launches: &launches,
        regions: &regions,
        hierarchy: &cluster.memory,
        allow_recompute: true,
        allow_reload: true,
    });
    let mut checkpointed = Vec::new();
    for cell in &scheduled.cells {
        if let Rewrite::Recompute { tensor, .. } = &cell.rewrite {
            if let Some(name) = tensor.strip_prefix("act:") {
                if !name.contains("#recv") {
                    checkpointed.push(name.to_string());
                }
            }
        }
    }
    Ok(DeviceStage { device: device.into(), schedule: scheduled, checkpointed })
}

fn update_resource(name: &str, backend: KernelBackend) -> KernelResource {
    let mut k = describe(&OpenOp::Identity, &format!("{name}.upd"), backend);
    k.registers_per_thread = 4;
    k.static_smem_bytes = 0;
    k.dynamic_smem_bytes = 0;
    k.launch.block = [32, 1, 1];
    k
}

/// Fixed batch for a graph. Embedding inputs are quantized to token ids.
/// An `LmHead` output uses class ids; every other output uses MSE.
pub fn example_batch(graph: &IrGraph, seed: u64) -> CatResult<(Tensor, TrainTarget)> {
    let input_id =
        *graph.inputs.first().ok_or_else(|| CatError::Wiring("graph has no input".into()))?;
    let in_node = graph.node(input_id).unwrap();
    let mut input = Tensor::lcg(in_node.output_ty.clone(), seed.max(1));
    for succ in graph.succs(input_id) {
        if let OpenOp::Embedding { vocab, .. } = &graph.node(succ).unwrap().op {
            let vocab = (*vocab).max(1);
            for x in &mut input.data {
                let u = (*x * 0.5 + 0.5).clamp(0.0, 0.999999);
                *x = (u * vocab as f32) as u32 as f32;
            }
        }
    }
    let out_id =
        *graph.outputs.last().ok_or_else(|| CatError::Wiring("graph has no output".into()))?;
    let out = graph.node(out_id).unwrap();
    let target = match &out.op {
        OpenOp::LmHead { vocab, .. } => {
            let vocab = (*vocab).max(1);
            let rows = out.output_ty.numel() / vocab;
            let mut state = seed.wrapping_add(99).max(1);
            let mut ids = Vec::with_capacity(rows);
            for _ in 0..rows {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                ids.push((state % vocab as u64) as u32);
            }
            TrainTarget::Classes(ids)
        }
        _ => TrainTarget::Mse(Tensor::lcg(out.output_ty.clone(), seed.wrapping_add(11).max(1))),
    };
    Ok((input, target))
}

pub fn loss_and_grad(pred: &Tensor, target: &TrainTarget) -> CatResult<(f32, Tensor)> {
    match target {
        TrainTarget::Mse(t) => mse_loss(pred, t),
        TrainTarget::Classes(ids) => cross_entropy(pred, ids),
    }
}

pub fn apply_update(
    p: &Tensor,
    g: &Tensor,
    opt: OptimizerKind,
    lr: f32,
    adam: &mut Option<AdamState>,
) -> CatResult<Tensor> {
    match opt {
        OptimizerKind::Sgd => Sgd::new(lr).put(p, g),
        OptimizerKind::Adam => {
            let state = adam.take().unwrap_or_else(|| AdamState::zeros_like(p));
            let (next, st) = Adam::new(lr).put(p, g, &state)?;
            *adam = Some(st);
            Ok(next)
        }
    }
}

/// Single-device reference for the same step the cluster runtime runs:
/// forward, loss, adjoint, then the optimizer. Checkpointing is exact on
/// this CPU, so a feasible distributed step matches these losses.
pub fn reference_train(
    graph: &IrGraph,
    input: &Tensor,
    target: &TrainTarget,
    opts: &TrainOptions,
) -> CatResult<Vec<f32>> {
    let mut params: HashMap<u32, indexmap::IndexMap<String, Tensor>> = HashMap::new();
    for node in graph.nodes.values() {
        params.insert(node.id.0, node.params.clone());
    }
    let mut adam: HashMap<(u32, String), AdamState> = HashMap::new();
    let order = graph.topo()?;
    let mut losses = Vec::new();
    for _ in 0..opts.steps {
        let mut acts: HashMap<u32, Tensor> = HashMap::new();
        for id in &order {
            let node = graph.node(*id).unwrap();
            if graph.inputs.contains(id) {
                acts.insert(id.0, input.clone());
                continue;
            }
            let mut op = node.clone();
            if let Some(p) = params.get(&id.0) {
                op.params = p.clone();
            }
            let inputs: Vec<Tensor> = graph.preds(*id).iter().map(|p| acts[&p.0].clone()).collect();
            acts.insert(id.0, kernel::eval_cpu(&op, &inputs)?);
        }
        let out_id = graph.outputs.last().unwrap().0;
        let (loss, dy) = loss_and_grad(&acts[&out_id], target)?;
        losses.push(loss);
        let mut grads: HashMap<u32, Tensor> = HashMap::new();
        grads.insert(out_id, dy);
        let mut dparams: HashMap<u32, indexmap::IndexMap<String, Tensor>> = HashMap::new();
        for id in order.iter().rev() {
            if graph.inputs.contains(id) {
                continue;
            }
            let Some(dy) = grads.get(&id.0).cloned() else { continue };
            let node = graph.node(*id).unwrap();
            let mut op = node.clone();
            if let Some(p) = params.get(&id.0) {
                op.params = p.clone();
            }
            let preds = graph.preds(*id);
            let inputs: Vec<Tensor> = preds.iter().map(|p| acts[&p.0].clone()).collect();
            let adj = adjoint_of(&op, &inputs, &acts[&id.0], &dy)?;
            for (p, dx) in preds.iter().zip(adj.dx) {
                if let Some(prev) = grads.get(&p.0) {
                    grads.insert(p.0, crate::kernel::adjoint::add_tensors(prev, &dx)?);
                } else {
                    grads.insert(p.0, dx);
                }
            }
            dparams.insert(id.0, adj.dparams);
        }
        for (id, dp) in dparams {
            let slot = params.entry(id).or_default();
            for (port, g) in dp {
                if let Some(p) = slot.get(&port).cloned() {
                    let mut st = adam.remove(&(id, port.clone()));
                    let next = apply_update(&p, &g, opts.opt, opts.lr, &mut st)?;
                    if let Some(st) = st {
                        adam.insert((id, port.clone()), st);
                    }
                    slot.insert(port, next);
                }
            }
        }
    }
    Ok(losses)
}

/// Consecutive hops are legal when they share a node or the cluster has
/// a directed link from the source node to the destination node.
pub fn hops_follow_links(cluster: &Cluster, hops: &[String]) -> bool {
    for pair in hops.windows(2) {
        if pair[0] == pair[1] {
            continue;
        }
        let Some(a) = cluster.node_of(&pair[0]) else { return false };
        let Some(b) = cluster.node_of(&pair[1]) else { return false };
        if a.id == b.id {
            continue;
        }
        if cluster.link(&a.id, &b.id).is_none() {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use catnn_core::architecture::{Device, Link, MemoryBudget, MemoryHierarchy, Node};
    use catnn_core::spec::{mlp_spec, transformer_block_spec};

    fn line() -> Cluster {
        Cluster {
            name: "line".into(),
            nodes: vec![
                Node::new("n0", vec![Device::cpu("n0.cpu")]),
                Node::new("n1", vec![Device::cpu("n1.cpu")]),
                Node::new("n2", vec![Device::cpu("n2.cpu")]),
            ],
            links: vec![
                Link {
                    src_node: "n0".into(),
                    dst_node: "n1".into(),
                    bandwidth_bps: 1e9,
                    latency_s: 1e-4,
                },
                Link {
                    src_node: "n1".into(),
                    dst_node: "n2".into(),
                    bandwidth_bps: 1e9,
                    latency_s: 1e-4,
                },
                Link {
                    src_node: "n2".into(),
                    dst_node: "n1".into(),
                    bandwidth_bps: 1e9,
                    latency_s: 1e-4,
                },
                Link {
                    src_node: "n1".into(),
                    dst_node: "n0".into(),
                    bandwidth_bps: 1e9,
                    latency_s: 1e-4,
                },
            ],
            memory: MemoryHierarchy::desktop_cpu(),
            budget: None,
        }
    }

    #[test]
    fn route_across_a_line_relays_the_middle_node() {
        let c = line();
        let hops = route(&c, "n0.cpu", "n2.cpu").unwrap();
        assert_eq!(hops, vec!["n0.cpu", "n1.cpu", "n2.cpu"]);
        assert!(hops_follow_links(&c, &hops));
    }

    #[test]
    fn missing_link_is_not_a_route() {
        let mut c = line();
        c.links.clear();
        let err = route(&c, "n0.cpu", "n2.cpu").unwrap_err();
        assert!(err.to_string().contains("no directed path"));
    }

    #[test]
    fn mlp_stage_names_a_code_overflow() {
        let arch = mlp_spec(4, 16, 32, 8).unwrap();
        let cluster = Cluster::single_cpu().with_budget(MemoryBudget {
            code: Some(1),
            buffer: 8 << 30,
            scratch: Some(1 << 20),
        });
        let staged = stage(&arch, &cluster, OptimizerKind::Sgd).unwrap();
        assert!(!staged.feasible);
        assert_eq!(staged.overflow_region, Some(MemoryRegionKind::Code));
    }

    #[test]
    fn mlp_reference_loss_decreases() {
        let arch = mlp_spec(4, 16, 32, 8).unwrap();
        let staged = stage(&arch, &Cluster::single_cpu(), OptimizerKind::Sgd).unwrap();
        assert!(staged.feasible, "{}", staged.rationale);
        let (input, target) = example_batch(&staged.graph, 7).unwrap();
        let opts =
            TrainOptions { steps: 6, lr: 0.05, opt: OptimizerKind::Sgd, ..TrainOptions::default() };
        let losses = reference_train(&staged.graph, &input, &target, &opts).unwrap();
        assert!(losses.last().unwrap() < losses.first().unwrap(), "{losses:?}");
    }

    #[test]
    fn deepseek_mixture_schedules_on_a_cluster_and_costs_more_than_one() {
        use catnn_core::spec::{deepseek_mixture, deepseek_v3_architecture, DeepSeekV3Config};
        let cfg = DeepSeekV3Config::tiny();
        let one = deepseek_v3_architecture(&cfg, 2, 4, true).unwrap();
        let mix = deepseek_mixture(&cfg, 2, 4, true).unwrap();
        let cluster = Cluster::demo();
        let a = stage(&one, &cluster, OptimizerKind::Sgd).unwrap();
        let b = stage(&mix, &cluster, OptimizerKind::Sgd).unwrap();
        assert!(a.feasible, "{}", a.rationale);
        assert!(b.feasible, "{}", b.rationale);
        assert!(b.graph.nodes.len() > a.graph.nodes.len());
        let bytes = |s: &StagedTrain| s.devices.iter().map(|d| d.schedule.peaks.buffer).sum::<u64>();
        assert!(bytes(&b) > bytes(&a), "mixture {} vs one {}", bytes(&b), bytes(&a));
        let graph = crate::lower::lower_architecture(&mix).unwrap();
        let mut env: std::collections::HashMap<u32, catnn_core::tensor::Tensor> =
            std::collections::HashMap::new();
        let input = graph.nodes.values().find(|n| graph.inputs.contains(&n.id)).unwrap();
        env.insert(input.id.0, catnn_core::tensor::Tensor::lcg(input.output_ty.clone(), 3));
        for id in graph.topo().unwrap() {
            if graph.inputs.contains(&id) {
                continue;
            }
            let node = graph.node(id).unwrap();
            let inputs: Vec<_> = graph
                .preds(id)
                .iter()
                .map(|p| env[&p.0].clone())
                .collect();
            let y = crate::kernel::eval_cpu(node, &inputs).unwrap();
            env.insert(id.0, y);
        }
        let out = &env[&graph.outputs[0].0];
        assert_eq!(out.ty.shape.dims, vec![2, 4, cfg.vocab_size]);
        assert!(out.data.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn transformer_reference_step_is_finite() {
        let arch = transformer_block_spec(1, 2, 4, 2).unwrap();
        let staged = stage(&arch, &Cluster::single_cpu(), OptimizerKind::Sgd).unwrap();
        assert!(staged.feasible, "{}", staged.rationale);
        let (input, target) = example_batch(&staged.graph, 3).unwrap();
        let opts = TrainOptions { steps: 1, lr: 0.01, ..TrainOptions::default() };
        let losses = reference_train(&staged.graph, &input, &target, &opts).unwrap();
        assert!(losses[0].is_finite(), "{losses:?}");
    }
}
