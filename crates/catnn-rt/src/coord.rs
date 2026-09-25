//! Bulk-synchronous cluster coordination.
//!
//! Each wave, every device thread runs the ops whose inputs are already
//! in its store and emits tensors that have to leave the device. The host
//! then walks each emitted value one hop along the staged route. A hop
//! whose remainder is a single device is delivered into that device's
//! store; a longer remainder is a relay inbox on the intermediate device,
//! which the next wave forwards. Forward values, adjoints, and parameter
//! updates all use this path. A device reads the tensor the route delivered,
//! not a side copy made before the send.

use std::collections::{HashMap, HashSet};
use std::thread;
use std::time::Instant;

use catnn_compile::ir::{IrGraph, NodeId};
use catnn_compile::kernel::adjoint::{add_tensors, adjoint_needs_output, adjoint_of};
use catnn_compile::pipeline::{
    self, loss_and_grad, route, DeviceStage, RouteRecord, StagedTrain, TrainTarget,
};
use catnn_compile::train::TrainOptions;
use catnn_compile::Compiled;
use catnn_core::architecture::{DeviceKind, MemoryRegionKind, Placement};
use catnn_core::lens::{AdamState, OptimizerKind};
use catnn_core::tensor::Tensor;
use catnn_core::{Architecture, CatError, CatResult};
use indexmap::IndexMap;

use crate::handler::DeviceHandler;
use crate::sheaf::LocalSection;

#[derive(Clone)]
struct Envelope {
    tag: String,
    tensor: Tensor,
    /// Hops still to visit, including the device that should receive this
    /// envelope next.
    rest: Vec<String>,
}

struct DeviceRt {
    tensors: HashMap<String, Tensor>,
    params: HashMap<u32, IndexMap<String, Tensor>>,
    adam: HashMap<(u32, String), AdamState>,
    pending: HashMap<(u32, String), Tensor>,
    fw_done: HashSet<u32>,
    bw_done: HashSet<u32>,
    loss_done: bool,
    updates_done: bool,
    dropped: bool,
    inbox: Vec<Envelope>,
    loss: Option<f32>,
    target: Option<TrainTarget>,
    owned: Vec<u32>,
    compute: Vec<u32>,
    owns_output: bool,
}

/// Result of executing a staged pipeline.
#[derive(Debug, Clone)]
pub struct ClusterTrainReport {
    pub losses: Vec<f32>,
    pub final_params_checksum: u64,
    pub feasible: bool,
    pub rationale: String,
    pub overflow_device: Option<String>,
    pub overflow_region: Option<MemoryRegionKind>,
    pub routes: Vec<RouteRecord>,
    pub placement: Vec<(String, String)>,
    pub messages: u64,
    pub bytes_moved: u64,
    pub devices: Vec<DeviceStage>,
}

pub struct ForwardRun {
    pub output: Tensor,
    pub bytes_moved: u64,
    pub messages: u64,
    pub sections: Vec<LocalSection>,
    pub used_gpu_fallback: bool,
    pub occupancy: Vec<(String, f64)>,
    pub latency_ms: f64,
}

pub fn train_on_cluster(
    arch: &Architecture,
    cluster: &catnn_core::architecture::Cluster,
    opts: &TrainOptions,
) -> CatResult<ClusterTrainReport> {
    let staged = pipeline::stage(arch, cluster, opts.opt)?;
    execute_staged(cluster, opts, staged)
}

pub fn train_placed(
    arch: &Architecture,
    cluster: &catnn_core::architecture::Cluster,
    opts: &TrainOptions,
    placement: Placement,
) -> CatResult<ClusterTrainReport> {
    let staged = pipeline::stage_with(arch, cluster, opts.opt, Some(placement))?;
    execute_staged(cluster, opts, staged)
}

fn execute_staged(
    cluster: &catnn_core::architecture::Cluster,
    opts: &TrainOptions,
    staged: StagedTrain,
) -> CatResult<ClusterTrainReport> {
    let placement_pairs: Vec<(String, String)> =
        staged.placement.assign.iter().map(|(op, dev)| (op.clone(), dev.clone())).collect();
    if !staged.feasible {
        return Ok(ClusterTrainReport {
            losses: vec![],
            final_params_checksum: 0,
            feasible: false,
            rationale: staged.rationale,
            overflow_device: staged.overflow_device,
            overflow_region: staged.overflow_region,
            routes: staged.routes,
            placement: placement_pairs,
            messages: 0,
            bytes_moved: 0,
            devices: staged.devices,
        });
    }
    let (input, target) = pipeline::example_batch(&staged.graph, opts.seed)?;
    let input_id = staged.graph.inputs[0].0;
    let output_id = staged.graph.outputs.last().unwrap().0;
    let mut states: HashMap<String, DeviceRt> = HashMap::new();
    let mut handlers: HashMap<String, DeviceHandler> = HashMap::new();
    for (_, dev) in cluster.devices() {
        let owned: Vec<u32> = staged
            .graph
            .nodes
            .values()
            .filter(|n| staged.placement.get(&n.name) == Some(dev.id.as_str()))
            .map(|n| n.id.0)
            .collect();
        let compute: Vec<u32> = owned.iter().copied().filter(|id| *id != input_id).collect();
        let out_name = &staged.graph.node(NodeId(output_id)).unwrap().name;
        let owns_output = staged.placement.get(out_name) == Some(dev.id.as_str());
        states.insert(
            dev.id.clone(),
            DeviceRt {
                tensors: HashMap::new(),
                params: HashMap::new(),
                adam: HashMap::new(),
                pending: HashMap::new(),
                fw_done: HashSet::new(),
                bw_done: HashSet::new(),
                loss_done: !owns_output,
                updates_done: compute.is_empty(),
                dropped: false,
                inbox: vec![],
                loss: None,
                target: None,
                owned,
                compute,
                owns_output,
            },
        );
        handlers.insert(dev.id.clone(), DeviceHandler::new(dev.clone()));
    }
    for node in staged.graph.nodes.values() {
        let dev = staged.placement.get(&node.name).unwrap();
        states.get_mut(dev).unwrap().params.insert(node.id.0, node.params.clone());
    }

    let mut losses = Vec::new();
    let mut messages = 0u64;
    let mut bytes_moved = 0u64;
    let order: Vec<u32> = staged.graph.topo()?.into_iter().map(|id| id.0).collect();
    let limit =
        (order.len() * 8 + staged.routes.iter().map(|r| r.hops.len()).sum::<usize>() + 8).max(8);

    for _step in 0..opts.steps {
        for state in states.values_mut() {
            state.tensors.clear();
            state.pending.clear();
            state.fw_done.clear();
            state.bw_done.clear();
            state.inbox.clear();
            state.loss = None;
            state.dropped = false;
            state.loss_done = !state.owns_output;
            state.updates_done = state.compute.is_empty();
        }
        let input_dev = staged
            .placement
            .get(&staged.graph.node(NodeId(input_id)).unwrap().name)
            .unwrap()
            .to_string();
        states
            .get_mut(&input_dev)
            .unwrap()
            .tensors
            .insert(format!("act:{input_id}"), input.clone());
        let output_dev = staged
            .placement
            .get(&staged.graph.node(NodeId(output_id)).unwrap().name)
            .unwrap()
            .to_string();
        states.get_mut(&output_dev).unwrap().target = Some(target.clone());

        let mut settled = false;
        for _wave in 0..limit {
            let fired = run_training_wave(
                &staged,
                &order,
                opts,
                &mut states,
                &mut handlers,
                &mut messages,
                &mut bytes_moved,
            )?;
            let inbox_empty = states.values().all(|s| s.inbox.is_empty());
            let ready = states
                .values()
                .all(|s| s.updates_done && s.owned.iter().all(|id| s.fw_done.contains(id)));
            if ready && inbox_empty {
                settled = true;
                break;
            }
            if fired == 0 && inbox_empty {
                let stuck: Vec<String> = states
                    .iter()
                    .filter(|(_, s)| {
                        !s.updates_done || s.owned.iter().any(|id| !s.fw_done.contains(id))
                    })
                    .map(|(d, _)| d.clone())
                    .collect();
                return Err(CatError::Wiring(format!(
                    "training coordination stalled; devices still waiting: {}",
                    stuck.join(", ")
                )));
            }
        }
        if !settled {
            return Err(CatError::Wiring("training exceeded the coordination wave limit".into()));
        }
        let loss = states.get(&output_dev).and_then(|s| s.loss).ok_or_else(|| {
            CatError::Wiring("output device finished the step without a loss".into())
        })?;
        losses.push(loss);
    }

    let mut checksum = 0u64;
    let mut ids: Vec<u32> = staged.graph.nodes.keys().copied().collect();
    ids.sort_unstable();
    for id in ids {
        let dev = staged.placement.get(&staged.graph.node(NodeId(id)).unwrap().name).unwrap();
        if let Some(params) = states.get(dev).and_then(|s| s.params.get(&id)) {
            for tensor in params.values() {
                checksum ^= tensor.checksum();
            }
        }
    }

    Ok(ClusterTrainReport {
        losses,
        final_params_checksum: checksum,
        feasible: true,
        rationale: staged.rationale,
        overflow_device: None,
        overflow_region: None,
        routes: staged.routes,
        placement: placement_pairs,
        messages,
        bytes_moved,
        devices: staged.devices,
    })
}

fn run_training_wave(
    staged: &StagedTrain,
    order: &[u32],
    opts: &TrainOptions,
    states: &mut HashMap<String, DeviceRt>,
    handlers: &mut HashMap<String, DeviceHandler>,
    messages: &mut u64,
    bytes_moved: &mut u64,
) -> CatResult<usize> {
    let ids: Vec<String> = {
        let mut v: Vec<String> = states.keys().cloned().collect();
        v.sort();
        v
    };
    let mut fired = 0usize;
    let mut outgoing_all: Vec<Envelope> = Vec::new();
    thread::scope(|scope| -> CatResult<()> {
        let mut joins = Vec::new();
        for id in &ids {
            let state = states.remove(id).unwrap();
            let handler = handlers.remove(id).unwrap();
            let device = id.clone();
            let graph = &staged.graph;
            let placement = &staged.placement;
            let routes = &staged.routes;
            let checkpointed = staged
                .devices
                .iter()
                .find(|d| d.device == device)
                .map(|d| d.checkpointed.as_slice())
                .unwrap_or(&[]);
            let opt = opts.opt;
            let lr = opts.lr;
            joins.push(scope.spawn(move || {
                training_device(
                    device,
                    state,
                    handler,
                    graph,
                    placement,
                    routes,
                    checkpointed,
                    order,
                    opt,
                    lr,
                )
            }));
        }
        for join in joins {
            let out = join.join().map_err(|_| CatError::Io("device thread panicked".into()))??;
            fired += out.fired + out.relayed;
            outgoing_all.extend(out.outgoing);
            states.insert(out.device.clone(), out.state);
            handlers.insert(out.device, out.handler);
        }
        Ok(())
    })?;
    deliver_all(outgoing_all, states, messages, bytes_moved)?;
    Ok(fired)
}

struct TrainWave {
    device: String,
    state: DeviceRt,
    handler: DeviceHandler,
    outgoing: Vec<Envelope>,
    fired: usize,
    relayed: usize,
}

#[allow(clippy::too_many_arguments)]
fn training_device(
    device: String,
    mut state: DeviceRt,
    mut handler: DeviceHandler,
    graph: &IrGraph,
    placement: &Placement,
    routes: &[RouteRecord],
    checkpointed: &[String],
    order: &[u32],
    opt: OptimizerKind,
    lr: f32,
) -> CatResult<TrainWave> {
    let mut outgoing = Vec::new();
    let mut relayed = 0usize;
    let mut fired = 0usize;
    let inbox = std::mem::take(&mut state.inbox);
    for env in inbox {
        if env.rest.first().map(String::as_str) != Some(device.as_str()) {
            return Err(CatError::Wiring(format!(
                "misrouted {} to {device}, rest {:?}",
                env.tag, env.rest
            )));
        }
        relayed += 1;
        handler.record_transfer(&format!("relay:{}", env.tag), env.tensor.checksum(), true);
        if env.rest.len() <= 1 {
            handler.record_transfer(&format!("relay:{}", env.tag), env.tensor.checksum(), false);
            state.tensors.insert(env.tag, env.tensor);
        } else {
            let tag = env.tag.clone();
            let tensor = env.tensor;
            let rest = env.rest[1..].to_vec();
            handler.record_transfer(&format!("relay:{tag}"), tensor.checksum(), false);
            outgoing.push(Envelope { tag, tensor, rest });
        }
    }

    let cap = order.len() * 2 + 4;
    for _ in 0..cap {
        let mut progressed = false;
        for id in order {
            if forward_one(
                *id,
                &device,
                &mut state,
                &mut handler,
                graph,
                placement,
                routes,
                &mut outgoing,
            )? {
                fired += 1;
                progressed = true;
            }
        }
        if loss_one(graph, &mut state)? {
            fired += 1;
            progressed = true;
        }
        if !state.dropped
            && state.loss_done
            && state.owned.iter().all(|id| state.fw_done.contains(id))
        {
            for name in checkpointed {
                if let Some(node) = graph.nodes.values().find(|n| n.name == *name) {
                    state.tensors.remove(&format!("act:{}", node.id.0));
                }
            }
            state.dropped = true;
            progressed = true;
        }
        for id in order.iter().rev() {
            if backward_one(
                *id,
                &device,
                &mut state,
                &mut handler,
                graph,
                placement,
                routes,
                &mut outgoing,
            )? {
                fired += 1;
                progressed = true;
            }
        }
        if !state.updates_done && state.compute.iter().all(|id| state.bw_done.contains(id)) {
            apply_pending(&mut state, opt, lr)?;
            state.updates_done = true;
            progressed = true;
        }
        if !progressed {
            break;
        }
    }
    Ok(TrainWave { device, state, handler, outgoing, fired, relayed })
}

#[allow(clippy::too_many_arguments)]
fn forward_one(
    id: u32,
    device: &str,
    state: &mut DeviceRt,
    handler: &mut DeviceHandler,
    graph: &IrGraph,
    placement: &Placement,
    routes: &[RouteRecord],
    outgoing: &mut Vec<Envelope>,
) -> CatResult<bool> {
    if state.fw_done.contains(&id) {
        return Ok(false);
    }
    let node = graph.node(NodeId(id)).unwrap();
    if placement.get(&node.name) != Some(device) {
        return Ok(false);
    }
    let tag = format!("act:{id}");
    if graph.inputs.iter().any(|n| n.0 == id) {
        if state.tensors.contains_key(&tag) {
            let tensor = state.tensors.get(&tag).unwrap().clone();
            fanout(id, &tensor, device, graph, placement, routes, outgoing)?;
            state.fw_done.insert(id);
            return Ok(true);
        }
        return Ok(false);
    }
    let preds = graph.preds(NodeId(id));
    if preds.iter().any(|p| !state.tensors.contains_key(&format!("act:{}", p.0))) {
        return Ok(false);
    }
    let inputs: Vec<Tensor> =
        preds.iter().map(|p| state.tensors[&format!("act:{}", p.0)].clone()).collect();
    let mut op = node.clone();
    if let Some(p) = state.params.get(&id) {
        op.params.clone_from(p);
    }
    let y = handler.run_kernel(&op, &inputs)?;
    fanout(id, &y, device, graph, placement, routes, outgoing)?;
    state.tensors.insert(tag, y);
    state.fw_done.insert(id);
    Ok(true)
}

fn loss_one(graph: &IrGraph, state: &mut DeviceRt) -> CatResult<bool> {
    if state.loss_done {
        return Ok(false);
    }
    let out = graph.outputs.last().unwrap().0;
    let Some(pred) = state.tensors.get(&format!("act:{out}")).cloned() else {
        return Ok(false);
    };
    let Some(target) = state.target.as_ref() else {
        return Ok(false);
    };
    let (loss, dy) = loss_and_grad(&pred, target)?;
    state.tensors.insert(format!("grad:{out}"), dy);
    state.loss = Some(loss);
    state.loss_done = true;
    Ok(true)
}

#[allow(clippy::too_many_arguments)]
fn backward_one(
    id: u32,
    device: &str,
    state: &mut DeviceRt,
    handler: &mut DeviceHandler,
    graph: &IrGraph,
    placement: &Placement,
    routes: &[RouteRecord],
    outgoing: &mut Vec<Envelope>,
) -> CatResult<bool> {
    if state.bw_done.contains(&id) {
        return Ok(false);
    }
    let node = graph.node(NodeId(id)).unwrap();
    if placement.get(&node.name) != Some(device) {
        return Ok(false);
    }
    if graph.inputs.iter().any(|n| n.0 == id) {
        if state.fw_done.contains(&id) {
            state.bw_done.insert(id);
            return Ok(true);
        }
        return Ok(false);
    }
    let succs = graph.succs(NodeId(id));
    let dy = if succs.is_empty() {
        match state.tensors.get(&format!("grad:{id}")).cloned() {
            Some(g) => g,
            None => return Ok(false),
        }
    } else {
        let mut acc: Option<Tensor> = None;
        for succ in &succs {
            let tag = format!("grad:{id}:from:{}", succ.0);
            let Some(g) = state.tensors.get(&tag) else {
                return Ok(false);
            };
            acc = Some(match acc {
                None => g.clone(),
                Some(a) => add_tensors(&a, g)?,
            });
        }
        match acc {
            Some(g) => g,
            None => return Ok(false),
        }
    };
    let preds = graph.preds(NodeId(id));
    for p in &preds {
        ensure_act(p.0, device, state, graph, placement)?;
    }
    if adjoint_needs_output(&node.op) {
        ensure_act(id, device, state, graph, placement)?;
    }
    let inputs: Vec<Tensor> =
        preds.iter().map(|p| state.tensors[&format!("act:{}", p.0)].clone()).collect();
    let output =
        state.tensors.get(&format!("act:{id}")).cloned().unwrap_or_else(|| inputs[0].clone());
    let mut op = node.clone();
    if let Some(p) = state.params.get(&id) {
        op.params.clone_from(p);
    }
    handler.note_kernel(&format!("{}.bwd", op.name), dy.checksum());
    let adj = adjoint_of(&op, &inputs, &output, &dy)?;
    for (pred, dx) in preds.iter().zip(adj.dx) {
        let tag = format!("grad:{}:from:{id}", pred.0);
        let pred_node = graph.node(*pred).unwrap();
        let dest = placement.get(&pred_node.name).unwrap();
        if dest == device {
            if let Some(prev) = state.tensors.get(&tag).cloned() {
                state.tensors.insert(tag, add_tensors(&prev, &dx)?);
            } else {
                state.tensors.insert(tag, dx);
            }
        } else {
            let hops = route_between(routes, id, pred.0, true)?;
            outgoing.push(Envelope { tag, tensor: dx, rest: hops[1..].to_vec() });
        }
    }
    for (port, g) in adj.dparams {
        state.pending.insert((id, port), g);
    }
    state.bw_done.insert(id);
    Ok(true)
}

fn ensure_act(
    id: u32,
    device: &str,
    state: &mut DeviceRt,
    graph: &IrGraph,
    placement: &Placement,
) -> CatResult<()> {
    let tag = format!("act:{id}");
    if state.tensors.contains_key(&tag) {
        return Ok(());
    }
    let node = graph.node(NodeId(id)).unwrap();
    if placement.get(&node.name) != Some(device) {
        return Err(CatError::Fibration(format!(
            "activation {} is not on {device} and cannot be rematerialized there",
            node.name
        )));
    }
    if graph.inputs.iter().any(|n| n.0 == id) {
        return Err(CatError::Fibration(format!(
            "input {} was checkpointed; the host value cannot be rematerialized",
            node.name
        )));
    }
    let preds = graph.preds(NodeId(id));
    for p in &preds {
        ensure_act(p.0, device, state, graph, placement)?;
    }
    let inputs: Vec<Tensor> =
        preds.iter().map(|p| state.tensors[&format!("act:{}", p.0)].clone()).collect();
    let mut op = node.clone();
    if let Some(p) = state.params.get(&id) {
        op.params.clone_from(p);
    }
    let y = catnn_compile::kernel::eval_cpu(&op, &inputs)?;
    state.tensors.insert(tag, y);
    Ok(())
}

fn apply_pending(state: &mut DeviceRt, opt: OptimizerKind, lr: f32) -> CatResult<()> {
    let pending = std::mem::take(&mut state.pending);
    for ((id, port), g) in pending {
        let Some(slot) = state.params.get_mut(&id) else { continue };
        let Some(p) = slot.get(&port).cloned() else { continue };
        let mut st = state.adam.remove(&(id, port.clone()));
        let next = pipeline::apply_update(&p, &g, opt, lr, &mut st)?;
        if let Some(st) = st {
            state.adam.insert((id, port.clone()), st);
        }
        slot.insert(port, next);
    }
    Ok(())
}

fn fanout(
    id: u32,
    tensor: &Tensor,
    device: &str,
    graph: &IrGraph,
    placement: &Placement,
    routes: &[RouteRecord],
    outgoing: &mut Vec<Envelope>,
) -> CatResult<()> {
    for succ in graph.succs(NodeId(id)) {
        let succ_node = graph.node(succ).unwrap();
        let dest = placement.get(&succ_node.name).unwrap();
        if dest == device {
            continue;
        }
        let hops = route_between(routes, id, succ.0, false)?;
        outgoing.push(Envelope {
            tag: format!("act:{id}"),
            tensor: tensor.clone(),
            rest: hops[1..].to_vec(),
        });
    }
    Ok(())
}

fn route_between(
    routes: &[RouteRecord],
    from: u32,
    to: u32,
    backward: bool,
) -> CatResult<&[String]> {
    routes
        .iter()
        .find(|r| r.backward == backward && r.from_node == from && r.to_node == to)
        .map(|r| r.hops.as_slice())
        .ok_or_else(|| {
            CatError::Wiring(format!("no staged route {from} -> {to} backward={backward}"))
        })
}

fn deliver_all(
    outgoing: Vec<Envelope>,
    states: &mut HashMap<String, DeviceRt>,
    messages: &mut u64,
    bytes_moved: &mut u64,
) -> CatResult<()> {
    for env in outgoing {
        let dest =
            env.rest.first().cloned().ok_or_else(|| {
                CatError::Wiring(format!("route for {} has no next hop", env.tag))
            })?;
        *messages += 1;
        *bytes_moved += env.tensor.ty.bytes() as u64;
        let slot = states
            .get_mut(&dest)
            .ok_or_else(|| CatError::Wiring(format!("route delivers to unknown device {dest}")))?;
        if env.rest.len() == 1 {
            slot.tensors.insert(env.tag, env.tensor);
        } else {
            slot.inbox.push(env);
        }
    }
    Ok(())
}

/// Forward execution. Device threads run placed kernels; tensors that cross
/// devices move along [`route`] and are what the consumer reads.
pub fn run_forward(compiled: &Compiled, input: &Tensor) -> CatResult<ForwardRun> {
    let wall = Instant::now();
    let graph = &compiled.graph;
    let order: Vec<u32> = graph.topo()?.into_iter().map(|id| id.0).collect();
    let input_id = graph.inputs[0].0;
    let output_id = graph.outputs.last().unwrap().0;
    let mut routes: Vec<RouteRecord> = Vec::new();
    for edge in &graph.edges {
        let src = graph.node(edge.src).unwrap();
        let dst = graph.node(edge.dst).unwrap();
        let sd = placed(compiled, &src.name);
        let dd = placed(compiled, &dst.name);
        if sd != dd {
            let hops = route(&compiled.cluster, sd, dd)?;
            routes.push(RouteRecord {
                tag: format!("act:{}→{}", src.name, dst.name),
                hops,
                bytes: graph.bytes_of(edge.src),
                from_node: edge.src.0,
                to_node: edge.dst.0,
                backward: false,
            });
        }
    }
    let mut states: HashMap<String, DeviceRt> = HashMap::new();
    let mut handlers: HashMap<String, DeviceHandler> = HashMap::new();
    let mut gpu_devs: HashSet<String> = HashSet::new();
    for (_, dev) in compiled.cluster.devices() {
        if matches!(dev.kind, DeviceKind::Gpu | DeviceKind::Accelerator) {
            gpu_devs.insert(dev.id.clone());
        }
        let owned: Vec<u32> = graph
            .nodes
            .values()
            .filter(|n| placed(compiled, &n.name) == dev.id)
            .map(|n| n.id.0)
            .collect();
        states.insert(
            dev.id.clone(),
            DeviceRt {
                tensors: HashMap::new(),
                params: HashMap::new(),
                adam: HashMap::new(),
                pending: HashMap::new(),
                fw_done: HashSet::new(),
                bw_done: HashSet::new(),
                loss_done: true,
                updates_done: true,
                dropped: true,
                inbox: vec![],
                loss: None,
                target: None,
                owned,
                compute: vec![],
                owns_output: false,
            },
        );
        handlers.insert(dev.id.clone(), DeviceHandler::new(dev.clone()));
    }
    for node in graph.nodes.values() {
        let dev = placed(compiled, &node.name);
        states.get_mut(dev).unwrap().params.insert(node.id.0, node.params.clone());
    }
    let input_dev = placed(compiled, &graph.node(NodeId(input_id)).unwrap().name).to_string();
    states.get_mut(&input_dev).unwrap().tensors.insert(format!("act:{input_id}"), input.clone());

    let mut messages = 0u64;
    let mut bytes_moved = 0u64;
    let mut used_gpu = false;
    let mut occupancy: HashMap<String, f64> = HashMap::new();
    let limit = (order.len() * 4 + routes.iter().map(|r| r.hops.len()).sum::<usize>() + 4).max(4);
    let mut settled = false;
    for _ in 0..limit {
        let ids: Vec<String> = {
            let mut v: Vec<String> = states.keys().cloned().collect();
            v.sort();
            v
        };
        let mut outgoing_all = Vec::new();
        let mut fired = 0usize;
        thread::scope(|scope| -> CatResult<()> {
            let mut joins = Vec::new();
            for id in &ids {
                let state = states.remove(id).unwrap();
                let handler = handlers.remove(id).unwrap();
                let device = id.clone();
                let is_gpu = gpu_devs.contains(&device);
                let routes = &routes;
                let order = &order;
                joins.push(scope.spawn(move || {
                    forward_device(
                        device,
                        state,
                        handler,
                        graph,
                        &compiled.realization.placement,
                        compiled.cluster.default_device().id.as_str(),
                        routes,
                        order,
                        is_gpu,
                    )
                }));
            }
            for join in joins {
                let out =
                    join.join().map_err(|_| CatError::Io("device thread panicked".into()))??;
                fired += out.fired + out.relayed;
                used_gpu |= out.used_gpu;
                *occupancy.entry(out.device.clone()).or_insert(0.0) += out.occupancy_ms;
                outgoing_all.extend(out.outgoing);
                states.insert(out.device.clone(), out.state);
                handlers.insert(out.device, out.handler);
            }
            Ok(())
        })?;
        deliver_all(outgoing_all, &mut states, &mut messages, &mut bytes_moved)?;
        let done = states.values().all(|s| s.owned.iter().all(|id| s.fw_done.contains(id)));
        let inbox_empty = states.values().all(|s| s.inbox.is_empty());
        if done && inbox_empty {
            settled = true;
            break;
        }
        if fired == 0 && inbox_empty {
            return Err(CatError::Wiring(
                "forward coordination stalled before every placed op ran".into(),
            ));
        }
    }
    if !settled {
        return Err(CatError::Wiring("forward exceeded the coordination wave limit".into()));
    }
    let output_dev = placed(compiled, &graph.node(NodeId(output_id)).unwrap().name);
    let output = states
        .get(output_dev)
        .and_then(|s| s.tensors.get(&format!("act:{output_id}")).cloned())
        .ok_or_else(|| CatError::Wiring("output device has no activation".into()))?;
    let mut sections: Vec<LocalSection> = handlers.into_values().map(|h| h.section).collect();
    sections.sort_by(|a, b| a.device.cmp(&b.device));
    let mut occ: Vec<(String, f64)> = occupancy.into_iter().collect();
    occ.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(ForwardRun {
        output,
        bytes_moved,
        messages,
        sections,
        used_gpu_fallback: used_gpu,
        occupancy: occ,
        latency_ms: wall.elapsed().as_secs_f64() * 1e3,
    })
}

struct FwdWave {
    device: String,
    state: DeviceRt,
    handler: DeviceHandler,
    outgoing: Vec<Envelope>,
    fired: usize,
    relayed: usize,
    used_gpu: bool,
    occupancy_ms: f64,
}

#[allow(clippy::too_many_arguments)]
fn forward_device(
    device: String,
    mut state: DeviceRt,
    mut handler: DeviceHandler,
    graph: &IrGraph,
    placement: &Placement,
    default_dev: &str,
    routes: &[RouteRecord],
    order: &[u32],
    is_gpu: bool,
) -> CatResult<FwdWave> {
    let t0 = Instant::now();
    let mut outgoing = Vec::new();
    let mut relayed = 0usize;
    let mut fired = 0usize;
    let mut used_gpu = false;
    let inbox = std::mem::take(&mut state.inbox);
    for env in inbox {
        relayed += 1;
        handler.record_transfer(&format!("relay:{}", env.tag), env.tensor.checksum(), true);
        if env.rest.len() <= 1 {
            state.tensors.insert(env.tag, env.tensor);
        } else {
            outgoing.push(Envelope {
                tag: env.tag,
                tensor: env.tensor,
                rest: env.rest[1..].to_vec(),
            });
        }
    }
    for _ in 0..order.len() + 2 {
        let mut progressed = false;
        for id in order {
            if state.fw_done.contains(id) {
                continue;
            }
            let node = graph.node(NodeId(*id)).unwrap();
            let home = placement.get(&node.name).unwrap_or(default_dev);
            if home != device {
                continue;
            }
            if graph.inputs.iter().any(|n| n.0 == *id) {
                if let Some(tensor) = state.tensors.get(&format!("act:{id}")).cloned() {
                    fanout(*id, &tensor, &device, graph, placement, routes, &mut outgoing)?;
                    state.fw_done.insert(*id);
                    fired += 1;
                    progressed = true;
                }
                continue;
            }
            let preds = graph.preds(NodeId(*id));
            if preds.iter().any(|p| !state.tensors.contains_key(&format!("act:{}", p.0))) {
                continue;
            }
            for p in &preds {
                let pred = graph.node(*p).unwrap();
                let src = placement.get(&pred.name).unwrap_or(default_dev);
                if src != device {
                    let tensor = &state.tensors[&format!("act:{}", p.0)];
                    handler.record_transfer(
                        &format!("xfer:{}→{device}", pred.name),
                        tensor.checksum(),
                        false,
                    );
                }
            }
            let inputs: Vec<Tensor> =
                preds.iter().map(|p| state.tensors[&format!("act:{}", p.0)].clone()).collect();
            let mut op = node.clone();
            if let Some(p) = state.params.get(id) {
                op.params.clone_from(p);
            }
            let y = handler.run_kernel(&op, &inputs)?;
            if is_gpu {
                used_gpu = true;
            }
            fanout(*id, &y, &device, graph, placement, routes, &mut outgoing)?;
            state.tensors.insert(format!("act:{id}"), y);
            state.fw_done.insert(*id);
            fired += 1;
            progressed = true;
        }
        if !progressed {
            break;
        }
    }
    Ok(FwdWave {
        device,
        state,
        handler,
        outgoing,
        fired,
        relayed,
        used_gpu,
        occupancy_ms: t0.elapsed().as_secs_f64() * 1e3,
    })
}

fn placed<'a>(compiled: &'a Compiled, name: &str) -> &'a str {
    compiled.realization.device_of(name).unwrap_or(compiled.cluster.default_device().id.as_str())
}

#[cfg(test)]
mod tests {
    use super::*;
    use catnn_compile::pipeline::{hops_follow_links, reference_train};
    use catnn_core::architecture::{Cluster, Device, Link, MemoryBudget, MemoryHierarchy, Node};
    use catnn_core::spec::mlp_spec;

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
    fn demo_mlp_matches_reference_and_uses_only_real_links() {
        let arch = mlp_spec(4, 16, 32, 8).unwrap();
        let cluster = Cluster::demo();
        let opts =
            TrainOptions { steps: 4, lr: 0.05, opt: OptimizerKind::Sgd, ..TrainOptions::default() };
        let report = train_on_cluster(&arch, &cluster, &opts).unwrap();
        assert!(report.feasible, "{}", report.rationale);
        assert!(report.losses.len() == 4, "{:?}", report.losses);
        assert!(
            report.losses.last().unwrap() < report.losses.first().unwrap(),
            "{:?}",
            report.losses
        );
        let staged = pipeline::stage(&arch, &cluster, OptimizerKind::Sgd).unwrap();
        let (input, target) = pipeline::example_batch(&staged.graph, opts.seed).unwrap();
        let reference = reference_train(&staged.graph, &input, &target, &opts).unwrap();
        for (got, expect) in report.losses.iter().zip(reference.iter()) {
            assert!((got - expect).abs() < 1e-5, "cluster {got} reference {expect}");
        }
        for route in &report.routes {
            assert!(hops_follow_links(&cluster, &route.hops), "{:?}", route.hops);
        }
    }

    #[test]
    fn line_relays_a_forward_and_a_grad_through_the_middle() {
        let arch = mlp_spec(2, 4, 4, 2).unwrap();
        let cluster = line();
        let mut placement = Placement::new();
        placement.put("mlp.in", "n0.cpu");
        placement.put("fc1", "n0.cpu");
        placement.put("act", "n2.cpu");
        placement.put("fc2", "n2.cpu");
        let opts =
            TrainOptions { steps: 2, lr: 0.05, opt: OptimizerKind::Sgd, ..TrainOptions::default() };
        let report = train_placed(&arch, &cluster, &opts, placement).unwrap();
        assert!(report.feasible, "{}", report.rationale);
        assert!(
            report.routes.iter().any(|r| r.hops == ["n0.cpu", "n1.cpu", "n2.cpu"]),
            "{:?}",
            report.routes.iter().map(|r| &r.hops).collect::<Vec<_>>()
        );
        assert!(report.messages >= 2, "messages {}", report.messages);
        let staged = pipeline::stage(&arch, &Cluster::single_cpu(), OptimizerKind::Sgd).unwrap();
        let (input, target) = pipeline::example_batch(&staged.graph, opts.seed).unwrap();
        let reference = reference_train(&staged.graph, &input, &target, &opts).unwrap();
        for (got, expect) in report.losses.iter().zip(reference.iter()) {
            assert!((got - expect).abs() < 1e-5, "cluster {got} reference {expect}");
        }
    }

    #[test]
    fn undersized_code_does_not_run() {
        let arch = mlp_spec(4, 16, 32, 8).unwrap();
        let cluster = Cluster::single_cpu().with_budget(MemoryBudget {
            code: Some(1),
            buffer: 8 << 30,
            scratch: Some(1 << 20),
        });
        let opts = TrainOptions::default();
        let report = train_on_cluster(&arch, &cluster, &opts).unwrap();
        assert!(!report.feasible);
        assert!(report.losses.is_empty());
        assert_eq!(report.overflow_region, Some(MemoryRegionKind::Code));
    }
}
