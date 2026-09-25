//! Network control plane and peer message passing.
//!
//! One member process (or thread) listens per cluster node. The coordinator
//! dials each member, installs the placement, and runs bulk-synchronous
//! waves. A tensor that leaves a node is a length-prefixed message to the
//! member that owns the next hop. Intra-node hops stay inside that member
//! and are applied on the following wave. The consumer reads the delivered
//! tensor.

use std::collections::{HashMap, HashSet};
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use catnn_compile::ir::{IrGraph, NodeId};
use catnn_compile::pipeline::{RouteRecord, StagedTrain, TrainTarget};
use catnn_compile::train::TrainOptions;
use catnn_compile::Compiled;
use catnn_core::architecture::{Cluster, Device, DeviceKind, Placement};
use catnn_core::lens::OptimizerKind;
use catnn_core::tensor::Tensor;
use catnn_core::{CatError, CatResult};
use serde::{Deserialize, Serialize};

use crate::coord::{self, DeviceRt, Envelope, ForwardRun};
use crate::handler::DeviceHandler;
use crate::sheaf::LocalSection;

/// A node member the coordinator can dial.
#[derive(Debug, Clone)]
pub struct MemberEndpoint {
    pub node_id: String,
    pub addr: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PeerInfo {
    node_id: String,
    addr: String,
}

#[derive(Debug, Serialize, Deserialize)]
enum Role {
    Control,
    Peer { node_id: String },
    Stop,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Install {
    node_id: String,
    graph: IrGraph,
    placement: Placement,
    routes: Vec<RouteRecord>,
    peers: Vec<PeerInfo>,
    devices: Vec<Device>,
    device_node: Vec<(String, String)>,
    default_device: String,
    opt: OptimizerKind,
    lr: f32,
    checkpointed: Vec<(String, Vec<String>)>,
    infer: bool,
    input_id: u32,
    output_id: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Begin {
    input_device: Option<String>,
    input: Option<Tensor>,
    input_tag: Option<String>,
    target_device: Option<String>,
    target: Option<TrainTarget>,
}

#[derive(Debug, Serialize, Deserialize)]
enum Ctrl {
    Install(Box<Install>),
    Begin(Begin),
    Wave { id: u64 },
    Collect,
    Shutdown,
}

#[derive(Debug, Serialize, Deserialize)]
struct WaveRep {
    fired: usize,
    pending: usize,
    ready: bool,
    loss: Option<f32>,
    messages: u64,
    bytes: u64,
}

#[derive(Debug, Serialize, Deserialize)]
struct DoneRep {
    checksum: u64,
    sections: Vec<LocalSection>,
    output: Option<Tensor>,
    used_gpu: bool,
    occupancy: Vec<(String, f64)>,
}

#[derive(Debug, Serialize, Deserialize)]
enum Reply {
    Hello { node_id: String },
    Installed,
    Wave(WaveRep),
    Done(DoneRep),
    Bye,
    Failed(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct WireEnv {
    id: u64,
    tag: String,
    tensor: Tensor,
    rest: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
enum Data {
    Tensor(WireEnv),
    Ack(u64),
}

struct PeerOut {
    write: Mutex<TcpStream>,
    acks: Mutex<HashSet<u64>>,
    cv: Condvar,
}

struct Member {
    node_id: String,
    plan: Option<Install>,
    states: HashMap<String, DeviceRt>,
    handlers: HashMap<String, DeviceHandler>,
    gpu: HashSet<String>,
    deferred: Vec<Envelope>,
    peers: HashMap<String, Arc<PeerOut>>,
    incoming: Arc<Mutex<Vec<Envelope>>>,
    seq: u64,
    occupancy: HashMap<String, f64>,
    used_gpu: bool,
    order: Vec<u32>,
}

/// Bind `bind` and serve until the coordinator sends `Shutdown`.
pub fn serve_member(bind: &str, node_id: &str) -> CatResult<()> {
    let listener = TcpListener::bind(bind).map_err(|e| CatError::Io(e.to_string()))?;
    let addr = listener.local_addr().map_err(|e| CatError::Io(e.to_string()))?;
    eprintln!("member {node_id} listening {addr}");
    serve_listener(listener, node_id)
}

fn serve_listener(listener: TcpListener, node_id: &str) -> CatResult<()> {
    let incoming = Arc::new(Mutex::new(Vec::<Envelope>::new()));
    let peers: Arc<Mutex<HashMap<String, Arc<PeerOut>>>> = Arc::new(Mutex::new(HashMap::new()));
    let state = Arc::new(Mutex::new(Member {
        node_id: node_id.to_string(),
        plan: None,
        states: HashMap::new(),
        handlers: HashMap::new(),
        gpu: HashSet::new(),
        deferred: Vec::new(),
        peers: HashMap::new(),
        incoming: incoming.clone(),
        seq: 1,
        occupancy: HashMap::new(),
        used_gpu: false,
        order: Vec::new(),
    }));
    let local = listener.local_addr().map_err(|e| CatError::Io(e.to_string()))?;
    loop {
        let (sock, _) = listener.accept().map_err(|e| CatError::Io(e.to_string()))?;
        let role: Role = read_frame(&sock)?;
        match role {
            Role::Stop => break,
            Role::Control => {
                let state = state.clone();
                let peers = peers.clone();
                let incoming = incoming.clone();
                let local = local.to_string();
                thread::spawn(move || {
                    if let Err(e) = control_loop(sock, state, peers, incoming, &local) {
                        eprintln!("member control ended: {e}");
                    }
                });
            }
            Role::Peer { node_id: from } => {
                register_peer(&peers, &state, from, sock)?;
            }
        }
    }
    Ok(())
}

fn register_peer(
    peers: &Arc<Mutex<HashMap<String, Arc<PeerOut>>>>,
    state: &Arc<Mutex<Member>>,
    from: String,
    sock: TcpStream,
) -> CatResult<()> {
    let incoming = state.lock().unwrap().incoming.clone();
    let link = link_from(sock, incoming)?;
    peers.lock().unwrap().insert(from.clone(), link.clone());
    state.lock().unwrap().peers.insert(from, link);
    Ok(())
}

fn link_from(sock: TcpStream, incoming: Arc<Mutex<Vec<Envelope>>>) -> CatResult<Arc<PeerOut>> {
    sock.set_nodelay(true).ok();
    let reader = sock.try_clone().map_err(|e| CatError::Io(e.to_string()))?;
    let link = Arc::new(PeerOut {
        write: Mutex::new(sock),
        acks: Mutex::new(HashSet::new()),
        cv: Condvar::new(),
    });
    let mine = link.clone();
    thread::spawn(move || peer_reader(reader, mine, incoming));
    Ok(link)
}

fn peer_reader(reader: TcpStream, link: Arc<PeerOut>, incoming: Arc<Mutex<Vec<Envelope>>>) {
    while let Ok(msg) = read_frame::<Data>(&reader) {
        match msg {
            Data::Tensor(env) => {
                incoming.lock().unwrap().push(Envelope {
                    tag: env.tag,
                    tensor: env.tensor,
                    rest: env.rest,
                });
                let mut w = link.write.lock().unwrap();
                if write_frame(&mut *w, &Data::Ack(env.id)).is_err() {
                    break;
                }
            }
            Data::Ack(id) => {
                link.acks.lock().unwrap().insert(id);
                link.cv.notify_all();
            }
        }
    }
}

fn control_loop(
    mut sock: TcpStream,
    state: Arc<Mutex<Member>>,
    peers: Arc<Mutex<HashMap<String, Arc<PeerOut>>>>,
    incoming: Arc<Mutex<Vec<Envelope>>>,
    local_addr: &str,
) -> CatResult<()> {
    sock.set_nodelay(true).ok();
    let node_id = state.lock().unwrap().node_id.clone();
    write_frame(&mut sock, &Reply::Hello { node_id })?;
    loop {
        let ctrl: Ctrl = match read_frame(&sock) {
            Ok(c) => c,
            Err(_) => break,
        };
        let reply = match ctrl {
            Ctrl::Install(plan) => match install(&state, &peers, &incoming, *plan) {
                Ok(()) => Reply::Installed,
                Err(e) => Reply::Failed(e.to_string()),
            },
            Ctrl::Begin(begin) => match begin_step(&state, begin) {
                Ok(()) => Reply::Installed,
                Err(e) => Reply::Failed(e.to_string()),
            },
            Ctrl::Wave { .. } => match wave(&state) {
                Ok(rep) => Reply::Wave(rep),
                Err(e) => Reply::Failed(e.to_string()),
            },
            Ctrl::Collect => match collect(&state) {
                Ok(rep) => Reply::Done(rep),
                Err(e) => Reply::Failed(e.to_string()),
            },
            Ctrl::Shutdown => {
                let _ = write_frame(&mut sock, &Reply::Bye);
                let _ = TcpStream::connect(local_addr).map(|mut s| {
                    write_frame(&mut s, &Role::Stop).ok();
                    s
                });
                break;
            }
        };
        write_frame(&mut sock, &reply)?;
        if matches!(reply, Reply::Failed(_)) {
            break;
        }
    }
    Ok(())
}

fn install(
    state: &Arc<Mutex<Member>>,
    peers: &Arc<Mutex<HashMap<String, Arc<PeerOut>>>>,
    incoming: &Arc<Mutex<Vec<Envelope>>>,
    plan: Install,
) -> CatResult<()> {
    let me = plan.node_id.clone();
    for peer in &plan.peers {
        if peer.node_id <= me {
            continue;
        }
        let sock = dial(&peer.addr)?;
        write_frame(&sock, &Role::Peer { node_id: me.clone() })?;
        let link = link_from(sock, incoming.clone())?;
        peers.lock().unwrap().insert(peer.node_id.clone(), link.clone());
        state.lock().unwrap().peers.insert(peer.node_id.clone(), link);
    }
    let deadline = Instant::now() + Duration::from_secs(5);
    let need: HashSet<String> =
        plan.peers.iter().map(|p| p.node_id.clone()).filter(|id| id != &me).collect();
    while Instant::now() < deadline {
        let have: HashSet<String> = state.lock().unwrap().peers.keys().cloned().collect();
        if need.is_subset(&have) {
            break;
        }
        thread::sleep(Duration::from_millis(10));
    }
    let have: HashSet<String> = state.lock().unwrap().peers.keys().cloned().collect();
    if !need.is_subset(&have) {
        return Err(CatError::Io(format!("member {me} missing peers")));
    }
    let mut member = state.lock().unwrap();
    let order = plan.graph.topo()?.into_iter().map(|id| id.0).collect();
    member.order = order;
    member.gpu = plan
        .devices
        .iter()
        .filter(|d| matches!(d.kind, DeviceKind::Gpu | DeviceKind::Accelerator))
        .map(|d| d.id.clone())
        .collect();
    member.states.clear();
    member.handlers.clear();
    for dev in &plan.devices {
        member.handlers.insert(dev.id.clone(), DeviceHandler::new(dev.clone()));
        member.states.insert(dev.id.clone(), fresh_rt(&plan, dev));
    }
    for node in plan.graph.nodes.values() {
        let dev = plan.placement.get(&node.name).unwrap_or(&plan.default_device);
        if let Some(slot) = member.states.get_mut(dev) {
            slot.params.insert(node.id.0, node.params.clone());
        }
    }
    member.plan = Some(plan);
    Ok(())
}

fn fresh_rt(plan: &Install, dev: &Device) -> DeviceRt {
    let owned: Vec<u32> = plan
        .graph
        .nodes
        .values()
        .filter(|n| plan.placement.get(&n.name) == Some(dev.id.as_str()))
        .map(|n| n.id.0)
        .collect();
    let compute: Vec<u32> = owned.iter().copied().filter(|id| *id != plan.input_id).collect();
    let out_name = plan.graph.node(NodeId(plan.output_id)).map(|n| n.name.as_str()).unwrap_or("");
    let owns_output = plan.placement.get(out_name) == Some(dev.id.as_str());
    DeviceRt {
        tensors: HashMap::new(),
        params: HashMap::new(),
        adam: HashMap::new(),
        pending: HashMap::new(),
        fw_done: HashSet::new(),
        bw_done: HashSet::new(),
        loss_done: plan.infer || !owns_output,
        updates_done: plan.infer || compute.is_empty(),
        dropped: plan.infer,
        inbox: vec![],
        loss: None,
        target: None,
        owned,
        compute: if plan.infer { vec![] } else { compute },
        owns_output,
    }
}

fn begin_step(state: &Arc<Mutex<Member>>, begin: Begin) -> CatResult<()> {
    let mut member = state.lock().unwrap();
    member.incoming.lock().unwrap().clear();
    member.deferred.clear();
    member.occupancy.clear();
    member.used_gpu = false;
    let infer = member.plan.as_ref().map(|p| p.infer).unwrap_or(false);
    for st in member.states.values_mut() {
        st.tensors.clear();
        st.pending.clear();
        st.fw_done.clear();
        st.bw_done.clear();
        st.inbox.clear();
        st.loss = None;
        st.dropped = infer;
        st.loss_done = infer || !st.owns_output;
        st.updates_done = infer || st.compute.is_empty();
        st.target = None;
    }
    if let (Some(dev), Some(tensor), Some(tag)) = (begin.input_device, begin.input, begin.input_tag)
    {
        if let Some(slot) = member.states.get_mut(&dev) {
            slot.tensors.insert(tag, tensor);
        }
    }
    if let (Some(dev), Some(target)) = (begin.target_device, begin.target) {
        if let Some(slot) = member.states.get_mut(&dev) {
            slot.target = Some(target);
        }
    }
    Ok(())
}

fn wave(state: &Arc<Mutex<Member>>) -> CatResult<WaveRep> {
    let mut member = state.lock().unwrap();
    let batch = std::mem::take(&mut *member.incoming.lock().unwrap());
    let deferred = std::mem::take(&mut member.deferred);
    let mut messages = 0u64;
    let mut bytes = 0u64;
    let mut together = batch;
    together.extend(deferred);
    coord::deliver_all(together, &mut member.states, &mut messages, &mut bytes)?;
    let plan = member.plan.clone().ok_or_else(|| CatError::Wiring("member has no plan".into()))?;
    let ids: Vec<String> = {
        let mut v: Vec<String> = member.states.keys().cloned().collect();
        v.sort();
        v
    };
    let mut fired = 0usize;
    let mut outgoing_all: Vec<Envelope> = Vec::new();
    let mut next_states = HashMap::new();
    let mut next_handlers = HashMap::new();
    let order = member.order.clone();
    let infer = plan.infer;
    let opt = plan.opt;
    let lr = plan.lr;
    let ckpt: HashMap<String, Vec<String>> = plan.checkpointed.iter().cloned().collect();
    thread::scope(|scope| -> CatResult<()> {
        let mut joins = Vec::new();
        for id in &ids {
            let st = member.states.remove(id).unwrap();
            let handler = member.handlers.remove(id).unwrap();
            let device = id.clone();
            let graph = &plan.graph;
            let placement = &plan.placement;
            let routes = &plan.routes;
            let checkpointed = ckpt.get(&device).map(Vec::as_slice).unwrap_or(&[]);
            let is_gpu = member.gpu.contains(&device);
            let default_dev = plan.default_device.clone();
            let order = order.clone();
            joins.push(scope.spawn(move || {
                if infer {
                    let wave = coord::forward_device(
                        device,
                        st,
                        handler,
                        graph,
                        placement,
                        &default_dev,
                        routes,
                        &order,
                        is_gpu,
                    )?;
                    let dev_name = wave.device.clone();
                    Ok((
                        dev_name,
                        wave.fired + wave.relayed,
                        wave.outgoing,
                        wave.state,
                        wave.handler,
                        wave.used_gpu,
                        wave.occupancy_ms,
                    ))
                } else {
                    let wave = coord::training_device(
                        device,
                        st,
                        handler,
                        graph,
                        placement,
                        routes,
                        checkpointed,
                        &order,
                        opt,
                        lr,
                    )?;
                    let dev_name = wave.device.clone();
                    Ok((
                        dev_name,
                        wave.fired + wave.relayed,
                        wave.outgoing,
                        wave.state,
                        wave.handler,
                        false,
                        0.0,
                    ))
                }
            }));
        }
        for join in joins {
            let (dev_occ, n, outgoing, st, handler, gpu, occ) =
                join.join().map_err(|_| CatError::Io("device thread panicked".into()))??;
            fired += n;
            if gpu {
                member.used_gpu = true;
            }
            *member.occupancy.entry(dev_occ).or_insert(0.0) += occ;
            outgoing_all.extend(outgoing);
            let dev = handler.device.id.clone();
            next_states.insert(dev.clone(), st);
            next_handlers.insert(dev, handler);
        }
        Ok(())
    })?;
    member.states = next_states;
    member.handlers = next_handlers;
    let device_node: HashMap<String, String> = plan.device_node.iter().cloned().collect();
    let mut remote: Vec<(String, Envelope)> = Vec::new();
    for env in outgoing_all {
        let next =
            env.rest.first().cloned().ok_or_else(|| {
                CatError::Wiring(format!("route for {} has no next hop", env.tag))
            })?;
        let node = device_node.get(&next).cloned().unwrap_or_default();
        if node == member.node_id {
            member.deferred.push(env);
        } else {
            remote.push((node, env));
        }
    }
    let ready = member.states.values().all(|s| {
        s.updates_done && s.owned.iter().all(|id| s.fw_done.contains(id)) && s.inbox.is_empty()
    });
    let loss = member.states.values().find_map(|s| s.loss);
    let peers = member.peers.clone();
    let mut seq = member.seq;
    drop(member);
    for (node, env) in remote {
        let link = peers.get(&node).cloned().ok_or_else(|| {
            CatError::Wiring(format!("no peer link to node {node} for {}", env.tag))
        })?;
        seq += 1;
        send_tensor(&link, seq, env)?;
    }
    let mut member = state.lock().unwrap();
    member.seq = seq;
    let pending = member.deferred.len() + member.incoming.lock().unwrap().len();
    Ok(WaveRep { fired, pending, ready, loss, messages, bytes })
}

fn send_tensor(link: &PeerOut, id: u64, env: Envelope) -> CatResult<()> {
    let wire = Data::Tensor(WireEnv { id, tag: env.tag, tensor: env.tensor, rest: env.rest });
    {
        let mut w = link.write.lock().unwrap();
        write_frame(&mut *w, &wire)?;
    }
    let deadline = Instant::now() + Duration::from_secs(30);
    let mut acks = link.acks.lock().unwrap();
    while !acks.contains(&id) {
        let now = Instant::now();
        if now >= deadline {
            return Err(CatError::Io(format!("peer ack {id} timed out")));
        }
        let (guard, timeout) = link.cv.wait_timeout(acks, deadline - now).unwrap();
        acks = guard;
        if timeout.timed_out() && !acks.contains(&id) {
            return Err(CatError::Io(format!("peer ack {id} timed out")));
        }
    }
    Ok(())
}

fn collect(state: &Arc<Mutex<Member>>) -> CatResult<DoneRep> {
    let member = state.lock().unwrap();
    let mut checksum = 0u64;
    for st in member.states.values() {
        for params in st.params.values() {
            for tensor in params.values() {
                checksum ^= tensor.checksum();
            }
        }
    }
    let plan = member.plan.as_ref();
    let output = plan.and_then(|p| {
        let name = p.graph.node(NodeId(p.output_id))?.name.clone();
        let dev = p.placement.get(&name)?.to_string();
        member.states.get(&dev)?.tensors.get(&format!("act:{}", p.output_id)).cloned()
    });
    let mut sections: Vec<LocalSection> =
        member.handlers.values().map(|h| h.section.clone()).collect();
    sections.sort_by(|a, b| a.device.cmp(&b.device));
    let mut occupancy: Vec<(String, f64)> =
        member.occupancy.iter().map(|(k, v)| (k.clone(), *v)).collect();
    occupancy.retain(|(k, _)| !k.is_empty());
    occupancy.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(DoneRep { checksum, sections, output, used_gpu: member.used_gpu, occupancy })
}

fn dial(addr: &str) -> CatResult<TcpStream> {
    let deadline = Instant::now() + Duration::from_secs(5);
    let mut last = String::from("not attempted");
    while Instant::now() < deadline {
        match TcpStream::connect(addr) {
            Ok(s) => return Ok(s),
            Err(e) => last = e.to_string(),
        }
        thread::sleep(Duration::from_millis(20));
    }
    Err(CatError::Io(format!("dial {addr}: {last}")))
}

fn write_frame<T: Serialize>(mut stream: impl Write, value: &T) -> CatResult<()> {
    let payload = serde_json::to_vec(value).map_err(|e| CatError::Io(e.to_string()))?;
    let len =
        u32::try_from(payload.len()).map_err(|_| CatError::Io("frame exceeds 4 GiB".into()))?;
    stream.write_all(&len.to_le_bytes()).map_err(|e| CatError::Io(e.to_string()))?;
    stream.write_all(&payload).map_err(|e| CatError::Io(e.to_string()))?;
    stream.flush().ok();
    Ok(())
}

fn read_frame<T: for<'de> Deserialize<'de>>(mut stream: impl Read) -> CatResult<T> {
    let mut lenb = [0u8; 4];
    stream.read_exact(&mut lenb).map_err(|e| CatError::Io(e.to_string()))?;
    let len = u32::from_le_bytes(lenb) as usize;
    if len > 64 << 20 {
        return Err(CatError::Io(format!("frame length {len} exceeds 64 MiB")));
    }
    let mut buf = vec![0u8; len];
    stream.read_exact(&mut buf).map_err(|e| CatError::Io(e.to_string()))?;
    serde_json::from_slice(&buf).map_err(|e| CatError::Io(e.to_string()))
}

struct Booted {
    endpoints: Vec<MemberEndpoint>,
    addrs: Vec<String>,
}

fn boot(cluster: &Cluster) -> CatResult<Booted> {
    let mut endpoints = Vec::new();
    let mut addrs = Vec::new();
    for node in &cluster.nodes {
        let listener = TcpListener::bind("127.0.0.1:0").map_err(|e| CatError::Io(e.to_string()))?;
        let addr = listener.local_addr().map_err(|e| CatError::Io(e.to_string()))?.to_string();
        let node_id = node.id.clone();
        let serve_addr = addr.clone();
        thread::spawn(move || {
            if let Err(e) = serve_listener(listener, &node_id) {
                eprintln!("member {node_id} ({serve_addr}) stopped: {e}");
            }
        });
        endpoints.push(MemberEndpoint { node_id: node.id.clone(), addr: addr.clone() });
        addrs.push(addr);
    }
    Ok(Booted { endpoints, addrs })
}

fn stop_all(addrs: &[String]) {
    for addr in addrs {
        if let Ok(mut s) = TcpStream::connect(addr) {
            let _ = write_frame(&mut s, &Role::Stop);
        }
    }
}

struct Session {
    socks: Vec<TcpStream>,
    node_ids: Vec<String>,
    addrs: Vec<String>,
}

impl Drop for Session {
    fn drop(&mut self) {
        for sock in &mut self.socks {
            let _ = write_frame(sock, &Ctrl::Shutdown);
        }
        stop_all(&self.addrs);
    }
}

fn connect_all(endpoints: &[MemberEndpoint]) -> CatResult<Vec<TcpStream>> {
    let mut socks = Vec::new();
    for ep in endpoints {
        let mut sock = dial(&ep.addr)?;
        sock.set_nodelay(true).ok();
        write_frame(&mut sock, &Role::Control)?;
        let hello: Reply = read_frame(&mut sock)?;
        match hello {
            Reply::Hello { node_id } if node_id == ep.node_id => socks.push(sock),
            Reply::Hello { node_id } => {
                return Err(CatError::Wiring(format!(
                    "member at {} is {node_id}, expected {}",
                    ep.addr, ep.node_id
                )));
            }
            other => {
                return Err(CatError::Io(format!(
                    "expected hello from {}, got {other:?}",
                    ep.node_id
                )));
            }
        }
    }
    Ok(socks)
}

fn install_all(
    cluster: &Cluster,
    socks: &mut [TcpStream],
    infer: bool,
    plan: &InstallShared,
) -> CatResult<()> {
    let peers: Vec<PeerInfo> = plan
        .endpoints
        .iter()
        .map(|e| PeerInfo { node_id: e.node_id.clone(), addr: e.addr.clone() })
        .collect();
    let device_node: Vec<(String, String)> =
        cluster.devices().map(|(n, d)| (d.id.clone(), n.id.clone())).collect();
    for (sock, ep) in socks.iter_mut().zip(plan.endpoints.iter()) {
        let devices: Vec<Device> = cluster
            .nodes
            .iter()
            .find(|n| n.id == ep.node_id)
            .map(|n| n.devices.clone())
            .unwrap_or_default();
        let checkpointed: Vec<(String, Vec<String>)> = plan
            .checkpointed
            .iter()
            .filter(|(dev, _)| devices.iter().any(|d| d.id == *dev))
            .cloned()
            .collect();
        let msg = Ctrl::Install(Box::new(Install {
            node_id: ep.node_id.clone(),
            graph: plan.graph.clone(),
            placement: plan.placement.clone(),
            routes: plan.routes.clone(),
            peers: peers.clone(),
            devices,
            device_node: device_node.clone(),
            default_device: cluster.default_device().id.clone(),
            opt: plan.opt,
            lr: plan.lr,
            checkpointed,
            infer,
            input_id: plan.input_id,
            output_id: plan.output_id,
        }));
        write_frame(sock, &msg)?;
    }
    for sock in socks.iter_mut() {
        expect_installed(sock)?;
    }
    Ok(())
}

fn expect_installed(sock: &mut TcpStream) -> CatResult<()> {
    match read_frame(sock)? {
        Reply::Installed => Ok(()),
        Reply::Failed(s) => Err(CatError::Wiring(s)),
        other => Err(CatError::Io(format!("expected installed, got {other:?}"))),
    }
}

struct InstallShared<'a> {
    endpoints: &'a [MemberEndpoint],
    graph: IrGraph,
    placement: Placement,
    routes: Vec<RouteRecord>,
    checkpointed: Vec<(String, Vec<String>)>,
    opt: OptimizerKind,
    lr: f32,
    input_id: u32,
    output_id: u32,
}

pub(crate) struct TrainNet {
    pub losses: Vec<f32>,
    pub checksum: u64,
    pub messages: u64,
    pub bytes_moved: u64,
}

pub(crate) fn run_training(
    cluster: &Cluster,
    opts: &TrainOptions,
    staged: &StagedTrain,
    attached: Option<&[MemberEndpoint]>,
) -> CatResult<TrainNet> {
    let booted = if attached.is_none() { Some(boot(cluster)?) } else { None };
    let endpoints: Vec<MemberEndpoint> =
        attached.map(|e| e.to_vec()).unwrap_or_else(|| booted.as_ref().unwrap().endpoints.clone());
    let owned_addrs = booted.map(|b| b.addrs).unwrap_or_default();
    let result = (|| {
        let mut socks = connect_all(&endpoints)?;
        let mut session = Session {
            socks: Vec::new(),
            node_ids: endpoints.iter().map(|e| e.node_id.clone()).collect(),
            addrs: owned_addrs.clone(),
        };
        let shared = InstallShared {
            endpoints: &endpoints,
            graph: staged.graph.clone(),
            placement: staged.placement.clone(),
            routes: staged.routes.clone(),
            checkpointed: staged
                .devices
                .iter()
                .map(|d| (d.device.clone(), d.checkpointed.clone()))
                .collect(),
            opt: opts.opt,
            lr: opts.lr,
            input_id: staged.graph.inputs[0].0,
            output_id: staged.graph.outputs.last().unwrap().0,
        };
        install_all(cluster, &mut socks, false, &shared)?;
        session.socks = socks;
        drive_train(cluster, opts, staged, &mut session)
    })();
    if result.is_err() {
        stop_all(&owned_addrs);
    }
    result
}

fn drive_train(
    cluster: &Cluster,
    opts: &TrainOptions,
    staged: &StagedTrain,
    session: &mut Session,
) -> CatResult<TrainNet> {
    let (input, target) = catnn_compile::pipeline::example_batch(&staged.graph, opts.seed)?;
    let input_id = staged.graph.inputs[0].0;
    let output_id = staged.graph.outputs.last().unwrap().0;
    let input_dev = staged
        .placement
        .get(&staged.graph.node(NodeId(input_id)).unwrap().name)
        .unwrap()
        .to_string();
    let output_dev = staged
        .placement
        .get(&staged.graph.node(NodeId(output_id)).unwrap().name)
        .unwrap()
        .to_string();
    let input_node = cluster.node_of(&input_dev).map(|n| n.id.clone()).unwrap_or_default();
    let output_node = cluster.node_of(&output_dev).map(|n| n.id.clone()).unwrap_or_default();
    let limit = (staged.graph.nodes.len() * 8
        + staged.routes.iter().map(|r| r.hops.len()).sum::<usize>()
        + 8)
    .max(8);
    let mut losses = Vec::new();
    let mut messages = 0u64;
    let mut bytes_moved = 0u64;
    let _ = (input_node, output_node);
    for _step in 0..opts.steps {
        begin_payload(
            session,
            cluster,
            Some((input_dev.clone(), input.clone(), format!("act:{input_id}"))),
            Some((output_dev.clone(), target.clone())),
        )?;
        let mut settled = false;
        for _wave in 0..limit {
            let reps = broadcast_wave(session)?;
            messages += reps.iter().map(|r| r.messages).sum::<u64>();
            bytes_moved += reps.iter().map(|r| r.bytes).sum::<u64>();
            let ready = reps.iter().all(|r| r.ready && r.pending == 0);
            let fired: usize = reps.iter().map(|r| r.fired).sum();
            if ready && fired == 0 {
                settled = true;
                let loss = reps.iter().find_map(|r| r.loss).ok_or_else(|| {
                    CatError::Wiring("output device finished the step without a loss".into())
                })?;
                losses.push(loss);
                break;
            }
            if fired == 0 && reps.iter().all(|r| r.pending == 0) {
                return Err(CatError::Wiring(
                    "training coordination stalled; a member is still waiting".into(),
                ));
            }
        }
        if !settled {
            return Err(CatError::Wiring("training exceeded the coordination wave limit".into()));
        }
    }
    let done = collect_all(session)?;
    let checksum = done.iter().fold(0u64, |acc, d| acc ^ d.checksum);
    Ok(TrainNet { losses, checksum, messages, bytes_moved })
}

fn broadcast_wave(session: &mut Session) -> CatResult<Vec<WaveRep>> {
    for sock in &mut session.socks {
        write_frame(sock, &Ctrl::Wave { id: 0 })?;
    }
    let mut reps = Vec::new();
    for sock in &mut session.socks {
        match read_frame(sock)? {
            Reply::Wave(rep) => reps.push(rep),
            Reply::Failed(s) => return Err(CatError::Wiring(s)),
            other => return Err(CatError::Io(format!("expected wave report, got {other:?}"))),
        }
    }
    Ok(reps)
}

fn collect_all(session: &mut Session) -> CatResult<Vec<DoneRep>> {
    for sock in &mut session.socks {
        write_frame(sock, &Ctrl::Collect)?;
    }
    let mut out = Vec::new();
    for sock in &mut session.socks {
        match read_frame(sock)? {
            Reply::Done(rep) => out.push(rep),
            Reply::Failed(s) => return Err(CatError::Wiring(s)),
            other => return Err(CatError::Io(format!("expected collect, got {other:?}"))),
        }
    }
    Ok(out)
}

pub(crate) fn run_forward(
    compiled: &Compiled,
    input: &Tensor,
    routes: Vec<RouteRecord>,
) -> CatResult<ForwardRun> {
    let wall = Instant::now();
    let cluster = &compiled.cluster;
    let booted = boot(cluster)?;
    let endpoints = booted.endpoints.clone();
    let result = (|| {
        let mut socks = connect_all(&endpoints)?;
        let mut session = Session {
            socks: Vec::new(),
            node_ids: endpoints.iter().map(|e| e.node_id.clone()).collect(),
            addrs: booted.addrs.clone(),
        };
        let graph = &compiled.graph;
        let input_id = graph.inputs[0].0;
        let output_id = graph.outputs.last().unwrap().0;
        let shared = InstallShared {
            endpoints: &endpoints,
            graph: graph.clone(),
            placement: compiled.realization.placement.clone(),
            routes,
            checkpointed: vec![],
            opt: OptimizerKind::Sgd,
            lr: 0.0,
            input_id,
            output_id,
        };
        install_all(cluster, &mut socks, true, &shared)?;
        session.socks = socks;
        let input_name = graph.node(NodeId(input_id)).unwrap().name.clone();
        let output_name = graph.node(NodeId(output_id)).unwrap().name.clone();
        let input_dev = compiled
            .realization
            .device_of(&input_name)
            .unwrap_or(cluster.default_device().id.as_str())
            .to_string();
        let _output_dev = compiled
            .realization
            .device_of(&output_name)
            .unwrap_or(cluster.default_device().id.as_str())
            .to_string();
        begin_payload(
            &mut session,
            cluster,
            Some((input_dev, input.clone(), format!("act:{input_id}"))),
            None,
        )?;
        let limit =
            (graph.nodes.len() * 4 + shared.routes.iter().map(|r| r.hops.len()).sum::<usize>() + 4)
                .max(4);
        let mut messages = 0u64;
        let mut bytes_moved = 0u64;
        let mut settled = false;
        for _ in 0..limit {
            let reps = broadcast_wave(&mut session)?;
            messages += reps.iter().map(|r| r.messages).sum::<u64>();
            bytes_moved += reps.iter().map(|r| r.bytes).sum::<u64>();
            let ready = reps.iter().all(|r| r.ready && r.pending == 0);
            let fired: usize = reps.iter().map(|r| r.fired).sum();
            if ready && fired == 0 {
                settled = true;
                break;
            }
            if fired == 0 && reps.iter().all(|r| r.pending == 0) {
                return Err(CatError::Wiring(
                    "forward coordination stalled before every placed op ran".into(),
                ));
            }
        }
        if !settled {
            return Err(CatError::Wiring("forward exceeded the coordination wave limit".into()));
        }
        let done = collect_all(&mut session)?;
        let output = done
            .iter()
            .find_map(|d| d.output.clone())
            .ok_or_else(|| CatError::Wiring("output device has no activation".into()))?;
        let mut sections = Vec::new();
        let mut used_gpu = false;
        let mut occupancy = Vec::new();
        for d in done {
            used_gpu |= d.used_gpu;
            sections.extend(d.sections);
            occupancy.extend(d.occupancy);
        }
        sections.sort_by(|a, b| a.device.cmp(&b.device));
        occupancy.sort_by(|a, b| a.0.cmp(&b.0));
        Ok(ForwardRun {
            output,
            bytes_moved,
            messages,
            sections,
            used_gpu_fallback: used_gpu,
            occupancy,
            latency_ms: wall.elapsed().as_secs_f64() * 1e3,
        })
    })();
    if result.is_err() {
        stop_all(&booted.addrs);
    }
    result
}

fn begin_payload(
    session: &mut Session,
    cluster: &Cluster,
    input: Option<(String, Tensor, String)>,
    target: Option<(String, TrainTarget)>,
) -> CatResult<()> {
    let node_of = |dev: &str| cluster.node_of(dev).map(|n| n.id.clone()).unwrap_or_default();
    let nodes = session.node_ids.clone();
    for (sock, node) in session.socks.iter_mut().zip(nodes.iter()) {
        let mut begin = Begin {
            input_device: None,
            input: None,
            input_tag: None,
            target_device: None,
            target: None,
        };
        if let Some((dev, tensor, tag)) = &input {
            if node_of(dev) == *node {
                begin.input_device = Some(dev.clone());
                begin.input = Some(tensor.clone());
                begin.input_tag = Some(tag.clone());
            }
        }
        if let Some((dev, tgt)) = &target {
            if node_of(dev) == *node {
                begin.target_device = Some(dev.clone());
                begin.target = Some(tgt.clone());
            }
        }
        write_frame(sock, &Ctrl::Begin(begin))?;
    }
    for sock in session.socks.iter_mut() {
        expect_installed(sock)?;
    }
    Ok(())
}

/// Train against members that are already listening.
pub fn train_attached(
    arch: &catnn_core::Architecture,
    cluster: &Cluster,
    opts: &TrainOptions,
    members: &[MemberEndpoint],
) -> CatResult<crate::coord::ClusterTrainReport> {
    let staged = catnn_compile::pipeline::stage(arch, cluster, opts.opt)?;
    crate::coord::execute_attached(cluster, opts, staged, Some(members))
}
