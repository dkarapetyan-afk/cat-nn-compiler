//! Hardware / environment category `H` (spec §6).
//!
//! Objects are devices and nodes; morphisms are *links* (transfer channels)
//! with bandwidth and latency. A [`Cluster`] is a diagram in `H`.

use serde::{Deserialize, Serialize};

use crate::architecture::memory::{DeviceMemory, MemoryBudget, MemoryHierarchy};

/// Kind of compute device. The runtime has a handler per kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DeviceKind {
    Cpu,
    Gpu,
    Accelerator,
}

impl DeviceKind {
    pub fn as_str(self) -> &'static str {
        match self {
            DeviceKind::Cpu => "cpu",
            DeviceKind::Gpu => "gpu",
            DeviceKind::Accelerator => "accel",
        }
    }

    /// Relative flop throughput used by the cost model (CPU = 1).
    pub fn flop_rate(self) -> f64 {
        match self {
            DeviceKind::Cpu => 1.0,
            DeviceKind::Gpu => 8.0,
            DeviceKind::Accelerator => 4.0,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Device {
    pub id: String,
    pub kind: DeviceKind,
    /// Device-local *buffer* memory in bytes (legacy single number).
    pub memory_bytes: u64,
    /// Optional peak GFLOP/s override; default from [`DeviceKind::flop_rate`].
    pub gflops: f64,
    /// Structured code / buffer / scratch object of `H`.
    #[serde(default)]
    pub regions: DeviceMemory,
}

impl Device {
    pub fn cpu(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            kind: DeviceKind::Cpu,
            memory_bytes: 8 << 30,
            gflops: 50.0,
            regions: DeviceMemory::desktop_cpu(),
        }
    }

    pub fn gpu(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            kind: DeviceKind::Gpu,
            memory_bytes: 8 << 30,
            gflops: 400.0,
            regions: DeviceMemory::desktop_cpu(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Node {
    pub id: String,
    pub devices: Vec<Device>,
}

impl Node {
    pub fn new(id: impl Into<String>, devices: Vec<Device>) -> Self {
        Self { id: id.into(), devices }
    }

    pub fn device(&self, id: &str) -> Option<&Device> {
        self.devices.iter().find(|d| d.id == id)
    }
}

/// A directed link in `H`. Cost of a transfer of `n` bytes is
/// `latency_s + n / bandwidth_bps`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Link {
    pub src_node: String,
    pub dst_node: String,
    pub bandwidth_bps: f64,
    pub latency_s: f64,
}

impl Link {
    pub fn transfer_seconds(&self, bytes: u64) -> f64 {
        self.latency_s + (bytes as f64) / self.bandwidth_bps.max(1.0)
    }
}

/// Heterogeneous cluster: objects of a diagram in `H`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Cluster {
    pub name: String,
    pub nodes: Vec<Node>,
    pub links: Vec<Link>,
    /// Memory object of `H`. Indexes the realization fibration: a smaller
    /// budget reindexes to more recompute / offload.
    #[serde(default)]
    pub memory: MemoryHierarchy,
    /// Optional compile-time per-device working-set cap.
    #[serde(default)]
    pub budget: Option<MemoryBudget>,
}

impl Cluster {
    /// Two nodes, each with a CPU; node0 also has a GPU. Inter-node link
    /// is a stand-in for a slow fabric; intra-node is implicit and cheap.
    pub fn demo() -> Self {
        Self {
            name: "demo-mixed".into(),
            nodes: vec![
                Node::new("node0", vec![Device::cpu("node0.cpu"), Device::gpu("node0.gpu")]),
                Node::new("node1", vec![Device::cpu("node1.cpu")]),
            ],
            links: vec![
                Link {
                    src_node: "node0".into(),
                    dst_node: "node1".into(),
                    bandwidth_bps: 1e9,
                    latency_s: 2e-4,
                },
                Link {
                    src_node: "node1".into(),
                    dst_node: "node0".into(),
                    bandwidth_bps: 1e9,
                    latency_s: 2e-4,
                },
            ],
            memory: MemoryHierarchy::desktop_cpu(),
            budget: None,
        }
    }

    pub fn single_cpu() -> Self {
        Self {
            name: "local-cpu".into(),
            nodes: vec![Node::new("node0", vec![Device::cpu("node0.cpu")])],
            links: vec![],
            memory: MemoryHierarchy::desktop_cpu(),
            budget: None,
        }
    }

    /// Apply a structured compile budget (CLI `--mem` / `--mem-per-device`).
    pub fn with_budget(mut self, budget: MemoryBudget) -> Self {
        self.budget = Some(budget);
        for n in &mut self.nodes {
            for d in &mut n.devices {
                d.memory_bytes = d.memory_bytes.min(budget.per_device());
                d.regions.apply_budget(&budget);
            }
        }
        self.memory.device.bytes = self.memory.device.bytes.min(budget.per_device());
        self.memory.regions.apply_budget(&budget);
        self
    }

    pub fn devices(&self) -> impl Iterator<Item = (&Node, &Device)> {
        self.nodes.iter().flat_map(|n| n.devices.iter().map(move |d| (n, d)))
    }

    pub fn find_device(&self, id: &str) -> Option<(&Node, &Device)> {
        self.devices().find(|(_, d)| d.id == id)
    }

    pub fn node_of(&self, device_id: &str) -> Option<&Node> {
        self.find_device(device_id).map(|(n, _)| n)
    }

    pub fn link(&self, src_node: &str, dst_node: &str) -> Option<&Link> {
        self.links.iter().find(|l| l.src_node == src_node && l.dst_node == dst_node)
    }

    /// Intra-node transfer is modelled as a fast implicit link.
    pub fn transfer_seconds(&self, src_dev: &str, dst_dev: &str, bytes: u64) -> f64 {
        if src_dev == dst_dev {
            return 0.0;
        }
        let src_node = self.node_of(src_dev).map(|n| n.id.as_str()).unwrap_or("");
        let dst_node = self.node_of(dst_dev).map(|n| n.id.as_str()).unwrap_or("");
        if src_node == dst_node {
            return 5e-6 + (bytes as f64) / 64e9;
        }
        if let Some(link) = self.link(src_node, dst_node) {
            return link.transfer_seconds(bytes);
        }
        // Unknown pair: conservative WAN-like cost.
        1e-3 + (bytes as f64) / 1e8
    }

    pub fn default_device(&self) -> &Device {
        &self.nodes[0].devices[0]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn demo_cluster_has_cpu_and_gpu() {
        let c = Cluster::demo();
        assert!(c.find_device("node0.gpu").is_some());
        assert!(c.find_device("node1.cpu").is_some());
        assert!(c.transfer_seconds("node0.cpu", "node1.cpu", 1024) > 0.0);
    }
}
