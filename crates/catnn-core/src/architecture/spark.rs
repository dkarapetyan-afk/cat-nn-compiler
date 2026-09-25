//! Hardware object: NVIDIA DGX Spark (GB10 Grace Blackwell).
//!
//! Specs cited from NVIDIA product pages and user docs (retrieved 2026-09-25):
//!
//! - Product page: <https://www.nvidia.com/en-us/products/workstations/dgx-spark/>
//!   GB10 Grace Blackwell Superchip; 128 GB coherent unified LPDDR5x;
//!   256-bit interface; **273 GB/s** memory bandwidth; ConnectX-7 **200 Gb/s**;
//!   “up to 1 petaFLOP of AI performance at FP4”; 20-core Arm
//!   (10× Cortex-X925 + 10× Cortex-A725); NVLink-C2C coherent CPU↔GPU.
//! - Clustering: <https://docs.nvidia.com/dgx/dgx-spark/spark-clustering.html>
//!   two QSFP ports, each up to 200 Gb/s via ConnectX-7.
//! - Cluster Assistant: <https://docs.nvidia.com/sync/latest/cluster-assistant.html>
//!   **officially supports 2–4 Sparks**. Direct cables: 2–3 nodes. Switch: 2–4.
//!   Eight-node switch topologies must be configured manually (not Cluster Assistant).
//! - CUDA compute capability **12.1 / `sm_121`**: NVIDIA forum confirmation
//!   (aniculescu, 2025-09-25) and SM121 hardware notes. Distinct from
//!   datacenter Blackwell `sm_100` (B200). No TMEM / tcgen05 on GB10.

use serde::{Deserialize, Serialize};

use crate::architecture::hardware::{Cluster, Device, DeviceKind, Link, Node};
use crate::error::{CatError, CatResult};

/// One DGX Spark, as a device in `H`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DgxSparkSpec {
    pub memory_bytes: u64,
    pub memory_bandwidth_bps: f64,
    pub connectx7_bits_per_sec: f64,
    /// NVIDIA “up to 1 PFLOP FP4”.
    pub fp4_flops: f64,
    pub sm_arch: String,
    pub compute_capability_major: u32,
    pub compute_capability_minor: u32,
    pub cpu_cores: u32,
    pub reserved_os_bytes: u64,
    pub reserved_workspace_bytes: u64,
}

impl DgxSparkSpec {
    pub fn official() -> Self {
        Self {
            memory_bytes: 128 * (1u64 << 30),
            memory_bandwidth_bps: 273.0 * 1e9,
            connectx7_bits_per_sec: 200.0 * 1e9,
            fp4_flops: 1.0e15,
            sm_arch: "sm_121".into(),
            compute_capability_major: 12,
            compute_capability_minor: 1,
            cpu_cores: 20,
            // Conservative: OS + CUDA context on unified memory.
            reserved_os_bytes: 20 * (1u64 << 30),
            reserved_workspace_bytes: 4 * (1u64 << 30),
        }
    }

    pub fn usable_bytes(&self) -> u64 {
        self.memory_bytes.saturating_sub(self.reserved_os_bytes + self.reserved_workspace_bytes)
    }

    pub fn connectx7_bytes_per_sec(&self) -> f64 {
        self.connectx7_bits_per_sec / 8.0
    }

    /// Peak FLOP/s scaled from the 1 PFLOP FP4 claim.
    /// BF16 ≈ 1/4, FP8 ≈ 1/2 of FP4 tensor throughput (not measured).
    pub fn peak_flops(&self, precision: Precision) -> f64 {
        match precision {
            Precision::Fp4 => self.fp4_flops,
            Precision::Fp8 => self.fp4_flops * 0.5,
            Precision::Bf16 => self.fp4_flops * 0.25,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Precision {
    Bf16,
    Fp8,
    Fp4,
}

impl Precision {
    pub fn as_str(self) -> &'static str {
        match self {
            Precision::Bf16 => "bf16",
            Precision::Fp8 => "fp8",
            Precision::Fp4 => "fp4",
        }
    }

    pub fn parse(s: &str) -> CatResult<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "bf16" | "bfloat16" => Ok(Precision::Bf16),
            "fp8" => Ok(Precision::Fp8),
            "fp4" | "nvfp4" => Ok(Precision::Fp4),
            other => Err(CatError::Parse(format!("unknown precision {other}"))),
        }
    }

    pub fn bits(self) -> u32 {
        match self {
            Precision::Bf16 => 16,
            Precision::Fp8 => 8,
            Precision::Fp4 => 4,
        }
    }

    pub fn packed_bytes(self, numel: u64) -> u64 {
        (numel * self.bits() as u64 + 7) / 8
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SparkTopology {
    /// Pairwise QSFP DAC. Officially 2–3 nodes.
    Direct,
    /// Each Spark ↔ switch at 200 Gb/s. Officially 2–4 nodes.
    Switch,
}

impl SparkTopology {
    pub fn officially_supported(self, n: usize) -> bool {
        match self {
            SparkTopology::Direct => (2..=3).contains(&n) || n == 1,
            SparkTopology::Switch => (2..=4).contains(&n) || n == 1,
        }
    }
}

/// Build a cluster of `n` Sparks. `n = 1` is a single desktop.
///
/// Official Cluster Assistant support is 2–4 nodes (see module docs).
/// `n = 8` is accepted as a *manual* topology and flagged unsupported.
pub fn spark_cluster(n: usize, topology: SparkTopology) -> CatResult<Cluster> {
    if n == 0 || n > 16 {
        return Err(CatError::Fibration(format!("Spark N={n} out of range 1..=16")));
    }
    let spec = DgxSparkSpec::official();
    let mut nodes = Vec::new();
    for i in 0..n {
        nodes.push(Node::new(
            format!("spark{i}"),
            vec![Device {
                id: format!("spark{i}.gb10"),
                kind: DeviceKind::Gpu,
                memory_bytes: spec.memory_bytes,
                gflops: spec.peak_flops(Precision::Fp4) / 1e9,
                regions: crate::architecture::memory::DeviceMemory::dgx_spark(),
            }],
        ));
    }
    let mut links = Vec::new();
    let bps = spec.connectx7_bytes_per_sec();
    // ConnectX-7 RoCE latency: ~1–2 µs typical; we use 2 µs.
    let latency_s = 2.0e-6;
    if n > 1 {
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }
                // Direct: full mesh (cables). Switch: also full logical mesh
                // at 200 Gb/s per pair (oversubscribed on a small switch,
                // which we do not model further).
                let _ = topology;
                links.push(Link {
                    src_node: format!("spark{i}"),
                    dst_node: format!("spark{j}"),
                    bandwidth_bps: bps,
                    latency_s,
                });
            }
        }
    }
    let tag = if topology.officially_supported(n) { "official" } else { "manual-unsupported" };
    let mut memory = crate::architecture::memory::MemoryHierarchy::dgx_spark();
    if n > 1 {
        if let Some(remote) = memory.remote.as_mut() {
            remote.bytes = spec.memory_bytes.saturating_mul((n - 1) as u64);
        }
    }
    Ok(Cluster { name: format!("dgx-spark-N{n}-{tag}"), nodes, links, memory, budget: None })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spec_matches_cited_numbers() {
        let s = DgxSparkSpec::official();
        assert_eq!(s.memory_bytes, 128 * (1u64 << 30));
        assert!((s.memory_bandwidth_bps - 273e9).abs() < 1.0);
        assert_eq!(s.sm_arch, "sm_121");
        assert_eq!((s.compute_capability_major, s.compute_capability_minor), (12, 1));
    }

    #[test]
    fn official_support_is_at_most_four() {
        assert!(SparkTopology::Switch.officially_supported(4));
        assert!(!SparkTopology::Switch.officially_supported(8));
        assert!(SparkTopology::Direct.officially_supported(2));
        assert!(!SparkTopology::Direct.officially_supported(4));
    }

    #[test]
    fn cluster_n4_has_mesh_links() {
        let c = spark_cluster(4, SparkTopology::Switch).unwrap();
        assert_eq!(c.nodes.len(), 4);
        assert_eq!(c.links.len(), 4 * 3);
    }
}
