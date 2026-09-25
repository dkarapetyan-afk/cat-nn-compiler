//! Memory as an object of `H` and a compile parameter.
//!
//! Each device has three **regions** with independent capacities and
//! bandwidths:
//!
//! - **code** — resident kernel modules / cubins, instruction cache,
//!   constant or parameter banks, per-launch argument space;
//! - **buffer** — global / device / unified memory for tensors;
//! - **scratch** — SMEM / shared memory, registers, L1/L2.
//!
//! The hierarchy (device, host, NVMe, remote) still describes *where*
//! buffer bytes can live. The budget is a structured
//! [`MemoryBudget`] (`code` / `buffer` / `scratch`), not a single number.
//! Realizations over a smaller memory object reindex to more kernel
//! reloads, rematerialization, or offload.

use serde::{Deserialize, Serialize};

use crate::error::{CatError, CatResult};

/// A distinct address space in the *hierarchy* (where bytes live).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryKind {
    /// Device HBM or Spark unified LPDDR5x.
    Device,
    /// Host DRAM (may be the *same* physical memory as Device on GB10).
    Host,
    /// Local NVMe / disk.
    Nvme,
    /// Another node's memory, reached over the cluster fabric.
    Remote,
}

impl MemoryKind {
    pub fn as_str(self) -> &'static str {
        match self {
            MemoryKind::Device => "device",
            MemoryKind::Host => "host",
            MemoryKind::Nvme => "nvme",
            MemoryKind::Remote => "remote",
        }
    }
}

/// A named region *inside* a device (what the bytes are for).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryRegionKind {
    Code,
    Buffer,
    Scratch,
}

impl MemoryRegionKind {
    pub fn as_str(self) -> &'static str {
        match self {
            MemoryRegionKind::Code => "code",
            MemoryRegionKind::Buffer => "buffer",
            MemoryRegionKind::Scratch => "scratch",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MemoryTier {
    pub kind: MemoryKind,
    pub bytes: u64,
    pub bandwidth_bps: f64,
    pub latency_s: f64,
}

impl MemoryTier {
    pub fn transfer_seconds(&self, bytes: u64) -> f64 {
        self.latency_s + (bytes as f64) / self.bandwidth_bps.max(1.0)
    }
}

/// One device-local region (code, buffer, or scratch).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MemoryRegion {
    pub kind: MemoryRegionKind,
    pub bytes: u64,
    pub bandwidth_bps: f64,
    pub latency_s: f64,
    /// `"cited"` or a short assumption note.
    pub provenance: String,
}

impl MemoryRegion {
    pub fn transfer_seconds(&self, bytes: u64) -> f64 {
        self.latency_s + (bytes as f64) / self.bandwidth_bps.max(1.0)
    }
}

/// Per-device split of code / buffer / scratch. This is part of the
/// object of `H` and indexes `R`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DeviceMemory {
    pub code: MemoryRegion,
    pub buffer: MemoryRegion,
    pub scratch: MemoryRegion,
}

impl DeviceMemory {
    /// Desktop CPU. Code size is a process I-cache / .text working set
    /// (**assumption**, not measured on this box). Buffer is the heap.
    /// Scratch is the per-thread stack + SIMD register file (**assumption**).
    pub fn desktop_cpu() -> Self {
        Self {
            code: MemoryRegion {
                kind: MemoryRegionKind::Code,
                bytes: 16 << 20,
                bandwidth_bps: 50e9,
                latency_s: 5e-9,
                provenance: "assumption: 16 MiB resident .text / I-cache working set \
                     (not measured; typical x86-64 process)"
                    .into(),
            },
            buffer: MemoryRegion {
                kind: MemoryRegionKind::Buffer,
                bytes: 8 << 30,
                bandwidth_bps: 50e9,
                latency_s: 1e-7,
                provenance: "assumption: 8 GiB heap for the CPU device object".into(),
            },
            scratch: MemoryRegion {
                kind: MemoryRegionKind::Scratch,
                bytes: 1 << 20,
                bandwidth_bps: 200e9,
                latency_s: 1e-9,
                provenance: "assumption: 1 MiB stack + SIMD register file per worker".into(),
            },
        }
    }

    /// DGX Spark / GB10 regions.
    ///
    /// *Buffer* is cited (128 GiB unified LPDDR5x @ 273 GB/s). *Code* and
    /// *scratch* are **stated assumptions**: NVIDIA does not publish GB10
    /// I-cache, cubin-residency, SMEM/block, or L2 sizes.
    pub fn dgx_spark() -> Self {
        Self {
            code: MemoryRegion {
                kind: MemoryRegionKind::Code,
                bytes: 64 << 20,
                bandwidth_bps: 273e9,
                latency_s: 1e-6,
                provenance: "assumption: 64 MiB resident module/cubin working set \
                     fetched from unified LPDDR5x @ 273 GB/s; I-cache size \
                     unpublished for GB10. Launch-arg bank folded into this \
                     region (4 KiB/launch typical)."
                    .into(),
            },
            buffer: MemoryRegion {
                kind: MemoryRegionKind::Buffer,
                bytes: 128 * (1u64 << 30),
                bandwidth_bps: 273e9,
                latency_s: 1e-7,
                provenance: "cited: NVIDIA DGX Spark product page — 128 GB unified \
                     LPDDR5x, 273 GB/s"
                    .into(),
            },
            scratch: MemoryRegion {
                kind: MemoryRegionKind::Scratch,
                bytes: 228 * 1024,
                bandwidth_bps: 8e12,
                latency_s: 1e-9,
                provenance: "assumption: 228 KiB SMEM/block from datacenter \
                     Blackwell B200 public figures; GB10 SMEM unpublished. \
                     Registers (255/thread) and L2 (assumed 16 MiB) are \
                     modelled as pressure against this cap, not a separate \
                     region."
                    .into(),
            },
        }
    }

    pub fn region(&self, kind: MemoryRegionKind) -> &MemoryRegion {
        match kind {
            MemoryRegionKind::Code => &self.code,
            MemoryRegionKind::Buffer => &self.buffer,
            MemoryRegionKind::Scratch => &self.scratch,
        }
    }

    pub fn apply_budget(&mut self, b: &MemoryBudget) {
        if let Some(c) = b.code {
            self.code.bytes = self.code.bytes.min(c);
        }
        self.buffer.bytes = self.buffer.bytes.min(b.buffer);
        if let Some(s) = b.scratch {
            self.scratch.bytes = self.scratch.bytes.min(s);
        }
    }
}

impl Default for DeviceMemory {
    fn default() -> Self {
        Self::desktop_cpu()
    }
}

/// Hierarchy of *where* buffer bytes can live (device / host / NVMe / remote).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MemoryHierarchy {
    pub device: MemoryTier,
    pub host: MemoryTier,
    #[serde(default)]
    pub nvme: Option<MemoryTier>,
    #[serde(default)]
    pub remote: Option<MemoryTier>,
    /// When true, `host` is the same physical pool as `device` (GB10).
    #[serde(default)]
    pub host_aliases_device: bool,
    /// Default per-device region split (overridden per [`crate::architecture::hardware::Device`]).
    #[serde(default)]
    pub regions: DeviceMemory,
}

impl MemoryHierarchy {
    pub fn desktop_cpu() -> Self {
        Self {
            device: MemoryTier {
                kind: MemoryKind::Device,
                bytes: 8 << 30,
                bandwidth_bps: 50e9,
                latency_s: 1e-7,
            },
            host: MemoryTier {
                kind: MemoryKind::Host,
                bytes: 16 << 30,
                bandwidth_bps: 25e9,
                latency_s: 2e-7,
            },
            nvme: None,
            remote: None,
            host_aliases_device: false,
            regions: DeviceMemory::desktop_cpu(),
        }
    }

    pub fn dgx_spark() -> Self {
        Self {
            device: MemoryTier {
                kind: MemoryKind::Device,
                bytes: 128 * (1u64 << 30),
                bandwidth_bps: 273e9,
                latency_s: 1e-7,
            },
            host: MemoryTier {
                kind: MemoryKind::Host,
                bytes: 0,
                bandwidth_bps: 273e9,
                latency_s: 1e-7,
            },
            nvme: Some(MemoryTier {
                kind: MemoryKind::Nvme,
                bytes: 1024 * (1u64 << 30),
                bandwidth_bps: 3.5e9,
                latency_s: 50e-6,
            }),
            remote: Some(MemoryTier {
                kind: MemoryKind::Remote,
                bytes: 128 * (1u64 << 30),
                bandwidth_bps: 200e9 / 8.0,
                latency_s: 2e-6,
            }),
            host_aliases_device: true,
            regions: DeviceMemory::dgx_spark(),
        }
    }

    pub fn offload_capacity(&self) -> u64 {
        let host = if self.host_aliases_device { 0 } else { self.host.bytes };
        host + self.nvme.as_ref().map(|t| t.bytes).unwrap_or(0)
    }

    pub fn offload_tier(&self) -> Option<&MemoryTier> {
        if !self.host_aliases_device && self.host.bytes > 0 {
            Some(&self.host)
        } else {
            self.nvme.as_ref()
        }
    }
}

impl Default for MemoryHierarchy {
    fn default() -> Self {
        Self::desktop_cpu()
    }
}

/// Structured per-device budget — a compile parameter of `H`.
///
/// `buffer` is required. `code` / `scratch` default to the device object's
/// published (or assumed) region size when omitted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryBudget {
    pub code: Option<u64>,
    pub buffer: u64,
    pub scratch: Option<u64>,
}

impl MemoryBudget {
    pub fn buffer_only(buffer: u64) -> Self {
        Self { code: None, buffer, scratch: None }
    }

    /// Alias used by older call sites: the buffer-region cap.
    pub fn per_device(&self) -> u64 {
        self.buffer
    }

    /// Parse `64GiB` **or** `code=4MiB,buffer=64GiB,scratch=128KiB`.
    pub fn parse(s: &str) -> CatResult<Self> {
        let t = s.trim();
        if t.contains('=') || t.contains(',') {
            let mut code = None;
            let mut buffer = None;
            let mut scratch = None;
            for part in t.split(',') {
                let part = part.trim();
                if part.is_empty() {
                    continue;
                }
                let (k, v) = part
                    .split_once('=')
                    .ok_or_else(|| CatError::Parse(format!("expected key=value in {part}")))?;
                let bytes = parse_bytes(v.trim())?;
                match k.trim().to_ascii_lowercase().as_str() {
                    "code" | "instruction" | "text" => code = Some(bytes),
                    "buffer" | "data" | "hbm" | "device" => buffer = Some(bytes),
                    "scratch" | "smem" | "shared" => scratch = Some(bytes),
                    other => return Err(CatError::Parse(format!("unknown memory region {other}"))),
                }
            }
            let buffer = buffer
                .ok_or_else(|| CatError::Parse("structured --mem needs a buffer=… entry".into()))?;
            Ok(Self { code, buffer, scratch })
        } else {
            Ok(Self::buffer_only(parse_bytes(t)?))
        }
    }
}

pub fn parse_bytes(s: &str) -> CatResult<u64> {
    let t = s.trim().replace('_', "");
    let lower = t.to_ascii_lowercase();
    let (num, mul) = if let Some(rest) = lower.strip_suffix("gib") {
        (rest, 1u64 << 30)
    } else if let Some(rest) = lower.strip_suffix("gb") {
        (rest, 1_000_000_000)
    } else if let Some(rest) = lower.strip_suffix("mib") {
        (rest, 1u64 << 20)
    } else if let Some(rest) = lower.strip_suffix("mb") {
        (rest, 1_000_000)
    } else if let Some(rest) = lower.strip_suffix("kib") {
        (rest, 1u64 << 10)
    } else if let Some(rest) = lower.strip_suffix("kb") {
        (rest, 1_000)
    } else if let Some(rest) = lower.strip_suffix('g') {
        (rest, 1u64 << 30)
    } else if let Some(rest) = lower.strip_suffix('m') {
        (rest, 1u64 << 20)
    } else {
        (lower.as_str(), 1)
    };
    let v: f64 = num.trim().parse().map_err(|_| CatError::Parse(format!("bad memory size {s}")))?;
    if v < 0.0 {
        return Err(CatError::Parse("negative memory".into()));
    }
    Ok((v * mul as f64) as u64)
}

/// Simulated peak occupancy of the three regions at one instant.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RegionPeaks {
    pub code: u64,
    pub buffer: u64,
    pub scratch: u64,
}

impl RegionPeaks {
    pub fn exceeds(&self, cap: &DeviceMemory) -> Option<MemoryRegionKind> {
        if self.code > cap.code.bytes {
            return Some(MemoryRegionKind::Code);
        }
        if self.buffer > cap.buffer.bytes {
            return Some(MemoryRegionKind::Buffer);
        }
        if self.scratch > cap.scratch.bytes {
            return Some(MemoryRegionKind::Scratch);
        }
        None
    }

    pub fn max_with(&self, other: &Self) -> Self {
        Self {
            code: self.code.max(other.code),
            buffer: self.buffer.max(other.buffer),
            scratch: self.scratch.max(other.scratch),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_gib() {
        assert_eq!(parse_bytes("64GiB").unwrap(), 64 * (1u64 << 30));
        assert_eq!(parse_bytes("8MiB").unwrap(), 8 << 20);
        assert_eq!(parse_bytes("1024").unwrap(), 1024);
    }

    #[test]
    fn parse_structured_budget() {
        let b = MemoryBudget::parse("code=4MiB,buffer=64GiB,scratch=128KiB").unwrap();
        assert_eq!(b.code, Some(4 << 20));
        assert_eq!(b.buffer, 64 * (1u64 << 30));
        assert_eq!(b.scratch, Some(128 << 10));
    }

    #[test]
    fn spark_host_does_not_add_capacity() {
        let h = MemoryHierarchy::dgx_spark();
        assert!(h.host_aliases_device);
        assert_eq!(h.offload_capacity(), 1024 * (1u64 << 30));
        assert_eq!(h.regions.buffer.bytes, 128 * (1u64 << 30));
        assert!(h.regions.buffer.provenance.contains("cited"));
        assert!(h.regions.code.provenance.contains("assumption"));
        assert!(h.regions.scratch.provenance.contains("assumption"));
    }
}
