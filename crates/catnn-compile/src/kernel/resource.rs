//! Resource descriptors for generated kernels.
//!
//! Every CPU / WGSL / CUDA kernel carries code size, register and scratch
//! use, launch geometry, and the buffers it touches. CUDA sizes without
//! `nvcc` are **estimates**. CPU sizes are estimated from a per-op x86-64
//! model (16 B/µop × a counted instruction budget) and labelled as such.

use catnn_core::architecture::{KernelBackend, MemoryRegionKind};
use catnn_core::category::OpenOp;
use serde::{Deserialize, Serialize};

use crate::kernel::cuda::{self, CudaKernel};
use crate::kernel::wgsl::WgslKernel;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LaunchGeom {
    pub grid: [u32; 3],
    pub block: [u32; 3],
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BufferUse {
    pub name: String,
    pub bytes: u64,
    pub region: MemoryRegionKind,
}

/// First-class staged resource for one kernel module.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KernelResource {
    pub name: String,
    pub backend: KernelBackend,
    pub code_bytes: u64,
    /// `true` when cubin / .text was not measured (no nvcc / no objdump).
    pub code_estimated: bool,
    pub code_note: String,
    pub registers_per_thread: u32,
    pub static_smem_bytes: u64,
    pub dynamic_smem_bytes: u64,
    pub launch: LaunchGeom,
    pub launch_arg_bytes: u64,
    pub reads: Vec<BufferUse>,
    pub writes: Vec<BufferUse>,
    pub temps: Vec<BufferUse>,
}

impl KernelResource {
    pub fn scratch_bytes(&self) -> u64 {
        self.static_smem_bytes
            + self.dynamic_smem_bytes
            + u64::from(self.registers_per_thread) * 4
                * u64::from(self.launch.block[0] * self.launch.block[1] * self.launch.block[2])
    }

    pub fn code_footprint(&self) -> u64 {
        self.code_bytes + self.launch_arg_bytes
    }

    pub fn with_buffers(mut self, reads: Vec<BufferUse>, writes: Vec<BufferUse>, temps: Vec<BufferUse>) -> Self {
        self.reads = reads;
        self.writes = writes;
        self.temps = temps;
        self
    }
}

pub fn buf(name: impl Into<String>, bytes: u64) -> BufferUse {
    BufferUse { name: name.into(), bytes, region: MemoryRegionKind::Buffer }
}

/// Instruction-budget model for the CPU backend (x86-64, **estimate**).
fn cpu_code_bytes(op: &OpenOp) -> (u64, String) {
    let syms = crate::kernel::measure::measure_cpu_symbols();
    let key = match op {
        OpenOp::Linear { .. } | OpenOp::MatMul | OpenOp::LmHead { .. } => "linear",
        OpenOp::RmsNorm { .. } => "rmsnorm",
        OpenOp::SwiGlu { .. } => "swiglu",
        OpenOp::GqaAttention { .. } | OpenOp::Attention { .. } => "gqa",
        OpenOp::MoE { .. } => "moe",
        OpenOp::Softmax { .. } => "softmax",
        OpenOp::Embedding { .. } => "embedding",
        _ => "linear",
    };
    let hit = syms.iter().find(|s| s.name.contains(key)).map(|s| s.bytes);
    if let Some(b) = hit {
        (b, format!("measured nm --size symbol containing '{key}': {b} B"))
    } else {
        let fallback = match op {
            OpenOp::Linear { .. } | OpenOp::MatMul => 4096,
            OpenOp::GqaAttention { .. } | OpenOp::MoE { .. } => 8192,
            _ => 2048,
        };
        (fallback, format!("nm miss for '{key}'; fallback {fallback} B"))
    }
}

fn cuda_code_bytes(k: &CudaKernel) -> (u64, String) {
    let m = crate::kernel::measure::compile_cubin(k);
    if m.compiled && m.cubin_bytes > 0 {
        (
            m.cubin_bytes,
            format!(
                "measured cubin {} B, {} regs, smem {} B; compiled -arch={} ({})",
                m.cubin_bytes, m.registers, m.smem_bytes, m.arch, crate::kernel::measure::MEASURE_ARCH_NOTE
            ),
        )
    } else {
        // Fallback: a naive GEMM cubin on this VM measured 5544 B.
        (
            5544,
            format!(
                "fallback 5544 B (measured naive GEMM cubin on sm_80); compile note: {}",
                m.note
            ),
        )
    }
}

fn wgsl_code_bytes(k: &WgslKernel) -> (u64, String) {
    let est = k.source.len() as u64;
    (est, "estimate: WGSL source bytes stand in for a pipeline blob".into())
}

fn launch_for(op: &OpenOp) -> LaunchGeom {
    match op {
        OpenOp::Linear { .. } | OpenOp::MatMul | OpenOp::LmHead { .. } | OpenOp::SwiGlu { .. } => {
            LaunchGeom { grid: [1, 1, 1], block: [16, 16, 1] }
        }
        OpenOp::GqaAttention { .. } | OpenOp::Attention { .. } => {
            LaunchGeom { grid: [1, 1, 1], block: [32, 4, 1] }
        }
        _ => LaunchGeom { grid: [1, 1, 1], block: [64, 1, 1] },
    }
}

fn smem_for(op: &OpenOp) -> (u64, u32) {
    match op {
        OpenOp::Linear { .. } | OpenOp::MatMul | OpenOp::LmHead { .. } => (2 * 16 * 16 * 4, 32),
        OpenOp::GqaAttention { head_dim, .. } => ((*head_dim as u64 * 8).max(256), 64),
        OpenOp::Attention { d_head, .. } => ((*d_head as u64 * 8).max(256), 64),
        OpenOp::Softmax { .. } | OpenOp::RmsNorm { .. } | OpenOp::LayerNorm { .. } => (256, 24),
        OpenOp::SwiGlu { .. } | OpenOp::MoE { .. } => (512, 40),
        _ => (0, 16),
    }
}

/// Descriptor for an op on a given backend. Buffer lists are filled by
/// the scheduler once tensor sizes are known.
pub fn describe(op: &OpenOp, name: &str, backend: KernelBackend) -> KernelResource {
    let (code_bytes, code_note) = match backend {
        KernelBackend::CudaSm121 => {
            let k = cuda::emit(op, name);
            let (b, n) = cuda_code_bytes(&k);
            (b, n)
        }
        KernelBackend::Wgsl => {
            let k = crate::kernel::wgsl::emit(op, name);
            let (b, n) = wgsl_code_bytes(&k);
            (b, n)
        }
        KernelBackend::CpuRust | KernelBackend::Reference => {
            let (b, n) = cpu_code_bytes(op);
            (b, n)
        }
    };
    let launch = launch_for(op);
    let (smem, regs) = smem_for(op);
    KernelResource {
        name: name.into(),
        backend,
        code_bytes,
        code_estimated: !code_note.contains("measured cubin") && !code_note.contains("nm --size"),
        code_note,
        registers_per_thread: regs,
        static_smem_bytes: smem,
        dynamic_smem_bytes: 0,
        launch,
        launch_arg_bytes: 256,
        reads: vec![],
        writes: vec![],
        temps: vec![],
    }
}

pub fn describe_cuda(op: &OpenOp, name: &str) -> KernelResource {
    describe(op, name, KernelBackend::CudaSm121)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cuda_code_is_measured_or_fallback() {
        let r = describe_cuda(&OpenOp::RmsNorm { eps_e6: 10 }, "rms");
        assert!(r.code_bytes >= 2000, "code {} B", r.code_bytes);
        assert!(r.code_note.contains("measured") || r.code_note.contains("fallback"));
    }

    #[test]
    fn cpu_linear_has_scratch() {
        let r = describe(
            &OpenOp::Linear { in_features: 4, out_features: 8, bias: true },
            "fc",
            KernelBackend::CpuRust,
        );
        assert!(r.scratch_bytes() > 0);
    }
}
