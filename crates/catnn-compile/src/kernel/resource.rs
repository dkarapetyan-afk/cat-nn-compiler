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
    let uops = match op {
        OpenOp::Identity | OpenOp::Reshape { .. } | OpenOp::Transpose { .. } => 32,
        OpenOp::Relu | OpenOp::Add | OpenOp::Mul | OpenOp::ResidualAdd => 64,
        OpenOp::Gelu | OpenOp::Softmax { .. } | OpenOp::RmsNorm { .. } | OpenOp::LayerNorm { .. } => {
            256
        }
        OpenOp::Linear { .. } | OpenOp::MatMul | OpenOp::LmHead { .. } | OpenOp::Embedding { .. } => {
            512
        }
        OpenOp::SwiGlu { .. } | OpenOp::Router { .. } => 768,
        OpenOp::Attention { .. } | OpenOp::GqaAttention { .. } | OpenOp::MoE { .. } => 2048,
    };
    let bytes = uops * 16;
    (
        bytes,
        format!(
            "estimate: {uops} µops × 16 B (x86-64); not objdump-measured"
        ),
    )
}

fn cuda_code_bytes(k: &CudaKernel) -> (u64, String) {
    // Naive cubin ≈ 1/4 of the CUDA C source for these tiny kernels.
    let est = (k.source.len() as u64 / 4).max(256);
    (
        est,
        "estimate: cubin ≈ source/4 (nvcc not run; not a measured .cubin)".into(),
    )
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
    let (code_bytes, code_note, code_estimated) = match backend {
        KernelBackend::CudaSm121 => {
            let k = cuda::emit(op, name);
            let (b, n) = cuda_code_bytes(&k);
            (b, n, true)
        }
        KernelBackend::Wgsl => {
            let k = crate::kernel::wgsl::emit(op, name);
            let (b, n) = wgsl_code_bytes(&k);
            (b, n, true)
        }
        KernelBackend::CpuRust | KernelBackend::Reference => {
            let (b, n) = cpu_code_bytes(op);
            (b, n, true)
        }
    };
    let launch = launch_for(op);
    let (smem, regs) = smem_for(op);
    KernelResource {
        name: name.into(),
        backend,
        code_bytes,
        code_estimated,
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
    fn cuda_code_is_labelled_estimate() {
        let r = describe_cuda(&OpenOp::RmsNorm { eps_e6: 10 }, "rms");
        assert!(r.code_estimated);
        assert!(r.code_bytes > 0);
        assert!(r.code_note.contains("estimate"));
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
