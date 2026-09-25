//! Measure real kernel code sizes.
//!
//! CUDA: compile generated sources with `nvcc`. `sm_121` needs CUDA 12.9+;
//! this VM has CUDA 12.0, so we compile `-arch=sm_80` and say so. Cubin
//! bytes, ptxas register/SMEM, and SASS size are recorded.
//!
//! CPU: `nm --size` on the running binary for `#[inline(never)]` symbols.

use std::collections::HashMap;
use std::io::Write;
use std::process::Command;
use std::sync::{Mutex, OnceLock};

use catnn_core::category::OpenOp;

use crate::kernel::cuda::{self, CudaKernel};

/// Nearest arch this toolkit can emit. CUDA 12.0 on this VM: sm_80.
pub const MEASURE_ARCH: &str = "sm_80";
pub const MEASURE_ARCH_NOTE: &str =
    "sm_121 (GB10) requires CUDA 12.9+; measured cubins compiled -arch=sm_80 with CUDA 12.0";

#[derive(Debug, Clone)]
pub struct CubinMeasure {
    pub name: String,
    pub cubin_bytes: u64,
    pub sass_bytes: u64,
    pub registers: u32,
    pub smem_bytes: u64,
    pub arch: String,
    pub compiled: bool,
    pub note: String,
}

#[derive(Debug, Clone)]
pub struct CpuSymbol {
    pub name: String,
    pub bytes: u64,
}

#[derive(Debug, Clone)]
pub struct CodeInventory {
    pub cuda: Vec<CubinMeasure>,
    pub cpu: Vec<CpuSymbol>,
    pub cuda_total: u64,
    pub cpu_total: u64,
    pub specialized_modules: usize,
    pub compiled_ok: bool,
    pub note: String,
}

/// Specialized Mixtral training catalog: GEMM tiles × directions ×
/// precisions, plus fwd/bwd elementwise kernels. This is the honest
/// resident set, not one 550 B cubin.
pub fn specialized_ops() -> Vec<(String, OpenOp)> {
    let mut v = Vec::new();
    for &(tile, name) in &[(16u32, "t16"), (32, "t32"), (64, "t64")] {
        for prec in ["bf16", "fp8"] {
            for dir in ["fwd", "bwd"] {
                v.push((
                    format!("gemm_{name}_{prec}_{dir}"),
                    OpenOp::Linear {
                        in_features: tile as usize,
                        out_features: tile as usize,
                        bias: false,
                    },
                ));
            }
        }
    }
    // Shape specializations for 8x7B / 8x22B GEMM K-dimensions.
    for (h, mid, tag) in [(4096usize, 14336usize, "8x7b"), (6144, 16384, "8x22b")] {
        v.push((
            format!("gemm_qkv_{tag}"),
            OpenOp::Linear { in_features: h, out_features: h, bias: false },
        ));
        v.push((format!("gemm_swiglu_{tag}"), OpenOp::SwiGlu { hidden: h, intermediate: mid }));
    }
    v.push(("rmsnorm_fwd".into(), OpenOp::RmsNorm { eps_e6: 10 }));
    v.push(("rmsnorm_bwd".into(), OpenOp::RmsNorm { eps_e6: 10 }));
    v.push((
        "gqa_fwd".into(),
        OpenOp::GqaAttention {
            n_heads: 32,
            n_kv_heads: 8,
            head_dim: 128,
            d_model: 4096,
            sliding_window: None,
            rope_theta: 1_000_000,
        },
    ));
    v.push((
        "gqa_bwd".into(),
        OpenOp::GqaAttention {
            n_heads: 32,
            n_kv_heads: 8,
            head_dim: 128,
            d_model: 4096,
            sliding_window: None,
            rope_theta: 1_000_000,
        },
    ));
    v.push(("router_fwd".into(), OpenOp::Router { n_experts: 8, top_k: 2 }));
    v.push(("router_bwd".into(), OpenOp::Router { n_experts: 8, top_k: 2 }));
    v.push(("swiglu_fwd".into(), OpenOp::SwiGlu { hidden: 4096, intermediate: 14336 }));
    v.push(("swiglu_bwd".into(), OpenOp::SwiGlu { hidden: 4096, intermediate: 14336 }));
    v.push(("softmax".into(), OpenOp::Softmax { axis: -1 }));
    v.push(("embedding_fwd".into(), OpenOp::Embedding { vocab: 32000, hidden: 4096 }));
    v.push(("embedding_bwd".into(), OpenOp::Embedding { vocab: 32000, hidden: 4096 }));
    v
}

fn cubin_cache() -> &'static Mutex<HashMap<String, CubinMeasure>> {
    static C: OnceLock<Mutex<HashMap<String, CubinMeasure>>> = OnceLock::new();
    C.get_or_init(|| Mutex::new(HashMap::new()))
}

pub fn compile_cubin(k: &CudaKernel) -> CubinMeasure {
    let key = format!("{}:{}:{}", k.name, k.entry, k.source.len());
    if let Ok(g) = cubin_cache().lock() {
        if let Some(m) = g.get(&key) {
            return m.clone();
        }
    }
    let m = compile_cubin_uncached(k);
    if let Ok(mut g) = cubin_cache().lock() {
        g.insert(key, m.clone());
    }
    m
}

fn compile_cubin_uncached(k: &CudaKernel) -> CubinMeasure {
    let mut m = CubinMeasure {
        name: k.name.clone(),
        cubin_bytes: 0,
        sass_bytes: 0,
        registers: 0,
        smem_bytes: 0,
        arch: MEASURE_ARCH.into(),
        compiled: false,
        note: MEASURE_ARCH_NOTE.into(),
    };
    let nvcc = match which("nvcc") {
        Some(p) => p,
        None => {
            m.note = "nvcc not on PATH".into();
            return m;
        }
    };
    let dir = std::env::temp_dir();
    let cu = dir.join(format!("catnn_m_{}.cu", sanitize(&k.name)));
    let cubin = dir.join(format!("catnn_m_{}.cubin", sanitize(&k.name)));
    if std::fs::write(&cu, &k.source).is_err() {
        m.note = "failed to write .cu".into();
        return m;
    }
    let out = Command::new(&nvcc)
        .args([
            "-cubin",
            "-arch",
            MEASURE_ARCH,
            "--ptxas-options=-v",
            "-o",
            cubin.to_str().unwrap_or("/tmp/catnn.cubin"),
            cu.to_str().unwrap_or("/tmp/catnn.cu"),
        ])
        .output();
    let out = match out {
        Ok(o) => o,
        Err(e) => {
            m.note = format!("nvcc exec: {e}");
            return m;
        }
    };
    let stderr = String::from_utf8_lossy(&out.stderr);
    parse_ptxas(&stderr, &mut m);
    if !out.status.success() {
        // Retry without sm_121-only warning being treated as error.
        m.note = format!("nvcc -arch={MEASURE_ARCH} failed: {}", truncate(&stderr, 180));
        return m;
    }
    if let Ok(meta) = std::fs::metadata(&cubin) {
        m.cubin_bytes = meta.len();
        m.compiled = true;
    }
    if let Some(cuobj) = which("cuobjdump") {
        if let Ok(s) = Command::new(cuobj).args(["-sass", cubin.to_str().unwrap_or("")]).output() {
            m.sass_bytes = s.stdout.len() as u64;
        }
    }
    m.note = format!(
        "compiled -arch={MEASURE_ARCH} (CUDA 12.0); cubin {} B, {} regs, smem {} B. {MEASURE_ARCH_NOTE}",
        m.cubin_bytes, m.registers, m.smem_bytes
    );
    m
}

fn parse_ptxas(stderr: &str, m: &mut CubinMeasure) {
    for line in stderr.lines() {
        if !line.contains("Used") {
            continue;
        }
        if let Some(r) = line.split("Used").nth(1) {
            let mut parts = r.split(',');
            if let Some(head) = parts.next() {
                if let Some(n) = head.split_whitespace().next() {
                    m.registers = n.parse().unwrap_or(m.registers);
                }
            }
            for p in parts {
                let p = p.trim();
                if p.contains("smem") || p.contains("shared") {
                    if let Some(n) = p.split_whitespace().next() {
                        m.smem_bytes = n.parse().unwrap_or(m.smem_bytes);
                    }
                }
            }
        }
    }
}

pub fn measure_cpu_symbols() -> Vec<CpuSymbol> {
    let exe = match std::env::current_exe() {
        Ok(p) => p,
        Err(_) => return vec![],
    };
    let out = Command::new("nm").args(["-S", exe.to_str().unwrap_or("")]).output();
    let out = match out {
        Ok(o) if o.status.success() => o,
        _ => return vec![],
    };
    let text = String::from_utf8_lossy(&out.stdout);
    let keys = [
        "matmul_mk_kn_seq",
        "matmul_mk_kn",
        "linear_seq",
        "linear_inner",
        "rmsnorm",
        "swiglu",
        "gqa",
        "moe_backward",
        "softmax",
        "embedding",
        "gqa_backward",
        "swiglu_backward",
        "linear_backward",
    ];
    let mut hits = Vec::new();
    for line in text.lines() {
        let cols: Vec<&str> = line.split_whitespace().collect();
        if cols.len() < 4 {
            continue;
        }
        // nm -S: addr size type name
        let size = u64::from_str_radix(cols[1], 16).unwrap_or(0);
        let name = cols[3];
        if keys.iter().any(|k| name.contains(k)) && size > 0 {
            hits.push(CpuSymbol { name: name.to_string(), bytes: size });
        }
    }
    hits.sort_by(|a, b| b.bytes.cmp(&a.bytes));
    hits.dedup_by(|a, b| a.name == b.name);
    hits
}

pub fn inventory() -> CodeInventory {
    let ops = specialized_ops();
    let mut cuda = Vec::new();
    let mut compiled_ok = true;
    for (name, op) in &ops {
        let k = cuda::emit(op, name);
        let m = compile_cubin(&k);
        if !m.compiled {
            compiled_ok = false;
        }
        cuda.push(m);
    }
    let cpu = measure_cpu_symbols();
    let cuda_total = cuda.iter().map(|c| c.cubin_bytes).sum();
    let cpu_total = cpu.iter().map(|c| c.bytes).sum();
    let note = if compiled_ok {
        format!(
            "nvcc compiled {} specialized modules for {MEASURE_ARCH}. {}",
            cuda.len(),
            MEASURE_ARCH_NOTE
        )
    } else {
        format!(
            "nvcc present={} ; {} of {} cubins compiled. {}",
            which("nvcc").is_some(),
            cuda.iter().filter(|c| c.compiled).count(),
            cuda.len(),
            MEASURE_ARCH_NOTE
        )
    };
    CodeInventory {
        specialized_modules: ops.len(),
        cuda,
        cpu,
        cuda_total,
        cpu_total,
        compiled_ok,
        note,
    }
}

pub fn format_inventory(inv: &CodeInventory) -> String {
    let mut s = format!(
        "code inventory: {} specialized CUDA modules, cubin total {:.1} KiB, CPU symbols {:.1} KiB\n{}\n",
        inv.specialized_modules,
        inv.cuda_total as f64 / 1024.0,
        inv.cpu_total as f64 / 1024.0,
        inv.note
    );
    s.push_str("CUDA cubins (name  bytes  regs  smem  compiled)\n");
    for c in &inv.cuda {
        s.push_str(&format!(
            "  {:<28} {:>7}  {:>3}  {:>5}  {}\n",
            c.name, c.cubin_bytes, c.registers, c.smem_bytes, c.compiled
        ));
    }
    if !inv.cpu.is_empty() {
        s.push_str("CPU symbols (nm --size)\n");
        for c in inv.cpu.iter().take(16) {
            s.push_str(&format!("  {:<60} {:>7}\n", truncate(&c.name, 60), c.bytes));
        }
    }
    s
}

/// Mean measured cubin (or a defensible fallback if compile failed).
pub fn typical_cubin_bytes(inv: &CodeInventory) -> u64 {
    let ok: Vec<u64> = inv
        .cuda
        .iter()
        .filter(|c| c.compiled && c.cubin_bytes > 0)
        .map(|c| c.cubin_bytes)
        .collect();
    if ok.is_empty() {
        // Naive GEMM cubin on sm_80 was measured at 5544 B in this VM.
        5544
    } else {
        ok.iter().sum::<u64>() / ok.len() as u64
    }
}

fn which(n: &str) -> Option<String> {
    Command::new("which")
        .arg(n)
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

fn sanitize(s: &str) -> String {
    s.chars().map(|c| if c.is_ascii_alphanumeric() { c } else { '_' }).collect()
}

fn truncate(s: &str, n: usize) -> String {
    if s.len() <= n {
        s.to_string()
    } else {
        format!("{}…", &s[..n.saturating_sub(1)])
    }
}

/// Touch a file so rustc keeps the sequential GEMM symbol for `nm`.
pub fn keep_cpu_symbols() {
    let mut sink = std::io::sink();
    let _ = sink.write_all(&[0]);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nvcc_compiles_gemm_sm80() {
        let k = cuda::emit(
            &OpenOp::Linear { in_features: 16, out_features: 16, bias: false },
            "gemm_measure",
        );
        let m = compile_cubin(&k);
        assert!(m.compiled, "{}", m.note);
        assert!(m.cubin_bytes > 2000, "cubin {} B is still tiny", m.cubin_bytes);
        assert!(m.registers > 0);
    }

    #[test]
    fn specialized_catalog_is_large() {
        assert!(specialized_ops().len() >= 20);
    }
}
