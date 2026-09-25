//! Mixtral *training* placement on DGX Spark: memory-parametrized schedule.
//!
//! Weights/grads are bf16; master weights and Adam `m`/`v` are fp32
//! (16 B/parameter unsharded). Optional fp8 mixed precision stores the
//! forward weights at 1 B. State is sharded by EP×TP×PP and then ZeRO
//! across DP. Peak occupancy is **simulated** (code / buffer / scratch).

use catnn_core::architecture::{
    spark_cluster, DeviceMemory, DgxSparkSpec, MemoryBudget, Precision, RegionPeaks, SparkTopology,
};
use catnn_core::spec::MixtralConfig;
use catnn_core::{CatError, CatResult};

use crate::kernel::resource::{describe_cuda, KernelResource};
use crate::schedule::{self, BufferReq, LaunchReq, ScheduleRequest};
use crate::spark_place::weight_elems_per_node;
use crate::train::kernel_resources_for_mixtral;
use catnn_core::category::OpenOp;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ZeroStage {
    None = 0,
    Os = 1,
    OsGrad = 2,
    OsGradParam = 3,
}

impl ZeroStage {
    pub fn as_u8(self) -> u8 {
        self as u8
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CkptPolicy {
    None,
    Layer,
    Selective,
}

/// ZeRO-Infinity-style NVMe offload of optimizer state and/or parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OffloadPolicy {
    None,
    /// Adam `m`/`v` + fp32 master on the Spark NVMe.
    Optimizer,
    /// Optimizer + parameters (working set = 2 layers).
    OptimizerParam,
}

impl OffloadPolicy {
    pub fn as_str(self) -> &'static str {
        match self {
            OffloadPolicy::None => "none",
            OffloadPolicy::Optimizer => "nvme-os",
            OffloadPolicy::OptimizerParam => "nvme-os+p",
        }
    }
}

/// Stated NVMe sequential bandwidth for the Spark SSD.
/// NVIDIA does not publish a DGX Spark SSD figure; 3.5 GB/s is a
/// conservative Gen4 NVMe assumption (typical 3–7 GB/s).
pub const SPARK_NVME_BPS: f64 = 3.5e9;

impl CkptPolicy {
    pub fn as_str(self) -> &'static str {
        match self {
            CkptPolicy::None => "none",
            CkptPolicy::Layer => "layer",
            CkptPolicy::Selective => "selective",
        }
    }
}

#[derive(Debug, Clone)]
pub struct TrainPlan {
    pub model: String,
    pub n: usize,
    pub mixed_fp8: bool,
    pub feasible: bool,
    pub ep: usize,
    pub tp: usize,
    pub pp: usize,
    pub dp: usize,
    pub zero: ZeroStage,
    pub ckpt: CkptPolicy,
    pub offload: OffloadPolicy,
    pub microbatch: usize,
    pub peaks: RegionPeaks,
    pub cap: DeviceMemory,
    pub usable_buffer: u64,
    pub recompute_overhead: f64,
    pub step_seconds: f64,
    pub tokens_per_sec: f64,
    pub seq: usize,
    pub global_batch: usize,
    pub min_n: Option<usize>,
    pub min_buffer: u64,
    pub pick_rationale: String,
    pub assumptions: String,
    pub officially_supported: bool,
}

fn divisors(n: usize) -> Vec<usize> {
    (1..=n).filter(|d| n % d == 0).collect()
}

fn bytes_per_param(mixed_fp8: bool) -> u64 {
    // bf16 weight + bf16 grad + fp32 master + fp32 m + fp32 v = 16
    // fp8 mixed: 1 + 2 + 4 + 4 + 4 = 15 (almost the same; we keep 16)
    let _ = mixed_fp8;
    16
}

#[allow(clippy::too_many_arguments)]
fn state_bytes_per_node(
    cfg: &MixtralConfig,
    ep: usize,
    tp: usize,
    pp: usize,
    dp: usize,
    zero: ZeroStage,
    mixed_fp8: bool,
) -> CatResult<u64> {
    let w = weight_elems_per_node(cfg, ep, tp, pp)?;
    let local = w.total();
    let bpp = bytes_per_param(mixed_fp8);
    // Unique params after MP; ZeRO then shards remaining replicas across DP.
    let shard = match zero {
        ZeroStage::None => 1,
        ZeroStage::Os => dp.max(1),
        ZeroStage::OsGrad => dp.max(1),
        ZeroStage::OsGradParam => dp.max(1),
    };
    // Optimizer states (8 B) always /shard under ZeRO≥1; grads (2 B) under ≥2;
    // params (2–4 B) under 3. We use a conservative 16/shard for ZeRO-3 and
    // interpolate.
    let per = match zero {
        ZeroStage::None => bpp,
        ZeroStage::Os => 8 / shard as u64 + 8,          // m/v sharded; params+grads full
        ZeroStage::OsGrad => 10 / shard as u64 + 6,     // m/v+grad sharded
        ZeroStage::OsGradParam => bpp / shard as u64,   // everything
    };
    Ok(local * per.max(1))
}

fn layer_elems(cfg: &MixtralConfig, ep: usize, tp: usize) -> CatResult<u64> {
    let w = weight_elems_per_node(cfg, ep, tp, 1)?;
    let layers = cfg.num_hidden_layers.max(1) as u64;
    Ok(w.total() / layers)
}

/// Bytes that must sit in the **buffer** region after ZeRO and offload.
#[allow(clippy::too_many_arguments)]
fn device_buffer_bytes(
    cfg: &MixtralConfig,
    ep: usize,
    tp: usize,
    pp: usize,
    dp: usize,
    zero: ZeroStage,
    mixed_fp8: bool,
    offload: OffloadPolicy,
) -> CatResult<u64> {
    let full = state_bytes_per_node(cfg, ep, tp, pp, dp, zero, mixed_fp8)?;
    match offload {
        OffloadPolicy::None => Ok(full),
        OffloadPolicy::Optimizer => {
            let local = weight_elems_per_node(cfg, ep, tp, pp)?.total();
            let wg = if mixed_fp8 { 3 } else { 4 };
            let shard = match zero {
                ZeroStage::OsGradParam => dp.max(1) as u64,
                _ => 1,
            };
            Ok(local * wg / shard)
        }
        OffloadPolicy::OptimizerParam => {
            // ZeRO-Infinity working set: current + prefetch layer of
            // weights/grads, plus Adam m/v for those two layers while they
            // update. Master + the rest of the state stay on NVMe.
            let layer = layer_elems(cfg, ep, tp)?;
            let wg = if mixed_fp8 { 3 } else { 4 };
            Ok(layer * (wg + 8) * 2)
        }
    }
}

fn offload_seconds(
    cfg: &MixtralConfig,
    ep: usize,
    tp: usize,
    pp: usize,
    offload: OffloadPolicy,
) -> CatResult<f64> {
    let local = weight_elems_per_node(cfg, ep, tp, pp)?.total() as f64;
    let bytes = match offload {
        OffloadPolicy::None => 0.0,
        OffloadPolicy::Optimizer => 2.0 * local * 12.0, // r/w master+m+v
        OffloadPolicy::OptimizerParam => 2.0 * local * 16.0,
    };
    Ok(bytes / SPARK_NVME_BPS)
}

fn act_bytes_per_node(
    cfg: &MixtralConfig,
    prec: Precision,
    pp: usize,
    tp: usize,
    seq: usize,
    micro: usize,
    ckpt: CkptPolicy,
) -> u64 {
    let h = cfg.hidden_size as u64;
    let layers_local = (cfg.num_hidden_layers as u64).div_ceil(pp as u64);
    let saved = match ckpt {
        CkptPolicy::None => layers_local * 4, // rms, attn, rms, moe
        CkptPolicy::Layer => layers_local,    // one activation / layer
        CkptPolicy::Selective => layers_local * 2,
    };
    let elem = micro as u64 * seq as u64 * h / tp.max(1) as u64;
    prec.packed_bytes(elem * saved)
}

#[allow(clippy::too_many_arguments)]
fn train_cost(
    cfg: &MixtralConfig,
    spec: &DgxSparkSpec,
    n: usize,
    ep: usize,
    tp: usize,
    pp: usize,
    dp: usize,
    seq: usize,
    global_batch: usize,
    mixed_fp8: bool,
    ckpt: CkptPolicy,
    offload: OffloadPolicy,
) -> f64 {
    let active = cfg.active_param_count().unwrap_or(1) as f64;
    let micro = (global_batch / dp.max(1)).max(1);
    let tokens = micro as f64 * seq as f64;
    // ~6 FLOP/param/token (fwd+bwd). EP/TP shard work; PP is 1F1B
    // (stage time = full/PP, pipeline time = (PP + #micro − 1) · stage).
    let flops = 6.0 * active * tokens / (ep * tp).max(1) as f64;
    let compute = flops / spec.peak_flops(if mixed_fp8 { Precision::Fp8 } else { Precision::Bf16 });
    let hbm = active * 2.0 * tokens * 2.0 / ((ep * tp).max(1) as f64) / spec.memory_bandwidth_bps;
    let link = spec.connectx7_bytes_per_sec();
    let gbytes = Precision::Bf16.packed_bytes(cfg.param_count().unwrap_or(1)) as f64
        / (ep * tp * pp * dp).max(1) as f64;
    let ar = if dp > 1 {
        2.0 * (dp as f64 - 1.0) / dp as f64 * gbytes / link
    } else {
        0.0
    };
    let tp_ar = if tp > 1 {
        4.0 * cfg.num_hidden_layers as f64 * 2.0 * (tp as f64 - 1.0) / tp as f64
            * cfg.hidden_size as f64
            * 2.0
            / link
            * tokens
    } else {
        0.0
    };
    let ck = match ckpt {
        CkptPolicy::None => 0.0,
        CkptPolicy::Layer => 0.30,
        CkptPolicy::Selective => 0.15,
    };
    let _ = n;
    let stage = compute.max(hbm) / pp.max(1) as f64;
    let pipe = (pp + micro - 1) as f64 * stage;
    let nvme = offload_seconds(cfg, ep, tp, pp, offload).unwrap_or(0.0);
    (pipe + ar + tp_ar) * (1.0 + ck) + nvme
}

#[allow(clippy::too_many_arguments)]
fn simulate_regions(
    cfg: &MixtralConfig,
    regions: &DeviceMemory,
    ep: usize,
    tp: usize,
    pp: usize,
    seq: usize,
    micro: usize,
    state: u64,
    act: u64,
) -> (RegionPeaks, f64, usize, bool, Option<catnn_core::architecture::MemoryRegionKind>) {
    let kernels: Vec<KernelResource> = kernel_resources_for_mixtral();
    let mut bufs = vec![BufferReq {
        id: "state".into(),
        bytes: state,
        live_from: 0,
        live_to: 8,
        checkpoint: false,
    }];
    bufs.push(BufferReq {
        id: "act".into(),
        bytes: act.max(4096),
        live_from: 1,
        live_to: 6,
        checkpoint: false,
    });
    let mut launches = Vec::new();
    let mut t = 1usize;
    for (i, k) in kernels.iter().enumerate() {
        launches.push(LaunchReq { at: t, kernel: k.clone() });
        let _ = (cfg, ep, tp, pp, seq, micro);
        t += 1;
        if i == 2 {
            // Second launch of the same expert kernel later → tests residency.
            launches.push(LaunchReq { at: t, kernel: k.clone() });
            t += 1;
        }
    }
    let hier = catnn_core::architecture::MemoryHierarchy::dgx_spark();
    let s = schedule::schedule(ScheduleRequest {
        buffers: &bufs,
        launches: &launches,
        regions,
        hierarchy: &hier,
        allow_recompute: true,
        allow_reload: true,
    });
    (s.peaks, s.recompute_overhead, s.kernel_reloads, s.feasible, s.overflow_region)
}

fn any_feasible(
    cfg: &MixtralConfig,
    n: usize,
    mixed_fp8: bool,
    seq: usize,
    global_batch: usize,
    usable: u64,
) -> bool {
    for pp in divisors(n) {
        let rest = n / pp;
        for ep in divisors(rest) {
            if cfg.num_local_experts % ep != 0 {
                continue;
            }
            let tp_dp = rest / ep;
            for tp in divisors(tp_dp) {
                if cfg.num_attention_heads % tp != 0 {
                    continue;
                }
                let dp = tp_dp / tp;
                for &zero in &[ZeroStage::OsGradParam, ZeroStage::OsGrad, ZeroStage::Os, ZeroStage::None]
                {
                    for &ckpt in &[CkptPolicy::Layer, CkptPolicy::Selective, CkptPolicy::None] {
                        for &offload in &[
                            OffloadPolicy::None,
                            OffloadPolicy::Optimizer,
                            OffloadPolicy::OptimizerParam,
                        ] {
                            let state = match device_buffer_bytes(
                                cfg, ep, tp, pp, dp, zero, mixed_fp8, offload,
                            ) {
                                Ok(s) => s,
                                Err(_) => continue,
                            };
                            let micro = (global_batch / dp.max(1)).max(1);
                            let act = act_bytes_per_node(
                                cfg,
                                if mixed_fp8 { Precision::Fp8 } else { Precision::Bf16 },
                                pp,
                                tp,
                                seq,
                                micro,
                                ckpt,
                            );
                            if state + act <= usable {
                                return true;
                            }
                        }
                    }
                }
            }
        }
    }
    false
}

fn first_feasible_n(
    cfg: &MixtralConfig,
    mixed_fp8: bool,
    seq: usize,
    global_batch: usize,
    budget: Option<MemoryBudget>,
) -> Option<usize> {
    let spec = DgxSparkSpec::official();
    let usable = budget.map(|b| b.per_device()).unwrap_or(spec.usable_bytes());
    [1usize, 2, 4, 8, 16, 32, 64]
        .into_iter()
        .find(|&cand| any_feasible(cfg, cand, mixed_fp8, seq, global_batch, usable))
}

pub fn plan_train(
    cfg: &MixtralConfig,
    n: usize,
    mixed_fp8: bool,
    seq: usize,
    global_batch: usize,
    budget: Option<MemoryBudget>,
) -> CatResult<TrainPlan> {
    if n == 0 {
        return Err(CatError::Fibration("N=0".into()));
    }
    let spec = DgxSparkSpec::official();
    let topo = if n <= 3 { SparkTopology::Direct } else { SparkTopology::Switch };
    let mut cluster = spark_cluster(n, topo)?;
    if let Some(b) = budget {
        cluster = cluster.with_budget(b);
    }
    let usable = cluster.budget.map(|b| b.per_device()).unwrap_or(spec.usable_bytes());
    let mut regions = DeviceMemory::dgx_spark();
    regions.buffer.bytes = regions.buffer.bytes.min(usable);
    if let Some(b) = budget {
        regions.apply_budget(&b);
    }

    let mut best: Option<TrainPlan> = None;
    for pp in divisors(n) {
        let rest = n / pp;
        for ep in divisors(rest) {
            if cfg.num_local_experts % ep != 0 {
                continue;
            }
            let tp_dp = rest / ep;
            for tp in divisors(tp_dp) {
                if cfg.num_attention_heads % tp != 0 {
                    continue;
                }
                let dp = tp_dp / tp;
                for &zero in &[ZeroStage::OsGradParam, ZeroStage::OsGrad, ZeroStage::Os, ZeroStage::None]
                {
                    for &ckpt in &[CkptPolicy::Layer, CkptPolicy::Selective, CkptPolicy::None] {
                    for &offload in &[
                        OffloadPolicy::None,
                        OffloadPolicy::Optimizer,
                        OffloadPolicy::OptimizerParam,
                    ] {
                        let state = device_buffer_bytes(
                            cfg, ep, tp, pp, dp, zero, mixed_fp8, offload,
                        )?;
                        let micro = (global_batch / dp.max(1)).max(1);
                        let act = act_bytes_per_node(
                            cfg,
                            if mixed_fp8 { Precision::Fp8 } else { Precision::Bf16 },
                            pp,
                            tp,
                            seq,
                            micro,
                            ckpt,
                        );
                        let (mut peaks, _rec, _reloads, sim_ok, overflow) =
                            simulate_regions(cfg, &regions, ep, tp, pp, seq, micro, state, act);
                        let buffer_need = state + act;
                        if !sim_ok {
                            let floor = kernel_region_floor();
                            peaks.code = floor.code;
                            peaks.scratch = floor.scratch;
                            peaks.buffer = buffer_need;
                        }
                        let fits = sim_ok && buffer_need <= regions.buffer.bytes && peaks.exceeds(&regions).is_none();
                        let _ = overflow;
                        let step = train_cost(
                            cfg, &spec, n, ep, tp, pp, dp, seq, global_batch, mixed_fp8, ckpt,
                            offload,
                        );
                        let ck_frac = match ckpt {
                            CkptPolicy::None => 0.0,
                            CkptPolicy::Layer => 0.30,
                            CkptPolicy::Selective => 0.15,
                        };
                        let tps = (global_batch * seq) as f64 / step.max(1e-12);
                        let plan = TrainPlan {
                            model: cfg.name.clone(),
                            n,
                            mixed_fp8,
                            feasible: fits,
                            ep,
                            tp,
                            pp,
                            dp,
                            zero,
                            ckpt,
                            offload,
                            microbatch: micro,
                            peaks,
                            cap: regions.clone(),
                            usable_buffer: regions.buffer.bytes,
                            recompute_overhead: ck_frac,
                            step_seconds: step,
                            tokens_per_sec: tps,
                            seq,
                            global_batch,
                            min_n: None,
                            min_buffer: buffer_need,
                            pick_rationale: String::new(),
                            assumptions: String::new(),
                            officially_supported: topo.officially_supported(n),
                        };
                        if fits {
                            best = Some(match best {
                                None => plan,
                                Some(b) => {
                                    if plan.step_seconds < b.step_seconds {
                                        plan
                                    } else {
                                        b
                                    }
                                }
                            });
                        } else if best.as_ref().map(|b| b.feasible).unwrap_or(false) {
                            // keep feasible
                        } else if best.as_ref().map(|b| b.min_buffer).unwrap_or(u64::MAX) > buffer_need {
                            best = Some(plan);
                        }
                    }
                    }
                }
            }
        }
    }
    let mut p = best.ok_or_else(|| CatError::Fibration("no train factorization".into()))?;
    p.min_n = first_feasible_n(cfg, mixed_fp8, seq, global_batch, budget);
    if !p.feasible {
        p.pick_rationale = format!(
            "infeasible on N={n}: need ≥ {:.1} GiB buffer/node (have {:.1}); \
             closest EP/TP/PP/DP={}/{}/{}/{} ZeRO-{} ckpt={} offload={}. \
             min feasible N={:?}; min buffer at this N is {:.1} GiB.",
            p.min_buffer as f64 / (1u64 << 30) as f64,
            p.usable_buffer as f64 / (1u64 << 30) as f64,
            p.ep,
            p.tp,
            p.pp,
            p.dp,
            p.zero.as_u8(),
            p.ckpt.as_str(),
            p.offload.as_str(),
            p.min_n,
            p.min_buffer as f64 / (1u64 << 30) as f64
        );
    } else {
        p.pick_rationale = format!(
            "min step {:.3}s (recompute +{:.0}%, micro={}); \
             EP/TP/PP/DP={}/{}/{}/{} ZeRO-{} ckpt={} offload={}; \
             peaks code={:.1}KiB buffer={:.1}GiB scratch={}B; \
             seq={} gbs={}.",
            p.step_seconds,
            p.recompute_overhead * 100.0,
            p.microbatch,
            p.ep,
            p.tp,
            p.pp,
            p.dp,
            p.zero.as_u8(),
            p.ckpt.as_str(),
            p.offload.as_str(),
            p.peaks.code as f64 / 1024.0,
            p.peaks.buffer as f64 / (1u64 << 30) as f64,
            p.peaks.scratch,
            p.seq,
            p.global_batch
        );
    }
    p.assumptions = format!(
        "training roofline: 6·active_params·tokens FLOPs @ one GB10 peak; \
         HBM 273 GB/s; DP reduce-scatter of bf16 grads over ConnectX-7 200 Gb/s; \
         two TP ring ARs/layer/token (fwd+bwd). 1F1B time is \
         (PP+#micro−1)·stage. Adam fp32 m/v + fp32 master + \
         bf16 weights/grads (16 B/param); ZeRO shards that state. \
         NVMe offload @ 3.5 GB/s (assumption: Gen4 NVMe; Spark SSD BW unpublished). \
         Layer checkpoint adds ~30% recompute. Cubin sizes measured with \
         nvcc -arch=sm_80 (CUDA 12.0; sm_121 needs 12.9+). \
         seq={seq} global_batch={global_batch}."
    );
    let _ = describe_cuda(&OpenOp::RmsNorm { eps_e6: 10 }, "rms");
    Ok(p)
}

pub fn sweep_train(
    models: &[MixtralConfig],
    ns: &[usize],
    seq: usize,
    global_batch: usize,
    budget: Option<MemoryBudget>,
    fp8: bool,
) -> CatResult<Vec<TrainPlan>> {
    let mut out = Vec::new();
    for cfg in models {
        for &n in ns {
            out.push(plan_train(cfg, n, false, seq, global_batch, budget)?);
            out.push(plan_train(cfg, n, true, seq, global_batch, budget)?);
            let _ = fp8;
        }
    }
    Ok(out)
}

pub fn format_train_table(plans: &[TrainPlan]) -> String {
    let mut s = String::from(
        "model              N  mp   feas  EP/TP/PP/DP  ZeRO  ckpt      offload     codeKiB  bufGiB  rec%   step_s   tok/s  micro seq  gbs  minN  minMGiB\n",
    );
    s.push_str(&"-".repeat(160));
    s.push('\n');
    for p in plans {
        s.push_str(&format!(
            "{:<18} {:>2}  {:<4} {:<5} {:>2}/{:<2}/{:<2}/{:<2}  z{:<3}  {:<9} {:<10} {:>7.1} {:>7.1} {:>5.0} {:>8.2} {:>7.2} {:>5} {:>4} {:>4} {:>4} {:>8.1}\n",
            p.model,
            p.n,
            if p.mixed_fp8 { "fp8" } else { "bf16" },
            if p.feasible { "yes" } else { "NO" },
            p.ep,
            p.tp,
            p.pp,
            p.dp,
            p.zero.as_u8(),
            p.ckpt.as_str(),
            p.offload.as_str(),
            p.peaks.code as f64 / 1024.0,
            p.peaks.buffer as f64 / (1u64 << 30) as f64,
            p.recompute_overhead * 100.0,
            p.step_seconds,
            p.tokens_per_sec,
            p.microbatch,
            p.seq,
            p.global_batch,
            p.min_n.unwrap_or(0),
            p.min_buffer as f64 / (1u64 << 30) as f64,
        ));
    }
    s
}

pub fn format_train_rationales(plans: &[TrainPlan]) -> String {
    let mut s = String::from("chosen training plan\n");
    s.push_str(&"-".repeat(120));
    s.push('\n');
    for p in plans {
        s.push_str(&format!(
            "{} N={} {} — {}\n",
            p.model,
            p.n,
            if p.mixed_fp8 { "fp8" } else { "bf16" },
            p.pick_rationale
        ));
    }
    s
}

fn kernel_region_floor() -> RegionPeaks {
    let kernels = kernel_resources_for_mixtral();
    RegionPeaks {
        code: kernels.iter().map(|k| k.code_footprint()).sum(),
        buffer: 0,
        scratch: kernels.iter().map(|k| k.scratch_bytes()).max().unwrap_or(0),
    }
}

/// Inference-plan helper: attach simulated region peaks for the decode kernels.
pub fn inference_region_peaks(weight_bytes: u64, kv: u64, act: u64) -> RegionPeaks {
    let kernels = kernel_resources_for_mixtral();
    let mut regions = DeviceMemory::dgx_spark();
    regions.buffer.bytes = 104 * (1u64 << 30);
    let bufs = vec![
        BufferReq { id: "w".into(), bytes: weight_bytes.max(1), live_from: 0, live_to: 3, checkpoint: false },
        BufferReq { id: "kv".into(), bytes: kv.max(1), live_from: 1, live_to: 3, checkpoint: false },
        BufferReq { id: "act".into(), bytes: act.max(1), live_from: 2, live_to: 2, checkpoint: false },
    ];
    let launches: Vec<LaunchReq> =
        kernels.into_iter().enumerate().map(|(i, k)| LaunchReq { at: i.min(3), kernel: k }).collect();
    let hier = catnn_core::architecture::MemoryHierarchy::dgx_spark();
    let s = schedule::schedule(ScheduleRequest {
        buffers: &bufs,
        launches: &launches,
        regions: &regions,
        hierarchy: &hier,
        allow_recompute: true,
        allow_reload: true,
    });
    if s.feasible {
        s.peaks
    } else {
        let floor = kernel_region_floor();
        RegionPeaks {
            code: floor.code,
            buffer: weight_bytes.saturating_add(kv).saturating_add(act),
            scratch: floor.scratch,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eight_x7b_n1_training_needs_nvme_offload() {
        let p = plan_train(&MixtralConfig::mixtral_8x7b(), 1, false, 2048, 8, None).unwrap();
        assert!(p.feasible, "{}", p.pick_rationale);
        assert_ne!(p.offload, OffloadPolicy::None, "{}", p.pick_rationale);
        assert!(p.min_buffer < 104 * (1u64 << 30), "offload should cut device buffer");
    }

    #[test]
    fn tiny_train_plan_fits_desktop_budget() {
        let p = plan_train(
            &MixtralConfig::tiny(),
            1,
            false,
            8,
            2,
            Some(MemoryBudget::buffer_only(8 << 30)),
        )
        .unwrap();
        assert!(p.feasible, "{}", p.pick_rationale);
        assert!(p.peaks.exceeds(&p.cap).is_none());
    }
}
