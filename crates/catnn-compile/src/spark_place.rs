//! Placement of Mixtral on a DGX Spark cluster: EP / TP / PP, KV budget,
//! all-to-all cost, feasibility. This is a realization in `R(A)` over `H`.

use catnn_core::architecture::{spark_cluster, Cluster, DgxSparkSpec, Precision, SparkTopology};
use catnn_core::category::OpenOp;
use catnn_core::spec::MixtralConfig;
use catnn_core::{CatError, CatResult};

use crate::kernel::cuda::{self, CudaKernel};

#[derive(Debug, Clone)]
pub struct MixtralPlan {
    pub model: String,
    pub n: usize,
    pub precision: Precision,
    pub feasible: bool,
    pub reason: String,
    pub officially_supported_cluster: bool,
    pub ep: usize,
    pub tp: usize,
    pub pp: usize,
    pub param_count: u64,
    pub active_params: u64,
    pub weight_bytes_total: u64,
    pub weight_bytes_per_node: u64,
    pub kv_bytes_per_node: u64,
    pub activation_bytes: u64,
    pub per_node_bytes: u64,
    pub usable_bytes: u64,
    pub all2all_bytes_per_decode: u64,
    pub all2all_seconds: f64,
    pub tokens_per_sec: f64,
    pub tok_s_assumptions: String,
    pub placement_summary: String,
    pub cuda: Vec<CudaKernel>,
    pub nvcc_validated: Option<bool>,
    pub nvcc_note: String,
}

fn divisors(n: usize) -> Vec<usize> {
    (1..=n).filter(|d| n % d == 0).collect()
}

/// Search EP/TP/PP so weights + KV + activations fit in one Spark's usable memory.
pub fn plan_mixtral(
    cfg: &MixtralConfig,
    n: usize,
    precision: Precision,
    topology: SparkTopology,
    context: usize,
    batch: usize,
) -> CatResult<MixtralPlan> {
    if n == 0 {
        return Err(CatError::Fibration("N=0".into()));
    }
    let spec = DgxSparkSpec::official();
    let cluster = spark_cluster(n, topology)?;
    let params = cfg.param_count()?;
    let active = cfg.active_param_count()?;
    let weight_total = precision.packed_bytes(params);
    let hd = cfg.head_dim()?;
    let usable = spec.usable_bytes();

    let mut best: Option<(usize, usize, usize, NodeMem)> = None;
    for pp in divisors(n) {
        let rest = n / pp;
        for ep in divisors(rest) {
            if cfg.num_local_experts % ep != 0 {
                continue;
            }
            let tp = rest / ep;
            if cfg.num_attention_heads % tp != 0 {
                continue;
            }
            if cfg.num_key_value_heads % tp != 0 && tp > cfg.num_key_value_heads {
                continue;
            }
            let mem = node_mem(cfg, precision, ep, tp, pp, context, batch, &spec)?;
            if mem.total <= usable {
                best = Some((ep, tp, pp, mem));
                break;
            }
            if best.as_ref().map(|(_, _, _, m)| m.total).unwrap_or(u64::MAX) > mem.total {
                best = Some((ep, tp, pp, mem));
            }
        }
        if best.as_ref().map(|(_, _, _, m)| m.total <= usable).unwrap_or(false) {
            break;
        }
    }

    let (ep, tp, pp, mem) = best.unwrap_or((1, 1, n.max(1), NodeMem::overflow(weight_total)));
    let feasible = mem.total <= usable;
    let reason = if feasible {
        format!(
            "fits: {:.1} GiB / {:.1} GiB usable on each of {n} Sparks (EP={ep} TP={tp} PP={pp})",
            mem.total as f64 / (1u64 << 30) as f64,
            usable as f64 / (1u64 << 30) as f64
        )
    } else {
        format!(
            "infeasible: need {:.1} GiB/node, usable {:.1} GiB (128 GiB − 20 GiB OS − 4 GiB workspace). \
             Closest shard EP={ep} TP={tp} PP={pp}.",
            mem.total as f64 / (1u64 << 30) as f64,
            usable as f64 / (1u64 << 30) as f64
        )
    };

    let remote = if ep > 1 { 1.0 - 1.0 / ep as f64 } else { 0.0 };
    let a2a = 2
        * cfg.num_experts_per_tok as u64
        * cfg.hidden_size as u64
        * cfg.num_hidden_layers as u64
        * batch as u64;
    let a2a_bytes = precision.packed_bytes(a2a);
    let a2a_remote = (a2a_bytes as f64 * remote) as u64;
    let link_bps = spec.connectx7_bytes_per_sec();
    let a2a_s = if ep > 1 { 2.0e-6 + a2a_remote as f64 / link_bps } else { 0.0 };

    // Roofline decode, batch tokens in parallel.
    let flops = 2.0 * active as f64 * batch as f64;
    let compute_s = flops / (n as f64 * spec.peak_flops(precision));
    let bytes_read = precision.packed_bytes(active) as f64 * batch as f64 / n.max(1) as f64;
    let mem_s = bytes_read / spec.memory_bandwidth_bps;
    let step = compute_s.max(mem_s).max(a2a_s).max(1e-12);
    let tps = batch as f64 / step;
    let assumptions = format!(
        "decode roofline: 2*active_params*B FLOPs @ N×peak (FP4=1 PFLOP from NVIDIA; \
         FP8=0.5×, BF16=0.25× — not measured); weight traffic active_params/N / 273 GB/s; \
         expert all-to-all over ConnectX-7 200 Gb/s with remote fraction 1-1/EP. \
         context={context} batch={batch}."
    );

    let cuda = vec![
        cuda::emit(&OpenOp::RmsNorm { eps_e6: 10 }, "rmsnorm"),
        cuda::emit(
            &OpenOp::SwiGlu { hidden: cfg.hidden_size, intermediate: cfg.intermediate_size },
            "swiglu",
        ),
        cuda::emit(
            &OpenOp::GqaAttention {
                n_heads: cfg.num_attention_heads,
                n_kv_heads: cfg.num_key_value_heads,
                head_dim: hd,
                d_model: cfg.hidden_size,
                sliding_window: cfg.sliding_window,
                rope_theta: cfg.rope_theta as u64,
            },
            "gqa",
        ),
        cuda::emit(
            &OpenOp::Router { n_experts: cfg.num_local_experts, top_k: cfg.num_experts_per_tok },
            "router",
        ),
        cuda::emit(
            &OpenOp::Linear {
                in_features: cfg.hidden_size,
                out_features: cfg.hidden_size,
                bias: false,
            },
            "gemm",
        ),
    ];
    let (nvcc_validated, nvcc_note) = match cuda::try_nvcc(&cuda[0]) {
        Ok(true) => (Some(true), "nvcc -arch=sm_121 accepted generated source".into()),
        Ok(false) => (None, "nvcc not on PATH; CUDA source generated, not compiled".into()),
        Err(e) => (Some(false), e),
    };

    let placement_summary = format!(
        "{}: experts striped EP={ep} ({} experts/node), attention TP={tp}, layers PP={pp} \
         ({} layers/stage). Devices: {}.",
        cluster.name,
        cfg.num_local_experts / ep.max(1),
        cfg.num_hidden_layers.div_ceil(pp.max(1)),
        cluster.nodes.iter().map(|n| n.id.as_str()).collect::<Vec<_>>().join(",")
    );

    Ok(MixtralPlan {
        model: cfg.name.clone(),
        n,
        precision,
        feasible,
        reason,
        officially_supported_cluster: topology.officially_supported(n),
        ep,
        tp,
        pp,
        param_count: params,
        active_params: active,
        weight_bytes_total: weight_total,
        weight_bytes_per_node: mem.weights,
        kv_bytes_per_node: mem.kv,
        activation_bytes: mem.act,
        per_node_bytes: mem.total,
        usable_bytes: usable,
        all2all_bytes_per_decode: a2a_remote,
        all2all_seconds: a2a_s,
        tokens_per_sec: tps,
        tok_s_assumptions: assumptions,
        placement_summary,
        cuda,
        nvcc_validated,
        nvcc_note,
    })
}

#[derive(Debug, Clone)]
struct NodeMem {
    weights: u64,
    kv: u64,
    act: u64,
    total: u64,
}

impl NodeMem {
    fn overflow(w: u64) -> Self {
        Self { weights: w, kv: 0, act: 0, total: w }
    }
}

#[allow(clippy::too_many_arguments)]
fn node_mem(
    cfg: &MixtralConfig,
    prec: Precision,
    ep: usize,
    tp: usize,
    pp: usize,
    context: usize,
    batch: usize,
    spec: &DgxSparkSpec,
) -> CatResult<NodeMem> {
    let h = cfg.hidden_size as u64;
    let mid = cfg.intermediate_size as u64;
    let v = cfg.vocab_size as u64;
    let layers = cfg.num_hidden_layers as u64;
    let hd = cfg.head_dim()? as u64;
    let q = cfg.num_attention_heads as u64 * hd;
    let kv_dim = cfg.num_key_value_heads as u64 * hd;
    let layers_local = layers.div_ceil(pp as u64);
    let experts_local = (cfg.num_local_experts / ep.max(1)) as u64;

    let embed = v * h / tp.max(1) as u64;
    let lm = if cfg.tie_word_embeddings { 0 } else { v * h / tp.max(1) as u64 };
    let attn = layers_local * (h * q + h * kv_dim + h * kv_dim + q * h) / tp.max(1) as u64;
    let norms = layers_local * 2 * h + h;
    let router = layers_local * h * cfg.num_local_experts as u64;
    let expert = 3 * h * mid;
    let experts = layers_local * experts_local * expert;
    // Embed/LM live on first/last PP stage; budget the max stage (worst case both).
    let weights = prec.packed_bytes(embed + lm + attn + norms + router + experts);

    let kv_elems = layers_local
        * 2
        * (cfg.num_key_value_heads as u64 / tp.max(1) as u64).max(1)
        * hd
        * context as u64
        * batch as u64;
    let kv = prec.packed_bytes(kv_elems);

    // Prefill workspace: one layer's hidden + SwiGLU mid for top-k experts.
    let act_elems =
        batch as u64 * context.min(2048) as u64 * (h + cfg.num_experts_per_tok as u64 * mid);
    let act = prec.packed_bytes(act_elems).max(spec.reserved_workspace_bytes / 4);

    Ok(NodeMem { weights, kv, act, total: weights + kv + act })
}

pub fn sweep(
    models: &[MixtralConfig],
    ns: &[usize],
    precs: &[Precision],
    context: usize,
    batch: usize,
) -> CatResult<Vec<MixtralPlan>> {
    let mut out = Vec::new();
    for cfg in models {
        for &n in ns {
            let topo = if n <= 3 { SparkTopology::Direct } else { SparkTopology::Switch };
            for &p in precs {
                out.push(plan_mixtral(cfg, n, p, topo, context, batch)?);
            }
        }
    }
    Ok(out)
}

/// Format the sweep as a fixed-width table.
pub fn format_table(plans: &[MixtralPlan]) -> String {
    let mut s = String::from(
        "model              N  prec  feasible  GiB/node  usable   EP/TP/PP  tok/s    official  note\n",
    );
    s.push_str(&"-".repeat(110));
    s.push('\n');
    for p in plans {
        s.push_str(&format!(
            "{:<18} {:>2}  {:<4}  {:<8}  {:>7.1}  {:>6.1}   {:>2}/{:<2}/{:<2}  {:>7.2}  {:<9} {}\n",
            p.model,
            p.n,
            p.precision.as_str(),
            if p.feasible { "yes" } else { "NO" },
            p.per_node_bytes as f64 / (1u64 << 30) as f64,
            p.usable_bytes as f64 / (1u64 << 30) as f64,
            p.ep,
            p.tp,
            p.pp,
            p.tokens_per_sec,
            if p.officially_supported_cluster { "yes" } else { "manual" },
            if p.feasible { "ok" } else { "OOM" },
        ));
    }
    s
}

/// Unused cluster handle so the fibration object is constructed during planning.
pub fn cluster_of(plan: &MixtralPlan) -> CatResult<Cluster> {
    let topo = if plan.n <= 3 { SparkTopology::Direct } else { SparkTopology::Switch };
    spark_cluster(plan.n, topo)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eight_x22b_bf16_one_spark_is_infeasible() {
        let p = plan_mixtral(
            &MixtralConfig::mixtral_8x22b(),
            1,
            Precision::Bf16,
            SparkTopology::Switch,
            4096,
            1,
        )
        .unwrap();
        assert!(!p.feasible, "{}", p.reason);
    }

    #[test]
    fn eight_x7b_fp4_one_spark_fits() {
        let p = plan_mixtral(
            &MixtralConfig::mixtral_8x7b(),
            1,
            Precision::Fp4,
            SparkTopology::Switch,
            4096,
            1,
        )
        .unwrap();
        assert!(p.feasible, "{}", p.reason);
    }

    #[test]
    fn sweep_covers_requested_grid() {
        let rows = sweep(
            &[MixtralConfig::mixtral_8x7b()],
            &[1, 2, 4, 8],
            &[Precision::Bf16, Precision::Fp8, Precision::Fp4],
            4096,
            1,
        )
        .unwrap();
        assert_eq!(rows.len(), 12);
        assert!(format_table(&rows).contains("mixtral-8x7b"));
    }
}
