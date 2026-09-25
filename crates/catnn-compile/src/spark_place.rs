//! Placement of Mixtral on a DGX Spark cluster: EP / TP / PP, KV budget,
//! communication, feasibility. This is a realization in `R(A)` over `H`.
//!
//! Memory, placement, and the decode roofline share one sharding model:
//!
//! - Expert SwiGLU GEMMs are partitioned by **EP × TP × PP**.
//! - Attention QKV/O is partitioned by **TP × PP** and **replicated across EP**.
//! - Embeddings / LM head are TP-sharded (vocab-parallel) and live on the
//!   first / last PP stage.
//! - RMSNorms and the router are treated as replicas on every rank that owns
//!   the layer (PP only).
//!
//! Search enumerates every factorization `EP * TP * PP = N` that respects
//! expert / head divisibility, drops plans that miss the per-node memory
//! budget, and keeps the **minimum decode-step** survivor. Decode time is
//! per-node HBM (273 GB/s) plus EP all-to-all, two TP ring all-reduces per
//! layer, and PP P2P — not `N` times the single-node rate for free.

use catnn_core::architecture::{spark_cluster, Cluster, DgxSparkSpec, Precision, SparkTopology};
use catnn_core::category::OpenOp;
use catnn_core::spec::MixtralConfig;
use catnn_core::{CatError, CatResult};

use crate::kernel::cuda::{self, CudaKernel};

/// Relative decode-step slack inside which a higher-EP plan is preferred
/// (Mixtral top-2 makes expert-parallel a natural default).
const EP_TIE_REL: f64 = 0.10;

#[derive(Debug, Clone)]
pub struct MixtralPlan {
    pub model: String,
    pub n: usize,
    pub precision: Precision,
    pub feasible: bool,
    pub reason: String,
    pub pick_rationale: String,
    pub officially_supported_cluster: bool,
    pub ep: usize,
    pub tp: usize,
    pub pp: usize,
    pub param_count: u64,
    pub active_params: u64,
    pub weight_bytes_total: u64,
    pub weight_bytes_per_node: u64,
    pub replica_weight_bytes: u64,
    pub kv_bytes_per_node: u64,
    pub activation_bytes: u64,
    pub per_node_bytes: u64,
    pub usable_bytes: u64,
    pub all2all_bytes_per_decode: u64,
    pub all2all_seconds: f64,
    pub tp_allreduce_seconds: f64,
    pub pp_p2p_seconds: f64,
    pub hbm_seconds: f64,
    pub compute_seconds: f64,
    pub step_seconds: f64,
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

/// Element counts for one node's resident weights under (EP, TP, PP).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WeightBreakdown {
    pub embed: u64,
    pub lm: u64,
    pub attn: u64,
    pub norms: u64,
    pub router: u64,
    pub experts: u64,
}

impl WeightBreakdown {
    pub fn total(&self) -> u64 {
        self.embed + self.lm + self.attn + self.norms + self.router + self.experts
    }

    /// Tensors that do not shrink with EP×TP (norms, router) plus the
    /// EP-replicated share of attention when `ep > 1` is already excluded
    /// from `attn` (attn is only `/ (TP*PP)`).
    pub fn replica_elems(&self) -> u64 {
        self.norms + self.router
    }

    pub fn shardable_elems(&self) -> u64 {
        self.embed + self.lm + self.attn + self.experts
    }
}

/// Per-node resident weight *elements* (not bytes).
pub fn weight_elems_per_node(
    cfg: &MixtralConfig,
    ep: usize,
    tp: usize,
    pp: usize,
) -> CatResult<WeightBreakdown> {
    if ep == 0 || tp == 0 || pp == 0 {
        return Err(CatError::Fibration("EP/TP/PP must be ≥ 1".into()));
    }
    let h = cfg.hidden_size as u64;
    let mid = cfg.intermediate_size as u64;
    let v = cfg.vocab_size as u64;
    let layers = cfg.num_hidden_layers as u64;
    let hd = cfg.head_dim()? as u64;
    let q = cfg.num_attention_heads as u64 * hd;
    let kv_dim = cfg.num_key_value_heads as u64 * hd;
    let layers_local = layers.div_ceil(pp as u64);
    let experts_local = (cfg.num_local_experts / ep) as u64;

    // Vocab-parallel embed / LM on the first / last PP stage. Budget the
    // worst-case node (a stage that owns both when PP = 1).
    let embed = v * h / tp as u64;
    let lm = if cfg.tie_word_embeddings { 0 } else { v * h / tp as u64 };
    // Attention is TP-sharded and PP-sliced; EP ranks each keep a full copy
    // of that TP shard (standard MoE + Megatron).
    let attn = layers_local * (h * q + h * kv_dim + h * kv_dim + q * h) / tp as u64;
    let norms = layers_local * 2 * h + h;
    let router = layers_local * h * cfg.num_local_experts as u64;
    let expert = 3 * h * mid;
    let experts = layers_local * experts_local * expert / tp as u64;
    Ok(WeightBreakdown { embed, lm, attn, norms, router, experts })
}

/// Search EP/TP/PP so weights + KV + activations fit, then keep the
/// feasible plan with the lowest estimated decode step.
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

    let scored = enumerate_plans(cfg, n, precision, context, batch, &spec)?;
    let feasible: Vec<&Scored> = scored.iter().filter(|s| s.mem.total <= usable).collect();

    let chosen = if let Some(best) = pick_min_latency(&feasible) {
        best
    } else {
        scored
            .iter()
            .min_by_key(|s| s.mem.total)
            .ok_or_else(|| CatError::Fibration("no EP/TP/PP factorization".into()))?
    };
    let mem = chosen.mem.clone();
    let dec = chosen.dec.clone();
    let ep = chosen.ep;
    let tp = chosen.tp;
    let pp = chosen.pp;
    let feasible_flag = mem.total <= usable;

    let pick_rationale = rationale(chosen, &feasible, feasible_flag, n, usable);
    let reason = if feasible_flag {
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

    let assumptions = format!(
        "decode roofline is per-node, not N× free: max(HBM, compute) + comm. \
         HBM = this rank's active weight bytes (attn/(TP) + E[top-{k} experts]/(EP·TP) \
         + replicas + LM head/TP) / 273 GB/s, summed over all layers (PP is sequential \
         at batch=1). Compute = 2·active_local FLOPs @ one GB10 peak (FP4=1 PFLOP from \
         NVIDIA; FP8=0.5×, BF16=0.25× — not measured). Comm over ConnectX-7 200 Gb/s: \
         EP all-to-all volume 2·top_k·H·L with remote fraction (EP-1)/EP; TP pays two \
         ring all-reduces per layer per token (2·(TP-1)/TP·H each, plus 2·(TP-1)·2µs \
         hops); PP adds (PP-1) hidden P2P sends. context={context} batch={batch}.",
        k = cfg.num_experts_per_tok
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
        "{}: experts striped EP={ep}×TP={tp} ({} expert-shards/node), attention TP={tp} \
         (replicated across EP), layers PP={pp} ({} layers/stage). Devices: {}.",
        cluster.name,
        (cfg.num_local_experts / ep.max(1)) as u64,
        cfg.num_hidden_layers.div_ceil(pp.max(1)),
        cluster.nodes.iter().map(|n| n.id.as_str()).collect::<Vec<_>>().join(",")
    );

    Ok(MixtralPlan {
        model: cfg.name.clone(),
        n,
        precision,
        feasible: feasible_flag,
        reason,
        pick_rationale,
        officially_supported_cluster: topology.officially_supported(n),
        ep,
        tp,
        pp,
        param_count: params,
        active_params: active,
        weight_bytes_total: weight_total,
        weight_bytes_per_node: mem.weights,
        replica_weight_bytes: mem.replicas,
        kv_bytes_per_node: mem.kv,
        activation_bytes: mem.act,
        per_node_bytes: mem.total,
        usable_bytes: usable,
        all2all_bytes_per_decode: dec.a2a_bytes,
        all2all_seconds: dec.a2a_s,
        tp_allreduce_seconds: dec.ar_s,
        pp_p2p_seconds: dec.pp_s,
        hbm_seconds: dec.hbm_s,
        compute_seconds: dec.compute_s,
        step_seconds: dec.step_s,
        tokens_per_sec: if dec.step_s > 0.0 { batch as f64 / dec.step_s } else { 0.0 },
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
    replicas: u64,
    kv: u64,
    act: u64,
    total: u64,
}

#[derive(Debug, Clone)]
struct DecodeCost {
    hbm_s: f64,
    compute_s: f64,
    a2a_s: f64,
    ar_s: f64,
    pp_s: f64,
    step_s: f64,
    a2a_bytes: u64,
    active_weight_bytes: u64,
}

#[derive(Debug, Clone)]
struct Scored {
    ep: usize,
    tp: usize,
    pp: usize,
    mem: NodeMem,
    dec: DecodeCost,
}

fn enumerate_plans(
    cfg: &MixtralConfig,
    n: usize,
    precision: Precision,
    context: usize,
    batch: usize,
    spec: &DgxSparkSpec,
) -> CatResult<Vec<Scored>> {
    let mut out = Vec::new();
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
            let mem = node_mem(cfg, precision, ep, tp, pp, context, batch, spec)?;
            let dec = decode_cost(cfg, precision, ep, tp, pp, batch, spec)?;
            out.push(Scored { ep, tp, pp, mem, dec });
        }
    }
    if out.is_empty() {
        return Err(CatError::Fibration(format!(
            "no EP/TP/PP with EP*TP*PP={n} divides experts={} heads={}",
            cfg.num_local_experts, cfg.num_attention_heads
        )));
    }
    Ok(out)
}

fn pick_min_latency<'a>(feasible: &[&'a Scored]) -> Option<&'a Scored> {
    let mut best: Option<&Scored> = None;
    for &s in feasible {
        best = Some(match best {
            None => s,
            Some(b) => better_plan(b, s),
        });
    }
    best
}

/// Lower decode step wins. Within [`EP_TIE_REL`], prefer higher EP (Mixtral
/// top-2). Then lower per-node bytes, then lower PP (pipeline is sequential
/// at batch-1 decode).
fn better_plan<'a>(a: &'a Scored, b: &'a Scored) -> &'a Scored {
    let sa = a.dec.step_s;
    let sb = b.dec.step_s;
    let close = {
        let m = sa.min(sb);
        m > 0.0 && (sa - sb).abs() / m <= EP_TIE_REL
    };
    if close {
        if b.ep != a.ep {
            return if b.ep > a.ep { b } else { a };
        }
        if b.mem.total != a.mem.total {
            return if b.mem.total < a.mem.total { b } else { a };
        }
        if b.pp != a.pp {
            return if b.pp < a.pp { b } else { a };
        }
        return a;
    }
    if sb < sa {
        b
    } else {
        a
    }
}

fn rationale(
    chosen: &Scored,
    feasible: &[&Scored],
    fits: bool,
    n: usize,
    usable: u64,
) -> String {
    let dec = &chosen.dec;
    let ms = |s: f64| s * 1e3;
    if !fits {
        return format!(
            "no plan fits {:.1} GiB usable; reporting min-memory shard EP={} TP={} PP={} \
             ({:.1} GiB/node, decode would be {:.2} ms).",
            usable as f64 / (1u64 << 30) as f64,
            chosen.ep,
            chosen.tp,
            chosen.pp,
            chosen.mem.total as f64 / (1u64 << 30) as f64,
            ms(dec.step_s)
        );
    }

    let runner = feasible
        .iter()
        .copied()
        .filter(|s| s.ep != chosen.ep || s.tp != chosen.tp || s.pp != chosen.pp)
        .min_by(|a, b| a.dec.step_s.partial_cmp(&b.dec.step_s).unwrap_or(std::cmp::Ordering::Equal));

    let mut why = format!(
        "min decode {:.2} ms among {} feasible EP*TP*PP={n} plans \
         (HBM {:.2} + compute {:.2} → max {:.2}; + EP a2a {:.2} + TP AR {:.2} + PP {:.2}); \
         picked EP={} TP={} PP={}.",
        ms(dec.step_s),
        feasible.len(),
        ms(dec.hbm_s),
        ms(dec.compute_s),
        ms(dec.hbm_s.max(dec.compute_s)),
        ms(dec.a2a_s),
        ms(dec.ar_s),
        ms(dec.pp_s),
        chosen.ep,
        chosen.tp,
        chosen.pp
    );

    if let Some(alt) = runner {
        let ratio = if alt.dec.step_s > 0.0 { alt.dec.step_s / dec.step_s.max(1e-18) } else { 1.0 };
        why.push_str(&format!(
            " Next is EP={} TP={} PP={} at {:.2} ms ({:.2}×).",
            alt.ep,
            alt.tp,
            alt.pp,
            ms(alt.dec.step_s),
            ratio
        ));
        if chosen.tp > 1 && alt.ep > chosen.ep {
            why.push_str(
                " EP does not cut attention HBM (attn is replicated across expert ranks); \
                 Mixtral top-2 all-to-all is a 2·k·H activation exchange, while TP=N \
                 splits every GEMM including experts and only pays two ring all-reduces \
                 of H per layer on 200 Gb/s.",
            );
        } else if chosen.ep > 1 && alt.tp > chosen.tp {
            why.push_str(
                " Top-2 routing makes EP natural: all-to-all of activations beat TP's \
                 two all-reduces/layer once the 10% EP tie-break applies, or EP's HBM \
                 (experts/EP, attn replicated) is the latency winner.",
            );
        } else if chosen.pp > 1 {
            why.push_str(
                " PP is sequential at batch-1 decode (no HBM win, plus P2P); chosen \
                 only because it is the latency-best *feasible* shard.",
            );
        }
    } else if n == 1 {
        why.push_str(" Single Spark: no EP/TP/PP comm.");
    }

    why
}

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
    let w = weight_elems_per_node(cfg, ep, tp, pp)?;
    let hd = cfg.head_dim()? as u64;
    let layers_local = (cfg.num_hidden_layers as u64).div_ceil(pp as u64);
    let weights = prec.packed_bytes(w.total());
    let replicas = prec.packed_bytes(w.replica_elems());

    // KV is TP-sharded and PP-sliced; EP ranks each keep the local-layer KV
    // (they all run attention).
    let kv_heads = (cfg.num_key_value_heads as u64 / tp as u64).max(1);
    let kv_elems = layers_local * 2 * kv_heads * hd * context as u64 * batch as u64;
    let kv = prec.packed_bytes(kv_elems);

    let h = cfg.hidden_size as u64;
    let mid = cfg.intermediate_size as u64;
    let act_elems =
        batch as u64 * context.min(2048) as u64 * (h + cfg.num_experts_per_tok as u64 * mid);
    let act = prec.packed_bytes(act_elems).max(spec.reserved_workspace_bytes / 4);

    Ok(NodeMem { weights, replicas, kv, act, total: weights + kv + act })
}

/// Decode step for `batch` tokens in parallel. HBM and FLOPs are **one
/// node's** traffic / work; communication is added, not hidden behind `max`.
fn decode_cost(
    cfg: &MixtralConfig,
    prec: Precision,
    ep: usize,
    tp: usize,
    pp: usize,
    batch: usize,
    spec: &DgxSparkSpec,
) -> CatResult<DecodeCost> {
    let h = cfg.hidden_size as u64;
    let mid = cfg.intermediate_size as u64;
    let v = cfg.vocab_size as u64;
    let layers = cfg.num_hidden_layers as u64;
    let hd = cfg.head_dim()? as u64;
    let q = cfg.num_attention_heads as u64 * hd;
    let kv_dim = cfg.num_key_value_heads as u64 * hd;
    let expert = 3 * h * mid;
    let top_k = cfg.num_experts_per_tok as u64;
    let bytes_e = prec.packed_bytes(1) as f64;
    let link = spec.connectx7_bytes_per_sec();
    let hop = 2.0e-6;

    // Batch-1 decode walks every layer sequentially. PP does not shrink that
    // HBM time: stages run one after another. EP/TP do: this rank only reads
    // its GEMM shards. Attention is *not* divided by EP.
    let attn_elems = layers * (h * q + h * kv_dim + h * kv_dim + q * h) / tp as u64;
    let expert_hits = top_k as f64 / ep as f64;
    let expert_elems = (layers as f64 * expert_hits * expert as f64 / tp as f64) as u64;
    let router_elems = layers * h * cfg.num_local_experts as u64;
    let norm_elems = layers * 2 * h + h;
    let lm_elems = if cfg.tie_word_embeddings { 0 } else { v * h / tp as u64 };
    let active_elems = attn_elems + expert_elems + router_elems + norm_elems + lm_elems;
    let active_bytes = prec.packed_bytes(active_elems).saturating_mul(batch as u64);

    let hbm_s = active_bytes as f64 / spec.memory_bandwidth_bps;
    let flops = 2.0 * active_elems as f64 * batch as f64;
    let compute_s = flops / spec.peak_flops(prec);

    // EP all-to-all: send hidden to top-k expert ranks and gather outputs.
    // Volume per token per layer = 2 * top_k * H * elem; remote (EP-1)/EP.
    // Token visits all layers (PP sequential).
    let a2a_elems_layer = 2 * top_k * h;
    let remote = if ep > 1 { (ep - 1) as f64 / ep as f64 } else { 0.0 };
    let a2a_bytes = prec.packed_bytes(a2a_elems_layer * layers);
    let a2a_remote = (a2a_bytes as f64 * remote) as u64;
    let a2a_s = if ep > 1 {
        layers as f64 * hop + a2a_remote as f64 * batch as f64 / link
    } else {
        0.0
    };

    // TP ring all-reduce of the residual stream. Megatron-style: one AR after
    // attention (row-parallel out-proj) and one after the MoE/MLP — two per
    // layer per token. Per-AR payload M = H * batch * elem.
    // Ring data per rank: 2*(TP-1)/TP * M; hops: 2*(TP-1).
    let ar_s = if tp > 1 {
        let m = h as f64 * batch as f64 * bytes_e;
        let vol = 2.0 * (tp - 1) as f64 / tp as f64 * m;
        let lat = 2.0 * (tp - 1) as f64 * hop;
        2.0 * layers as f64 * (lat + vol / link)
    } else {
        0.0
    };

    // PP P2P: one hidden activation send per stage boundary, sequential.
    let pp_s = if pp > 1 {
        let payload = prec.packed_bytes(h) as f64 * batch as f64;
        (pp - 1) as f64 * (hop + payload / link)
    } else {
        0.0
    };

    let step_s = hbm_s.max(compute_s) + a2a_s + ar_s + pp_s;
    Ok(DecodeCost {
        hbm_s,
        compute_s,
        a2a_s,
        ar_s,
        pp_s,
        step_s: step_s.max(1e-12),
        a2a_bytes: a2a_remote,
        active_weight_bytes: active_bytes,
    })
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

/// One-line pick rationale per plan, for the sweep report.
pub fn format_rationales(plans: &[MixtralPlan]) -> String {
    let mut s = String::from("chosen-plan rationale\n");
    s.push_str(&"-".repeat(110));
    s.push('\n');
    for p in plans {
        s.push_str(&format!(
            "{} N={} {}: EP/TP/PP={}/{}/{} — {}\n",
            p.model,
            p.n,
            p.precision.as_str(),
            p.ep,
            p.tp,
            p.pp,
            p.pick_rationale
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

    fn gib(b: u64) -> f64 {
        b as f64 / (1u64 << 30) as f64
    }

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
        assert!(format_rationales(&rows).contains("min decode"));
    }

    #[test]
    fn per_node_weight_bytes_shrink_with_ep_tp_pp() {
        let cfg = MixtralConfig::mixtral_8x7b();
        let prec = Precision::Bf16;
        let full = weight_elems_per_node(&cfg, 1, 1, 1).unwrap();
        let spec = DgxSparkSpec::official();

        // (EP, TP, PP) triples with product 8.
        for (ep, tp, pp) in [(1, 8, 1), (8, 1, 1), (2, 2, 2), (1, 1, 8), (2, 4, 1)] {
            let got = weight_elems_per_node(&cfg, ep, tp, pp).unwrap();
            let scale = (ep * tp * pp) as u64;
            assert_eq!(
                got.experts,
                full.experts / scale,
                "experts must shrink 1/(EP·TP·PP) (EP={ep} TP={tp} PP={pp})"
            );
            assert_eq!(
                got.attn,
                full.attn / (tp * pp) as u64,
                "attn shrinks 1/(TP·PP); EP replicas it (EP={ep} TP={tp} PP={pp})"
            );
            assert_eq!(got.embed, full.embed / tp as u64);
            assert_eq!(got.lm, full.lm / tp as u64);

            let mem = node_mem(&cfg, prec, ep, tp, pp, 4096, 1, &spec).unwrap();
            assert_eq!(mem.weights, prec.packed_bytes(got.total()));

            // Per-node weight bytes ≈ 1/(EP·TP·PP) of GEMMs, plus replicas
            // (embeddings, norms, router) and the EP-copied attention shard.
            let gemm_share = prec.packed_bytes(full.experts / scale + full.attn / (tp * pp) as u64);
            let replica = prec.packed_bytes(got.embed + got.lm + got.norms + got.router);
            let expect = gemm_share + replica;
            let slack = expect.abs_diff(mem.weights) as f64 / expect.max(1) as f64;
            assert!(
                slack < 0.02,
                "per-node weights {:.3} GiB vs 1/(EP·TP·PP) GEMMs + replicas {:.3} \
                 (EP={ep} TP={tp} PP={pp}, rel {slack})",
                gib(mem.weights),
                gib(expect)
            );
        }
    }

    #[test]
    fn chosen_plan_weight_bytes_match_breakdown() {
        let cfg = MixtralConfig::mixtral_8x7b();
        let p = plan_mixtral(&cfg, 8, Precision::Bf16, SparkTopology::Switch, 4096, 1).unwrap();
        let w = weight_elems_per_node(&cfg, p.ep, p.tp, p.pp).unwrap();
        assert_eq!(p.weight_bytes_per_node, Precision::Bf16.packed_bytes(w.total()));
        // With any N=8 factorization, per-node weights must be well below
        // the unsharded 8x7B bf16 footprint (~88 GiB of weights).
        let unsharded = plan_mixtral(&cfg, 1, Precision::Bf16, SparkTopology::Switch, 4096, 1)
            .unwrap();
        assert!(
            p.weight_bytes_per_node * 2 < unsharded.weight_bytes_per_node,
            "N=8 weights {:.1} GiB should be far below N=1 {:.1} GiB",
            gib(p.weight_bytes_per_node),
            gib(unsharded.weight_bytes_per_node)
        );
    }

    #[test]
    fn tp_allreduce_and_ep_all_to_all_are_modeled() {
        let cfg = MixtralConfig::mixtral_8x7b();
        let spec = DgxSparkSpec::official();
        let tp = decode_cost(&cfg, Precision::Bf16, 1, 8, 1, 1, &spec).unwrap();
        assert!(tp.ar_s > 0.0, "TP=8 must pay ring all-reduce");
        assert_eq!(tp.a2a_s, 0.0);
        // Two ARs / layer × 32 layers × 2·(8-1) hops × 2 µs = 1.792 ms of
        // latency alone, before payload.
        assert!(
            tp.ar_s > 1.5e-3,
            "TP=8 two-AR/layer latency floor, got {} s",
            tp.ar_s
        );

        let ep = decode_cost(&cfg, Precision::Bf16, 8, 1, 1, 1, &spec).unwrap();
        assert!(ep.a2a_s > 0.0, "EP=8 must pay all-to-all");
        assert_eq!(ep.ar_s, 0.0);

        let pp = decode_cost(&cfg, Precision::Bf16, 1, 1, 8, 1, &spec).unwrap();
        assert!(pp.pp_s > 0.0);
        assert_eq!(pp.ar_s, 0.0);
    }

    #[test]
    fn decode_is_not_free_n_times_single_node() {
        let cfg = MixtralConfig::mixtral_8x7b();
        let a = plan_mixtral(&cfg, 1, Precision::Bf16, SparkTopology::Switch, 4096, 1).unwrap();
        let b = plan_mixtral(&cfg, 8, Precision::Bf16, SparkTopology::Switch, 4096, 1).unwrap();
        assert!(a.feasible && b.feasible);
        // Perfect N× would require zero comm and a perfect 1/N HBM split.
        // TP-AR / EP-a2a are added, so tok/s must stay ≤ N × single-node.
        assert!(
            b.tokens_per_sec <= a.tokens_per_sec * b.n as f64 * 1.001,
            "N=8 tok/s {:.2} exceeded 8× N=1 {:.2}",
            b.tokens_per_sec,
            a.tokens_per_sec
        );
        if b.tp > 1 {
            assert!(b.tp_allreduce_seconds > 0.0);
        }
        if b.ep > 1 {
            assert!(b.all2all_seconds > 0.0);
        }
        // HBM time must drop if GEMMs are sharded; comm must be the reason
        // we are not exactly N×.
        assert!(b.hbm_seconds < a.hbm_seconds);
        assert!(b.step_seconds > b.hbm_seconds.max(b.compute_seconds));
    }

    #[test]
    fn search_picks_min_latency_among_feasible() {
        let cfg = MixtralConfig::mixtral_8x7b();
        let spec = DgxSparkSpec::official();
        let p = plan_mixtral(&cfg, 8, Precision::Bf16, SparkTopology::Switch, 4096, 1).unwrap();
        let all = enumerate_plans(&cfg, 8, Precision::Bf16, 4096, 1, &spec).unwrap();
        let usable = spec.usable_bytes();
        let feas: Vec<&Scored> = all.iter().filter(|s| s.mem.total <= usable).collect();
        let best = pick_min_latency(&feas).unwrap();
        assert_eq!((p.ep, p.tp, p.pp), (best.ep, best.tp, best.pp));
        assert!(p.pick_rationale.contains("min decode"));
        // First-feasible used to lock EP=1,TP=N. Whatever wins now must be
        // the scored minimum, not an iteration-order accident.
        for s in &feas {
            let close = {
                let m = s.dec.step_s.min(best.dec.step_s);
                m > 0.0 && (s.dec.step_s - best.dec.step_s).abs() / m <= EP_TIE_REL
            };
            if close {
                assert!(best.ep >= s.ep || best.dec.step_s < s.dec.step_s);
            } else {
                assert!(best.dec.step_s <= s.dec.step_s + 1e-15);
            }
        }
    }
}
