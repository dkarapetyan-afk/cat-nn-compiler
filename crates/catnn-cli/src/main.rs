//! `catnn` CLI: compile, run, adapt, inspect.

use std::fs;
use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use catnn_compile::kernel::moe::{self, MixtralWeights};
use catnn_compile::spark_place::{self, format_rationales, format_table};
use catnn_compile::{compile, Compiled};
use catnn_core::architecture::{spark_cluster, Cluster, Precision, SparkTopology};
use catnn_core::spec::{
    mixtral_architecture, mlp_spec, parse_auto, transformer_block_spec, MixtralConfig, NetworkSpec,
};
use catnn_core::tensor::Tensor;
use catnn_core::Architecture;
use catnn_rt::execute;
use clap::{Parser, Subcommand};

#[derive(Parser, Debug)]
#[command(name = "catnn", about = "Categorical compiler for neural networks")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// Lower a spec to an IR graph, place it, generate kernels.
    Compile {
        /// Path to .toml / .json / .catnn, or the built-in name `mlp` / `transformer`.
        spec: String,
        #[arg(long, default_value = "demo")]
        cluster: String,
        #[arg(short, long)]
        output: Option<PathBuf>,
    },
    /// Compile and execute on the cluster model. Checks against the reference.
    Run {
        spec: String,
        #[arg(long, default_value = "demo")]
        cluster: String,
        #[arg(long)]
        adapt: bool,
        #[arg(long, default_value_t = 42)]
        seed: u64,
    },
    /// One adaptation step of the unlawful optic.
    Adapt {
        spec: String,
        #[arg(long, default_value = "demo")]
        cluster: String,
        #[arg(long, default_value_t = 42)]
        seed: u64,
    },
    /// Print architecture, fibre of realizations, and (if compiled) placement.
    Inspect {
        spec: String,
        #[arg(long, default_value = "demo")]
        cluster: String,
    },
    /// Place Mixtral 8x7B / 8x22B on N DGX Sparks × {bf16,fp8,fp4}.
    MixtralSweep {
        #[arg(long, default_value = "both")]
        model: String,
        #[arg(long, default_value_t = 4096)]
        context: usize,
        #[arg(long, default_value_t = 1)]
        batch: usize,
        /// Optional safetensors path (not executed unless the file exists).
        #[arg(long)]
        weights: Option<PathBuf>,
    },
    /// Numerically check the scaled-down Mixtral against the naïve reference.
    MixtralTiny {
        #[arg(long, default_value_t = 7)]
        seed: u64,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Compile { spec, cluster, output } => cmd_compile(&spec, &cluster, output.as_ref()),
        Cmd::Run { spec, cluster, adapt, seed } => cmd_run(&spec, &cluster, adapt, seed),
        Cmd::Adapt { spec, cluster, seed } => cmd_run(&spec, &cluster, true, seed),
        Cmd::Inspect { spec, cluster } => cmd_inspect(&spec, &cluster),
        Cmd::MixtralSweep { model, context, batch, weights } => {
            cmd_mixtral_sweep(&model, context, batch, weights.as_deref())
        }
        Cmd::MixtralTiny { seed } => cmd_mixtral_tiny(seed),
    }
}

fn load_cluster(name: &str) -> Result<Cluster> {
    if name == "demo" {
        return Ok(Cluster::demo());
    }
    if name == "cpu" {
        return Ok(Cluster::single_cpu());
    }
    if let Some(rest) = name.strip_prefix("spark") {
        let n: usize = rest.parse().unwrap_or(1);
        let topo = if n <= 3 { SparkTopology::Direct } else { SparkTopology::Switch };
        return Ok(spark_cluster(n.max(1), topo)?);
    }
    let src = fs::read_to_string(name).with_context(|| format!("read cluster {name}"))?;
    Ok(toml::from_str(&src).or_else(|_| serde_json::from_str(&src))?)
}

fn load_arch(spec: &str) -> Result<Architecture> {
    match spec {
        "mlp" => Ok(mlp_spec(4, 16, 32, 8)?),
        "transformer" | "transformer_block" => Ok(transformer_block_spec(2, 8, 16, 4)?),
        "mixtral-tiny" => Ok(mixtral_architecture(&MixtralConfig::tiny(), 2, 4, true)?),
        "mixtral" | "mixtral-8x7b" => {
            Ok(mixtral_architecture(&MixtralConfig::mixtral_8x7b(), 1, 1, false)?)
        }
        "mixtral-8x22b" => Ok(mixtral_architecture(&MixtralConfig::mixtral_8x22b(), 1, 1, false)?),
        path => {
            let src = fs::read_to_string(path).with_context(|| format!("read spec {path}"))?;
            let parsed: NetworkSpec = parse_auto(&src, Some(path))?;
            Ok(parsed.to_architecture()?)
        }
    }
}

fn infer_input(arch: &Architecture, seed: u64) -> Result<Tensor> {
    let ty = arch
        .input()
        .ports
        .values()
        .next()
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("architecture has no input port"))?;
    Ok(Tensor::lcg(ty, seed))
}

fn cmd_compile(spec: &str, cluster: &str, output: Option<&PathBuf>) -> Result<()> {
    if spec.starts_with("mixtral") && cluster.starts_with("spark") {
        let n: usize = cluster.strip_prefix("spark").unwrap_or("1").parse().unwrap_or(1);
        let cfg = if spec.contains("22") {
            MixtralConfig::mixtral_8x22b()
        } else if spec.contains("tiny") {
            MixtralConfig::tiny()
        } else {
            MixtralConfig::mixtral_8x7b()
        };
        let topo = if n <= 3 { SparkTopology::Direct } else { SparkTopology::Switch };
        let plan = spark_place::plan_mixtral(&cfg, n.max(1), Precision::Bf16, topo, 4096, 1)?;
        println!("spark plan: {}", plan.reason);
        println!("  EP/TP/PP = {}/{}/{}", plan.ep, plan.tp, plan.pp);
        println!("  why: {}", plan.pick_rationale);
        println!("  {}", plan.placement_summary);
        println!("  CUDA sm_121 kernels: {} ({})", plan.cuda.len(), plan.nvcc_note);
        if !plan.feasible {
            println!("INFEASIBLE — no bogus schedule emitted as executable.");
        }
    }
    let arch = load_arch(spec)?;
    let cluster = load_cluster(cluster)?;
    let compiled = compile(&arch, &cluster)?;
    println!("architecture: {}", arch.id);
    println!("ir nodes: {}", compiled.graph.nodes.len());
    println!("realization: {}", compiled.realization.id);
    println!("placement:");
    for (op, dev) in &compiled.realization.placement.assign {
        println!("  {op:>16} → {dev}");
    }
    if !compiled.wgsl.is_empty() {
        println!("wgsl kernels generated and naga-validated: {}", compiled.wgsl.len());
        for (name, _) in &compiled.wgsl {
            println!("  {name}");
        }
    } else {
        println!("wgsl kernels: none (all ops placed on CPU)");
    }
    if let Some(path) = output {
        let json = serde_json::to_string_pretty(&InspectDump::from_compiled(&arch, &compiled))?;
        fs::write(path, json)?;
        println!("wrote {}", path.display());
    }
    Ok(())
}

fn cmd_mixtral_tiny(seed: u64) -> Result<()> {
    let cfg = MixtralConfig::tiny();
    let arch = mixtral_architecture(&cfg, 2, 4, true)?;
    println!("architecture: {} ({} 1-morphisms)", arch.id, arch.components().len());
    println!(
        "categorical MoE: experts are a coproduct of {} SwiGLU 1-morphisms; router is an optic (PutGet fails)",
        cfg.num_local_experts
    );
    let w = MixtralWeights::seeded(&cfg, seed)?;
    let tokens = [1u32, 2, 3, 4, 5, 6, 7, 8];
    let a = moe::forward(&cfg, &w, &tokens, 2, 4)?;
    let b = moe::reference::forward(&cfg, &w, &tokens, 2, 4)?;
    let err = a.max_abs_diff(&b)?;
    println!("output shape: {}", a.ty);
    println!("max_abs_error vs independent reference: {:.2e}", err);
    println!("executed: CPU Mixtral interpreter + naïve reference (seeded weights, not HF)");
    println!("not executed: safetensors / Hugging Face transformers / GPU");
    if err > 2e-4 {
        bail!("tiny Mixtral numerical check failed");
    }
    Ok(())
}

fn cmd_mixtral_sweep(
    model: &str,
    context: usize,
    batch: usize,
    weights: Option<&std::path::Path>,
) -> Result<()> {
    let models: Vec<MixtralConfig> = match model {
        "8x7b" | "mixtral-8x7b" => vec![MixtralConfig::mixtral_8x7b()],
        "8x22b" | "mixtral-8x22b" => vec![MixtralConfig::mixtral_8x22b()],
        _ => vec![MixtralConfig::mixtral_8x7b(), MixtralConfig::mixtral_8x22b()],
    };
    let ns = [1usize, 2, 4, 8];
    let precs = [Precision::Bf16, Precision::Fp8, Precision::Fp4];
    let rows = spark_place::sweep(&models, &ns, &precs, context, batch)?;
    println!("DGX Spark Mixtral placement sweep");
    println!("GB10: 128 GiB LPDDR5x @ 273 GB/s, ConnectX-7 200 Gb/s, sm_121, ≤1 PFLOP FP4.");
    println!("Official multi-node: Cluster Assistant 2–4 Sparks (direct 2–3, switch 2–4). N=8 is manual.");
    println!("Usable memory = 128 − 20 (OS) − 4 (workspace) GiB. context={context} batch={batch}.");
    println!();
    print!("{}", format_table(&rows));
    println!();
    print!("{}", format_rationales(&rows));
    println!();
    if let Some(p) = rows.first() {
        println!("tok/s assumptions: {}", p.tok_s_assumptions);
        println!(
            "CUDA: {} kernels/plan targeting {}. {}",
            p.cuda.len(),
            p.cuda[0].arch,
            p.nvcc_note
        );
    }
    if let Some(path) = weights {
        match moe::try_load_safetensors(path) {
            Ok(msg) => println!("weights: {msg}"),
            Err(e) => println!("weights: {e} (optional path; not executed in CI)"),
        }
    } else {
        println!("weights: no --weights given; full-size safetensors were not loaded.");
    }
    println!("executed: architecture build (shape-only for 8x7B/8x22B), placement, CUDA emit.");
    println!("not executed: full-size forward, GPU kernels, HF reference.");
    Ok(())
}

fn cmd_run(spec: &str, cluster: &str, adapt: bool, seed: u64) -> Result<()> {
    if spec == "mixtral-tiny" {
        return cmd_mixtral_tiny(seed);
    }
    if spec == "mixtral" || spec == "mixtral-8x7b" || spec == "mixtral-8x22b" {
        bail!("full-size Mixtral cannot run on this VM (~93–280 GB weights). Use `catnn mixtral-tiny` or `catnn mixtral-sweep`.");
    }
    let arch = load_arch(spec)?;
    let cluster = load_cluster(cluster)?;
    let compiled = compile(&arch, &cluster)?;
    let input = infer_input(&arch, seed)?;
    let report = execute(&compiled, &input, adapt)?;
    println!("architecture: {}", arch.id);
    println!("realization: {}", report.compiled.realization.id);
    println!("output shape: {}", report.output.ty);
    println!("output checksum: {:016x}", report.output.checksum());
    println!("latency_ms: {:.3}", report.observation.latency_ms);
    println!("bytes_moved: {}", report.observation.bytes_moved);
    println!("max_abs_error vs reference: {:.2e}", report.observation.max_abs_error);
    println!(
        "sheaf coherent: {} ({} events, {} issues)",
        report.history.is_coherent(),
        report.history.events.len(),
        report.history.issues.len()
    );
    for n in &report.observation.notes {
        println!("note: {n}");
    }
    if let Some(step) = &report.adapt {
        println!("adapt step: {}", step.cell.name);
        println!("  rewrite: {:?}", step.cell.rewrite);
        println!("  witness verified: {}", step.cell.is_verified());
        println!("  estimated/observed improvement: {}", step.improved);
        println!("  optic is unlawful ({})", step.unlawful.law_that_fails);
        println!("  policy.step: {}", step.policy.step);
        println!("  residual: {}", step.policy.residual_note);
        println!("  latency {:.3}ms → {:.3}ms", step.before.latency_ms, step.after.latency_ms);
    }
    if report.observation.max_abs_error > 1e-3 {
        bail!("numerical check failed");
    }
    Ok(())
}

fn cmd_inspect(spec: &str, cluster: &str) -> Result<()> {
    let arch = load_arch(spec)?;
    let cluster = load_cluster(cluster)?;
    println!("== architecture {} ==", arch.id);
    println!("input:  {}", arch.input());
    println!("output: {}", arch.output());
    println!("state:  {}", arch.system.state);
    println!("components (1-morphisms):");
    for c in arch.components() {
        println!("  {}: {}  {} → {}  (state {})", c.name, c.op.name(), c.input, c.output, c.state);
    }
    println!("hom tree: {}", arch.hom.name);
    let fibre = catnn_core::enumerate_realizations(&arch, &cluster);
    println!("== fibre R({}) : {} realizations ==", arch.id, fibre.objects.len());
    for r in &fibre.objects {
        println!("  {} on cluster {}", r.id, r.cluster);
        for (op, dev) in &r.placement.assign {
            println!("      {op} @ {dev} ({:?})", r.backend_of(op));
        }
    }
    let compiled = compile(&arch, &cluster)?;
    println!("== compiled placement ({}) ==", compiled.realization.id);
    for (op, dev) in &compiled.realization.placement.assign {
        println!("  {op} → {dev}");
    }
    println!("fusions: {:?}", compiled.realization.fusions);
    println!("generated WGSL: {}", compiled.wgsl.len());
    Ok(())
}

#[derive(serde::Serialize)]
struct InspectDump {
    architecture: String,
    realization: String,
    placement: Vec<(String, String)>,
    wgsl_ops: Vec<String>,
    ir_nodes: Vec<String>,
}

impl InspectDump {
    fn from_compiled(arch: &Architecture, compiled: &Compiled) -> Self {
        Self {
            architecture: arch.id.clone(),
            realization: compiled.realization.id.clone(),
            placement: compiled
                .realization
                .placement
                .assign
                .iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect(),
            wgsl_ops: compiled.wgsl.iter().map(|(n, _)| n.clone()).collect(),
            ir_nodes: compiled.graph.nodes.values().map(|n| n.name.clone()).collect(),
        }
    }
}
