//! `catnn` CLI: compile, run, adapt, inspect.

use std::fs;
use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use catnn_compile::kernel::moe::{self, MixtralWeights};
use catnn_compile::spark_place::{self, format_rationales, format_table};
use catnn_compile::spark_train::{self, format_train_rationales, format_train_table};
use catnn_compile::train::{self, TrainOptions};
use catnn_compile::{compile, Compiled};
use catnn_core::architecture::{spark_cluster, Cluster, MemoryBudget, Precision, SparkTopology};
use catnn_core::lens::OptimizerKind;
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
        /// Structured budget: `64GiB` or `code=4MiB,buffer=64GiB,scratch=128KiB`.
        #[arg(long)]
        mem: Option<String>,
        #[arg(long)]
        mem_per_device: Option<String>,
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
        #[arg(long)]
        mem: Option<String>,
        #[arg(long)]
        mem_per_device: Option<String>,
    },
    /// One adaptation step of the unlawful optic.
    Adapt {
        spec: String,
        #[arg(long, default_value = "demo")]
        cluster: String,
        #[arg(long, default_value_t = 42)]
        seed: u64,
        #[arg(long)]
        mem: Option<String>,
        #[arg(long)]
        mem_per_device: Option<String>,
    },
    /// Print architecture, fibre of realizations, and (if compiled) placement.
    Inspect {
        spec: String,
        #[arg(long, default_value = "demo")]
        cluster: String,
        #[arg(long)]
        mem: Option<String>,
        #[arg(long)]
        mem_per_device: Option<String>,
    },
    /// CPU training loop (MLP or mixtral-tiny) under a memory budget.
    Train {
        spec: String,
        #[arg(long, default_value_t = 8)]
        steps: usize,
        #[arg(long, default_value = "sgd")]
        opt: String,
        #[arg(long, default_value_t = 0.05)]
        lr: f64,
        #[arg(long, default_value_t = 7)]
        seed: u64,
        #[arg(long)]
        mem: Option<String>,
        #[arg(long)]
        mem_per_device: Option<String>,
    },
    /// Place Mixtral *training* on N Sparks (bf16+fp32 Adam, optional fp8).
    MixtralTrainSweep {
        #[arg(long, default_value = "both")]
        model: String,
        #[arg(long, default_value_t = 2048)]
        seq: usize,
        #[arg(long, default_value_t = 8)]
        global_batch: usize,
        #[arg(long)]
        mem: Option<String>,
        #[arg(long)]
        mem_per_device: Option<String>,
        #[arg(long, default_value_t = false)]
        fp8: bool,
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
        Cmd::Compile { spec, cluster, output, mem, mem_per_device } => {
            cmd_compile(&spec, &cluster, output.as_ref(), mem.as_deref(), mem_per_device.as_deref())
        }
        Cmd::Run { spec, cluster, adapt, seed, mem, mem_per_device } => {
            cmd_run(&spec, &cluster, adapt, seed, mem.as_deref(), mem_per_device.as_deref())
        }
        Cmd::Adapt { spec, cluster, seed, mem, mem_per_device } => {
            cmd_run(&spec, &cluster, true, seed, mem.as_deref(), mem_per_device.as_deref())
        }
        Cmd::Inspect { spec, cluster, mem, mem_per_device } => {
            cmd_inspect(&spec, &cluster, mem.as_deref(), mem_per_device.as_deref())
        }
        Cmd::Train { spec, steps, opt, lr, seed, mem, mem_per_device } => {
            cmd_train(&spec, steps, &opt, lr, seed, mem.as_deref(), mem_per_device.as_deref())
        }
        Cmd::MixtralSweep { model, context, batch, weights } => {
            cmd_mixtral_sweep(&model, context, batch, weights.as_deref())
        }
        Cmd::MixtralTrainSweep { model, seq, global_batch, mem, mem_per_device, fp8 } => {
            cmd_mixtral_train_sweep(
                &model,
                seq,
                global_batch,
                mem.as_deref(),
                mem_per_device.as_deref(),
                fp8,
            )
        }
        Cmd::MixtralTiny { seed } => cmd_mixtral_tiny(seed),
    }
}

fn parse_mem_flags(mem: Option<&str>, mem_per_device: Option<&str>) -> Result<Option<MemoryBudget>> {
    if let Some(s) = mem.or(mem_per_device) {
        Ok(Some(MemoryBudget::parse(s)?))
    } else {
        Ok(None)
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

fn cmd_compile(
    spec: &str,
    cluster: &str,
    output: Option<&PathBuf>,
    mem: Option<&str>,
    mem_per_device: Option<&str>,
) -> Result<()> {
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
        println!(
            "  region peaks: code={:.1} KiB buffer={:.1} GiB scratch={} B",
            plan.peak_code as f64 / 1024.0,
            plan.peak_buffer as f64 / (1u64 << 30) as f64,
            plan.peak_scratch
        );
        if !plan.feasible {
            println!("INFEASIBLE — no bogus schedule emitted as executable.");
        }
    }
    let arch = load_arch(spec)?;
    let mut cluster = load_cluster(cluster)?;
    if let Some(b) = parse_mem_flags(mem, mem_per_device)? {
        cluster = cluster.with_budget(b);
        println!(
            "memory budget: code={:?} buffer={} scratch={:?}",
            b.code, b.buffer, b.scratch
        );
    }
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

fn cmd_run(
    spec: &str,
    cluster: &str,
    adapt: bool,
    seed: u64,
    mem: Option<&str>,
    mem_per_device: Option<&str>,
) -> Result<()> {
    if spec == "mixtral-tiny" {
        return cmd_mixtral_tiny(seed);
    }
    if spec == "mixtral" || spec == "mixtral-8x7b" || spec == "mixtral-8x22b" {
        bail!("full-size Mixtral cannot run on this VM (~93–280 GB weights). Use `catnn mixtral-tiny` or `catnn mixtral-sweep`.");
    }
    let arch = load_arch(spec)?;
    let mut cluster = load_cluster(cluster)?;
    if let Some(b) = parse_mem_flags(mem, mem_per_device)? {
        cluster = cluster.with_budget(b);
    }
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

fn cmd_inspect(
    spec: &str,
    cluster: &str,
    mem: Option<&str>,
    mem_per_device: Option<&str>,
) -> Result<()> {
    let arch = load_arch(spec)?;
    let mut cluster = load_cluster(cluster)?;
    if let Some(b) = parse_mem_flags(mem, mem_per_device)? {
        cluster = cluster.with_budget(b);
    }
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
    println!(
        "H memory: buffer {} B, code {} B, scratch {} B ({})",
        cluster.memory.regions.buffer.bytes,
        cluster.memory.regions.code.bytes,
        cluster.memory.regions.scratch.bytes,
        cluster.memory.regions.buffer.provenance
    );
    Ok(())
}

fn cmd_train(
    spec: &str,
    steps: usize,
    opt: &str,
    lr: f64,
    seed: u64,
    mem: Option<&str>,
    mem_per_device: Option<&str>,
) -> Result<()> {
    let opts = TrainOptions {
        steps,
        lr: lr as f32,
        opt: OptimizerKind::parse(opt)?,
        budget: parse_mem_flags(mem, mem_per_device)?,
        seed,
    };
    let report = if spec == "mixtral-tiny" || spec.contains("mixtral") {
        if spec.contains("8x7") || spec.contains("8x22") {
            bail!("full-size Mixtral training is planned, not executed. Use mixtral-tiny or mixtral-train-sweep.");
        }
        train::train_mixtral_tiny(&opts)?
    } else {
        train::train_mlp(&opts)?
    };
    println!("train {spec} opt={opt} steps={steps} lr={lr}");
    println!("losses: {:?}", report.losses);
    println!("final params checksum: {:016x}", report.final_params_checksum);
    println!(
        "schedule: feasible={} peaks code={} buffer={} scratch={} ({})",
        report.schedule.feasible,
        report.schedule.peaks.code,
        report.schedule.peaks.buffer,
        report.schedule.peaks.scratch,
        report.schedule.rationale
    );
    if !report.schedule.feasible {
        println!(
            "INFEASIBLE region: {:?}",
            report.schedule.overflow_region
        );
    }
    println!("executed: CPU training step (forward, loss, adjoint, optimizer).");
    println!("not executed: GPU kernels, full-size Mixtral, HF weights.");
    Ok(())
}

fn cmd_mixtral_train_sweep(
    model: &str,
    seq: usize,
    global_batch: usize,
    mem: Option<&str>,
    mem_per_device: Option<&str>,
    fp8: bool,
) -> Result<()> {
    let models: Vec<MixtralConfig> = match model {
        "8x7b" | "mixtral-8x7b" => vec![MixtralConfig::mixtral_8x7b()],
        "8x22b" | "mixtral-8x22b" => vec![MixtralConfig::mixtral_8x22b()],
        _ => vec![MixtralConfig::mixtral_8x7b(), MixtralConfig::mixtral_8x22b()],
    };
    let budget = parse_mem_flags(mem, mem_per_device)?;
    let rows = spark_train::sweep_train(&models, &[1, 2, 4, 8], seq, global_batch, budget, fp8)?;
    println!("DGX Spark Mixtral *training* sweep");
    println!("bf16 weights/grads + fp32 master + Adam m/v (16 B/param). Optional --fp8.");
    println!("seq={seq} global_batch={global_batch}. Usable buffer defaults to 104 GiB.");
    println!();
    print!("{}", format_train_table(&rows));
    println!();
    print!("{}", format_train_rationales(&rows));
    println!();
    if let Some(p) = rows.first() {
        println!("assumptions: {}", p.assumptions);
    }
    match train::mixtral_layer_timeline_excerpt() {
        Ok(ex) => print!("{ex}"),
        Err(e) => println!("timeline excerpt: {e}"),
    }
    println!("executed: placement search + simulated timeline (CPU).");
    println!("not executed: full-size training, GPU, nvcc cubin measurement.");
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
