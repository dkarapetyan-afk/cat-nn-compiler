//! `catnn` CLI: compile, run, adapt, inspect.

use std::fs;
use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use catnn_compile::{compile, Compiled};
use catnn_core::architecture::Cluster;
use catnn_core::spec::{mlp_spec, parse_auto, transformer_block_spec, NetworkSpec};
use catnn_core::tensor::Tensor;
use catnn_core::Architecture;
use catnn_rt::execute;

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
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Compile { spec, cluster, output } => cmd_compile(&spec, &cluster, output.as_ref()),
        Cmd::Run { spec, cluster, adapt, seed } => cmd_run(&spec, &cluster, adapt, seed),
        Cmd::Adapt { spec, cluster, seed } => cmd_run(&spec, &cluster, true, seed),
        Cmd::Inspect { spec, cluster } => cmd_inspect(&spec, &cluster),
    }
}

fn load_cluster(name: &str) -> Result<Cluster> {
    if name == "demo" {
        return Ok(Cluster::demo());
    }
    if name == "cpu" {
        return Ok(Cluster::single_cpu());
    }
    let src = fs::read_to_string(name).with_context(|| format!("read cluster {name}"))?;
    Ok(toml::from_str(&src).or_else(|_| serde_json::from_str(&src))?)
}

fn load_arch(spec: &str) -> Result<Architecture> {
    match spec {
        "mlp" => Ok(mlp_spec(4, 16, 32, 8)?),
        "transformer" | "transformer_block" => Ok(transformer_block_spec(2, 8, 16, 4)?),
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

fn cmd_run(spec: &str, cluster: &str, adapt: bool, seed: u64) -> Result<()> {
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
        println!(
            "  latency {:.3}ms → {:.3}ms",
            step.before.latency_ms, step.after.latency_ms
        );
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
        println!(
            "  {}: {}  {} → {}  (state {})",
            c.name,
            c.op.name(),
            c.input,
            c.output,
            c.state
        );
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
