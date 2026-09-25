//! Adaptive loop (spec §§7, 8, 10).
//!
//! `A --compile--> R --run--> τ --observe--> O --optimize--> A'`
//!
//! The optimizer is an [`catnn_core::lens::Optic`] with residual
//! [`PolicyState`], not a lawful lens. It actually mutates the realization:
//! fuse a linear+relu on the same device, or move the slowest op onto the
//! fastest device, when the estimated cost improves.

use catnn_compile::place::{estimate, propose_fusions};
use catnn_compile::Compiled;
use catnn_core::architecture::{DeviceKind, KernelBackend, KernelChoice};
use catnn_core::category::{EquivWitness, Rewrite, TwoCell};
use catnn_core::lens::{Observation, Optic, PolicyState, UnlawfulWitness};
use catnn_core::CatResult;

use crate::scheduler::{run_with_reference, RunResult};

/// One step of the adaptive loop, recorded as data.
#[derive(Debug, Clone)]
pub struct AdaptStep {
    pub before: Observation,
    pub after: Observation,
    pub cell: TwoCell,
    pub policy: PolicyState,
    pub improved: bool,
    pub unlawful: UnlawfulWitness,
}

pub fn adapt_optic() -> Optic<Compiled, Observation, PolicyState> {
    Optic::new("adapt_optic", |compiled, _p| {
        Observation {
            latency_ms: 0.0,
            bytes_moved: 0,
            kernel_count: compiled.graph.nodes.len(),
            device_occupancy: vec![],
            max_abs_error: 0.0,
            notes: vec!["view-only; run to observe".into()],
        }
    }, |compiled, obs, policy| {
        let mut next = compiled.clone();
        let mut p = policy.clone();
        p.step += 1;
        p.last_latency_ms = obs.latency_ms;
        p.last_max_abs = obs.max_abs_error;
        p.residual_note = format!(
            "step {}: last latency {:.3}ms; PutGet does not hold",
            p.step, obs.latency_ms
        );

        // 1. Fuse linear+relu when they share a device.
        let fusions = propose_fusions(&next.graph, &next.realization.placement);
        if !fusions.is_empty() && next.realization.fusions.is_empty() {
            next.realization.fusions = fusions;
        }

        // 2. Move the last compute-heavy op onto the fastest device if that
        //    reduces the estimated cost.
        if let Some((_, fastest)) = next
            .cluster
            .devices()
            .max_by(|a, b| a.1.gflops.partial_cmp(&b.1.gflops).unwrap())
        {
            let heavy = next
                .graph
                .nodes
                .values()
                .rev()
                .find(|n| matches!(n.op, catnn_core::category::OpenOp::Linear { .. }
                    | catnn_core::category::OpenOp::Attention { .. }))
                .map(|n| n.name.clone());
            if let Some(op) = heavy {
                let before = estimate(&next.graph, &next.realization.placement, &next.cluster);
                let old = next.realization.device_of(&op).unwrap_or("").to_string();
                next.realization.placement.put(&op, &fastest.id);
                let after = estimate(&next.graph, &next.realization.placement, &next.cluster);
                if after.total() * p.replace_threshold >= before.total() {
                    // Revert if not enough estimated gain.
                    next.realization.placement.put(&op, old);
                } else if fastest.kind == DeviceKind::Gpu {
                    if let Some(k) = next.realization.kernels.iter_mut().find(|k| k.op == op) {
                        k.backend = KernelBackend::Wgsl;
                    } else {
                        next.realization.kernels.push(KernelChoice {
                            op,
                            backend: KernelBackend::Wgsl,
                        });
                    }
                }
            }
        }
        (next, p)
    })
}

/// Compile → run → observe → update, once. Returns the new compiled
/// programme and the recorded step.
pub fn adapt_once(
    compiled: &Compiled,
    input: &catnn_core::tensor::Tensor,
    policy: &PolicyState,
) -> CatResult<(Compiled, AdaptStep, RunResult)> {
    let before_run = run_with_reference(compiled, input)?;
    let optic = adapt_optic();
    let (next, new_policy) = optic.step(compiled, &before_run.observation, policy);
    let after_run = run_with_reference(&next, input)?;

    let improved = after_run.observation.latency_ms < before_run.observation.latency_ms
        || next.realization.fusions.len() > compiled.realization.fusions.len()
        || estimate(&next.graph, &next.realization.placement, &next.cluster).total()
            < estimate(&compiled.graph, &compiled.realization.placement, &compiled.cluster).total();

    let mut cell = TwoCell {
        name: format!("adapt#{}", new_policy.step),
        source_name: compiled.realization.id.clone(),
        target_name: next.realization.id.clone(),
        rewrite: if next.realization.fusions.len() > compiled.realization.fusions.len() {
            Rewrite::Fuse {
                ops: next
                    .realization
                    .fusions
                    .first()
                    .map(|g| g.ops.clone())
                    .unwrap_or_default(),
            }
        } else {
            Rewrite::Reassign {
                op: "heavy".into(),
                from_device: "prev".into(),
                to_device: "fastest".into(),
            }
        },
        witness: EquivWitness::relaxed(),
    };
    cell.witness.accept(after_run.observation.max_abs_error).ok();

    let step = AdaptStep {
        before: before_run.observation.clone(),
        after: after_run.observation.clone(),
        cell,
        policy: new_policy,
        improved,
        unlawful: UnlawfulWitness::optimizer(),
    };
    Ok((next, step, after_run))
}

#[cfg(test)]
mod tests {
    use super::*;
    use catnn_compile::compile;
    use catnn_core::architecture::Cluster;
    use catnn_core::spec::mlp_spec;
    use catnn_core::tensor::{Tensor, TensorType};

    #[test]
    fn one_adapt_step_preserves_numerics() {
        let arch = mlp_spec(4, 16, 32, 8).unwrap();
        let compiled = compile(&arch, &Cluster::demo()).unwrap();
        let x = Tensor::lcg(TensorType::f32([4, 16]), 3);
        let (_next, step, _run) = adapt_once(&compiled, &x, &PolicyState::default()).unwrap();
        assert!(step.after.max_abs_error < 1e-3);
        assert_eq!(step.unlawful.law_that_fails, "PutGet");
    }
}
