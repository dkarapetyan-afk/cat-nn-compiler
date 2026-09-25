use catnn_compile::compile;
use catnn_core::architecture::Cluster;
use catnn_core::spec::{mlp_spec, parse_auto, transformer_block_spec};
use catnn_core::tensor::{Tensor, TensorType};
use catnn_rt::{adapt_once, execute};

#[test]
fn mlp_from_catnn_file_runs() {
    let src = include_str!("../../../examples/mlp.catnn");
    let spec = parse_auto(src, Some("mlp.catnn")).unwrap();
    let arch = spec.to_architecture().unwrap();
    let compiled = compile(&arch, &Cluster::demo()).unwrap();
    let x = Tensor::lcg(TensorType::f32([4, 16]), 1);
    let report = execute(&compiled, &x, false).unwrap();
    assert!(report.observation.max_abs_error < 1e-4);
    assert!(report.history.is_coherent());
}

#[test]
fn transformer_matches_reference() {
    let arch = transformer_block_spec(2, 8, 16, 4).unwrap();
    let compiled = compile(&arch, &Cluster::demo()).unwrap();
    let x = Tensor::lcg(TensorType::f32([2, 8, 16]), 11);
    let report = execute(&compiled, &x, false).unwrap();
    assert!(report.observation.max_abs_error < 1e-4, "{}", report.observation.max_abs_error);
    assert!(report.history.is_coherent());
}

#[test]
fn adapt_mlp_keeps_equivalence() {
    let arch = mlp_spec(4, 16, 32, 8).unwrap();
    let compiled = compile(&arch, &Cluster::demo()).unwrap();
    let x = Tensor::lcg(TensorType::f32([4, 16]), 5);
    let (next, step, _) =
        adapt_once(&compiled, &x, &catnn_core::lens::PolicyState::default()).unwrap();
    assert!(step.cell.witness.observed_max_abs.unwrap() < 1e-3);
    assert_eq!(next.graph.nodes.len(), compiled.graph.nodes.len());
}
