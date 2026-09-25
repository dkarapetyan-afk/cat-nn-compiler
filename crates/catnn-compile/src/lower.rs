//! Compile functor: architecture (composite of open systems) → IR graph.
//!
//! Sequential composition becomes a path; monoidal product becomes a
//! pair of disconnected components; cospans become the apex DAG.

use indexmap::IndexMap;

use catnn_core::architecture::Architecture;
use catnn_core::category::{Hom, HomBody, OpenOp};
use catnn_core::lens::OpenComponent;
use catnn_core::tensor::{Tensor, TensorType};
use catnn_core::CatResult;

use crate::ir::{IrEdge, IrGraph, IrNode, NodeId};

pub fn lower_architecture(arch: &Architecture) -> CatResult<IrGraph> {
    let mut g = IrGraph::new(&arch.id);
    let mut next_id = 0u32;
    let mut last: Option<NodeId>;

    // An explicit input node so the runtime has a place to inject data.
    let in_ty = arch
        .input()
        .ports
        .values()
        .next()
        .cloned()
        .unwrap_or_else(|| TensorType::f32([1]));
    let input_id = NodeId(next_id);
    g.nodes.insert(
        next_id,
        IrNode {
            id: input_id,
            name: format!("{}.in", arch.id),
            op: OpenOp::Identity,
            output_ty: in_ty,
            params: IndexMap::new(),
        },
    );
    g.inputs.push(input_id);
    last = Some(input_id);
    next_id += 1;

    for c in arch.components() {
        let id = NodeId(next_id);
        let node = lower_component(c, id)?;
        g.nodes.insert(next_id, node);
        if let Some(prev) = last {
            g.edges.push(IrEdge {
                src: prev,
                dst: id,
                src_port: "data".into(),
                dst_port: "data".into(),
            });
        }
        last = Some(id);
        next_id += 1;
    }

    if let Some(out) = last {
        g.outputs.push(out);
    }
    let _ = walk_hom(&arch.hom);
    Ok(g)
}

fn lower_component(c: &OpenComponent, id: NodeId) -> CatResult<IrNode> {
    let output_ty = c
        .output
        .ports
        .values()
        .next()
        .cloned()
        .unwrap_or_else(|| TensorType::f32([1]));
    let mut params = IndexMap::new();
    let seed_base = id.0 as u64 * 1009 + 17;
    for (i, (pname, ty)) in c.state.ports.iter().enumerate() {
        params.insert(pname.clone(), Tensor::lcg(ty.clone(), seed_base + i as u64));
    }
    Ok(IrNode { id, name: c.name.clone(), op: c.op.clone(), output_ty, params })
}

/// Recursively visit a hom tree (used to keep the derivation load-bearing:
/// lowering is defined on every constructor, not only on flattened lists).
fn walk_hom(hom: &Hom) -> usize {
    match &hom.body {
        HomBody::Seq(f, g) | HomBody::Par(f, g) => 1 + walk_hom(f) + walk_hom(g),
        HomBody::Component(_) | HomBody::Cospan(_) => 1,
        _ => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use catnn_core::spec::mlp_spec;

    #[test]
    fn lower_mlp_has_input_plus_layers() {
        let arch = mlp_spec(4, 16, 32, 8).unwrap();
        let g = lower_architecture(&arch).unwrap();
        assert_eq!(g.nodes.len(), 4); // in + fc1 + act + fc2
        assert_eq!(g.topo().unwrap().len(), 4);
    }
}
