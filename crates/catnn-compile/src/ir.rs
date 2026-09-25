//! Compiler IR: a DAG of tensor operations obtained by flattening a `Hom`.
//!
//! This is the image of the architecture under the compile functor
//! `A → Graph`. Edges carry tensor types; nodes carry [`OpenOp`]s.

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

use catnn_core::category::OpenOp;
use catnn_core::tensor::{Tensor, TensorType};
use catnn_core::CatError;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NodeId(pub u32);

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IrNode {
    pub id: NodeId,
    pub name: String,
    pub op: OpenOp,
    pub output_ty: TensorType,
    /// Parameter tensors keyed by state-port name (weights).
    pub params: IndexMap<String, Tensor>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IrEdge {
    pub src: NodeId,
    pub dst: NodeId,
    pub src_port: String,
    pub dst_port: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IrGraph {
    pub name: String,
    pub nodes: IndexMap<u32, IrNode>,
    pub edges: Vec<IrEdge>,
    pub inputs: Vec<NodeId>,
    pub outputs: Vec<NodeId>,
}

impl IrGraph {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            nodes: IndexMap::new(),
            edges: vec![],
            inputs: vec![],
            outputs: vec![],
        }
    }

    pub fn node(&self, id: NodeId) -> Option<&IrNode> {
        self.nodes.get(&id.0)
    }

    pub fn preds(&self, id: NodeId) -> Vec<NodeId> {
        self.edges.iter().filter(|e| e.dst == id).map(|e| e.src).collect()
    }

    pub fn succs(&self, id: NodeId) -> Vec<NodeId> {
        self.edges.iter().filter(|e| e.src == id).map(|e| e.dst).collect()
    }

    /// Kahn topo-sort. Errors on cycles.
    pub fn topo(&self) -> Result<Vec<NodeId>, CatError> {
        let mut indeg: IndexMap<u32, usize> = self.nodes.keys().map(|k| (*k, 0usize)).collect();
        for e in &self.edges {
            *indeg.get_mut(&e.dst.0).unwrap() += 1;
        }
        let mut q: Vec<u32> = indeg.iter().filter(|(_, d)| **d == 0).map(|(k, _)| *k).collect();
        q.sort_unstable();
        let mut out = Vec::new();
        while let Some(n) = q.pop() {
            out.push(NodeId(n));
            let mut nexts: Vec<u32> =
                self.edges.iter().filter(|e| e.src.0 == n).map(|e| e.dst.0).collect();
            nexts.sort_unstable();
            for s in nexts {
                let d = indeg.get_mut(&s).unwrap();
                *d -= 1;
                if *d == 0 {
                    q.push(s);
                }
            }
        }
        if out.len() != self.nodes.len() {
            return Err(CatError::Wiring("IR graph has a cycle".into()));
        }
        Ok(out)
    }

    pub fn bytes_of(&self, id: NodeId) -> u64 {
        self.node(id).map(|n| n.output_ty.bytes() as u64).unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use catnn_core::tensor::TensorType;

    #[test]
    fn topo_linear_chain() {
        let mut g = IrGraph::new("t");
        g.nodes.insert(
            0,
            IrNode {
                id: NodeId(0),
                name: "a".into(),
                op: OpenOp::Identity,
                output_ty: TensorType::f32([2]),
                params: IndexMap::new(),
            },
        );
        g.nodes.insert(
            1,
            IrNode {
                id: NodeId(1),
                name: "b".into(),
                op: OpenOp::Relu,
                output_ty: TensorType::f32([2]),
                params: IndexMap::new(),
            },
        );
        g.edges.push(IrEdge {
            src: NodeId(0),
            dst: NodeId(1),
            src_port: "data".into(),
            dst_port: "data".into(),
        });
        assert_eq!(g.topo().unwrap(), vec![NodeId(0), NodeId(1)]);
    }
}
