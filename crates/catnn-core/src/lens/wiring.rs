//! Wiring diagrams and structured cospans (spec §4, §6).
//!
//! A [`StructuredCospan`] is `L → Apex ← R`: left and right interfaces
//! include into an internal graph. Sequential composition is a pushout of
//! apexes along the shared interface (implemented as a port-identifying
//! graph merge — the finite-set pushout).

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

use crate::category::interface::Interface;
use crate::error::{CatError, CatResult};
use crate::lens::open::OpenComponent;

/// A directed wire between component ports.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Wire {
    pub src_box: String,
    pub src_port: String,
    pub dst_box: String,
    pub dst_port: String,
}

/// Apex of a structured cospan: a finite graph of open components.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Apex {
    pub name: String,
    pub boxes: IndexMap<String, OpenComponent>,
    pub wires: Vec<Wire>,
}

impl Apex {
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into(), boxes: IndexMap::new(), wires: vec![] }
    }

    pub fn add_box(&mut self, c: OpenComponent) {
        self.boxes.insert(c.name.clone(), c);
    }
}

/// Structured cospan `left → apex ← right`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StructuredCospan {
    pub left: Interface,
    pub right: Interface,
    pub apex: Apex,
    /// Inclusion of left ports into apex boxes: `(left_port, box, box_port)`.
    pub inbound: Vec<(String, String, String)>,
    /// Inclusion of apex boxes into right ports: `(box, box_port, right_port)`.
    pub outbound: Vec<(String, String, String)>,
}

impl StructuredCospan {
    pub fn apex_name(&self) -> &str {
        &self.apex.name
    }

    pub fn components(&self) -> Vec<&OpenComponent> {
        self.apex.boxes.values().collect()
    }

    /// Sequential composition of cospans: pushout of apexes along `self.right ≅ other.left`.
    pub fn compose(&self, other: &StructuredCospan) -> CatResult<StructuredCospan> {
        self.right.same_as(&other.left).map_err(|_| CatError::InterfaceMismatch {
            expected: self.right.to_string(),
            actual: other.left.to_string(),
        })?;

        let mut apex = Apex::new(format!("{}▷{}", self.apex.name, other.apex.name));
        for c in self.apex.boxes.values() {
            apex.add_box(c.clone());
        }
        for c in other.apex.boxes.values() {
            let mut c = c.clone();
            if apex.boxes.contains_key(&c.name) {
                c.name = format!("{}/{}", other.apex.name, c.name);
            }
            apex.add_box(c);
        }
        apex.wires.extend(self.apex.wires.iter().cloned());
        apex.wires.extend(other.apex.wires.iter().cloned());

        // Pushout: identify self.outbound ports with other.inbound ports
        // that share the same interface port name.
        for (sbox, sport, rport) in &self.outbound {
            for (lport, dbox, dport) in &other.inbound {
                if rport == lport {
                    apex.wires.push(Wire {
                        src_box: sbox.clone(),
                        src_port: sport.clone(),
                        dst_box: dbox.clone(),
                        dst_port: dport.clone(),
                    });
                }
            }
        }

        Ok(StructuredCospan {
            left: self.left.clone(),
            right: other.right.clone(),
            apex,
            inbound: self.inbound.clone(),
            outbound: other.outbound.clone(),
        })
    }

    /// Lift a sequential chain of components to a cospan (straight wiring).
    pub fn from_chain(name: impl Into<String>, parts: &[OpenComponent]) -> CatResult<Self> {
        if parts.is_empty() {
            return Err(CatError::Wiring("empty chain".into()));
        }
        let mut apex = Apex::new(name);
        for c in parts {
            apex.add_box(c.clone());
        }
        for pair in parts.windows(2) {
            let src = &pair[0];
            let dst = &pair[1];
            src.output.same_as(&dst.input).map_err(|_| CatError::InterfaceMismatch {
                expected: src.output.to_string(),
                actual: dst.input.to_string(),
            })?;
            for port in src.output.port_names() {
                apex.wires.push(Wire {
                    src_box: src.name.clone(),
                    src_port: port.to_string(),
                    dst_box: dst.name.clone(),
                    dst_port: port.to_string(),
                });
            }
        }
        let first = &parts[0];
        let last = parts.last().unwrap();
        let inbound = first
            .input
            .port_names()
            .map(|p| (p.to_string(), first.name.clone(), p.to_string()))
            .collect();
        let outbound = last
            .output
            .port_names()
            .map(|p| (last.name.clone(), p.to_string(), p.to_string()))
            .collect();
        Ok(StructuredCospan {
            left: first.input.clone(),
            right: last.output.clone(),
            apex,
            inbound,
            outbound,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lens::open::OpenComponent;

    #[test]
    fn cospan_compose_glues_on_shared_interface() {
        let fc1 = OpenComponent::linear("fc1", 2, 4, 8, true);
        let mid_out = fc1.output.clone();
        let act = OpenComponent::relu("act", mid_out);
        let fc2 = OpenComponent::linear("fc2", 2, 8, 2, true);
        let c1 = StructuredCospan::from_chain("c1", &[fc1]).unwrap();
        let c2 = StructuredCospan::from_chain("c2", &[act, fc2]).unwrap();
        let glued = c1.compose(&c2).unwrap();
        assert!(glued.apex.boxes.len() >= 3);
        assert!(!glued.apex.wires.is_empty());
    }
}
