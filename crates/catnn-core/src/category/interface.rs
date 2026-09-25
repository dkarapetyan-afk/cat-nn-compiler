//! Objects of the tensor SMC: finite products of named tensor ports.
//!
//! An [`Interface`] is a *system boundary* in the sense of AISys (spec §3):
//! the domain or codomain of an open component. The monoidal product is
//! disjoint union of ports (names are namespaced on collision).

use std::fmt;

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

use crate::error::{CatError, CatResult};
use crate::tensor::TensorType;

/// A named port on an interface.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Port {
    pub name: String,
    pub ty: TensorType,
}

impl Port {
    pub fn new(name: impl Into<String>, ty: TensorType) -> Self {
        Self { name: name.into(), ty }
    }
}

/// An object of `Tens`. Order of ports is part of the object (strict product).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Interface {
    pub name: String,
    pub ports: IndexMap<String, TensorType>,
}

impl Interface {
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into(), ports: IndexMap::new() }
    }

    pub fn unit() -> Self {
        Self::new("I")
    }

    pub fn is_unit(&self) -> bool {
        self.ports.is_empty()
    }

    pub fn port(mut self, name: impl Into<String>, ty: TensorType) -> Self {
        self.ports.insert(name.into(), ty);
        self
    }

    pub fn single(name: impl Into<String>, ty: TensorType) -> Self {
        let name = name.into();
        Self::new(name.clone()).port("data", ty)
    }

    pub fn get(&self, port: &str) -> CatResult<&TensorType> {
        self.ports.get(port).ok_or_else(|| CatError::UnknownPort {
            interface: self.name.clone(),
            port: port.to_string(),
        })
    }

    pub fn same_as(&self, other: &Interface) -> CatResult<()> {
        if self.ports.len() != other.ports.len() {
            return Err(CatError::InterfaceMismatch {
                expected: self.to_string(),
                actual: other.to_string(),
            });
        }
        for (name, ty) in &self.ports {
            let other_ty = other.ports.get(name).ok_or_else(|| CatError::InterfaceMismatch {
                expected: self.to_string(),
                actual: other.to_string(),
            })?;
            ty.same_as(other_ty).map_err(|_| CatError::InterfaceMismatch {
                expected: format!("{}@{ty}", name),
                actual: format!("{}@{other_ty}", name),
            })?;
        }
        Ok(())
    }

    /// Monoidal product `A ⊗ B`. Port names from `B` are prefixed with `b.`
    /// when they collide with `A`.
    pub fn tensor(&self, other: &Interface) -> Interface {
        let mut ports = self.ports.clone();
        for (name, ty) in &other.ports {
            let key = if ports.contains_key(name) {
                format!("{}.{}", other.name, name)
            } else {
                name.clone()
            };
            ports.insert(key, ty.clone());
        }
        let name = if self.is_unit() {
            other.name.clone()
        } else if other.is_unit() {
            self.name.clone()
        } else {
            format!("{}⊗{}", self.name, other.name)
        };
        Interface { name, ports }
    }

    pub fn bytes(&self) -> usize {
        self.ports.values().map(|t| t.bytes()).sum()
    }

    pub fn port_names(&self) -> impl Iterator<Item = &str> {
        self.ports.keys().map(|s| s.as_str())
    }
}

impl fmt::Display for Interface {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}{{", self.name)?;
        for (i, (n, ty)) in self.ports.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{n}: {ty}")?;
        }
        write!(f, "}}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::TensorType;

    #[test]
    fn tensor_product_renames_collisions() {
        let a = Interface::single("A", TensorType::f32([2, 2]));
        let b = Interface::single("B", TensorType::f32([2, 2]));
        let p = a.tensor(&b);
        assert!(p.ports.contains_key("data"));
        assert!(p.ports.contains_key("B.data") || p.ports.len() == 2);
    }

    #[test]
    fn unit_is_identity_for_tensor() {
        let a = Interface::single("A", TensorType::f32([3]));
        let u = Interface::unit();
        assert_eq!(a.tensor(&u).ports, a.ports);
        assert_eq!(u.tensor(&a).ports, a.ports);
    }
}
