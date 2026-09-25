//! Strict-ish symmetric monoidal category of tensor interfaces.
//!
//! The mathematical object is `Tens`: objects are [`Interface`]s, morphisms
//! are [`Hom`]s. We truncate ∞-coherence to *checkable* unit / associator /
//! braiding morphisms (spec §9: "∞-coherence truncated to 2-cells").
//!
//! Sequential composition `g ∘ f` is defined only when `cod(f) ≅ dom(g)`
//! by portwise shape/dtype equality. That check is the type system.

use serde::{Deserialize, Serialize};

use crate::category::interface::Interface;
use crate::category::morphism::{Hom, HomBody};
use crate::error::{CatError, CatResult};

/// The tensor SMC as a concrete category (one object collection, one hom collection).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Tens;

impl Tens {
    pub fn id(obj: &Interface) -> Hom {
        Hom {
            name: format!("id_{}", obj.name),
            dom: obj.clone(),
            cod: obj.clone(),
            body: HomBody::Id,
        }
    }

    /// Sequential composition `g ∘ f` (apply `f` then `g`).
    pub fn compose(f: &Hom, g: &Hom) -> CatResult<Hom> {
        f.cod.same_as(&g.dom).map_err(|_| CatError::InterfaceMismatch {
            expected: f.cod.to_string(),
            actual: g.dom.to_string(),
        })?;
        Ok(Hom {
            name: format!("({}∘{})", g.name, f.name),
            dom: f.dom.clone(),
            cod: g.cod.clone(),
            body: HomBody::Seq(Box::new(f.clone()), Box::new(g.clone())),
        })
    }

    /// Monoidal product of morphisms `f ⊗ g : A⊗C → B⊗D`.
    pub fn tensor(f: &Hom, g: &Hom) -> Hom {
        Hom {
            name: format!("({}⊗{})", f.name, g.name),
            dom: f.dom.tensor(&g.dom),
            cod: f.cod.tensor(&g.cod),
            body: HomBody::Par(Box::new(f.clone()), Box::new(g.clone())),
        }
    }

    pub fn unit_object() -> Interface {
        Interface::unit()
    }

    /// Associator `α : (A⊗B)⊗C → A⊗(B⊗C)`. Strict on ports: we rebuild the
    /// product in the opposite association. The *witness* that these are
    /// equal as objects is a 2-cell checked by [`crate::category::two_cell`].
    pub fn associator(a: &Interface, b: &Interface, c: &Interface) -> Hom {
        let left = a.tensor(b).tensor(c);
        let right = a.tensor(&b.tensor(c));
        Hom {
            name: format!("α[{}⊗{}⊗{}]", a.name, b.name, c.name),
            dom: left,
            cod: right,
            body: HomBody::Associator,
        }
    }

    pub fn left_unitor(a: &Interface) -> Hom {
        Hom {
            name: format!("λ[{}]", a.name),
            dom: Self::unit_object().tensor(a),
            cod: a.clone(),
            body: HomBody::LeftUnitor,
        }
    }

    pub fn right_unitor(a: &Interface) -> Hom {
        Hom {
            name: format!("ρ[{}]", a.name),
            dom: a.tensor(&Self::unit_object()),
            cod: a.clone(),
            body: HomBody::RightUnitor,
        }
    }

    /// Braiding `σ : A⊗B → B⊗A`.
    pub fn braiding(a: &Interface, b: &Interface) -> Hom {
        Hom {
            name: format!("σ[{}⊗{}]", a.name, b.name),
            dom: a.tensor(b),
            cod: b.tensor(a),
            body: HomBody::Braid,
        }
    }

    /// Fold a list with `compose`. Empty list is not an identity (no object).
    pub fn compose_all(homs: &[Hom]) -> CatResult<Hom> {
        let mut iter = homs.iter();
        let first = iter
            .next()
            .ok_or_else(|| CatError::Wiring("compose_all on empty list".into()))?;
        let mut acc = first.clone();
        for h in iter {
            acc = Self::compose(&acc, h)?;
        }
        Ok(acc)
    }
}

/// Flatten a hom tree into the sequence of atomic operations (left-to-right
/// application order). Used as a denotation for associativity tests: the
/// associator of composition is the identity on this flattening.
pub fn flatten_seq(hom: &Hom) -> Vec<String> {
    match &hom.body {
        HomBody::Seq(f, g) => {
            let mut v = flatten_seq(f);
            v.extend(flatten_seq(g));
            v
        }
        HomBody::Par(f, g) => {
            vec![format!("par[{}|{}]", flatten_seq(f).join(";"), flatten_seq(g).join(";"))]
        }
        HomBody::Id => vec![format!("id:{}", hom.dom.name)],
        HomBody::Associator => vec!["α".into()],
        HomBody::LeftUnitor => vec!["λ".into()],
        HomBody::RightUnitor => vec!["ρ".into()],
        HomBody::Braid => vec!["σ".into()],
        HomBody::Component(c) => vec![c.name.clone()],
        HomBody::Cospan(c) => vec![format!("cospan:{}", c.apex_name())],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::category::morphism::OpenOp;
    use crate::lens::open::OpenComponent;
    use crate::tensor::TensorType;

    fn dummy(name: &str, n: usize) -> Hom {
        let iface = Interface::single(name, TensorType::f32([n]));
        OpenComponent::identity(name, iface).as_hom()
    }

    #[test]
    fn compose_associative_on_flattening() {
        let f = dummy("f", 4);
        let g = dummy("g", 4);
        let h = dummy("h", 4);
        let left = Tens::compose(&Tens::compose(&f, &g).unwrap(), &h).unwrap();
        let right = Tens::compose(&f, &Tens::compose(&g, &h).unwrap()).unwrap();
        assert_eq!(flatten_seq(&left), flatten_seq(&right));
    }

    #[test]
    fn compose_rejects_type_mismatch() {
        let f = dummy("f", 4);
        let g = OpenComponent::identity("g", Interface::single("g", TensorType::f32([8]))).as_hom();
        assert!(Tens::compose(&f, &g).is_err());
    }

    #[test]
    fn unit_laws_on_flattening() {
        let f = dummy("f", 3);
        let id_a = Tens::id(&f.dom);
        let id_b = Tens::id(&f.cod);
        let left = Tens::compose(&id_a, &f).unwrap();
        let right = Tens::compose(&f, &id_b).unwrap();
        assert_eq!(flatten_seq(&left), flatten_seq(&f));
        assert_eq!(flatten_seq(&right), flatten_seq(&f));
    }

    #[test]
    fn identity_component_exists() {
        let _ = OpenOp::Identity;
    }
}
