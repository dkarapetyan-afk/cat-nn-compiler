//! Property tests for the SMC laws, exported so `catnn-core` tests share them.

use crate::category::interface::Interface;
use crate::category::smc::{flatten_seq, Tens};
use crate::lens::open::OpenComponent;
use crate::tensor::TensorType;

pub fn typed_id_chain(width: usize, names: &[&str]) -> Vec<crate::category::morphism::Hom> {
    names
        .iter()
        .map(|n| {
            OpenComponent::identity(*n, Interface::single(*n, TensorType::f32([width]))).as_hom()
        })
        .collect()
}

pub fn associativity_holds(width: usize) -> bool {
    let homs = typed_id_chain(width, &["f", "g", "h"]);
    let f = &homs[0];
    let g = &homs[1];
    let h = &homs[2];
    let left = Tens::compose(&Tens::compose(f, g).unwrap(), h).unwrap();
    let right = Tens::compose(f, &Tens::compose(g, h).unwrap()).unwrap();
    flatten_seq(&left) == flatten_seq(&right)
}

pub fn unit_laws_hold(width: usize) -> bool {
    let f = &typed_id_chain(width, &["f"])[0];
    let id_a = Tens::id(&f.dom);
    let id_b = Tens::id(&f.cod);
    let left = Tens::compose(&id_a, f).unwrap();
    let right = Tens::compose(f, &id_b).unwrap();
    flatten_seq(&left) == flatten_seq(f) && flatten_seq(&right) == flatten_seq(f)
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn compose_associative(width in 1usize..16) {
            prop_assert!(associativity_holds(width));
        }

        #[test]
        fn compose_units(width in 1usize..16) {
            prop_assert!(unit_laws_hold(width));
        }
    }

    #[test]
    fn monoidal_unit_object_is_empty() {
        assert!(Tens::unit_object().is_unit());
    }
}
