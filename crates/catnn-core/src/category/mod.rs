//! Category-theoretic spine: `Tens` as an SMC, 1-morphisms, 2-cells.
//!
//! Mapping from the design document:
//! - objects of AISys / interfaces  → [`interface::Interface`]
//! - 1-morphisms (open components)  → [`morphism::Hom`]
//! - sequential / monoidal compose  → [`smc::Tens`]
//! - 2-morphisms (refinements)      → [`two_cell::TwoCell`]

pub mod interface;
pub mod laws;
pub mod morphism;
pub mod smc;
pub mod two_cell;

pub use interface::Interface;
pub use morphism::{Hom, HomBody, OpenOp};
pub use smc::{flatten_seq, Tens};
pub use two_cell::{EquivWitness, Rewrite, TwoCell};
