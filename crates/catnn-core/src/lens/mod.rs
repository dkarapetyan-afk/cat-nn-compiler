//! Lenses, optics, open systems, wiring diagrams (spec §§1,4,7).
//!
//! - [`lawful::LawfulLens`] — PutGet / GetPut / PutPut, checked.
//! - [`optic::Optic`] — residual policy; laws are *not* claimed.
//! - [`open::OpenComponent`] — Moore/Mealy open dynamical systems.
//! - [`wiring::StructuredCospan`] — composition by pushout.

pub mod lawful;
pub mod moe;
pub mod open;
pub mod optic;
pub mod wiring;

pub use lawful::{first_of_pair, identity_f64, LawfulLens, LensSpec, PairState};
pub use moe::{ExpertFamily, GateView, MoEBlock, RouterResidual};
pub use open::{CompositeSystem, OpenComponent};
pub use optic::{Observation, Optic, PolicyState, UnlawfulWitness};
pub use wiring::{Apex, StructuredCospan, Wire};
