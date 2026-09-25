//! Classical lenses with checkable PutGet / GetPut / PutPut (spec §1).
//!
//! Higher coherences (Capriotti–Danielsson–Vezzosi) are truncated: we store
//! the three 1-laws as predicates on a finite sample of states/views. A
//! [`LawfulLens`] that fails the predicates is rejected; the optimizer is
//! *not* a lawful lens (see [`crate::lens::optic`]).

use serde::{Deserialize, Serialize};

use crate::error::{CatError, CatResult};

/// A classical lens `S ⇄ V` with explicit get/put.
///
/// The function fields are not serializable; [`LensSpec`] is the
/// description that is. Tests reconstruct the functions from the spec.
#[derive(Clone)]
pub struct LawfulLens<S, V> {
    pub name: String,
    pub get: fn(&S) -> V,
    pub put: fn(&S, &V) -> S,
}

impl<S, V> LawfulLens<S, V> {
    pub fn new(name: impl Into<String>, get: fn(&S) -> V, put: fn(&S, &V) -> S) -> Self {
        Self { name: name.into(), get, put }
    }

    pub fn view(&self, s: &S) -> V {
        (self.get)(s)
    }

    pub fn update(&self, s: &S, v: &V) -> S {
        (self.put)(s, v)
    }
}

impl<S, V> LawfulLens<S, V>
where
    S: PartialEq,
    V: PartialEq,
{
    /// PutGet: `get(put(s, v)) == v`.
    pub fn law_put_get(&self, s: &S, v: &V) -> CatResult<()> {
        let s2 = self.update(s, v);
        let v2 = self.view(&s2);
        if v2 == *v {
            Ok(())
        } else {
            Err(CatError::Unlawful(format!("{}: PutGet failed", self.name)))
        }
    }

    /// GetPut: `put(s, get(s)) == s`.
    pub fn law_get_put(&self, s: &S) -> CatResult<()> {
        let v = self.view(s);
        let s2 = self.update(s, &v);
        if s2 == *s {
            Ok(())
        } else {
            Err(CatError::Unlawful(format!("{}: GetPut failed", self.name)))
        }
    }

    /// PutPut: `put(put(s, v1), v2) == put(s, v2)`.
    pub fn law_put_put(&self, s: &S, v1: &V, v2: &V) -> CatResult<()> {
        let left = self.update(&self.update(s, v1), v2);
        let right = self.update(s, v2);
        if left == right {
            Ok(())
        } else {
            Err(CatError::Unlawful(format!("{}: PutPut failed", self.name)))
        }
    }

    pub fn check_laws(&self, s: &S, v1: &V, v2: &V) -> CatResult<()> {
        self.law_put_get(s, v1)?;
        self.law_get_put(s)?;
        self.law_put_put(s, v1, v2)?;
        Ok(())
    }
}

/// Serialisable description of a stock lens used in tests and the adaptive loop.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum LensSpec {
    /// Identity lens on a named field of a record (used for unit tests).
    Identity { name: String },
    /// Project a single `f64` field out of a two-field record.
    FirstOfPair,
}

/// Concrete state used to exercise lawful lenses in property tests.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PairState {
    pub first: f64,
    pub second: f64,
}

pub fn first_of_pair() -> LawfulLens<PairState, f64> {
    LawfulLens::new("first_of_pair", |s| s.first, |s, v| PairState { first: *v, second: s.second })
}

pub fn identity_f64() -> LawfulLens<f64, f64> {
    LawfulLens::new("id_f64", |s| *s, |_s, v| *v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_of_pair_satisfies_laws() {
        let lens = first_of_pair();
        let s = PairState { first: 1.0, second: 2.0 };
        lens.check_laws(&s, &3.0, &4.0).unwrap();
    }

    #[test]
    fn identity_satisfies_laws() {
        identity_f64().check_laws(&1.5, &2.5, &0.0).unwrap();
    }
}
