//! Rust embedded DSL: typed builders that *are* open systems.
//!
//! Composition methods return `CatResult` so a shape error is a failed
//! morphism, not a later compiler surprise.

use crate::architecture::Architecture;
use crate::error::CatResult;
use crate::lens::open::{CompositeSystem, OpenComponent};
use crate::spec::ast::{ComposeStyle, LayerSpec, NetworkSpec};

/// Fluent builder that records a [`NetworkSpec`] *and* the assembled
/// [`CompositeSystem`]. Both surfaces stay in sync.
pub struct NetworkBuilder {
    spec: NetworkSpec,
    system: Option<CompositeSystem>,
}

impl NetworkBuilder {
    pub fn new(name: impl Into<String>, batch: usize) -> Self {
        Self {
            spec: NetworkSpec {
                name: name.into(),
                batch,
                layers: vec![],
                compose: ComposeStyle::Sequential,
            },
            system: None,
        }
    }

    pub fn linear(mut self, name: impl Into<String>, din: usize, dout: usize) -> CatResult<Self> {
        let name = name.into();
        let c = match &self.system {
            Some(s) => OpenComponent::linear_on(&name, s.output.clone(), din, dout, true),
            None => OpenComponent::linear(&name, self.spec.batch.max(1), din, dout, true),
        };
        self.spec.layers.push(LayerSpec::Linear {
            name,
            in_features: din,
            out_features: dout,
            bias: true,
        });
        self.push(c)?;
        Ok(self)
    }

    pub fn relu(mut self, name: impl Into<String>) -> CatResult<Self> {
        let name = name.into();
        let input = self
            .system
            .as_ref()
            .map(|s| s.output.clone())
            .ok_or_else(|| crate::error::CatError::Parse("relu with empty prefix".into()))?;
        let c = OpenComponent::relu(&name, input);
        self.spec.layers.push(LayerSpec::Relu { name });
        self.push(c)?;
        Ok(self)
    }

    pub fn gelu(mut self, name: impl Into<String>) -> CatResult<Self> {
        let name = name.into();
        let input = self
            .system
            .as_ref()
            .map(|s| s.output.clone())
            .ok_or_else(|| crate::error::CatError::Parse("gelu with empty prefix".into()))?;
        let c = OpenComponent::gelu(&name, input);
        self.spec.layers.push(LayerSpec::Gelu { name });
        self.push(c)?;
        Ok(self)
    }

    pub fn softmax(mut self, name: impl Into<String>) -> CatResult<Self> {
        let name = name.into();
        let input = self
            .system
            .as_ref()
            .map(|s| s.output.clone())
            .ok_or_else(|| crate::error::CatError::Parse("softmax with empty prefix".into()))?;
        let c = OpenComponent::softmax(&name, input, -1);
        self.spec.layers.push(LayerSpec::Softmax { name, axis: -1 });
        self.push(c)?;
        Ok(self)
    }

    pub fn layer_norm(mut self, name: impl Into<String>, features: usize) -> CatResult<Self> {
        let name = name.into();
        let input = self
            .system
            .as_ref()
            .map(|s| s.output.clone())
            .ok_or_else(|| crate::error::CatError::Parse("layernorm with empty prefix".into()))?;
        let c = OpenComponent::layer_norm(&name, input, features);
        self.spec.layers.push(LayerSpec::LayerNorm { name, features });
        self.push(c)?;
        Ok(self)
    }

    pub fn attention(
        mut self,
        name: impl Into<String>,
        seq: usize,
        d_model: usize,
        n_heads: usize,
    ) -> CatResult<Self> {
        let name = name.into();
        let c = OpenComponent::attention(&name, self.spec.batch.max(1), seq, d_model, n_heads)?;
        self.spec.layers.push(LayerSpec::Attention { name, seq, d_model, n_heads });
        self.push(c)?;
        Ok(self)
    }

    fn push(&mut self, c: OpenComponent) -> CatResult<()> {
        self.system = Some(match self.system.take() {
            None => CompositeSystem::singleton(c),
            Some(s) => s.then(&c)?,
        });
        Ok(())
    }

    pub fn spec(&self) -> &NetworkSpec {
        &self.spec
    }

    pub fn finish(self) -> CatResult<Architecture> {
        self.spec.to_architecture()
    }
}

/// Stock examples used by the CLI and tests.
pub fn mlp_spec(batch: usize, din: usize, hidden: usize, dout: usize) -> CatResult<Architecture> {
    NetworkBuilder::new("mlp", batch)
        .linear("fc1", din, hidden)?
        .relu("act")?
        .linear("fc2", hidden, dout)?
        .finish()
}

pub fn transformer_block_spec(
    batch: usize,
    seq: usize,
    d_model: usize,
    n_heads: usize,
) -> CatResult<Architecture> {
    // Sequential stand-in for a block: LN → MHA → LN → Linear → GELU → Linear.
    // Residual adds are representable as components but need a two-input
    // wiring; the compiler inserts skip-add IR nodes when lowering this
    // named pattern. The architecture itself is the sequential backbone.
    NetworkBuilder::new("transformer_block", batch)
        .attention("mha", seq, d_model, n_heads)?
        .layer_norm("ln1", d_model)?
        .linear("ff1", d_model, d_model * 2)?
        .gelu("ff_act")?
        .linear("ff2", d_model * 2, d_model)?
        .layer_norm("ln2", d_model)?
        .finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dsl_rejects_bad_linear_chain() {
        let err = NetworkBuilder::new("bad", 2)
            .linear("a", 4, 8)
            .unwrap()
            .linear("b", 3, 3);
        assert!(err.is_err());
    }

    #[test]
    fn dsl_mlp_and_transformer() {
        assert_eq!(mlp_spec(4, 16, 32, 8).unwrap().components().len(), 3);
        assert_eq!(transformer_block_spec(2, 8, 16, 4).unwrap().components().len(), 6);
    }
}
