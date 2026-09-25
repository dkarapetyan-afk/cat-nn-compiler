//! Spec layer: Rust DSL + TOML/JSON/`.catnn` → the same IR.

pub mod ast;
pub mod dsl;
pub mod parse;

pub use ast::{ComposeStyle, LayerSpec, NetworkSpec};
pub use dsl::{mlp_spec, transformer_block_spec, NetworkBuilder};
pub use parse::{parse_auto, parse_catnn, parse_json, parse_toml};
