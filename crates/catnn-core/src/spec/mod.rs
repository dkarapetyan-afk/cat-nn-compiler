//! Spec layer: Rust DSL + TOML/JSON/`.catnn` → the same IR.

pub mod ast;
pub mod deepseek;
pub mod dsl;
pub mod mixtral;
pub mod parse;

pub use ast::{ComposeStyle, LayerSpec, NetworkSpec};
pub use deepseek::{
    deepseek_connected, deepseek_mixture, deepseek_trunk, deepseek_v3_architecture, DeepSeekV3Config,
};
pub use dsl::{mlp_spec, transformer_block_spec, NetworkBuilder};
pub use mixtral::{mixtral_architecture, MixtralConfig};
pub use parse::{parse_auto, parse_catnn, parse_json, parse_toml};
