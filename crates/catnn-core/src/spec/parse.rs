//! Textual formats: TOML, JSON, and a small `.catnn` syntax.
//!
//! All three deserialize into [`crate::spec::ast::NetworkSpec`].

use crate::error::{CatError, CatResult};
use crate::spec::ast::{ComposeStyle, LayerSpec, NetworkSpec};

pub fn parse_toml(src: &str) -> CatResult<NetworkSpec> {
    toml::from_str(src).map_err(|e| CatError::Parse(e.to_string()))
}

pub fn parse_json(src: &str) -> CatResult<NetworkSpec> {
    serde_json::from_str(src).map_err(|e| CatError::Parse(e.to_string()))
}

/// Infer format from path suffix or, failing that, from the first non-ws char.
pub fn parse_auto(src: &str, path_hint: Option<&str>) -> CatResult<NetworkSpec> {
    if let Some(p) = path_hint {
        if p.ends_with(".toml") {
            return parse_toml(src);
        }
        if p.ends_with(".json") {
            return parse_json(src);
        }
        if p.ends_with(".catnn") {
            return parse_catnn(src);
        }
    }
    let trimmed = src.trim_start();
    if trimmed.starts_with('{') {
        parse_json(src)
    } else if trimmed.starts_with("network") || trimmed.starts_with("component") {
        parse_catnn(src)
    } else {
        parse_toml(src)
    }
}

/// Minimal custom syntax:
///
/// ```text
/// network mlp batch 4
/// linear fc1 16 32
/// relu act
/// linear fc2 32 8
/// compose sequential
/// ```
pub fn parse_catnn(src: &str) -> CatResult<NetworkSpec> {
    let mut spec = NetworkSpec::default();
    for (lineno, raw) in src.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') || line.starts_with("//") {
            continue;
        }
        let mut toks = line.split_whitespace();
        let kw = toks.next().ok_or_else(|| CatError::Parse(format!("line {lineno}: empty")))?;
        match kw {
            "network" => {
                spec.name = toks
                    .next()
                    .ok_or_else(|| CatError::Parse("network needs a name".into()))?
                    .to_string();
                if let Some("batch") = toks.next() {
                    spec.batch = toks
                        .next()
                        .and_then(|s| s.parse().ok())
                        .ok_or_else(|| CatError::Parse("batch needs an integer".into()))?;
                }
            }
            "linear" => {
                let name = need(toks.next(), "linear name")?;
                let din = parse_usize(toks.next(), "in_features")?;
                let dout = parse_usize(toks.next(), "out_features")?;
                spec.layers.push(LayerSpec::Linear {
                    name,
                    in_features: din,
                    out_features: dout,
                    bias: true,
                });
            }
            "relu" => spec.layers.push(LayerSpec::Relu { name: need(toks.next(), "relu name")? }),
            "gelu" => spec.layers.push(LayerSpec::Gelu { name: need(toks.next(), "gelu name")? }),
            "softmax" => spec
                .layers
                .push(LayerSpec::Softmax { name: need(toks.next(), "softmax name")?, axis: -1 }),
            "layernorm" => {
                let name = need(toks.next(), "layernorm name")?;
                let features = parse_usize(toks.next(), "features")?;
                spec.layers.push(LayerSpec::LayerNorm { name, features });
            }
            "attention" => {
                let name = need(toks.next(), "attention name")?;
                let seq = parse_usize(toks.next(), "seq")?;
                let d_model = parse_usize(toks.next(), "d_model")?;
                let n_heads = parse_usize(toks.next(), "n_heads")?;
                spec.layers.push(LayerSpec::Attention { name, seq, d_model, n_heads });
            }
            "compose" => {
                let style = toks.next().unwrap_or("sequential");
                spec.compose = match style {
                    "sequential" => ComposeStyle::Sequential,
                    "parallel" => ComposeStyle::Parallel,
                    other => return Err(CatError::Parse(format!("unknown compose {other}"))),
                };
            }
            other => {
                return Err(CatError::Parse(format!("line {lineno}: unknown keyword {other}")))
            }
        }
    }
    if spec.batch == 0 {
        spec.batch = 1;
    }
    Ok(spec)
}

fn need(tok: Option<&str>, what: &str) -> CatResult<String> {
    tok.map(|s| s.to_string()).ok_or_else(|| CatError::Parse(format!("missing {what}")))
}

fn parse_usize(tok: Option<&str>, what: &str) -> CatResult<usize> {
    let s = need(tok, what)?;
    s.parse().map_err(|_| CatError::Parse(format!("expected integer for {what}, got {s}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_catnn_mlp() {
        let src = r#"
network mlp batch 4
linear fc1 16 32
relu act
linear fc2 32 8
compose sequential
"#;
        let spec = parse_catnn(src).unwrap();
        assert_eq!(spec.layers.len(), 3);
        spec.to_architecture().unwrap();
    }

    #[test]
    fn parse_toml_mlp() {
        let src = r#"
name = "mlp"
batch = 4
compose = "sequential"

[[layers]]
op = "linear"
name = "fc1"
in_features = 16
out_features = 32

[[layers]]
op = "relu"
name = "act"

[[layers]]
op = "linear"
name = "fc2"
in_features = 32
out_features = 8
"#;
        let spec = parse_toml(src).unwrap();
        assert_eq!(spec.name, "mlp");
        spec.to_architecture().unwrap();
    }
}
