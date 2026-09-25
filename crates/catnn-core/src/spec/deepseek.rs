//! DeepSeek-V3 as a diagram, and composites of those diagrams.
//!
//! Hyperparameters follow the public `deepseek-ai/DeepSeek-V3` config:
//! MLA (q/kv LoRA, decoupled RoPE), three dense SwiGLU layers, then
//! sigmoid-routed MoE with one shared expert. A second copy is not a
//! special runner: [`Architecture::connect`] sequences two trunks, and
//! [`Architecture::mixture`] runs two models on the same input.

use crate::architecture::Architecture;
use crate::category::morphism::OpenOp;
use crate::error::{CatError, CatResult};
use crate::lens::moe::hidden_interface;
use crate::lens::open::OpenComponent;
use crate::tensor::{DType, TensorType};
use crate::category::interface::Interface;

#[derive(Debug, Clone, PartialEq)]
pub struct DeepSeekV3Config {
    pub name: String,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub moe_intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub q_lora_rank: usize,
    pub kv_lora_rank: usize,
    pub qk_nope_head_dim: usize,
    pub qk_rope_head_dim: usize,
    pub v_head_dim: usize,
    pub n_routed_experts: usize,
    pub n_shared_experts: usize,
    pub num_experts_per_tok: usize,
    pub n_group: usize,
    pub topk_group: usize,
    pub first_k_dense_replace: usize,
    pub routed_scaling_factor: f32,
    pub rms_norm_eps: f32,
    pub rope_theta: f64,
}

impl DeepSeekV3Config {
    /// Public DeepSeek-V3 widths. Weights are not materialized.
    pub fn v3() -> Self {
        Self {
            name: "deepseek-v3".into(),
            vocab_size: 129_280,
            hidden_size: 7168,
            intermediate_size: 18432,
            moe_intermediate_size: 2048,
            num_hidden_layers: 61,
            num_attention_heads: 128,
            q_lora_rank: 1536,
            kv_lora_rank: 512,
            qk_nope_head_dim: 128,
            qk_rope_head_dim: 64,
            v_head_dim: 128,
            n_routed_experts: 256,
            n_shared_experts: 1,
            num_experts_per_tok: 8,
            n_group: 8,
            topk_group: 4,
            first_k_dense_replace: 3,
            routed_scaling_factor: 2.5,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
        }
    }

    /// Same generators, small enough to execute on CPU.
    pub fn tiny() -> Self {
        Self {
            name: "deepseek-v3-tiny".into(),
            vocab_size: 32,
            hidden_size: 32,
            intermediate_size: 48,
            moe_intermediate_size: 16,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            q_lora_rank: 8,
            kv_lora_rank: 8,
            qk_nope_head_dim: 4,
            qk_rope_head_dim: 4,
            v_head_dim: 4,
            n_routed_experts: 4,
            n_shared_experts: 1,
            num_experts_per_tok: 2,
            n_group: 2,
            topk_group: 1,
            first_k_dense_replace: 1,
            routed_scaling_factor: 2.5,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
        }
    }

    pub fn param_count(&self) -> CatResult<u64> {
        self.validate()?;
        let h = self.hidden_size as u64;
        let heads = self.num_attention_heads as u64;
        let q_lora = self.q_lora_rank as u64;
        let kv_lora = self.kv_lora_rank as u64;
        let nope = self.qk_nope_head_dim as u64;
        let rope = self.qk_rope_head_dim as u64;
        let v = self.v_head_dim as u64;
        let q_head = nope + rope;
        let attn = h * q_lora
            + q_lora
            + q_lora * heads * q_head
            + h * kv_lora
            + kv_lora
            + kv_lora * heads * nope
            + kv_lora * heads * v
            + h * rope
            + heads * v * h;
        let dense_layers = self.first_k_dense_replace as u64;
        let moe_layers = (self.num_hidden_layers as u64).saturating_sub(dense_layers);
        let swiglu = |mid: u64| 3 * h * mid;
        let dense = dense_layers * swiglu(self.intermediate_size as u64);
        let expert = swiglu(self.moe_intermediate_size as u64);
        let routed = moe_layers * self.n_routed_experts as u64 * expert;
        let shared = moe_layers * self.n_shared_experts as u64 * expert;
        let gate = moe_layers * h * self.n_routed_experts as u64;
        let embed = self.vocab_size as u64 * h;
        let layers = self.num_hidden_layers as u64;
        // Two RMSNorms per layer plus the final norm.
        let norms = (layers * 2 + 1) * h;
        Ok(embed * 2 + attn * layers + dense + routed + shared + gate + norms)
    }

    pub fn validate(&self) -> CatResult<()> {
        if self.n_group == 0 || self.n_routed_experts % self.n_group != 0 {
            return Err(CatError::ShapeMismatch("routed experts must divide into groups".into()));
        }
        if self.topk_group == 0 || self.topk_group > self.n_group {
            return Err(CatError::ShapeMismatch("topk_group out of range".into()));
        }
        if self.num_experts_per_tok == 0
            || self.num_experts_per_tok > self.n_routed_experts
        {
            return Err(CatError::ShapeMismatch("top-k out of range".into()));
        }
        if self.first_k_dense_replace > self.num_hidden_layers {
            return Err(CatError::ShapeMismatch("dense prefix longer than the stack".into()));
        }
        Ok(())
    }

    fn mla_op(&self) -> OpenOp {
        OpenOp::Mla {
            n_heads: self.num_attention_heads,
            q_lora: self.q_lora_rank,
            kv_lora: self.kv_lora_rank,
            qk_nope: self.qk_nope_head_dim,
            qk_rope: self.qk_rope_head_dim,
            v_dim: self.v_head_dim,
            rope_theta: self.rope_theta as u64,
        }
    }

    fn moe_op(&self) -> OpenOp {
        OpenOp::DeepSeekMoe {
            n_routed: self.n_routed_experts,
            n_shared: self.n_shared_experts,
            top_k: self.num_experts_per_tok,
            n_group: self.n_group,
            topk_group: self.topk_group,
            hidden: self.hidden_size,
            moe_intermediate: self.moe_intermediate_size,
            routed_scale_milli: (self.routed_scaling_factor * 1000.0).round() as u32,
        }
    }
}

fn rms(name: &str, h: usize, eps: f32, input: Interface) -> OpenComponent {
    let op = OpenOp::RmsNorm { eps_e6: (eps * 1e6) as u32 };
    let mut output = input.clone();
    output.name = format!("{name}.Y");
    OpenComponent {
        name: name.into(),
        state: op.state_interface(name, Some(h)),
        input,
        output,
        op,
        moore: false,
    }
}

fn swiglu_op(name: &str, h: usize, mid: usize, input: Interface) -> OpenComponent {
    let op = OpenOp::SwiGlu { hidden: h, intermediate: mid };
    OpenComponent {
        name: name.into(),
        state: op.state_interface(name, Some(h)),
        input,
        output: hidden_interface(&format!("{name}.Y"), 0, 0, h),
        op,
        moore: false,
    }
}

/// Hidden → hidden trunk (no embed, no LM head). Two of these connect.
pub fn deepseek_trunk(
    cfg: &DeepSeekV3Config,
    batch: usize,
    seq: usize,
    materialize_params: bool,
) -> CatResult<Architecture> {
    cfg.validate()?;
    let h = cfg.hidden_size;
    let mut cur = hidden_interface("h", batch, seq, h);
    let mut parts = Vec::new();
    for layer in 0..cfg.num_hidden_layers {
        let rms1 = rms(&format!("L{layer}.rms_attn"), h, cfg.rms_norm_eps, cur.clone());
        cur = rms1.output.clone();
        parts.push(rms1);
        let op = cfg.mla_op();
        let mla = OpenComponent {
            name: format!("L{layer}.mla"),
            state: op.state_interface(&format!("L{layer}.mla"), Some(h)),
            input: cur.clone(),
            output: hidden_interface(&format!("L{layer}.mla.Y"), batch, seq, h),
            op,
            moore: false,
        };
        cur = mla.output.clone();
        parts.push(mla);
        let rms2 = rms(&format!("L{layer}.rms_ffn"), h, cfg.rms_norm_eps, cur.clone());
        cur = rms2.output.clone();
        parts.push(rms2);
        let ffn = if layer < cfg.first_k_dense_replace {
            let mut c = swiglu_op(
                &format!("L{layer}.dense"),
                h,
                cfg.intermediate_size,
                cur.clone(),
            );
            c.output = hidden_interface(&format!("L{layer}.dense.Y"), batch, seq, h);
            c
        } else {
            let op = cfg.moe_op();
            OpenComponent {
                name: format!("L{layer}.moe"),
                state: op.state_interface(&format!("L{layer}.moe"), Some(h)),
                input: cur.clone(),
                output: hidden_interface(&format!("L{layer}.moe.Y"), batch, seq, h),
                op,
                moore: false,
            }
        };
        cur = ffn.output.clone();
        parts.push(ffn);
    }
    let mut arch = Architecture::from_components(format!("{}-trunk", cfg.name), parts)?;
    arch.materialize_params = materialize_params;
    Ok(arch)
}

/// Tokens → logits, one DeepSeek-V3.
pub fn deepseek_v3_architecture(
    cfg: &DeepSeekV3Config,
    batch: usize,
    seq: usize,
    materialize_params: bool,
) -> CatResult<Architecture> {
    cfg.validate()?;
    let h = cfg.hidden_size;
    let tokens = Interface::single("tokens", TensorType::new([batch, seq], DType::I32));
    let embed_op = OpenOp::Embedding { vocab: cfg.vocab_size, hidden: h };
    let embed = OpenComponent {
        name: "embed".into(),
        state: embed_op.state_interface("embed", Some(h)),
        input: tokens,
        output: hidden_interface("embed.Y", batch, seq, h),
        op: embed_op,
        moore: false,
    };
    let trunk = deepseek_trunk(cfg, batch, seq, materialize_params)?;
    let mut parts = vec![embed];
    parts.extend(trunk.components().iter().cloned());
    let mut cur = parts.last().unwrap().output.clone();
    let final_norm = rms("final_norm", h, cfg.rms_norm_eps, cur);
    cur = final_norm.output.clone();
    parts.push(final_norm);
    let lm_op = OpenOp::LmHead { hidden: h, vocab: cfg.vocab_size };
    parts.push(OpenComponent {
        name: "lm_head".into(),
        state: lm_op.state_interface("lm_head", Some(h)),
        input: cur,
        output: Interface::single("logits", TensorType::f32([batch, seq, cfg.vocab_size])),
        op: lm_op,
        moore: false,
    });
    let mut arch = Architecture::from_components(&cfg.name, parts)?;
    arch.materialize_params = materialize_params;
    Ok(arch)
}

/// One trunk wired into another. The composite is still `Hidden → Hidden`.
pub fn deepseek_connected(
    cfg: &DeepSeekV3Config,
    batch: usize,
    seq: usize,
    materialize_params: bool,
) -> CatResult<Architecture> {
    let a = deepseek_trunk(cfg, batch, seq, materialize_params)?;
    let b = deepseek_trunk(cfg, batch, seq, materialize_params)?;
    a.connect(&b)
}

/// Two full models on the same tokens; a scalar gate mixes the logits.
pub fn deepseek_mixture(
    cfg: &DeepSeekV3Config,
    batch: usize,
    seq: usize,
    materialize_params: bool,
) -> CatResult<Architecture> {
    let a = deepseek_v3_architecture(cfg, batch, seq, materialize_params)?;
    let b = deepseek_v3_architecture(cfg, batch, seq, materialize_params)?;
    a.mixture(&b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn v3_param_count_is_about_671b() {
        let n = DeepSeekV3Config::v3().param_count().unwrap();
        assert!(n > 660_000_000_000 && n < 690_000_000_000, "got {n}");
    }

    #[test]
    fn connected_trunks_typecheck_and_full_models_do_not_chain_on_logits() {
        let cfg = DeepSeekV3Config::tiny();
        let connected = deepseek_connected(&cfg, 2, 4, true).unwrap();
        let one = deepseek_trunk(&cfg, 2, 4, true).unwrap();
        assert!(connected.components().len() >= one.components().len() * 2);
        let model = deepseek_v3_architecture(&cfg, 2, 4, true).unwrap();
        assert!(model.connect(&model).is_err());
    }

    #[test]
    fn mixture_of_two_models_is_one_diagram() {
        let cfg = DeepSeekV3Config::tiny();
        let one = deepseek_v3_architecture(&cfg, 2, 4, true).unwrap();
        let mix = deepseek_mixture(&cfg, 2, 4, true).unwrap();
        assert!(mix.components().len() > one.components().len() * 2);
        assert_eq!(mix.input().ports["data"].shape.dims, vec![2, 4]);
        assert_eq!(mix.output().ports["data"].shape.dims, vec![2, 4, 32]);
    }
}
