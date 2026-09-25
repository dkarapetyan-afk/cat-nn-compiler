//! Mixtral architecture as a typed composite of open systems.
//!
//! Hyperparameters are copied from the official Hugging Face `config.json`
//! files (retrieved 2026-09-25):
//!
//! - [`MixtralConfig::mixtral_8x7b`]: `mistralai/Mixtral-8x7B-v0.1`
//!   <https://huggingface.co/mistralai/Mixtral-8x7B-v0.1/raw/main/config.json>
//! - [`MixtralConfig::mixtral_8x22b`]: `mistralai/Mixtral-8x22B-v0.1`
//!   <https://huggingface.co/mistralai/Mixtral-8x22B-v0.1/raw/main/config.json>
//!
//! The official 8x7B / 8x22B configs set `sliding_window: null`. The
//! Mixtral paper (Jiang et al. 2024, arXiv:2401.04088) inherits Mistral 7B's
//! 4096-token sliding window; we store that separately and implement the
//! mechanism so a non-null window is respected when present.

use serde::{Deserialize, Serialize};

use crate::architecture::Architecture;
use crate::category::interface::Interface;
use crate::category::morphism::OpenOp;
use crate::error::{CatError, CatResult};
use crate::lens::moe::{hidden_interface, MoEBlock};
use crate::lens::open::OpenComponent;
use crate::tensor::{DType, TensorType};

/// Mixtral / Mixtral-like decoder config. Fields match HF `MixtralConfig`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MixtralConfig {
    pub name: String,
    pub hf_id: String,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub num_local_experts: usize,
    pub num_experts_per_tok: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f64,
    /// Official HF field; `None` for both released Mixtral configs.
    pub sliding_window: Option<usize>,
    /// Paper / Mistral-7B heritage (4096). Not used unless `sliding_window` is set.
    pub paper_sliding_window: Option<usize>,
    pub hidden_act: String,
    pub tie_word_embeddings: bool,
    pub torch_dtype: String,
    pub bos_token_id: u32,
    pub eos_token_id: u32,
}

impl MixtralConfig {
    /// `mistralai/Mixtral-8x7B-v0.1` `config.json` (HF, `transformers_version` 4.36.0.dev0).
    pub fn mixtral_8x7b() -> Self {
        Self {
            name: "mixtral-8x7b-v0.1".into(),
            hf_id: "mistralai/Mixtral-8x7B-v0.1".into(),
            vocab_size: 32000,
            hidden_size: 4096,
            intermediate_size: 14336,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: 8,
            num_local_experts: 8,
            num_experts_per_tok: 2,
            max_position_embeddings: 32768,
            rms_norm_eps: 1e-5,
            rope_theta: 1_000_000.0,
            sliding_window: None,
            paper_sliding_window: Some(4096),
            hidden_act: "silu".into(),
            tie_word_embeddings: false,
            torch_dtype: "bfloat16".into(),
            bos_token_id: 1,
            eos_token_id: 2,
        }
    }

    /// `mistralai/Mixtral-8x22B-v0.1` `config.json` (HF, `transformers_version` 4.38.0).
    pub fn mixtral_8x22b() -> Self {
        Self {
            name: "mixtral-8x22b-v0.1".into(),
            hf_id: "mistralai/Mixtral-8x22B-v0.1".into(),
            vocab_size: 32000,
            hidden_size: 6144,
            intermediate_size: 16384,
            num_hidden_layers: 56,
            num_attention_heads: 48,
            num_key_value_heads: 8,
            num_local_experts: 8,
            num_experts_per_tok: 2,
            max_position_embeddings: 65536,
            rms_norm_eps: 1e-5,
            rope_theta: 1_000_000.0,
            sliding_window: None,
            paper_sliding_window: Some(4096),
            hidden_act: "silu".into(),
            tie_word_embeddings: false,
            torch_dtype: "bfloat16".into(),
            bos_token_id: 1,
            eos_token_id: 2,
        }
    }

    /// Same architecture, small enough to execute on this VM.
    pub fn tiny() -> Self {
        Self {
            name: "mixtral-tiny".into(),
            hf_id: "catnn/mixtral-tiny (architecture-faithful scale-down)".into(),
            vocab_size: 32,
            hidden_size: 32,
            intermediate_size: 64,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            num_local_experts: 4,
            num_experts_per_tok: 2,
            max_position_embeddings: 32,
            rms_norm_eps: 1e-5,
            rope_theta: 10_000.0,
            sliding_window: Some(8),
            paper_sliding_window: Some(8),
            hidden_act: "silu".into(),
            tie_word_embeddings: false,
            torch_dtype: "float32".into(),
            bos_token_id: 1,
            eos_token_id: 2,
        }
    }

    pub fn head_dim(&self) -> CatResult<usize> {
        if self.num_attention_heads == 0 || self.hidden_size % self.num_attention_heads != 0 {
            return Err(CatError::ShapeMismatch(format!(
                "hidden {} not divisible by heads {}",
                self.hidden_size, self.num_attention_heads
            )));
        }
        Ok(self.hidden_size / self.num_attention_heads)
    }

    pub fn kv_dim(&self) -> CatResult<usize> {
        Ok(self.num_key_value_heads * self.head_dim()?)
    }

    /// Total parameter count (elements), matching the HF Mixtral layout:
    /// embed, per-layer (RMS×2, GQA, router, 8×SwiGLU), final RMS, untied LM head.
    pub fn param_count(&self) -> CatResult<u64> {
        let h = self.hidden_size as u64;
        let mid = self.intermediate_size as u64;
        let v = self.vocab_size as u64;
        let layers = self.num_hidden_layers as u64;
        let hd = self.head_dim()? as u64;
        let q = self.num_attention_heads as u64 * hd;
        let kv = self.num_key_value_heads as u64 * hd;
        let embed = v * h;
        let lm = if self.tie_word_embeddings { 0 } else { v * h };
        let attn = layers * (h * q + h * kv + h * kv + q * h);
        let norms = layers * 2 * h + h;
        let router = layers * h * self.num_local_experts as u64;
        let expert = 3 * h * mid;
        let experts = layers * self.num_local_experts as u64 * expert;
        Ok(embed + lm + attn + norms + router + experts)
    }

    pub fn active_param_count(&self) -> CatResult<u64> {
        let total = self.param_count()?;
        let h = self.hidden_size as u64;
        let mid = self.intermediate_size as u64;
        let layers = self.num_hidden_layers as u64;
        let expert = 3 * h * mid;
        let idle_experts =
            layers * (self.num_local_experts - self.num_experts_per_tok) as u64 * expert;
        Ok(total.saturating_sub(idle_experts))
    }
}

/// Build the Mixtral 1-morphism as a sequential composite of open systems.
/// Residuals are implicit in the GQA/MoE kernels (see `catnn-compile::kernel::moe`).
pub fn mixtral_architecture(
    cfg: &MixtralConfig,
    batch: usize,
    seq: usize,
    materialize_params: bool,
) -> CatResult<Architecture> {
    let h = cfg.hidden_size;
    let hd = cfg.head_dim()?;
    let tokens = Interface::single(
        format!("{}.tokens", cfg.name),
        TensorType::new([batch, seq], DType::I32),
    );
    let embed_op = OpenOp::Embedding { vocab: cfg.vocab_size, hidden: h };
    let embed = OpenComponent {
        name: "embed".into(),
        state: embed_op.state_interface("embed", Some(h)),
        input: tokens,
        output: hidden_interface("embed.Y", batch, seq, h),
        op: embed_op,
        moore: false,
    };
    let mut parts = vec![embed];
    let mut cur = hidden_interface("h", batch, seq, h);

    for layer in 0..cfg.num_hidden_layers {
        let rms1_op = OpenOp::RmsNorm { eps_e6: (cfg.rms_norm_eps * 1e6) as u32 };
        let rms1 = OpenComponent {
            name: format!("L{layer}.rms_attn"),
            state: rms1_op.state_interface(&format!("L{layer}.rms_attn"), Some(h)),
            input: cur.clone(),
            output: {
                let mut o = cur.clone();
                o.name = format!("L{layer}.rms_attn.Y");
                o
            },
            op: rms1_op,
            moore: false,
        };
        cur = rms1.output.clone();
        parts.push(rms1);

        let gqa_op = OpenOp::GqaAttention {
            n_heads: cfg.num_attention_heads,
            n_kv_heads: cfg.num_key_value_heads,
            head_dim: hd,
            d_model: h,
            sliding_window: cfg.sliding_window,
            rope_theta: cfg.rope_theta as u64,
        };
        let gqa = OpenComponent {
            name: format!("L{layer}.gqa"),
            state: gqa_op.state_interface(&format!("L{layer}.gqa"), Some(h)),
            input: cur.clone(),
            output: hidden_interface(&format!("L{layer}.gqa.Y"), batch, seq, h),
            op: gqa_op,
            moore: false,
        };
        cur = gqa.output.clone();
        parts.push(gqa);

        let rms2_op = OpenOp::RmsNorm { eps_e6: (cfg.rms_norm_eps * 1e6) as u32 };
        let rms2 = OpenComponent {
            name: format!("L{layer}.rms_moe"),
            state: rms2_op.state_interface(&format!("L{layer}.rms_moe"), Some(h)),
            input: cur.clone(),
            output: {
                let mut o = cur.clone();
                o.name = format!("L{layer}.rms_moe.Y");
                o
            },
            op: rms2_op,
            moore: false,
        };
        cur = rms2.output.clone();
        parts.push(rms2);

        let moe = MoEBlock::new(
            format!("L{layer}.moe"),
            cfg.num_local_experts,
            cfg.num_experts_per_tok,
            h,
            cfg.intermediate_size,
            cur.clone(),
        )?;
        cur = moe.component.output.clone();
        parts.push(moe.component);
    }

    let fn_op = OpenOp::RmsNorm { eps_e6: (cfg.rms_norm_eps * 1e6) as u32 };
    let final_norm = OpenComponent {
        name: "final_norm".into(),
        state: fn_op.state_interface("final_norm", Some(h)),
        input: cur.clone(),
        output: hidden_interface("final_norm.Y", batch, seq, h),
        op: fn_op,
        moore: false,
    };
    cur = final_norm.output.clone();
    parts.push(final_norm);

    let lm_op = OpenOp::LmHead { hidden: h, vocab: cfg.vocab_size };
    let lm = OpenComponent {
        name: "lm_head".into(),
        state: lm_op.state_interface("lm_head", Some(h)),
        input: cur,
        output: Interface::single("logits", TensorType::f32([batch, seq, cfg.vocab_size])),
        op: lm_op,
        moore: false,
    };
    parts.push(lm);

    let mut arch = Architecture::from_components(&cfg.name, parts)?;
    arch.materialize_params = materialize_params;
    Ok(arch)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hf_8x7b_param_count_is_about_47b() {
        let n = MixtralConfig::mixtral_8x7b().param_count().unwrap();
        // HF reports 46.7B; our layout is the same tensors.
        assert!(n > 46_000_000_000 && n < 48_000_000_000, "got {n}");
    }

    #[test]
    fn hf_8x22b_param_count_is_about_141b() {
        let n = MixtralConfig::mixtral_8x22b().param_count().unwrap();
        assert!(n > 130_000_000_000 && n < 150_000_000_000, "got {n}");
    }

    #[test]
    fn tiny_architecture_typechecks() {
        let cfg = MixtralConfig::tiny();
        let arch = mixtral_architecture(&cfg, 2, 4, true).unwrap();
        assert!(arch.components().len() > 4);
        assert_eq!(arch.output().ports["data"].shape.dims, vec![2, 4, 32]);
    }
}
