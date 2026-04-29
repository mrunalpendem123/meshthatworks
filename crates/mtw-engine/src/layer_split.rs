//! Layer-split inference engine — the orchestrator.
//!
//! Stitches multiple partial-loaded `SwiftLMEngine` peers together so that
//! every token's forward pass crosses the pipeline:
//!
//! ```text
//! prompt tokens
//!     │
//!     ▼
//! peer[0]  (MTW_LAYER_RANGE=0,K)        embedTokens + layers 0..K
//!     │ activation [B, L, hidden]
//!     ▼
//! peer[1]  (MTW_LAYER_RANGE=K+1,M)      layers K+1..M
//!     │ activation
//!     ⋮
//! peer[n-1] (MTW_LAYER_RANGE=M+1,N-1)   layers M+1..N-1 + norm + lm_head
//!     │ logits [B, L, vocab]
//!     ▼
//! argmax → next_token, loop
//! ```
//!
//! Scope of this v1:
//! - **Greedy decoding only.** No temperature/top-p sampling.
//! - **No KV cache.** Every generation step re-runs the full sequence.
//!   Quadratic in output length; fine for a 5–10-token demo, not for prod.
//!   (SwiftLM's `/v1/layer-forward` is stateless — adding a per-session
//!   cache there is meaningful follow-up work.)
//! - **Localhost peers.** All `SwiftLMEngine`s in `peers` are HTTP. To split
//!   across machines, wrap each peer in an iroh `mtw/layer/0` proxy — the
//!   wire format is identical, so it's a transport swap, not a redesign.
//! - **Hand-built Qwen3 ChatML template.** No `apply_chat_template` from a
//!   transformers-equivalent — we build the prompt by hand because
//!   `tokenizers` (Rust) doesn't ship chat templates.
//!
//! The point of the v1: prove a real model can produce a real token from a
//! real forward pass that physically traverses ≥2 partial-loaded SwiftLMs.
//! Speed is a follow-up problem.

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use anyhow::Context;
use async_trait::async_trait;
use tokenizers::Tokenizer;

use crate::{
    ActivationTensor, ChatRequest, ChatResponse, InferenceEngine, LayerPeer, ModelInfo,
};

pub struct LayerSplitEngine {
    info: ModelInfo,
    /// Layer-split peers, in pipeline order. `peers[0]` owns the embedding +
    /// first layer slice; `peers[N-1]` owns the last layer slice + final
    /// norm + lm_head. Implementations include local `SwiftLMEngine` (HTTP)
    /// and iroh-transported `IrohLayerPeer` (cross-machine).
    peers: Vec<Arc<dyn LayerPeer>>,
    tokenizer: Tokenizer,
    /// EOS token id from `tokenizer_config.json`. Stops generation early.
    eos_token: u32,
}

impl LayerSplitEngine {
    /// Build the engine from a list of layer-split peers and the model
    /// directory (used to load tokenizer + read config metadata).
    pub fn new(peers: Vec<Arc<dyn LayerPeer>>, model_dir: &Path) -> anyhow::Result<Self> {
        anyhow::ensure!(!peers.is_empty(), "LayerSplitEngine needs at least one peer");

        let tok_path = model_dir.join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&tok_path)
            .map_err(|e| anyhow::anyhow!("load tokenizer at {}: {e}", tok_path.display()))?;

        let eos_token = read_eos_token_id(model_dir)
            .with_context(|| format!("read EOS from {}", model_dir.display()))?;

        // Pull config metadata so we can publish a reasonable model_info to
        // the rest of the system. We don't reuse the per-peer ModelInfo
        // because each peer only knows about its own slice.
        let cfg_path = model_dir.join("config.json");
        let cfg_text = std::fs::read_to_string(&cfg_path)
            .with_context(|| format!("read {}", cfg_path.display()))?;
        let cfg: serde_json::Value = serde_json::from_str(&cfg_text)?;
        let num_layers = cfg
            .get("num_hidden_layers")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;
        let hidden_size = cfg
            .get("hidden_size")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;
        let num_experts = cfg.get("num_experts").and_then(|v| v.as_u64()).map(|v| v as usize);
        let num_experts_per_tok = cfg
            .get("num_experts_per_tok")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize);
        let name = model_dir
            .file_name()
            .map(|s| format!("{} (mesh×{})", s.to_string_lossy(), peers.len()))
            .unwrap_or_else(|| "mesh".into());

        Ok(Self {
            info: ModelInfo {
                name,
                num_layers,
                hidden_size,
                num_experts,
                num_experts_per_tok,
            },
            peers,
            tokenizer,
            eos_token,
        })
    }

    /// Hand-built Qwen3 ChatML template, with thinking mode DISABLED.
    ///
    /// Qwen3 reasoning models default to emitting a `<think>...</think>`
    /// chain-of-thought block before the actual answer. SwiftLM's
    /// `/v1/chat/completions` path has a `thinking=disabled` config that
    /// suppresses this; our `/v1/layer-forward` is a lower-level endpoint
    /// that doesn't, so we have to do it via the prompt.
    ///
    /// The trick (mirrored from upstream Qwen3 chat-template):
    /// append `<think>\n\n</think>\n\n` after the assistant tag, signalling
    /// to the model "thinking is already done". Without this we sample
    /// chain-of-thought tokens (nuclear / sword / blood / ...) instead of
    /// the answer, because greedy argmax over the raw logits picks the
    /// `<think>` token (id 151667) first.
    fn build_chatml_prompt(&self, req: &ChatRequest) -> String {
        let mut s = String::new();
        for m in &req.messages {
            s.push_str("<|im_start|>");
            s.push_str(&m.role);
            s.push('\n');
            s.push_str(&m.content);
            s.push_str("<|im_end|>\n");
        }
        s.push_str("<|im_start|>assistant\n<think>\n\n</think>\n\n");
        s
    }

    async fn forward_once(&self, tokens: &[i32]) -> anyhow::Result<ActivationTensor> {
        // First peer: tokens → activation.
        let mut act = self
            .peers[0]
            .run_partial_tokens(tokens, vec![1, tokens.len()])
            .await
            .context("first peer run_partial_tokens")?;

        // Middle/last peers: activation → activation (or logits, on the last).
        for (i, peer) in self.peers.iter().enumerate().skip(1) {
            act = peer
                .run_partial_activation(act)
                .await
                .with_context(|| format!("peer[{i}] run_partial_activation"))?;
        }
        Ok(act)
    }
}

#[async_trait]
impl InferenceEngine for LayerSplitEngine {
    fn model_info(&self) -> &ModelInfo {
        &self.info
    }

    async fn chat_complete(&self, req: ChatRequest) -> anyhow::Result<ChatResponse> {
        let t0 = Instant::now();

        let prompt_text = self.build_chatml_prompt(&req);
        let encoding = self
            .tokenizer
            .encode(prompt_text.clone(), false)
            .map_err(|e| anyhow::anyhow!("tokenize prompt: {e}"))?;
        let mut tokens: Vec<i32> = encoding.get_ids().iter().map(|&u| u as i32).collect();
        let prompt_token_count = tokens.len();
        eprintln!(
            "[layer_split] prompt template ({} chars):\n--- BEGIN ---\n{}\n--- END ---",
            prompt_text.len(), prompt_text
        );
        eprintln!(
            "[layer_split] encoded to {} tokens: {:?}",
            tokens.len(),
            tokens
        );

        let max_new = req.max_tokens.unwrap_or(50).max(1);
        let mut generated_ids: Vec<u32> = Vec::with_capacity(max_new);

        for step in 0..max_new {
            let logits = self.forward_once(&tokens).await?;

            // Logits shape is `[1, seq_len, vocab_size]`. We sample from the
            // LAST sequence position (the next-token slot).
            let vocab_size = *logits
                .shape
                .last()
                .ok_or_else(|| anyhow::anyhow!("empty logits shape"))?;
            anyhow::ensure!(
                logits.data.len() >= vocab_size,
                "logits underflow: data.len()={} vocab_size={}",
                logits.data.len(),
                vocab_size
            );
            let last_pos = &logits.data[logits.data.len() - vocab_size..];

            // Greedy: argmax. (Temperature/top-p is a follow-up.)
            let next_token = last_pos
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i as u32)
                .ok_or_else(|| anyhow::anyhow!("argmax over empty logits"))?;

            tracing::debug!(step, token = next_token, "layer-split: sampled token");

            if next_token == self.eos_token {
                break;
            }
            generated_ids.push(next_token);
            tokens.push(next_token as i32);
        }

        let content = self
            .tokenizer
            .decode(&generated_ids, true)
            .map_err(|e| anyhow::anyhow!("decode generated ids: {e}"))?;

        Ok(ChatResponse {
            prompt_tokens: prompt_token_count,
            completion_tokens: generated_ids.len(),
            content,
            latency_ms: t0.elapsed().as_millis(),
        })
    }
}

/// Walk the standard HF metadata files for an EOS token id. We try, in
/// order: `tokenizer_config.json::eos_token_id`, `config.json::eos_token_id`,
/// `generation_config.json::eos_token_id`. Falls back to a sensible default
/// (151645 — Qwen3's `<|im_end|>`) so a missing field doesn't kill the
/// whole engine.
fn read_eos_token_id(model_dir: &Path) -> anyhow::Result<u32> {
    let candidates = [
        "tokenizer_config.json",
        "config.json",
        "generation_config.json",
    ];
    for fname in candidates {
        let path = model_dir.join(fname);
        if !path.exists() {
            continue;
        }
        let text = std::fs::read_to_string(&path)?;
        let val: serde_json::Value = serde_json::from_str(&text)?;
        if let Some(id) = val.get("eos_token_id") {
            // Could be a single int or a list of ints (some configs).
            if let Some(n) = id.as_u64() {
                return Ok(n as u32);
            }
            if let Some(arr) = id.as_array() {
                if let Some(n) = arr.first().and_then(|v| v.as_u64()) {
                    return Ok(n as u32);
                }
            }
        }
    }
    tracing::warn!("no eos_token_id found in model dir, defaulting to 151645 (Qwen3 <|im_end|>)");
    Ok(151645)
}
