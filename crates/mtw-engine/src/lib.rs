//! Per-node inference engine for MeshThatWorks.
//!
//! Defines the [`InferenceEngine`] trait that every per-node engine must implement,
//! plus a deterministic [`MockEngine`] for mesh development and a [`SwiftLMEngine`]
//! that talks to a local SharpAI SwiftLM process over its OpenAI-compatible HTTP API.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

pub mod layer_split;
pub mod mock;
pub mod swiftlm;

pub use layer_split::LayerSplitEngine;
pub use mock::MockEngine;
pub use swiftlm::SwiftLMEngine;

/// Activation tensor flowing through the mesh.
///
/// Row-major, `f32` for v1. A future version will carry dtype and use `Vec<u8>`
/// so we can pass bf16/Q4 over the wire without decoding.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivationTensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

impl ActivationTensor {
    pub fn zeros(shape: Vec<usize>) -> Self {
        let n: usize = shape.iter().product();
        Self {
            shape,
            data: vec![0.0; n],
        }
    }

    pub fn expected_len(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn is_well_formed(&self) -> bool {
        self.data.len() == self.expected_len()
    }
}

/// Static metadata about the model this engine serves.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub name: String,
    pub num_layers: usize,
    pub hidden_size: usize,
    pub num_experts: Option<usize>,
    pub num_experts_per_tok: Option<usize>,
}

/// A single chat message in OpenAI-compatible format.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

impl ChatMessage {
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: "user".into(),
            content: content.into(),
        }
    }

    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: "system".into(),
            content: content.into(),
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: "assistant".into(),
            content: content.into(),
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ChatRequest {
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub temperature: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatResponse {
    pub content: String,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub latency_ms: u128,
}

/// One peer in a layer-split pipeline. Either a local `SwiftLMEngine`
/// (HTTP to a co-located SwiftLM) or an iroh-transported `IrohLayerPeer`
/// (forwards to a remote `mtw serve` over `mtw/layer-forward/0`). The two
/// methods mirror SwiftLM's `/v1/layer-forward` HTTP contract:
/// `run_partial_tokens` for the first peer (embeds tokens + runs its slice),
/// `run_partial_activation` for middle/last peers (consumes the previous
/// peer's activation, returns either the next activation or final logits).
#[async_trait]
pub trait LayerPeer: Send + Sync {
    /// First-peer entry point: send raw token IDs, get the post-slice
    /// activation `[batch, seq, hidden]`.
    async fn run_partial_tokens(
        &self,
        tokens: &[i32],
        shape: Vec<usize>,
    ) -> anyhow::Result<ActivationTensor>;

    /// Middle/last-peer entry point: send a previous peer's activation,
    /// get either the next activation or final logits.
    async fn run_partial_activation(
        &self,
        input: ActivationTensor,
    ) -> anyhow::Result<ActivationTensor>;
}

/// The per-node inference engine. Implementations:
/// - [`MockEngine`] — deterministic stand-in, useful for mesh development and tests.
/// - [`SwiftLMEngine`] — drives a local SwiftLM process over its OpenAI-compatible API.
/// - [`LayerSplitEngine`] — orchestrates layer-split inference across multiple [`LayerPeer`]s.
#[async_trait]
pub trait InferenceEngine: Send + Sync {
    fn model_info(&self) -> &ModelInfo;

    /// Whole-model chat completion. This is the path SwiftLM natively exposes.
    async fn chat_complete(&self, req: ChatRequest) -> anyhow::Result<ChatResponse>;

    /// Per-layer forward pass. Optional — engines that only expose whole-model
    /// generation (like SwiftLM via HTTP) return an error here. `LayerSplitEngine`
    /// and `MockEngine` implement it for mesh coordination work.
    async fn run_layer(
        &self,
        _layer_idx: usize,
        _activations: ActivationTensor,
    ) -> anyhow::Result<ActivationTensor> {
        anyhow::bail!("run_layer is not supported by this engine")
    }
}
