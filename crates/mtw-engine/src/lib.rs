//! Per-node inference engine for MeshThatWorks.
//!
//! Defines the [`InferenceEngine`] trait that every per-node engine must implement,
//! plus a deterministic [`MockEngine`] for mesh development. When the MLX FFI lands,
//! an `MlxEngine` will plug in behind the same trait without any change to callers.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

pub mod mock;

pub use mock::MockEngine;

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

/// Run one transformer layer at a time. A mesh coordinator calls this on the
/// engine that owns each layer, passing activations between peers over iroh.
#[async_trait]
pub trait InferenceEngine: Send + Sync {
    fn model_info(&self) -> &ModelInfo;

    async fn run_layer(
        &self,
        layer_idx: usize,
        activations: ActivationTensor,
    ) -> anyhow::Result<ActivationTensor>;
}
