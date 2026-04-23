//! Deterministic stand-in for a real inference engine.
//!
//! `MockEngine::run_layer` adds `layer_idx + 1` to every activation element.
//! That transform is invertible and additive across layers, so a full pipeline
//! pass visiting layers `[0, 1, …, N-1]` leaves each element bumped by
//! `1 + 2 + … + N = N(N+1)/2`. Tests assert that exact sum to prove every
//! expected layer actually ran, in order.

use async_trait::async_trait;

use crate::{ActivationTensor, InferenceEngine, ModelInfo};

pub struct MockEngine {
    info: ModelInfo,
}

impl MockEngine {
    pub fn new(info: ModelInfo) -> Self {
        Self { info }
    }

    /// Preset matching OLMoE-1B-7B-0125-Instruct — the current dev model.
    pub fn olmoe() -> Self {
        Self::new(ModelInfo {
            name: "OLMoE-1B-7B-0125-Instruct-4bit (mock)".into(),
            num_layers: 16,
            hidden_size: 2048,
            num_experts: Some(64),
            num_experts_per_tok: Some(8),
        })
    }
}

#[async_trait]
impl InferenceEngine for MockEngine {
    fn model_info(&self) -> &ModelInfo {
        &self.info
    }

    async fn run_layer(
        &self,
        layer_idx: usize,
        mut activations: ActivationTensor,
    ) -> anyhow::Result<ActivationTensor> {
        if layer_idx >= self.info.num_layers {
            anyhow::bail!(
                "layer_idx {layer_idx} out of range for {} layers",
                self.info.num_layers
            );
        }
        if !activations.is_well_formed() {
            anyhow::bail!(
                "activation data.len()={} but shape {:?} implies {}",
                activations.data.len(),
                activations.shape,
                activations.expected_len()
            );
        }
        let bump = (layer_idx + 1) as f32;
        for x in activations.data.iter_mut() {
            *x += bump;
        }
        Ok(activations)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn single_layer_bumps_by_one() {
        let engine = MockEngine::olmoe();
        let out = engine
            .run_layer(0, ActivationTensor::zeros(vec![2, 2, 2048]))
            .await
            .unwrap();
        assert!(out.data.iter().all(|&x| x == 1.0));
    }

    #[tokio::test]
    async fn full_pipeline_sums_triangular() {
        let engine = MockEngine::olmoe();
        let n = engine.model_info().num_layers;
        let mut x = ActivationTensor::zeros(vec![1, 4, 2048]);
        for layer in 0..n {
            x = engine.run_layer(layer, x).await.unwrap();
        }
        let expected = (n * (n + 1) / 2) as f32;
        assert!(
            x.data.iter().all(|&v| v == expected),
            "expected all elements = {expected}, got e.g. {:?}",
            &x.data[..4]
        );
    }

    #[tokio::test]
    async fn rejects_out_of_range_layer() {
        let engine = MockEngine::olmoe();
        let err = engine
            .run_layer(9999, ActivationTensor::zeros(vec![1, 1, 2048]))
            .await
            .unwrap_err();
        assert!(err.to_string().contains("out of range"));
    }

    #[tokio::test]
    async fn rejects_malformed_tensor() {
        let engine = MockEngine::olmoe();
        let bad = ActivationTensor {
            shape: vec![2, 2048],
            data: vec![0.0; 100],
        };
        let err = engine.run_layer(0, bad).await.unwrap_err();
        assert!(err.to_string().contains("data.len"));
    }

    #[tokio::test]
    async fn dyn_dispatch_works() {
        let engine: Box<dyn InferenceEngine> = Box::new(MockEngine::olmoe());
        let out = engine
            .run_layer(0, ActivationTensor::zeros(vec![1, 1, 8]))
            .await
            .unwrap();
        assert_eq!(out.data, vec![1.0; 8]);
    }
}
