//! End-to-end test of the `mtw/layer-forward/0` ALPN.
//!
//! Two iroh endpoints in-process. On one we register a `LayerForwardHandler`
//! backed by a stub `LayerPeer`; the other dials via `IrohLayerPeer` and
//! verifies the round-trip preserves shape + data + handles both
//! `Tokens` and `Activation` inputs.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use iroh::{Endpoint, SecretKey, endpoint::presets, protocol::Router};
use mtw_core::layer_forward::{IrohLayerPeer, LAYER_FORWARD_ALPN, LayerForwardHandler};
use mtw_engine::{ActivationTensor, LayerPeer};

async fn fresh_endpoint() -> Endpoint {
    let mut bytes = [0u8; 32];
    rand::Rng::fill(&mut rand::thread_rng(), &mut bytes);
    let secret = SecretKey::from_bytes(&bytes);
    let endpoint = Endpoint::builder(presets::N0)
        .secret_key(secret)
        .bind()
        .await
        .expect("bind iroh endpoint");
    endpoint.online().await;
    endpoint
}

/// Test stub: synthesizes an activation matching the input shape, with each
/// element set to `tag` so the caller can verify which branch ran. Tokens
/// branch returns shape from caller; activation branch returns input shape.
struct StubPeer {
    tokens_tag: f32,
    activation_tag: f32,
}

#[async_trait]
impl LayerPeer for StubPeer {
    async fn run_partial_tokens(
        &self,
        _tokens: &[i32],
        shape: Vec<usize>,
    ) -> anyhow::Result<ActivationTensor> {
        let n: usize = shape.iter().product();
        Ok(ActivationTensor {
            shape,
            data: vec![self.tokens_tag; n],
        })
    }

    async fn run_partial_activation(
        &self,
        input: ActivationTensor,
    ) -> anyhow::Result<ActivationTensor> {
        let n: usize = input.shape.iter().product();
        Ok(ActivationTensor {
            shape: input.shape,
            data: vec![self.activation_tag; n],
        })
    }
}

#[tokio::test]
async fn tokens_round_trip() {
    let responder = fresh_endpoint().await;
    let caller = fresh_endpoint().await;
    let stub: Arc<dyn LayerPeer> = Arc::new(StubPeer {
        tokens_tag: 7.0,
        activation_tag: -1.0,
    });
    let _router = Router::builder(responder.clone())
        .accept(LAYER_FORWARD_ALPN, LayerForwardHandler::new(stub))
        .spawn();
    let responder_addr = responder.addr();

    let client = IrohLayerPeer::new(caller, responder_addr.id)
        .with_timeout(Duration::from_secs(15));

    let tokens = vec![1i32, 2, 3, 4];
    let out = client
        .run_partial_tokens(&tokens, vec![1, 4, 8])
        .await
        .expect("run_partial_tokens");

    assert_eq!(out.shape, vec![1, 4, 8]);
    assert_eq!(out.data.len(), 32);
    assert!(out.data.iter().all(|&x| x == 7.0));
}

#[tokio::test]
async fn activation_round_trip() {
    let responder = fresh_endpoint().await;
    let caller = fresh_endpoint().await;
    let stub: Arc<dyn LayerPeer> = Arc::new(StubPeer {
        tokens_tag: 7.0,
        activation_tag: -1.0,
    });
    let _router = Router::builder(responder.clone())
        .accept(LAYER_FORWARD_ALPN, LayerForwardHandler::new(stub))
        .spawn();
    let responder_addr = responder.addr();

    let client = IrohLayerPeer::new(caller, responder_addr.id)
        .with_timeout(Duration::from_secs(15));

    let input = ActivationTensor {
        shape: vec![1, 2, 4],
        data: vec![0.5; 8],
    };
    let out = client
        .run_partial_activation(input)
        .await
        .expect("run_partial_activation");

    assert_eq!(out.shape, vec![1, 2, 4]);
    assert!(out.data.iter().all(|&x| x == -1.0));
}

#[tokio::test]
async fn malformed_activation_returns_remote_error() {
    let responder = fresh_endpoint().await;
    let caller = fresh_endpoint().await;
    let stub: Arc<dyn LayerPeer> = Arc::new(StubPeer {
        tokens_tag: 7.0,
        activation_tag: -1.0,
    });
    let _router = Router::builder(responder.clone())
        .accept(LAYER_FORWARD_ALPN, LayerForwardHandler::new(stub))
        .spawn();
    let responder_addr = responder.addr();

    let client = IrohLayerPeer::new(caller, responder_addr.id)
        .with_timeout(Duration::from_secs(15));

    // shape says 1×2×4=8 but data has 5 — handler should reject.
    let bad = ActivationTensor {
        shape: vec![1, 2, 4],
        data: vec![0.0; 5],
    };
    let err = client
        .run_partial_activation(bad)
        .await
        .expect_err("expected error from malformed tensor");
    let msg = format!("{err:#}");
    assert!(msg.contains("activation"), "unexpected error: {msg}");
}
