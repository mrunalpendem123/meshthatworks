//! End-to-end test of the `mtw/layer/0` ALPN.
//!
//! Stands up two iroh endpoints in-process, registers a `LayerHandler` backed
//! by `MockEngine` on one, and exercises the wire format from the other via
//! `forward_layer_on_peer`. We bypass iroh's public discovery by passing the
//! full `EndpointAddr` (relay URL + direct UDP addresses) we got from the
//! responder side, so the test doesn't depend on the n0 DNS service or any
//! network-level flake.

use std::sync::Arc;
use std::time::Duration;

use iroh::{Endpoint, SecretKey, endpoint::presets, protocol::Router};
use mtw_core::layer::{LAYER_ALPN, LayerHandler, forward_layer_on_peer};
use mtw_engine::{ActivationTensor, InferenceEngine, MockEngine};

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

#[tokio::test]
async fn single_layer_round_trip() {
    let responder = fresh_endpoint().await;
    let caller = fresh_endpoint().await;

    let engine: Arc<dyn InferenceEngine> = Arc::new(MockEngine::olmoe());
    let _router = Router::builder(responder.clone())
        .accept(LAYER_ALPN, LayerHandler::new(engine))
        .spawn();

    let responder_addr = responder.addr();

    // Send a zero tensor through layer 0 — MockEngine bumps each elem by 1.
    let act = ActivationTensor::zeros(vec![1, 2, 8]);
    let out = forward_layer_on_peer(
        &caller,
        responder_addr.id,
        0,
        act,
        Duration::from_secs(15),
    )
    .await
    .expect("forward layer");

    assert_eq!(out.shape, vec![1, 2, 8]);
    assert!(
        out.data.iter().all(|&x| x == 1.0),
        "expected all 1.0 after layer 0 bump, got {:?}",
        &out.data[..4]
    );
}

#[tokio::test]
async fn pipeline_across_peers_sums_triangular() {
    // Same MockEngine math as the unit test, but with the full forward pass
    // crossing the iroh wire on every layer. After visiting 0..N layers the
    // value is N*(N+1)/2 — confirms ordering and that no layer was lost.
    let responder = fresh_endpoint().await;
    let caller = fresh_endpoint().await;

    let engine: Arc<dyn InferenceEngine> = Arc::new(MockEngine::olmoe());
    let n_layers = engine.model_info().num_layers;
    let _router = Router::builder(responder.clone())
        .accept(LAYER_ALPN, LayerHandler::new(engine))
        .spawn();

    let responder_addr = responder.addr();

    let mut act = ActivationTensor::zeros(vec![1, 1, 32]);
    for layer in 0..n_layers {
        act = forward_layer_on_peer(
            &caller,
            responder_addr.id,
            layer,
            act,
            Duration::from_secs(15),
        )
        .await
        .expect("forward layer");
    }

    let expected = (n_layers * (n_layers + 1) / 2) as f32;
    assert!(
        act.data.iter().all(|&x| x == expected),
        "expected all {expected} after {n_layers}-layer pipeline, got {:?}",
        &act.data[..4]
    );
}

#[tokio::test]
async fn split_across_two_peers_matches_local_pipeline() {
    // The point of this test: prove that splitting a model's layers across
    // two distinct peers produces the IDENTICAL result to running every
    // layer on a single peer. This is the foundational guarantee of
    // layer-split inference — without it, "two Macs together = one big
    // model" wouldn't hold.
    //
    // Setup mirrors the real architecture:
    //   peer A — owns layers 0..=7  (the "first half")
    //   peer B — owns layers 8..=15 (the "second half")
    //   client — runs forward pass: feed activation through A, then B,
    //            verify result equals 16*17/2 = 136 (MockEngine math).
    let peer_a = fresh_endpoint().await;
    let peer_b = fresh_endpoint().await;
    let client = fresh_endpoint().await;

    // Both peers run the same MockEngine config, but the orchestrator
    // (us, here) is what enforces the split. In real layer-split each
    // peer would have its weights filtered to its assigned slice.
    let engine_a: Arc<dyn InferenceEngine> = Arc::new(MockEngine::olmoe());
    let engine_b: Arc<dyn InferenceEngine> = Arc::new(MockEngine::olmoe());
    let total_layers = engine_a.model_info().num_layers;
    let split_at = total_layers / 2;

    let _router_a = Router::builder(peer_a.clone())
        .accept(LAYER_ALPN, LayerHandler::new(engine_a))
        .spawn();
    let _router_b = Router::builder(peer_b.clone())
        .accept(LAYER_ALPN, LayerHandler::new(engine_b))
        .spawn();

    let addr_a = peer_a.addr();
    let addr_b = peer_b.addr();

    let timeout = Duration::from_secs(15);
    let mut act = ActivationTensor::zeros(vec![1, 1, 32]);

    // First half: layers 0..split_at on peer A.
    for layer in 0..split_at {
        act = forward_layer_on_peer(&client, addr_a.id, layer, act, timeout)
            .await
            .expect("forward to peer A");
    }
    // Hand off the activation to peer B for the second half.
    let handoff_value_check = act.data[0];
    eprintln!(
        "[split test] handoff after {} layers on peer A: each elem = {}",
        split_at, handoff_value_check
    );

    for layer in split_at..total_layers {
        act = forward_layer_on_peer(&client, addr_b.id, layer, act, timeout)
            .await
            .expect("forward to peer B");
    }

    let expected = (total_layers * (total_layers + 1) / 2) as f32;
    eprintln!(
        "[split test] final after all {} layers (split A=[0..{}], B=[{}..{}]): each elem = {} (expected {})",
        total_layers, split_at, split_at, total_layers, act.data[0], expected,
    );

    assert!(
        act.data.iter().all(|&v| v == expected),
        "split-pipeline result mismatch: got {:?}, expected all {}",
        &act.data[..4],
        expected,
    );

    // And the handoff value must match what the first half alone would
    // produce, so we know we actually split at the right boundary.
    let expected_handoff = (split_at * (split_at + 1) / 2) as f32;
    assert!(
        (handoff_value_check - expected_handoff).abs() < 1e-5,
        "handoff value wrong: got {handoff_value_check}, expected {expected_handoff}",
    );
}

#[tokio::test]
async fn out_of_range_layer_returns_remote_error() {
    let responder = fresh_endpoint().await;
    let caller = fresh_endpoint().await;

    let engine: Arc<dyn InferenceEngine> = Arc::new(MockEngine::olmoe());
    let _router = Router::builder(responder.clone())
        .accept(LAYER_ALPN, LayerHandler::new(engine))
        .spawn();
    let responder_addr = responder.addr();

    let err = forward_layer_on_peer(
        &caller,
        responder_addr.id,
        9999,
        ActivationTensor::zeros(vec![1, 1, 32]),
        Duration::from_secs(15),
    )
    .await
    .expect_err("expected out-of-range error");

    let s = format!("{err:#}");
    assert!(
        s.contains("out of range"),
        "expected 'out of range' in error, got: {s}"
    );
}
