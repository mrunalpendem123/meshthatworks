//! Always-on mesh node.
//!
//! `mtw serve` binds the device's persistent identity and stays up to answer
//! mesh protocols — currently just `mtw/health/0`. The inference ALPN and
//! OpenAI API come later; serve is the process they'll hang off.

use std::sync::Arc;

use anyhow::Context;
use iroh::{Endpoint, SecretKey, endpoint::presets, protocol::Router};
use mtw_engine::{InferenceEngine, LayerPeer};

use crate::health::{HEALTH_ALPN, HealthHandler};
use crate::infer::{INFER_ALPN, InferHandler};
use crate::layer::{LAYER_ALPN, LayerHandler};
use crate::layer_forward::{LAYER_FORWARD_ALPN, LayerForwardHandler};

pub async fn run(
    secret: SecretKey,
    engine: Arc<dyn InferenceEngine>,
    layer_peer: Option<Arc<dyn LayerPeer>>,
) -> anyhow::Result<()> {
    let info = engine.model_info().clone();
    let endpoint = Endpoint::builder(presets::N0)
        .secret_key(secret)
        .bind()
        .await
        .context("bind serve endpoint")?;

    endpoint.online().await;
    let id = endpoint.id();

    let mut router_builder = Router::builder(endpoint)
        .accept(
            HEALTH_ALPN,
            HealthHandler {
                model_info: info.clone(),
            },
        )
        .accept(INFER_ALPN, InferHandler::new(engine.clone()))
        .accept(LAYER_ALPN, LayerHandler::new(engine.clone()));

    let mut alpns = vec![
        std::str::from_utf8(HEALTH_ALPN).unwrap(),
        std::str::from_utf8(INFER_ALPN).unwrap(),
        std::str::from_utf8(LAYER_ALPN).unwrap(),
    ];
    if let Some(peer) = layer_peer {
        router_builder = router_builder.accept(LAYER_FORWARD_ALPN, LayerForwardHandler::new(peer));
        alpns.push(std::str::from_utf8(LAYER_FORWARD_ALPN).unwrap());
    }
    let router = router_builder.spawn();

    let peer_count = crate::peers::load()
        .map(|list| list.peers.len())
        .unwrap_or(0);

    println!("mtw serve: online");
    println!("endpoint id: {id}");
    println!("model:       {}", info.name);
    println!("layers:      {}", info.num_layers);
    println!("peers known: {peer_count}");
    println!();
    println!("ALPNs:       {}", alpns.join(", "));
    println!("press ctrl-c to stop.");

    tokio::signal::ctrl_c()
        .await
        .context("install ctrl-c handler")?;
    println!();
    println!("shutting down…");
    router
        .shutdown()
        .await
        .map_err(|e| anyhow::anyhow!("router shutdown: {e:?}"))?;
    Ok(())
}
