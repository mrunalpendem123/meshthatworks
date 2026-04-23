//! Always-on mesh node.
//!
//! `mtw serve` binds the device's persistent identity and stays up to answer
//! mesh protocols — currently just `mtw/health/0`. The inference ALPN and
//! OpenAI API come later; serve is the process they'll hang off.

use std::sync::Arc;

use anyhow::Context;
use iroh::{Endpoint, SecretKey, endpoint::presets, protocol::Router};
use mtw_engine::InferenceEngine;

use crate::health::{HEALTH_ALPN, HealthHandler};

pub async fn run(secret: SecretKey, engine: Arc<dyn InferenceEngine>) -> anyhow::Result<()> {
    let info = engine.model_info().clone();
    let endpoint = Endpoint::builder(presets::N0)
        .secret_key(secret)
        .bind()
        .await
        .context("bind serve endpoint")?;

    endpoint.online().await;
    let id = endpoint.id();

    let router = Router::builder(endpoint)
        .accept(
            HEALTH_ALPN,
            HealthHandler {
                model_info: info.clone(),
            },
        )
        .spawn();

    let peer_count = crate::peers::load()
        .map(|list| list.peers.len())
        .unwrap_or(0);

    println!("mtw serve: online");
    println!("endpoint id: {id}");
    println!("model:       {}", info.name);
    println!("layers:      {}", info.num_layers);
    println!("peers known: {peer_count}");
    println!();
    println!("ALPNs:       {}", std::str::from_utf8(HEALTH_ALPN).unwrap());
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
