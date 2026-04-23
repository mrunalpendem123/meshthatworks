//! `mtw status` — pings every peer in `~/.mtw/peers.json` and prints health.

use std::time::Duration;

use anyhow::Context;
use iroh::{Endpoint, EndpointId, SecretKey, endpoint::presets};

use crate::health::ping_peer;

const PING_TIMEOUT: Duration = Duration::from_secs(5);

pub async fn run(secret: SecretKey) -> anyhow::Result<()> {
    let endpoint = Endpoint::builder(presets::N0)
        .secret_key(secret)
        .bind()
        .await
        .context("bind status endpoint")?;

    let peers = crate::peers::load().unwrap_or_default();

    println!("mtw status");
    println!("==========");
    println!("self id:     {}", endpoint.id());
    println!("peers known: {}", peers.peers.len());
    println!();

    if peers.peers.is_empty() {
        println!(
            "no peers paired. run `mtw pair` on one device and `mtw join <invite>` on the other."
        );
        endpoint.close().await;
        return Ok(());
    }

    println!(
        "pinging {} peer(s), timeout {:?} each...",
        peers.peers.len(),
        PING_TIMEOUT
    );
    println!();
    println!(
        "  {:<16}  {:>6}  {:>4}  {:<40}  {}",
        "peer", "status", "rtt", "model", "their peer count"
    );
    println!(
        "  {:<16}  {:>6}  {:>4}  {:<40}  {}",
        "----------------", "------", "----", "----------------------------------------", "-----"
    );

    for peer in &peers.peers {
        let short = peer.id.get(..16).unwrap_or(&peer.id);
        let peer_id: EndpointId = match peer.id.parse() {
            Ok(id) => id,
            Err(err) => {
                println!("  {short}  INVALID id ({err})");
                continue;
            }
        };
        match ping_peer(&endpoint, peer_id, PING_TIMEOUT).await {
            Ok((pong, rtt)) => {
                let model = pong
                    .model_info
                    .map(|m| m.name)
                    .unwrap_or_else(|| "(unknown)".into());
                println!(
                    "  {short}  {:>6}  {:>3}ms  {:<40}  {}",
                    "UP",
                    rtt.as_millis(),
                    truncate(&model, 40),
                    pong.peer_count
                );
            }
            Err(err) => {
                let summary = err.to_string();
                println!(
                    "  {short}  {:>6}  {:>4}  {}",
                    "DOWN",
                    "-",
                    truncate(&summary, 80)
                );
            }
        }
    }

    endpoint.close().await;
    Ok(())
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("{}…", &s[..max.saturating_sub(1)])
    }
}
