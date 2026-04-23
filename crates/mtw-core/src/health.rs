//! Peer health-check protocol.
//!
//! `HealthHandler` answers ping requests on the `mtw/health/0` ALPN with model
//! metadata and the responder's peer count. `ping_peer` is the client side —
//! opens a bi-stream to a known peer, sends a random-nonced ping, and returns
//! the pong plus measured round-trip time. Used by `mtw status`.

use std::time::{Duration, Instant};

use anyhow::{Context, bail};
use iroh::{
    Endpoint, EndpointAddr, EndpointId,
    endpoint::Connection,
    protocol::{AcceptError, ProtocolHandler},
};
use mtw_engine::ModelInfo;
use serde::{Deserialize, Serialize};

pub const HEALTH_ALPN: &[u8] = b"mtw/health/0";

#[derive(Debug, Serialize, Deserialize)]
pub struct Ping {
    pub nonce: u64,
    pub include_model_info: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Pong {
    pub nonce: u64,
    pub model_info: Option<ModelInfo>,
    pub peer_count: usize,
}

#[derive(Debug, Clone)]
pub struct HealthHandler {
    pub model_info: ModelInfo,
}

impl ProtocolHandler for HealthHandler {
    async fn accept(&self, conn: Connection) -> Result<(), AcceptError> {
        if let Err(err) = self.handle(conn).await {
            tracing::warn!(%err, "health handler: failed");
        }
        Ok(())
    }
}

impl HealthHandler {
    async fn handle(&self, conn: Connection) -> anyhow::Result<()> {
        let (mut send, mut recv) = conn.accept_bi().await.context("accept bi")?;
        let buf = recv.read_to_end(4096).await.context("read ping")?;
        let ping: Ping = serde_json::from_slice(&buf).context("parse ping")?;

        let peer_count = crate::peers::load()
            .map(|list| list.peers.len())
            .unwrap_or(0);
        let pong = Pong {
            nonce: ping.nonce,
            model_info: ping.include_model_info.then(|| self.model_info.clone()),
            peer_count,
        };
        let reply = serde_json::to_vec(&pong)?;
        send.write_all(&reply).await.context("write pong")?;
        send.finish().context("finish send")?;
        conn.closed().await;
        Ok(())
    }
}

pub async fn ping_peer(
    endpoint: &Endpoint,
    peer_id: EndpointId,
    timeout: Duration,
) -> anyhow::Result<(Pong, Duration)> {
    let start = Instant::now();
    let pong = tokio::time::timeout(timeout, send_ping(endpoint, peer_id))
        .await
        .map_err(|_| anyhow::anyhow!("timed out after {timeout:?}"))??;
    Ok((pong, start.elapsed()))
}

async fn send_ping(endpoint: &Endpoint, peer_id: EndpointId) -> anyhow::Result<Pong> {
    let conn = endpoint
        .connect(EndpointAddr::from(peer_id), HEALTH_ALPN)
        .await
        .context("connect to peer")?;
    let (mut send, mut recv) = conn.open_bi().await.context("open bi")?;
    let nonce: u64 = rand::random();
    let ping = Ping {
        nonce,
        include_model_info: true,
    };
    send.write_all(&serde_json::to_vec(&ping)?)
        .await
        .context("write ping")?;
    send.finish().context("finish send")?;

    let buf = recv.read_to_end(8192).await.context("read pong")?;
    let pong: Pong = serde_json::from_slice(&buf).context("parse pong")?;
    if pong.nonce != nonce {
        bail!("nonce mismatch: expected {nonce}, got {}", pong.nonce);
    }
    conn.close(0u32.into(), b"bye");
    Ok(pong)
}
