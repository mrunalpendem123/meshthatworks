//! Peer-to-peer inference RPC (`mtw/infer/0` ALPN).
//!
//! One peer asks another to run a full chat completion using the remote's
//! local engine. Wire format is JSON for now — simple to debug; if the
//! payloads grow (activations, large prompts) we can swap to bincode at
//! one call site without touching the handler or client callers.
//!
//! Shape inspired by Parallax's `forward.proto` but scoped down to v1's
//! "delegate a whole chat to another peer" rather than per-layer activation
//! pipelining, which stays a future protocol (`mtw/layer/0`).

use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, bail};
use iroh::{
    Endpoint, EndpointAddr, EndpointId,
    endpoint::Connection,
    protocol::{AcceptError, ProtocolHandler},
};
use mtw_engine::{ChatRequest, ChatResponse, InferenceEngine};
use serde::{Deserialize, Serialize};

pub const INFER_ALPN: &[u8] = b"mtw/infer/0";

const MAX_FRAME_BYTES: usize = 64 * 1024 * 1024; // 64 MB ceiling per side

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferRequestWire {
    pub version: u32,
    pub nonce: u64,
    pub request: ChatRequest,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferResponseWire {
    pub version: u32,
    pub nonce: u64,
    /// `Ok(response)` on success, `Err(display_string)` when the remote engine
    /// returned an error. We don't ship opaque Rust error types over the wire.
    pub result: Result<ChatResponse, String>,
}

#[derive(Clone)]
pub struct InferHandler {
    engine: Arc<dyn InferenceEngine>,
}

impl InferHandler {
    pub fn new(engine: Arc<dyn InferenceEngine>) -> Self {
        Self { engine }
    }
}

impl std::fmt::Debug for InferHandler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InferHandler")
            .field("model", &self.engine.model_info().name)
            .finish()
    }
}

impl ProtocolHandler for InferHandler {
    async fn accept(&self, conn: Connection) -> Result<(), AcceptError> {
        if let Err(err) = self.handle(conn).await {
            tracing::warn!(%err, "infer handler: failed");
        }
        Ok(())
    }
}

impl InferHandler {
    async fn handle(&self, conn: Connection) -> anyhow::Result<()> {
        let peer = conn.remote_id();
        let (mut send, mut recv) = conn.accept_bi().await.context("accept bi")?;

        let buf = recv
            .read_to_end(MAX_FRAME_BYTES)
            .await
            .context("read infer request")?;
        let req: InferRequestWire =
            serde_json::from_slice(&buf).context("parse infer request")?;

        tracing::info!(
            %peer,
            nonce = req.nonce,
            msgs = req.request.messages.len(),
            max_tokens = ?req.request.max_tokens,
            "infer handler: dispatching to local engine"
        );

        let out = match self.engine.chat_complete(req.request).await {
            Ok(resp) => InferResponseWire {
                version: 1,
                nonce: req.nonce,
                result: Ok(resp),
            },
            Err(err) => InferResponseWire {
                version: 1,
                nonce: req.nonce,
                result: Err(format!("{err:#}")),
            },
        };
        let bytes = serde_json::to_vec(&out)?;
        send.write_all(&bytes).await.context("write reply")?;
        send.finish().context("finish send")?;
        conn.closed().await;
        Ok(())
    }
}

pub async fn infer_on_peer(
    endpoint: &Endpoint,
    peer_id: EndpointId,
    request: ChatRequest,
    timeout: Duration,
) -> anyhow::Result<ChatResponse> {
    let nonce: u64 = rand::random();
    let wire = InferRequestWire {
        version: 1,
        nonce,
        request,
    };

    let fut = async {
        let conn = endpoint
            .connect(EndpointAddr::from(peer_id), INFER_ALPN)
            .await
            .context("connect to peer")?;
        let (mut send, mut recv) = conn.open_bi().await.context("open bi")?;

        let bytes = serde_json::to_vec(&wire)?;
        send.write_all(&bytes).await.context("write request")?;
        send.finish().context("finish send")?;

        let buf = recv
            .read_to_end(MAX_FRAME_BYTES)
            .await
            .context("read response")?;
        let resp: InferResponseWire =
            serde_json::from_slice(&buf).context("parse response")?;
        if resp.nonce != nonce {
            bail!(
                "infer: nonce mismatch (expected {nonce}, got {})",
                resp.nonce
            );
        }
        conn.close(0u32.into(), b"bye");
        anyhow::Ok(resp)
    };

    let resp = tokio::time::timeout(timeout, fut)
        .await
        .map_err(|_| anyhow::anyhow!("infer on peer: timeout after {timeout:?}"))??;

    match resp.result {
        Ok(ok) => Ok(ok),
        Err(msg) => bail!("remote engine error: {msg}"),
    }
}
