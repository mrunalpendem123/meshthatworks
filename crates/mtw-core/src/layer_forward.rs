//! Range-based layer-forward RPC (`mtw/layer-forward/0` ALPN).
//!
//! Cross-machine bridge for `LayerSplitEngine`. Each peer in the layer-split
//! pipeline runs `mtw serve`, which exposes a *range slice* of the model
//! (set via `MTW_LAYER_RANGE=K,M` on the local SwiftLM). The orchestrator on
//! one machine dials the other's `mtw/layer-forward/0` ALPN and asks it to
//! run its slice — sending tokens (first peer) or an activation
//! (middle/last peers), receiving an activation (or final logits on last).
//!
//! Why a separate ALPN from `mtw/layer/0`:
//! - `mtw/layer/0` is **single-layer**: one RPC = one transformer layer.
//!   It targets `engine.run_layer(layer_idx, activation)` and works with
//!   `MockEngine` for tests, but `SwiftLMEngine::run_layer` is `bail!`'d.
//! - `mtw/layer-forward/0` is **range-based**: one RPC = the peer's whole
//!   layer slice. Maps directly onto SwiftLM's HTTP `/v1/layer-forward`,
//!   which is the only path that actually drives a forked SwiftLM with
//!   `MTW_LAYER_RANGE` set.
//!
//! Wire format: bincode (same as `mtw/layer/0`). An f32 activation for
//! Qwen3-30B (`hidden=5120`, seq=128) is ~2.5 MB — JSON would 6× that.
//!
//! The protocol shape mirrors `infer.rs` and `layer.rs` (request + nonce +
//! result enum) so all three stay grep-able.

use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, bail};
use iroh::{
    Endpoint, EndpointAddr, EndpointId,
    endpoint::Connection,
    protocol::{AcceptError, ProtocolHandler},
};
use mtw_engine::{ActivationTensor, LayerPeer};
use serde::{Deserialize, Serialize};

pub const LAYER_FORWARD_ALPN: &[u8] = b"mtw/layer-forward/0";

/// 128 MB ceiling per side. Matches `mtw/layer/0`.
const MAX_FRAME_BYTES: usize = 128 * 1024 * 1024;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LayerForwardInput {
    /// First peer in the pipeline — raw token IDs to embed and run.
    Tokens { tokens: Vec<i32>, shape: Vec<usize> },
    /// Middle/last peer — activation from the previous peer.
    Activation(ActivationTensor),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerForwardRequest {
    pub version: u32,
    pub nonce: u64,
    pub input: LayerForwardInput,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerForwardResponse {
    pub version: u32,
    pub nonce: u64,
    /// `Ok(activation)` on success, `Err(display_string)` if the local peer
    /// rejected the request (malformed tensor, layer-range mismatch, etc.).
    pub result: Result<ActivationTensor, String>,
}

#[derive(Clone)]
pub struct LayerForwardHandler {
    peer: Arc<dyn LayerPeer>,
}

impl LayerForwardHandler {
    pub fn new(peer: Arc<dyn LayerPeer>) -> Self {
        Self { peer }
    }
}

impl std::fmt::Debug for LayerForwardHandler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LayerForwardHandler").finish()
    }
}

impl ProtocolHandler for LayerForwardHandler {
    async fn accept(&self, conn: Connection) -> Result<(), AcceptError> {
        if let Err(err) = self.handle(conn).await {
            tracing::warn!(%err, "layer-forward handler: failed");
        }
        Ok(())
    }
}

impl LayerForwardHandler {
    async fn handle(&self, conn: Connection) -> anyhow::Result<()> {
        let peer_id = conn.remote_id();
        let (mut send, mut recv) = conn.accept_bi().await.context("accept bi")?;

        let buf = recv
            .read_to_end(MAX_FRAME_BYTES)
            .await
            .context("read layer-forward request")?;
        let cfg = bincode_config();
        let (req, _): (LayerForwardRequest, usize) =
            bincode::serde::decode_from_slice(&buf, cfg).context("decode request")?;

        tracing::info!(%peer_id, nonce = req.nonce, "layer-forward: dispatching to local peer");

        let result = match req.input {
            LayerForwardInput::Tokens { tokens, shape } => {
                self.peer.run_partial_tokens(&tokens, shape).await
            }
            LayerForwardInput::Activation(act) => {
                if !act.is_well_formed() {
                    Err(anyhow::anyhow!(
                        "activation data.len()={} but shape {:?} implies {}",
                        act.data.len(),
                        act.shape,
                        act.expected_len()
                    ))
                } else {
                    self.peer.run_partial_activation(act).await
                }
            }
        };

        let response = LayerForwardResponse {
            version: 1,
            nonce: req.nonce,
            result: result.map_err(|e| format!("{e:#}")),
        };

        let bytes = bincode::serde::encode_to_vec(&response, cfg).context("encode response")?;
        send.write_all(&bytes).await.context("write reply")?;
        send.finish().context("finish send")?;
        conn.closed().await;
        Ok(())
    }
}

/// Iroh-transported `LayerPeer`. Dials a remote `mtw serve` and proxies the
/// call over `mtw/layer-forward/0`. Use this when constructing a
/// `LayerSplitEngine` with peers on different machines.
pub struct IrohLayerPeer {
    endpoint: Endpoint,
    peer_id: EndpointId,
    timeout: Duration,
}

impl IrohLayerPeer {
    pub fn new(endpoint: Endpoint, peer_id: EndpointId) -> Self {
        Self {
            endpoint,
            peer_id,
            timeout: Duration::from_secs(900),
        }
    }

    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    async fn round_trip(
        &self,
        input: LayerForwardInput,
    ) -> anyhow::Result<ActivationTensor> {
        let nonce: u64 = rand::random();
        let req = LayerForwardRequest {
            version: 1,
            nonce,
            input,
        };
        let cfg = bincode_config();

        let fut = async {
            let conn = self
                .endpoint
                .connect(EndpointAddr::from(self.peer_id), LAYER_FORWARD_ALPN)
                .await
                .context("connect to peer")?;
            let (mut send, mut recv) = conn.open_bi().await.context("open bi")?;

            let bytes = bincode::serde::encode_to_vec(&req, cfg).context("encode request")?;
            send.write_all(&bytes).await.context("write request")?;
            send.finish().context("finish send")?;

            let buf = recv
                .read_to_end(MAX_FRAME_BYTES)
                .await
                .context("read response")?;
            let (resp, _): (LayerForwardResponse, usize) =
                bincode::serde::decode_from_slice(&buf, cfg).context("decode response")?;
            if resp.nonce != nonce {
                bail!(
                    "layer-forward: nonce mismatch (expected {nonce}, got {})",
                    resp.nonce
                );
            }
            conn.close(0u32.into(), b"bye");
            anyhow::Ok(resp)
        };

        let resp = tokio::time::timeout(self.timeout, fut)
            .await
            .map_err(|_| anyhow::anyhow!("layer-forward: timeout after {:?}", self.timeout))??;
        match resp.result {
            Ok(act) => Ok(act),
            Err(msg) => bail!("remote peer error: {msg}"),
        }
    }
}

#[async_trait::async_trait]
impl LayerPeer for IrohLayerPeer {
    async fn run_partial_tokens(
        &self,
        tokens: &[i32],
        shape: Vec<usize>,
    ) -> anyhow::Result<ActivationTensor> {
        self.round_trip(LayerForwardInput::Tokens {
            tokens: tokens.to_vec(),
            shape,
        })
        .await
    }

    async fn run_partial_activation(
        &self,
        input: ActivationTensor,
    ) -> anyhow::Result<ActivationTensor> {
        self.round_trip(LayerForwardInput::Activation(input)).await
    }
}

fn bincode_config() -> bincode::config::Configuration {
    bincode::config::standard()
}
