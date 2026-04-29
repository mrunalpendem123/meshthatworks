//! Per-layer activation-passing RPC (`mtw/layer/0` ALPN).
//!
//! This is the wire protocol for layer-split inference. One peer asks another
//! to run a single transformer layer's forward pass, sending the input
//! activation tensor and receiving the output. The orchestrator on the
//! caller side stitches consecutive `forward_layer_on_peer` calls together
//! to form a full forward pass that crosses the mesh.
//!
//! Wire format choice: **bincode**, not JSON. An f32 activation tensor for
//! a typical hidden_size=4096 / seq_len=128 prompt is 2 MB raw — JSON would
//! 6× balloon it with `[1.0, 2.0, ...]` text. Bincode is `mem::size_of::<f32>()`
//! per element plus tiny len/shape headers. Same `serde::{Serialize,
//! Deserialize}` derives as the JSON path so it costs nothing at the type
//! level.
//!
//! The protocol shape (request + nonce + result enum) mirrors `infer.rs` so
//! the patterns stay grep-able. Only the encoding and the ALPN bytes change.
//!
//! Reference: `EXO`'s `pipeline_auto_parallel` (`auto_parallel.py`) does the
//! moral equivalent over MLX's distributed ring (raw TCP). Here we ride
//! iroh's QUIC stream instead, which gives us multiplexing, encryption, and
//! NAT traversal for free — the same substrate as `mtw/infer/0`.

use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, bail};
use iroh::{
    Endpoint, EndpointAddr, EndpointId,
    endpoint::Connection,
    protocol::{AcceptError, ProtocolHandler},
};
use mtw_engine::{ActivationTensor, InferenceEngine};
use serde::{Deserialize, Serialize};

pub const LAYER_ALPN: &[u8] = b"mtw/layer/0";

/// 128 MB ceiling per side. Largest realistic single-layer activation for
/// a 30B model: ~4096 × 8192 × 4 bytes ≈ 128 MB. We accept that as the cap.
const MAX_FRAME_BYTES: usize = 128 * 1024 * 1024;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerForwardRequest {
    pub version: u32,
    pub nonce: u64,
    pub layer_idx: usize,
    pub activation: ActivationTensor,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerForwardResponse {
    pub version: u32,
    pub nonce: u64,
    /// `Ok(activation)` on success, `Err(display_string)` if the remote engine
    /// rejected the request (out-of-range layer, malformed tensor, etc.).
    pub result: Result<ActivationTensor, String>,
}

#[derive(Clone)]
pub struct LayerHandler {
    engine: Arc<dyn InferenceEngine>,
}

impl LayerHandler {
    pub fn new(engine: Arc<dyn InferenceEngine>) -> Self {
        Self { engine }
    }
}

impl std::fmt::Debug for LayerHandler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LayerHandler")
            .field("model", &self.engine.model_info().name)
            .field("layers", &self.engine.model_info().num_layers)
            .finish()
    }
}

impl ProtocolHandler for LayerHandler {
    async fn accept(&self, conn: Connection) -> Result<(), AcceptError> {
        if let Err(err) = self.handle(conn).await {
            tracing::warn!(%err, "layer handler: failed");
        }
        Ok(())
    }
}

impl LayerHandler {
    async fn handle(&self, conn: Connection) -> anyhow::Result<()> {
        let peer = conn.remote_id();
        let (mut send, mut recv) = conn.accept_bi().await.context("accept bi")?;

        let buf = recv
            .read_to_end(MAX_FRAME_BYTES)
            .await
            .context("read layer request")?;
        let cfg = bincode_config();
        let (req, _): (LayerForwardRequest, usize) =
            bincode::serde::decode_from_slice(&buf, cfg).context("decode layer request")?;

        tracing::info!(
            %peer,
            nonce = req.nonce,
            layer = req.layer_idx,
            shape = ?req.activation.shape,
            "layer handler: dispatching to engine.run_layer"
        );

        let out = match self
            .engine
            .run_layer(req.layer_idx, req.activation)
            .await
        {
            Ok(act) => LayerForwardResponse {
                version: 1,
                nonce: req.nonce,
                result: Ok(act),
            },
            Err(err) => LayerForwardResponse {
                version: 1,
                nonce: req.nonce,
                result: Err(format!("{err:#}")),
            },
        };

        let bytes = bincode::serde::encode_to_vec(&out, cfg).context("encode layer response")?;
        send.write_all(&bytes).await.context("write reply")?;
        send.finish().context("finish send")?;
        conn.closed().await;
        Ok(())
    }
}

/// Dial a peer and ask it to run a single layer's forward pass. The
/// orchestrator (`LayerSplitEngine`) calls this once per layer that lives on
/// a remote peer, stitching the activations across the mesh.
pub async fn forward_layer_on_peer(
    endpoint: &Endpoint,
    peer_id: EndpointId,
    layer_idx: usize,
    activation: ActivationTensor,
    timeout: Duration,
) -> anyhow::Result<ActivationTensor> {
    let nonce: u64 = rand::random();
    let wire = LayerForwardRequest {
        version: 1,
        nonce,
        layer_idx,
        activation,
    };

    let fut = async {
        let conn = endpoint
            .connect(EndpointAddr::from(peer_id), LAYER_ALPN)
            .await
            .context("connect to peer")?;
        let (mut send, mut recv) = conn.open_bi().await.context("open bi")?;

        let cfg = bincode_config();
        let bytes = bincode::serde::encode_to_vec(&wire, cfg).context("encode request")?;
        send.write_all(&bytes).await.context("write request")?;
        send.finish().context("finish send")?;

        let buf = recv
            .read_to_end(MAX_FRAME_BYTES)
            .await
            .context("read response")?;
        let (resp, _): (LayerForwardResponse, usize) =
            bincode::serde::decode_from_slice(&buf, cfg).context("decode response")?;
        if resp.nonce != nonce {
            bail!("layer: nonce mismatch (expected {nonce}, got {})", resp.nonce);
        }
        conn.close(0u32.into(), b"bye");
        anyhow::Ok(resp)
    };

    let resp = tokio::time::timeout(timeout, fut)
        .await
        .map_err(|_| anyhow::anyhow!("forward_layer_on_peer: timeout after {timeout:?}"))??;

    match resp.result {
        Ok(act) => Ok(act),
        Err(msg) => bail!("remote layer error: {msg}"),
    }
}

fn bincode_config() -> bincode::config::Configuration {
    bincode::config::standard()
}
