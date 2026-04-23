//! Diagnostic iroh echo — Milestone 1 transport sanity check.
//!
//! `listen` opens an iroh endpoint under the device's persistent identity,
//! advertises the echo ALPN, and copies any bytes received on a bidirectional
//! stream straight back. `dial` connects to a printed endpoint id, sends a
//! message, and prints the reply.

use anyhow::Context;
use iroh::{Endpoint, EndpointAddr, EndpointId, endpoint::presets};

pub const ECHO_ALPN: &[u8] = b"mtw/echo/0";

pub async fn listen() -> anyhow::Result<()> {
    let secret = crate::identity::load_or_create()?;
    let endpoint = Endpoint::builder(presets::N0)
        .secret_key(secret)
        .alpns(vec![ECHO_ALPN.to_vec()])
        .bind()
        .await
        .context("bind iroh endpoint")?;

    endpoint.online().await;

    let id = endpoint.id();
    println!("mtw echo: listening");
    println!("endpoint id: {id}");
    println!();
    println!("from another terminal or device, run:");
    println!("    mtw echo dial {id} <message>");
    println!();
    println!("press ctrl-c to stop.");

    while let Some(incoming) = endpoint.accept().await {
        tokio::spawn(async move {
            if let Err(err) = handle_connection(incoming).await {
                tracing::warn!(%err, "echo connection failed");
            }
        });
    }

    Ok(())
}

async fn handle_connection(incoming: iroh::endpoint::Incoming) -> anyhow::Result<()> {
    let conn = incoming.await.context("accept connection")?;
    let remote = conn.remote_id();
    let (mut send, mut recv) = conn.accept_bi().await.context("accept bi stream")?;

    let bytes = tokio::io::copy(&mut recv, &mut send)
        .await
        .context("echo copy")?;
    send.finish().context("finish send stream")?;

    println!("echoed {bytes} byte(s) to {remote}");
    conn.closed().await;
    Ok(())
}

pub async fn dial(target: &str, message: &str) -> anyhow::Result<()> {
    let id: EndpointId = target
        .parse()
        .with_context(|| format!("parse endpoint id from {target:?}"))?;
    let addr = EndpointAddr::from(id);

    let secret = crate::identity::load_or_create()?;
    let endpoint = Endpoint::builder(presets::N0)
        .secret_key(secret)
        .bind()
        .await
        .context("bind client endpoint")?;

    let conn = endpoint
        .connect(addr, ECHO_ALPN)
        .await
        .context("connect to target")?;

    let (mut send, mut recv) = conn.open_bi().await.context("open bi stream")?;
    send.write_all(message.as_bytes())
        .await
        .context("write message")?;
    send.finish().context("finish send stream")?;

    let reply = recv
        .read_to_end(1024 * 1024)
        .await
        .context("read echo reply")?;

    println!("{}", String::from_utf8_lossy(&reply));

    conn.close(0u32.into(), b"bye");
    endpoint.close().await;
    Ok(())
}
