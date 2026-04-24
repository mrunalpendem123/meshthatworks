//! Invite-code pairing between two devices on the same home mesh.
//!
//! The pairing device generates an 8-char passcode, prints an invite string
//! containing its `EndpointId` plus the passcode, and listens on the
//! `mtw/pair/0` ALPN. The joining device parses the invite, connects, and
//! presents the passcode. On success both sides persist the other's id to
//! `~/.mtw/peers.json`.
//!
//! Passcode entropy: 8 chars × log2(32) = 40 bits. Interactive iroh
//! handshakes cap attacker throughput at roughly one attempt per second,
//! so brute-force is infeasible for the short window an invite is live.

use anyhow::{Context, bail};
use iroh::{Endpoint, EndpointAddr, EndpointId, SecretKey, endpoint::presets};
use rand::Rng;
use serde::{Deserialize, Serialize};

pub const PAIR_ALPN: &[u8] = b"mtw/pair/0";

const CODE_ALPHABET: &[u8] = b"abcdefghijkmnpqrstuvwxyz23456789";
const CODE_LEN: usize = 8;
const ID_HEX_LEN: usize = 64;
const INVITE_PREFIX: &str = "mtw-invite:";

#[derive(Debug, Serialize, Deserialize)]
struct JoinerHello {
    version: u32,
    code: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct PairerReply {
    accepted: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    reason: Option<String>,
}

fn generate_code() -> String {
    let mut rng = rand::thread_rng();
    (0..CODE_LEN)
        .map(|_| {
            let i = rng.gen_range(0..CODE_ALPHABET.len());
            CODE_ALPHABET[i] as char
        })
        .collect()
}

fn encode_invite(id: &EndpointId, code: &str) -> String {
    format!("{INVITE_PREFIX}{id}-{code}")
}

fn decode_invite(invite: &str) -> anyhow::Result<(EndpointId, String)> {
    let rest = invite
        .strip_prefix(INVITE_PREFIX)
        .with_context(|| format!("invite must start with {INVITE_PREFIX}"))?;

    if rest.len() < ID_HEX_LEN + 2 {
        bail!("invite is too short");
    }
    let id_part = &rest[..ID_HEX_LEN];
    let sep = rest.as_bytes()[ID_HEX_LEN];
    if sep != b'-' {
        bail!("invite separator missing between id and code");
    }
    let code = &rest[ID_HEX_LEN + 1..];
    if code.len() != CODE_LEN {
        bail!("invite passcode must be {CODE_LEN} chars, got {}", code.len());
    }
    let id: EndpointId = id_part
        .parse()
        .context("invite contains an invalid endpoint id")?;
    Ok((id, code.to_string()))
}

/// An active pairing session. `invite` is ready to share immediately;
/// call [`PairSession::wait_for_peer`] to block until a peer redeems it.
pub struct PairSession {
    pub invite: String,
    endpoint: Endpoint,
    code: String,
}

impl PairSession {
    pub async fn start(secret: SecretKey) -> anyhow::Result<Self> {
        let endpoint = Endpoint::builder(presets::N0)
            .secret_key(secret)
            .alpns(vec![PAIR_ALPN.to_vec()])
            .bind()
            .await
            .context("bind pairing endpoint")?;
        endpoint.online().await;
        let id = endpoint.id();
        let code = generate_code();
        let invite = encode_invite(&id, &code);
        Ok(Self {
            invite,
            endpoint,
            code,
        })
    }

    /// Accept connections until one presents the correct passcode. Returns
    /// the joiner's endpoint id and records them to `~/.mtw/peers.json`.
    pub async fn wait_for_peer(self) -> anyhow::Result<EndpointId> {
        while let Some(incoming) = self.endpoint.accept().await {
            match handle_join_attempt(incoming, &self.code).await {
                Ok(peer_id) => {
                    crate::peers::record(&peer_id.to_string())?;
                    self.endpoint.close().await;
                    return Ok(peer_id);
                }
                Err(err) => {
                    tracing::warn!(%err, "rejected pairing attempt");
                }
            }
        }
        anyhow::bail!("pairing endpoint closed before a peer joined")
    }
}

/// CLI convenience: start a session, print the invite, wait for a peer.
pub async fn pair(secret: SecretKey) -> anyhow::Result<()> {
    let session = PairSession::start(secret).await?;
    println!("mtw pair: waiting for a peer to join.");
    println!();
    println!("share this invite with your other device:");
    println!();
    println!("    {}", session.invite);
    println!();
    println!("on the other device, run:  mtw join <invite>");
    println!();
    println!("press ctrl-c to cancel.");
    let peer_id = session.wait_for_peer().await?;
    println!();
    println!("paired with {peer_id}");
    println!("recorded in ~/.mtw/peers.json");
    Ok(())
}

async fn handle_join_attempt(
    incoming: iroh::endpoint::Incoming,
    expected_code: &str,
) -> anyhow::Result<EndpointId> {
    let conn = incoming.await.context("accept incoming")?;
    let peer_id = conn.remote_id();
    let (mut send, mut recv) = conn.accept_bi().await.context("accept bi stream")?;

    let buf = recv.read_to_end(4096).await.context("read joiner hello")?;
    let hello: JoinerHello = serde_json::from_slice(&buf).context("parse joiner hello")?;

    let accepted = hello.version == 1 && hello.code == expected_code;
    let reply = if accepted {
        PairerReply {
            accepted: true,
            reason: None,
        }
    } else {
        PairerReply {
            accepted: false,
            reason: Some("invalid code".into()),
        }
    };
    let reply_bytes = serde_json::to_vec(&reply)?;
    send.write_all(&reply_bytes).await.context("write reply")?;
    send.finish().context("finish reply stream")?;
    conn.closed().await;

    if !accepted {
        bail!("invalid code from {peer_id}");
    }
    Ok(peer_id)
}

pub async fn join(secret: SecretKey, invite: &str) -> anyhow::Result<()> {
    let (target_id, code) = decode_invite(invite)?;

    let endpoint = Endpoint::builder(presets::N0)
        .secret_key(secret)
        .bind()
        .await
        .context("bind joiner endpoint")?;

    let conn = endpoint
        .connect(EndpointAddr::from(target_id), PAIR_ALPN)
        .await
        .context("connect to pairer")?;

    let (mut send, mut recv) = conn.open_bi().await.context("open bi stream")?;
    let hello = JoinerHello { version: 1, code };
    let hello_bytes = serde_json::to_vec(&hello)?;
    send.write_all(&hello_bytes).await.context("write hello")?;
    send.finish().context("finish hello stream")?;

    let reply_bytes = recv.read_to_end(4096).await.context("read reply")?;
    let reply: PairerReply = serde_json::from_slice(&reply_bytes).context("parse reply")?;
    if !reply.accepted {
        let reason = reply.reason.unwrap_or_else(|| "unknown".into());
        bail!("pairing rejected: {reason}");
    }

    crate::peers::record(&target_id.to_string())?;
    println!("paired with {target_id}");
    println!("recorded in ~/.mtw/peers.json");

    conn.close(0u32.into(), b"bye");
    endpoint.close().await;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invite_roundtrip() {
        let mut bytes = [0u8; 32];
        rand::thread_rng().fill(&mut bytes[..]);
        let secret = SecretKey::from_bytes(&bytes);
        let id = secret.public();
        let code = generate_code();
        let invite = encode_invite(&id, &code);
        let (decoded_id, decoded_code) = decode_invite(&invite).unwrap();
        assert_eq!(decoded_id, id);
        assert_eq!(decoded_code, code);
    }

    #[test]
    fn invite_rejects_bad_prefix() {
        assert!(decode_invite("not-an-invite:xxx").is_err());
    }

    #[test]
    fn invite_rejects_wrong_code_length() {
        let id = "00".repeat(32);
        assert!(decode_invite(&format!("{INVITE_PREFIX}{id}-short")).is_err());
    }

    #[test]
    fn generated_code_is_right_length_and_alphabet() {
        let code = generate_code();
        assert_eq!(code.len(), CODE_LEN);
        assert!(
            code.bytes().all(|b| CODE_ALPHABET.contains(&b)),
            "code {code} contains chars outside the alphabet"
        );
    }
}
