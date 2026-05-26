//! Swarm discovery over iroh-gossip.
//!
//! Every node broadcasts a small presence message on a shared room topic every
//! few seconds; everyone keeps a live, TTL'd list of who's online. That powers
//! "see all active nodes → click join" without trading per-pair invite codes.
//!
//! Gossip needs ≥1 bootstrap peer to join a topic (no magic global list), so we
//! seed from the node's already-paired peers (`peers.json`). Pair once with
//! anyone (or open a room link) and you're on the swarm; discovery handles the
//! rest.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::Context;
use bytes::Bytes;
use futures_lite::StreamExt;
use iroh::{Endpoint, EndpointId};
use iroh_gossip::{api::Event, net::Gossip, proto::TopicId};
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;

/// Fixed 32-byte topic for the public MeshThatWorks swarm room.
pub const ROOM_TOPIC: TopicId = TopicId::from_bytes(*b"meshthatworks-swarm-v1-discovery");

/// Drop nodes we haven't heard from in this long.
const TTL_SECS: u64 = 20;

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Presence {
    pub name: String,
    pub endpoint_id: String,
    pub model: String,
    pub ts: u64,
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DiscoveredNode {
    pub endpoint_id: String,
    pub name: String,
    pub model: String,
    pub age_secs: u64,
}

/// Live registry of discovered nodes: `endpoint_id -> (presence, last_seen_unix)`.
#[derive(Clone, Default)]
pub struct Discovered {
    inner: Arc<Mutex<HashMap<String, (Presence, u64)>>>,
}

impl Discovered {
    pub async fn upsert(&self, p: Presence) {
        let now = now_unix();
        self.inner
            .lock()
            .await
            .insert(p.endpoint_id.clone(), (p, now));
    }

    /// Online nodes seen within the TTL, freshest first.
    pub async fn list(&self) -> Vec<DiscoveredNode> {
        let now = now_unix();
        let mut v: Vec<DiscoveredNode> = self
            .inner
            .lock()
            .await
            .values()
            .filter(|(_, seen)| now.saturating_sub(*seen) <= TTL_SECS)
            .map(|(p, seen)| DiscoveredNode {
                endpoint_id: p.endpoint_id.clone(),
                name: p.name.clone(),
                model: p.model.clone(),
                age_secs: now.saturating_sub(*seen),
            })
            .collect();
        v.sort_by_key(|n| n.age_secs);
        v
    }
}

fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Build the Gossip protocol on `endpoint`. Register the returned handler with
/// the router via `router_builder.accept(iroh_gossip::net::GOSSIP_ALPN, gossip.clone())`.
pub fn build_gossip(endpoint: Endpoint) -> Gossip {
    Gossip::builder().spawn(endpoint)
}

/// Join the room, broadcast our presence every 5s, and record everyone else's.
/// Runs until the topic stream ends.
pub async fn run_discovery(
    gossip: Gossip,
    name: String,
    endpoint_id: String,
    model: String,
    bootstrap: Vec<EndpointId>,
    discovered: Discovered,
) -> anyhow::Result<()> {
    let topic = gossip
        .subscribe(ROOM_TOPIC, bootstrap)
        .await
        .context("gossip subscribe to swarm room")?;
    let (sender, mut receiver) = topic.split();

    // Broadcast presence every 5s (ts refreshed each tick).
    tokio::spawn(async move {
        loop {
            let p = Presence {
                name: name.clone(),
                endpoint_id: endpoint_id.clone(),
                model: model.clone(),
                ts: now_unix(),
            };
            if let Ok(bytes) = serde_json::to_vec(&p) {
                let _ = sender.broadcast(Bytes::from(bytes)).await;
            }
            tokio::time::sleep(Duration::from_secs(5)).await;
        }
    });

    // Record everyone else's presence.
    while let Some(ev) = receiver.next().await {
        if let Ok(Event::Received(msg)) = ev {
            if let Ok(p) = serde_json::from_slice::<Presence>(&msg.content) {
                discovered.upsert(p).await;
            }
        }
    }
    Ok(())
}
