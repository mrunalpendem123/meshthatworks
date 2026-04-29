//! Persistent list of paired peers, stored at `~/.mtw/peers.json`.

use anyhow::Context;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PeerList {
    #[serde(default)]
    pub peers: Vec<Peer>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Peer {
    pub id: String,
    pub paired_at: u64,
}

pub fn peers_path() -> anyhow::Result<PathBuf> {
    Ok(crate::identity::config_dir()?.join("peers.json"))
}

pub fn load() -> anyhow::Result<PeerList> {
    let path = peers_path()?;
    if !path.exists() {
        return Ok(PeerList::default());
    }
    let data = std::fs::read_to_string(&path)
        .with_context(|| format!("read {}", path.display()))?;
    serde_json::from_str(&data).with_context(|| format!("parse {}", path.display()))
}

pub fn save(list: &PeerList) -> anyhow::Result<()> {
    let path = peers_path()?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let json = serde_json::to_string_pretty(list)?;
    std::fs::write(&path, json)?;
    Ok(())
}

pub fn record(id: &str) -> anyhow::Result<()> {
    let mut list = load()?;
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    if let Some(existing) = list.peers.iter_mut().find(|p| p.id == id) {
        existing.paired_at = now;
    } else {
        list.peers.push(Peer {
            id: id.to_string(),
            paired_at: now,
        });
    }
    save(&list)
}

/// Remove a peer by full id. Returns `true` if a peer was removed.
pub fn remove(id: &str) -> anyhow::Result<bool> {
    let mut list = load()?;
    let before = list.peers.len();
    list.peers.retain(|p| p.id != id);
    let removed = list.peers.len() < before;
    if removed {
        save(&list)?;
    }
    Ok(removed)
}
