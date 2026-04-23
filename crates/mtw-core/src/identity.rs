//! Per-device persistent identity.
//!
//! Loads (or generates and stores) a 32-byte iroh `SecretKey` at
//! `~/.mtw/identity.bin`. Every `mtw` subcommand that binds an endpoint
//! reuses this key so the device keeps a stable `EndpointId` across runs.

use anyhow::{Context, bail};
use iroh::SecretKey;
use rand::RngCore;
use std::path::PathBuf;

pub fn config_dir() -> anyhow::Result<PathBuf> {
    let home = std::env::var_os("HOME").context("$HOME not set")?;
    Ok(PathBuf::from(home).join(".mtw"))
}

pub fn load_or_create() -> anyhow::Result<SecretKey> {
    let dir = config_dir()?;
    let path = dir.join("identity.bin");

    if path.exists() {
        let bytes = std::fs::read(&path)
            .with_context(|| format!("read identity at {}", path.display()))?;
        if bytes.len() != 32 {
            bail!(
                "identity file {} is {} bytes, expected 32",
                path.display(),
                bytes.len()
            );
        }
        let mut arr = [0u8; 32];
        arr.copy_from_slice(&bytes);
        return Ok(SecretKey::from_bytes(&arr));
    }

    std::fs::create_dir_all(&dir)
        .with_context(|| format!("create {}", dir.display()))?;

    let mut bytes = [0u8; 32];
    rand::thread_rng().fill_bytes(&mut bytes);
    let key = SecretKey::from_bytes(&bytes);

    write_private(&path, &bytes)
        .with_context(|| format!("write identity to {}", path.display()))?;

    Ok(key)
}

#[cfg(unix)]
fn write_private(path: &std::path::Path, bytes: &[u8; 32]) -> anyhow::Result<()> {
    use std::os::unix::fs::OpenOptionsExt;
    use std::io::Write;
    let mut file = std::fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .mode(0o600)
        .open(path)?;
    file.write_all(bytes)?;
    Ok(())
}

#[cfg(not(unix))]
fn write_private(path: &std::path::Path, bytes: &[u8; 32]) -> anyhow::Result<()> {
    std::fs::write(path, bytes)?;
    Ok(())
}
