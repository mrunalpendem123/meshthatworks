//! Filesystem locations the desktop app shares with the `mtw` CLI.
//!
//! These mirror the constants baked into `mtw-cli` so the GUI and the
//! bundled sidecar agree on where SwiftLM, models, and the config live.

use std::path::PathBuf;

pub fn home() -> PathBuf {
    dirs::home_dir().unwrap_or_else(|| PathBuf::from("."))
}

/// `~/.meshthatworks-deps` — SwiftLM build + downloaded models.
pub fn deps_dir() -> PathBuf {
    if let Ok(p) = std::env::var("MTW_DEPS") {
        return PathBuf::from(p);
    }
    home().join(".meshthatworks-deps")
}

pub fn swiftlm_dir() -> PathBuf {
    deps_dir().join("SwiftLM")
}

/// The compiled SwiftLM binary the engine spawns.
pub fn swiftlm_binary() -> PathBuf {
    swiftlm_dir().join(".build/arm64-apple-macosx/release/SwiftLM")
}

pub fn models_dir() -> PathBuf {
    deps_dir().join("models")
}

/// `~/.mtw` — identity, peer list, active-model pointer.
pub fn mtw_config_dir() -> PathBuf {
    home().join(".mtw")
}

pub fn active_model_file() -> PathBuf {
    mtw_config_dir().join("active-model")
}

pub fn peers_file() -> PathBuf {
    mtw_config_dir().join("peers.json")
}
