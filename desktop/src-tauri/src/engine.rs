//! Supervises the bundled `mtw serve` sidecar and reports its health.
//!
//! `mtw serve` owns the SwiftLM child process and exposes the OpenAI-compatible
//! proxy on `localhost:9337`. We spawn it as a Tauri sidecar, stream its logs to
//! the frontend, and run a single poller that turns `/healthz` + `/status` into
//! `engine-status` events the UI renders as the "node connecting → ready" arc.

use serde::Serialize;
use tauri::{AppHandle, Emitter, Manager};
use tauri_plugin_shell::process::{CommandChild, CommandEvent};
use tauri_plugin_shell::ShellExt;
use tokio::sync::Mutex;

pub const PROXY_URL: &str = "http://127.0.0.1:9337";

/// Path to the SwiftLM engine bundled inside the app (Resources/swiftlm/SwiftLM),
/// if present. When bundled, the app is self-contained — no Xcode, no Metal
/// Toolchain, no 30-min build. Returns None in dev builds without the resource.
pub fn bundled_swiftlm(app: &AppHandle) -> Option<std::path::PathBuf> {
    let p = app.path().resource_dir().ok()?.join("swiftlm").join("SwiftLM");
    p.exists().then_some(p)
}

/// Kill any engine processes squatting our ports. `mtw serve`'s SwiftLM child is
/// NOT reaped when we SIGKILL the sidecar, so without this a restart or model
/// switch leaves an orphaned SwiftLM on 9876 that blocks the new engine from
/// binding — the "node stuck on connecting" bug. Best-effort; ignores errors.
/// Worker's layer slice (`N/2..N`) computed from the active model's config, so
/// it loads the upper half while the head loads the lower half.
fn worker_range_from_active_model() -> Option<(usize, usize)> {
    let home = dirs::home_dir()?;
    let active = std::fs::read_to_string(home.join(".mtw").join("active-model")).ok()?;
    let dir = std::path::PathBuf::from(active.trim());
    let cfg = std::fs::read_to_string(dir.join("config.json")).ok()?;
    let v: serde_json::Value = serde_json::from_str(&cfg).ok()?;
    // Only Qwen3 (dense + MoE) and Llama can slice. For anything else (lfm2_moe,
    // olmoe, …) return None so the worker full-loads instead of failing to load
    // a slice it doesn't support.
    let arch = v.get("model_type").and_then(|x| x.as_str()).unwrap_or("");
    if !matches!(arch, "qwen3" | "qwen3_moe" | "llama") {
        return None;
    }
    let n = v.get("num_hidden_layers")?.as_u64()? as usize;
    Some((n / 2, n))
}

fn free_engine_ports() {
    use std::process::Command;
    // The orphaned bundled SwiftLM child (matches Resources/swiftlm/SwiftLM).
    let _ = Command::new("pkill").args(["-9", "-f", "swiftlm/SwiftLM"]).status();
    // Backstop: kill whatever still holds the engine/proxy ports.
    for port in ["9876", "9337"] {
        if let Ok(out) = Command::new("lsof").args(["-ti", &format!("tcp:{port}")]).output() {
            for pid in String::from_utf8_lossy(&out.stdout).split_whitespace() {
                let _ = Command::new("kill").args(["-9", pid]).status();
            }
        }
    }
}

#[derive(Default)]
pub struct Engine {
    inner: Mutex<EngineInner>,
}

/// How this node participates in a layer-split across paired devices.
#[derive(Clone, Default, PartialEq)]
pub enum SplitMode {
    /// Single-node — run the whole model locally.
    #[default]
    Off,
    /// Drive generation: run layers `0..N/2` locally, pipeline the rest to this
    /// peer (its endpoint id) over iroh.
    Head(String),
    /// Serve layers `N/2..N` over iroh for a paired head.
    Worker,
}

#[derive(Default)]
struct EngineInner {
    child: Option<CommandChild>,
    /// Whether the user wants the engine up. Drives the poller's "stopped"
    /// vs "starting" interpretation when no health response is available yet.
    intended: bool,
    /// Layer-split role this node should (re)start in.
    split: SplitMode,
}

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct EngineStatus {
    /// "stopped" | "starting" | "ready" | "error"
    pub phase: String,
    pub healthy: bool,
    pub message: String,
    /// The `/status` payload when the node is up, else null.
    pub status: Option<serde_json::Value>,
}

impl Engine {
    pub async fn is_running(&self) -> bool {
        self.inner.lock().await.child.is_some()
    }

    pub async fn intended(&self) -> bool {
        self.inner.lock().await.intended
    }

    /// Set the layer-split role applied on the next (re)start.
    pub async fn set_split(&self, mode: SplitMode) {
        self.inner.lock().await.split = mode;
    }

    /// Short label of the current split role, for the UI.
    pub async fn split_label(&self) -> String {
        match &self.inner.lock().await.split {
            SplitMode::Off => "off".into(),
            SplitMode::Head(p) => format!("head→{}", &p[..p.len().min(12)]),
            SplitMode::Worker => "worker".into(),
        }
    }

    /// Spawn `mtw serve` if it is not already running.
    pub async fn start(&self, app: &AppHandle) -> Result<(), String> {
        let mut g = self.inner.lock().await;
        g.intended = true;
        if g.child.is_some() {
            return Ok(());
        }

        // Clear any orphaned engine from a prior crash or hard kill (a SIGKILL'd
        // `mtw serve` leaves its SwiftLM child on 9876) so the new one can bind.
        free_engine_ports();
        tokio::time::sleep(std::time::Duration::from_millis(700)).await;

        // Use the bundled SwiftLM if present so the app is self-contained.
        let mut args: Vec<String> = vec!["serve".into()];
        if let Some(slm) = bundled_swiftlm(app) {
            args.push("--swiftlm".into());
            args.push(slm.to_string_lossy().into_owned());
        }
        // Layer-split flags: head pipelines to a peer over iroh; worker serves
        // its upper slice. Off → normal single-node serve.
        match &g.split {
            SplitMode::Off => {}
            SplitMode::Head(peer) => {
                args.push("--split-peer".into());
                args.push(peer.clone());
            }
            SplitMode::Worker => {
                if let Some((lo, hi)) = worker_range_from_active_model() {
                    args.push("--layer-range".into());
                    args.push(format!("{lo},{hi}"));
                }
            }
        }
        let cmd = app
            .shell()
            .sidecar("mtw")
            .map_err(|e| format!("locate mtw sidecar: {e}"))?
            .args(args);
        let (mut rx, child) = cmd.spawn().map_err(|e| format!("spawn mtw serve: {e}"))?;
        g.child = Some(child);
        drop(g);

        // Pump the sidecar's stdout/stderr to the UI as a live log.
        let app_log = app.clone();
        tauri::async_runtime::spawn(async move {
            while let Some(event) = rx.recv().await {
                match event {
                    CommandEvent::Stdout(bytes) | CommandEvent::Stderr(bytes) => {
                        let line = String::from_utf8_lossy(&bytes).trim_end().to_string();
                        if !line.is_empty() {
                            let _ = app_log.emit("engine-log", line);
                        }
                    }
                    CommandEvent::Terminated(payload) => {
                        let _ = app_log.emit(
                            "engine-log",
                            format!("[mtw serve exited: code={:?}]", payload.code),
                        );
                        // Clear the handle so the poller reports the real state.
                        if let Some(engine) = app_log.try_state::<Engine>() {
                            engine.inner.lock().await.child = None;
                        }
                        break;
                    }
                    _ => {}
                }
            }
        });

        let _ = app.emit(
            "engine-status",
            EngineStatus {
                phase: "starting".into(),
                healthy: false,
                message: "Spawning engine…".into(),
                status: None,
            },
        );
        Ok(())
    }

    /// Kill the sidecar and stop intending it to run.
    pub async fn stop(&self) -> Result<(), String> {
        let mut g = self.inner.lock().await;
        g.intended = false;
        if let Some(child) = g.child.take() {
            let _ = child.kill();
        }
        // Reap the orphaned SwiftLM child (the sidecar's SIGKILL doesn't), so a
        // following start() can bind the ports cleanly.
        free_engine_ports();
        Ok(())
    }
}

/// One long-lived loop, started at app setup, that translates the proxy's
/// health into `engine-status` events roughly twice a second.
pub fn spawn_poller(app: AppHandle) {
    tauri::async_runtime::spawn(async move {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_millis(1200))
            .build()
            .unwrap_or_default();
        let mut last_phase = String::new();

        loop {
            tokio::time::sleep(std::time::Duration::from_millis(1500)).await;
            let engine = match app.try_state::<Engine>() {
                Some(e) => e,
                None => continue,
            };
            let intended = engine.intended().await;
            let has_child = engine.is_running().await;

            let status = if client
                .get(format!("{PROXY_URL}/healthz"))
                .send()
                .await
                .map(|r| r.status().is_success())
                .unwrap_or(false)
            {
                let payload = client
                    .get(format!("{PROXY_URL}/status"))
                    .send()
                    .await
                    .ok();
                let body = match payload {
                    Some(r) => r.json::<serde_json::Value>().await.ok(),
                    None => None,
                };
                EngineStatus {
                    phase: "ready".into(),
                    healthy: true,
                    message: "Node online".into(),
                    status: body,
                }
            } else if has_child || intended {
                EngineStatus {
                    phase: "starting".into(),
                    healthy: false,
                    message: "Node connecting…".into(),
                    status: None,
                }
            } else {
                EngineStatus {
                    phase: "stopped".into(),
                    healthy: false,
                    message: "Engine offline".into(),
                    status: None,
                }
            };

            // Always emit on a phase change; otherwise refresh "ready" so the
            // UI can tick uptime, but stay quiet on repeated stopped frames.
            if status.phase != last_phase || status.phase == "ready" {
                last_phase = status.phase.clone();
                let _ = app.emit("engine-status", status);
            }
        }
    });
}
