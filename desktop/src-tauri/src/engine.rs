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

#[derive(Default)]
pub struct Engine {
    inner: Mutex<EngineInner>,
}

#[derive(Default)]
struct EngineInner {
    child: Option<CommandChild>,
    /// Whether the user wants the engine up. Drives the poller's "stopped"
    /// vs "starting" interpretation when no health response is available yet.
    intended: bool,
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

    /// Spawn `mtw serve` if it is not already running.
    pub async fn start(&self, app: &AppHandle) -> Result<(), String> {
        let mut g = self.inner.lock().await;
        g.intended = true;
        if g.child.is_some() {
            return Ok(());
        }

        // Use the bundled SwiftLM if present so the app is self-contained.
        let mut args: Vec<String> = vec!["serve".into()];
        if let Some(slm) = bundled_swiftlm(app) {
            args.push("--swiftlm".into());
            args.push(slm.to_string_lossy().into_owned());
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
            child.kill().map_err(|e| format!("kill mtw serve: {e}"))?;
        }
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
