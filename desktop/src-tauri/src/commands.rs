//! Tauri commands the React frontend invokes.
//!
//! Everything that touches the network (the local proxy on :9337 and
//! huggingface.co) happens here in Rust via reqwest, then streams to the
//! webview over IPC events / channels. That sidesteps macOS App Transport
//! Security blocking cleartext http to localhost from WKWebView.

use std::io::Write as _;
use std::path::PathBuf;

use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use tauri::ipc::Channel;
use tauri::{AppHandle, Emitter, State};
use tauri_plugin_shell::process::{CommandChild, CommandEvent};
use tauri_plugin_shell::ShellExt;
use tokio::sync::Mutex;

use crate::engine::{Engine, PROXY_URL};
use crate::paths;

// ───────────────────────────────────────────────────────── setup

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SetupState {
    pub metal_available: bool,
    pub swiftlm_installed: bool,
    pub swiftlm_path: String,
    pub model_installed: bool,
    pub active_model: Option<String>,
    pub active_model_name: Option<String>,
    pub installed_count: usize,
    pub deps_dir: String,
    pub models_dir: String,
    /// SwiftLM built AND at least one model on disk AND an active model chosen.
    pub ready: bool,
}

fn metal_available() -> bool {
    std::process::Command::new("xcrun")
        .args(["-sdk", "macosx", "metal", "--version"])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// True only if `dir` contains real weights (`.safetensors`), not just a
/// config — i.e. a complete download, not a half-finished one.
fn dir_has_weights(dir: &std::path::Path) -> bool {
    std::fs::read_dir(dir)
        .map(|rd| {
            rd.filter_map(|e| e.ok())
                .any(|e| e.path().extension().map_or(false, |x| x == "safetensors"))
        })
        .unwrap_or(false)
}

fn read_active_model() -> Option<PathBuf> {
    let raw = std::fs::read_to_string(paths::active_model_file()).ok()?;
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return None;
    }
    let dir = PathBuf::from(trimmed);
    // Never report a not-downloaded model as active — a config-only path is a
    // failed/partial download and would make the UI show "Active" with no
    // weights for the engine to load.
    dir_has_weights(&dir).then_some(dir)
}

#[tauri::command]
pub fn get_setup_state(app: AppHandle) -> SetupState {
    setup_state(&app)
}

pub fn setup_state(app: &AppHandle) -> SetupState {
    // Prefer the engine bundled inside the app — when present, the user needs
    // no Xcode, no Metal Toolchain, no build step.
    let bundled = crate::engine::bundled_swiftlm(app);
    let swiftlm = bundled.clone().unwrap_or_else(paths::swiftlm_binary);
    let swiftlm_present = bundled.is_some() || paths::swiftlm_binary().is_file();

    let installed = installed_model_dirs();
    let active = read_active_model();
    let active_name = active
        .as_ref()
        .and_then(|p| p.file_name())
        .map(|n| n.to_string_lossy().to_string());
    let model_installed = !installed.is_empty();
    let active_valid = active
        .as_ref()
        .map(|p| p.join("config.json").is_file())
        .unwrap_or(false);

    SetupState {
        // A bundled engine ships precompiled Metal shaders, so the toolchain
        // is irrelevant; only fall back to probing it for a source build.
        metal_available: bundled.is_some() || metal_available(),
        swiftlm_installed: swiftlm_present,
        swiftlm_path: swiftlm.display().to_string(),
        model_installed,
        active_model: active.map(|p| p.display().to_string()),
        active_model_name: active_name,
        installed_count: installed.len(),
        deps_dir: paths::deps_dir().display().to_string(),
        models_dir: paths::models_dir().display().to_string(),
        ready: swiftlm_present && model_installed && active_valid,
    }
}

// ───────────────────────────────────────────────────────── models

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct InstalledModel {
    pub dir_name: String,
    pub path: String,
    pub size_bytes: u64,
    pub is_active: bool,
}

fn installed_model_dirs() -> Vec<PathBuf> {
    let mut out = Vec::new();
    if let Ok(rd) = std::fs::read_dir(paths::models_dir()) {
        for entry in rd.flatten() {
            let p = entry.path();
            // Require weights, not just config — a config-only dir is a
            // half-finished download and must show as "not installed" so the UI
            // offers a Download (not "Use this model").
            let has_weights = std::fs::read_dir(&p)
                .map(|rd| {
                    rd.filter_map(|e| e.ok())
                        .any(|e| e.path().extension().map_or(false, |x| x == "safetensors"))
                })
                .unwrap_or(false);
            if p.is_dir() && p.join("config.json").is_file() && has_weights {
                out.push(p);
            }
        }
    }
    out.sort();
    out
}

fn dir_size(dir: &std::path::Path) -> u64 {
    let mut total = 0;
    if let Ok(rd) = std::fs::read_dir(dir) {
        for entry in rd.flatten() {
            if let Ok(meta) = entry.metadata() {
                if meta.is_file() {
                    total += meta.len();
                }
            }
        }
    }
    total
}

#[tauri::command]
pub fn list_installed_models() -> Vec<InstalledModel> {
    let active = read_active_model();
    installed_model_dirs()
        .into_iter()
        .map(|p| {
            let is_active = active.as_ref().map(|a| a == &p).unwrap_or(false);
            InstalledModel {
                dir_name: p.file_name().unwrap_or_default().to_string_lossy().to_string(),
                path: p.display().to_string(),
                size_bytes: dir_size(&p),
                is_active,
            }
        })
        .collect()
}

#[tauri::command]
pub fn set_active_model(dir_name: String) -> Result<String, String> {
    let dir = paths::models_dir().join(&dir_name);
    if !dir.is_dir() {
        return Err(format!("not installed: {dir_name}"));
    }
    if !dir.join("config.json").is_file() {
        return Err(format!("{dir_name} has no config.json"));
    }
    // Require actual weights — config.json alone means a half-finished download,
    // which would silently activate a model the engine can't load.
    let has_weights = std::fs::read_dir(&dir)
        .map(|rd| {
            rd.filter_map(|e| e.ok())
                .any(|e| e.path().extension().map_or(false, |x| x == "safetensors"))
        })
        .unwrap_or(false);
    if !has_weights {
        return Err(format!("{dir_name} isn't fully downloaded yet (no .safetensors)"));
    }
    let abs = dir
        .canonicalize()
        .map_err(|e| format!("canonicalize: {e}"))?;
    let file = paths::active_model_file();
    if let Some(parent) = file.parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("mkdir {}: {e}", parent.display()))?;
    }
    std::fs::write(&file, format!("{}\n", abs.display())).map_err(|e| format!("write: {e}"))?;
    Ok(abs.display().to_string())
}

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct DownloadProgress {
    dir_name: String,
    file: String,
    file_index: usize,
    file_count: usize,
    received: u64,
    total: u64,
    overall_received: u64,
    overall_total: u64,
    phase: String, // "listing" | "downloading" | "done" | "error"
}

#[derive(Deserialize)]
struct HfEntry {
    #[serde(rename = "type")]
    kind: String,
    path: String,
    #[serde(default)]
    size: u64,
}

/// Download every file in a HuggingFace repo into `~/.meshthatworks-deps/models/<dir_name>`,
/// emitting `download-progress` events. Sets the model active when finished.
#[tauri::command]
pub async fn download_model(app: AppHandle, repo: String, dir_name: String) -> Result<(), String> {
    let dest = paths::models_dir().join(&dir_name);
    std::fs::create_dir_all(&dest).map_err(|e| format!("mkdir {}: {e}", dest.display()))?;

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(3600))
        .build()
        .map_err(|e| e.to_string())?;

    let emit = |p: DownloadProgress| {
        let _ = app.emit("download-progress", p);
    };

    emit(DownloadProgress {
        dir_name: dir_name.clone(),
        file: String::new(),
        file_index: 0,
        file_count: 0,
        received: 0,
        total: 0,
        overall_received: 0,
        overall_total: 0,
        phase: "listing".into(),
    });

    let tree_url = format!("https://huggingface.co/api/models/{repo}/tree/main?recursive=true");
    let entries: Vec<HfEntry> = client
        .get(&tree_url)
        .header("user-agent", "meshthatworks-desktop")
        .send()
        .await
        .map_err(|e| format!("list files: {e}"))?
        .json()
        .await
        .map_err(|e| format!("parse file list: {e}"))?;

    let files: Vec<HfEntry> = entries.into_iter().filter(|e| e.kind == "file").collect();
    if files.is_empty() {
        return Err(format!("no files found for {repo}"));
    }
    let file_count = files.len();
    let overall_total: u64 = files.iter().map(|f| f.size).sum();
    let mut overall_received: u64 = 0;

    for (idx, entry) in files.iter().enumerate() {
        let file_url = format!("https://huggingface.co/{repo}/resolve/main/{}", entry.path);
        let target = dest.join(&entry.path);
        if let Some(parent) = target.parent() {
            std::fs::create_dir_all(parent).map_err(|e| format!("mkdir: {e}"))?;
        }

        // Skip files already fully downloaded (resume across restarts).
        if let Ok(meta) = std::fs::metadata(&target) {
            if entry.size > 0 && meta.len() == entry.size {
                overall_received += entry.size;
                emit(DownloadProgress {
                    dir_name: dir_name.clone(),
                    file: entry.path.clone(),
                    file_index: idx + 1,
                    file_count,
                    received: entry.size,
                    total: entry.size,
                    overall_received,
                    overall_total,
                    phase: "downloading".into(),
                });
                continue;
            }
        }

        let part = target.with_extension("part");
        let resp = client
            .get(&file_url)
            .header("user-agent", "meshthatworks-desktop")
            .send()
            .await
            .map_err(|e| format!("download {}: {e}", entry.path))?;
        if !resp.status().is_success() {
            return Err(format!("download {}: HTTP {}", entry.path, resp.status()));
        }
        let total = resp.content_length().unwrap_or(entry.size);
        let mut file =
            std::fs::File::create(&part).map_err(|e| format!("create {}: {e}", part.display()))?;
        let mut received: u64 = 0;
        let mut stream = resp.bytes_stream();
        let mut last_tick = std::time::Instant::now();
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|e| format!("stream {}: {e}", entry.path))?;
            file.write_all(&chunk).map_err(|e| format!("write: {e}"))?;
            received += chunk.len() as u64;
            // Throttle UI events to ~10/s to avoid flooding the bridge.
            if last_tick.elapsed().as_millis() >= 100 {
                last_tick = std::time::Instant::now();
                emit(DownloadProgress {
                    dir_name: dir_name.clone(),
                    file: entry.path.clone(),
                    file_index: idx + 1,
                    file_count,
                    received,
                    total,
                    overall_received: overall_received + received,
                    overall_total,
                    phase: "downloading".into(),
                });
            }
        }
        file.flush().ok();
        drop(file);
        std::fs::rename(&part, &target).map_err(|e| format!("rename: {e}"))?;
        overall_received += received;
    }

    // Activate the freshly downloaded model.
    let _ = set_active_model(dir_name.clone());

    emit(DownloadProgress {
        dir_name,
        file: String::new(),
        file_index: file_count,
        file_count,
        received: overall_total,
        total: overall_total,
        overall_received: overall_total,
        overall_total,
        phase: "done".into(),
    });
    Ok(())
}

// ───────────────────────────────────────────────────────── engine control

#[tauri::command]
pub async fn start_engine(app: AppHandle, engine: State<'_, Engine>) -> Result<(), String> {
    engine.start(&app).await
}

#[tauri::command]
pub async fn stop_engine(engine: State<'_, Engine>) -> Result<(), String> {
    engine.stop().await
}

#[tauri::command]
pub async fn restart_engine(app: AppHandle, engine: State<'_, Engine>) -> Result<(), String> {
    engine.stop().await?;
    tokio::time::sleep(std::time::Duration::from_millis(900)).await;
    engine.start(&app).await
}

/// Set this node's layer-split role and restart the engine so it takes effect.
/// role = "off" | "head" | "worker"; `peer` is the worker's endpoint id (head only).
#[tauri::command]
pub async fn set_split_mode(
    app: AppHandle,
    engine: State<'_, Engine>,
    role: String,
    peer: Option<String>,
) -> Result<(), String> {
    use crate::engine::SplitMode;
    let mode = match role.as_str() {
        "head" => SplitMode::Head(peer.ok_or("head mode needs a peer endpoint id")?),
        "worker" => SplitMode::Worker,
        _ => SplitMode::Off,
    };
    engine.set_split(mode).await;
    engine.stop().await?;
    tokio::time::sleep(std::time::Duration::from_millis(900)).await;
    engine.start(&app).await
}

/// Current split role label, for the UI.
#[tauri::command]
pub async fn split_status(engine: State<'_, Engine>) -> Result<String, String> {
    Ok(engine.split_label().await)
}

#[derive(serde::Serialize, serde::Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct DiscoveredNode {
    pub endpoint_id: String,
    pub name: String,
    pub model: String,
    pub age_secs: u64,
}

/// Live list of swarm nodes the engine has discovered over gossip (named, no
/// invite codes). Empty if the engine isn't up yet.
#[tauri::command]
pub async fn discovered_nodes() -> Result<Vec<DiscoveredNode>, String> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(3))
        .build()
        .map_err(|e| e.to_string())?;
    match client.get(format!("{}/discovered", crate::engine::PROXY_URL)).send().await {
        Ok(r) => r.json().await.map_err(|e| e.to_string()),
        Err(_) => Ok(Vec::new()), // engine not up yet — show empty, not an error
    }
}

#[tauri::command]
pub async fn node_status() -> Result<serde_json::Value, String> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_millis(1500))
        .build()
        .map_err(|e| e.to_string())?;
    client
        .get(format!("{PROXY_URL}/status"))
        .send()
        .await
        .map_err(|e| format!("status: {e}"))?
        .json()
        .await
        .map_err(|e| format!("parse status: {e}"))
}

// ───────────────────────────────────────────────────────── peers / pairing

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PeerInfo {
    pub id: String,
    pub paired_at: u64,
}

#[tauri::command]
pub fn list_peers() -> Vec<PeerInfo> {
    let raw = match std::fs::read_to_string(paths::peers_file()) {
        Ok(s) => s,
        Err(_) => return Vec::new(),
    };
    let value: serde_json::Value = match serde_json::from_str(&raw) {
        Ok(v) => v,
        Err(_) => return Vec::new(),
    };
    value
        .get("peers")
        .and_then(|p| p.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|p| {
                    Some(PeerInfo {
                        id: p.get("id")?.as_str()?.to_string(),
                        paired_at: p.get("paired_at").and_then(|v| v.as_u64()).unwrap_or(0),
                    })
                })
                .collect()
        })
        .unwrap_or_default()
}

/// Holds the running `mtw pair` child so the UI can cancel it.
#[derive(Default)]
pub struct PairState(pub Mutex<Option<CommandChild>>);

#[tauri::command]
pub async fn pair_start(app: AppHandle, pair: State<'_, PairState>) -> Result<(), String> {
    // Cancel any prior session first.
    if let Some(child) = pair.0.lock().await.take() {
        let _ = child.kill();
    }
    let (mut rx, child) = app
        .shell()
        .sidecar("mtw")
        .map_err(|e| e.to_string())?
        .args(["pair"])
        .spawn()
        .map_err(|e| format!("spawn mtw pair: {e}"))?;
    *pair.0.lock().await = Some(child);

    let app2 = app.clone();
    tauri::async_runtime::spawn(async move {
        while let Some(event) = rx.recv().await {
            match event {
                CommandEvent::Stdout(b) | CommandEvent::Stderr(b) => {
                    let line = String::from_utf8_lossy(&b).trim_end().to_string();
                    if let Some(invite) = line.split_whitespace().find(|w| w.starts_with("mtw-invite:")) {
                        let _ = app2.emit("pair-invite", invite.to_string());
                    }
                    if line.contains("paired with") {
                        let _ = app2.emit("pair-success", line.clone());
                    }
                    if !line.is_empty() {
                        let _ = app2.emit("pair-log", line);
                    }
                }
                CommandEvent::Terminated(_) => {
                    let _ = app2.emit("pair-ended", ());
                    break;
                }
                _ => {}
            }
        }
    });
    Ok(())
}

#[tauri::command]
pub async fn pair_cancel(pair: State<'_, PairState>) -> Result<(), String> {
    if let Some(child) = pair.0.lock().await.take() {
        child.kill().map_err(|e| e.to_string())?;
    }
    Ok(())
}

#[tauri::command]
pub async fn join_mesh(app: AppHandle, invite: String) -> Result<(), String> {
    let invite = invite.trim().to_string();
    if !invite.starts_with("mtw-invite:") {
        return Err("that does not look like an invite (should start with mtw-invite:)".into());
    }
    let (mut rx, _child) = app
        .shell()
        .sidecar("mtw")
        .map_err(|e| e.to_string())?
        .args(["join", &invite])
        .spawn()
        .map_err(|e| format!("spawn mtw join: {e}"))?;

    let mut ok = false;
    let mut last = String::new();
    while let Some(event) = rx.recv().await {
        match event {
            CommandEvent::Stdout(b) | CommandEvent::Stderr(b) => {
                let line = String::from_utf8_lossy(&b).trim_end().to_string();
                if !line.is_empty() {
                    let _ = app.emit("pair-log", line.clone());
                    last = line.clone();
                }
                if line.contains("paired with") || line.contains("joined") {
                    ok = true;
                }
            }
            CommandEvent::Terminated(payload) => {
                if payload.code == Some(0) || ok {
                    return Ok(());
                }
                return Err(if last.is_empty() {
                    "join failed".into()
                } else {
                    last
                });
            }
            _ => {}
        }
    }
    if ok {
        Ok(())
    } else {
        Err("join ended without confirmation".into())
    }
}

// ───────────────────────────────────────────────────────── first-run setup

/// Build SwiftLM into `~/.meshthatworks-deps/SwiftLM`. Streams output as
/// `setup-log` events and a final `setup-status` ("done" | "error").
#[tauri::command]
pub async fn run_setup(app: AppHandle) -> Result<(), String> {
    if !metal_available() {
        let msg = "Xcode + the Metal Toolchain are required to build SwiftLM. Install Xcode from the App Store, then run: xcodebuild -downloadComponent MetalToolchain";
        let _ = app.emit("setup-status", serde_json::json!({"phase":"error","message":msg}));
        return Err(msg.into());
    }

    let deps = paths::deps_dir();
    let slm = paths::swiftlm_dir();
    let bin = paths::swiftlm_binary();
    let script = format!(
        r#"set -e
mkdir -p "{deps}/models"
if [ -x "{bin}" ]; then echo "SwiftLM already built at {bin}"; exit 0; fi
if [ ! -d "{slm}/.git" ]; then
  echo "Cloning SwiftLM (~3 GB)…"
  git clone --recursive --depth 1 https://github.com/SharpAI/SwiftLM "{slm}"
fi
cd "{slm}"
echo "Building SwiftLM (release) — this takes ~30 min the first time…"
swift build -c release
test -x "{bin}"
echo "SwiftLM ready at {bin}""#,
        deps = deps.display(),
        slm = slm.display(),
        bin = bin.display(),
    );

    let _ = app.emit("setup-status", serde_json::json!({"phase":"running","message":"Preparing the engine…"}));

    use tokio::io::{AsyncBufReadExt, BufReader};
    let mut child = tokio::process::Command::new("sh")
        .arg("-c")
        .arg(&script)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .map_err(|e| format!("spawn setup: {e}"))?;

    let stdout = child.stdout.take();
    let stderr = child.stderr.take();
    let app_out = app.clone();
    if let Some(out) = stdout {
        tauri::async_runtime::spawn(async move {
            let mut lines = BufReader::new(out).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                let _ = app_out.emit("setup-log", line);
            }
        });
    }
    let app_err = app.clone();
    if let Some(err) = stderr {
        tauri::async_runtime::spawn(async move {
            let mut lines = BufReader::new(err).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                let _ = app_err.emit("setup-log", line);
            }
        });
    }

    let status = child.wait().await.map_err(|e| format!("setup wait: {e}"))?;
    if status.success() {
        let _ = app.emit("setup-status", serde_json::json!({"phase":"done","message":"Engine ready"}));
        Ok(())
    } else {
        let msg = "SwiftLM build failed — see the log above.";
        let _ = app.emit("setup-status", serde_json::json!({"phase":"error","message":msg}));
        Err(msg.into())
    }
}

// ───────────────────────────────────────────────────────── chat streaming

#[derive(Deserialize)]
pub struct ChatMsg {
    pub role: String,
    pub content: String,
}

#[derive(Clone, Serialize)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum ChatEvent {
    Delta { content: String },
    Done,
    Error { message: String },
}

#[tauri::command]
pub async fn chat_stream(
    messages: Vec<ChatMsg>,
    max_tokens: Option<u32>,
    temperature: Option<f32>,
    on_event: Channel<ChatEvent>,
) -> Result<(), String> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(900))
        .build()
        .map_err(|e| e.to_string())?;

    let body = serde_json::json!({
        "model": "local",
        "stream": true,
        "max_tokens": max_tokens.unwrap_or(1024),
        "temperature": temperature.unwrap_or(0.7),
        "messages": messages.iter().map(|m| serde_json::json!({"role": m.role, "content": m.content})).collect::<Vec<_>>(),
    });

    let resp = match client
        .post(format!("{PROXY_URL}/v1/chat/completions"))
        .json(&body)
        .send()
        .await
    {
        Ok(r) => r,
        Err(e) => {
            let _ = on_event.send(ChatEvent::Error {
                message: format!("could not reach the engine: {e}"),
            });
            return Ok(());
        }
    };

    if !resp.status().is_success() {
        let code = resp.status();
        let text = resp.text().await.unwrap_or_default();
        let _ = on_event.send(ChatEvent::Error {
            message: format!("engine returned {code}: {text}"),
        });
        return Ok(());
    }

    let mut stream = resp.bytes_stream();
    let mut buf = String::new();
    while let Some(chunk) = stream.next().await {
        let chunk = match chunk {
            Ok(c) => c,
            Err(e) => {
                let _ = on_event.send(ChatEvent::Error { message: e.to_string() });
                return Ok(());
            }
        };
        buf.push_str(&String::from_utf8_lossy(&chunk));
        // Process complete lines; keep any trailing partial line in buf.
        while let Some(nl) = buf.find('\n') {
            let line = buf[..nl].trim().to_string();
            buf.drain(..=nl);
            let Some(data) = line.strip_prefix("data:") else { continue };
            let data = data.trim();
            if data == "[DONE]" {
                let _ = on_event.send(ChatEvent::Done);
                return Ok(());
            }
            if let Ok(json) = serde_json::from_str::<serde_json::Value>(data) {
                if let Some(content) = json
                    .get("choices")
                    .and_then(|c| c.get(0))
                    .and_then(|c| c.get("delta"))
                    .and_then(|d| d.get("content"))
                    .and_then(|c| c.as_str())
                {
                    if !content.is_empty() {
                        let _ = on_event.send(ChatEvent::Delta {
                            content: content.to_string(),
                        });
                    }
                }
            }
        }
    }
    let _ = on_event.send(ChatEvent::Done);
    Ok(())
}

#[tauri::command]
pub fn open_url(app: AppHandle, url: String) -> Result<(), String> {
    use tauri_plugin_opener::OpenerExt;
    app.opener()
        .open_url(url, None::<&str>)
        .map_err(|e| e.to_string())
}
