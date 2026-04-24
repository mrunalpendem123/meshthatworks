//! MeshThatWorks terminal UI.
//!
//! Single-binary TUI (ratatui + crossterm) that shows the live state of a
//! running `mtw serve` — identity, paired peers, recent activity, and an
//! embedded chat against the local OpenAI-compatible proxy.
//!
//! Visual design goals:
//! - Opening splash that shows the mesh coming online
//! - Tab bar: Dashboard · Chat · Peers · Models · Help
//! - Rounded borders, muted palette with a single accent colour
//! - Animated spinners for streaming / pinging
//! - Activity feed so you can see requests flowing in real time
//!
//! Key bindings:
//!   Tab / Shift-Tab   cycle tabs       (or 1–5 for direct)
//!   Enter             send chat prompt  (Chat tab)
//!   any char          typed into input  (Chat tab)
//!   Ctrl-L            clear chat
//!   Ctrl-R            force ping round
//!   ? or F1           toggle help overlay
//!   q (outside chat) / Esc / Ctrl-C    quit

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::Context;
use crossterm::{
    event::{
        DisableMouseCapture, EnableMouseCapture, Event, EventStream, KeyCode, KeyEventKind,
        KeyModifiers,
    },
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use futures_util::StreamExt;
use iroh::{Endpoint, EndpointId, endpoint::presets};
use mtw_api::NodeStatus;
use mtw_core::{
    health::{Pong, ping_peer},
    peers::{Peer, PeerList},
};
use ratatui::{
    Terminal,
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout, Margin, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span, Text},
    widgets::{Block, BorderType, Borders, Clear, Paragraph, Tabs, Wrap},
};
use serde_json::Value;
use tokio::sync::Mutex;

// ---------------------------------------------------------- palette

const ACCENT: Color = Color::Rgb(115, 192, 222); // soft cyan
const ACCENT_DIM: Color = Color::Rgb(70, 115, 135);
const OK: Color = Color::Rgb(163, 207, 141);
const WARN: Color = Color::Rgb(220, 180, 90);
const ERR: Color = Color::Rgb(230, 110, 110);
const FG: Color = Color::Rgb(220, 222, 226);
const MUTED: Color = Color::Rgb(125, 130, 140);
const BG_ALT: Color = Color::Rgb(35, 38, 46);

// ---------------------------------------------------------- public api

pub struct DashboardArgs {
    pub url: String,
    pub tick: Duration,
    pub ping_every: Duration,
}

// ---------------------------------------------------------- state

#[derive(Debug, Default, Clone)]
struct PeerHealth {
    last_rtt_ms: Option<u128>,
    last_model: Option<String>,
    last_error: Option<String>,
    last_checked: Option<Instant>,
}

#[derive(Debug, Clone)]
struct ChatTurn {
    role: String,
    content: String,
    tokens: Option<usize>,
    elapsed_ms: Option<u128>,
}

#[derive(Debug, Clone)]
struct Activity {
    at: Instant,
    kind: ActivityKind,
    text: String,
}

#[derive(Debug, Clone, Copy)]
enum ActivityKind {
    Info,
    Ok,
    Warn,
    Err,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Tab {
    Dashboard,
    Chat,
    Peers,
    Models,
    Help,
}

impl Tab {
    const ORDER: [Tab; 5] = [Tab::Dashboard, Tab::Chat, Tab::Peers, Tab::Models, Tab::Help];
    fn index(&self) -> usize {
        Self::ORDER.iter().position(|t| t == self).unwrap()
    }
    fn title(&self) -> &'static str {
        match self {
            Tab::Dashboard => "Dashboard",
            Tab::Chat => "Chat",
            Tab::Peers => "Peers",
            Tab::Models => "Models",
            Tab::Help => "Help",
        }
    }
}

#[derive(Debug, Clone)]
struct ModelRow {
    id: String,
    owned_by: String,
}

#[derive(Default)]
struct SharedState {
    tab: Option<Tab>, // None during splash
    splash_started_at: Option<Instant>,

    node: Option<NodeStatus>,
    node_error: Option<String>,

    peers: Vec<Peer>,
    peers_error: Option<String>,
    peer_health: HashMap<String, PeerHealth>,

    models: Vec<ModelRow>,
    models_error: Option<String>,

    chat: Vec<ChatTurn>,
    input: String,
    streaming: bool,
    chat_error: Option<String>,

    activity: VecDeque<Activity>,

    last_status_refresh: Option<Instant>,
    last_ping_round: Option<Instant>,
    last_models_refresh: Option<Instant>,

    spinner_frame: usize,
    show_help_overlay: bool,
}

impl SharedState {
    fn new() -> Self {
        let mut s = Self::default();
        s.tab = None;
        s.splash_started_at = Some(Instant::now());
        s.push_activity(ActivityKind::Info, "dashboard: starting");
        s
    }

    fn snapshot(&self) -> Self {
        Self {
            tab: self.tab,
            splash_started_at: self.splash_started_at,
            node: self.node.clone(),
            node_error: self.node_error.clone(),
            peers: self.peers.clone(),
            peers_error: self.peers_error.clone(),
            peer_health: self.peer_health.clone(),
            models: self.models.clone(),
            models_error: self.models_error.clone(),
            chat: self.chat.clone(),
            input: self.input.clone(),
            streaming: self.streaming,
            chat_error: self.chat_error.clone(),
            activity: self.activity.clone(),
            last_status_refresh: self.last_status_refresh,
            last_ping_round: self.last_ping_round,
            last_models_refresh: self.last_models_refresh,
            spinner_frame: self.spinner_frame,
            show_help_overlay: self.show_help_overlay,
        }
    }

    fn push_activity(&mut self, kind: ActivityKind, text: impl Into<String>) {
        let entry = Activity {
            at: Instant::now(),
            kind,
            text: text.into(),
        };
        self.activity.push_front(entry);
        while self.activity.len() > 200 {
            self.activity.pop_back();
        }
    }
}

// ---------------------------------------------------------- entry

pub async fn run(args: DashboardArgs) -> anyhow::Result<()> {
    let state = Arc::new(Mutex::new(SharedState::new()));

    let secret = mtw_core::identity::load_or_create()?;
    let endpoint = Endpoint::builder(presets::N0)
        .secret_key(secret)
        .bind()
        .await
        .context("bind iroh endpoint")?;
    state.lock().await.push_activity(
        ActivityKind::Ok,
        format!("iroh endpoint bound: {}", short_id(&endpoint.id().to_string())),
    );

    let refresh_handle = {
        let state = state.clone();
        let url = args.url.clone();
        let endpoint = endpoint.clone();
        let ping_every = args.ping_every;
        tokio::spawn(async move { refresh_loop(state, url, endpoint, ping_every).await })
    };

    enable_raw_mode().context("enable raw mode")?;
    let mut stdout = std::io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;
    terminal.hide_cursor().ok();

    let result = ui_loop(&mut terminal, state.clone(), &args).await;

    disable_raw_mode().ok();
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture
    )
    .ok();
    terminal.show_cursor().ok();

    refresh_handle.abort();
    let _ = refresh_handle.await;
    endpoint.close().await;
    result
}

// ---------------------------------------------------------- ui loop

const SPLASH_DURATION: Duration = Duration::from_millis(1200);

async fn ui_loop(
    terminal: &mut Terminal<CrosstermBackend<std::io::Stdout>>,
    state: Arc<Mutex<SharedState>>,
    args: &DashboardArgs,
) -> anyhow::Result<()> {
    let mut events = EventStream::new();
    let mut ticker = tokio::time::interval(args.tick);
    let mut spinner_tick = tokio::time::interval(Duration::from_millis(120));

    loop {
        // Transition from splash to first tab once its timer elapses.
        {
            let mut s = state.lock().await;
            if s.tab.is_none() {
                if let Some(t) = s.splash_started_at {
                    if t.elapsed() >= SPLASH_DURATION {
                        s.tab = Some(Tab::Dashboard);
                        s.push_activity(ActivityKind::Ok, "dashboard: ready");
                    }
                }
            }
        }
        let snap = state.lock().await.snapshot();
        terminal.draw(|f| render(f, &snap))?;

        tokio::select! {
            _ = ticker.tick() => {}
            _ = spinner_tick.tick() => {
                let mut s = state.lock().await;
                s.spinner_frame = s.spinner_frame.wrapping_add(1);
            }
            maybe = events.next() => match maybe {
                Some(Ok(Event::Key(key))) if key.kind == KeyEventKind::Press => {
                    if handle_key(key, &state, &args.url).await {
                        return Ok(());
                    }
                }
                Some(Err(e)) => tracing::warn!(%e, "event stream"),
                _ => {}
            }
        }
    }
}

async fn handle_key(
    key: crossterm::event::KeyEvent,
    state: &Arc<Mutex<SharedState>>,
    url: &str,
) -> bool {
    let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
    let in_chat = {
        let s = state.lock().await;
        s.tab == Some(Tab::Chat)
    };

    // Global: always-on keybindings
    match key.code {
        KeyCode::Char('c') if ctrl => return true,
        KeyCode::Esc => {
            let mut s = state.lock().await;
            if s.show_help_overlay {
                s.show_help_overlay = false;
                return false;
            }
            return true;
        }
        KeyCode::Char('?') | KeyCode::F(1) => {
            let mut s = state.lock().await;
            s.show_help_overlay = !s.show_help_overlay;
            return false;
        }
        KeyCode::Tab | KeyCode::Right => {
            let mut s = state.lock().await;
            if let Some(t) = s.tab {
                let i = (t.index() + 1) % Tab::ORDER.len();
                s.tab = Some(Tab::ORDER[i]);
            }
            return false;
        }
        KeyCode::BackTab | KeyCode::Left => {
            let mut s = state.lock().await;
            if let Some(t) = s.tab {
                let i = (t.index() + Tab::ORDER.len() - 1) % Tab::ORDER.len();
                s.tab = Some(Tab::ORDER[i]);
            }
            return false;
        }
        KeyCode::Char('r') if ctrl => {
            let mut s = state.lock().await;
            s.last_status_refresh = None;
            s.last_ping_round = None;
            s.last_models_refresh = None;
            s.push_activity(ActivityKind::Info, "manual refresh requested");
            return false;
        }
        KeyCode::Char('l') if ctrl && in_chat => {
            let mut s = state.lock().await;
            s.chat.clear();
            s.chat_error = None;
            s.push_activity(ActivityKind::Info, "chat cleared");
            return false;
        }
        _ => {}
    }

    // Direct tab navigation via digits (anywhere except when typing)
    if !in_chat {
        if let KeyCode::Char(c) = key.code {
            if let Some(d) = c.to_digit(10) {
                let d = d as usize;
                if (1..=Tab::ORDER.len()).contains(&d) {
                    state.lock().await.tab = Some(Tab::ORDER[d - 1]);
                    return false;
                }
            }
            if c == 'q' {
                return true;
            }
        }
    }

    // Chat input
    if in_chat {
        match key.code {
            KeyCode::Backspace => {
                let mut s = state.lock().await;
                s.input.pop();
            }
            KeyCode::Enter => {
                send_prompt(state.clone(), url.to_string()).await;
            }
            KeyCode::Char(c) => {
                let mut s = state.lock().await;
                if !s.streaming {
                    s.input.push(c);
                }
            }
            _ => {}
        }
    }
    false
}

// ---------------------------------------------------------- send + stream

async fn send_prompt(state: Arc<Mutex<SharedState>>, url: String) {
    let (model_name, history, prompt_text) = {
        let mut s = state.lock().await;
        if s.streaming || s.input.trim().is_empty() {
            return;
        }
        let prompt = std::mem::take(&mut s.input);
        let prompt_clone = prompt.clone();
        s.chat.push(ChatTurn {
            role: "user".into(),
            content: prompt,
            tokens: None,
            elapsed_ms: None,
        });
        s.chat.push(ChatTurn {
            role: "assistant".into(),
            content: String::new(),
            tokens: None,
            elapsed_ms: None,
        });
        s.streaming = true;
        s.chat_error = None;
        s.push_activity(ActivityKind::Info, format!("chat: {}", truncate(prompt_clone.clone(), 60)));

        let history: Vec<_> = s
            .chat
            .iter()
            .filter(|t| !t.content.is_empty() && !t.content.starts_with("[error:"))
            .map(|t| serde_json::json!({"role": t.role, "content": t.content}))
            .collect();

        let model_name = s
            .node
            .as_ref()
            .map(|n| n.model.name.clone())
            .unwrap_or_else(|| "mesh".into());
        (model_name, history, prompt_clone)
    };

    let state_clone = state.clone();
    tokio::spawn(async move {
        let started = Instant::now();
        let res = stream_chat(url, model_name, history, state_clone.clone()).await;
        let mut s = state_clone.lock().await;
        s.streaming = false;
        match res {
            Ok(tok) => {
                let ms = started.elapsed().as_millis();
                if let Some(last) = s.chat.last_mut() {
                    last.tokens = Some(tok);
                    last.elapsed_ms = Some(ms);
                }
                s.push_activity(
                    ActivityKind::Ok,
                    format!("reply: {} tok in {}ms (prompt: {})", tok, ms, truncate(prompt_text, 40)),
                );
            }
            Err(e) => {
                let pretty = format!("{e:#}");
                s.chat_error = Some(pretty.clone());
                if let Some(last) = s.chat.last_mut() {
                    if last.role == "assistant" && last.content.is_empty() {
                        last.content = format!("[error: {pretty}]");
                    }
                }
                s.push_activity(ActivityKind::Err, format!("chat error: {pretty}"));
            }
        }
    });
}

async fn stream_chat(
    url: String,
    model: String,
    messages: Vec<Value>,
    state: Arc<Mutex<SharedState>>,
) -> anyhow::Result<usize> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(900))
        .build()?;
    let body = serde_json::json!({
        "model": model,
        "messages": messages,
        "max_tokens": 400,
        "stream": true,
    });
    let resp = client
        .post(format!("{url}/v1/chat/completions"))
        .header("Accept", "text/event-stream")
        .json(&body)
        .send()
        .await
        .with_context(|| format!("POST {url}/v1/chat/completions"))?;
    let status = resp.status();
    if !status.is_success() {
        let body = resp.text().await.unwrap_or_default();
        anyhow::bail!("HTTP {}: {}", status, truncate(body, 200));
    }
    let mut stream = resp.bytes_stream();
    let mut buf = String::new();
    let mut tokens: usize = 0;

    let process_frame = |buf: &mut String,
                         tokens: &mut usize,
                         state: &Arc<Mutex<SharedState>>|
     -> Option<bool> {
        // Accept both \n\n and \r\n\r\n delimiters.
        let mut idx = buf.find("\n\n");
        if idx.is_none() {
            idx = buf.find("\r\n\r\n").map(|i| i + 2);
        }
        let Some(i) = idx else {
            return None;
        };
        let frame: String = buf.drain(..=i + 1).collect();
        for line in frame.lines() {
            let line = line.trim_end_matches('\r');
            let Some(payload) = line.strip_prefix("data: ").or_else(|| line.strip_prefix("data:")) else {
                continue;
            };
            let payload = payload.trim();
            if payload.is_empty() {
                continue;
            }
            if payload == "[DONE]" {
                return Some(true);
            }
            let v: Value = match serde_json::from_str(payload) {
                Ok(v) => v,
                Err(_) => continue,
            };
            if let Some(delta) = v["choices"][0]["delta"]["content"].as_str() {
                *tokens += 1;
                let delta = delta.to_string();
                let state = state.clone();
                tokio::spawn(async move {
                    let mut s = state.lock().await;
                    if let Some(last) = s.chat.last_mut() {
                        last.content.push_str(&delta);
                    }
                });
            }
        }
        Some(false)
    };

    while let Some(chunk) = stream.next().await {
        let chunk = chunk.context("read chunk")?;
        buf.push_str(&String::from_utf8_lossy(&chunk));
        while let Some(done) = process_frame(&mut buf, &mut tokens, &state) {
            if done {
                return Ok(tokens);
            }
        }
    }
    // Flush any trailing line even without final \n\n.
    for line in buf.lines() {
        let line = line.trim();
        let Some(payload) = line.strip_prefix("data: ").or_else(|| line.strip_prefix("data:")) else {
            continue;
        };
        let payload = payload.trim();
        if payload.is_empty() || payload == "[DONE]" {
            continue;
        }
        if let Ok(v) = serde_json::from_str::<Value>(payload) {
            if let Some(delta) = v["choices"][0]["delta"]["content"].as_str() {
                tokens += 1;
                let delta = delta.to_string();
                let state = state.clone();
                tokio::spawn(async move {
                    let mut s = state.lock().await;
                    if let Some(last) = s.chat.last_mut() {
                        last.content.push_str(&delta);
                    }
                });
            }
        }
    }
    Ok(tokens)
}

// ---------------------------------------------------------- refresh loop

async fn refresh_loop(
    state: Arc<Mutex<SharedState>>,
    url: String,
    endpoint: Endpoint,
    ping_every: Duration,
) {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(3))
        .build()
        .expect("build http client");
    let mut poll = tokio::time::interval(Duration::from_millis(500));
    loop {
        poll.tick().await;

        // /status
        match client.get(format!("{url}/status")).send().await {
            Ok(r) if r.status().is_success() => match r.json::<NodeStatus>().await {
                Ok(n) => {
                    let mut s = state.lock().await;
                    let first = s.node.is_none();
                    s.node = Some(n);
                    s.node_error = None;
                    s.last_status_refresh = Some(Instant::now());
                    if first {
                        s.push_activity(ActivityKind::Ok, "connected to /status on proxy");
                    }
                }
                Err(e) => state.lock().await.node_error = Some(format!("parse /status: {e}")),
            },
            Ok(r) => state.lock().await.node_error = Some(format!("/status http {}", r.status())),
            Err(e) => {
                let mut s = state.lock().await;
                let first = s.node.is_some() || s.node_error.is_none();
                s.node = None;
                s.node_error = Some(format!("/status unreachable: {e}"));
                if first {
                    s.push_activity(ActivityKind::Warn, format!("/status unreachable"));
                }
            }
        }

        // /v1/models (refresh every 5s)
        let refresh_models = state
            .lock()
            .await
            .last_models_refresh
            .map(|t| t.elapsed() >= Duration::from_secs(5))
            .unwrap_or(true);
        if refresh_models {
            match client.get(format!("{url}/v1/models")).send().await {
                Ok(r) if r.status().is_success() => match r.json::<Value>().await {
                    Ok(v) => {
                        let list: Vec<ModelRow> = v["data"]
                            .as_array()
                            .map(|arr| {
                                arr.iter()
                                    .map(|m| ModelRow {
                                        id: m["id"].as_str().unwrap_or("").into(),
                                        owned_by: m["owned_by"].as_str().unwrap_or("").into(),
                                    })
                                    .collect()
                            })
                            .unwrap_or_default();
                        let mut s = state.lock().await;
                        s.models = list;
                        s.models_error = None;
                        s.last_models_refresh = Some(Instant::now());
                    }
                    Err(e) => state.lock().await.models_error = Some(format!("{e}")),
                },
                Ok(r) => state.lock().await.models_error = Some(format!("http {}", r.status())),
                Err(e) => state.lock().await.models_error = Some(format!("{e}")),
            }
        }

        // peers.json
        match mtw_core::peers::load() {
            Ok(PeerList { peers }) => {
                let mut s = state.lock().await;
                if s.peers.len() != peers.len() {
                    s.push_activity(
                        ActivityKind::Info,
                        format!("peers.json → {} entries", peers.len()),
                    );
                }
                s.peers = peers;
                s.peers_error = None;
            }
            Err(e) => state.lock().await.peers_error = Some(format!("{e}")),
        }

        let should_ping = state
            .lock()
            .await
            .last_ping_round
            .map(|t| t.elapsed() >= ping_every)
            .unwrap_or(true);
        if should_ping {
            let peers = state.lock().await.peers.clone();
            for peer in peers {
                let id: EndpointId = match peer.id.parse() {
                    Ok(id) => id,
                    Err(e) => {
                        state
                            .lock()
                            .await
                            .peer_health
                            .insert(peer.id.clone(), PeerHealth {
                                last_error: Some(format!("bad id: {e}")),
                                last_checked: Some(Instant::now()),
                                ..PeerHealth::default()
                            });
                        continue;
                    }
                };
                let result = ping_peer(&endpoint, id, Duration::from_secs(3)).await;
                let mut s = state.lock().await;
                let short = short_id(&peer.id);
                let slot = s.peer_health.entry(peer.id.clone()).or_default();
                slot.last_checked = Some(Instant::now());
                match result {
                    Ok((Pong { model_info, .. }, rtt)) => {
                        let was_down = slot.last_rtt_ms.is_none();
                        slot.last_rtt_ms = Some(rtt.as_millis());
                        slot.last_model = model_info.map(|m| m.name);
                        slot.last_error = None;
                        if was_down {
                            s.push_activity(
                                ActivityKind::Ok,
                                format!("peer {short} UP {}ms", rtt.as_millis()),
                            );
                        }
                    }
                    Err(e) => {
                        let was_up = slot.last_rtt_ms.is_some();
                        slot.last_rtt_ms = None;
                        slot.last_error = Some(format!("{e}"));
                        if was_up {
                            s.push_activity(ActivityKind::Warn, format!("peer {short} DOWN"));
                        }
                    }
                }
            }
            state.lock().await.last_ping_round = Some(Instant::now());
        }
    }
}

// ---------------------------------------------------------- render

fn render(f: &mut ratatui::Frame, s: &SharedState) {
    // Splash
    if s.tab.is_none() {
        render_splash(f, s);
        return;
    }

    let area = f.area();
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // tab bar
            Constraint::Min(10),   // active tab body
            Constraint::Length(2), // footer
        ])
        .split(area);

    render_tabbar(f, rows[0], s);

    match s.tab.unwrap() {
        Tab::Dashboard => render_tab_dashboard(f, rows[1], s),
        Tab::Chat => render_tab_chat(f, rows[1], s),
        Tab::Peers => render_tab_peers(f, rows[1], s),
        Tab::Models => render_tab_models(f, rows[1], s),
        Tab::Help => render_tab_help(f, rows[1]),
    }

    render_footer(f, rows[2], s);

    if s.show_help_overlay {
        render_help_overlay(f, area);
    }
}

fn render_splash(f: &mut ratatui::Frame, s: &SharedState) {
    let area = f.area();
    let logo = vec![
        "",
        "   ███╗   ███╗████████╗██╗    ██╗",
        "   ████╗ ████║╚══██╔══╝██║    ██║",
        "   ██╔████╔██║   ██║   ██║ █╗ ██║",
        "   ██║╚██╔╝██║   ██║   ██║███╗██║",
        "   ██║ ╚═╝ ██║   ██║   ╚███╔███╔╝",
        "   ╚═╝     ╚═╝   ╚═╝    ╚══╝╚══╝ ",
        "",
        "      meshthatworks · dashboard  ",
        "",
    ];
    let pct = s
        .splash_started_at
        .map(|t| (t.elapsed().as_millis() as f32 / SPLASH_DURATION.as_millis() as f32).min(1.0))
        .unwrap_or(1.0);
    let spinner = spinner_char(s.spinner_frame);
    let status = match (s.node.is_some(), pct >= 1.0) {
        (true, _) => "connected".to_string(),
        (false, true) => "waiting for /status…".to_string(),
        (false, _) => format!("{spinner} booting"),
    };

    let mut lines: Vec<Line> = Vec::new();
    for l in logo {
        lines.push(Line::from(Span::styled(
            l.to_string(),
            Style::default().fg(ACCENT),
        )));
    }
    lines.push(Line::from(Span::styled(
        format!("   {status}"),
        Style::default().fg(MUTED),
    )));
    lines.push(Line::from(""));

    // bar
    let bar_w: usize = 30;
    let fill = (bar_w as f32 * pct) as usize;
    let mut bar = String::from("   ");
    bar.push('[');
    bar.push_str(&"█".repeat(fill));
    bar.push_str(&"░".repeat(bar_w.saturating_sub(fill)));
    bar.push(']');
    bar.push(' ');
    bar.push_str(&format!("{}%", (pct * 100.0) as u32));
    lines.push(Line::from(Span::styled(bar, Style::default().fg(ACCENT_DIM))));

    let text = Text::from(lines);
    let w = 46;
    let h = text.lines.len() as u16 + 2;
    let cx = area.width.saturating_sub(w) / 2;
    let cy = area.height.saturating_sub(h) / 2;
    let centre = Rect {
        x: cx,
        y: cy,
        width: w.min(area.width),
        height: h.min(area.height),
    };
    f.render_widget(Paragraph::new(text).alignment(ratatui::layout::Alignment::Left), centre);
}

fn render_tabbar(f: &mut ratatui::Frame, area: Rect, s: &SharedState) {
    let titles: Vec<Line> = Tab::ORDER
        .iter()
        .enumerate()
        .map(|(i, t)| {
            Line::from(vec![
                Span::styled(format!(" {} ", i + 1), Style::default().fg(MUTED)),
                Span::styled(
                    t.title(),
                    Style::default().fg(FG).add_modifier(Modifier::BOLD),
                ),
                Span::raw(" "),
            ])
        })
        .collect();

    let selected = s.tab.map(|t| t.index()).unwrap_or(0);

    let block = Block::default()
        .borders(Borders::BOTTOM)
        .border_type(BorderType::Plain)
        .border_style(Style::default().fg(ACCENT_DIM))
        .title(Line::from(vec![
            Span::styled("  ▎mtw ", Style::default().fg(ACCENT).add_modifier(Modifier::BOLD)),
            Span::styled("dashboard ", Style::default().fg(MUTED)),
        ]));

    let tabs = Tabs::new(titles)
        .select(selected)
        .style(Style::default().fg(MUTED))
        .highlight_style(
            Style::default()
                .bg(BG_ALT)
                .fg(ACCENT)
                .add_modifier(Modifier::BOLD),
        )
        .divider(Span::styled("│", Style::default().fg(ACCENT_DIM)))
        .block(block);
    f.render_widget(tabs, area);
}

fn render_footer(f: &mut ratatui::Frame, area: Rect, s: &SharedState) {
    let spinner = if s.streaming {
        Span::styled(
            format!("{} streaming", spinner_char(s.spinner_frame)),
            Style::default().fg(WARN),
        )
    } else {
        Span::styled("● idle", Style::default().fg(OK))
    };

    let status = Line::from(vec![
        Span::styled("  mesh: ", Style::default().fg(MUTED)),
        Span::raw(format!("{} peers", s.peers.len())),
        Span::styled("   proxy: ", Style::default().fg(MUTED)),
        Span::raw(match &s.node {
            Some(_) => "● up",
            None => "○ down",
        }),
        Span::styled("   last /status: ", Style::default().fg(MUTED)),
        Span::raw(rel_instant(&s.last_status_refresh)),
        Span::styled("   state: ", Style::default().fg(MUTED)),
        spinner,
    ]);

    let hints = Line::from(vec![
        Span::styled("  ← → / Tab", Style::default().fg(ACCENT)),
        Span::styled(" switch  ", Style::default().fg(MUTED)),
        Span::styled("?", Style::default().fg(ACCENT)),
        Span::styled(" help  ", Style::default().fg(MUTED)),
        Span::styled("Ctrl-R", Style::default().fg(ACCENT)),
        Span::styled(" refresh  ", Style::default().fg(MUTED)),
        Span::styled("Ctrl-C", Style::default().fg(ACCENT)),
        Span::styled(" quit", Style::default().fg(MUTED)),
    ]);

    f.render_widget(
        Paragraph::new(vec![status, hints])
            .block(Block::default().borders(Borders::TOP).border_style(Style::default().fg(ACCENT_DIM))),
        area,
    );
}

// ---------------------------------------------------------- tab: dashboard

fn render_tab_dashboard(f: &mut ratatui::Frame, area: Rect, s: &SharedState) {
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(10), Constraint::Min(6)])
        .split(area);

    let top = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(60), Constraint::Percentage(40)])
        .split(rows[0]);

    render_panel_self(f, top[0], s);
    render_panel_peers_summary(f, top[1], s);
    render_panel_activity(f, rows[1], s);
}

fn render_panel_self(f: &mut ratatui::Frame, area: Rect, s: &SharedState) {
    let block = rounded_block(" Self ");
    let inner = block.inner(area);
    f.render_widget(block, area);

    let lines: Vec<Line> = if let Some(n) = &s.node {
        let uptime = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs().saturating_sub(n.started_at_unix))
            .unwrap_or(0);
        vec![
            kv_line("endpoint id", &short_id(&n.endpoint_id)),
            kv_line("proxy", &n.proxy_url),
            kv_line("upstream", &n.upstream_url),
            kv_line(
                "model",
                &format!(
                    "{}  ({} layers · hidden {})",
                    n.model.name, n.model.num_layers, n.model.hidden_size
                ),
            ),
            kv_line("ALPNs", &n.alpns.join("  ·  ")),
            kv_line("uptime", &format_duration(Duration::from_secs(uptime))),
            kv_line("mtw version", &n.version),
        ]
    } else if let Some(err) = &s.node_error {
        vec![
            Line::from(Span::styled(
                format!("  {} cannot reach /status", spinner_char(s.spinner_frame)),
                Style::default().fg(ERR),
            )),
            Line::from(Span::styled(
                format!("    {}", truncate(err.clone(), 60)),
                Style::default().fg(MUTED),
            )),
            Line::from(""),
            Line::from(Span::styled(
                "    run `mtw serve` in another terminal.",
                Style::default().fg(MUTED),
            )),
        ]
    } else {
        vec![Line::from(Span::styled(
            format!("  {} polling /status …", spinner_char(s.spinner_frame)),
            Style::default().fg(WARN),
        ))]
    };

    f.render_widget(
        Paragraph::new(lines).wrap(Wrap { trim: false }),
        inner.inner(Margin { horizontal: 1, vertical: 0 }),
    );
}

fn render_panel_peers_summary(f: &mut ratatui::Frame, area: Rect, s: &SharedState) {
    let title = format!(" Peers ({}) ", s.peers.len());
    let block = rounded_block(&title);
    let inner = block.inner(area);
    f.render_widget(block, area);

    let mut lines: Vec<Line> = Vec::new();
    if s.peers.is_empty() {
        lines.push(Line::from(Span::styled(
            "  no peers paired",
            Style::default().fg(MUTED),
        )));
        lines.push(Line::from(Span::styled(
            "  mtw pair / mtw join <invite>",
            Style::default().fg(MUTED).add_modifier(Modifier::DIM),
        )));
    } else {
        for peer in &s.peers {
            let h = s.peer_health.get(&peer.id);
            let short = short_id(&peer.id);
            let (dot, colour, rtt) = match h {
                Some(h) if h.last_rtt_ms.is_some() => (
                    "●",
                    OK,
                    format!("{:>4}ms", h.last_rtt_ms.unwrap()),
                ),
                Some(_) => ("○", ERR, "  -  ".into()),
                None => ("○", WARN, "  …  ".into()),
            };
            lines.push(Line::from(vec![
                Span::styled(format!("  {dot}  "), Style::default().fg(colour)),
                Span::raw(short),
                Span::styled(format!("   {rtt}"), Style::default().fg(MUTED)),
            ]));
        }
    }

    f.render_widget(
        Paragraph::new(lines).wrap(Wrap { trim: false }),
        inner.inner(Margin { horizontal: 1, vertical: 0 }),
    );
}

fn render_panel_activity(f: &mut ratatui::Frame, area: Rect, s: &SharedState) {
    let block = rounded_block(" Activity ");
    let inner = block.inner(area);
    f.render_widget(block, area);

    let h = inner.height as usize;
    let mut lines: Vec<Line> = Vec::with_capacity(h);
    for a in s.activity.iter().take(h.saturating_sub(1)) {
        let colour = match a.kind {
            ActivityKind::Info => MUTED,
            ActivityKind::Ok => OK,
            ActivityKind::Warn => WARN,
            ActivityKind::Err => ERR,
        };
        let icon = match a.kind {
            ActivityKind::Info => "•",
            ActivityKind::Ok => "✓",
            ActivityKind::Warn => "!",
            ActivityKind::Err => "✗",
        };
        let age = rel_instant(&Some(a.at));
        lines.push(Line::from(vec![
            Span::styled(format!("  {icon} "), Style::default().fg(colour)),
            Span::styled(format!("{age:<7}  "), Style::default().fg(MUTED)),
            Span::raw(a.text.clone()),
        ]));
    }
    if lines.is_empty() {
        lines.push(Line::from(Span::styled(
            "  (nothing yet — events will appear here as the mesh runs)",
            Style::default().fg(MUTED),
        )));
    }
    f.render_widget(
        Paragraph::new(lines).wrap(Wrap { trim: false }),
        inner.inner(Margin { horizontal: 1, vertical: 0 }),
    );
}

// ---------------------------------------------------------- tab: chat

fn render_tab_chat(f: &mut ratatui::Frame, area: Rect, s: &SharedState) {
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(6), Constraint::Length(3)])
        .split(area);

    // chat history
    let block = rounded_block(" Chat ");
    let inner = block.inner(rows[0]);
    f.render_widget(block, rows[0]);

    let width = inner.width.saturating_sub(10) as usize;
    let mut lines: Vec<Line> = Vec::new();
    for turn in &s.chat {
        let (prefix, colour) = if turn.role == "user" {
            (" you ▸  ", OK)
        } else {
            (" asst ◂ ", ACCENT)
        };
        let wrapped = wrap_text(&turn.content, width);
        for (i, l) in wrapped.iter().enumerate() {
            let p = if i == 0 {
                Span::styled(
                    prefix.to_string(),
                    Style::default().fg(colour).add_modifier(Modifier::BOLD),
                )
            } else {
                Span::styled("        ", Style::default().fg(colour))
            };
            lines.push(Line::from(vec![p, Span::raw(l.clone())]));
        }
        if turn.role == "assistant" {
            if let (Some(n), Some(ms)) = (turn.tokens, turn.elapsed_ms) {
                let rate = n as f64 / (ms as f64 / 1000.0).max(0.01);
                lines.push(Line::from(Span::styled(
                    format!("         {n} tok · {:.1}s · {:.2} tok/s", ms as f64 / 1000.0, rate),
                    Style::default().fg(MUTED),
                )));
            } else if s.streaming && std::ptr::eq(turn, s.chat.last().unwrap_or(turn)) {
                let sp = spinner_char(s.spinner_frame);
                lines.push(Line::from(Span::styled(
                    format!("         {sp} streaming…"),
                    Style::default().fg(WARN),
                )));
            }
        }
        lines.push(Line::from(""));
    }
    if lines.is_empty() {
        lines.push(Line::from(Span::styled(
            "  type a prompt below and press Enter",
            Style::default().fg(MUTED),
        )));
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            "  Ctrl-L clears history · Ctrl-C quits",
            Style::default().fg(MUTED).add_modifier(Modifier::DIM),
        )));
    }
    // Keep tail that fits
    let h = inner.height as usize;
    let start = lines.len().saturating_sub(h);
    let lines = lines[start..].to_vec();
    f.render_widget(
        Paragraph::new(lines).wrap(Wrap { trim: false }),
        inner.inner(Margin { horizontal: 1, vertical: 0 }),
    );

    // input
    let input_title = if let Some(e) = &s.chat_error {
        format!(" Input · error: {} ", truncate(e.clone(), 50))
    } else {
        " Input ".into()
    };
    let input_block = rounded_block(&input_title);
    let input_inner = input_block.inner(rows[1]);
    f.render_widget(input_block, rows[1]);
    let (prompt_text, prompt_style) = if s.streaming {
        (" wait   ", Style::default().fg(WARN))
    } else {
        (" you ▸  ", Style::default().fg(OK).add_modifier(Modifier::BOLD))
    };
    let cursor = if s.streaming {
        spinner_char(s.spinner_frame).to_string()
    } else {
        "▎".into()
    };
    let line = Line::from(vec![
        Span::styled(prompt_text.to_string(), prompt_style),
        Span::raw(s.input.clone()),
        Span::styled(
            cursor,
            Style::default().fg(ACCENT).add_modifier(Modifier::SLOW_BLINK),
        ),
    ]);
    f.render_widget(
        Paragraph::new(line),
        input_inner.inner(Margin { horizontal: 1, vertical: 0 }),
    );
}

// ---------------------------------------------------------- tab: peers

fn render_tab_peers(f: &mut ratatui::Frame, area: Rect, s: &SharedState) {
    let block = rounded_block(" Peers ");
    let inner = block.inner(area);
    f.render_widget(block, area);

    let header = Line::from(Span::styled(
        format!(
            "  {:<24}{:<10}{:>10}  {}",
            "peer id", "status", "rtt", "model (via mtw/health/0)"
        ),
        Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
    ));
    let mut lines = vec![header, Line::from("")];
    if s.peers.is_empty() {
        lines.push(Line::from(Span::styled(
            "  no peers paired yet.",
            Style::default().fg(MUTED),
        )));
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            "  to pair:   run `mtw pair` on one device → copy the invite",
            Style::default().fg(MUTED),
        )));
        lines.push(Line::from(Span::styled(
            "              run `mtw join <invite>` on the other device",
            Style::default().fg(MUTED),
        )));
    } else {
        for peer in &s.peers {
            let h = s.peer_health.get(&peer.id);
            let short = short_id(&peer.id);
            let (dot, colour, label, rtt, note) = match h {
                Some(h) if h.last_rtt_ms.is_some() => (
                    "●",
                    OK,
                    "UP",
                    format!("{}ms", h.last_rtt_ms.unwrap()),
                    h.last_model.clone().unwrap_or_else(|| "(unknown)".into()),
                ),
                Some(h) => (
                    "○",
                    ERR,
                    "DOWN",
                    "-".into(),
                    truncate(h.last_error.clone().unwrap_or_default(), 50),
                ),
                None => (
                    "○",
                    WARN,
                    "…",
                    "-".into(),
                    "pinging…".into(),
                ),
            };
            lines.push(Line::from(vec![
                Span::raw(format!("  {:<24}", short)),
                Span::styled(format!("{dot} {:<6}", label), Style::default().fg(colour)),
                Span::styled(format!("{rtt:>10}"), Style::default().fg(MUTED)),
                Span::raw("  "),
                Span::raw(note),
            ]));
        }
    }

    f.render_widget(
        Paragraph::new(lines).wrap(Wrap { trim: false }),
        inner.inner(Margin { horizontal: 1, vertical: 0 }),
    );
}

// ---------------------------------------------------------- tab: models

fn render_tab_models(f: &mut ratatui::Frame, area: Rect, s: &SharedState) {
    let block = rounded_block(" Models ");
    let inner = block.inner(area);
    f.render_widget(block, area);

    let mut lines = vec![
        Line::from(Span::styled(
            format!("  {:<3}  {:<60}  {}", "#", "id", "owned_by"),
            Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
        )),
        Line::from(""),
    ];
    if let Some(err) = &s.models_error {
        lines.push(Line::from(Span::styled(
            format!("  error: {}", truncate(err.clone(), 80)),
            Style::default().fg(ERR),
        )));
    }
    if s.models.is_empty() && s.models_error.is_none() {
        lines.push(Line::from(Span::styled(
            format!("  {} fetching /v1/models…", spinner_char(s.spinner_frame)),
            Style::default().fg(MUTED),
        )));
    }
    for (i, m) in s.models.iter().enumerate() {
        lines.push(Line::from(vec![
            Span::styled(
                format!("  {:<3}  ", i + 1),
                Style::default().fg(MUTED),
            ),
            Span::styled(
                format!("{:<60}", truncate(m.id.clone(), 60)),
                Style::default().fg(FG),
            ),
            Span::styled(
                format!("  {}", m.owned_by),
                Style::default().fg(MUTED),
            ),
        ]));
    }
    f.render_widget(
        Paragraph::new(lines).wrap(Wrap { trim: false }),
        inner.inner(Margin { horizontal: 1, vertical: 0 }),
    );
}

// ---------------------------------------------------------- tab: help

fn render_tab_help(f: &mut ratatui::Frame, area: Rect) {
    let block = rounded_block(" Help ");
    let inner = block.inner(area);
    f.render_widget(block, area);

    let rows: Vec<(&str, &str)> = vec![
        ("← / → or Tab / Shift-Tab", "cycle tabs  (works inside Chat too)"),
        ("1 / 2 / 3 / 4 / 5", "jump to Dashboard / Chat / Peers / Models / Help"),
        ("Enter (Chat)", "send prompt"),
        ("Ctrl-L (Chat)", "clear chat history"),
        ("Ctrl-R", "force an immediate peer-ping round"),
        ("?  or  F1", "toggle this help overlay"),
        ("Esc or Ctrl-C or q", "quit"),
    ];

    let mut lines = vec![
        Line::from(Span::styled(
            "  Keybindings",
            Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
        )),
        Line::from(""),
    ];
    for (k, v) in rows {
        lines.push(Line::from(vec![
            Span::styled(format!("  {:<22}", k), Style::default().fg(ACCENT)),
            Span::raw(v.to_string()),
        ]));
    }
    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled(
        "  Tips",
        Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
    )));
    lines.push(Line::from(""));
    lines.push(Line::from(
        "  - If /status is red, check `mtw serve` is running in another terminal.",
    ));
    lines.push(Line::from(
        "  - Peers are paired once via `mtw pair` + `mtw join <invite>`; reflected here.",
    ));
    lines.push(Line::from(
        "  - Chat streams over the local :9337 proxy, which forwards to SwiftLM.",
    ));
    f.render_widget(
        Paragraph::new(lines).wrap(Wrap { trim: false }),
        inner.inner(Margin { horizontal: 1, vertical: 0 }),
    );
}

fn render_help_overlay(f: &mut ratatui::Frame, area: Rect) {
    let w = 62_u16.min(area.width.saturating_sub(4));
    let h = 14_u16.min(area.height.saturating_sub(4));
    let r = Rect {
        x: (area.width.saturating_sub(w)) / 2,
        y: (area.height.saturating_sub(h)) / 2,
        width: w,
        height: h,
    };
    f.render_widget(Clear, r);
    render_tab_help(f, r);
}

// ---------------------------------------------------------- helpers

fn rounded_block(title: &str) -> Block<'_> {
    Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(ACCENT_DIM))
        .title(Span::styled(
            title.to_string(),
            Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
        ))
        .title_alignment(ratatui::layout::Alignment::Left)
}

fn kv_line(k: &str, v: &str) -> Line<'static> {
    Line::from(vec![
        Span::styled(format!("  {:<13}", k), Style::default().fg(MUTED)),
        Span::styled(v.to_string(), Style::default().fg(FG)),
    ])
}

fn short_id(id: &str) -> String {
    if id.len() > 22 {
        format!("{}…{}", &id[..14], &id[id.len() - 4..])
    } else {
        id.to_string()
    }
}

fn truncate(s: String, max: usize) -> String {
    if s.len() <= max {
        s
    } else {
        format!("{}…", &s[..max.saturating_sub(1)])
    }
}

fn rel_instant(i: &Option<Instant>) -> String {
    match i {
        None => "n/a".into(),
        Some(t) => {
            let secs = t.elapsed().as_secs_f32();
            if secs < 1.0 {
                "just now".into()
            } else if secs < 60.0 {
                format!("{:.0}s ago", secs)
            } else if secs < 3600.0 {
                format!("{:.0}m ago", secs / 60.0)
            } else {
                format!("{:.0}h ago", secs / 3600.0)
            }
        }
    }
}

fn format_duration(d: Duration) -> String {
    let s = d.as_secs();
    if s < 60 {
        format!("{s}s")
    } else if s < 3600 {
        format!("{}m {}s", s / 60, s % 60)
    } else {
        format!("{}h {}m", s / 3600, (s % 3600) / 60)
    }
}

fn spinner_char(frame: usize) -> char {
    const FRAMES: &[char] = &['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];
    FRAMES[frame % FRAMES.len()]
}

fn wrap_text(text: &str, width: usize) -> Vec<String> {
    if width == 0 {
        return vec![text.to_string()];
    }
    let mut out = Vec::new();
    for paragraph in text.split('\n') {
        if paragraph.is_empty() {
            out.push(String::new());
            continue;
        }
        let mut current = String::new();
        for word in paragraph.split(' ') {
            if current.is_empty() {
                current.push_str(word);
            } else if current.len() + 1 + word.len() <= width {
                current.push(' ');
                current.push_str(word);
            } else {
                out.push(std::mem::take(&mut current));
                current.push_str(word);
            }
        }
        if !current.is_empty() {
            out.push(current);
        }
    }
    out
}
