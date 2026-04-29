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
//!   ← / →             cycle tabs       (filter category on Models tab)
//!   ↑ / ↓             move cursor      (Models tab)
//!   D                 delete model     (Models tab, installed only)
//!   Enter             send chat prompt  (Chat tab)
//!   any char          typed into input  (Chat tab)
//!   Ctrl-L            clear chat
//!   Ctrl-O            launch opencode against our proxy (Chat tab)
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
    pair::PairSession,
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

use crate::catalog::{CATALOG, CatalogModel, Category, Compat};

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
    /// When set, the dashboard signals this Notify whenever it changes the
    /// active model. The supervisor that spawned the engine listens and
    /// restarts SwiftLM with the new model — the user gets a live swap
    /// instead of having to ctrl-C and rerun.
    pub engine_restart: Option<Arc<tokio::sync::Notify>>,
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

/// Chat tab modes. `Chat` is plain LLM dialog; `Code` runs an in-UI agent
/// loop that lets the model read/write files and run shell commands in the
/// dashboard's working directory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum ChatMode {
    #[default]
    Chat,
    Code,
}

impl ChatMode {
    fn label(&self) -> &'static str {
        match self {
            ChatMode::Chat => "Chat",
            ChatMode::Code => "Code",
        }
    }
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

#[derive(Debug, Clone)]
enum Overlay {
    None,
    Help,
    Pair {
        invite: Option<String>,
        status: String,
        done: Option<String>, // Some(peer_id) on success
        error: Option<String>,
    },
    Join {
        input: String,
        status: String,
        error: Option<String>,
        in_flight: bool,
    },
    DeleteConfirm {
        dir_name: String,
        size_bytes: u64,
    },
}

#[derive(Debug, Clone)]
pub struct InstalledModel {
    pub dir_name: String,
    pub abs_path: std::path::PathBuf,
    pub size_bytes: u64,
}

impl Default for Overlay {
    fn default() -> Self {
        Overlay::None
    }
}

#[derive(Default)]
struct SharedState {
    tab: Option<Tab>, // None during splash
    splash_started_at: Option<Instant>,
    overlay: Overlay,

    node: Option<NodeStatus>,
    node_error: Option<String>,

    peers: Vec<Peer>,
    peers_error: Option<String>,
    peer_health: HashMap<String, PeerHealth>,

    models: Vec<ModelRow>,
    models_error: Option<String>,

    installed: Vec<InstalledModel>,
    /// Index into a flat (installed ++ filtered_catalog) list for selection
    models_cursor: usize,
    models_filter: Category,
    /// `~/.mtw/active-model` value at the last refresh — the model the next
    /// `mtw start` will load. May differ from `node.model.name` (which is
    /// the model the *running* engine has loaded).
    active_model_dir: Option<std::path::PathBuf>,
    /// Set by the parent `mtw dashboard` runner when it owns the engine
    /// supervisor. The model picker pings it on a successful set so the
    /// supervisor restarts SwiftLM with the new model.
    engine_restart: Option<Arc<tokio::sync::Notify>>,

    chat: Vec<ChatTurn>,
    input: String,
    streaming: bool,
    chat_error: Option<String>,

    activity: VecDeque<Activity>,

    last_status_refresh: Option<Instant>,
    last_ping_round: Option<Instant>,
    last_models_refresh: Option<Instant>,

    spinner_frame: usize,

    /// Toggled with Ctrl-O. In `Code` mode, the chat runs as an in-UI agent
    /// loop with file/shell tools instead of a plain LLM chat.
    chat_mode: ChatMode,

    /// Result of the connectivity self-check (run once at dashboard startup).
    /// `None` while the probe is still in flight.
    connectivity: Option<crate::doctor::Report>,
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
            overlay: self.overlay.clone(),
            node: self.node.clone(),
            node_error: self.node_error.clone(),
            peers: self.peers.clone(),
            peers_error: self.peers_error.clone(),
            peer_health: self.peer_health.clone(),
            models: self.models.clone(),
            models_error: self.models_error.clone(),
            installed: self.installed.clone(),
            models_cursor: self.models_cursor,
            models_filter: self.models_filter,
            active_model_dir: self.active_model_dir.clone(),
            engine_restart: self.engine_restart.clone(),
            chat: self.chat.clone(),
            input: self.input.clone(),
            streaming: self.streaming,
            chat_error: self.chat_error.clone(),
            activity: self.activity.clone(),
            last_status_refresh: self.last_status_refresh,
            last_ping_round: self.last_ping_round,
            last_models_refresh: self.last_models_refresh,
            spinner_frame: self.spinner_frame,
            chat_mode: self.chat_mode,
            connectivity: self.connectivity.clone(),
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
    let state = Arc::new(Mutex::new({
        let mut s = SharedState::new();
        s.engine_restart = args.engine_restart.clone();
        s
    }));

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

    // One-shot connectivity probe at startup. Runs in the background so the
    // dashboard frame paints immediately; the badge updates ~5–10s later when
    // STUN + HTTP probes finish.
    {
        let state = state.clone();
        tokio::spawn(async move {
            state.lock().await.push_activity(
                ActivityKind::Info,
                "connectivity: probing IPv6 / IPv4 / NAT…",
            );
            let report = crate::doctor::collect().await;
            let summary = report.short_summary();
            let mut s = state.lock().await;
            let kind = match report.connectivity() {
                crate::doctor::Connectivity::Direct => ActivityKind::Ok,
                crate::doctor::Connectivity::Relay => ActivityKind::Warn,
                crate::doctor::Connectivity::NoInternet => ActivityKind::Err,
            };
            s.push_activity(kind, format!("connectivity: {summary}"));
            s.connectivity = Some(report);
        });
    }

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
    let (in_chat, in_overlay, overlay_kind) = {
        let s = state.lock().await;
        let kind = match &s.overlay {
            Overlay::None => 0u8,
            Overlay::Help => 1,
            Overlay::Pair { .. } => 2,
            Overlay::Join { .. } => 3,
            Overlay::DeleteConfirm { .. } => 4,
        };
        (s.tab == Some(Tab::Chat), kind != 0, kind)
    };

    // When an overlay is open, route input to it. Only Esc and Ctrl-C escape.
    if in_overlay {
        if ctrl && matches!(key.code, KeyCode::Char('c')) {
            return true;
        }
        if matches!(key.code, KeyCode::Esc) {
            state.lock().await.overlay = Overlay::None;
            return false;
        }
        // DeleteConfirm overlay: Y confirms, N cancels.
        if overlay_kind == 4 {
            match key.code {
                KeyCode::Char('y') | KeyCode::Char('Y') => {
                    confirm_delete(state.clone()).await;
                }
                KeyCode::Char('n') | KeyCode::Char('N') | KeyCode::Enter => {
                    state.lock().await.overlay = Overlay::None;
                }
                _ => {}
            }
            return false;
        }
        match overlay_kind {
            2 => {
                // Pair overlay — nothing to type, Esc closes handled above.
                return false;
            }
            3 => {
                // Join overlay — text input.
                match key.code {
                    KeyCode::Backspace => {
                        let mut s = state.lock().await;
                        if let Overlay::Join { input, .. } = &mut s.overlay {
                            input.pop();
                        }
                    }
                    KeyCode::Enter => {
                        submit_join(state.clone()).await;
                    }
                    KeyCode::Char(c) => {
                        let mut s = state.lock().await;
                        if let Overlay::Join { input, in_flight, .. } = &mut s.overlay {
                            if !*in_flight {
                                input.push(c);
                            }
                        }
                    }
                    _ => {}
                }
                return false;
            }
            _ => return false, // Help overlay: just Esc.
        }
    }

    // Global keys
    match key.code {
        KeyCode::Char('c') if ctrl => return true,
        KeyCode::Esc => return true,
        KeyCode::Char('?') | KeyCode::F(1) => {
            state.lock().await.overlay = Overlay::Help;
            return false;
        }
        KeyCode::Tab => {
            let mut s = state.lock().await;
            if let Some(t) = s.tab {
                let i = (t.index() + 1) % Tab::ORDER.len();
                s.tab = Some(Tab::ORDER[i]);
            }
            return false;
        }
        KeyCode::BackTab => {
            let mut s = state.lock().await;
            if let Some(t) = s.tab {
                let i = (t.index() + Tab::ORDER.len() - 1) % Tab::ORDER.len();
                s.tab = Some(Tab::ORDER[i]);
            }
            return false;
        }
        KeyCode::Right => {
            let mut s = state.lock().await;
            if s.tab == Some(Tab::Models) {
                let cur = Category::FILTERS
                    .iter()
                    .position(|c| *c == s.models_filter)
                    .unwrap_or(0);
                s.models_filter = Category::FILTERS[(cur + 1) % Category::FILTERS.len()];
                s.models_cursor = 0;
            } else if let Some(t) = s.tab {
                let i = (t.index() + 1) % Tab::ORDER.len();
                s.tab = Some(Tab::ORDER[i]);
            }
            return false;
        }
        KeyCode::Left => {
            let mut s = state.lock().await;
            if s.tab == Some(Tab::Models) {
                let cur = Category::FILTERS
                    .iter()
                    .position(|c| *c == s.models_filter)
                    .unwrap_or(0);
                let n = Category::FILTERS.len();
                s.models_filter = Category::FILTERS[(cur + n - 1) % n];
                s.models_cursor = 0;
            } else if let Some(t) = s.tab {
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
        KeyCode::Char('o') if ctrl && in_chat => {
            // Ctrl-O cycles between Chat and Code modes (in-UI agentic coding).
            let mut s = state.lock().await;
            s.chat_mode = match s.chat_mode {
                ChatMode::Chat => ChatMode::Code,
                ChatMode::Code => ChatMode::Chat,
            };
            let label = s.chat_mode.label();
            s.push_activity(ActivityKind::Info, format!("chat mode → {label}"));
            return false;
        }
        _ => {}
    }

    // Models-tab actions
    let in_models = state.lock().await.tab == Some(Tab::Models);
    if in_models {
        match key.code {
            KeyCode::Up => {
                let mut s = state.lock().await;
                s.models_cursor = s.models_cursor.saturating_sub(1);
                return false;
            }
            KeyCode::Down => {
                let mut s = state.lock().await;
                let avail = CATALOG.iter().filter(|m| m.matches(s.models_filter)).count();
                let total = s.installed.len() + avail;
                if total > 0 {
                    s.models_cursor = (s.models_cursor + 1).min(total - 1);
                }
                return false;
            }
            KeyCode::Char('d') | KeyCode::Char('D') => {
                let s = state.lock().await;
                if s.models_cursor < s.installed.len() {
                    let m = &s.installed[s.models_cursor];
                    let dir_name = m.dir_name.clone();
                    let size_bytes = m.size_bytes;
                    drop(s);
                    state.lock().await.overlay = Overlay::DeleteConfirm {
                        dir_name,
                        size_bytes,
                    };
                }
                return false;
            }
            KeyCode::Enter => {
                let s = state.lock().await;
                let cursor = s.models_cursor;
                if cursor < s.installed.len() {
                    // Installed model — set as active.
                    let m = s.installed[cursor].clone();
                    drop(s);
                    set_active_model_action(state.clone(), m.abs_path, m.dir_name).await;
                } else {
                    // Catalog model — download then set as active.
                    let avail: Vec<CatalogModel> = CATALOG
                        .iter()
                        .copied()
                        .filter(|m| m.matches(s.models_filter))
                        .collect();
                    let idx = cursor - s.installed.len();
                    if let Some(m) = avail.get(idx).copied() {
                        drop(s);
                        download_and_activate(state.clone(), m).await;
                    }
                }
                return false;
            }
            _ => {}
        }
    }

    // Peers-tab actions: P to pair, J to join, X to forget DOWN peers.
    let in_peers = state.lock().await.tab == Some(Tab::Peers);
    if in_peers {
        if let KeyCode::Char('p') | KeyCode::Char('P') = key.code {
            start_pair(state.clone()).await;
            return false;
        }
        if let KeyCode::Char('j') | KeyCode::Char('J') = key.code {
            state.lock().await.overlay = Overlay::Join {
                input: String::new(),
                status: "paste the invite string from the other device, then Enter".into(),
                error: None,
                in_flight: false,
            };
            return false;
        }
        if let KeyCode::Char('x') | KeyCode::Char('X') = key.code {
            forget_down_peers(state.clone()).await;
            return false;
        }
    }

    // Digits jump tabs, unless typing in Chat input.
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

    // Chat input.
    if in_chat {
        match key.code {
            KeyCode::Backspace => {
                state.lock().await.input.pop();
            }
            KeyCode::Enter => {
                if handle_slash_command(state.clone()).await {
                    return false;
                }
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

/// Slash commands typed in the chat input — Mac-friendly mode toggling and
/// utilities. Returns true if the input was a slash command (and was handled),
/// so the caller skips the LLM dispatch.
///
/// Only **known** commands trigger handling, so absolute paths like
/// `/Users/foo/bar` (e.g. dragged-in files) fall through to the normal
/// chat path and are attachment-detected there.
async fn handle_slash_command(state: Arc<Mutex<SharedState>>) -> bool {
    let raw = {
        let s = state.lock().await;
        s.input.trim().to_string()
    };
    if !raw.starts_with('/') {
        return false;
    }
    let body = raw.trim_start_matches('/');
    // First "token" is what's before any whitespace OR another '/'. That way
    // /Users/foo's first token is "Users", not the whole path.
    let first = body
        .split(|c: char| c.is_whitespace() || c == '/')
        .next()
        .unwrap_or("")
        .to_lowercase();
    let known = matches!(
        first.as_str(),
        "code" | "chat" | "clear" | "help" | "?"
    );
    if !known {
        return false;
    }
    let mut s = state.lock().await;
    s.input.clear();
    match first.as_str() {
        "code" => {
            s.chat_mode = ChatMode::Code;
            s.push_activity(ActivityKind::Info, "chat mode → Code");
        }
        "chat" => {
            s.chat_mode = ChatMode::Chat;
            s.push_activity(ActivityKind::Info, "chat mode → Chat");
        }
        "clear" => {
            s.chat.clear();
            s.chat_error = None;
            s.push_activity(ActivityKind::Info, "chat cleared");
        }
        "help" | "?" => {
            s.overlay = Overlay::Help;
        }
        _ => unreachable!(),
    }
    true
}

// ---------------------------------------------------------- pair / join

/// Remove every peer that has been pinged at least once and is currently DOWN
/// (no successful RTT in `peer_health`). Persists the new list to
/// `~/.mtw/peers.json` and updates the dashboard's in-memory state. Peers that
/// have not been pinged yet are kept — they might just be slow to come up.
async fn forget_down_peers(state: Arc<Mutex<SharedState>>) {
    let stale_ids: Vec<String> = {
        let s = state.lock().await;
        s.peers
            .iter()
            .filter(|p| {
                matches!(
                    s.peer_health.get(&p.id),
                    Some(h) if h.last_rtt_ms.is_none()
                )
            })
            .map(|p| p.id.clone())
            .collect()
    };
    if stale_ids.is_empty() {
        let mut s = state.lock().await;
        s.push_activity(
            ActivityKind::Info,
            "no DOWN peers to forget — every peer is either UP or hasn't been pinged yet",
        );
        return;
    }
    let mut removed = 0usize;
    for id in &stale_ids {
        match mtw_core::peers::remove(id) {
            Ok(true) => removed += 1,
            Ok(false) => {}
            Err(e) => {
                let mut s = state.lock().await;
                s.push_activity(
                    ActivityKind::Err,
                    format!("forget peer {}: {e}", short_id(id)),
                );
            }
        }
    }
    let mut s = state.lock().await;
    s.peers.retain(|p| !stale_ids.contains(&p.id));
    for id in &stale_ids {
        s.peer_health.remove(id);
    }
    s.push_activity(
        ActivityKind::Ok,
        format!("forgot {removed} DOWN peer{}", if removed == 1 { "" } else { "s" }),
    );
}

async fn start_pair(state: Arc<Mutex<SharedState>>) {
    {
        let mut s = state.lock().await;
        if matches!(s.overlay, Overlay::Pair { .. }) {
            return; // already running
        }
        s.overlay = Overlay::Pair {
            invite: None,
            status: "binding pairing endpoint…".into(),
            done: None,
            error: None,
        };
        s.push_activity(ActivityKind::Info, "pair: starting session");
    }

    tokio::spawn(async move {
        let secret = match mtw_core::identity::load_or_create() {
            Ok(s) => s,
            Err(e) => {
                let mut st = state.lock().await;
                st.overlay = Overlay::Pair {
                    invite: None,
                    status: String::new(),
                    done: None,
                    error: Some(format!("identity: {e:#}")),
                };
                return;
            }
        };
        match PairSession::start(secret).await {
            Ok(session) => {
                let invite = session.invite.clone();
                {
                    let mut st = state.lock().await;
                    st.overlay = Overlay::Pair {
                        invite: Some(invite.clone()),
                        status: "waiting for a peer to join…".into(),
                        done: None,
                        error: None,
                    };
                    st.push_activity(ActivityKind::Ok, format!("pair: invite ready"));
                }
                match session.wait_for_peer().await {
                    Ok(peer_id) => {
                        let mut st = state.lock().await;
                        st.overlay = Overlay::Pair {
                            invite: Some(invite),
                            status: "paired".into(),
                            done: Some(peer_id.to_string()),
                            error: None,
                        };
                        st.push_activity(
                            ActivityKind::Ok,
                            format!("pair: peer {} joined", short_id(&peer_id.to_string())),
                        );
                    }
                    Err(e) => {
                        let mut st = state.lock().await;
                        st.overlay = Overlay::Pair {
                            invite: Some(invite),
                            status: String::new(),
                            done: None,
                            error: Some(format!("{e:#}")),
                        };
                        st.push_activity(ActivityKind::Err, format!("pair: {e}"));
                    }
                }
            }
            Err(e) => {
                let mut st = state.lock().await;
                st.overlay = Overlay::Pair {
                    invite: None,
                    status: String::new(),
                    done: None,
                    error: Some(format!("{e:#}")),
                };
                st.push_activity(ActivityKind::Err, format!("pair: {e}"));
            }
        }
    });
}

async fn confirm_delete(state: Arc<Mutex<SharedState>>) {
    let dir_name = {
        let s = state.lock().await;
        match &s.overlay {
            Overlay::DeleteConfirm { dir_name, .. } => dir_name.clone(),
            _ => return,
        }
    };
    let home = match std::env::var_os("HOME") {
        Some(h) => h,
        None => return,
    };
    let path = std::path::Path::new(&home)
        .join("Desktop/meshthatworks-deps/models")
        .join(&dir_name);
    let result = tokio::task::spawn_blocking({
        let path = path.clone();
        move || std::fs::remove_dir_all(&path)
    })
    .await;
    let mut s = state.lock().await;
    match result {
        Ok(Ok(())) => {
            s.installed.retain(|m| m.dir_name != dir_name);
            if s.models_cursor >= s.installed.len() + CATALOG.len() {
                s.models_cursor = s.installed.len().saturating_sub(1);
            }
            s.push_activity(
                ActivityKind::Ok,
                format!("deleted {}", dir_name),
            );
        }
        Ok(Err(e)) => {
            s.push_activity(
                ActivityKind::Err,
                format!("delete {} failed: {}", dir_name, e),
            );
        }
        Err(e) => {
            s.push_activity(
                ActivityKind::Err,
                format!("delete task panicked: {}", e),
            );
        }
    }
    s.overlay = Overlay::None;
}

// ---------------------------------------------------------- model picker

/// Persist `path` as the active model (`~/.mtw/active-model`). Logs to the
/// activity feed. Restart `mtw start` to actually load it — the engine is
/// constructed once at startup, so a live swap is a future feature.
async fn set_active_model_action(
    state: Arc<Mutex<SharedState>>,
    path: std::path::PathBuf,
    dir_name: String,
) {
    match mtw_core::active_model::set(&path) {
        Ok(()) => {
            let restart = {
                let mut s = state.lock().await;
                let restart = s.engine_restart.clone();
                let msg = if restart.is_some() {
                    format!("active model: {} — restarting engine…", dir_name)
                } else {
                    format!(
                        "active model: {} — restart `mtw start` to load it",
                        dir_name
                    )
                };
                s.push_activity(ActivityKind::Ok, msg);
                restart
            };
            // Ping the supervisor outside the state lock — it will abort the
            // current engine task and respawn SwiftLM with the new model.
            if let Some(n) = restart {
                n.notify_one();
            }
        }
        Err(e) => {
            let mut s = state.lock().await;
            s.push_activity(
                ActivityKind::Err,
                format!("set active model: {e}"),
            );
        }
    }
}

/// Download a catalog model from Hugging Face into
/// `~/.meshthatworks-deps/models/<dir>/`, then set it as active. Streams
/// each file with progress lines into the activity feed. Tolerates files
/// (e.g. `generation_config.json`) that may not exist for every repo.
async fn download_and_activate(state: Arc<Mutex<SharedState>>, m: CatalogModel) {
    let dest = match std::env::var_os("HOME") {
        Some(home) => std::path::PathBuf::from(home)
            .join(".meshthatworks-deps/models")
            .join(m.dir_name),
        None => {
            state.lock().await.push_activity(
                ActivityKind::Err,
                "$HOME not set — cannot pick install dir",
            );
            return;
        }
    };

    if dest.join("config.json").is_file() && dest.join("model.safetensors").is_file() {
        // Already there — just set active.
        set_active_model_action(state.clone(), dest, m.dir_name.into()).await;
        return;
    }

    state.lock().await.push_activity(
        ActivityKind::Info,
        format!("downloading {} (~{:.1} GB) — this happens once", m.name, m.size_gb),
    );

    let state_clone = state.clone();
    tokio::spawn(async move {
        if let Err(e) = std::fs::create_dir_all(&dest) {
            state_clone.lock().await.push_activity(
                ActivityKind::Err,
                format!("create model dir: {e}"),
            );
            return;
        }
        match list_hf_files(m.hf_repo).await {
            Ok(files) => {
                let total_files = files.len();
                for (i, file) in files.iter().enumerate() {
                    let dest_file = dest.join(file);
                    if dest_file.is_file() {
                        continue;
                    }
                    let url = format!(
                        "https://huggingface.co/{}/resolve/main/{}",
                        m.hf_repo, file
                    );
                    state_clone.lock().await.push_activity(
                        ActivityKind::Info,
                        format!("  [{}/{}] fetching {file}", i + 1, total_files),
                    );
                    if let Err(e) = stream_to_file(&url, &dest_file).await {
                        state_clone.lock().await.push_activity(
                            ActivityKind::Err,
                            format!("  download {file}: {e}"),
                        );
                        return;
                    }
                }
                state_clone.lock().await.push_activity(
                    ActivityKind::Ok,
                    format!("downloaded {}", m.name),
                );
                set_active_model_action(state_clone, dest, m.dir_name.into()).await;
            }
            Err(e) => {
                state_clone.lock().await.push_activity(
                    ActivityKind::Err,
                    format!("list files for {}: {e}", m.hf_repo),
                );
            }
        }
    });
}

/// Hit `https://huggingface.co/api/models/<repo>/tree/main` and return the
/// list of file paths. Skips entries that are directories.
async fn list_hf_files(repo: &str) -> anyhow::Result<Vec<String>> {
    #[derive(serde::Deserialize)]
    struct Entry {
        #[serde(rename = "type")]
        kind: String,
        path: String,
    }
    let url = format!("https://huggingface.co/api/models/{repo}/tree/main");
    let entries: Vec<Entry> = reqwest::Client::new()
        .get(&url)
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    Ok(entries
        .into_iter()
        .filter(|e| e.kind == "file")
        .map(|e| e.path)
        .collect())
}

/// Stream a URL to `dest`. Atomic via .part suffix + rename so a crashed
/// download leaves no half-file at the destination.
async fn stream_to_file(url: &str, dest: &std::path::Path) -> anyhow::Result<()> {
    use tokio::io::AsyncWriteExt;
    let part = dest.with_extension("part");
    let resp = reqwest::Client::new()
        .get(url)
        .send()
        .await?
        .error_for_status()?;
    let mut stream = resp.bytes_stream();
    let mut file = tokio::fs::File::create(&part).await?;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        file.write_all(&chunk).await?;
    }
    file.flush().await?;
    drop(file);
    tokio::fs::rename(&part, dest).await?;
    Ok(())
}

async fn submit_join(state: Arc<Mutex<SharedState>>) {
    let invite = {
        let mut s = state.lock().await;
        match &mut s.overlay {
            Overlay::Join { input, in_flight, error, status } => {
                if *in_flight || input.trim().is_empty() {
                    return;
                }
                *in_flight = true;
                *error = None;
                *status = "contacting peer…".into();
                input.trim().to_string()
            }
            _ => return,
        }
    };

    tokio::spawn(async move {
        let secret = match mtw_core::identity::load_or_create() {
            Ok(s) => s,
            Err(e) => {
                let mut st = state.lock().await;
                if let Overlay::Join { in_flight, error, .. } = &mut st.overlay {
                    *in_flight = false;
                    *error = Some(format!("identity: {e:#}"));
                }
                return;
            }
        };
        match mtw_core::pair::join(secret, &invite).await {
            Ok(()) => {
                let mut st = state.lock().await;
                st.overlay = Overlay::None;
                st.push_activity(ActivityKind::Ok, "join: paired successfully");
            }
            Err(e) => {
                let mut st = state.lock().await;
                if let Overlay::Join { in_flight, error, status, .. } = &mut st.overlay {
                    *in_flight = false;
                    *error = Some(format!("{e:#}"));
                    *status = "try again or Esc to cancel".into();
                }
                st.push_activity(ActivityKind::Err, format!("join: {e}"));
            }
        }
    });
}

// ---------------------------------------------------------- send + stream

async fn send_prompt(state: Arc<Mutex<SharedState>>, url: String) {
    let mode = state.lock().await.chat_mode;
    match mode {
        ChatMode::Chat => run_chat_turn(state, url).await,
        ChatMode::Code => run_agent_turn(state, url).await,
    }
}

/// Result of preparing a user prompt: any absolute paths in the raw input
/// that exist on disk get pulled in as `<file>`/`<dir>` context blocks, so
/// drag-and-drop "just works" the way it does in opencode/Cursor.
struct PreparedPrompt {
    /// Final text inlined into the chat history (and seen by the model).
    text: String,
    /// How many paths were attached — purely for the activity feed.
    attached: usize,
}

const ATTACH_FILE_BUDGET_BYTES: usize = 16 * 1024;
const ATTACH_DIR_ENTRY_LIMIT: usize = 200;

async fn prepare_prompt(raw: &str) -> PreparedPrompt {
    // Tokenize on whitespace. macOS terminal drag pastes one path per drop,
    // separated by spaces. Backslash-escaped spaces in paths (`/Users/foo\ bar`)
    // would split incorrectly here — we accept that as a known limitation,
    // since Desktop drops in this app are usually unescaped.
    let mut attachments: Vec<String> = Vec::new();
    let mut other_words: Vec<&str> = Vec::new();

    for token in raw.split_whitespace() {
        // Strip a trailing colon/comma a user may have typed after pasting.
        let cleaned = token.trim_end_matches([':', ',']);
        if cleaned.starts_with('/') && std::path::Path::new(cleaned).exists() {
            attachments.push(cleaned.to_string());
        } else {
            other_words.push(token);
        }
    }

    let cleaned_prompt = other_words.join(" ");

    if attachments.is_empty() {
        return PreparedPrompt {
            text: raw.to_string(),
            attached: 0,
        };
    }

    let mut preamble = String::from("[Attached for context]\n\n");
    for path_str in &attachments {
        let p = std::path::Path::new(path_str);
        let is_dir = p.is_dir();
        if is_dir {
            preamble.push_str(&format!("<dir path=\"{path_str}\">\n"));
            match tokio::fs::read_dir(p).await {
                Ok(mut rd) => {
                    let mut entries = Vec::new();
                    while let Ok(Some(e)) = rd.next_entry().await {
                        let is_dir = e
                            .file_type()
                            .await
                            .map(|t| t.is_dir())
                            .unwrap_or(false);
                        let n = e.file_name().to_string_lossy().to_string();
                        entries.push(if is_dir { format!("{n}/") } else { n });
                        if entries.len() >= ATTACH_DIR_ENTRY_LIMIT {
                            entries.push(format!(
                                "...(more entries truncated at {ATTACH_DIR_ENTRY_LIMIT})"
                            ));
                            break;
                        }
                    }
                    entries.sort();
                    preamble.push_str(&entries.join("\n"));
                }
                Err(e) => preamble.push_str(&format!("(error: {e})")),
            }
            preamble.push_str("\n</dir>\n\n");
        } else {
            preamble.push_str(&format!("<file path=\"{path_str}\">\n"));
            match tokio::fs::read(p).await {
                Ok(bytes) => {
                    // Skip obvious binary blobs.
                    let is_text = std::str::from_utf8(&bytes).is_ok();
                    if !is_text {
                        preamble.push_str(&format!(
                            "(binary file, {} bytes — skipped)",
                            bytes.len()
                        ));
                    } else if bytes.len() > ATTACH_FILE_BUDGET_BYTES {
                        let head = &bytes[..ATTACH_FILE_BUDGET_BYTES];
                        preamble.push_str(&String::from_utf8_lossy(head));
                        preamble.push_str(&format!(
                            "\n...[truncated: {ATTACH_FILE_BUDGET_BYTES} of {} bytes shown]",
                            bytes.len()
                        ));
                    } else {
                        preamble.push_str(&String::from_utf8_lossy(&bytes));
                    }
                }
                Err(e) => preamble.push_str(&format!("(error: {e})")),
            }
            preamble.push_str("\n</file>\n\n");
        }
    }

    let final_question = if cleaned_prompt.is_empty() {
        "What do you make of this? Help me understand or work with it.".to_string()
    } else {
        cleaned_prompt
    };

    PreparedPrompt {
        text: format!("{preamble}{final_question}"),
        attached: attachments.len(),
    }
}

async fn run_chat_turn(state: Arc<Mutex<SharedState>>, url: String) {
    let raw = {
        let mut s = state.lock().await;
        if s.streaming || s.input.trim().is_empty() {
            return;
        }
        std::mem::take(&mut s.input)
    };
    let prepared = prepare_prompt(&raw).await;
    if prepared.attached > 0 {
        state.lock().await.push_activity(
            ActivityKind::Info,
            format!("attached {} path(s) to prompt", prepared.attached),
        );
    }

    let (model_name, history, prompt_text) = {
        let mut s = state.lock().await;
        let prompt = prepared.text;
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
            .map(|t| serde_json::json!({
                "role": if t.role == "tool_result" { "user" } else { t.role.as_str() },
                "content": t.content,
            }))
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
                let raw = format!("{e:#}");
                let pretty = friendly_chat_error(&raw);
                s.chat_error = Some(pretty.clone());
                if let Some(last) = s.chat.last_mut() {
                    if last.role == "assistant" && last.content.is_empty() {
                        last.content = pretty.clone();
                    }
                }
                s.push_activity(ActivityKind::Err, format!("chat error: {pretty}"));
            }
        }
    });
}

const AGENT_MAX_STEPS: usize = 12;

fn agent_system_prompt(cwd: &str) -> String {
    format!(
        "You are MeshThatWorks Code, an in-terminal coding agent.\n\
         Working directory: {cwd}\n\
         \n\
         You have these tools. To call one, emit its tag in your reply. The runtime will\n\
         execute it and feed the output back as the next user message. Then you decide what\n\
         to do next.\n\
         \n\
           <read path=\"relative/path.ext\"/>           — show file contents\n\
           <list path=\"relative/dir\"/>                — list a directory\n\
           <bash>shell command</bash>                  — run a command in the working dir\n\
           <write path=\"relative/path.ext\">           — replace (or create) a file with the\n\
           <full new file contents go here>             body between the tags. ALWAYS include\n\
           </write>                                     the entire file, never a diff.\n\
           <done/>                                     — emit when the task is complete.\n\
         \n\
         Rules:\n\
         - One tool per reply. Think briefly, then call exactly one tool, OR emit <done/>.\n\
         - Read files before editing them. Don't fabricate paths or contents.\n\
         - After <bash>, look at stdout/stderr/exit. After <read>, you'll see the file body.\n\
         - Use <bash>cargo build</bash> or similar to verify changes.\n\
         - Keep prose outside of tags short — the user mostly cares about the result.\n\
         \n\
         Begin.\n"
    )
}

async fn run_agent_turn(state: Arc<Mutex<SharedState>>, url: String) {
    let raw = {
        let mut s = state.lock().await;
        if s.streaming || s.input.trim().is_empty() {
            return;
        }
        std::mem::take(&mut s.input)
    };
    let prepared = prepare_prompt(&raw).await;
    if prepared.attached > 0 {
        state.lock().await.push_activity(
            ActivityKind::Info,
            format!("attached {} path(s) to prompt", prepared.attached),
        );
    }

    let (model_name, prompt_text) = {
        let mut s = state.lock().await;
        let prompt = prepared.text;
        let prompt_clone = prompt.clone();
        s.chat.push(ChatTurn {
            role: "user".into(),
            content: prompt,
            tokens: None,
            elapsed_ms: None,
        });
        s.streaming = true;
        s.chat_error = None;
        s.push_activity(
            ActivityKind::Info,
            format!("code: {}", truncate(prompt_clone.clone(), 60)),
        );

        let model_name = s
            .node
            .as_ref()
            .map(|n| n.model.name.clone())
            .unwrap_or_else(|| "mesh".into());
        (model_name, prompt_clone)
    };

    let cwd = std::env::current_dir()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|_| ".".into());
    let system = agent_system_prompt(&cwd);

    let state_clone = state.clone();
    let url_clone = url.clone();
    tokio::spawn(async move {
        let mut steps = 0usize;
        let mut last_err: Option<String> = None;

        loop {
            if steps >= AGENT_MAX_STEPS {
                state_clone.lock().await.push_activity(
                    ActivityKind::Warn,
                    format!("agent: hit step cap ({AGENT_MAX_STEPS}) — stopping"),
                );
                break;
            }
            steps += 1;

            // Build the message history for this iteration.
            let history = {
                let s = state_clone.lock().await;
                let mut msgs = vec![serde_json::json!({"role": "system", "content": system})];
                for t in &s.chat {
                    if t.content.is_empty() || t.content.starts_with("[error:") {
                        continue;
                    }
                    let role = match t.role.as_str() {
                        "tool_result" => "user",
                        "assistant" => "assistant",
                        _ => "user",
                    };
                    msgs.push(serde_json::json!({"role": role, "content": t.content}));
                }
                msgs
            };

            // Push an empty assistant turn that stream_chat will fill.
            state_clone.lock().await.chat.push(ChatTurn {
                role: "assistant".into(),
                content: String::new(),
                tokens: None,
                elapsed_ms: None,
            });

            let started = Instant::now();
            let res = stream_chat(
                url_clone.clone(),
                model_name.clone(),
                history,
                state_clone.clone(),
            )
            .await;
            let elapsed = started.elapsed().as_millis();

            let assistant_text = {
                let mut s = state_clone.lock().await;
                match res {
                    Ok(tok) => {
                        if let Some(last) = s.chat.last_mut() {
                            last.tokens = Some(tok);
                            last.elapsed_ms = Some(elapsed);
                        }
                    }
                    Err(e) => {
                        let pretty = friendly_chat_error(&format!("{e:#}"));
                        if let Some(last) = s.chat.last_mut() {
                            if last.role == "assistant" && last.content.is_empty() {
                                last.content = pretty.clone();
                            }
                        }
                        last_err = Some(pretty);
                        break;
                    }
                }
                s.chat.last().map(|t| t.content.clone()).unwrap_or_default()
            };

            // Parse for tool calls. Stop if <done/> or no tools.
            if assistant_text.contains("<done/>")
                || assistant_text.contains("<done />")
                || assistant_text.contains("<done>")
            {
                state_clone
                    .lock()
                    .await
                    .push_activity(ActivityKind::Ok, format!("agent: done in {steps} step(s)"));
                break;
            }

            let calls = parse_tool_calls(&assistant_text);
            if calls.is_empty() {
                // No tool, no <done/> — model just talked. End the turn.
                state_clone.lock().await.push_activity(
                    ActivityKind::Info,
                    "agent: no tool call in reply — stopping",
                );
                break;
            }

            // Execute each tool call in order, push its result as a tool_result turn.
            for call in calls {
                let label = call.label();
                state_clone
                    .lock()
                    .await
                    .push_activity(ActivityKind::Info, format!("tool: {label}"));
                let result = execute_tool(&call).await;
                let body = format!("<tool_result for=\"{label}\">\n{result}\n</tool_result>");
                state_clone.lock().await.chat.push(ChatTurn {
                    role: "tool_result".into(),
                    content: body,
                    tokens: None,
                    elapsed_ms: None,
                });
            }
        }

        let mut s = state_clone.lock().await;
        s.streaming = false;
        if let Some(e) = last_err {
            s.chat_error = Some(e.clone());
            s.push_activity(ActivityKind::Err, format!("agent error: {e}"));
        }
        let _ = prompt_text;
    });
}

#[derive(Debug)]
enum ToolCall {
    Read { path: String },
    List { path: String },
    Bash { cmd: String },
    Write { path: String, body: String },
}

impl ToolCall {
    fn label(&self) -> &'static str {
        match self {
            ToolCall::Read { .. } => "read",
            ToolCall::List { .. } => "list",
            ToolCall::Bash { .. } => "bash",
            ToolCall::Write { .. } => "write",
        }
    }
}

fn parse_tool_calls(text: &str) -> Vec<ToolCall> {
    let mut out = Vec::new();
    // Self-closing: <read path="..."/>  and  <list path="..."/>
    for tag in ["read", "list"] {
        let needle = format!("<{tag}");
        let mut cursor = 0usize;
        while let Some(found) = text[cursor..].find(&needle) {
            let start = cursor + found;
            let after = start + needle.len();
            let rest = &text[after..];
            let close = rest.find("/>").map(|i| (i, 2usize));
            let close = close.or_else(|| rest.find('>').map(|i| (i, 1usize)));
            let Some((rel_end, close_len)) = close else {
                break;
            };
            let attrs = &rest[..rel_end];
            let path = extract_attr(attrs, "path").unwrap_or_default();
            if !path.is_empty() {
                out.push(match tag {
                    "read" => ToolCall::Read { path: path.into() },
                    "list" => ToolCall::List { path: path.into() },
                    _ => unreachable!(),
                });
            }
            cursor = after + rel_end + close_len;
        }
    }

    // Block tags with bodies: <bash>...</bash>  and  <write path="…">...</write>
    for tag in ["bash", "write"] {
        let open = format!("<{tag}");
        let close = format!("</{tag}>");
        let mut cursor = 0usize;
        while let Some(found) = text[cursor..].find(&open) {
            let start = cursor + found;
            let after = start + open.len();
            let rest = &text[after..];
            let Some(gt) = rest.find('>') else { break };
            let attrs = &rest[..gt];
            let body_start = after + gt + 1;
            let Some(end) = text[body_start..].find(&close) else {
                break;
            };
            let body = &text[body_start..body_start + end];
            match tag {
                "bash" => out.push(ToolCall::Bash {
                    cmd: body.trim().to_string(),
                }),
                "write" => {
                    let path = extract_attr(attrs, "path").unwrap_or_default();
                    if !path.is_empty() {
                        // Strip a single leading newline from body if present.
                        let body = body.strip_prefix('\n').unwrap_or(body);
                        out.push(ToolCall::Write {
                            path: path.into(),
                            body: body.to_string(),
                        });
                    }
                }
                _ => {}
            }
            cursor = body_start + end + close.len();
        }
    }
    out
}

fn extract_attr<'a>(attrs: &'a str, key: &str) -> Option<&'a str> {
    let needle = format!("{key}=");
    let i = attrs.find(&needle)?;
    let rest = &attrs[i + needle.len()..];
    let quote = rest.chars().next()?;
    if quote != '"' && quote != '\'' {
        return None;
    }
    let body = &rest[1..];
    let end = body.find(quote)?;
    Some(&body[..end])
}

async fn execute_tool(call: &ToolCall) -> String {
    const MAX_BYTES: usize = 8 * 1024;

    fn truncate_for_model(s: String) -> String {
        if s.len() <= MAX_BYTES {
            s
        } else {
            let cut = &s[..MAX_BYTES];
            format!(
                "{cut}\n...[truncated: {} bytes shown of {}]",
                MAX_BYTES,
                s.len()
            )
        }
    }

    match call {
        ToolCall::Read { path } => match tokio::fs::read_to_string(path).await {
            Ok(body) => truncate_for_model(body),
            Err(e) => format!("error: read {path}: {e}"),
        },
        ToolCall::List { path } => match tokio::fs::read_dir(path).await {
            Ok(mut rd) => {
                let mut entries = Vec::new();
                while let Ok(Some(e)) = rd.next_entry().await {
                    let is_dir = e
                        .file_type()
                        .await
                        .map(|t| t.is_dir())
                        .unwrap_or(false);
                    let name = e.file_name().to_string_lossy().to_string();
                    entries.push(if is_dir {
                        format!("{name}/")
                    } else {
                        name
                    });
                }
                entries.sort();
                truncate_for_model(entries.join("\n"))
            }
            Err(e) => format!("error: list {path}: {e}"),
        },
        ToolCall::Bash { cmd } => {
            let output = tokio::process::Command::new("bash")
                .arg("-lc")
                .arg(cmd)
                .output()
                .await;
            match output {
                Ok(o) => {
                    let stdout = String::from_utf8_lossy(&o.stdout).to_string();
                    let stderr = String::from_utf8_lossy(&o.stderr).to_string();
                    let exit = o.status.code().unwrap_or(-1);
                    truncate_for_model(format!(
                        "exit: {exit}\n--- stdout ---\n{stdout}--- stderr ---\n{stderr}"
                    ))
                }
                Err(e) => format!("error: spawn bash: {e}"),
            }
        }
        ToolCall::Write { path, body } => {
            if let Some(parent) = std::path::Path::new(path).parent() {
                if !parent.as_os_str().is_empty() {
                    let _ = tokio::fs::create_dir_all(parent).await;
                }
            }
            match tokio::fs::write(path, body.as_bytes()).await {
                Ok(()) => format!("wrote {} bytes to {path}", body.len()),
                Err(e) => format!("error: write {path}: {e}"),
            }
        }
    }
}

/// Make a chat error human-readable. Connection-refused at :9337 is the
/// dominant case (user opened `mtw dashboard` without starting the engine);
/// the raw `tcp connect error: Connection refused (os error 61)` is useless
/// to a non-developer. Replace it with a clear pointer to `mtw start`.
fn friendly_chat_error(raw: &str) -> String {
    if raw.contains("Connection refused")
        || raw.contains("os error 61")
        || raw.contains("ConnectionRefused")
    {
        return "the engine is not running. run  mtw start  to bring it up, then try this prompt again.".into();
    }
    if raw.contains("timed out") || raw.contains("timeout") {
        return format!(
            "the engine timed out. it may be loading the model (cold start can take ~30s). try again in a moment.\n\n[underlying: {}]",
            truncate(raw.to_string(), 200)
        );
    }
    raw.to_string()
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

    // Accumulate deltas from every parseable line in a frame, then take the
    // state lock ONCE per frame to apply them — instead of `tokio::spawn`
    // per token, which (a) costs a task allocation per token and (b) loses
    // ordering when multiple spawns race for the lock.
    let mut apply = |frame_deltas: &str, state: &Arc<Mutex<SharedState>>| -> futures_util::future::BoxFuture<'_, ()> {
        let s = state.clone();
        let owned = frame_deltas.to_string();
        Box::pin(async move {
            if owned.is_empty() {
                return;
            }
            let mut st = s.lock().await;
            if let Some(last) = st.chat.last_mut() {
                last.content.push_str(&owned);
            }
        })
    };

    while let Some(chunk) = stream.next().await {
        let chunk = chunk.context("read chunk")?;
        buf.push_str(&String::from_utf8_lossy(&chunk));

        loop {
            // Find the next SSE event boundary: \n\n or \r\n\r\n.
            let mut idx = buf.find("\n\n");
            if idx.is_none() {
                idx = buf.find("\r\n\r\n").map(|i| i + 2);
            }
            let Some(i) = idx else { break };
            let frame: String = buf.drain(..=i + 1).collect();

            let mut frame_deltas = String::new();
            let mut done = false;
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
                    done = true;
                    break;
                }
                let v: Value = match serde_json::from_str(payload) {
                    Ok(v) => v,
                    Err(_) => continue,
                };
                if let Some(delta) = v["choices"][0]["delta"]["content"].as_str() {
                    tokens += 1;
                    frame_deltas.push_str(delta);
                }
            }
            apply(&frame_deltas, &state).await;
            if done {
                return Ok(tokens);
            }
        }
    }

    // Flush trailing line even without final \n\n.
    let mut frame_deltas = String::new();
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
                frame_deltas.push_str(delta);
            }
        }
    }
    apply(&frame_deltas, &state).await;
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

        // installed models on disk
        if let Some(home) = std::env::var_os("HOME") {
            let dir = std::path::Path::new(&home)
                .join(".meshthatworks-deps/models");
            if let Ok(entries) = std::fs::read_dir(&dir) {
                let mut found: Vec<InstalledModel> = Vec::new();
                for e in entries.flatten() {
                    let path = e.path();
                    if !path.is_dir() {
                        continue;
                    }
                    let dir_name = match path.file_name().and_then(|s| s.to_str()) {
                        Some(s) => s.to_string(),
                        None => continue,
                    };
                    let size = du(&path).unwrap_or(0);
                    found.push(InstalledModel {
                        dir_name,
                        abs_path: path,
                        size_bytes: size,
                    });
                }
                found.sort_by(|a, b| b.size_bytes.cmp(&a.size_bytes));
                state.lock().await.installed = found;
            }
        }

        // active model setting (`~/.mtw/active-model`)
        let active = mtw_core::active_model::load().ok().flatten();
        state.lock().await.active_model_dir = active;

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

    match &s.overlay {
        Overlay::None => {}
        Overlay::Help => render_help_overlay(f, area),
        Overlay::Pair { invite, status, done, error } => {
            render_pair_overlay(f, area, invite.as_deref(), status, done.as_deref(), error.as_deref());
        }
        Overlay::Join { input, status, error, in_flight } => {
            render_join_overlay(f, area, input, status, error.as_deref(), *in_flight, s.spinner_frame);
        }
        Overlay::DeleteConfirm { dir_name, size_bytes } => {
            render_delete_overlay(f, area, dir_name, *size_bytes);
        }
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

    let (conn_glyph, conn_text, conn_color) = match &s.connectivity {
        None => ("◌", "probing…".to_string(), MUTED),
        Some(r) => match r.connectivity() {
            crate::doctor::Connectivity::Direct => ("●", r.short_summary(), OK),
            crate::doctor::Connectivity::Relay => ("●", r.short_summary(), WARN),
            crate::doctor::Connectivity::NoInternet => ("●", r.short_summary(), ERR),
        },
    };

    let status = Line::from(vec![
        Span::styled("  mesh: ", Style::default().fg(MUTED)),
        Span::raw(format!("{} peers", s.peers.len())),
        Span::styled("   proxy: ", Style::default().fg(MUTED)),
        Span::raw(match &s.node {
            Some(_) => "● up",
            None => "○ down",
        }),
        Span::styled("   net: ", Style::default().fg(MUTED)),
        Span::styled(conn_glyph.to_string(), Style::default().fg(conn_color)),
        Span::raw(format!(" {conn_text}")),
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
        .constraints([
            Constraint::Length(10),
            Constraint::Length(7),
            Constraint::Min(4),
        ])
        .split(area);

    let top = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(60), Constraint::Percentage(40)])
        .split(rows[0]);

    render_panel_self(f, top[0], s);
    render_panel_peers_summary(f, top[1], s);
    render_panel_connectivity(f, rows[1], s);
    render_panel_activity(f, rows[2], s);
}

fn render_panel_connectivity(f: &mut ratatui::Frame, area: Rect, s: &SharedState) {
    let block = rounded_block(" Connectivity ");
    let inner = block.inner(area);
    f.render_widget(block, area);

    let lines: Vec<Line> = match &s.connectivity {
        None => vec![
            Line::from(Span::styled(
                format!(
                    "  {} probing IPv6 / IPv4 / NAT type — first pairing prediction in ~5s…",
                    spinner_char(s.spinner_frame)
                ),
                Style::default().fg(MUTED),
            )),
        ],
        Some(r) => {
            let (verdict_color, verdict_text) = match r.connectivity() {
                crate::doctor::Connectivity::Direct => (
                    OK,
                    "✓ direct connection — pairing will skip relay".to_string(),
                ),
                crate::doctor::Connectivity::Relay => (
                    WARN,
                    "⚠ relay required — your NAT blocks hole-punching, traffic goes through n0 relays"
                        .to_string(),
                ),
                crate::doctor::Connectivity::NoInternet => (
                    ERR,
                    "✗ no internet reachable — neither IPv6 nor public IPv4 works".to_string(),
                ),
            };

            let ipv6_line = match (&r.ipv6, &r.ipv6_error) {
                (Some(a), _) => format!("IPv6  ✓ {}", a),
                (None, Some(_)) => "IPv6  ✗ no v6 route".to_string(),
                (None, None) => "IPv6  ?".to_string(),
            };
            let ipv4_line = match (&r.ipv4_public, &r.ipv4_error) {
                (Some(ip), _) => {
                    let cgnat = r.cgnat.unwrap_or(false);
                    if cgnat {
                        format!("IPv4  ⚠ {ip}  (CGNAT)")
                    } else {
                        format!("IPv4  ✓ {ip}")
                    }
                }
                (None, Some(_)) => "IPv4  ✗".to_string(),
                (None, None) => "IPv4  ?".to_string(),
            };
            let nat_line = match (&r.nat_type, &r.nat_error) {
                (Some(crate::doctor::NatVerdict::EndpointIndependent { .. }), _) => {
                    "NAT   ✓ endpoint-independent — hole-punch works".to_string()
                }
                (Some(crate::doctor::NatVerdict::Symmetric { .. }), _) => {
                    "NAT   ✗ symmetric — hole-punch fails, must relay".to_string()
                }
                (None, Some(_)) => "NAT   ?".to_string(),
                (None, None) => "NAT   ?".to_string(),
            };

            vec![
                Line::from(Span::styled(
                    format!("  {verdict_text}"),
                    Style::default().fg(verdict_color).add_modifier(Modifier::BOLD),
                )),
                Line::from(""),
                Line::from(Span::styled(
                    format!("    {ipv6_line}"),
                    Style::default().fg(MUTED),
                )),
                Line::from(Span::styled(
                    format!("    {ipv4_line}"),
                    Style::default().fg(MUTED),
                )),
                Line::from(Span::styled(
                    format!("    {nat_line}"),
                    Style::default().fg(MUTED),
                )),
            ]
        }
    };

    f.render_widget(
        Paragraph::new(lines).wrap(Wrap { trim: false }),
        inner.inner(Margin { horizontal: 1, vertical: 0 }),
    );
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
    // If the engine isn't reachable, surface a banner row above the chat
    // pane. A banner is far more discoverable than a single ⊘ in the footer
    // when the user is looking at the Chat tab.
    let engine_down = s.last_status_refresh.is_some() && s.node.is_none();
    let rows = if engine_down {
        Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3), // banner
                Constraint::Min(6),
                Constraint::Length(3),
            ])
            .split(area)
    } else {
        Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Min(6), Constraint::Length(3)])
            .split(area)
    };

    let mut chat_idx = 0usize;
    let mut input_idx = 1usize;
    if engine_down {
        let banner_block = Block::default()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(Style::default().fg(ERR))
            .title(Span::styled(
                " Engine offline ",
                Style::default().fg(ERR).add_modifier(Modifier::BOLD),
            ));
        let banner_inner = banner_block.inner(rows[0]);
        f.render_widget(banner_block, rows[0]);
        let line = Line::from(vec![
            Span::styled("  the engine is not running. start it with  ", Style::default().fg(MUTED)),
            Span::styled("mtw start", Style::default().fg(ACCENT).add_modifier(Modifier::BOLD)),
            Span::styled("  (or  ", Style::default().fg(MUTED)),
            Span::styled("mtw serve", Style::default().fg(ACCENT)),
            Span::styled("  in another terminal).", Style::default().fg(MUTED)),
        ]);
        f.render_widget(
            Paragraph::new(line),
            banner_inner.inner(Margin { horizontal: 1, vertical: 0 }),
        );
        chat_idx = 1;
        input_idx = 2;
    }

    // chat history
    let title = match s.chat_mode {
        ChatMode::Chat => " Chat · type /code to switch to Code mode ".to_string(),
        ChatMode::Code => " Chat · mode: CODE (agent · reads/edits files, runs shell) · /chat to switch back ".to_string(),
    };
    let block = rounded_block(&title);
    let inner = block.inner(rows[chat_idx]);
    f.render_widget(block, rows[chat_idx]);

    let width = inner.width.saturating_sub(10) as usize;
    let mut lines: Vec<Line> = Vec::new();
    for turn in &s.chat {
        let (prefix, colour) = match turn.role.as_str() {
            "user" => (" you  ▸ ", OK),
            "tool_result" => (" tool ◂ ", WARN),
            _ => (" asst ◂ ", ACCENT),
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
        match s.chat_mode {
            ChatMode::Chat => {
                lines.push(Line::from(Span::styled(
                    "  type a prompt below and press Enter",
                    Style::default().fg(MUTED),
                )));
                lines.push(Line::from(""));
                lines.push(Line::from(Span::styled(
                    "  /code   — switch to Code mode (in-UI coding agent · file + shell tools)",
                    Style::default().fg(ACCENT).add_modifier(Modifier::DIM),
                )));
            }
            ChatMode::Code => {
                lines.push(Line::from(Span::styled(
                    "  CODE MODE — describe a coding task and the model will use tools:",
                    Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
                )));
                lines.push(Line::from(Span::styled(
                    "    <read path=…/>    <list path=…/>    <bash>…</bash>    <write path=…>…</write>    <done/>",
                    Style::default().fg(MUTED),
                )));
                lines.push(Line::from(""));
                lines.push(Line::from(Span::styled(
                    "  Tool calls run in your current directory. No confirmation prompts — be specific.",
                    Style::default().fg(WARN).add_modifier(Modifier::DIM),
                )));
                lines.push(Line::from(Span::styled(
                    "  /chat   — switch back to plain Chat mode",
                    Style::default().fg(MUTED).add_modifier(Modifier::DIM),
                )));
            }
        }
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            "  /clear clears history · /help shows all keys · Ctrl-C quits",
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
    let input_inner = input_block.inner(rows[input_idx]);
    f.render_widget(input_block, rows[input_idx]);
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
        lines.push(Line::from(vec![
            Span::raw("  "),
            Span::styled(
                "[P]",
                Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
            ),
            Span::raw("  pair this device   — shows an invite string you share with the other device"),
        ]));
        lines.push(Line::from(vec![
            Span::raw("  "),
            Span::styled(
                "[J]",
                Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
            ),
            Span::raw("  join another device — paste an invite you received"),
        ]));
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            "  on the OTHER device, run `mtw dashboard`, come to Peers, press P or J.",
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
        let any_up = s
            .peers
            .iter()
            .any(|p| matches!(s.peer_health.get(&p.id), Some(h) if h.last_rtt_ms.is_some()));
        if !any_up {
            lines.push(Line::from(""));
            lines.push(Line::from(Span::styled(
                "  every peer is DOWN — they're saved but unreachable. Re-pair, or remove with X.",
                Style::default().fg(MUTED),
            )));
        }
    }

    // Persistent keybinding footer — visible whether peers are listed or not.
    lines.push(Line::from(""));
    lines.push(Line::from(vec![
        Span::raw("  "),
        Span::styled("[P]", Style::default().fg(ACCENT).add_modifier(Modifier::BOLD)),
        Span::styled("  pair this device   ", Style::default().fg(MUTED)),
        Span::styled("[J]", Style::default().fg(ACCENT).add_modifier(Modifier::BOLD)),
        Span::styled("  join with an invite   ", Style::default().fg(MUTED)),
        Span::styled("[X]", Style::default().fg(ACCENT).add_modifier(Modifier::BOLD)),
        Span::styled("  forget every peer that is DOWN", Style::default().fg(MUTED)),
    ]));

    f.render_widget(
        Paragraph::new(lines).wrap(Wrap { trim: false }),
        inner.inner(Margin { horizontal: 1, vertical: 0 }),
    );
}

// ---------------------------------------------------------- tab: models

fn render_tab_models(f: &mut ratatui::Frame, area: Rect, s: &SharedState) {
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // filter bar
            Constraint::Min(8),    // body
        ])
        .split(area);

    // ---- filter bar
    let filter_block = rounded_block(" Filter by use case (←/→) ");
    let inner = filter_block.inner(rows[0]);
    f.render_widget(filter_block, rows[0]);
    let mut spans: Vec<Span> = Vec::new();
    for (i, c) in Category::FILTERS.iter().enumerate() {
        if i > 0 {
            spans.push(Span::styled("  ·  ", Style::default().fg(ACCENT_DIM)));
        }
        let style = if *c == s.models_filter {
            Style::default()
                .fg(ACCENT)
                .add_modifier(Modifier::BOLD)
                .bg(BG_ALT)
        } else {
            Style::default().fg(MUTED)
        };
        spans.push(Span::styled(format!(" {} ", c.label()), style));
    }
    f.render_widget(
        Paragraph::new(Line::from(spans)),
        inner.inner(Margin { horizontal: 1, vertical: 0 }),
    );

    // ---- body: installed first, then catalog filtered
    let total_disk: u64 = s.installed.iter().map(|i| i.size_bytes).sum();
    let body_title = format!(
        " Installed ({}, {}) · Available ({}) ",
        s.installed.len(),
        format_size(total_disk),
        CATALOG.iter().filter(|m| m.matches(s.models_filter)).count(),
    );
    let body_block = rounded_block(&body_title);
    let body_inner = body_block.inner(rows[1]);
    f.render_widget(body_block, rows[1]);

    let mut lines: Vec<Line> = Vec::new();

    // installed section
    lines.push(Line::from(Span::styled(
        "  ▎ INSTALLED",
        Style::default().fg(OK).add_modifier(Modifier::BOLD),
    )));
    if s.installed.is_empty() {
        lines.push(Line::from(Span::styled(
            "  (none — pull a model from the Available list below)",
            Style::default().fg(MUTED),
        )));
    } else {
        let installed_idx_end = s.installed.len();
        for (i, m) in s.installed.iter().enumerate() {
            let is_selected = i == s.models_cursor;
            let prefix = if is_selected {
                Span::styled(
                    "  ▶ ",
                    Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
                )
            } else {
                Span::raw("    ")
            };
            let loaded = s
                .node
                .as_ref()
                .map(|n| n.model.name == m.dir_name)
                .unwrap_or(false);
            let picked = s
                .active_model_dir
                .as_ref()
                .and_then(|p| p.file_name().and_then(|s| s.to_str()))
                .map(|name| name == m.dir_name)
                .unwrap_or(false);
            let active_tag = if loaded {
                Span::styled(
                    "  [loaded]",
                    Style::default().fg(OK).add_modifier(Modifier::BOLD),
                )
            } else if picked {
                Span::styled(
                    "  ★ active",
                    Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
                )
            } else {
                Span::raw("")
            };
            lines.push(Line::from(vec![
                prefix,
                Span::styled(
                    format!("{:<48}", truncate(m.dir_name.clone(), 48)),
                    Style::default().fg(FG),
                ),
                Span::styled(
                    format!(" {:>9}", format_size(m.size_bytes)),
                    Style::default().fg(MUTED),
                ),
                active_tag,
            ]));
        }
        let _ = installed_idx_end; // marker for future selection-aware code
    }

    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled(
        "  ▎ AVAILABLE (curated MoE catalog)",
        Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
    )));
    let filtered: Vec<&CatalogModel> = CATALOG.iter().filter(|m| m.matches(s.models_filter)).collect();
    let cursor_in_avail = s.models_cursor.checked_sub(s.installed.len());
    for (i, m) in filtered.iter().enumerate() {
        let installed = s.installed.iter().any(|inst| inst.dir_name == m.dir_name);
        let is_selected = cursor_in_avail.map(|c| c == i).unwrap_or(false);
        let prefix = if is_selected {
            Span::styled(
                "  ▶ ",
                Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
            )
        } else {
            Span::raw("    ")
        };
        let installed_tag = if installed {
            Span::styled(
                "  [installed]",
                Style::default().fg(OK),
            )
        } else {
            Span::raw("")
        };
        let compat_color = match m.compat {
            Compat::Recommended => OK,
            Compat::Tight => WARN,
            Compat::NeedsBigger => MUTED,
        };
        lines.push(Line::from(vec![
            prefix,
            Span::styled(
                format!("{:<28}", truncate(m.name.into(), 28)),
                Style::default().fg(FG).add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                format!(" {:>8}", format!("{:.1} GB", m.size_gb)),
                Style::default().fg(MUTED),
            ),
            Span::styled(
                format!(" · {}", m.compat.label()),
                Style::default().fg(compat_color),
            ),
            installed_tag,
        ]));
        // categories + note
        let cats: Vec<&str> = m.categories.iter().map(|c| c.label()).collect();
        lines.push(Line::from(vec![
            Span::raw("        "),
            Span::styled(
                format!("{}  ·  {}", cats.join(", "), m.note),
                Style::default().fg(MUTED),
            ),
        ]));
    }

    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled(
        "  ↑/↓ select   Enter use this model (downloads if needed)   D delete   ←/→ filter",
        Style::default().fg(MUTED),
    )));

    // Keep a window of lines that fits.
    let h = body_inner.height as usize;
    let cursor_pos = s.models_cursor;
    let estimated_line = if cursor_pos < s.installed.len() {
        cursor_pos + 1
    } else {
        s.installed.len() + 3 + (cursor_pos - s.installed.len()) * 2
    };
    let start = estimated_line.saturating_sub(h.saturating_sub(2));
    let end = (start + h).min(lines.len());
    let lines = lines[start..end].to_vec();

    f.render_widget(
        Paragraph::new(lines).wrap(Wrap { trim: false }),
        body_inner.inner(Margin { horizontal: 1, vertical: 0 }),
    );
}

// ---------------------------------------------------------- tab: help

fn render_tab_help(f: &mut ratatui::Frame, area: Rect) {
    let block = rounded_block(" Help ");
    let inner = block.inner(area);
    f.render_widget(block, area);

    let rows: Vec<(&str, &str)> = vec![
        ("Tab / Shift-Tab", "cycle tabs  (works inside Chat too)"),
        ("1 / 2 / 3 / 4 / 5", "jump to Dashboard / Chat / Peers / Models / Help"),
        ("← / →", "cycle tabs  (or filter on Models tab)"),
        ("↑ / ↓ (Models)", "move cursor through model list"),
        ("Enter (Models)", "use selected model — downloads from HF if not installed, then sets it as active for next mtw start"),
        ("D (Models)", "delete selected installed model"),
        ("Enter (Chat)", "send prompt"),
        ("/code  (Chat)", "switch to Code mode (in-UI coding agent)"),
        ("/chat  (Chat)", "switch back to plain Chat mode"),
        ("/clear (Chat)", "clear chat history"),
        ("Ctrl-L (Chat)", "clear chat history (alt)"),
        ("Ctrl-O (Chat)", "toggle Chat / Code mode (alt)"),
        ("P (Peers)", "pair this device — shows invite string"),
        ("J (Peers)", "join another device — paste an invite"),
        ("X (Peers)", "forget every peer that is currently DOWN"),
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

fn render_delete_overlay(
    f: &mut ratatui::Frame,
    area: Rect,
    dir_name: &str,
    size_bytes: u64,
) {
    let w = 80;
    let h = 11;
    let r = center_rect(area, w, h);
    f.render_widget(Clear, r);
    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(ERR))
        .title(Span::styled(
            " Delete model ",
            Style::default().fg(ERR).add_modifier(Modifier::BOLD),
        ));
    let inner = block.inner(r);
    f.render_widget(block, r);

    let lines = vec![
        Line::from(""),
        Line::from(Span::styled(
            "  This will remove the model directory and free disk space:",
            Style::default().fg(MUTED),
        )),
        Line::from(""),
        Line::from(vec![
            Span::raw("    "),
            Span::styled(
                dir_name.to_string(),
                Style::default().fg(FG).add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                format!("   ({} on disk)", format_size(size_bytes)),
                Style::default().fg(MUTED),
            ),
        ]),
        Line::from(""),
        Line::from(Span::styled(
            "  Cannot be undone — you'd need to re-download.",
            Style::default().fg(WARN),
        )),
        Line::from(""),
        Line::from(vec![
            Span::raw("    "),
            Span::styled(
                "[Y]",
                Style::default().fg(ERR).add_modifier(Modifier::BOLD),
            ),
            Span::raw(" delete   "),
            Span::styled(
                "[N / Esc]",
                Style::default().fg(OK).add_modifier(Modifier::BOLD),
            ),
            Span::raw(" cancel"),
        ]),
    ];
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

fn center_rect(area: Rect, w: u16, h: u16) -> Rect {
    let w = w.min(area.width.saturating_sub(4));
    let h = h.min(area.height.saturating_sub(4));
    Rect {
        x: (area.width.saturating_sub(w)) / 2,
        y: (area.height.saturating_sub(h)) / 2,
        width: w,
        height: h,
    }
}

fn render_pair_overlay(
    f: &mut ratatui::Frame,
    area: Rect,
    invite: Option<&str>,
    status: &str,
    done: Option<&str>,
    error: Option<&str>,
) {
    let w = 90;
    let h = 14;
    let r = center_rect(area, w, h);
    f.render_widget(Clear, r);
    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(ACCENT))
        .title(Span::styled(
            " Pair a new device ",
            Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
        ));
    let inner = block.inner(r);
    f.render_widget(block, r);

    let mut lines: Vec<Line> = Vec::new();
    if let Some(inv) = invite {
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            "  share this invite with the OTHER device:",
            Style::default().fg(MUTED),
        )));
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            format!("    {inv}"),
            Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
        )));
        lines.push(Line::from(""));
        if let Some(peer) = done {
            lines.push(Line::from(Span::styled(
                format!("  ✓ paired with {}", short_id(peer)),
                Style::default().fg(OK).add_modifier(Modifier::BOLD),
            )));
            lines.push(Line::from(Span::styled(
                "    (recorded in ~/.mtw/peers.json — Esc to close)",
                Style::default().fg(MUTED),
            )));
        } else if let Some(err) = error {
            lines.push(Line::from(Span::styled(
                format!("  ✗ {err}"),
                Style::default().fg(ERR),
            )));
        } else {
            lines.push(Line::from(Span::styled(
                format!("  {status}"),
                Style::default().fg(MUTED),
            )));
        }
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            "    on the other device, open `mtw dashboard`, go to Peers, press J,",
            Style::default().fg(MUTED),
        )));
        lines.push(Line::from(Span::styled(
            "    paste this invite, and press Enter.",
            Style::default().fg(MUTED),
        )));
    } else if let Some(err) = error {
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            format!("  ✗ failed: {err}"),
            Style::default().fg(ERR),
        )));
    } else {
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            format!("  {} {status}", spinner_char(0)),
            Style::default().fg(MUTED),
        )));
    }
    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled(
        "  Esc  cancel / close",
        Style::default().fg(MUTED),
    )));
    f.render_widget(
        Paragraph::new(lines).wrap(Wrap { trim: false }),
        inner.inner(Margin { horizontal: 1, vertical: 0 }),
    );
}

fn render_join_overlay(
    f: &mut ratatui::Frame,
    area: Rect,
    input: &str,
    status: &str,
    error: Option<&str>,
    in_flight: bool,
    spinner: usize,
) {
    let w = 90;
    let h = 12;
    let r = center_rect(area, w, h);
    f.render_widget(Clear, r);
    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(ACCENT))
        .title(Span::styled(
            " Join a device ",
            Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
        ));
    let inner = block.inner(r);
    f.render_widget(block, r);

    let cursor: &str = if in_flight { "" } else { "▎" };
    let mut lines: Vec<Line> = Vec::new();
    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled(
        "  paste the invite string (starts with `mtw-invite:`):",
        Style::default().fg(MUTED),
    )));
    lines.push(Line::from(""));
    lines.push(Line::from(vec![
        Span::styled(
            "    invite ▸ ",
            Style::default().fg(OK).add_modifier(Modifier::BOLD),
        ),
        Span::raw(input.to_string()),
        Span::styled(
            cursor.to_string(),
            Style::default().fg(ACCENT).add_modifier(Modifier::SLOW_BLINK),
        ),
    ]));
    lines.push(Line::from(""));
    let status_span = if let Some(err) = error {
        Span::styled(format!("  ✗ {err}"), Style::default().fg(ERR))
    } else if in_flight {
        Span::styled(
            format!("  {} {status}", spinner_char(spinner)),
            Style::default().fg(WARN),
        )
    } else {
        Span::styled(format!("  {status}"), Style::default().fg(MUTED))
    };
    lines.push(Line::from(status_span));
    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled(
        "  Enter submit · Esc cancel",
        Style::default().fg(MUTED),
    )));
    f.render_widget(
        Paragraph::new(lines).wrap(Wrap { trim: false }),
        inner.inner(Margin { horizontal: 1, vertical: 0 }),
    );
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

/// Recursive directory size in bytes, no symlink-following.
fn du(path: &std::path::Path) -> std::io::Result<u64> {
    let meta = std::fs::symlink_metadata(path)?;
    if meta.file_type().is_symlink() {
        return Ok(0);
    }
    if meta.is_file() {
        return Ok(meta.len());
    }
    if meta.is_dir() {
        let mut total = 0u64;
        for entry in std::fs::read_dir(path)? {
            let e = entry?;
            total = total.saturating_add(du(&e.path()).unwrap_or(0));
        }
        return Ok(total);
    }
    Ok(0)
}

fn format_size(bytes: u64) -> String {
    let gb = bytes as f64 / 1e9;
    if gb >= 1.0 {
        format!("{:.1} GB", gb)
    } else {
        format!("{:.0} MB", bytes as f64 / 1e6)
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
