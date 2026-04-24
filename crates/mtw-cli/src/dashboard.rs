//! Unified terminal dashboard for a MeshThatWorks node.
//!
//! Single view shows everything at once:
//! - **Self** (top-left) + **Peers** (top-right)  — live status
//! - **Chat** (middle) — conversation area with an input line
//! - **Status** + shortcuts (bottom)
//!
//! Background tasks keep `Self` and `Peers` fresh; the chat panel is
//! wired to `POST /v1/chat/completions` with `stream: true` on the local
//! mtw-api proxy, so replies stream in token-by-token.
//!
//! Default key bindings:
//! - Any printable character + Backspace edit the input line
//! - **Enter** sends the prompt
//! - **Ctrl-C** or **Esc** quit the dashboard
//! - **Ctrl-R** forces an immediate peer-ping round
//! - **Ctrl-L** clears the chat history

use std::collections::HashMap;
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
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Wrap},
};
use serde_json::Value;
use tokio::sync::Mutex;

pub struct DashboardArgs {
    pub url: String,
    pub tick: Duration,
    pub ping_every: Duration,
}

#[derive(Debug, Default, Clone)]
struct PeerHealth {
    last_rtt_ms: Option<u128>,
    last_model: Option<String>,
    last_error: Option<String>,
}

#[derive(Debug, Clone)]
struct ChatTurn {
    role: String,
    content: String,
    completion_tokens: Option<usize>,
    elapsed_ms: Option<u128>,
}

#[derive(Default)]
struct SharedState {
    node: Option<NodeStatus>,
    node_error: Option<String>,
    peers: Vec<Peer>,
    peers_error: Option<String>,
    peer_health: HashMap<String, PeerHealth>,
    last_status_refresh: Option<Instant>,
    last_ping_round: Option<Instant>,

    chat: Vec<ChatTurn>,
    input: String,
    streaming: bool,
    chat_error: Option<String>,
}

impl SharedState {
    fn snapshot(&self) -> SharedState {
        Self {
            node: self.node.clone(),
            node_error: self.node_error.clone(),
            peers: self.peers.clone(),
            peers_error: self.peers_error.clone(),
            peer_health: self.peer_health.clone(),
            last_status_refresh: self.last_status_refresh,
            last_ping_round: self.last_ping_round,
            chat: self.chat.clone(),
            input: self.input.clone(),
            streaming: self.streaming,
            chat_error: self.chat_error.clone(),
        }
    }
}

pub async fn run(args: DashboardArgs) -> anyhow::Result<()> {
    let state = Arc::new(Mutex::new(SharedState::default()));

    let secret = mtw_core::identity::load_or_create()?;
    let endpoint = Endpoint::builder(presets::N0)
        .secret_key(secret)
        .bind()
        .await
        .context("bind iroh endpoint for dashboard pings")?;

    let refresh_state = state.clone();
    let refresh_url = args.url.clone();
    let refresh_endpoint = endpoint.clone();
    let refresh_handle = tokio::spawn(async move {
        refresh_loop(
            refresh_state,
            refresh_url,
            refresh_endpoint,
            args.ping_every,
        )
        .await
    });

    enable_raw_mode().context("enable raw mode")?;
    let mut stdout = std::io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

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

async fn ui_loop(
    terminal: &mut Terminal<CrosstermBackend<std::io::Stdout>>,
    state: Arc<Mutex<SharedState>>,
    args: &DashboardArgs,
) -> anyhow::Result<()> {
    let mut events = EventStream::new();
    let mut ticker = tokio::time::interval(args.tick);

    loop {
        let snap = state.lock().await.snapshot();
        terminal.draw(|f| render(f, &snap))?;

        tokio::select! {
            _ = ticker.tick() => {}
            maybe = events.next() => match maybe {
                Some(Ok(Event::Key(key))) if key.kind == KeyEventKind::Press => {
                    let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
                    match key.code {
                        KeyCode::Char('c') if ctrl => return Ok(()),
                        KeyCode::Esc => return Ok(()),
                        KeyCode::Char('r') if ctrl => {
                            let mut s = state.lock().await;
                            s.last_status_refresh = None;
                            s.last_ping_round = None;
                        }
                        KeyCode::Char('l') if ctrl => {
                            let mut s = state.lock().await;
                            s.chat.clear();
                            s.chat_error = None;
                        }
                        KeyCode::Backspace => {
                            let mut s = state.lock().await;
                            s.input.pop();
                        }
                        KeyCode::Enter => {
                            send_prompt(state.clone(), args.url.clone()).await;
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
                Some(Err(e)) => tracing::warn!(%e, "event stream error"),
                _ => {}
            }
        }
    }
}

async fn send_prompt(state: Arc<Mutex<SharedState>>, url: String) {
    let (prompt, history) = {
        let mut s = state.lock().await;
        if s.streaming || s.input.trim().is_empty() {
            return;
        }
        let prompt = std::mem::take(&mut s.input);
        s.chat.push(ChatTurn {
            role: "user".into(),
            content: prompt.clone(),
            completion_tokens: None,
            elapsed_ms: None,
        });
        s.chat.push(ChatTurn {
            role: "assistant".into(),
            content: String::new(),
            completion_tokens: None,
            elapsed_ms: None,
        });
        s.streaming = true;
        s.chat_error = None;
        let history: Vec<_> = s
            .chat
            .iter()
            .filter(|t| t.role != "assistant" || !t.content.is_empty())
            .map(|t| serde_json::json!({"role": t.role, "content": t.content}))
            .collect();
        (prompt, history)
    };

    let state_clone = state.clone();
    tokio::spawn(async move {
        let result = stream_chat(url, history, state_clone.clone()).await;
        let mut s = state_clone.lock().await;
        s.streaming = false;
        if let Err(e) = result {
            s.chat_error = Some(format!("{e:#}"));
            if let Some(last) = s.chat.last_mut() {
                if last.role == "assistant" && last.content.is_empty() {
                    last.content = format!("[error: {e}]");
                }
            }
        }
        drop(prompt);
    });
}

async fn stream_chat(
    url: String,
    messages: Vec<Value>,
    state: Arc<Mutex<SharedState>>,
) -> anyhow::Result<()> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(600))
        .build()?;
    let body = serde_json::json!({
        "model": "mesh",
        "messages": messages,
        "max_tokens": 400,
        "stream": true,
    });
    let started = Instant::now();
    let resp = client
        .post(format!("{url}/v1/chat/completions"))
        .json(&body)
        .send()
        .await
        .context("POST /v1/chat/completions")?;
    let resp = resp
        .error_for_status()
        .context("server returned error status")?;
    let mut stream = resp.bytes_stream();
    let mut buf = String::new();
    let mut tokens: usize = 0;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.context("read chunk")?;
        buf.push_str(&String::from_utf8_lossy(&chunk));
        while let Some(idx) = buf.find("\n\n") {
            let frame: String = buf.drain(..=idx + 1).collect();
            for line in frame.lines() {
                let Some(payload) = line.strip_prefix("data: ") else {
                    continue;
                };
                if payload == "[DONE]" {
                    let mut s = state.lock().await;
                    if let Some(last) = s.chat.last_mut() {
                        last.completion_tokens = Some(tokens);
                        last.elapsed_ms = Some(started.elapsed().as_millis());
                    }
                    return Ok(());
                }
                let v: Value = match serde_json::from_str(payload) {
                    Ok(v) => v,
                    Err(_) => continue,
                };
                if let Some(delta) = v["choices"][0]["delta"]["content"].as_str() {
                    tokens += 1;
                    let mut s = state.lock().await;
                    if let Some(last) = s.chat.last_mut() {
                        last.content.push_str(delta);
                    }
                }
            }
        }
    }
    let mut s = state.lock().await;
    if let Some(last) = s.chat.last_mut() {
        last.completion_tokens = Some(tokens);
        last.elapsed_ms = Some(started.elapsed().as_millis());
    }
    Ok(())
}

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

        match client.get(format!("{url}/status")).send().await {
            Ok(r) if r.status().is_success() => match r.json::<NodeStatus>().await {
                Ok(n) => {
                    let mut s = state.lock().await;
                    s.node = Some(n);
                    s.node_error = None;
                    s.last_status_refresh = Some(Instant::now());
                }
                Err(e) => state.lock().await.node_error = Some(format!("parse /status: {e}")),
            },
            Ok(r) => state.lock().await.node_error = Some(format!("/status http {}", r.status())),
            Err(e) => state.lock().await.node_error = Some(format!("/status unreachable: {e}")),
        }

        match mtw_core::peers::load() {
            Ok(PeerList { peers }) => {
                let mut s = state.lock().await;
                s.peers = peers;
                s.peers_error = None;
            }
            Err(e) => state.lock().await.peers_error = Some(format!("{e}")),
        }

        let should_ping = {
            let s = state.lock().await;
            s.last_ping_round.map(|t| t.elapsed() >= ping_every).unwrap_or(true)
        };
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
                                ..PeerHealth::default()
                            });
                        continue;
                    }
                };
                let result = ping_peer(&endpoint, id, Duration::from_secs(3)).await;
                let mut s = state.lock().await;
                let slot = s.peer_health.entry(peer.id.clone()).or_default();
                match result {
                    Ok((Pong { model_info, .. }, rtt)) => {
                        slot.last_rtt_ms = Some(rtt.as_millis());
                        slot.last_model = model_info.map(|m| m.name);
                        slot.last_error = None;
                    }
                    Err(e) => {
                        slot.last_rtt_ms = None;
                        slot.last_error = Some(format!("{e}"));
                    }
                }
            }
            state.lock().await.last_ping_round = Some(Instant::now());
        }
    }
}

// --------------------------------------------------- render

fn render(f: &mut ratatui::Frame, s: &SharedState) {
    let area = f.area();
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(2),  // title
            Constraint::Length(10), // self + peers row
            Constraint::Min(8),     // chat area
            Constraint::Length(3),  // input line
            Constraint::Length(3),  // status + hints
        ])
        .split(area);

    render_title(f, rows[0]);
    let top = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(rows[1]);
    render_self(f, top[0], s);
    render_peers(f, top[1], s);
    render_chat(f, rows[2], s);
    render_input(f, rows[3], s);
    render_footer(f, rows[4], s);
}

fn render_title(f: &mut ratatui::Frame, area: Rect) {
    let now = humantime::format_rfc3339_seconds(SystemTime::now()).to_string();
    let title = Line::from(vec![
        Span::styled(
            "meshthatworks ",
            Style::default()
                .add_modifier(Modifier::BOLD)
                .fg(Color::Cyan),
        ),
        Span::raw("· dashboard "),
        Span::styled(now, Style::default().add_modifier(Modifier::DIM)),
    ]);
    f.render_widget(
        Paragraph::new(title).block(Block::default().borders(Borders::BOTTOM)),
        area,
    );
}

fn render_self(f: &mut ratatui::Frame, area: Rect, s: &SharedState) {
    let lines: Vec<Line> = if let Some(n) = &s.node {
        let since = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs().saturating_sub(n.started_at_unix))
            .unwrap_or(0);
        vec![
            kv("endpoint id", &short_id(&n.endpoint_id)),
            kv("proxy", &n.proxy_url),
            kv("upstream", &n.upstream_url),
            kv(
                "model",
                &format!(
                    "{}  ({} layers, hidden={})",
                    n.model.name, n.model.num_layers, n.model.hidden_size
                ),
            ),
            kv("ALPNs", &n.alpns.join(", ")),
            kv("uptime", &format_duration(Duration::from_secs(since))),
            kv("mtw version", &n.version),
        ]
    } else if let Some(err) = &s.node_error {
        vec![
            Line::from(Span::styled(
                "cannot reach /status",
                Style::default().fg(Color::Red),
            )),
            Line::from(Span::styled(
                err.clone(),
                Style::default().add_modifier(Modifier::DIM),
            )),
        ]
    } else {
        vec![Line::from(Span::styled(
            "polling /status …",
            Style::default().add_modifier(Modifier::DIM),
        ))]
    };
    f.render_widget(
        Paragraph::new(lines)
            .block(Block::default().borders(Borders::ALL).title(" Self "))
            .wrap(Wrap { trim: false }),
        area,
    );
}

fn render_peers(f: &mut ratatui::Frame, area: Rect, s: &SharedState) {
    let title = format!(" Peers ({}) ", s.peers.len());
    let header = Line::from(Span::styled(
        format!(
            "  {:<22}{:<8}{:>7}  {}",
            "peer id", "status", "rtt", "model"
        ),
        Style::default().add_modifier(Modifier::BOLD),
    ));
    let mut lines = vec![header];
    if s.peers.is_empty() {
        lines.push(Line::from(Span::styled(
            "  no peers paired. `mtw pair` / `mtw join <invite>`",
            Style::default().add_modifier(Modifier::DIM),
        )));
    } else {
        for peer in &s.peers {
            let short = short_id(&peer.id);
            let h = s.peer_health.get(&peer.id);
            let (status_span, rtt_span, note) = match h {
                Some(h) if h.last_rtt_ms.is_some() => (
                    Span::styled("● UP  ", Style::default().fg(Color::Green)),
                    Span::raw(format!("{:>5}ms", h.last_rtt_ms.unwrap())),
                    h.last_model.clone().unwrap_or_else(|| "(no model)".into()),
                ),
                Some(h) => (
                    Span::styled("○ DOWN", Style::default().fg(Color::Red)),
                    Span::styled("     - ", Style::default().add_modifier(Modifier::DIM)),
                    truncate(h.last_error.clone().unwrap_or_default(), 40),
                ),
                None => (
                    Span::styled("…ping ", Style::default().fg(Color::Yellow)),
                    Span::styled("     - ", Style::default().add_modifier(Modifier::DIM)),
                    "first round".into(),
                ),
            };
            lines.push(Line::from(vec![
                Span::raw(format!("  {:<22}", short)),
                status_span,
                Span::raw(" "),
                rtt_span,
                Span::raw("  "),
                Span::raw(truncate(note, 40)),
            ]));
        }
    }
    f.render_widget(
        Paragraph::new(lines)
            .block(Block::default().borders(Borders::ALL).title(title))
            .wrap(Wrap { trim: false }),
        area,
    );
}

fn render_chat(f: &mut ratatui::Frame, area: Rect, s: &SharedState) {
    let inner_height = area.height.saturating_sub(2) as usize; // minus borders
    let mut lines: Vec<Line> = Vec::new();
    for turn in &s.chat {
        let prefix = if turn.role == "user" {
            Span::styled("you ▸ ", Style::default().fg(Color::Green).add_modifier(Modifier::BOLD))
        } else {
            Span::styled("asst ◂ ", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD))
        };
        let body_lines = wrap_text(&turn.content, area.width.saturating_sub(8) as usize);
        for (i, body) in body_lines.iter().enumerate() {
            if i == 0 {
                lines.push(Line::from(vec![prefix.clone(), Span::raw(body.clone())]));
            } else {
                lines.push(Line::from(vec![
                    Span::raw("       "),
                    Span::raw(body.clone()),
                ]));
            }
        }
        if turn.role == "assistant" {
            if let (Some(n), Some(ms)) = (turn.completion_tokens, turn.elapsed_ms) {
                let rate = n as f64 / (ms as f64 / 1000.0).max(0.01);
                lines.push(Line::from(Span::styled(
                    format!("       [{n} tok, {:.1}s, {:.2} tok/s]", ms as f64 / 1000.0, rate),
                    Style::default().add_modifier(Modifier::DIM),
                )));
            } else if s.streaming
                && std::ptr::eq(turn, s.chat.last().unwrap_or(turn))
            {
                lines.push(Line::from(Span::styled(
                    "       ⟂ streaming…",
                    Style::default().fg(Color::Yellow),
                )));
            }
        }
        lines.push(Line::from(""));
    }
    if lines.is_empty() {
        lines.push(Line::from(Span::styled(
            "  (type a prompt below and press Enter)",
            Style::default().add_modifier(Modifier::DIM),
        )));
    }
    // Keep only the tail that fits.
    let start = lines.len().saturating_sub(inner_height);
    let lines = lines[start..].to_vec();
    f.render_widget(
        Paragraph::new(lines)
            .block(Block::default().borders(Borders::ALL).title(" Chat "))
            .wrap(Wrap { trim: false }),
        area,
    );
}

fn render_input(f: &mut ratatui::Frame, area: Rect, s: &SharedState) {
    let prompt = if s.streaming { "wait  " } else { "you ▸ " };
    let prompt_style = if s.streaming {
        Style::default().fg(Color::Yellow)
    } else {
        Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)
    };
    let cursor = if s.streaming { "" } else { "▎" };
    let line = Line::from(vec![
        Span::styled(prompt, prompt_style),
        Span::raw(s.input.clone()),
        Span::styled(
            cursor,
            Style::default().fg(Color::Green).add_modifier(Modifier::SLOW_BLINK),
        ),
    ]);
    let title = if let Some(err) = &s.chat_error {
        format!(" Input · {} ", truncate(err.clone(), 60))
    } else {
        " Input ".into()
    };
    f.render_widget(
        Paragraph::new(line).block(Block::default().borders(Borders::ALL).title(title)),
        area,
    );
}

fn render_footer(f: &mut ratatui::Frame, area: Rect, s: &SharedState) {
    let fmt_instant = |i: &Option<Instant>| match i {
        Some(t) => format!("{:.1}s", t.elapsed().as_secs_f32()),
        None => "n/a".into(),
    };
    let status_line = Line::from(vec![
        Span::styled("  /status: ", Style::default().add_modifier(Modifier::DIM)),
        Span::raw(fmt_instant(&s.last_status_refresh)),
        Span::styled("   ping: ", Style::default().add_modifier(Modifier::DIM)),
        Span::raw(fmt_instant(&s.last_ping_round)),
        Span::styled("   state: ", Style::default().add_modifier(Modifier::DIM)),
        Span::raw(if s.streaming { "streaming" } else { "idle" }),
    ]);
    let hints = Line::from(vec![
        Span::styled("  Enter", Style::default().fg(Color::Yellow)),
        Span::raw(" send    "),
        Span::styled("Ctrl-L", Style::default().fg(Color::Yellow)),
        Span::raw(" clear    "),
        Span::styled("Ctrl-R", Style::default().fg(Color::Yellow)),
        Span::raw(" ping now    "),
        Span::styled("Ctrl-C / Esc", Style::default().fg(Color::Yellow)),
        Span::raw(" quit"),
    ]);
    f.render_widget(
        Paragraph::new(vec![status_line, hints])
            .block(Block::default().borders(Borders::TOP)),
        area,
    );
}

fn kv(k: &str, v: &str) -> Line<'static> {
    Line::from(vec![
        Span::styled(
            format!("  {:<13}", k),
            Style::default()
                .fg(Color::Gray)
                .add_modifier(Modifier::DIM),
        ),
        Span::raw(v.to_string()),
    ])
}

fn short_id(id: &str) -> String {
    if id.len() > 20 {
        format!("{}…{}", &id[..12], &id[id.len() - 4..])
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
