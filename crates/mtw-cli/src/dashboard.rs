//! Terminal dashboard showing a node's mesh state.
//!
//! Three panels:
//! - **Self** — what `GET /status` on the local proxy returns
//! - **Peers** — each entry in `~/.mtw/peers.json` with live iroh ping status
//! - **Status line** — last-refresh timestamps and shortcuts
//!
//! Background task pings peers every `ping_every` seconds; the UI thread
//! redraws on every tick or input event. Quit with `q`, force-refresh
//! peers with `r`.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::Context;
use crossterm::{
    event::{DisableMouseCapture, EnableMouseCapture, Event, EventStream, KeyCode, KeyEventKind},
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
use tokio::sync::Mutex;

pub struct DashboardArgs {
    pub url: String,
    pub tick: Duration,
    pub ping_every: Duration,
}

#[derive(Debug, Default, Clone)]
struct PeerHealth {
    last_attempt: Option<Instant>,
    last_rtt_ms: Option<u128>,
    last_model: Option<String>,
    last_error: Option<String>,
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
}

pub async fn run(args: DashboardArgs) -> anyhow::Result<()> {
    let state = Arc::new(Mutex::new(SharedState::default()));

    // Bind a client iroh endpoint once, reuse for all peer pings.
    let secret = mtw_core::identity::load_or_create()?;
    let endpoint = Endpoint::builder(presets::N0)
        .secret_key(secret)
        .bind()
        .await
        .context("bind iroh endpoint for dashboard pings")?;

    // Spawn background refresh task.
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

    // Set up terminal.
    enable_raw_mode().context("enable raw mode")?;
    let mut stdout = std::io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let result = ui_loop(&mut terminal, state, args.tick).await;

    // Restore terminal.
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
    tick: Duration,
) -> anyhow::Result<()> {
    let mut events = EventStream::new();
    let mut ticker = tokio::time::interval(tick);

    loop {
        // Read a snapshot under the lock, then release before rendering.
        let snapshot = state.lock().await.clone_snapshot();
        terminal.draw(|f| render(f, &snapshot))?;

        tokio::select! {
            _ = ticker.tick() => {}
            maybe = events.next() => match maybe {
                Some(Ok(Event::Key(key))) if key.kind == KeyEventKind::Press => {
                    match key.code {
                        KeyCode::Char('q') | KeyCode::Esc => return Ok(()),
                        KeyCode::Char('r') => {
                            // Force an immediate refresh by clearing last_ping_round.
                            let mut s = state.lock().await;
                            s.last_status_refresh = None;
                            s.last_ping_round = None;
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

impl SharedState {
    fn clone_snapshot(&self) -> SharedState {
        Self {
            node: self.node.clone(),
            node_error: self.node_error.clone(),
            peers: self.peers.clone(),
            peers_error: self.peers_error.clone(),
            peer_health: self.peer_health.clone(),
            last_status_refresh: self.last_status_refresh,
            last_ping_round: self.last_ping_round,
        }
    }
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

        // Refresh /status.
        match client.get(format!("{url}/status")).send().await {
            Ok(resp) if resp.status().is_success() => match resp.json::<NodeStatus>().await {
                Ok(s) => {
                    let mut st = state.lock().await;
                    st.node = Some(s);
                    st.node_error = None;
                    st.last_status_refresh = Some(Instant::now());
                }
                Err(err) => {
                    let mut st = state.lock().await;
                    st.node_error = Some(format!("parse /status: {err}"));
                }
            },
            Ok(resp) => {
                let mut st = state.lock().await;
                st.node_error = Some(format!("/status http {}", resp.status()));
            }
            Err(err) => {
                let mut st = state.lock().await;
                st.node_error = Some(format!("/status unreachable: {err}"));
            }
        }

        // Refresh peers.json.
        match mtw_core::peers::load() {
            Ok(PeerList { peers }) => {
                let mut st = state.lock().await;
                st.peers = peers;
                st.peers_error = None;
            }
            Err(err) => {
                let mut st = state.lock().await;
                st.peers_error = Some(format!("{err}"));
            }
        }

        // Ping peers at the configured cadence.
        let should_ping = {
            let st = state.lock().await;
            match st.last_ping_round {
                None => true,
                Some(t) => t.elapsed() >= ping_every,
            }
        };
        if should_ping {
            let peers = state.lock().await.peers.clone();
            for peer in peers {
                let id: EndpointId = match peer.id.parse() {
                    Ok(id) => id,
                    Err(err) => {
                        let mut st = state.lock().await;
                        st.peer_health.insert(
                            peer.id.clone(),
                            PeerHealth {
                                last_error: Some(format!("bad id: {err}")),
                                ..PeerHealth::default()
                            },
                        );
                        continue;
                    }
                };
                let start = Instant::now();
                let result = ping_peer(&endpoint, id, Duration::from_secs(3)).await;
                let mut st = state.lock().await;
                let slot = st.peer_health.entry(peer.id.clone()).or_default();
                slot.last_attempt = Some(start);
                match result {
                    Ok((Pong { model_info, .. }, rtt)) => {
                        slot.last_rtt_ms = Some(rtt.as_millis());
                        slot.last_model = model_info.map(|m| m.name);
                        slot.last_error = None;
                    }
                    Err(err) => {
                        slot.last_rtt_ms = None;
                        slot.last_error = Some(format!("{err}"));
                    }
                }
            }
            state.lock().await.last_ping_round = Some(Instant::now());
        }
    }
}

// ---------------------------------------------------------------- rendering

fn render(f: &mut ratatui::Frame, state: &SharedState) {
    let area = f.area();
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),                                  // title bar
            Constraint::Length(10),                                 // self panel
            Constraint::Min(6),                                     // peers panel (flex)
            Constraint::Length(4),                                  // status line
            Constraint::Length(1),                                  // shortcuts
        ])
        .split(area);

    render_title(f, chunks[0]);
    render_self(f, chunks[1], state);
    render_peers(f, chunks[2], state);
    render_status_line(f, chunks[3], state);
    render_shortcuts(f, chunks[4]);
}

fn render_title(f: &mut ratatui::Frame, area: Rect) {
    let now = humantime::format_rfc3339_seconds(SystemTime::now()).to_string();
    let title = Line::from(vec![
        Span::styled("meshthatworks ", Style::default().add_modifier(Modifier::BOLD).fg(Color::Cyan)),
        Span::raw("· dashboard "),
        Span::styled(now, Style::default().add_modifier(Modifier::DIM)),
    ]);
    let block = Block::default().borders(Borders::BOTTOM);
    let p = Paragraph::new(title).block(block);
    f.render_widget(p, area);
}

fn render_self(f: &mut ratatui::Frame, area: Rect, state: &SharedState) {
    let lines: Vec<Line> = if let Some(n) = &state.node {
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
    } else if let Some(err) = &state.node_error {
        vec![
            Line::from(Span::styled(
                "cannot reach /status",
                Style::default().fg(Color::Red),
            )),
            Line::from(Span::styled(err.clone(), Style::default().add_modifier(Modifier::DIM))),
            Line::from(""),
            Line::from(Span::styled(
                "run `mtw serve` in another terminal, or use --url",
                Style::default().add_modifier(Modifier::DIM),
            )),
        ]
    } else {
        vec![Line::from(Span::styled(
            "polling /status …",
            Style::default().add_modifier(Modifier::DIM),
        ))]
    };

    let block = Block::default().borders(Borders::ALL).title(" Self ");
    let p = Paragraph::new(lines).block(block).wrap(Wrap { trim: false });
    f.render_widget(p, area);
}

fn render_peers(f: &mut ratatui::Frame, area: Rect, state: &SharedState) {
    let title = format!(" Peers ({}) ", state.peers.len());
    let block = Block::default().borders(Borders::ALL).title(title);

    let header = Line::from(vec![
        Span::styled(
            format!("  {:<20}", "peer id (short)"),
            Style::default().add_modifier(Modifier::BOLD),
        ),
        Span::styled(format!("{:<8}", "status"), Style::default().add_modifier(Modifier::BOLD)),
        Span::styled(format!("{:>8}", "rtt"), Style::default().add_modifier(Modifier::BOLD)),
        Span::raw("  "),
        Span::styled("model / note", Style::default().add_modifier(Modifier::BOLD)),
    ]);

    let mut lines = vec![header, Line::from("")];

    if state.peers.is_empty() {
        lines.push(Line::from(Span::styled(
            "  no peers paired yet. run `mtw pair` / `mtw join <invite>`.",
            Style::default().add_modifier(Modifier::DIM),
        )));
    } else {
        for peer in &state.peers {
            let short = short_id(&peer.id);
            let health = state.peer_health.get(&peer.id);
            let (status_cell, rtt_cell, note_cell) = match health {
                Some(h) if h.last_rtt_ms.is_some() => (
                    Span::styled("● UP   ", Style::default().fg(Color::Green)),
                    Span::raw(format!("{:>5}ms", h.last_rtt_ms.unwrap_or(0))),
                    Span::raw(
                        h.last_model.clone().unwrap_or_else(|| "(no model)".into()),
                    ),
                ),
                Some(h) => (
                    Span::styled("○ DOWN ", Style::default().fg(Color::Red)),
                    Span::styled("     - ", Style::default().add_modifier(Modifier::DIM)),
                    Span::styled(
                        truncate(h.last_error.clone().unwrap_or_default(), 60),
                        Style::default().add_modifier(Modifier::DIM),
                    ),
                ),
                None => (
                    Span::styled("… ping", Style::default().fg(Color::Yellow)),
                    Span::styled("     - ", Style::default().add_modifier(Modifier::DIM)),
                    Span::styled("waiting for first round", Style::default().add_modifier(Modifier::DIM)),
                ),
            };

            lines.push(Line::from(vec![
                Span::raw(format!("  {:<20}", short)),
                status_cell,
                Span::raw(" "),
                rtt_cell,
                Span::raw("  "),
                note_cell,
            ]));
        }
    }

    if let Some(err) = &state.peers_error {
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            format!("  peers.json error: {err}"),
            Style::default().fg(Color::Red),
        )));
    }

    let p = Paragraph::new(lines).block(block).wrap(Wrap { trim: false });
    f.render_widget(p, area);
}

fn render_status_line(f: &mut ratatui::Frame, area: Rect, state: &SharedState) {
    let fmt_instant = |i: &Option<Instant>| match i {
        Some(t) => format!("{:.1}s ago", t.elapsed().as_secs_f32()),
        None => "never".into(),
    };

    let line1 = Line::from(vec![
        Span::styled("last /status: ", Style::default().add_modifier(Modifier::DIM)),
        Span::raw(fmt_instant(&state.last_status_refresh)),
        Span::styled("   last ping round: ", Style::default().add_modifier(Modifier::DIM)),
        Span::raw(fmt_instant(&state.last_ping_round)),
    ]);

    let block = Block::default().borders(Borders::ALL).title(" Status ");
    let p = Paragraph::new(vec![line1]).block(block);
    f.render_widget(p, area);
}

fn render_shortcuts(f: &mut ratatui::Frame, area: Rect) {
    let line = Line::from(vec![
        Span::styled("  q ", Style::default().fg(Color::Yellow)),
        Span::raw("quit"),
        Span::styled("   r ", Style::default().fg(Color::Yellow)),
        Span::raw("force refresh"),
    ]);
    f.render_widget(Paragraph::new(line), area);
}

fn kv(k: &str, v: &str) -> Line<'static> {
    Line::from(vec![
        Span::styled(
            format!("  {:<13}", k),
            Style::default().fg(Color::Gray).dim(),
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

// Suppress the "unused" warning on Modifier — kept in scope because adding
// bold/italic to lines is a very near-term polish step.
#[allow(dead_code)]
fn _use_modifier() -> Modifier {
    Modifier::BOLD
}
