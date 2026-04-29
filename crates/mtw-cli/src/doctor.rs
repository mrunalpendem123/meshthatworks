//! `mtw doctor` — connectivity self-check.
//!
//! Reports the four signals that determine whether two peers will pair
//! directly or fall back to relayed traffic:
//!
//! 1. **IPv6 reachability.** If both peers have v6, NAT is moot; iroh uses
//!    the v6 path and direct connection essentially always works.
//! 2. **Public IPv4 + CGNAT detection.** If your public v4 lives in
//!    `100.64.0.0/10` you're carrier-grade-NAT'd and hole-punching usually
//!    fails — relay fallback is unavoidable without a v6 path or VPN.
//! 3. **NAT type via STUN.** Sends a binding request to two public STUN
//!    servers; if the mapped port is the same, the NAT is endpoint-
//!    independent (hole-punching works); if different, it's symmetric NAT
//!    (hole-punching fails).
//! 4. **macOS application firewall state.** Reports if it's blocking
//!    inbound iroh traffic.
//!
//! Output is one screen of plain text. Both peers run `mtw doctor` and can
//! eyeball each other's output to predict what their pairing will look
//! like before doing it.

use std::net::{IpAddr, Ipv4Addr, SocketAddr, ToSocketAddrs};
use std::time::Duration;

use anyhow::Context;
use tokio::io::AsyncWriteExt;
use tokio::net::{TcpStream, UdpSocket};
use tokio::time::timeout;

const PROBE_TIMEOUT: Duration = Duration::from_secs(3);
const HTTP_TIMEOUT: Duration = Duration::from_secs(5);

// Known IPv6 endpoints that *should* always be reachable if v6 is working.
// Cloudflare's public DNS is the most reliable single target.
const IPV6_PROBE: &str = "[2606:4700:4700::1111]:443";

// Two public STUN servers run by Google. Different IPs; if the mapped port
// we get back from each differs, our NAT is endpoint-dependent (= symmetric).
const STUN_A: &str = "stun.l.google.com:19302";
const STUN_B: &str = "stun1.l.google.com:19302";

#[derive(Debug, Clone)]
pub struct Report {
    pub ipv6: Option<String>, // address string if reachable, None if not
    pub ipv6_error: Option<String>,
    pub ipv4_public: Option<Ipv4Addr>,
    pub ipv4_error: Option<String>,
    pub cgnat: Option<bool>,
    pub nat_type: Option<NatVerdict>,
    pub nat_error: Option<String>,
    pub firewall: FirewallStatus,
}

#[derive(Debug, Clone)]
pub enum NatVerdict {
    /// Mapped port stayed the same across two STUN servers — full-cone or
    /// restricted-cone NAT, hole-punching works ~99%.
    EndpointIndependent { mapped: SocketAddr },
    /// Mapped port differed — symmetric NAT, hole-punching ~always fails.
    Symmetric { a: SocketAddr, b: SocketAddr },
}

#[derive(Debug, Clone)]
pub enum FirewallStatus {
    Off,
    OnAllowSigned,
    OnBlockAll,
    Unknown(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Connectivity {
    /// Direct path open: IPv6 reachable, OR friendly v4 NAT.
    Direct,
    /// Will need relay fallback: CGNAT or symmetric NAT, no v6.
    Relay,
    /// Couldn't even reach the public internet — both probes failed.
    NoInternet,
}

impl Report {
    /// Coarse classification — what the dashboard footer should show.
    pub fn connectivity(&self) -> Connectivity {
        if self.ipv6.is_some() {
            return Connectivity::Direct;
        }
        if self.ipv4_public.is_none() && self.nat_type.is_none() {
            return Connectivity::NoInternet;
        }
        if matches!(self.cgnat, Some(true))
            || matches!(self.nat_type, Some(NatVerdict::Symmetric { .. }))
        {
            return Connectivity::Relay;
        }
        if matches!(self.nat_type, Some(NatVerdict::EndpointIndependent { .. })) {
            return Connectivity::Direct;
        }
        Connectivity::Relay
    }

    pub fn short_summary(&self) -> String {
        match self.connectivity() {
            Connectivity::Direct => {
                if self.ipv6.is_some() {
                    "direct (IPv6)".into()
                } else {
                    "direct (v4 punch)".into()
                }
            }
            Connectivity::Relay => "relay (NAT bound)".into(),
            Connectivity::NoInternet => "offline".into(),
        }
    }
}

pub async fn run() -> anyhow::Result<()> {
    println!("mtw doctor — environment check");
    println!();

    let local = check_local_setup();
    print_local(&local);

    println!();
    println!("running network probes (~5–10s)…");
    println!();
    let report = collect().await;
    render(&report);

    println!();
    print_next_step(&local);
    Ok(())
}

#[derive(Debug, Clone)]
pub struct LocalSetup {
    pub mtw_on_path: bool,
    pub xcode_metal: bool,
    pub xcode_metal_err: Option<String>,
    pub swiftlm_built: bool,
    pub swiftlm_path: std::path::PathBuf,
    pub model_present: bool,
    pub model_path: std::path::PathBuf,
}

pub fn check_local_setup() -> LocalSetup {
    let swiftlm_path = crate::default_swiftlm_binary();
    let model_path = crate::default_model_dir();

    // mtw is the binary that's currently running, so it's on PATH iff `which mtw`
    // resolves to anything. We don't strictly need this to be on PATH for the
    // user — but it's handy to flag.
    let mtw_on_path = std::process::Command::new("sh")
        .arg("-c")
        .arg("command -v mtw")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);

    let metal_check = std::process::Command::new("xcrun")
        .args(["-sdk", "macosx", "metal", "--version"])
        .output();
    let (xcode_metal, xcode_metal_err) = match metal_check {
        Ok(o) if o.status.success() => (true, None),
        Ok(o) => (
            false,
            Some(String::from_utf8_lossy(&o.stderr).trim().to_string()),
        ),
        Err(e) => (false, Some(format!("{e}"))),
    };

    let swiftlm_built = swiftlm_path.is_file()
        && std::fs::metadata(&swiftlm_path)
            .map(|m| m.permissions().readonly() == false || true) // existence is enough
            .unwrap_or(false);

    let model_present = model_path.join("config.json").is_file()
        && model_path.join("model.safetensors").is_file();

    LocalSetup {
        mtw_on_path,
        xcode_metal,
        xcode_metal_err,
        swiftlm_built,
        swiftlm_path,
        model_present,
        model_path,
    }
}

fn print_local(s: &LocalSetup) {
    fn pad(label: &str) -> String {
        format!("  {label:<16}")
    }
    println!("local setup");
    print!("{}", pad("mtw on PATH"));
    if s.mtw_on_path {
        println!("✓");
    } else {
        println!("✗  add ~/.local/bin to PATH:  echo 'export PATH=\"$HOME/.local/bin:$PATH\"' >> ~/.zshrc");
    }
    print!("{}", pad("Xcode + Metal"));
    if s.xcode_metal {
        println!("✓");
    } else {
        let hint = match &s.xcode_metal_err {
            Some(e) if e.contains("xcode-select") => {
                "install Xcode from the App Store, then: xcodebuild -downloadComponent MetalToolchain"
            }
            _ => "install Xcode + run: xcodebuild -downloadComponent MetalToolchain",
        };
        println!("✗  {hint}");
    }
    print!("{}", pad("SwiftLM binary"));
    if s.swiftlm_built {
        println!("✓  {}", s.swiftlm_path.display());
    } else {
        println!(
            "✗  not at {}. Run scripts/bootstrap.sh, or build manually: \
             git clone https://github.com/SharpAI/SwiftLM ~/.meshthatworks-deps/SwiftLM \
             && cd ~/.meshthatworks-deps/SwiftLM && swift build -c release",
            s.swiftlm_path.display()
        );
    }
    print!("{}", pad("Model"));
    if s.model_present {
        println!("✓  {}", s.model_path.display());
    } else {
        // Look for any other model dir under ~/.meshthatworks-deps/models — the
        // dashboard catalog might have downloaded one with a different name.
        let other = list_other_models(&s.model_path);
        if !other.is_empty() {
            println!("✓  found other models — pass --model <path>");
            for p in &other {
                println!("                     {}", p.display());
            }
        } else {
            println!("✗  no model installed yet — open `mtw dashboard` and pick one from the Models tab");
        }
    }
}

fn list_other_models(default_path: &std::path::Path) -> Vec<std::path::PathBuf> {
    let parent = match default_path.parent() {
        Some(p) => p.to_path_buf(),
        None => return Vec::new(),
    };
    let Ok(rd) = std::fs::read_dir(&parent) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for entry in rd.flatten() {
        let path = entry.path();
        if path.is_dir() && path.join("config.json").is_file() {
            out.push(path);
        }
    }
    out
}

fn print_next_step(s: &LocalSetup) {
    println!("──────────────────────────────────────────────────────────");
    if !s.xcode_metal {
        println!("Next: install Xcode (App Store) + the Metal toolchain.");
        println!("       xcodebuild -downloadComponent MetalToolchain");
        return;
    }
    if !s.swiftlm_built {
        println!("Next: run the bootstrap to clone & build SwiftLM (~30 min).");
        println!("       curl -sSL https://raw.githubusercontent.com/mrunalpendem123/meshthatworks/master/scripts/bootstrap.sh | sh");
        return;
    }
    if !s.model_present && list_other_models(&s.model_path).is_empty() {
        println!("Next: pick a model.");
        println!("       mtw dashboard          # opens the Models tab — choose one and download");
        return;
    }
    println!("Everything is set up. ✨");
    println!();
    println!("Next: run  mtw start");
    println!("       (this spawns the engine and opens the live dashboard)");
}

pub async fn collect() -> Report {
    let (ipv6_result, ipv4_result, nat_result) = tokio::join!(
        probe_ipv6(),
        probe_ipv4_public(),
        probe_nat_type(),
    );

    let firewall = check_macos_firewall();

    let (ipv6, ipv6_error) = match ipv6_result {
        Ok(addr) => (Some(addr), None),
        Err(e) => (None, Some(format!("{e:#}"))),
    };
    let (ipv4_public, ipv4_error) = match ipv4_result {
        Ok(addr) => (Some(addr), None),
        Err(e) => (None, Some(format!("{e:#}"))),
    };
    let cgnat = ipv4_public.map(is_cgnat);
    let (nat_type, nat_error) = match nat_result {
        Ok(v) => (Some(v), None),
        Err(e) => (None, Some(format!("{e:#}"))),
    };

    Report {
        ipv6,
        ipv6_error,
        ipv4_public,
        ipv4_error,
        cgnat,
        nat_type,
        nat_error,
        firewall,
    }
}

/// Try to TCP-connect to a known public IPv6 endpoint. Success means the OS
/// has a working v6 route to the internet.
async fn probe_ipv6() -> anyhow::Result<String> {
    let addr: SocketAddr = IPV6_PROBE.parse().context("parse v6 probe target")?;
    let stream = timeout(PROBE_TIMEOUT, TcpStream::connect(addr))
        .await
        .context("v6 connect timed out")?
        .context("v6 connect failed")?;
    // We just want to know it's reachable; close immediately.
    let local = stream.local_addr().ok();
    drop(stream);
    Ok(local
        .map(|s| s.ip().to_string())
        .unwrap_or_else(|| "(unknown)".into()))
}

/// Hit a public reflector (`ifconfig.co`) over IPv4 to learn our externally-
/// visible IPv4. We use `reqwest` (already a workspace dep) for the HTTP
/// request, forced to v4 by passing through `to_socket_addrs` filtering.
async fn probe_ipv4_public() -> anyhow::Result<Ipv4Addr> {
    let client = reqwest::Client::builder()
        .timeout(HTTP_TIMEOUT)
        .local_address(Some(IpAddr::V4(Ipv4Addr::UNSPECIFIED)))
        .build()
        .context("build http client")?;
    let body = client
        .get("https://ifconfig.co/ip")
        .header("user-agent", "curl/mtw-doctor")
        .send()
        .await
        .context("ifconfig.co request")?
        .error_for_status()
        .context("ifconfig.co status")?
        .text()
        .await
        .context("ifconfig.co body")?;
    body.trim()
        .parse::<Ipv4Addr>()
        .with_context(|| format!("parse public ipv4 from {:?}", body.trim()))
}

/// Send STUN binding requests to two different STUN servers and compare the
/// mapped (public_ip, public_port) values. Same → endpoint-independent
/// mapping (good NAT). Different → symmetric NAT (bad NAT).
///
/// We use a single local UDP socket for both probes so the same source
/// `(ip, port)` is presented to both servers.
async fn probe_nat_type() -> anyhow::Result<NatVerdict> {
    let sock = UdpSocket::bind("0.0.0.0:0")
        .await
        .context("bind local udp")?;

    let a = stun_binding_request(&sock, STUN_A).await.context("stun A")?;
    let b = stun_binding_request(&sock, STUN_B).await.context("stun B")?;

    if a == b {
        Ok(NatVerdict::EndpointIndependent { mapped: a })
    } else {
        Ok(NatVerdict::Symmetric { a, b })
    }
}

/// Minimal STUN binding request — sufficient for "what's my mapped address".
/// We don't bother with attribute parsing for anything beyond `XOR-MAPPED-ADDRESS`.
async fn stun_binding_request(sock: &UdpSocket, target: &str) -> anyhow::Result<SocketAddr> {
    // Resolve target to a v4 SocketAddr — STUN servers we picked are v4-only.
    let target_addr = target
        .to_socket_addrs()
        .with_context(|| format!("resolve {target}"))?
        .find(|a| a.is_ipv4())
        .ok_or_else(|| anyhow::anyhow!("no v4 address for {target}"))?;

    // RFC 5389 STUN binding request:
    //   header (20 bytes):
    //     0x0001 (binding request)
    //     0x0000 (length of message attributes — 0)
    //     0x2112A442 (magic cookie)
    //     12 bytes transaction id (random)
    let mut req = [0u8; 20];
    req[0..2].copy_from_slice(&0x0001u16.to_be_bytes());
    req[2..4].copy_from_slice(&0u16.to_be_bytes());
    req[4..8].copy_from_slice(&0x2112A442u32.to_be_bytes());
    let tid: [u8; 12] = rand_bytes();
    req[8..20].copy_from_slice(&tid);

    timeout(PROBE_TIMEOUT, sock.send_to(&req, target_addr))
        .await
        .context("stun send timed out")?
        .context("stun send")?;

    let mut buf = [0u8; 512];
    let (n, _from) = timeout(PROBE_TIMEOUT, sock.recv_from(&mut buf))
        .await
        .context("stun recv timed out")?
        .context("stun recv")?;
    parse_xor_mapped(&buf[..n])
}

fn rand_bytes() -> [u8; 12] {
    let mut out = [0u8; 12];
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    for (i, b) in now.to_le_bytes().iter().enumerate() {
        out[i] = *b;
    }
    // Last 4 bytes from a stable hash of the local time as well — good enough.
    let h = (now.wrapping_mul(2654435761)) as u32;
    out[8..12].copy_from_slice(&h.to_le_bytes());
    out
}

fn parse_xor_mapped(buf: &[u8]) -> anyhow::Result<SocketAddr> {
    if buf.len() < 20 {
        anyhow::bail!("stun reply too short");
    }
    let msg_type = u16::from_be_bytes([buf[0], buf[1]]);
    if msg_type != 0x0101 {
        anyhow::bail!("not a STUN success response (type={msg_type:#x})");
    }
    let attr_len = u16::from_be_bytes([buf[2], buf[3]]) as usize;
    let body = &buf[20..];
    if body.len() < attr_len {
        anyhow::bail!("stun body shorter than declared length");
    }
    // Walk attributes. Looking for XOR-MAPPED-ADDRESS (0x0020) — the modern
    // form that's NAT-rewrite-safe. Fall back to MAPPED-ADDRESS (0x0001).
    let mut cursor = 0usize;
    let mut xor_mapped: Option<SocketAddr> = None;
    let mut mapped: Option<SocketAddr> = None;
    while cursor + 4 <= attr_len {
        let typ = u16::from_be_bytes([body[cursor], body[cursor + 1]]);
        let len = u16::from_be_bytes([body[cursor + 2], body[cursor + 3]]) as usize;
        let val_start = cursor + 4;
        let val_end = val_start + len;
        if val_end > body.len() {
            break;
        }
        let val = &body[val_start..val_end];
        match typ {
            0x0020 => {
                if let Some(a) = parse_xor_mapped_attr(val) {
                    xor_mapped = Some(a);
                }
            }
            0x0001 => {
                if let Some(a) = parse_plain_mapped_attr(val) {
                    mapped = Some(a);
                }
            }
            _ => {}
        }
        // Attributes are 4-byte aligned.
        let pad = (4 - (len % 4)) % 4;
        cursor = val_end + pad;
    }
    xor_mapped
        .or(mapped)
        .ok_or_else(|| anyhow::anyhow!("no MAPPED-ADDRESS in stun reply"))
}

fn parse_xor_mapped_attr(val: &[u8]) -> Option<SocketAddr> {
    if val.len() < 4 {
        return None;
    }
    let family = val[1];
    let xport = u16::from_be_bytes([val[2], val[3]]);
    let port = xport ^ 0x2112; // top 16 bits of magic cookie
    match family {
        0x01 => {
            // IPv4
            if val.len() < 8 {
                return None;
            }
            let mut octets = [0u8; 4];
            for i in 0..4 {
                octets[i] = val[4 + i] ^ [0x21, 0x12, 0xA4, 0x42][i];
            }
            Some(SocketAddr::new(IpAddr::V4(Ipv4Addr::from(octets)), port))
        }
        _ => None, // we don't need v6 mapped here
    }
}

fn parse_plain_mapped_attr(val: &[u8]) -> Option<SocketAddr> {
    if val.len() < 8 {
        return None;
    }
    let family = val[1];
    let port = u16::from_be_bytes([val[2], val[3]]);
    if family != 0x01 {
        return None;
    }
    let octets = [val[4], val[5], val[6], val[7]];
    Some(SocketAddr::new(IpAddr::V4(Ipv4Addr::from(octets)), port))
}

/// CGNAT range per RFC 6598: 100.64.0.0/10 (= 100.64.0.0 — 100.127.255.255).
fn is_cgnat(addr: Ipv4Addr) -> bool {
    let octets = addr.octets();
    octets[0] == 100 && (octets[1] & 0xC0) == 0x40
}

/// Read macOS Application Firewall state via `defaults read`. Values:
/// 0 = off, 1 = on, allow signed apps, 2 = on, block all.
fn check_macos_firewall() -> FirewallStatus {
    let out = std::process::Command::new("defaults")
        .arg("read")
        .arg("/Library/Preferences/com.apple.alf")
        .arg("globalstate")
        .output();
    match out {
        Ok(o) if o.status.success() => match String::from_utf8_lossy(&o.stdout).trim() {
            "0" => FirewallStatus::Off,
            "1" => FirewallStatus::OnAllowSigned,
            "2" => FirewallStatus::OnBlockAll,
            other => FirewallStatus::Unknown(other.into()),
        },
        Ok(o) => FirewallStatus::Unknown(format!(
            "defaults exit={:?}",
            o.status.code()
        )),
        Err(e) => FirewallStatus::Unknown(format!("defaults: {e}")),
    }
}

fn render(r: &Report) {
    fn pad(label: &str) -> String {
        format!("  {label:<16}")
    }

    // IPv6
    print!("{}", pad("IPv6"));
    match (&r.ipv6, &r.ipv6_error) {
        (Some(addr), _) => println!("✓ reachable  (local: {addr})"),
        (None, Some(e)) => println!("✗ no v6 route  ({e})"),
        (None, None) => println!("? unknown"),
    }

    // IPv4
    print!("{}", pad("IPv4 (public)"));
    match (&r.ipv4_public, &r.ipv4_error) {
        (Some(ip), _) => {
            let cgnat = r.cgnat.unwrap_or(false);
            let note = if cgnat {
                "  ⚠ CGNAT — hole-punching will likely fail on v4"
            } else {
                ""
            };
            println!("✓ {ip}{note}");
        }
        (None, Some(e)) => println!("✗ couldn't determine  ({e})"),
        (None, None) => println!("? unknown"),
    }

    // NAT
    print!("{}", pad("NAT type"));
    match (&r.nat_type, &r.nat_error) {
        (Some(NatVerdict::EndpointIndependent { mapped }), _) => println!(
            "✓ endpoint-independent ({mapped})  hole-punching works"
        ),
        (Some(NatVerdict::Symmetric { a, b }), _) => println!(
            "✗ symmetric  (mapped {a} ≠ {b})  hole-punching FAILS — relay required"
        ),
        (None, Some(e)) => println!("? STUN probe failed  ({e})"),
        (None, None) => println!("? unknown"),
    }

    // Firewall
    print!("{}", pad("macOS firewall"));
    match &r.firewall {
        FirewallStatus::Off => println!("✓ off"),
        FirewallStatus::OnAllowSigned => {
            println!("⚠ on, allowing signed apps  (mtw must be allowed)")
        }
        FirewallStatus::OnBlockAll => {
            println!("✗ on, blocking all incoming  (turn off or whitelist mtw)")
        }
        FirewallStatus::Unknown(s) => println!("? unknown  ({s})"),
    }

    println!();
    print!("{}", pad("Verdict"));
    println!("{}", verdict(r));
    println!();
    println!("  Send your friend the same `mtw doctor` output to compare.");
    println!("  If both peers have IPv6, you'll connect direct over v6 — done.");
    println!("  If either is CGNAT or symmetric NAT on v4, expect relay fallback.");
}

fn verdict(r: &Report) -> &'static str {
    if r.ipv6.is_some() {
        "✓ IPv6 available — direct connection essentially always works"
    } else if matches!(r.cgnat, Some(true))
        || matches!(r.nat_type, Some(NatVerdict::Symmetric { .. }))
    {
        "✗ NAT will block hole-punching — connections will use the relay"
    } else if matches!(
        r.nat_type,
        Some(NatVerdict::EndpointIndependent { .. })
    ) {
        "✓ IPv4 NAT is friendly — hole-punching should succeed"
    } else {
        "? mixed signals — try pairing and see"
    }
}
