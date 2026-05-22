# MeshThatWorks — macOS app

A native macOS app for [MeshThatWorks](https://github.com/mrunalpendem123/meshthatworks):
run frontier open-source MoE models across the Apple Silicon devices you already
own. Deep-space UI inspired by node dashboards like LayerEdge — a dotted globe,
radar rings, glass pills — wrapped around the existing `mtw` engine.

Built with **Tauri 2 + React + Vite + Tailwind**. The Rust `mtw` CLI is embedded
as a sidecar, so the app is self-contained: it spawns `mtw serve`, supervises the
SwiftLM engine, and talks to the OpenAI-compatible proxy on `localhost:9337`.

## What's inside

```
desktop/
├── src/                      React frontend
│   ├── screens/              Welcome · Onboarding · Dashboard · Chat · Mesh · Models
│   ├── components/           Globe (cobe) · Brand · TopBar · ui atoms
│   └── lib/                  Tauri bindings · store · catalog · types
└── src-tauri/                Rust side
    ├── src/engine.rs         supervises the `mtw serve` sidecar + health poller
    ├── src/commands.rs       setup detect · model download · pair/join · chat stream
    ├── binaries/             the bundled `mtw` sidecar (per target triple)
    └── tauri.conf.json
```

**Design decision:** every network call (to `localhost:9337` and to
huggingface.co) happens in Rust via `reqwest` and is streamed to the webview over
Tauri IPC channels/events. This sidesteps macOS App Transport Security blocking
cleartext HTTP to localhost from `WKWebView`.

## Develop

```bash
cd desktop
pnpm install
./scripts/sync-sidecar.sh   # builds mtw and copies it into src-tauri/binaries/
pnpm tauri dev
```

`sync-sidecar.sh` is required on a fresh checkout — the sidecar binary is
git-ignored. The first run still needs SwiftLM + a model; the app walks you
through both on the Onboarding screen (the engine itself takes ~30 min to build
once, and needs Xcode + the Metal Toolchain).

## Build a distributable

```bash
pnpm tauri build
```

Outputs to `src-tauri/target/release/bundle/`:
- `macos/MeshThatWorks.app`
- `dmg/MeshThatWorks_0.1.0_aarch64.dmg`

### Signing & notarization (your Apple ID)

`tauri build` ad-hoc-signs by default — fine for running on your own Mac
(Gatekeeper: right-click → Open the first time). To distribute to other people
you need a **Developer ID Application** certificate, which requires the paid
Apple Developer Program (an Apple ID alone is not enough).

Once enrolled, set these and rebuild — Tauri signs the app and the bundled
sidecar, then notarizes:

```bash
export APPLE_SIGNING_IDENTITY="Developer ID Application: Your Name (TEAMID)"
export APPLE_ID="you@example.com"
export APPLE_PASSWORD="app-specific-password"   # appleid.apple.com → App-Specific Passwords
export APPLE_TEAM_ID="TEAMID"
pnpm tauri build
```

See https://v2.tauri.app/distribute/sign/macos/ for the full flow.
