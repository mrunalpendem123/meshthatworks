#!/usr/bin/env bash
# Build the `mtw` CLI from the workspace and drop it where Tauri expects the
# sidecar binary (named with the Rust host target triple). Run this before
# `pnpm tauri dev` / `pnpm tauri build` on a fresh checkout.
set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"        # desktop/
ROOT="$(cd "$HERE/.." && pwd)"                   # repo root
TRIPLE="$(rustc -vV | sed -n 's/host: //p')"
DEST="$HERE/src-tauri/binaries/mtw-$TRIPLE"

echo "==> Building mtw (release)…"
cargo build --release --manifest-path "$ROOT/Cargo.toml" -p mtw-cli --bin mtw

mkdir -p "$HERE/src-tauri/binaries"
cp "$ROOT/target/release/mtw" "$DEST"
chmod +x "$DEST"
echo "==> Sidecar ready: $DEST"
