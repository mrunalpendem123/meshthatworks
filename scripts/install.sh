#!/usr/bin/env bash
# Build and install the `mtw` CLI to ~/.local/bin (or $MTW_INSTALL_DIR).
#
# Usage:
#   ./scripts/install.sh
#   MTW_INSTALL_DIR=/usr/local/bin ./scripts/install.sh

set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${MTW_INSTALL_DIR:-$HOME/.local/bin}"

cd "$REPO"

echo "==> Building mtw (release)…"
cargo build --release --bin mtw

mkdir -p "$DEST"
install -m 0755 "$REPO/target/release/mtw" "$DEST/mtw"

echo "==> Installed: $DEST/mtw"
"$DEST/mtw" --version

case ":$PATH:" in
    *":$DEST:"*) ;;
    *)
        echo
        echo "note: $DEST is not on your \$PATH yet. Add it with:"
        echo "    echo 'export PATH=\"$DEST:\$PATH\"' >> ~/.zshrc"
        ;;
esac
