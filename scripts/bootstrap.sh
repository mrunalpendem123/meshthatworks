#!/usr/bin/env bash
# One-line installer for MeshThatWorks.
#
#   curl -sSL https://raw.githubusercontent.com/mrunalpendem123/meshthatworks/master/scripts/bootstrap.sh | sh

set -euo pipefail

REPO_URL="${MTW_REPO:-https://github.com/mrunalpendem123/meshthatworks.git}"
SRC_DIR="${MTW_HOME:-$HOME/.meshthatworks}"
BIN_DIR="${MTW_INSTALL_DIR:-$HOME/.local/bin}"

step() { printf "\033[1;36m==>\033[0m %s\n" "$1"; }
warn() { printf "\033[1;33m==>\033[0m %s\n" "$1" >&2; }
fail() { printf "\033[1;31m==>\033[0m %s\n" "$1" >&2; exit 1; }

command -v git >/dev/null   || fail "git is required. Install Xcode Command Line Tools: xcode-select --install"
command -v cargo >/dev/null || fail "Rust is required. Install: curl https://sh.rustup.rs -sSf | sh"

if [ -d "$SRC_DIR/.git" ]; then
    step "Updating $SRC_DIR"
    git -C "$SRC_DIR" pull --ff-only
else
    step "Cloning to $SRC_DIR"
    git clone --depth 1 "$REPO_URL" "$SRC_DIR"
fi

step "Building (first time takes a few minutes)"
cd "$SRC_DIR"
cargo build --release --bin mtw

mkdir -p "$BIN_DIR"
install -m 0755 "$SRC_DIR/target/release/mtw" "$BIN_DIR/mtw"
step "Installed: $BIN_DIR/mtw"

case ":$PATH:" in
    *":$BIN_DIR:"*)
        ;;
    *)
        warn "Add $BIN_DIR to your PATH:"
        echo "    echo 'export PATH=\"$BIN_DIR:\$PATH\"' >> ~/.zshrc"
        echo "    source ~/.zshrc"
        ;;
esac

echo
step "Next: run  mtw doctor"
