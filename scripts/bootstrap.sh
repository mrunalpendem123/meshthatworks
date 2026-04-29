#!/usr/bin/env bash
# MeshThatWorks — one-command setup.
#
#   curl -sSL https://raw.githubusercontent.com/mrunalpendem123/meshthatworks/master/scripts/bootstrap.sh | sh
#
# Installs:  rustup (if missing) + this repo + SwiftLM + a starter model + mtw
# Time:      first run is ~30 min (mostly the SwiftLM build) and ~5 GB of disk
# After:     run  mtw start
#
# Optional env:
#   MTW_HOME=/path           where to clone meshthatworks      (default: ~/.meshthatworks)
#   MTW_DEPS=/path           where SwiftLM + models live       (default: ~/.meshthatworks-deps)
#   MTW_INSTALL_DIR=/path    where the mtw binary goes         (default: ~/.local/bin)
#   MTW_MODEL=name           Hugging Face mlx-community model  (default: OLMoE-1B-7B-0125-Instruct-4bit)
#   MTW_SKIP_SWIFTLM=1       skip SwiftLM clone+build (advanced)
#   MTW_SKIP_MODEL=1         skip model download (advanced)

set -euo pipefail

REPO_URL="${MTW_REPO:-https://github.com/mrunalpendem123/meshthatworks.git}"
SRC_DIR="${MTW_HOME:-$HOME/.meshthatworks}"
DEPS_DIR="${MTW_DEPS:-$HOME/.meshthatworks-deps}"
BIN_DIR="${MTW_INSTALL_DIR:-$HOME/.local/bin}"
MODEL_NAME="${MTW_MODEL:-OLMoE-1B-7B-0125-Instruct-4bit}"

# ───────────────────────────────────────────────────────── helpers
step()  { printf "\033[1;36m==>\033[0m %s\n" "$1"; }
warn()  { printf "\033[1;33m==>\033[0m %s\n" "$1" >&2; }
fail()  { printf "\033[1;31m==>\033[0m %s\n" "$1" >&2; exit 1; }
have()  { command -v "$1" >/dev/null 2>&1; }

# ───────────────────────────────────────────────────────── 1. base tools
step "Checking base tools"

if ! have git; then
    warn "git missing → triggering Xcode Command Line Tools install"
    xcode-select --install || true
    fail "Re-run this script after the Command Line Tools install completes."
fi

if ! have cargo; then
    step "Rust missing → installing via rustup (non-interactive)"
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable --profile minimal
    # shellcheck disable=SC1091
    . "$HOME/.cargo/env"
fi

# Full Xcode + Metal Toolchain are required by SwiftLM. Detect, but don't try
# to install — that's an Apple-ID-gated process.
if ! xcrun -sdk macosx metal --version >/dev/null 2>&1; then
    warn "Xcode + Metal Toolchain not installed."
    warn "  1. Install Xcode from the App Store"
    warn "  2. Run:  xcodebuild -downloadComponent MetalToolchain"
    if [ "${MTW_SKIP_SWIFTLM:-0}" != "1" ]; then
        fail "SwiftLM cannot build without Metal. Install Xcode and re-run."
    fi
fi

# ───────────────────────────────────────────────────────── 2. repo
step "Cloning / updating $SRC_DIR"
if [ -d "$SRC_DIR/.git" ]; then
    git -C "$SRC_DIR" pull --ff-only
else
    git clone --depth 1 "$REPO_URL" "$SRC_DIR"
fi

# ───────────────────────────────────────────────────────── 3. mtw
step "Building mtw (first time ~5 min)"
cd "$SRC_DIR"
cargo build --release --bin mtw

mkdir -p "$BIN_DIR"
install -m 0755 "$SRC_DIR/target/release/mtw" "$BIN_DIR/mtw"
step "Installed: $BIN_DIR/mtw"

# ───────────────────────────────────────────────────────── 4. SwiftLM
SLM_DIR="$DEPS_DIR/SwiftLM"
SLM_BIN="$SLM_DIR/.build/arm64-apple-macosx/release/SwiftLM"

if [ "${MTW_SKIP_SWIFTLM:-0}" = "1" ]; then
    warn "MTW_SKIP_SWIFTLM=1 → not building SwiftLM (mtw will fail at runtime until a binary is at $SLM_BIN)"
elif [ -x "$SLM_BIN" ]; then
    step "SwiftLM already built at $SLM_BIN"
else
    step "Cloning + building SwiftLM (~30 min, ~3 GB disk)"
    mkdir -p "$DEPS_DIR"
    if [ ! -d "$SLM_DIR/.git" ]; then
        git clone --recursive --depth 1 https://github.com/SharpAI/SwiftLM "$SLM_DIR"
    fi
    cd "$SLM_DIR"
    swift build -c release
    if [ ! -x "$SLM_BIN" ]; then
        fail "SwiftLM build finished but binary not at $SLM_BIN"
    fi
fi

# ───────────────────────────────────────────────────────── 5. starter model
MODEL_DIR="$DEPS_DIR/models/$MODEL_NAME"
HF_REPO="mlx-community/$MODEL_NAME"

if [ "${MTW_SKIP_MODEL:-0}" = "1" ]; then
    warn "MTW_SKIP_MODEL=1 → not downloading a model"
elif [ -f "$MODEL_DIR/config.json" ] && [ -f "$MODEL_DIR/model.safetensors" ]; then
    step "Model already present at $MODEL_DIR"
else
    step "Downloading $MODEL_NAME (~3.6 GB, ~5–15 min)"
    mkdir -p "$MODEL_DIR"
    for f in config.json model.safetensors model.safetensors.index.json tokenizer.json tokenizer_config.json special_tokens_map.json generation_config.json; do
        if [ -f "$MODEL_DIR/$f" ]; then continue; fi
        url="https://huggingface.co/$HF_REPO/resolve/main/$f"
        # Some files (model.safetensors.index.json, generation_config.json)
        # may legitimately not exist for every model. Try, but tolerate 404.
        if curl -L --fail -sS -o "$MODEL_DIR/$f.part" "$url"; then
            mv "$MODEL_DIR/$f.part" "$MODEL_DIR/$f"
            step "  fetched $f"
        else
            rm -f "$MODEL_DIR/$f.part"
            warn "  $f not on HF (skipped — may be optional)"
        fi
    done
    [ -f "$MODEL_DIR/config.json" ] && [ -f "$MODEL_DIR/model.safetensors" ] \
        || fail "model download incomplete — re-run the script"
fi

# ───────────────────────────────────────────────────────── 6. PATH
case ":$PATH:" in
    *":$BIN_DIR:"*) ;;
    *)
        warn "$BIN_DIR is not on your PATH — adding it now"
        SHELL_RC="$HOME/.zshrc"
        [ -f "$HOME/.bashrc" ] && SHELL_RC="$HOME/.bashrc"
        echo "export PATH=\"$BIN_DIR:\$PATH\"" >> "$SHELL_RC"
        export PATH="$BIN_DIR:$PATH"
        warn "appended to $SHELL_RC — open a new terminal to pick it up"
        ;;
esac

# ───────────────────────────────────────────────────────── 7. doctor + next step
echo
step "All set. Running mtw doctor:"
echo
"$BIN_DIR/mtw" doctor || true

echo
step "To start the engine + open the dashboard:"
echo
echo "    mtw start"
echo
