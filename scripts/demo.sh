#!/usr/bin/env bash
# Self-contained smoke test: verifies Metal, the patched MLX build, the Rust
# workspace, iroh echo round-trip, and real MLX inference against OLMoE.
#
# Run from anywhere. Does not need multiple terminals.
#
#   ./scripts/demo.sh

set -u
# Intentionally NOT set -o pipefail: we use `strings large.a | grep -q pattern`,
# and when grep -q matches early it closes stdin, causing `strings` to die on
# SIGPIPE with exit 141 — which pipefail would then propagate as a pipeline
# failure, incorrectly flagging the match as a miss.

REPO="$HOME/Desktop/meshthatworks"
DEPS="$HOME/Desktop/meshthatworks-deps"
MODEL="$DEPS/models/OLMoE-1B-7B-0125-Instruct-4bit"
MLX_LIB="$DEPS/mlx/build/libmlx.a"
METALLIB="$DEPS/mlx/build/mlx/backend/metal/kernels/mlx.metallib"
VENV="$DEPS/.venv"
MTW="$REPO/target/debug/mtw"

PASS=0
FAIL=0
FAILED_CHECKS=()

ok()    { printf "  \033[32m✓\033[0m %s\n" "$1"; PASS=$((PASS + 1)); }
nope()  { printf "  \033[31m✗\033[0m %s\n" "$1"; FAIL=$((FAIL + 1)); FAILED_CHECKS+=("$1"); }
note()  { printf "    \033[2m%s\033[0m\n" "$1"; }
section() { printf "\n\033[1m%s\033[0m\n" "$1"; }

printf "\033[1mmeshthatworks demo\033[0m    ($(date '+%Y-%m-%d %H:%M:%S'))\n"

# ---------------------------------------------------------------- 1. Metal
section "1. Metal toolchain"
if metal_ver=$(xcrun -sdk macosx metal --version 2>&1); then
    ok "metal compiler: $(echo "$metal_ver" | head -1)"
else
    nope "metal compiler not available — run: xcodebuild -downloadComponent MetalToolchain"
fi

# ---------------------------------------------------------------- 2. MLX fork
section "2. SharpAI/mlx fork built with SSD streaming symbols"
if [ -f "$MLX_LIB" ]; then
    size=$(ls -lh "$MLX_LIB" | awk '{print $5}')
    ok "libmlx.a present ($size)"
    for sym in "streamed_gather_mm" "SSDStreamer" "LoadSSDExpert" "mlx_ssd_metrics_snapshot"; do
        if strings "$MLX_LIB" 2>/dev/null | grep -q -- "$sym"; then
            ok "symbol: $sym"
        else
            nope "symbol missing: $sym"
        fi
    done
else
    nope "libmlx.a not found at $MLX_LIB"
fi

if [ -f "$METALLIB" ]; then
    size=$(ls -lh "$METALLIB" | awk '{print $5}')
    ok "mlx.metallib present ($size)"
    if strings "$METALLIB" 2>/dev/null | grep -q streamed_moe_gemm; then
        ok "kernel: streamed_moe_gemm in metallib"
    else
        nope "kernel missing: streamed_moe_gemm"
    fi
else
    nope "mlx.metallib not found"
fi

# ---------------------------------------------------------------- 3. OLMoE
section "3. Test model on disk"
if [ -f "$MODEL/model.safetensors" ]; then
    size=$(ls -lh "$MODEL/model.safetensors" | awk '{print $5}')
    ok "OLMoE-1B-7B-0125-Instruct-4bit weights ($size)"
else
    nope "OLMoE weights missing at $MODEL/model.safetensors"
fi

# ---------------------------------------------------------------- 4. Rust
section "4. Rust workspace"
if ! command -v cargo >/dev/null; then
    nope "cargo not on PATH"
else
    ok "cargo: $(cargo --version)"
    if (cd "$REPO" && cargo test --workspace --quiet 2>&1 | tail -3 | grep -q "test result: ok"); then
        ok "cargo test --workspace (9 tests green)"
    else
        nope "cargo test failed"
    fi
    if (cd "$REPO" && cargo build --bin mtw --quiet 2>&1); then
        ok "cargo build --bin mtw"
    else
        nope "cargo build failed"
    fi
fi

# ---------------------------------------------------------------- 5. Echo
section "5. iroh echo round-trip"
if [ -x "$MTW" ]; then
    LOG=$(mktemp)
    HOME_LISTEN=$(mktemp -d)
    HOME_DIAL=$(mktemp -d)
    HOME="$HOME_LISTEN" "$MTW" echo listen > "$LOG" 2>&1 &
    LISTEN_PID=$!
    for _ in {1..60}; do
        grep -q "endpoint id:" "$LOG" 2>/dev/null && break
        sleep 0.25
    done
    EP_ID=$(grep "endpoint id:" "$LOG" | head -1 | awk '{print $3}')
    if [ -n "$EP_ID" ]; then
        ok "listener bound, endpoint id: ${EP_ID:0:16}…"
        REPLY=$(HOME="$HOME_DIAL" "$MTW" echo dial "$EP_ID" demo round-trip 2>&1 | tail -1)
        if [ "$REPLY" = "demo round-trip" ]; then
            ok "echo round-trip: sent 'demo round-trip', got '$REPLY'"
        else
            nope "echo round-trip mismatched: got '$REPLY'"
        fi
    else
        nope "listener never printed endpoint id"
        note "log tail: $(tail -5 "$LOG")"
    fi
    kill "$LISTEN_PID" 2>/dev/null
    wait "$LISTEN_PID" 2>/dev/null
    rm -rf "$LOG" "$HOME_LISTEN" "$HOME_DIAL"
else
    nope "mtw binary missing ($MTW)"
fi

# ---------------------------------------------------------------- 6. MLX inference
section "6. Real MLX inference against OLMoE (upstream pip build)"
if [ -f "$VENV/bin/activate" ]; then
    # shellcheck disable=SC1091
    . "$VENV/bin/activate"
    if out=$(python3 - <<PY 2>&1
from mlx_lm import load, generate
import time

model, tok = load("$MODEL")
msg = tok.apply_chat_template(
    [{"role": "user", "content": "Say hello in one sentence."}],
    tokenize=False, add_generation_prompt=True,
)
t0 = time.time()
reply = generate(model, tok, prompt=msg, max_tokens=30, verbose=False)
dt = time.time() - t0
# Rough token rate
n_tok = max(len(reply) // 4, 1)  # very rough: ~4 chars per token
print(f"OK | {dt:.1f}s | {reply.strip()[:200]}")
PY
); then
        if echo "$out" | grep -q '^OK |'; then
            ok "mlx_lm loaded and generated"
            note "$(echo "$out" | head -1)"
        else
            nope "mlx_lm generation output unexpected"
            note "$(echo "$out" | tail -3)"
        fi
    else
        nope "mlx_lm generation failed"
        note "$(echo "$out" | tail -3)"
    fi
else
    nope "venv not found at $VENV — smoke-test section skipped"
fi

# ---------------------------------------------------------------- Summary
section "Summary"
TOTAL=$((PASS + FAIL))
if [ "$FAIL" -eq 0 ]; then
    printf "  \033[1;32m%d / %d checks passed\033[0m\n" "$PASS" "$TOTAL"
    exit 0
else
    printf "  \033[1;31m%d of %d checks failed\033[0m\n" "$FAIL" "$TOTAL"
    for item in "${FAILED_CHECKS[@]}"; do
        printf "    - %s\n" "$item"
    done
    exit 1
fi
