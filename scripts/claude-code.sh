#!/usr/bin/env bash
# Print the env vars Claude Code needs to talk to a local `mtw serve`.
#
# Usage:
#   eval "$(./scripts/claude-code.sh)"
#   claude
#
# Or copy the lines into your shell profile.

set -euo pipefail

PORT="${MTW_PROXY_PORT:-9337}"
URL="http://127.0.0.1:${PORT}"

if ! curl -sf "${URL}/healthz" >/dev/null 2>&1; then
    echo "# warning: mtw serve is not responding at ${URL}/healthz" >&2
    echo "# start it first with:  mtw serve" >&2
fi

cat <<EOF
# Point Claude Code (and any other OpenAI-compatible client) at mtw.
export ANTHROPIC_BASE_URL="${URL}"
export ANTHROPIC_API_KEY="local-mtw-no-auth"
# Some tools use OpenAI-style env names; mtw answers /v1/* the same way.
export OPENAI_BASE_URL="${URL}/v1"
export OPENAI_API_KEY="local-mtw-no-auth"
EOF
