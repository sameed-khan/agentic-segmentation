#!/usr/bin/env bash
# Isolated experimental pi launch check:
#   - no discovered/configured extensions, skills, prompts, themes, context files, MCP
#   - only our extension + the Anthropic OAuth shim (needed for subscription billing)
#   - auth inherited from ~/.pi/agent/auth.json
#   - only seg_* tools exposed
# Expected: model calls seg_ping, then answers with the circle colour, background colour, digit.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
AUTH_SHIM="${AUTH_SHIM:-$HOME/.pi/agent/npm/node_modules/@gotgenes/pi-anthropic-auth}"
MODEL="${MODEL:-anthropic/claude-sonnet-5-5}"

pi -ne -e "$HERE/ping_ext.ts" -e "$AUTH_SHIM" \
   --no-mcp -ns -np -nc --no-themes -na --no-session \
   --tools 'seg_*' --model "$MODEL" \
   --system-prompt "You are a test agent. Use tools when asked." \
   -p "Call seg_ping once. Then report: (1) the text it returned, (2) the colour of the circle, (3) the background colour, (4) the digit drawn in the image. Also list every tool name you have access to."
