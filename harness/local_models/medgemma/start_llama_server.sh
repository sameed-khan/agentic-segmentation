#!/usr/bin/env bash
# Start MedGemma 1.5 4B IT (Q4_K_M GGUF + mmproj-F16) via llama.cpp llama-server,
# OpenAI-compatible, port 8091 (Phase 5 deployment-realism arm).
#
# Weights (downloaded by setup_a100.sh): unsloth/medgemma-1.5-4b-it-GGUF
#   medgemma-1.5-4b-it-Q4_K_M.gguf (~2.49 GB) + mmproj-F16.gguf (~0.85 GB)
# Community conversion (NOT an official Google GGUF); HAI-DEF terms apply.
#
# Usage: bash harness/local_models/medgemma/start_llama_server.sh [--tmux]
#   env overrides: LLAMA_CPP_DIR, MEDGEMMA_MODEL_DIR, PORT, CTX_SIZE
set -euo pipefail

LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-/opt/llama.cpp}"
MEDGEMMA_MODEL_DIR="${MEDGEMMA_MODEL_DIR:-$HOME/models/medgemma15}"
PORT="${PORT:-8091}"
CTX_SIZE="${CTX_SIZE:-8192}"
HOST="${HOST:-127.0.0.1}"

SERVER="$LLAMA_CPP_DIR/build/bin/llama-server"
MODEL="$MEDGEMMA_MODEL_DIR/medgemma-1.5-4b-it-Q4_K_M.gguf"
MMPROJ="$MEDGEMMA_MODEL_DIR/mmproj-F16.gguf"

[ -x "$SERVER" ] || { echo "llama-server not built: $SERVER (run llama_cpp_build.sh first)" >&2; exit 1; }
[ -f "$MODEL" ] || { echo "missing $MODEL (run setup_a100.sh to download; accept HAI-DEF terms + HF_TOKEN)" >&2; exit 1; }
[ -f "$MMPROJ" ] || { echo "missing $MMPROJ (same repo; must pair with this quant, never mix mmproj files)" >&2; exit 1; }

CMD=("$SERVER"
  --model "$MODEL"
  --mmproj "$MMPROJ"
  --n-gpu-layers 99
  --ctx-size "$CTX_SIZE"
  --parallel 1
  --jinja
  --host "$HOST"
  --port "$PORT"
)

if [ "${1:-}" = "--tmux" ]; then
  # detached tmux session, same name the runner queries for logs
  tmux new-session -d -s medgemma-q4 -c "$MEDGEMMA_MODEL_DIR" "${CMD[*]}"
  echo "started tmux session medgemma-q4 (port $PORT); attach: tmux attach -t medgemma-q4"
else
  exec "${CMD[@]}"
fi
