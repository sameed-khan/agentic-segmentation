#!/usr/bin/env bash
# Alternative BF16 arms via vLLM (OpenAI-compatible, port 8094).
#
# The DEFAULT BF16 arm is the Transformers server (start_bf16_server.py) — Google's
# documented loading path. vLLM (>=0.15) is the serving alternative to validate, not
# the reference; run at most one of the two on port 8094.
#
# Usage: bash harness/local_models/medgemma/start_bf16_vllm.sh [--tmux]
set -euo pipefail

MODEL_ID="${MODEL_ID:-google/medgemma-1.5-4b-it}"
PORT="${PORT:-8094}"
HOST="${HOST:-127.0.0.1}"

if [ "${1:-}" = "--tmux" ]; then
  tmux new-session -d -s medgemma-bf16 -c "$PWD" \
    "vllm serve $MODEL_ID --dtype bfloat16 --host $HOST --port $PORT --max-model-len 8192"
  echo "started tmux session medgemma-bf16 (port $PORT)"
else
  exec vllm serve "$MODEL_ID" --dtype bfloat16 --host "$HOST" --port "$PORT" --max-model-len 8192
fi
