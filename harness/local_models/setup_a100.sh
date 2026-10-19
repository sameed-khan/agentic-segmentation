#!/usr/bin/env bash
# Phase 5 A100 setup: venvs, weights, llama.cpp build, services in tmux (one-time).
#
# Prerequisites on the A100 box:
#   - uv, git, cmake, NVIDIA driver + CUDA 12.x runtime (sm_80 / A100)
#   - Hugging Face account that has ACCEPTED THE HAI-DEF TERMS for MedGemma:
#       https://developers.google.com/health-ai-developer-foundations/terms
#     (google/medgemma-1.5-4b-it is gated; the unsloth GGUF mirror carries the same
#     terms). Export HF_TOKEN with read access before running this script:
#       export HF_TOKEN=hf_...
#   - 80 GB VRAM plan (everything coexists; see research/phase5_implementation.md):
#       nni ~5 GB + Molmo2 BF16 ~19 GB + GUI-Actor BF16 ~7 GB
#       + MedGemma BF16 ~10 GB + MedGemma Q4 ~5 GB
#
# Usage: bash harness/local_models/setup_a100.sh [--weights-only|--services-only]
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="${MODELS_DIR:-$HOME/models}"
GUI_ACTOR_DIR="${GUI_ACTOR_DIR:-/opt/GUI-Actor}"
GUI_ACTOR_COMMIT="${GUI_ACTOR_COMMIT:-d98d1bbd01862f9112114b83b032f492c365a173}"

step() { printf '\n=== %s\n' "$*"; }

# ---------------------------------------------------------------- 1. uv projects
if [ "${1:-}" != "--services-only" ]; then
  step "1/5 creating uv venvs (pinned versions per the research notes)"
  uv sync --project "$HERE/pointd_gui_actor"            # transformers==4.51.3, qwen-vl-utils==0.0.8, accelerate==1.1.1
  uv sync --project "$HERE/pointd_molmo2"               # transformers==4.57.1 (trust_remote_code at load)
  uv sync --project "$HERE/medgemma"                    # transformers==4.57.1 (validate vs the 1.5 card)

  # ------------------------------------------------------------- 2. weights
  step "2/5 downloading weights into $MODELS_DIR (GUI-Actor, Molmo2 public; MedGemma needs HF_TOKEN)"
  uv run --project "$HERE/pointd_gui_actor" --with "huggingface_hub[cli]" python - "$MODELS_DIR" <<'EOF'
import os, sys
from huggingface_hub import snapshot_download
models_dir = sys.argv[1]
targets = [
    ("microsoft/GUI-Actor-2B-Qwen2-VL", os.path.join(models_dir, "GUI-Actor-2B-Qwen2-VL")),
    ("allenai/Molmo2-8B",               os.path.join(models_dir, "Molmo2-8B")),
    ("google/medgemma-1.5-4b-it",       os.path.join(models_dir, "medgemma-1.5-4b-it")),
]
for repo, dest in targets:
    print(f"downloading {repo} -> {dest}")
    snapshot_download(repo, local_dir=dest, token=os.environ.get("HF_TOKEN"))
EOF
  # MedGemma GGUF (deployment arm): Q4_K_M quant + its F16 mmproj from the same repo.
  uv run --project "$HERE/medgemma" --with "huggingface_hub[cli]" python - "$MODELS_DIR" <<'EOF'
import os, sys
from huggingface_hub import hf_hub_download
models_dir = sys.argv[1]
dest = os.path.join(models_dir, "medgemma15")
for fname in ["medgemma-1.5-4b-it-Q4_K_M.gguf", "mmproj-F16.gguf"]:
    path = hf_hub_download("unsloth/medgemma-1.5-4b-it-GGUF", fname, local_dir=dest,
                           token=os.environ.get("HF_TOKEN"))
    print(f"downloaded {path}")
EOF

  # ------------------------------------------------------------- 3. llama.cpp
  step "3/5 building llama.cpp (CUDA, sm_80, pinned commit)"
  bash "$HERE/medgemma/llama_cpp_build.sh" "${LLAMA_CPP_DIR:-/opt/llama.cpp}"

  # ------------------------------------------------------------- 4. GUI-Actor repo
  step "4/5 cloning official microsoft/GUI-Actor at pinned commit"
  if [ ! -d "$GUI_ACTOR_DIR/.git" ]; then
    git clone https://github.com/microsoft/GUI-Actor "$GUI_ACTOR_DIR"
  fi
  git -C "$GUI_ACTOR_DIR" fetch origin main
  git -C "$GUI_ACTOR_DIR" checkout --force "$GUI_ACTOR_COMMIT"
fi

# ---------------------------------------------------------------- 5. services
if [ "${1:-}" != "--weights-only" ]; then
  step "5/5 starting services in tmux (loopback only)"
  tmux has-session -t pointd-gui 2>/dev/null || \
  GUI_ACTOR_REPO="$GUI_ACTOR_DIR" MODELS_DIR="$MODELS_DIR" tmux new-session -d -s pointd-gui -c "$HERE/pointd_gui_actor" \
    "uv run --project $HERE/pointd_gui_actor python server.py \
       --model-id $MODELS_DIR/GUI-Actor-2B-Qwen2-VL --gui-actor-repo $GUI_ACTOR_DIR --port 8092"
  tmux has-session -t pointd-molmo 2>/dev/null || \
  tmux new-session -d -s pointd-molmo -c "$HERE/pointd_molmo2" \
    "uv run --project $HERE/pointd_molmo2 python server.py \
       --model-id $MODELS_DIR/Molmo2-8B --port 8093"
  bash "$HERE/medgemma/start_llama_server.sh" --tmux
  tmux has-session -t medgemma-bf16 2>/dev/null || \
  tmux new-session -d -s medgemma-bf16 -c "$HERE/medgemma" \
    "uv run --project $HERE/medgemma python start_bf16_server.py \
       --model-id $MODELS_DIR/medgemma-1.5-4b-it --port 8094"

  step "service status (wait for weight loads, then run smoke_test.py)"
  tmux ls 2>/dev/null || true
  cat <<'EOF'

Next:  python harness/local_models/smoke_test.py --prepare-dir <results/phase2_dev/prepare>
Ports: 8091 medgemma-q4 | 8092 pointd-gui | 8093 pointd-molmo | 8094 medgemma-bf16
Logs:  tmux attach -t <session>   (pointd-gui | pointd-molmo | medgemma-q4 | medgemma-bf16)
EOF
fi
