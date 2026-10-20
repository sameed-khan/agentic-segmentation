#!/usr/bin/env bash
# Download the two new Phase 5 Study A pointer checkpoints into $HOME/models and
# record the resolved pinned revisions in weights_manifest_extra.json
# (weights_manifest.json is left untouched — it belongs to setup_a100.sh).
# Revisions are resolved ONCE via the HF API and passed explicitly to
# snapshot_download, so download and manifest always agree (setup_a100.sh convention).
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
MODELS_DIR="${MODELS_DIR:-$HOME/models}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"   # harness/local_models -> worktree root

set -a; source /home/ubuntu/agentic-segmentation/.env; set +a

uv run --project "$ROOT/harness/local_models/pointd_molmopoint" --with "huggingface_hub[cli]" \
  python - "$MODELS_DIR" <<'EOF'
import json, os, sys
from huggingface_hub import HfApi, snapshot_download
models_dir = sys.argv[1]
targets = {
    "allenai/MolmoPoint-8B": os.path.join(models_dir, "MolmoPoint-8B"),
    "Qwen/Qwen3-VL-8B-Instruct": os.path.join(models_dir, "Qwen3-VL-8B-Instruct"),
}
resolved = {}
for repo, dest in targets.items():
    sha = HfApi().model_info(repo, token=os.environ.get("HF_TOKEN")).sha
    print(f"[weights] downloading {repo}@{sha} -> {dest}", flush=True)
    snapshot_download(repo, local_dir=dest, revision=sha, token=os.environ.get("HF_TOKEN"))
    resolved[repo] = {"revision": sha, "local_dir": dest}
    print(f"[weights] {repo} pinned at {sha}", flush=True)
manifest_path = os.path.join(models_dir, "weights_manifest_extra.json")
existing = {"snapshots": {}}
if os.path.exists(manifest_path):
    with open(manifest_path) as f:
        existing = json.load(f)
existing["snapshots"].update(resolved)
existing["note"] = ("pinned HF revisions for the Phase 5 Study A pointer additions "
                    "(pointd_molmopoint / pointd_qwen3vl; recorded separately from "
                    "setup_a100.sh's weights_manifest.json, which is not edited)")
with open(manifest_path, "w") as f:
    json.dump(existing, f, indent=2)
print(f"[weights] wrote {manifest_path}", flush=True)
EOF
