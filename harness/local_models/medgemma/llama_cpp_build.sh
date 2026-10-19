#!/usr/bin/env python3
"""Build llama.cpp for CUDA at a pinned commit (A100 = sm_80), for MedGemma Q4_K_M.

The commit is pinned for reproducibility (master when written, 2026-10-19); after a
successful MedGemma smoke test on the A100, treat this script as the frozen build
recipe (record the built binary's sha256 in the run metadata).

Usage:  bash harness/local_models/medgemma/llama_cpp_build.sh [DEST_DIR]
        (default DEST_DIR=/opt/llama.cpp; override with LLAMA_CPP_COMMIT / CMAKE_JOBS)
"""
set -euo pipefail

DEST_DIR="${1:-/opt/llama.cpp}"
LLAMA_CPP_COMMIT="${LLAMA_CPP_COMMIT:-1b29621bacc6a446bd523fc790316a24a06e886f}"
CMAKE_JOBS="${CMAKE_JOBS:-$(nproc)}"
CUDA_ARCH="${CUDA_ARCH:-80}"   # A100 = sm_80 (never build for other archs on this box)

if [ ! -d "$DEST_DIR/.git" ]; then
  git clone https://github.com/ggml-org/llama.cpp "$DEST_DIR"
fi
cd "$DEST_DIR"
git fetch origin master --tags
git checkout --force "$LLAMA_CPP_COMMIT"

cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
      -DGGML_NATIVE=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
      -DLLAMA_BUILD_SERVER=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j"$CMAKE_JOBS" --target llama-server llama-cli

echo
echo "built llama-server at commit $LLAMA_CPP_COMMIT (sm_$CUDA_ARCH):"
echo "  $DEST_DIR/build/bin/llama-server"
sha256sum "$DEST_DIR/build/bin/llama-server" || true
