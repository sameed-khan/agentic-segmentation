#!/usr/bin/env bash
# Tear down: close the SSH master (and forward), stop the exec stream, release the Colab VM.
set -uo pipefail
ssh -O exit colab-nnint 2>/dev/null
pkill -f "colab exec -s nnint -f" 2>/dev/null
colab stop -s nnint
colab usage 2>/dev/null | head -3
