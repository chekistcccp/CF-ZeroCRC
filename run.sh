#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if [ "$#" -ne 0 ]; then
  echo "Usage: bash run.sh" >&2
  exit 2
fi

SYSTEM_PYTHON="${PYTHON_BIN:-python3}"
VENV=".venv"
if [ ! -x "$VENV/bin/python" ]; then
  "$SYSTEM_PYTHON" -m venv "$VENV"
fi
PYTHON="$VENV/bin/python"

DEPS_KEY="$(sha256sum requirements.txt pyproject.toml | sha256sum | cut -d' ' -f1)"
INSTALLED_KEY="$(cat "$VENV/.cfzerocrc-deps" 2>/dev/null || true)"
if [ "$INSTALLED_KEY" != "$DEPS_KEY" ]; then
  "$PYTHON" -m pip install --upgrade pip
  "$PYTHON" -m pip install torch torchvision
  "$PYTHON" -m pip install -r requirements.txt -e .
  printf '%s\n' "$DEPS_KEY" > "$VENV/.cfzerocrc-deps"
fi

"$PYTHON" -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable in WSL2; check the NVIDIA driver and WSL GPU passthrough"'
"$PYTHON" scripts/run_all.py
