#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if [ "$#" -ne 0 ]; then
  echo "Usage: conda activate cfzerocrc && bash run.sh" >&2
  exit 2
fi

if [ -z "${CONDA_PREFIX:-}" ] || [ "${CONDA_DEFAULT_ENV:-}" = "base" ]; then
  echo "请先在 WSL2 中手动创建并激活非 base 的 Conda 环境：conda activate cfzerocrc" >&2
  exit 1
fi

PYTHON="$CONDA_PREFIX/bin/python"
if [ ! -x "$PYTHON" ]; then
  echo "当前 Conda 环境缺少 Python：$PYTHON" >&2
  exit 1
fi

"$PYTHON" - <<'PY'
import sys

modules = (
    "torch", "torchvision", "diffusers", "transformers", "accelerate",
    "safetensors", "modelscope", "huggingface_hub", "sentencepiece",
    "google.protobuf", "nibabel", "numpy", "scipy", "skimage", "sklearn",
    "PIL", "yaml", "pandas", "tqdm", "matplotlib", "cfzerocrc",
)
errors = []
for name in modules:
    try:
        __import__(name)
    except Exception as exc:
        errors.append(f"{name}: {exc}")
if errors:
    raise SystemExit(
        "当前 Conda 环境缺少依赖或依赖无法导入。请先手动安装 requirements.txt 和项目本身：\n"
        + "\n".join(errors)
    )
import torch
if not torch.cuda.is_available():
    raise SystemExit("当前 Conda 环境的 PyTorch 无法使用 CUDA；请检查 WSL2 驱动及 PyTorch 的 CUDA 安装。")
print(f"使用 Conda 环境：{sys.prefix}")
PY

"$PYTHON" scripts/run_all.py
