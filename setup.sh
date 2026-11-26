#!/usr/bin/env bash
# Create .venv and install everything. On a machine with an NVIDIA GPU this installs the CUDA build of torch.
set -euo pipefail
cd "$(dirname "$0")"
python3 -m venv .venv
.venv/bin/python -m pip install -q --upgrade pip
if command -v nvidia-smi >/dev/null 2>&1; then
  .venv/bin/python -m pip install -q -r requirements.txt            # PyPI torch on Linux is the CUDA build
else
  .venv/bin/python -m pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cpu
  .venv/bin/python -m pip install -q -r requirements.txt
fi
.venv/bin/python -c "import torch; print('torch', torch.__version__, '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none, will run on CPU')"
