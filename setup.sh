#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export PIP_DISABLE_PIP_VERSION_CHECK=1
.venv/bin/python -m pip install --timeout 120 'torch==2.11.0+cu126' 'torchaudio==2.11.0+cu126' --index-url https://download.pytorch.org/whl/cu126
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip freeze > environment.freeze.txt
.venv/bin/python -c 'import torch; print(torch.__version__, torch.cuda.is_available()); from nemo.collections.asr.models import ASRModel; print("ASR_IMPORT_OK")'
touch ENV_READY
