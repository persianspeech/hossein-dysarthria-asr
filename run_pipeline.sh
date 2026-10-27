#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON="${HOSSEIN_PYTHON:-.venv-reference/bin/python}"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export OMP_NUM_THREADS=4
python3 -u run_pipeline.py --python "$PYTHON"
