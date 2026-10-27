#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON="${HOSSEIN_PYTHON:-.venv-reference/bin/python}"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export OMP_NUM_THREADS=4
"$PYTHON" verify_data.py
"$PYTHON" -m pip freeze > environment.freeze.txt
if [ ! -f smoke.json ]; then "$PYTHON" -u train.py smoke; fi
if [ ! -f resume_verification.json ]; then "$PYTHON" -u train.py smoke-resume; fi
if [ ! -f training_completed.json ]; then "$PYTHON" -u train.py train; fi
if [ ! -f final_results.json ]; then "$PYTHON" -u train.py final; fi
"$PYTHON" -u build_model_release.py
