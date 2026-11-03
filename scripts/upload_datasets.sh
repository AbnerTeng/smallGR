#!/usr/bin/env bash
set -euo pipefail
source "${SMALLGR_VENV:-$HOME/git/rar/.venv}/bin/activate"
export CUDA_VISIBLE_DEVICES=1
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
exec python -m src.upload_datasets "$@"
