#!/usr/bin/env bash
set -euo pipefail
source "${SMALLGR_VENV:-$HOME/git/rar/.venv}/bin/activate"
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    export CUDA_VISIBLE_DEVICES=1
fi
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export PYTHONPATH="$(cd "$(dirname "$0")/.." && pwd)${PYTHONPATH:+:$PYTHONPATH}"
exec python -m src "$@"
