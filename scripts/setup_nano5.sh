#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ "$(pwd -P)" != /work/abner0803/* ]]; then
    echo "Run setup from the project copy under /work/abner0803." >&2
    exit 1
fi
source scripts/nano5_env.sh
mkdir -p "$XDG_CACHE_HOME" "$WANDB_CONFIG_DIR" "$WANDB_DATA_DIR" logs
# Axolotl and several small Python packages require source packaging.
# This lock does not include CUDA extensions that need compilation.
uv python install 3.12.13 --no-bin
uv sync --frozen --no-install-project --managed-python --python 3.12.13
source "$SMALLGR_VENV/bin/activate"
python - "$@" <<'PY'
import sys
from huggingface_hub import snapshot_download
from src.data import HF_DATASET, HF_REVISION

snapshot_download(
    HF_DATASET, repo_type='dataset', revision=HF_REVISION,
    allow_patterns=['data/*.jsonl', 'artifacts/*'],
)

sizes = sys.argv[1:] or ['small', 'base']
if any(size not in ('small', 'base', 'large', '3b') for size in sizes):
    raise ValueError('Model sizes must be small, base, large, or 3b')
for size in sizes:
    snapshot_download(
        f'google-t5/t5-{size}',
        allow_patterns=[
            'config.json', 'generation_config.json', 'model.safetensors',
            'spiece.model', 'tokenizer*', 'special_tokens_map.json', 'added_tokens.json',
        ],
    )
PY
