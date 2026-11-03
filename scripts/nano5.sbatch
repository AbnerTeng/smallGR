#!/usr/bin/env bash
#SBATCH --account=mst115094
#SBATCH --job-name=smallgr-smoke
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
if [[ "$(pwd -P)" != /work/abner0803/* ]]; then
    echo "Submit from the project copy under /work/abner0803." >&2
    exit 1
fi
source scripts/nano5_env.sh
source "$SMALLGR_VENV/bin/activate"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1

python - <<'PY'
import torch
assert torch.cuda.is_available(), 'No GPU visible in the Slurm allocation'
assert torch.cuda.device_count() == 1, 'Expected one allocated GPU'
print('PyTorch:', torch.__version__, 'CUDA:', torch.version.cuda)
print('GPU:', torch.cuda.get_device_name(0), flush=True)
PY

if [[ $# -eq 0 ]]; then
    if [[ ! -f data/smoke/nq/manifest.json ]]; then
        python -m src prepare --dataset Abner0803/NQ320k-semid --smoke --output-dir data/smoke/nq
    fi
    set -- smoke --data-dir data/smoke/nq
fi
bash scripts/run.sh "$@"
touch "logs/${SLURM_JOB_ID}.done"
