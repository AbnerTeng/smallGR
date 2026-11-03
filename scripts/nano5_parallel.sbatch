#!/usr/bin/env bash
#SBATCH --account=mst115094
#SBATCH --job-name=smallgr-parallel
#SBATCH --partition=dev
#SBATCH --array=0-1%2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
if [[ "$(pwd -P)" != /work/abner0803/* ]]; then
    echo "Submit from the project copy under /work/abner0803." >&2
    exit 1
fi
source scripts/nano5_env.sh
export HF_HUB_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

case "${2:-small}" in
    small) models=(small base) ;;
    large) models=(large 3b) ;;
    *) echo "Model group must be small or large" >&2; exit 1 ;;
esac
size="${models[$SLURM_ARRAY_TASK_ID]}"
mode="${1:-smoke}"
run_id="${3:-$SLURM_ARRAY_JOB_ID}"
case "$mode" in
    smoke)
        data_dir="data/smoke/nq-t5-$size"
        output_dir="outputs/smoke-parallel/$run_id/t5-$size"
        smoke_args=(--smoke)
        ;;
    train)
        data_dir="data/nq-t5-$size"
        output_dir="checkpoints/parallel/$run_id/t5-$size"
        smoke_args=()
        ;;
    *)
        echo "Usage: sbatch [options] scripts/nano5_parallel.sbatch [smoke|train] [small|large] [previous-run-id]" >&2
        exit 1
        ;;
esac

resume_args=()
if [[ -n "${3:-}" ]]; then
    if [[ ! "$run_id" =~ ^[0-9]+$ ]]; then
        echo "Previous run ID must be numeric" >&2
        exit 1
    fi
    if [[ -f "$output_dir/results.json" ]]; then
        echo "T5-$size already completed: $output_dir/results.json"
        touch "logs/${SLURM_JOB_ID}.done"
        exit 0
    fi
    checkpoint=$("$SMALLGR_VENV/bin/python" - "$output_dir" <<'PY'
import sys
from pathlib import Path

checkpoints = [p for p in Path(sys.argv[1]).glob('checkpoint-*')
               if p.name.removeprefix('checkpoint-').isdigit()
               and (p / 'trainer_state.json').is_file()]
if not checkpoints:
    raise SystemExit(f'No checkpoint to resume in {sys.argv[1]}')
print(max(checkpoints, key=lambda p: int(p.name.removeprefix('checkpoint-'))))
PY
    )
    resume_args=(--resume "$checkpoint")
fi

# Separate prepared directories let both tasks start without a writer race.
if [[ ! -f "$data_dir/manifest.json" ]]; then
    bash scripts/run.sh prepare --dataset Abner0803/NQ320k-semid \
        --output-dir "$data_dir" "${smoke_args[@]}"
fi
bash scripts/nano5.sbatch train --config "configs/t5-$size.yaml" \
    --data-dir "$data_dir" --output-dir "$output_dir" \
    --wandb-mode "${SMALLGR_WANDB_MODE:-offline}" "${smoke_args[@]}" "${resume_args[@]}"
