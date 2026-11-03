#!/usr/bin/env bash
set -euo pipefail

cd /work/abner0803/smallGR
mkdir -p logs

case "${1:-train}" in
    smoke)
        partition=dev
        walltime=01:00:00
        mode=smoke
        resume_args=()
        ;;
    train)
        partition=normal2
        walltime=48:00:00
        mode=train
        resume_args=()
        ;;
    *)
        if [[ ! "$1" =~ ^[0-9]+$ ]]; then
            echo "Usage: ./train_large.sh [train|smoke|previous-run-id]" >&2
            exit 1
        fi
        partition=normal2
        walltime=48:00:00
        mode=train
        resume_args=("$1")
        ;;
esac

exec sbatch \
    --account=mst115094 \
    --partition="$partition" \
    --time="$walltime" \
    --job-name=smallgr-large \
    --array=0-1%2 \
    scripts/nano5_parallel.sbatch "$mode" large "${resume_args[@]}"
