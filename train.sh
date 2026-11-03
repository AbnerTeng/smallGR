#!/usr/bin/env bash
set -euo pipefail

cd /work/abner0803/smallGR
mkdir -p logs

exec sbatch \
    --account=mst115094 \
    --partition=normal \
    --time=24:00:00 \
    --job-name=smallgr-train \
    --array=0-1%2 \
    scripts/nano5_parallel.sbatch train
