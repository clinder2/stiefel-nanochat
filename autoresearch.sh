#!/bin/bash
set -euo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export TRAIN_DEVICE=cpu
export TRAIN_WORKERS="${TRAIN_WORKERS:-${SLURM_CPUS_PER_TASK:-32}}"
export DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-2}"

command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"

[ -d ".venv" ] || uv venv
uv sync --extra cpu
source .venv/bin/activate

python prepare.py
python train.py
