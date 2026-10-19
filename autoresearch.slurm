#!/bin/bash
#SBATCH --job-name=autoresearch
#SBATCH --account=gts-mtao8
#SBATCH --partition=cpu-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00
#SBATCH --mem=120G
#SBATCH --output=Report-%j-ComparisonV1_StiefelAdam_sweeps.out
#SBATCH --error=Report-%j-ComparisonV1_StiefelAdam_sweeps.err

cd "$SLURM_SUBMIT_DIR"
export TRAIN_WORKERS="$SLURM_CPUS_PER_TASK"
export DEVICE_BATCH_SIZE=2

bash autoresearch.sh
