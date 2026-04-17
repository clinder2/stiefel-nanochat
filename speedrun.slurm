#!/bin/bash
#SBATCH --job-name=speedrun-gpt
#SBATCH --account=gts-mtao8
#SBATCH --nodes=4 --ntasks-per-node=16
#SBATCH --time=04:00:00
#SBATCH --mem=100G
#SBATCH --partition=cpu
#SBATCH --output=./logs/task.out
#SBATCH -oReport-%j.out   

# Change to the directory from which the job was submitted
cd $SLURM_SUBMIT_DIR

# Run your bash script
bash speedrun.sh
