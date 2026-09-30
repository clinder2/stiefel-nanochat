#!/bin/bash
#SBATCH --mail-type=BEGIN
#SBATCH --mail-user=clinder9@gatech.edu
#SBATCH --job-name=speedrun-gpt
#SBATCH --account=gts-mtao8
#SBATCH --nodes=1 --ntasks-per-node=1 --gres=gpu:h100:1
#SBATCH --time=15:00:00
#SBATCH --mem=100G
#SBATCH --partition=gpu-h100
#SBATCH --output=./logs/task.out
#SBATCH -oReport-%j.out   

# Change to the directory from which the job was submitted
cd $SLURM_SUBMIT_DIR

# Run your bash script
bash speedrun.sh
