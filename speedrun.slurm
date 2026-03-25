#!/bin/bash
#SBATCH --job-name=speedrun-gpt
#SBATCH --account=gts-mtao8
#SBATCH --nodes=1 --ntasks-per-node=1 --gres=gpu:v100:1        
#SBATCH --time=00:08:00
#SBATCH --mem=100G
#SBATCH --partition=gpu-v100
#SBATCH --output=./logs/task.out
#SBATCH -oReport-%j.out   
#SBATCH --mail-user=clinder9@gatech.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# Change to the directory from which the job was submitted
cd $SLURM_SUBMIT_DIR

# Run your bash script
bash speedrun.sh
