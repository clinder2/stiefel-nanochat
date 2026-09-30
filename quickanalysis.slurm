#!/bin/bash
#SBATCH --job-name=quick
#SBATCH --account=gts-mtao8
#SBATCH --nodes=1 --ntasks-per-node=1
#SBATCH --time=00:10:00
#SBATCH --mem=100G
#SBATCH --partition=cpu
#SBATCH --output=./logs/task.out
#SBATCH -oReport-%j.out   

# Change to the directory from which the job was submitted
cd $SLURM_SUBMIT_DIR

# Run your bash script
bash quickfile.sh