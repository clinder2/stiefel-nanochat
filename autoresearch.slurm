#!/bin/bash
#SBATCH --job-name=autoresearch
#SBATCH --account=gts-mtao8
#SBATCH --nodes=1 --ntasks-per-node=1 --gres=gpu:v100:1 #--cpus-per-task=2   
#SBATCH --time=04:00:00 #3
#SBATCH --mem=120G
#SBATCH --partition=gpu-v100
##### #SBATCH -C amd
#SBATCH --output=./logs/autoresearch_sweeps.out
#SBATCH -oReport-%j-SA_sweeps.out   

# Change to the directory from which the job was submitted
cd $SLURM_SUBMIT_DIR

# Run your bash script
#bash train_single_node.sh
bash autoresearch.sh