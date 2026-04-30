# # -----------------------------------------------------------------------------
# # Python venv setup with uv

# # install uv (if not already installed)
# command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh

# source $HOME/.local/bin/env

# # create a .venv local virtual environment (if it doesn't exist)
# [ -d ".venv" ] || uv venv
# # install the repo dependencies
# uv sync
# # activate venv so that `python` uses the project's venv instead of system python
# source .venv/bin/activate
# echo "here"
# uv run prepare.py
# echo "here"
# uv run train.py


export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/r-mtao8-0/.cache/autoresearch"
mkdir -p $NANOCHAT_BASE_DIR
echo "here"

# -----------------------------------------------------------------------------
# Python venv setup with uv

# install uv (if not already installed)
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh

source $HOME/.local/bin/env

echo "here"
# create a .venv local virtual environment (if it doesn't exist)
[ -d ".venv" ] || uv venv
# install the repo dependencies
uv sync
# activate venv so that `python` uses the project's venv instead of system python
source .venv/bin/activate

echo "here"
#torchrun prepare.py
# python3 -m train

echo "here"
#orchrun train_fullscale.py
torchrun train_numheads.py

# #!/bin/bash

# ###############################################################################
# # PACE PHOENIX CPU Training Script
# #
# # Initializes venv and runs train.py with distributed hyperparameter configs
# # Uses multiprocessing.Pool.starmap to distribute configs across CPU cores
# #
# # Usage:
# #   ./run_pace_phoenix.sh                    # Interactive submission
# #   sbatch run_pace_phoenix.sh               # SLURM batch submission
# #   ./run_pace_phoenix.sh --slurm             # Create SLURM submission script
# #
# # For interactive testing on login node:
# #   ./run_pace_phoenix.sh --test              # Limited grid, fewer workers
# #
# ###############################################################################

# set -e

# # ============================================================================
# # SLURM Configuration (if running via sbatch)
# # ============================================================================
# #SBATCH --job-name=autoresearch-train
# #SBATCH --partition=cpu
# #SBATCH --qos=debug
# #SBATCH --nodes=1
# #SBATCH --ntasks-per-node=64
# #SBATCH --time=04:00:00
# #SBATCH --output=logs/slurm_%j.log
# #SBATCH --error=logs/slurm_%j.err
# #SBATCH --mail-type=END,FAIL
# #SBATCH --mail-user=$USER@gatech.edu

# # ============================================================================
# # Configuration
# # ============================================================================

# PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# VENV_DIR="${PROJECT_DIR}/.venv"
# LOG_DIR="${PROJECT_DIR}/logs"
# mkdir -p "${LOG_DIR}"

# # Get CPU info
# if [ -n "$SLURM_NTASKS_PER_NODE" ]; then
#     # Running under SLURM
#     NUM_CORES=$SLURM_NTASKS_PER_NODE
#     SUBMIT_HOST="PHOENIX SLURM"
# else
#     # Running interactively
#     NUM_CORES=$(nproc)
#     SUBMIT_HOST="Local/Interactive"
# fi

# # Determine number of workers
# # For PHOENIX: Use fewer workers than cores to leave headroom for system/I/O
# NUM_WORKERS="${NUM_WORKERS:-$(( NUM_CORES / 2 ))}"
# if [ $NUM_WORKERS -lt 1 ]; then
#     NUM_WORKERS=1
# fi

# # Parse command line arguments
# TEST_MODE=false
# CREATE_SLURM=false

# while [[ $# -gt 0 ]]; do
#     case $1 in
#         --test)
#             TEST_MODE=true
#             NUM_WORKERS=2
#             shift
#             ;;
#         --slurm)
#             CREATE_SLURM=true
#             shift
#             ;;
#         *)
#             echo "Unknown option: $1"
#             exit 1
#             ;;
#     esac
# done

# # ============================================================================
# # Logging Setup
# # ============================================================================

# TIMESTAMP=$(date +%Y%m%d_%H%M%S)
# LOG_FILE="${LOG_DIR}/train_${TIMESTAMP}.log"

# # Redirect all output to log file and console
# exec 1> >(tee -a "${LOG_FILE}")
# exec 2>&1

# # Colors
# GREEN='\033[0;32m'
# YELLOW='\033[1;33m'
# BLUE='\033[0;34m'
# RED='\033[0;31m'
# NC='\033[0m'

# log_info() {
#     echo -e "${GREEN}[INFO]${NC} $1"
# }

# log_warn() {
#     echo -e "${YELLOW}[WARN]${NC} $1"
# }

# log_error() {
#     echo -e "${RED}[ERROR]${NC} $1"
# }

# log_step() {
#     echo -e "${BLUE}[STEP]${NC} $1"
# }

# # ============================================================================
# # Create SLURM Submission Script (optional)
# # ============================================================================

# if [ "$CREATE_SLURM" = true ]; then
#     log_step "Creating SLURM submission script"
    
#     SLURM_SCRIPT="${PROJECT_DIR}/submit_pace_phoenix.sh"
#     cat > "$SLURM_SCRIPT" << 'SLURM_EOF'
# #!/bin/bash
# #SBATCH --job-name=autoresearch-train
# #SBATCH --partition=cpu
# #SBATCH --qos=debug
# #SBATCH --nodes=1
# #SBATCH --ntasks-per-node=64
# #SBATCH --time=04:00:00
# #SBATCH --output=logs/slurm_%j.log
# #SBATCH --error=logs/slurm_%j.err
# #SBATCH --mail-type=END,FAIL

# # Run the training script
# cd "$(dirname "${BASH_SOURCE[0]}")"
# ./run_pace_phoenix.sh
# SLURM_EOF
    
#     chmod +x "$SLURM_SCRIPT"
#     log_info "Created: $SLURM_SCRIPT"
#     log_info "Submit with: sbatch submit_pace_phoenix.sh"
#     exit 0
# fi

# # ============================================================================
# # Step 1: System Information
# # ============================================================================

# log_step "Step 1: System Information"

# echo ""
# log_info "Submission Host: $SUBMIT_HOST"
# echo "  Available CPU Cores: $NUM_CORES"
# echo "  Multiprocessing Workers: $NUM_WORKERS"
# echo "  Log File: $LOG_FILE"

# if [ -n "$SLURM_JOB_ID" ]; then
#     echo "  SLURM Job ID: $SLURM_JOB_ID"
#     echo "  SLURM Partition: $SLURM_PARTITION"
# fi

# if [ "$TEST_MODE" = true ]; then
#     log_warn "Running in TEST MODE with limited hyperparameter grid"
# fi

# echo ""

# # ============================================================================
# # Step 2: Python Virtual Environment
# # ============================================================================

# log_step "Step 2: Setting up Python virtual environment"

# if [ ! -d "$VENV_DIR" ]; then
#     log_info "Creating virtual environment..."
#     python3 -m venv "$VENV_DIR"
# else
#     log_info "Virtual environment already exists"
# fi

# log_info "Activating virtual environment..."
# source "$VENV_DIR/bin/activate"

# # log_info "Upgrading pip, setuptools, wheel..."
# # pip install --quiet --upgrade pip setuptools wheel
# uv sync
# # log_info "Installing project dependencies..."
# # if [ -f "${PROJECT_DIR}/pyproject.toml" ]; then
# #     pip install --quiet -e "${PROJECT_DIR}"
# # else
# #     # Fallback: install core dependencies
# #     pip install --quiet torch networkx
# # fi

# log_info "Python environment ready"
# echo "  Python: $(python --version)"

# echo ""

# # ============================================================================
# # Step 3: Data Preparation
# # ============================================================================

# log_step "Step 3: Data preparation"

# if [ ! -f "${PROJECT_DIR}/data.bin" ] && [ ! -d "${PROJECT_DIR}/data" ]; then
#     log_info "Running prepare.py..."
#     python "${PROJECT_DIR}/prepare.py" 2>&1 | tee -a "${LOG_FILE}"
    
#     if [ $? -eq 0 ]; then
#         log_info "Data preparation successful"
#     else
#         log_error "Data preparation failed!"
#         exit 1
#     fi
# else
#     log_info "Data already prepared (data.bin or data/ exists)"
# fi

# echo ""

# # ============================================================================
# # Step 4: Environment Configuration
# # ============================================================================

# log_step "Step 4: Configuring training environment"

# # CPU optimization settings
# export OMP_NUM_THREADS=$((NUM_CORES / NUM_WORKERS))
# export OPENBLAS_NUM_THREADS=$((NUM_CORES / NUM_WORKERS))
# export MKL_NUM_THREADS=$((NUM_CORES / NUM_WORKERS))
# export TORCH_NUM_THREADS=$((NUM_CORES / NUM_WORKERS))
# export NUMEXPR_NUM_THREADS=$((NUM_CORES / NUM_WORKERS))

# # Memory settings
# export PYTORCH_ALLOC_CONF="expandable_segments:True"
# export MALLOC_ARENA_MAX=2
# export MALLOC_TRIM_THRESHOLD_=100000
# export MALLOC_MMAP_THRESHOLD_=1000000

# # Python settings
# export PYTHONUNBUFFERED=1
# export PYTHONHASHSEED=0

# # Disable progress bars during data loading
# export HF_HUB_DISABLE_PROGRESS_BARS=1

# log_info "Environment Configuration:"
# echo "  OMP_NUM_THREADS: ${OMP_NUM_THREADS}"
# echo "  PYTORCH_ALLOC_CONF: ${PYTORCH_ALLOC_CONF}"
# echo "  MALLOC_ARENA_MAX: ${MALLOC_ARENA_MAX}"

# echo ""

# # ============================================================================
# # Step 5: Create Training Runner Script
# # ============================================================================

# log_step "Step 5: Preparing training command"

# TRAIN_RUNNER="${PROJECT_DIR}/train_runner_${TIMESTAMP}.py"

# cat > "$TRAIN_RUNNER" << 'PYTHON_EOF'
# """
# Training runner for PACE PHOENIX
# Sets up hyperparameter grids and distributes via multiprocessing.Pool.starmap
# """

# import os
# import sys
# import csv
# import itertools
# import torch
# import torch.multiprocessing as mp

# # Add project to path
# sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# # Import training function
# from train import train

# # Detect CPU-only mode for PHOENIX
# device_type = "cpu"
# device = torch.device(device_type)

# import os

# if __name__ == '__main__':
#     # Number of worker processes (from environment or auto-detect)
#     num_workers = int(os.environ.get('TRAIN_WORKERS', 8))
    
#     # Hyperparameter grids
#     # During test mode, these can be overridden via environment
#     test_mode = os.environ.get('TEST_MODE', 'false').lower() == 'true'
    
#     if test_mode:
#         print("[TEST MODE] Using limited hyperparameter grid")
#         stiefel_beta1_grid = [0.9]
#         stiefel_beta2_grid = [0.99]
#         stiefel_lr_grid = [1e-4]
#         stiefel_momentum_grid = [0.9]
#         model_scales = [10]
#         batch_size = [2**16]
#         stiefel_type = ['SGD']
#     else:
#         # Full grid - customize as needed
#         stiefel_beta1_grid = [0.9]
#         stiefel_beta2_grid = [0.99]
#         stiefel_lr_grid = [1e-4]
#         stiefel_momentum_grid = [0.9]
#         model_scales = [10]
#         batch_size = [2**16]
#         stiefel_type = ['SGD']
    
#     # Generate hyperparameter combinations
#     hp_list = itertools.product(
#         model_scales,
#         stiefel_lr_grid,
#         stiefel_momentum_grid,
#         stiefel_beta1_grid,
#         stiefel_beta2_grid,
#         batch_size,
#         stiefel_type
#     )
    
#     hp_dict_list = [
#         dict(zip(
#             ['model_scale', 'stiefel_lr', 'stiefel_momentum', 'stiefel_beta1', 
#              'stiefel_beta2', 'total_batch_size', 'stiefel_type'],
#             vals
#         ))
#         for vals in hp_list
#     ]
    
#     print(f"Total hyperparameter configurations: {len(hp_dict_list)}")
#     print(f"Using {num_workers} worker processes")
#     print(f"Configurations: {hp_dict_list}")
#     print()
    
#     # Use spawn context for multiprocessing (more stable for CPU)
#     ctx = mp.get_context('spawn')
    
#     # Distribute configurations across workers using starmap
#     with ctx.Pool(num_workers) as pool:
#         output = pool.starmap(
#             train,
#             [(config, device_type, device) for config in hp_dict_list]
#         )
    
#     # Write results to TSV
#     results_file = 'results.tsv'
#     with open(results_file, 'a', newline='') as f:
#         writer = csv.writer(f, delimiter='\t')
        
#         # Write header if file is empty
#         if f.tell() == 0:
#             writer.writerow([
#                 'model_scale', 'stiefel_type', 'stiefel_lr', 'stiefel_momentum',
#                 'stiefel_beta1', 'stiefel_beta2', 'val_bpb', 'training_seconds',
#                 'total_seconds', 'peak_vram_mb', 'mfu_percent', 'total_tokens_M',
#                 'num_steps', 'num_params_M'
#             ])
        
#         # Write results
#         for result in output:
#             writer.writerow([
#                 result[k] for k in [
#                     'model_scale', 'stiefel_type', 'stiefel_lr', 'stiefel_momentum',
#                     'stiefel_beta1', 'stiefel_beta2', 'val_bpb', 'training_seconds',
#                     'total_seconds', 'peak_vram_mb', 'mfu_percent', 'total_tokens_M',
#                     'num_steps', 'num_params_M'
#                 ]
#             ])
    
#     print(f"Results written to {results_file}")
# PYTHON_EOF

# log_info "Created training runner: $(basename $TRAIN_RUNNER)"

# echo ""

# # ============================================================================
# # Step 6: Execute Training
# # ============================================================================

# log_step "Step 6: Starting distributed training"

# echo "Command: python $TRAIN_RUNNER"
# echo "=========================================="
# echo ""

# export TRAIN_WORKERS=$NUM_WORKERS
# if [ "$TEST_MODE" = true ]; then
#     export TEST_MODE=true
# fi

# python "$TRAIN_RUNNER"
# EXIT_CODE=$?

# echo ""
# echo "=========================================="

# if [ $EXIT_CODE -eq 0 ]; then
#     log_info "Training completed successfully"
# else
#     log_error "Training failed with exit code $EXIT_CODE"
#     exit $EXIT_CODE
# fi

# echo ""

# # ============================================================================
# # Step 7: Summary
# # ============================================================================

# log_step "Training Summary"

# if [ -f results.tsv ]; then
#     log_info "Results saved to: results.tsv"
#     echo "Number of completed runs:"
#     wc -l results.tsv | awk '{print "  " $1}'
#     echo ""
#     echo "Latest results:"
#     tail -3 results.tsv
# else
#     log_warn "No results file found"
# fi

# log_info "Log file: $LOG_FILE"
# log_info "Done!"