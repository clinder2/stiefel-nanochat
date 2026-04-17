#!/bin/bash
###############################################################################
# Single-Node Multi-Core Training Script (64 Cores)
#
# Optimized for single node with 64 CPU cores
# - Runs prepare.py once
# - Configures optimal threading
# - Launches train.py with multiprocessing pool
# - CPU affinity pinning for best performance
#
# Usage:
#   ./train_single_node.sh
#   ./train_single_node.sh --num-workers 16  # Override worker count
#   ./train_single_node.sh --debug            # Enable debug output
#
###############################################################################

set -e

# ============================================================================
# Configuration
# ============================================================================

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${PROJECT_DIR}/.venv"
LOG_DIR="${PROJECT_DIR}/logs"
mkdir -p "${LOG_DIR}"

# Single node setup
TOTAL_CORES=$(nproc)
NUM_WORKERS="${NUM_WORKERS:-1}"           # Multiprocessing pool size (train.py default)
THREADS_PER_WORKER=$((TOTAL_CORES / NUM_WORKERS))

# Logging
LOG_FILE="${LOG_DIR}/training_$(date +%Y%m%d_%H%M%S).log"
exec 1> >(tee -a "${LOG_FILE}")
exec 2>&1

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

# Parse arguments
DEBUG=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --num-workers)
            NUM_WORKERS="$2"
            shift 2
            ;;
        --threads-per-worker)
            THREADS_PER_WORKER="$2"
            shift 2
            ;;
        --debug)
            DEBUG=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# ============================================================================
# Helper Functions
# ============================================================================

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

cleanup() {
    log_info "Cleaning up..."
    pkill -f "python.*train.py" 2>/dev/null || true
    pkill -f "python.*prepare.py" 2>/dev/null || true
}

trap cleanup EXIT

# ============================================================================
# Step 1: System Information
# ============================================================================

log_step "Step 1: Detecting system configuration"

echo ""
log_info "System Information:"
echo "  Total CPU Cores: ${TOTAL_CORES}"

# Check available memory
if command -v free &> /dev/null; then
    TOTAL_MEM_GB=$(free -g | awk 'NR==2{printf "%.0f", $2}')
    AVAILABLE_MEM_GB=$(free -g | awk 'NR==2{printf "%.0f", $7}')
    log_info "Total Memory: ${TOTAL_MEM_GB}GB"
    log_info "Available Memory: ${AVAILABLE_MEM_GB}GB"
    
    # Warn if memory might be insufficient for large models
    if [ $TOTAL_MEM_GB -lt 32 ]; then
        log_warn "Low memory detected (${TOTAL_MEM_GB}GB). Large models may fail."
        log_info "Consider reducing model scale or batch size in train.py"
    fi
elif command -v vm_stat &> /dev/null; then
    # macOS memory check
    VM_STAT=$(vm_stat)
    PAGES_FREE=$(echo "$VM_STAT" | awk '/Pages free/ {print $3}' | tr -d '.')
    PAGES_ACTIVE=$(echo "$VM_STAT" | awk '/Pages active/ {print $3}' | tr -d '.')
    PAGE_SIZE=$(echo "$VM_STAT" | awk '/page size of/ {print $8}' | tr -d 'bytes:')
    
    FREE_MEM_MB=$((PAGES_FREE * PAGE_SIZE / 1024 / 1024))
    ACTIVE_MEM_MB=$((PAGES_ACTIVE * PAGE_SIZE / 1024 / 1024))
    
    log_info "Free Memory: ${FREE_MEM_MB}MB"
    log_info "Active Memory: ${ACTIVE_MEM_MB}MB"
fi

echo "  Multiprocessing Pool Workers: ${NUM_WORKERS}"
echo "  Threads per Worker: ${THREADS_PER_WORKER}"
echo "  Total Compute: ${NUM_WORKERS} workers × ${THREADS_PER_WORKER} threads = ${TOTAL_CORES} cores"

# Recalculate if needed
if [ $((NUM_WORKERS * THREADS_PER_WORKER)) -gt $TOTAL_CORES ]; then
    log_warn "Oversubscription detected! Reducing threads per worker."
    THREADS_PER_WORKER=$((TOTAL_CORES / NUM_WORKERS))
    log_info "Adjusted threads per worker to: ${THREADS_PER_WORKER}"
fi

# Check for useful tools
if command -v taskset &> /dev/null; then
    HAS_TASKSET=true
    log_info "taskset available: CPU affinity will be used"
else
    HAS_TASKSET=false
    log_warn "taskset not available: CPU affinity disabled"
fi

if command -v numactl &> /dev/null; then
    HAS_NUMACTL=true
    log_info "numactl available: NUMA-aware execution will be used"
else
    HAS_NUMACTL=false
    log_info "numactl not available: standard execution"
fi

echo ""

# ============================================================================
# Step 2: Setup Python Environment
# ============================================================================

log_step "Step 2: Setting up Python environment"

if [ ! -d "$VENV_DIR" ]; then
    log_info "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

log_info "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

log_info "Upgrading pip..."
#pip install --quiet --upgrade pip setuptools wheel 2>/dev/null
uv sync
# log_info "Installing dependencies..."
# if [ -f "${PROJECT_DIR}/pyproject.toml" ]; then
#     pip install --quiet -e "${PROJECT_DIR}" 2>/dev/null
# else
#     pip install --quiet torch networkx 2>/dev/null
# fi

# Try to install jemalloc if not available (helps with memory issues)
if ! command -v jemalloc-config &> /dev/null; then
    log_info "Attempting to install jemalloc for better memory allocation..."
    if command -v apt-get &> /dev/null; then
        sudo apt-get update -qq && sudo apt-get install -qq -y libjemalloc-dev 2>/dev/null || log_warn "Could not install jemalloc via apt-get"
    elif command -v yum &> /dev/null; then
        sudo yum install -q -y jemalloc-devel 2>/dev/null || log_warn "Could not install jemalloc via yum"
    elif command -v brew &> /dev/null; then
        brew install jemalloc 2>/dev/null || log_warn "Could not install jemalloc via brew"
    fi
fi

log_info "Environment ready"
echo ""

# ============================================================================
# Step 3: Data Preparation
# ============================================================================

log_step "Step 3: Data preparation (prepare.py)"

if [ ! -f "${PROJECT_DIR}/data.bin" ] && [ ! -d "${PROJECT_DIR}/data" ]; then
    log_info "Running prepare.py..."
    python "${PROJECT_DIR}/prepare.py" 2>&1 | tee -a "${LOG_FILE}"
    
    if [ $? -eq 0 ]; then
        log_info "Data preparation successful"
    else
        log_error "Data preparation failed!"
        exit 1
    fi
else
    log_info "Data already prepared, skipping prepare.py"
fi

echo ""

# ============================================================================
# Step 4: Configure Threading Environment
# ============================================================================

log_step "Step 4: Configuring threading environment"

# Calculate optimal threading based on cores
# Prevent oversubscription: keep some cores free for system
RESERVED_CORES=4
AVAILABLE_CORES=$((TOTAL_CORES - RESERVED_CORES))
OPTIMIZED_THREADS=$((AVAILABLE_CORES / NUM_WORKERS))

if [ $OPTIMIZED_THREADS -lt 1 ]; then
    OPTIMIZED_THREADS=1
fi

log_info "Threading Configuration:"
echo "  Reserved for system: ${RESERVED_CORES} cores"
echo "  Available cores: ${AVAILABLE_CORES}"
echo "  Optimized threads per worker: ${OPTIMIZED_THREADS}"

# ============================================================================
# Step 4.5: Memory Configuration (Critical for CPU Training)
# ============================================================================

log_step "Step 4.5: Configuring memory allocation"

# Set memory limits to prevent glibc malloc errors
ulimit -v unlimited 2>/dev/null || log_warn "Could not set unlimited virtual memory"
ulimit -m unlimited 2>/dev/null || log_warn "Could not set unlimited memory"

# PyTorch CPU memory settings
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0  # Disable MPS memory limits
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512  # For CUDA, but doesn't hurt CPU

# Memory allocator settings to prevent glibc malloc issues
export MALLOC_ARENA_MAX=2  # Limit glibc malloc arenas to prevent fragmentation
export MALLOC_TRIM_THRESHOLD_=100000
export MALLOC_MMAP_THRESHOLD_=1000000
export MALLOC_MMAP_MAX_=65536

# Alternative: Use jemalloc if available (better for PyTorch CPU)
if command -v jemalloc-config &> /dev/null; then
    log_info "Using jemalloc for better memory allocation"
    export LD_PRELOAD=$(jemalloc-config --libdir)/libjemalloc.so.$(jemalloc-config --revision)
    export MALLOC_CONF="background_thread:true,metadata_thp:auto,dirty_decay_ms:30000,muzzy_decay_ms:30000"
elif command -v tcmalloc &> /dev/null; then
    log_info "Using tcmalloc for better memory allocation"
    export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4 2>/dev/null || \
    export LD_PRELOAD=/usr/lib/libtcmalloc.so.4 2>/dev/null || \
    log_warn "tcmalloc found but could not set LD_PRELOAD"
else
    log_warn "Neither jemalloc nor tcmalloc found. Using system malloc (may cause issues)"
    log_info "To fix: sudo apt-get install libjemalloc-dev  # Ubuntu/Debian"
    log_info "Or: sudo yum install jemalloc-devel          # CentOS/RHEL"
fi

# Additional memory settings for large models
export OMP_PROC_BIND=close  # Bind threads to cores for better memory locality
export GOMP_CPU_AFFINITY="0-$((TOTAL_CORES-1))"  # GCC OpenMP CPU affinity

log_info "Memory Configuration:"
echo "  MALLOC_ARENA_MAX: ${MALLOC_ARENA_MAX:-default}"
echo "  MALLOC_TRIM_THRESHOLD_: ${MALLOC_TRIM_THRESHOLD_}"
echo "  MALLOC_MMAP_THRESHOLD_: ${MALLOC_MMAP_THRESHOLD_}"
echo "  LD_PRELOAD: ${LD_PRELOAD:-none}"

# Set threading environment variables
export OMP_NUM_THREADS=${OPTIMIZED_THREADS}
export OPENBLAS_NUM_THREADS=${OPTIMIZED_THREADS}
export MKL_NUM_THREADS=${OPTIMIZED_THREADS}
export TORCH_NUM_THREADS=${OPTIMIZED_THREADS}
export NUMEXPR_NUM_THREADS=${OPTIMIZED_THREADS}
export VECLIB_MAXIMUM_THREADS=${OPTIMIZED_THREADS}

# Multiprocessing configuration
export TORCH_NUM_WORKERS=0          # Disable DataLoader workers (use multiprocessing instead)
export PYTHONUNBUFFERED=1           # Unbuffered output for real-time logging

# Debug mode
if [ "$DEBUG" = true ]; then
    export TORCH_CPP_LOG_LEVEL=INFO
    export TORCH_DISTRIBUTED_DEBUG=INFO
    export PYTHONVERBOSE=1
    log_info "Debug mode enabled"
fi

echo ""

# ============================================================================
# Step 5: Prepare Launch Command
# ============================================================================

log_step "Step 5: Launching training"

log_info "Launch Configuration:"
echo "  Script: ${PROJECT_DIR}/train.py"
echo "  Workers: ${NUM_WORKERS}"
echo "  Threads per worker: ${OPTIMIZED_THREADS}"
echo "  Log file: ${LOG_FILE}"

# Build command based on available tools
TRAIN_CMD="python ${PROJECT_DIR}/train.py"

if [ "$HAS_NUMACTL" = true ] && [ $TOTAL_CORES -gt 32 ]; then
    # Use numactl for NUMA-aware execution on systems with >32 cores
    log_info "Using numactl for NUMA-aware execution"
    TRAIN_CMD="numactl --interleave=all ${TRAIN_CMD}"
elif [ "$HAS_TASKSET" = true ]; then
    # Use taskset for CPU affinity to prevent context switching
    CORE_AFFINITY="0-$((TOTAL_CORES-1))"
    log_info "Using CPU affinity: cores ${CORE_AFFINITY}"
    TRAIN_CMD="taskset -c ${CORE_AFFINITY} ${TRAIN_CMD}"
fi
echo ${TRAIN_CMD}
echo ""
log_info "Executing: ${TRAIN_CMD}"
echo "=========================================="
echo ""

# ============================================================================
# Step 6: Execute Training
# ============================================================================

# Run the training command
eval ${TRAIN_CMD}
TRAIN_EXIT_CODE=$?

echo ""
echo "=========================================="

# ============================================================================
# Finalization
# ============================================================================

if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    log_step "Training completed successfully!"
    echo ""
    log_info "Results saved to: ${LOG_FILE}"
    log_info "Check logs/node_0.log or logs/training_*.log for details"
else
    log_error "Training failed with exit code ${TRAIN_EXIT_CODE}"
    echo ""
    log_info "Logs saved to: ${LOG_FILE}"
    exit ${TRAIN_EXIT_CODE}
fi

echo ""
log_info "Deactivating virtual environment..."
deactivate

exit 0