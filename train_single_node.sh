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