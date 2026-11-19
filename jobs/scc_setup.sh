#!/bin/bash -l
# One-time environment setup on SCC (run on a login node from the repo root).
# Weights come from the shared read-only cache (/projectnb/bu-mlb/materials/hf_cache/hub); no download.
set -e
module unload miniconda academic-ml 2>/dev/null || true   # conflicts with the python3 module
module load python3/3.12.4
module load cuda/12.8
python -m venv .venv
source .venv/bin/activate
export PIP_CACHE_DIR=$PWD/.pip_cache      # home is at quota
pip install --upgrade pip
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
mkdir -p logs hf_home
