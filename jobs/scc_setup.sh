#!/bin/bash -l
# One-time environment setup on SCC (run on a login node from the repo root).
set -e
module load python3/3.11 2>/dev/null || module load python3
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
mkdir -p logs
# Pre-download weights on the login node (compute nodes may lack internet):
export HF_HOME=${HF_HOME:-/projectnb/CHANGE_ME_PROJECT/$USER/hf_cache}
python -c "from huggingface_hub import snapshot_download as s; [s(m) for m in ['Qwen/Qwen3-VL-2B-Instruct','Qwen/Qwen3-VL-8B-Instruct']]"
