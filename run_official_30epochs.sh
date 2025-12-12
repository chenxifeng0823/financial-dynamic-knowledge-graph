#!/usr/bin/env bash

# Official submission run: train all models for 30 epochs with best-known configs.
# This script assumes you are already on a GPU node (e.g., inside srun) and that
# the repo root is the current directory.
#
# Usage:
#   bash run_official_30epochs.sh

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

echo "[Official] Using repo dir: $REPO_DIR"

# Activate env if available
if [ -f "$REPO_DIR/kg_env/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$REPO_DIR/kg_env/bin/activate"
  echo "[Official] Activated venv at $REPO_DIR/kg_env"
else
  echo "[Official] WARNING: kg_env not found, using current Python environment."
fi

# Reduce risk of CUDA memory fragmentation (recommended by PyTorch OOM message)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Group 1: Simple baselines (ComplEx excluded to avoid resource issues)
SIMPLE_MODELS=("TransE" "DistMult" "TemporalTransE")

for MODEL in "${SIMPLE_MODELS[@]}"; do
  echo "[Official] Running baseline model: $MODEL"
  python main.py \
    --model_type "$MODEL" \
    --data_root data/FinDKG \
    --epochs 30 \
    --lr 0.001 \
    --weight_decay 0.0 \
    --dropout 0.0 \
    --grad_clip_norm 0.0 \
    --embed_dim 128 \
    --submission_run
done

# Group 2: Deep temporal models
DEEP_MODELS=("RNN" "Attention")

for MODEL in "${DEEP_MODELS[@]}"; do
  echo "[Official] Running deep temporal model: $MODEL"
  if [ "$MODEL" = "Attention" ]; then
    # Best Attention configuration from sweeps
    python main.py \
      --model_type Attention \
      --data_root data/FinDKG \
      --epochs 30 \
      --lr 0.0002 \
      --weight_decay 0.0001 \
      --dropout 0.1 \
      --grad_clip_norm 2.0 \
      --embed_dim 256 \
      --num_heads 8 \
      --num_gconv_layers 2 \
      --window_size 10 \
      --submission_run
  else
    # RNN baseline configuration (similar to earlier official runs)
    python main.py \
      --model_type RNN \
      --data_root data/FinDKG \
      --epochs 30 \
      --lr 0.00005 \
      --weight_decay 0.0001 \
      --dropout 0.1 \
      --grad_clip_norm 2.0 \
      --embed_dim 128 \
      --submission_run
  fi
done


