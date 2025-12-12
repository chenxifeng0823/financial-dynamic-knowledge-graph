#!/usr/bin/env bash

# Sanity check run: train all models for 2 epochs to verify that
# the full pipeline executes without errors (including evaluation).
#
# Usage:
#   source kg_env/bin/activate   # if not already active
#   bash run_sanity_2epochs.sh

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

echo "[Sanity] Using repo dir: $REPO_DIR"

if [ -f "$REPO_DIR/kg_env/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$REPO_DIR/kg_env/bin/activate"
  echo "[Sanity] Activated venv at $REPO_DIR/kg_env"
else
  echo "[Sanity] WARNING: kg_env not found, using current Python environment."
fi

# Reduce CUDA fragmentation risk
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

EPOCHS=2

##############################
# 1) Simple baselines (ComplEx excluded to avoid resource issues)
##############################

SIMPLE_MODELS=("TransE" "DistMult" "TemporalTransE")

for MODEL in "${SIMPLE_MODELS[@]}"; do
  echo "[Sanity] Running baseline model: $MODEL for $EPOCHS epochs"
  python main.py \
    --model_type "$MODEL" \
    --data_root data/FinDKG \
    --epochs "$EPOCHS" \
    --lr 0.001 \
    --weight_decay 0.0 \
    --dropout 0.0 \
    --grad_clip_norm 0.0 \
    --embed_dim 128
done

##############################
# 2) Deep temporal models
##############################

echo "[Sanity] Running deep temporal model: RNN for $EPOCHS epochs"
python main.py \
  --model_type RNN \
  --data_root data/FinDKG \
  --epochs "$EPOCHS" \
  --lr 0.00005 \
  --weight_decay 0.0001 \
  --dropout 0.1 \
  --grad_clip_norm 2.0 \
  --embed_dim 128

echo "[Sanity] Running deep temporal model: Attention for $EPOCHS epochs"
python main.py \
  --model_type Attention \
  --data_root data/FinDKG \
  --epochs "$EPOCHS" \
  --lr 0.0002 \
  --weight_decay 0.0001 \
  --dropout 0.1 \
  --grad_clip_norm 2.0 \
  --embed_dim 256 \
  --num_heads 8 \
  --num_gconv_layers 2 \
  --window_size 10


