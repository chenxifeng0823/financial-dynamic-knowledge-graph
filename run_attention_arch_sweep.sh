#!/usr/bin/env bash

# Stage B: Architecture sweep for the Temporal Attention model.
# This sweep keeps optimization hyperparameters fixed (centered around lr=2e-4)
# and varies architectural knobs like embed_dim, num_heads, window_size, and
# num_gconv_layers.
#
# Usage:
#   1) Create a sweep in the WandB UI using the YAML this script writes at:
#        sweeps/attention_architecture.yaml
#      (or let this script try to create one via the API if you have perms).
#   2) Then run:
#        bash run_attention_arch_sweep.sh <entity/project/sweep_id>
#
# If you omit the sweep ID, the script will *attempt* to create a sweep via
# the WandB API using the local YAML, but this may fail with 403 on some setups.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

echo "[ArchSweep] Using repo dir: $REPO_DIR"

##############################
# 1) Load WandB API key
##############################

WANDB_ENV_FILE="$REPO_DIR/wandb.env"
if [ -f "$WANDB_ENV_FILE" ]; then
  echo "[ArchSweep] Loading WANDB_API_KEY from wandb.env"
  WANDB_API_LINE=$(grep -m1 '^WANDB_API_KEY=' "$WANDB_ENV_FILE" || true)
  if [ -n "$WANDB_API_LINE" ]; then
    # shellcheck disable=SC2163
    export "$WANDB_API_LINE"
  else
    echo "[ArchSweep] WARNING: No WANDB_API_KEY= line found in wandb.env"
  fi
else
  echo "[ArchSweep] WARNING: wandb.env not found; ensure WANDB_API_KEY is set."
fi

##############################
# 2) Activate Python env
##############################

if command -v conda &> /dev/null; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  if conda env list | grep -q "^kg_env"; then
    echo "[ArchSweep] Activating conda env: kg_env"
    conda activate kg_env
  else
    echo "[ArchSweep] Conda env 'kg_env' not found; using current Python env."
  fi
elif [ -f "$REPO_DIR/kg_env/bin/activate" ]; then
  echo "[ArchSweep] Activating venv at $REPO_DIR/kg_env"
  # shellcheck disable=SC1091
  source "$REPO_DIR/kg_env/bin/activate"
else
  echo "[ArchSweep] No 'kg_env' environment found; using current Python env."
fi

##############################
# 3) Ensure sweep config exists
##############################

SWEEP_DIR="$REPO_DIR/sweeps"
SWEEP_CONFIG="$SWEEP_DIR/attention_architecture.yaml"

mkdir -p "$SWEEP_DIR"

if [ ! -f "$SWEEP_CONFIG" ]; then
  echo "[ArchSweep] Creating architecture sweep config at $SWEEP_CONFIG"
  cat > "$SWEEP_CONFIG" << 'EOF'
program: main.py
method: bayes
name: attention_model_architecture_sweep
metric:
  name: Val/MRR
  goal: maximize

parameters:
  model_type:
    value: Attention
  data_root:
    value: data/FinDKG

  # Training setup fixed for Stage B (architecture-only sweep)
  epochs:
    value: 20
  lr:
    value: 0.0002
  dropout:
    value: 0.1
  weight_decay:
    value: 0.0001
  grad_clip_norm:
    value: 2.0

  # Architecture knobs
  embed_dim:
    values: [64, 128, 256]
  num_heads:
    values: [4, 8]
  window_size:
    values: [5, 10, 20]
  num_gconv_layers:
    values: [1, 2, 3]

  seed:
    values: [41, 42, 43]
EOF
else
  echo "[ArchSweep] Using existing architecture sweep config at $SWEEP_CONFIG"
fi

##############################
# 4) Create or reuse sweep
##############################

# Default to the same entity/project you used for Stage A sweeps
PROJECT_NAME="kgraph sweepx"
WANDB_ENTITY="${WANDB_ENTITY:-zjduleo-nvidia}"

if [ $# -ge 1 ]; then
  SWEEP_ID="$1"
  echo "[ArchSweep] Using provided sweep ID: $SWEEP_ID"
else
  echo "[ArchSweep] Creating new WandB sweep for entity '$WANDB_ENTITY', project '$PROJECT_NAME'"
  if command -v wandb &> /dev/null; then
    SWEEP_OUTPUT=$(wandb sweep --entity "$WANDB_ENTITY" --project "$PROJECT_NAME" "$SWEEP_CONFIG" || true)
  else
    SWEEP_OUTPUT=$(python -m wandb.cli sweep --entity "$WANDB_ENTITY" --project "$PROJECT_NAME" "$SWEEP_CONFIG" || true)
  fi
  echo "$SWEEP_OUTPUT"
  SWEEP_ID=$(echo "$SWEEP_OUTPUT" | awk '/wandb agent /{print $NF}')
  if [ -z "${SWEEP_ID:-}" ]; then
    echo "[ArchSweep] ERROR: Failed to parse sweep ID from wandb sweep output."
    echo "[ArchSweep] If API creation is not allowed, create the sweep in the WandB UI"
    echo "[ArchSweep] using $SWEEP_CONFIG and then re-run:"
    echo "  bash run_attention_arch_sweep.sh <entity/project/sweep_id>"
    exit 1
  fi
  echo "[ArchSweep] Created sweep ID: $SWEEP_ID"
fi

##############################
# 5) Run sweep agent
##############################

echo "[ArchSweep] Starting WandB agent for sweep: $SWEEP_ID"
if command -v wandb &> /dev/null; then
  wandb agent "$SWEEP_ID"
else
  python -m wandb.cli agent "$SWEEP_ID"
fi


