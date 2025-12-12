#!/usr/bin/env bash

# Stage A hyperparameter sweep for the Temporal Attention model.
# Usage:
#   bash run_attention_stageA_sweep.sh
#     - or -
#   WANDB_ENTITY=<your_entity> bash run_attention_stageA_sweep.sh
#
# Optional: pass an existing sweep ID if you've already created one:
#   bash run_attention_stageA_sweep.sh your_entity/findkg-fix-rnn/abcd1234

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

echo "[StageA] Using repo dir: $REPO_DIR"

##############################
# 1) Load WandB API key (from wandb.env if present)
##############################

WANDB_ENV_FILE="$REPO_DIR/wandb.env"
if [ -f "$WANDB_ENV_FILE" ]; then
  echo "[StageA] Loading WANDB_API_KEY from wandb.env"
  # Only read the line that defines WANDB_API_KEY to avoid parsing errors
  WANDB_API_LINE=$(grep -m1 '^WANDB_API_KEY=' "$WANDB_ENV_FILE" || true)
  if [ -n "$WANDB_API_LINE" ]; then
    # shellcheck disable=SC2163
    export "$WANDB_API_LINE"
  else
    echo "[StageA] WARNING: No WANDB_API_KEY= line found in wandb.env"
  fi
else
  echo "[StageA] WARNING: wandb.env not found; ensure WANDB_API_KEY is set in your environment."
fi

##############################
# 2) Activate Python env
##############################

if command -v conda &> /dev/null; then
  # Initialize conda in this shell
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  if conda env list | grep -q "^kg_env"; then
    echo "[StageA] Activating conda env: kg_env"
    conda activate kg_env
  else
    echo "[StageA] Conda env 'kg_env' not found; using current Python env."
  fi
elif [ -f "$HOME/kg_env/bin/activate" ]; then
  echo "[StageA] Activating venv at \$HOME/kg_env"
  # shellcheck disable=SC1091
  source "$HOME/kg_env/bin/activate"
else
  echo "[StageA] No 'kg_env' environment found; using current Python env."
fi

##############################
# 3) Ensure sweep config exists
##############################

SWEEP_DIR="$REPO_DIR/sweeps"
SWEEP_CONFIG="$SWEEP_DIR/attention_stageA.yaml"

mkdir -p "$SWEEP_DIR"

if [ ! -f "$SWEEP_CONFIG" ]; then
  echo "[StageA] Creating sweep config at $SWEEP_CONFIG"
  cat > "$SWEEP_CONFIG" << 'EOF'
program: main.py
method: bayes
name: attention_stageA_temporal_transformer
metric:
  name: Val/MRR
  goal: maximize

parameters:
  model_type:
    value: Attention
  data_root:
    value: data/FinDKG
  epochs:
    value: 20

  # Learning rate: discrete log-spaced values in a sensible range
  lr:
    values: [2e-5, 5e-5, 1e-4, 2e-4, 3e-4]

  weight_decay:
    values: [0.0, 1e-5, 1e-4, 5e-4]

  dropout:
    values: [0.0, 0.1, 0.3]

  grad_clip_norm:
    values: [0.5, 1.0, 2.0, 5.0]

  window_size:
    values: [5, 10, 20]

  num_heads:
    values: [4]

  num_gconv_layers:
    values: [1, 2, 3]

  seed:
    values: [100]
EOF
else
  echo "[StageA] Using existing sweep config at $SWEEP_CONFIG"
fi

##############################
# 4) Create or reuse sweep
##############################

PROJECT_NAME="findkg-fix-rnn"
WANDB_ENTITY="${WANDB_ENTITY:-ldu}"  # change if your WandB entity is different

if [ $# -ge 1 ]; then
  SWEEP_ID="$1"
  echo "[StageA] Using provided sweep ID: $SWEEP_ID"
else
  echo "[StageA] Creating new WandB sweep for entity '$WANDB_ENTITY', project '$PROJECT_NAME'"
  # Prefer the CLI if available; otherwise, fall back to python -m wandb.cli
  # It prints a line like:
  #   wandb: Run sweep agent with: wandb agent <entity>/<project>/<sweep_id>
  if command -v wandb &> /dev/null; then
    SWEEP_OUTPUT=$(wandb sweep --entity "$WANDB_ENTITY" --project "$PROJECT_NAME" "$SWEEP_CONFIG")
  else
    SWEEP_OUTPUT=$(python -m wandb.cli sweep --entity "$WANDB_ENTITY" --project "$PROJECT_NAME" "$SWEEP_CONFIG")
  fi
  echo "$SWEEP_OUTPUT"
  SWEEP_ID=$(echo "$SWEEP_OUTPUT" | awk '/wandb agent /{print $NF}')
  if [ -z "${SWEEP_ID:-}" ]; then
    echo "[StageA] ERROR: Failed to parse sweep ID from wandb sweep output."
    exit 1
  fi
  echo "[StageA] Created sweep ID: $SWEEP_ID"
fi

##############################
# 5) Run sweep agent
##############################

echo "[StageA] Starting WandB agent for sweep: $SWEEP_ID"
if command -v wandb &> /dev/null; then
  wandb agent "$SWEEP_ID"
else
  python -m wandb.cli agent "$SWEEP_ID"
fi


