#!/bin/bash
################################################################################
# OFFICIAL RUN: 50 Epochs for All Models
################################################################################

cd /lustre/fsw/portfolios/general/users/ldu/financial-dynamic-knowledge-graph

# Setup environment
source kg_env/bin/activate
export DGLBACKEND=pytorch

# Load wandb API key
if [ -f wandb.env ]; then
    export $(cat wandb.env | xargs)
fi

# Create logs directory
mkdir -p logs

TIMESTAMP=$(date +%Y-%m-%d_%H%M%S)
LOG_FILE="logs/official_run_50epochs_${TIMESTAMP}.log"

echo "=================================================================="
echo "OFFICIAL RUN: 50 Epochs"
echo "=================================================================="
echo "Dataset: FinDKG"
echo "Epochs: 50"
echo "Log file: $LOG_FILE"
echo "=================================================================="
echo ""

# Write header to log file
{
    echo "=================================================================="
    echo "OFFICIAL RUN: 50 Epochs"
    echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=================================================================="
    echo ""
} > "$LOG_FILE"

EPOCHS=50
SEED=100

################################################################################
# GROUP 1: BASELINES (Simple Models)
################################################################################
SIMPLE_MODELS=("TransE" "DistMult" "ComplEx" "TemporalTransE")
SIMPLE_EMBED_DIM=128
SIMPLE_LR=0.001

for MODEL in "${SIMPLE_MODELS[@]}"; do
    {
        echo ""
        echo "=================================================================="
        echo "Training: $MODEL (Group: Baselines)"
        echo "=================================================================="
        echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
        echo ""
    } | tee -a "$LOG_FILE"
    
    python -u main.py \
        --model_type "$MODEL" \
        --data_root data/FinDKG \
        --epochs $EPOCHS \
        --embed_dim $SIMPLE_EMBED_DIM \
        --lr $SIMPLE_LR \
        --seed $SEED \
        2>&1 | tee -a "$LOG_FILE"
        
    # Clean up wandb runs
    rm -rf wandb/run-* 2>/dev/null
done

################################################################################
# GROUP 2: DEEP TEMPORAL (Deep Models)
################################################################################
DEEP_MODELS=("RNN" "Attention")
DEEP_EMBED_DIM=128
DEEP_LR=0.00005  # Low LR for stability (especially Attention)

for MODEL in "${DEEP_MODELS[@]}"; do
    {
        echo ""
        echo "=================================================================="
        echo "Training: $MODEL (Group: Deep_Temporal)"
        echo "=================================================================="
        echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
        echo ""
    } | tee -a "$LOG_FILE"
    
    python -u main.py \
        --model_type "$MODEL" \
        --data_root data/FinDKG \
        --epochs $EPOCHS \
        --embed_dim $DEEP_EMBED_DIM \
        --lr $DEEP_LR \
        --seed $SEED \
        2>&1 | tee -a "$LOG_FILE"
        
    # Clean up wandb runs
    rm -rf wandb/run-* 2>/dev/null
done

echo ""
echo "=================================================================="
echo "OFFICIAL RUN COMPLETE"
echo "=================================================================="

