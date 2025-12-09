#!/bin/bash
################################################################################
# FULL TRAINING: Train All 6 Financial Knowledge Graph Models
# 
# Training Strategy:
# - Train on full training data (multiple epochs)
# - Use validation data to select best model (early stopping via best MRR)
# - Evaluate final model on test data
# - Track all experiments with Weights & Biases
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

# Create timestamped log file
TIMESTAMP=$(date +%Y-%m-%d_%H%M%S)
LOG_FILE="logs/full_training_${TIMESTAMP}.log"

echo "=================================================================="
echo "FULL TRAINING: All Financial KG Models"
echo "=================================================================="
echo "Dataset: FinDKG (13,645 entities, 15 relations, 119K triplets)"
echo "Models: 6 (TransE, DistMult, ComplEx, TemporalTransE, RNN, Attention)"
echo "Strategy: Train → Validate → Test"
echo "Log file: $LOG_FILE"
echo "=================================================================="
echo ""

# Write header to log file
{
    echo "=================================================================="
    echo "FULL TRAINING: All Financial KG Models"
    echo "=================================================================="
    echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "Dataset: FinDKG (13,645 entities, 15 relations, 119K triplets)"
    echo "Strategy:"
    echo "  1. Train on training data with multiple epochs"
    echo "  2. Validate after each epoch, save best model by Val MRR"
    echo "  3. Load best model and evaluate on test data"
    echo "=================================================================="
    echo ""
} > "$LOG_FILE"

# Track results
PASSED=()
FAILED=()
START_TIME=$(date +%s)

################################################################################
# HYPERPARAMETERS
################################################################################

# Simple models: Higher learning rate, moderate epochs
SIMPLE_EPOCHS=50
SIMPLE_LR=0.001
SIMPLE_EMBED_DIM=128

# Deep models: Lower learning rate, more epochs
DEEP_EPOCHS=50
DEEP_LR=0.0005
DEEP_EMBED_DIM=128

SEED=33

################################################################################
# SIMPLE MODELS (TransE, DistMult, ComplEx, TemporalTransE)
################################################################################

SIMPLE_MODELS=("TransE" "DistMult" "ComplEx" "TemporalTransE")

for MODEL in "${SIMPLE_MODELS[@]}"; do
    {
        echo ""
        echo "=================================================================="
        echo "Training: $MODEL"
        echo "=================================================================="
        echo "Hyperparameters:"
        echo "  - Epochs: $SIMPLE_EPOCHS"
        echo "  - Embedding dim: $SIMPLE_EMBED_DIM"
        echo "  - Learning rate: $SIMPLE_LR"
        echo "  - Seed: $SEED"
        echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
        echo ""
    } | tee -a "$LOG_FILE"
    
    # Run training
    python -u main.py \
        --model_type "$MODEL" \
        --data_root data/FinDKG \
        --epochs $SIMPLE_EPOCHS \
        --embed_dim $SIMPLE_EMBED_DIM \
        --lr $SIMPLE_LR \
        --seed $SEED \
        2>&1 | tee -a "$LOG_FILE"
    
    EXIT_CODE=${PIPESTATUS[0]}
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo "✅ $MODEL: SUCCESS" | tee -a "$LOG_FILE"
        PASSED+=("$MODEL")
    else
        echo "❌ $MODEL: FAILED (exit code: $EXIT_CODE)" | tee -a "$LOG_FILE"
        FAILED+=("$MODEL")
    fi
    
    # Clean up wandb runs to save space
    rm -rf wandb/run-* 2>/dev/null
    
    {
        echo "Finished: $(date '+%Y-%m-%d %H:%M:%S')"
        echo ""
    } | tee -a "$LOG_FILE"
done

################################################################################
# DEEP TEMPORAL MODELS (RNN, Attention)
################################################################################

DEEP_MODELS=("RNN" "Attention")

for MODEL in "${DEEP_MODELS[@]}"; do
    {
        echo ""
        echo "=================================================================="
        echo "Training: $MODEL"
        echo "=================================================================="
        echo "Hyperparameters:"
        echo "  - Epochs: $DEEP_EPOCHS"
        echo "  - Embedding dim: $DEEP_EMBED_DIM (divisible by num_heads=4)"
        echo "  - Learning rate: $DEEP_LR"
        echo "  - Seed: $SEED"
        echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
        echo ""
    } | tee -a "$LOG_FILE"
    
    # Run training
    python -u main.py \
        --model_type "$MODEL" \
        --data_root data/FinDKG \
        --epochs $DEEP_EPOCHS \
        --embed_dim $DEEP_EMBED_DIM \
        --lr $DEEP_LR \
        --seed $SEED \
        2>&1 | tee -a "$LOG_FILE"
    
    EXIT_CODE=${PIPESTATUS[0]}
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo "✅ $MODEL: SUCCESS" | tee -a "$LOG_FILE"
        PASSED+=("$MODEL")
    else
        echo "❌ $MODEL: FAILED (exit code: $EXIT_CODE)" | tee -a "$LOG_FILE"
        FAILED+=("$MODEL")
    fi
    
    # Clean up wandb runs to save space
    rm -rf wandb/run-* 2>/dev/null
    
    {
        echo "Finished: $(date '+%Y-%m-%d %H:%M:%S')"
        echo ""
    } | tee -a "$LOG_FILE"
done

################################################################################
# SUMMARY
################################################################################

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

{
    echo ""
    echo "=================================================================="
    echo "FULL TRAINING COMPLETE"
    echo "=================================================================="
    echo "Finished: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "Total Duration: ${HOURS}h ${MINUTES}m ${SECONDS}s"
    echo ""
    echo "RESULTS:"
    echo "--------"
    echo "✅ Passed (${#PASSED[@]}/6): ${PASSED[*]}"
    echo "❌ Failed (${#FAILED[@]}/6): ${FAILED[*]}"
    echo ""
    echo "Training Details:"
    echo "  - Simple models (TransE, DistMult, ComplEx, TemporalTransE):"
    echo "    * Epochs: $SIMPLE_EPOCHS, LR: $SIMPLE_LR, Embed: $SIMPLE_EMBED_DIM"
    echo "  - Deep models (RNN, Attention):"
    echo "    * Epochs: $DEEP_EPOCHS, LR: $DEEP_LR, Embed: $DEEP_EMBED_DIM"
    echo ""
    echo "Each model:"
    echo "  1. Trained on full training set"
    echo "  2. Validated after each epoch (saved best by Val MRR)"
    echo "  3. Loaded best model and evaluated on test set"
    echo ""
    echo "View results in Weights & Biases:"
    echo "  https://wandb.ai/zjduleo-nvidia/findkg-model-comparison"
    echo ""
    echo "Complete log saved to: $LOG_FILE"
    echo "=================================================================="
} | tee -a "$LOG_FILE"

# Exit with error if any model failed
if [ ${#FAILED[@]} -gt 0 ]; then
    exit 1
else
    exit 0
fi
