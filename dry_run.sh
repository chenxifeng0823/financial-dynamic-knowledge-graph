#!/bin/bash
################################################################################
# UNIFIED DRY RUN: Test All 6 Financial Knowledge Graph Models
# Purpose: Validate all models can train without errors
# Features: Detailed logging, error capture, progress tracking
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
mkdir -p dry_run_logs

echo "=================================================================="
echo "DRY RUN: Testing All Financial KG Models"
echo "=================================================================="
echo "Dataset: FinDKG (13,645 entities, 15 relations, 119K triplets)"
echo "Models: 6 (TransE, DistMult, ComplEx, TemporalTransE, RNN, Attention)"
echo "Settings: 1 epoch (quick validation)"
echo "Logs: dry_run_logs/"
echo "=================================================================="
echo ""

# Track results
PASSED=()
FAILED=()
START_TIME=$(date +%s)

################################################################################
# SIMPLE MODELS (embed_dim=50)
################################################################################

SIMPLE_MODELS=("TransE" "DistMult" "ComplEx" "TemporalTransE")

for MODEL in "${SIMPLE_MODELS[@]}"; do
    echo ""
    echo "========================================"
    echo "Testing: $MODEL (embed_dim=50)"
    echo "========================================"
    
    LOG_FILE="dry_run_logs/${MODEL}.log"
    
    # Run with unbuffered output and save to log
    python -u main.py \
        --model_type "$MODEL" \
        --data_root data/FinDKG \
        --epochs 1 \
        --embed_dim 50 \
        --lr 0.001 \
        --seed 42 \
        2>&1 | tee "$LOG_FILE"
    
    EXIT_CODE=${PIPESTATUS[0]}
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo "✅ $MODEL: SUCCESS"
        PASSED+=("$MODEL")
    else
        echo "❌ $MODEL: FAILED (exit code: $EXIT_CODE)"
        echo "   Check $LOG_FILE for details"
        FAILED+=("$MODEL")
    fi
    
    # Clean up wandb runs to save space (keep only final run)
    rm -rf wandb/run-* 2>/dev/null
    
    echo ""
done

################################################################################
# DEEP TEMPORAL MODELS (embed_dim=64, divisible by num_heads=4)
################################################################################

DEEP_MODELS=("RNN" "Attention")

for MODEL in "${DEEP_MODELS[@]}"; do
    echo ""
    echo "========================================"
    echo "Testing: $MODEL (embed_dim=64)"
    echo "========================================"
    
    LOG_FILE="dry_run_logs/${MODEL}.log"
    
    # Run with unbuffered output and save to log
    python -u main.py \
        --model_type "$MODEL" \
        --data_root data/FinDKG \
        --epochs 1 \
        --embed_dim 64 \
        --lr 0.0005 \
        --seed 42 \
        2>&1 | tee "$LOG_FILE"
    
    EXIT_CODE=${PIPESTATUS[0]}
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo "✅ $MODEL: SUCCESS"
        PASSED+=("$MODEL")
    else
        echo "❌ $MODEL: FAILED (exit code: $EXIT_CODE)"
        echo "   Check $LOG_FILE for details"
        FAILED+=("$MODEL")
    fi
    
    # Clean up wandb runs to save space
    rm -rf wandb/run-* 2>/dev/null
    
    echo ""
done

################################################################################
# SUMMARY
################################################################################

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo ""
echo "=================================================================="
echo "DRY RUN COMPLETE"
echo "=================================================================="
echo "Duration: ${DURATION}s"
echo ""
echo "RESULTS:"
echo "--------"
echo "✅ Passed (${#PASSED[@]}/6): ${PASSED[*]}"
echo "❌ Failed (${#FAILED[@]}/6): ${FAILED[*]}"
echo ""
echo "Logs saved to: dry_run_logs/"
echo "=================================================================="

# Exit with error if any model failed
if [ ${#FAILED[@]} -gt 0 ]; then
    exit 1
else
    exit 0
fi

