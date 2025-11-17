#!/bin/bash
# Full training run for KGTransformer PyG (150 epochs, no early stopping)

echo "========================================="
echo "KGTransformer PyG - Full Training"
echo "========================================="
echo "Configuration:"
echo "  Epochs: 150"
echo "  Learning rate: 0.0005"
echo "  Early stopping: Disabled"
echo "  Device: CUDA"
echo "  Seed: 42"
echo "========================================="
echo ""
echo "Starting training at: $(date)"
echo ""

python train_kgt_pyg.py \
    --device cuda \
    --epochs 150 \
    --lr 0.0005 \
    --seed 42 \
    --save_model \
    --eval_rankings \
    --no_early_stop

echo ""
echo "Training completed at: $(date)"
echo "========================================="
