# Evaluation Metrics

This directory contains evaluation metrics for knowledge graph embedding models, including both standard KG metrics and temporal-specific metrics from the FinDKG paper.

## Modules

### `metrics.py`
Standard knowledge graph evaluation metrics for static models:
- **MRR (Mean Reciprocal Rank)**: Average of 1/rank for all test triplets
- **Hits@K**: Proportion of correct entities ranked in top K
- **Mean Rank**: Average rank of correct entities
- **Median Rank**: Median rank of correct entities

Supports both **raw** and **filtered** evaluation settings.

### `temporal_metrics.py`
Temporal knowledge graph evaluation metrics from the original FinDKG paper:

#### Link Prediction Metrics
- **MRR (Mean Reciprocal Rank)**: Primary metric for link prediction
- **Recall@K (Hits@K)**: Proportion of correct predictions in top K
  - K = 1, 3, 10, 100 (as used in FinDKG paper)
- **Mean Rank (MR)**: Average rank of true entities
- **Median Rank**: Median rank of true entities

#### Time Prediction Metrics
- **MAE (Mean Absolute Error)**: Average absolute difference between predicted and true event times
- **RMSE (Root Mean Squared Error)**: Square root of average squared differences
- **MSE (Mean Squared Error)**: Average squared differences

## Usage

### For Static Models (TransE, DistMult, ComplEx, TemporalTransE)

```python
from src.evaluation import evaluate_model

metrics = evaluate_model(
    model=model,
    test_data=test_triplets,
    num_entities=num_entities,
    batch_size=100,
    filtered=True,  # Use filtered evaluation
    all_triplets=all_triplets  # Required for filtered evaluation
)

print(f"MRR: {metrics['mrr']:.4f}")
print(f"Hits@10: {metrics['hits@10']:.4f}")
```

### For Temporal Models (KGTransformer)

```python
from src.evaluation.temporal_metrics import (
    compute_ranking_metrics,
    print_ranking_metrics,
    save_results_to_file
)

# Compute ranks during evaluation
ranks = []
for batch in data_loader:
    # ... model forward pass ...
    # ... compute rank for each prediction ...
    ranks.append(rank)

# Compute all metrics
metrics = compute_ranking_metrics(ranks, k_values=[1, 3, 10, 100])

# Pretty print results
print_ranking_metrics(metrics, phase="Test", epoch=10)

# Save to file
save_results_to_file(
    filepath="results/test_results.txt",
    metrics=metrics,
    config={'model': 'KGTransformer', 'split': 'test'}
)
```

### Using Evaluation Scripts

#### For Static Models
```bash
python evaluate.py \
    --model temporal_transe \
    --checkpoint checkpoints/best_model.pth \
    --split test \
    --filtered
```

#### For KGTransformer (PyG)
```bash
python evaluate_kgt_pyg.py \
    --checkpoint checkpoints/kgt_pyg_best.pth \
    --split test \
    --eval_batch_interval 1
```

## Metrics Comparison

### Standard KG Metrics vs FinDKG Metrics

| Metric | Standard KG | FinDKG | Notes |
|--------|-------------|---------|-------|
| MRR | ✓ | ✓ | Same definition |
| Hits@1 | ✓ | ✓ (REC1) | Same definition |
| Hits@3 | ✓ | ✓ (REC3) | Same definition |
| Hits@10 | ✓ | ✓ (REC10) | Same definition |
| Hits@100 | - | ✓ (REC100) | Additional in FinDKG |
| Mean Rank | ✓ | ✓ | Same definition |
| Median Rank | ✓ | ✓ | Same definition |
| MAE | - | ✓ | For time prediction |
| RMSE | - | ✓ | For time prediction |

## Evaluation Protocols

### Static Models
1. Load trained model checkpoint
2. Load test data
3. For each test triplet (s, r, o, t):
   - Score all possible objects: (s, r, o', t)
   - Rank the true object o
   - Optionally filter out other valid triplets
4. Compute metrics from ranks

### Temporal Models (KGTransformer)
1. Load trained model checkpoint
2. Initialize dynamic embeddings
3. Process test data in **temporal order** (sequential batching)
4. For each batch:
   - Evaluate link prediction on current batch
   - Update dynamic embeddings
   - Continue to next batch
5. Compute metrics from all ranks

**Important**: Temporal models must process data sequentially to maintain temporal consistency.

## Output Format

### Standard Output
```
======================================================================
Test Results
======================================================================
MRR:         0.234567

Recall@K (Hits@K):
  Recall@  1: 0.123456
  Recall@  3: 0.234567
  Recall@ 10: 0.345678
  Recall@100: 0.456789

Mean Rank:   123.45
Median Rank: 67.00
======================================================================
```

### FinDKG Format (for comparison with paper)
```
seed,REC1,REC3,REC10,MRR
0,0.123456,0.234567,0.345678,0.234567
```

## References

1. **Original FinDKG Paper**: Li et al., "FinDKG: Dynamic Knowledge Graphs with Large Language Models for Detecting Global Trends in Financial Markets"
2. **DGL Implementation**: https://github.com/xiaohui-victor-li/FinDKG
3. **Standard KG Evaluation**: Bordes et al., "Translating Embeddings for Modeling Multi-relational Data" (TransE paper)

