"""
Evaluation metrics for knowledge graph embedding models.

Implements standard metrics:
- Mean Reciprocal Rank (MRR)
- Hits@K (K=1, 3, 10)
- Filtered evaluation (removes other valid triplets from ranking)
"""

import torch
import numpy as np
from tqdm import tqdm


def compute_ranks(model, triplets, all_entities, batch_size=100, filtered=False, 
                  all_triplets=None):
    """
    Compute ranks for triplet predictions.
    
    Args:
        model: Knowledge graph model
        triplets: Test triplets [num_test, 4] (subj, rel, obj, time)
        all_entities: Total number of entities
        batch_size: Batch size for evaluation
        filtered: If True, remove other valid triplets from ranking
        all_triplets: All valid triplets for filtered evaluation
        
    Returns:
        ranks: Ranks for each triplet
    """
    model.eval()
    ranks = []
    
    # Convert to tensors
    if not isinstance(triplets, torch.Tensor):
        triplets = torch.from_numpy(triplets)
    
    device = next(model.parameters()).device
    
    with torch.no_grad():
        for i in tqdm(range(0, len(triplets), batch_size), desc="Evaluating"):
            batch = triplets[i:i + batch_size].to(device)
            batch_size_actual = batch.size(0)
            
            subjects = batch[:, 0]
            relations = batch[:, 1]
            objects = batch[:, 2]
            
            # Score against all entities for object prediction
            # Expand to [batch_size, num_entities]
            subjects_expanded = subjects.unsqueeze(1).expand(batch_size_actual, all_entities)
            relations_expanded = relations.unsqueeze(1).expand(batch_size_actual, all_entities)
            all_objs = torch.arange(all_entities, device=device).unsqueeze(0).expand(batch_size_actual, all_entities)
            
            # Compute scores for all possible objects
            scores = model.score(subjects_expanded, relations_expanded, all_objs)
            
            # Filtered setting: set scores of other valid triplets to -inf
            if filtered and all_triplets is not None:
                for j in range(batch_size_actual):
                    subj = subjects[j].item()
                    rel = relations[j].item()
                    true_obj = objects[j].item()
                    
                    # Find all valid objects for this (subject, relation) pair
                    mask = (all_triplets[:, 0] == subj) & (all_triplets[:, 1] == rel)
                    valid_objs = all_triplets[mask, 2]
                    
                    # Set scores of other valid objects to -inf (except true object)
                    for valid_obj in valid_objs:
                        if valid_obj != true_obj:
                            scores[j, valid_obj] = float('-inf')
            
            # Get ranks
            for j in range(batch_size_actual):
                true_obj = objects[j]
                score_true = scores[j, true_obj]
                
                # Count how many scores are greater than the true score
                rank = (scores[j] > score_true).sum().item() + 1
                ranks.append(rank)
    
    return np.array(ranks)


def compute_metrics(ranks):
    """
    Compute evaluation metrics from ranks.
    
    Args:
        ranks: Array of ranks for each triplet
        
    Returns:
        Dictionary with metrics
    """
    metrics = {}
    
    # Mean Reciprocal Rank
    metrics['mrr'] = np.mean(1.0 / ranks)
    
    # Hits@K
    for k in [1, 3, 10]:
        metrics[f'hits@{k}'] = np.mean(ranks <= k)
    
    # Additional statistics
    metrics['mean_rank'] = np.mean(ranks)
    metrics['median_rank'] = np.median(ranks)
    
    return metrics


def evaluate_model(model, test_data, num_entities, batch_size=100, 
                   filtered=False, all_triplets=None):
    """
    Evaluate a knowledge graph model.
    
    Args:
        model: Knowledge graph model
        test_data: Test triplets
        num_entities: Total number of entities
        batch_size: Batch size for evaluation
        filtered: Use filtered evaluation
        all_triplets: All valid triplets (for filtered evaluation)
        
    Returns:
        Dictionary with evaluation metrics
    """
    print(f"\n{'='*70}")
    print(f"Evaluation Mode: {'Filtered' if filtered else 'Raw'}")
    print(f"{'='*70}")
    
    # Compute ranks
    ranks = compute_ranks(
        model=model,
        triplets=test_data,
        all_entities=num_entities,
        batch_size=batch_size,
        filtered=filtered,
        all_triplets=all_triplets
    )
    
    # Compute metrics
    metrics = compute_metrics(ranks)
    
    # Print results
    print(f"\nResults:")
    print(f"  MRR: {metrics['mrr']:.4f}")
    print(f"  Hits@1: {metrics['hits@1']:.4f}")
    print(f"  Hits@3: {metrics['hits@3']:.4f}")
    print(f"  Hits@10: {metrics['hits@10']:.4f}")
    print(f"  Mean Rank: {metrics['mean_rank']:.2f}")
    print(f"  Median Rank: {metrics['median_rank']:.2f}")
    print(f"{'='*70}\n")
    
    return metrics


# Example usage
if __name__ == "__main__":
    print("Evaluation metrics module")
    print("Import this module to use evaluation functions")
    print("\nExample:")
    print("  from src.evaluation.metrics import evaluate_model")
    print("  metrics = evaluate_model(model, test_data, num_entities)")

