"""
Evaluation metrics for temporal knowledge graph models.

This module implements evaluation metrics from the original FinDKG paper:
- Link Prediction: MRR, Recall@K (Hits@K)
- Time Prediction: MAE, RMSE

Based on the original DGL implementation's eval_utils.py
"""

import numpy as np
import torch
from typing import List, Dict, Union


class RankingMetric:
    """Ranking metrics for link prediction evaluation."""
    
    @classmethod
    def mean_reciprocal_rank(cls, true_ranks: List[float]) -> float:
        """
        Compute Mean Reciprocal Rank (MRR).
        
        Args:
            true_ranks: List of ranks for true entities
            
        Returns:
            MRR score
        """
        return np.mean([1.0 / r for r in true_ranks])
    
    @classmethod
    def recall(cls, true_ranks: List[float], k: int = 10) -> float:
        """
        Compute Recall@K (also known as Hits@K).
        
        Args:
            true_ranks: List of ranks for true entities
            k: Cutoff for recall computation
            
        Returns:
            Recall@K score
        """
        return sum(np.array(true_ranks) <= k) * 1.0 / len(true_ranks)
    
    @classmethod
    def mean_rank(cls, true_ranks: List[float]) -> float:
        """
        Compute Mean Rank (MR).
        
        Args:
            true_ranks: List of ranks for true entities
            
        Returns:
            Mean Rank score
        """
        return np.mean(true_ranks)
    
    @classmethod
    def median_rank(cls, true_ranks: List[float]) -> float:
        """
        Compute Median Rank.
        
        Args:
            true_ranks: List of ranks for true entities
            
        Returns:
            Median Rank score
        """
        return np.median(true_ranks)


class RegressionMetric:
    """Regression metrics for time prediction evaluation."""
    
    @classmethod
    def mean_absolute_error(cls, diffs: List[float]) -> float:
        """
        Compute Mean Absolute Error (MAE).
        
        Args:
            diffs: List of differences between predicted and true values
            
        Returns:
            MAE score
        """
        return np.mean([abs(diff) for diff in diffs])
    
    @classmethod
    def mean_squared_error(cls, diffs: List[float]) -> float:
        """
        Compute Mean Squared Error (MSE).
        
        Args:
            diffs: List of differences between predicted and true values
            
        Returns:
            MSE score
        """
        return np.mean([diff ** 2 for diff in diffs])
    
    @classmethod
    def root_mean_squared_error(cls, diffs: List[float]) -> float:
        """
        Compute Root Mean Squared Error (RMSE).
        
        Args:
            diffs: List of differences between predicted and true values
            
        Returns:
            RMSE score
        """
        return np.sqrt(cls.mean_squared_error(diffs))


def compute_ranking_metrics(ranks: List[float], 
                            k_values: List[int] = [1, 3, 10, 100]) -> Dict[str, float]:
    """
    Compute all ranking metrics from a list of ranks.
    
    Args:
        ranks: List of ranks for true entities
        k_values: List of K values for Recall@K computation
        
    Returns:
        Dictionary containing all ranking metrics
    """
    metrics = {}
    
    # MRR
    metrics['MRR'] = RankingMetric.mean_reciprocal_rank(ranks)
    
    # Recall@K (Hits@K)
    for k in k_values:
        metrics[f'REC{k}'] = RankingMetric.recall(ranks, k)
        metrics[f'Hits@{k}'] = RankingMetric.recall(ranks, k)  # Alias
    
    # Mean and Median Rank
    metrics['MR'] = RankingMetric.mean_rank(ranks)
    metrics['MedianR'] = RankingMetric.median_rank(ranks)
    
    return metrics


def compute_regression_metrics(diffs: List[float]) -> Dict[str, float]:
    """
    Compute all regression metrics from a list of differences.
    
    Args:
        diffs: List of differences between predicted and true values
        
    Returns:
        Dictionary containing all regression metrics
    """
    metrics = {}
    
    # MAE and RMSE
    metrics['MAE'] = RegressionMetric.mean_absolute_error(diffs)
    metrics['RMSE'] = RegressionMetric.root_mean_squared_error(diffs)
    metrics['MSE'] = RegressionMetric.mean_squared_error(diffs)
    
    return metrics


def print_ranking_metrics(metrics: Dict[str, float], 
                         phase: str = "Evaluation",
                         epoch: int = None):
    """
    Pretty print ranking metrics.
    
    Args:
        metrics: Dictionary of metrics
        phase: Evaluation phase name (Train/Valid/Test)
        epoch: Optional epoch number
    """
    print("=" * 70)
    if epoch is not None:
        print(f"[Epoch {epoch}] {phase} Results")
    else:
        print(f"{phase} Results")
    print("=" * 70)
    
    # MRR
    if 'MRR' in metrics:
        print(f"MRR:         {metrics['MRR']:.6f}")
    
    # Recall@K
    print("\nRecall@K (Hits@K):")
    for k in [1, 3, 10, 100]:
        key = f'REC{k}' if f'REC{k}' in metrics else f'Hits@{k}'
        if key in metrics:
            print(f"  Recall@{k:3d}: {metrics[key]:.6f}")
    
    # Mean and Median Rank
    if 'MR' in metrics:
        print(f"\nMean Rank:   {metrics['MR']:.2f}")
    if 'MedianR' in metrics:
        print(f"Median Rank: {metrics['MedianR']:.2f}")
    
    print("=" * 70)


def print_regression_metrics(metrics: Dict[str, float],
                             phase: str = "Evaluation",
                             epoch: int = None):
    """
    Pretty print regression metrics.
    
    Args:
        metrics: Dictionary of metrics
        phase: Evaluation phase name (Train/Valid/Test)
        epoch: Optional epoch number
    """
    print("=" * 70)
    if epoch is not None:
        print(f"[Epoch {epoch}] {phase} Results")
    else:
        print(f"{phase} Results")
    print("=" * 70)
    
    if 'MAE' in metrics:
        print(f"MAE:  {metrics['MAE']:.6f}")
    if 'RMSE' in metrics:
        print(f"RMSE: {metrics['RMSE']:.6f}")
    if 'MSE' in metrics:
        print(f"MSE:  {metrics['MSE']:.6f}")
    
    print("=" * 70)


def save_results_to_file(filepath: str, 
                         metrics: Dict[str, float],
                         config: Dict = None):
    """
    Save evaluation results to a file.
    
    Args:
        filepath: Path to save results
        metrics: Dictionary of metrics
        config: Optional configuration dictionary
    """
    with open(filepath, 'w') as f:
        if config:
            f.write("Configuration:\n")
            for key, value in config.items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")
        
        f.write("Metrics:\n")
        for key, value in metrics.items():
            if isinstance(value, float):
                f.write(f"  {key}: {value:.6f}\n")
            else:
                f.write(f"  {key}: {value}\n")


# Example usage
if __name__ == "__main__":
    print("Temporal Knowledge Graph Evaluation Metrics")
    print("=" * 70)
    
    # Example ranking metrics
    example_ranks = [1, 3, 5, 2, 10, 1, 4, 6, 8, 2]
    ranking_metrics = compute_ranking_metrics(example_ranks)
    print_ranking_metrics(ranking_metrics, phase="Example")
    
    print("\n")
    
    # Example regression metrics
    example_diffs = [0.5, -0.3, 0.8, -0.2, 0.1, -0.6, 0.4, -0.1, 0.3, -0.4]
    regression_metrics = compute_regression_metrics(example_diffs)
    print_regression_metrics(regression_metrics, phase="Example")

