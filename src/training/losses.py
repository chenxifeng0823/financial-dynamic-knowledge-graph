"""
Loss functions for knowledge graph embedding models.
"""

import torch
import torch.nn as nn


def margin_ranking_loss(pos_scores, neg_scores, margin=1.0):
    """
    Margin-based ranking loss used by TransE.
    
    Loss = max(0, margin - pos_score + neg_score)
    
    Args:
        pos_scores: Scores for positive triplets [batch_size]
        neg_scores: Scores for negative triplets [batch_size, num_negatives]
        margin: Margin value (default: 1.0)
    
    Returns:
        Scalar loss value
    """
    pos_scores = pos_scores.unsqueeze(1)  # [batch_size, 1]
    losses = torch.clamp(margin - pos_scores + neg_scores, min=0)
    return losses.mean()


def bce_loss(pos_scores, neg_scores):
    """
    Binary cross-entropy loss for knowledge graph embeddings.
    
    Treats positive triplets as class 1, negative triplets as class 0.
    
    Args:
        pos_scores: Scores for positive triplets [batch_size]
        neg_scores: Scores for negative triplets [batch_size, num_negatives]
    
    Returns:
        Scalar loss value
    """
    # Positive loss
    pos_loss = -torch.log(torch.sigmoid(pos_scores) + 1e-10).mean()
    
    # Negative loss
    neg_loss = -torch.log(1 - torch.sigmoid(neg_scores) + 1e-10).mean()
    
    return pos_loss + neg_loss


def softplus_loss(pos_scores, neg_scores):
    """
    Softplus loss (log-sigmoid loss).
    
    Args:
        pos_scores: Scores for positive triplets [batch_size]
        neg_scores: Scores for negative triplets [batch_size, num_negatives]
    
    Returns:
        Scalar loss value
    """
    pos_loss = torch.nn.functional.softplus(-pos_scores).mean()
    neg_loss = torch.nn.functional.softplus(neg_scores).mean()
    
    return pos_loss + neg_loss

