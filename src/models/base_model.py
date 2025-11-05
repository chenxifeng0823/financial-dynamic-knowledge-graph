"""
Base model class for all knowledge graph embedding models.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class BaseKGModel(nn.Module, ABC):
    """Abstract base class for knowledge graph embedding models."""
    
    def __init__(self, num_entities, num_relations, embedding_dim):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
    
    @abstractmethod
    def score(self, subjects, relations, objects):
        """
        Compute scores for triplets.
        
        Args:
            subjects: Subject entity IDs [batch_size]
            relations: Relation IDs [batch_size]
            objects: Object entity IDs [batch_size]
        
        Returns:
            Scores for the triplets [batch_size]
        """
        pass
    
    @abstractmethod
    def forward(self, batch):
        """
        Forward pass for training.
        
        Args:
            batch: Dictionary with 'positive' and 'negatives' triplets
        
        Returns:
            Dictionary with 'pos_scores' and 'neg_scores'
        """
        pass
    
    def predict(self, subjects, relations, objects=None):
        """
        Predict scores for evaluation.
        
        Args:
            subjects: Subject entity IDs
            relations: Relation IDs
            objects: Object entity IDs (if None, score against all entities)
        
        Returns:
            Scores
        """
        if objects is None:
            # Score against all entities
            batch_size = subjects.size(0)
            all_entities = torch.arange(self.num_entities, device=subjects.device)
            
            # Expand for broadcasting
            subjects = subjects.unsqueeze(1).expand(batch_size, self.num_entities)
            relations = relations.unsqueeze(1).expand(batch_size, self.num_entities)
            objects = all_entities.unsqueeze(0).expand(batch_size, self.num_entities)
            
            scores = self.score(subjects, relations, objects)
            return scores
        else:
            return self.score(subjects, relations, objects)

