"""
DistMult: Embedding Entities and Relations for Learning and Inference in Knowledge Bases
Paper: https://arxiv.org/abs/1412.6575

DistMult uses bilinear scoring: score = h^T diag(r) t
"""

import torch
import torch.nn as nn
from .base_model import BaseKGModel


class DistMult(BaseKGModel):
    """DistMult model using bilinear scoring."""
    
    def __init__(self, num_entities, num_relations, embedding_dim=100):
        super().__init__(num_entities, num_relations, embedding_dim)
        
        # Entity embeddings
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        
        # Relation embeddings (diagonal matrix represented as vector)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)
        
        # Initialize embeddings
        nn.init.xavier_uniform_(self.entity_embeddings.weight)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)
    
    def score(self, subjects, relations, objects):
        """
        Compute DistMult scores: <h, r, t> = sum(h * r * t)
        
        Higher value = higher score (more plausible triplet)
        """
        h = self.entity_embeddings(subjects)
        r = self.relation_embeddings(relations)
        t = self.entity_embeddings(objects)
        
        # DistMult scoring: element-wise multiplication and sum
        score = torch.sum(h * r * t, dim=-1)
        
        return score
    
    def forward(self, batch):
        """Forward pass for training."""
        # Positive triplets
        pos_triplets = batch['positive']
        pos_subjects = pos_triplets[:, 0]
        pos_relations = pos_triplets[:, 1]
        pos_objects = pos_triplets[:, 2]
        
        # Negative triplets
        neg_triplets = batch['negatives']
        batch_size, num_negatives = neg_triplets.shape[0], neg_triplets.shape[1]
        
        neg_triplets_flat = neg_triplets.view(-1, 4)
        neg_subjects = neg_triplets_flat[:, 0]
        neg_relations = neg_triplets_flat[:, 1]
        neg_objects = neg_triplets_flat[:, 2]
        
        # Compute scores
        pos_scores = self.score(pos_subjects, pos_relations, pos_objects)
        neg_scores = self.score(neg_subjects, neg_relations, neg_objects)
        neg_scores = neg_scores.view(batch_size, num_negatives)
        
        return {
            'pos_scores': pos_scores,
            'neg_scores': neg_scores
        }

