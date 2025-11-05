"""
TransE: Translating Embeddings for Modeling Multi-relational Data
Paper: https://papers.nips.cc/paper/2013/file/1cecc7a77928ca8133fa24680a88d2f9-Paper.pdf

TransE models relations as translations in the embedding space: h + r ≈ t
"""

import torch
import torch.nn as nn
from .base_model import BaseKGModel


class TransE(BaseKGModel):
    """TransE model: h + r ≈ t"""
    
    def __init__(self, num_entities, num_relations, embedding_dim=100, margin=1.0, p_norm=2):
        super().__init__(num_entities, num_relations, embedding_dim)
        
        self.margin = margin
        self.p_norm = p_norm
        
        # Entity embeddings
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        
        # Relation embeddings
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)
        
        # Initialize embeddings
        nn.init.xavier_uniform_(self.entity_embeddings.weight)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)
    
    def score(self, subjects, relations, objects):
        """
        Compute TransE scores: -||h + r - t||_p
        
        Lower distance = higher score (more plausible triplet)
        """
        h = self.entity_embeddings(subjects)
        r = self.relation_embeddings(relations)
        t = self.entity_embeddings(objects)
        
        # TransE scoring: -||h + r - t||
        score = -torch.norm(h + r - t, p=self.p_norm, dim=-1)
        
        return score
    
    def forward(self, batch):
        """
        Forward pass for training with positive and negative triplets.
        """
        # Positive triplets: [batch_size, 4] (subj, rel, obj, time)
        pos_triplets = batch['positive']
        pos_subjects = pos_triplets[:, 0]
        pos_relations = pos_triplets[:, 1]
        pos_objects = pos_triplets[:, 2]
        
        # Negative triplets: [batch_size, num_negatives, 4]
        neg_triplets = batch['negatives']
        batch_size, num_negatives = neg_triplets.shape[0], neg_triplets.shape[1]
        
        # Flatten negatives for scoring
        neg_triplets_flat = neg_triplets.view(-1, 4)
        neg_subjects = neg_triplets_flat[:, 0]
        neg_relations = neg_triplets_flat[:, 1]
        neg_objects = neg_triplets_flat[:, 2]
        
        # Compute scores
        pos_scores = self.score(pos_subjects, pos_relations, pos_objects)
        neg_scores = self.score(neg_subjects, neg_relations, neg_objects)
        
        # Reshape negative scores
        neg_scores = neg_scores.view(batch_size, num_negatives)
        
        return {
            'pos_scores': pos_scores,
            'neg_scores': neg_scores
        }
    
    def normalize_embeddings(self):
        """Normalize entity embeddings to unit sphere (important for TransE)."""
        with torch.no_grad():
            self.entity_embeddings.weight.data = torch.nn.functional.normalize(
                self.entity_embeddings.weight.data, p=2, dim=1
            )

