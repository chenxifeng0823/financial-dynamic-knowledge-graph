"""
ComplEx: Complex Embeddings for Simple Link Prediction
Paper: https://arxiv.org/abs/1606.06357

ComplEx extends DistMult to complex space to handle asymmetric relations.
"""

import torch
import torch.nn as nn
from .base_model import BaseKGModel


class ComplEx(BaseKGModel):
    """ComplEx model using complex embeddings."""
    
    def __init__(self, num_entities, num_relations, embedding_dim=100):
        super().__init__(num_entities, num_relations, embedding_dim)
        
        # Entity embeddings (real and imaginary parts)
        self.entity_embeddings_real = nn.Embedding(num_entities, embedding_dim)
        self.entity_embeddings_imag = nn.Embedding(num_entities, embedding_dim)
        
        # Relation embeddings (real and imaginary parts)
        self.relation_embeddings_real = nn.Embedding(num_relations, embedding_dim)
        self.relation_embeddings_imag = nn.Embedding(num_relations, embedding_dim)
        
        # Initialize embeddings
        nn.init.xavier_uniform_(self.entity_embeddings_real.weight)
        nn.init.xavier_uniform_(self.entity_embeddings_imag.weight)
        nn.init.xavier_uniform_(self.relation_embeddings_real.weight)
        nn.init.xavier_uniform_(self.relation_embeddings_imag.weight)
    
    def score(self, subjects, relations, objects):
        """
        Compute ComplEx scores: Re(<h, r, conj(t)>)
        
        where h, r, t are complex vectors.
        """
        # Get real and imaginary parts
        h_real = self.entity_embeddings_real(subjects)
        h_imag = self.entity_embeddings_imag(subjects)
        
        r_real = self.relation_embeddings_real(relations)
        r_imag = self.relation_embeddings_imag(relations)
        
        t_real = self.entity_embeddings_real(objects)
        t_imag = self.entity_embeddings_imag(objects)
        
        # ComplEx scoring: Re(<h, r, conj(t)>)
        # conj(t) = t_real - i*t_imag
        # <h, r, conj(t)> = (h_real + i*h_imag) * (r_real + i*r_imag) * (t_real - i*t_imag)
        
        # Compute real part of the product
        score = (
            h_real * r_real * t_real +
            h_real * r_imag * t_imag +
            h_imag * r_real * t_imag -
            h_imag * r_imag * t_real
        ).sum(dim=-1)
        
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

