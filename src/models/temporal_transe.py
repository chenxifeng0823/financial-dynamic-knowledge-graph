"""
Temporal TransE: TransE with time embeddings.

Extension of TransE that incorporates temporal information:
    score = -||h + r + t_emb - tail||
    
where t_emb is a learned embedding for the timestamp.

Paper references:
- "Temporal Knowledge Graph Completion using a Linear Temporal Regularizer" (2018)
- Various temporal KG embedding papers extend TransE this way
"""

import torch
import torch.nn as nn
from .base_model import BaseKGModel


class TemporalTransE(BaseKGModel):
    """
    Temporal TransE: h + r + time_embedding ≈ t
    
    Extends TransE by adding time-specific embeddings to capture temporal dynamics.
    """
    
    def __init__(self, num_entities, num_relations, num_timestamps, 
                 embedding_dim=100, time_dim=50, margin=1.0, p_norm=2):
        """
        Initialize Temporal TransE.
        
        Args:
            num_entities: Number of entities
            num_relations: Number of relations
            num_timestamps: Number of unique timestamps
            embedding_dim: Dimension of entity/relation embeddings
            time_dim: Dimension of time embeddings
            margin: Margin for ranking loss
            p_norm: Norm to use (1 or 2)
        """
        super().__init__(num_entities, num_relations, embedding_dim)
        
        self.margin = margin
        self.p_norm = p_norm
        self.time_dim = time_dim
        self.num_timestamps = num_timestamps
        
        # Entity embeddings
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        
        # Relation embeddings
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)
        
        # Time embeddings (new!)
        self.time_embeddings = nn.Embedding(num_timestamps, time_dim)
        
        # Project time embeddings to same dimension as entity embeddings
        self.time_projection = nn.Linear(time_dim, embedding_dim)
        
        # Initialize embeddings
        nn.init.xavier_uniform_(self.entity_embeddings.weight)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)
        nn.init.xavier_uniform_(self.time_embeddings.weight)
        nn.init.xavier_uniform_(self.time_projection.weight)
    
    def score(self, subjects, relations, objects, times=None):
        """
        Compute Temporal TransE scores: -||h + r + time_proj - t||
        
        Args:
            subjects: Subject entity IDs
            relations: Relation IDs
            objects: Object entity IDs
            times: Timestamp IDs (optional, for backward compatibility)
        """
        h = self.entity_embeddings(subjects)
        r = self.relation_embeddings(relations)
        t = self.entity_embeddings(objects)
        
        if times is not None:
            # Get time embeddings and project to entity dimension
            time_emb = self.time_embeddings(times)
            time_vec = self.time_projection(time_emb)
            
            # Temporal TransE: h + r + time_vec ≈ t
            score = -torch.norm(h + r + time_vec - t, p=self.p_norm, dim=-1)
        else:
            # Fall back to regular TransE if no time info
            score = -torch.norm(h + r - t, p=self.p_norm, dim=-1)
        
        return score
    
    def forward(self, batch):
        """
        Forward pass for training with temporal information.
        """
        # Positive triplets: [batch_size, 4] (subj, rel, obj, time)
        pos_triplets = batch['positive']
        pos_subjects = pos_triplets[:, 0]
        pos_relations = pos_triplets[:, 1]
        pos_objects = pos_triplets[:, 2]
        pos_times = pos_triplets[:, 3]  # ← Now we use this!
        
        # Negative triplets: [batch_size, num_negatives, 4]
        neg_triplets = batch['negatives']
        batch_size, num_negatives = neg_triplets.shape[0], neg_triplets.shape[1]
        
        # Flatten negatives for scoring
        neg_triplets_flat = neg_triplets.view(-1, 4)
        neg_subjects = neg_triplets_flat[:, 0]
        neg_relations = neg_triplets_flat[:, 1]
        neg_objects = neg_triplets_flat[:, 2]
        neg_times = neg_triplets_flat[:, 3]
        
        # Compute scores with temporal information
        pos_scores = self.score(pos_subjects, pos_relations, pos_objects, pos_times)
        neg_scores = self.score(neg_subjects, neg_relations, neg_objects, neg_times)
        
        # Reshape negative scores
        neg_scores = neg_scores.view(batch_size, num_negatives)
        
        return {
            'pos_scores': pos_scores,
            'neg_scores': neg_scores
        }
    
    def normalize_embeddings(self):
        """Normalize entity embeddings to unit sphere."""
        with torch.no_grad():
            self.entity_embeddings.weight.data = torch.nn.functional.normalize(
                self.entity_embeddings.weight.data, p=2, dim=1
            )

