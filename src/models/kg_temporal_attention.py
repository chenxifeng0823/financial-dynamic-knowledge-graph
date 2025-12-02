"""
KGTransformer with Temporal Attention (Transformer) instead of RNN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple, deque

from .pyg_modules.rgcn import RGCN
from .pyg_modules.graph_transformer import KGTransformer
from .pyg_modules.temporal_attention import GraphTemporalTransformerConv

# New State Structure for Attention Model
# history: Tensor [N, Window_Size, D]
# timestamps: Tensor [N, Window_Size]
AttentionState = namedtuple('AttentionState', ['history', 'timestamps', 'mask'], defaults=[None, None, None])

# Multi-aspect embedding structure (modified for attention state)
# structural: Standard Tensor [N, D] (Latest state)
# temporal: AttentionState tuple (History window)
MultiAspectEmbedding = namedtuple('MultiAspectEmbedding', ['structural', 'temporal'], defaults=[None, None])


class EmbeddingUpdaterAttention(nn.Module):
    """
    Embedding Updater with Temporal Attention.
    
    Replaces RNN with Transformer Attention for temporal updates.
    Maintains a sliding window of history for each node.
    """
    
    def __init__(
        self,
        num_nodes,
        in_dim,
        structural_hid_dim,
        temporal_hid_dim,
        num_rels,
        rel_embed_dim,
        num_node_types=1,
        graph_structural_conv='KGT', # 'KGT' or 'RGCN'
        num_gconv_layers=2,
        window_size=10,
        num_attn_heads=4,
        dropout=0.1,
        activation=None,
        graph_name=None
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.window_size = window_size
        self.structural_hid_dim = structural_hid_dim
        
        # 1. Structural Encoder (Spatial GNN)
        # This part is same as baseline - extracting features from current graph
        if graph_structural_conv == 'RGCN':
            self.graph_conv = RGCN(
                in_dim, structural_hid_dim, structural_hid_dim,
                n_layers=num_gconv_layers,
                num_rels=num_rels,
                regularizer="bdd",
                num_bases=100,
                dropout=dropout,
                activation=activation,
                layer_norm=False
            )
        elif graph_structural_conv == 'KGT':
            self.graph_conv = KGTransformer(
                in_dim, structural_hid_dim, num_attn_heads, structural_hid_dim,
                n_layers=num_gconv_layers,
                num_nodes=num_node_types,
                num_rels=num_rels,
                layer_norm=True,
                dropout=dropout
            )
            
        # 2. Temporal Attention Module
        # Replaces the RNN. Takes current spatial features + history -> New State
        self.temporal_attn = GraphTemporalTransformerConv(
            in_dim=structural_hid_dim, # Input is output of GNN
            d_model=temporal_hid_dim,
            num_heads=num_attn_heads,
            window_size=window_size,
            dropout=dropout
        )
        
        # 3. Relation Update (Keep simple RNN or Linear for now to isolate node effect)
        # For simplicity in this experiment, we'll use a GRU for relations as they are global
        self.relation_updater = nn.GRUCell(rel_embed_dim, rel_embed_dim)

    def update_history(self, current_state, current_time, old_history):
        """
        Updates the sliding window history buffer.
        
        Args:
            current_state: [Batch, D] - New computed states
            current_time: Scalar - Current timestamp
            old_history: AttentionState(history, timestamps, mask)
                history: [Batch, Window, D]
                timestamps: [Batch, Window]
                mask: [Batch, Window] (1=valid, 0=padding)
        
        Returns:
            New AttentionState
        """
        batch_size = current_state.size(0)
        device = current_state.device
        
        if old_history.history is None:
            # Initialize empty history
            # We create a buffer of size [Batch, Window, D]
            # But we only fill the last slot
            new_hist_tensor = torch.zeros(batch_size, self.window_size, current_state.size(-1), device=device)
            new_time_tensor = torch.zeros(batch_size, self.window_size, device=device)
            new_mask = torch.zeros(batch_size, self.window_size, device=device)
            
            # Add current as first entry (at the end of window)
            new_hist_tensor[:, -1, :] = current_state
            new_time_tensor[:, -1] = current_time
            new_mask[:, -1] = 1
            
            return AttentionState(new_hist_tensor, new_time_tensor, new_mask)
        
        else:
            # Shift window left
            # history: [Batch, Window, D] -> [Batch, Window-1, D]
            prev_hist = old_history.history[:, 1:, :]
            prev_time = old_history.timestamps[:, 1:]
            prev_mask = old_history.mask[:, 1:]
            
            # Append new state
            new_hist_tensor = torch.cat([prev_hist, current_state.unsqueeze(1)], dim=1)
            
            # Append new time
            curr_time_expanded = torch.full((batch_size, 1), current_time, device=device)
            new_time_tensor = torch.cat([prev_time, curr_time_expanded], dim=1)
            
            # Append new mask (1)
            curr_mask = torch.ones((batch_size, 1), device=device)
            new_mask = torch.cat([prev_mask, curr_mask], dim=1)
            
            return AttentionState(new_hist_tensor, new_time_tensor, new_mask)

    def forward(self, batch_data, static_entity_emb, dynamic_entity_emb, dynamic_relation_emb, device):
        """
        Args:
            batch_data: PyG Data object
            static_entity_emb: [N, D]
            dynamic_entity_emb: MultiAspectEmbedding
                structural: [N, D] (Latest state)
                temporal: AttentionState (History window)
            dynamic_relation_emb: [R, D]
        """
        # 1. Spatial Convolution (Get current features)
        # Use static embeddings + graph structure to get spatial features
        # Note: In baseline, we used static structural embeddings
        batch_nodes = batch_data.node_id.cpu()
        batch_static_emb = static_entity_emb[batch_nodes].to(device)
        
        if isinstance(self.graph_conv, RGCN):
            # RGCN needs edge norm
            from .pyg_modules.embedding_updater import node_norm_to_edge_norm
            edge_norm = node_norm_to_edge_norm(batch_data)
            spatial_features = self.graph_conv(
                batch_static_emb,
                batch_data.edge_index,
                batch_data.edge_type.long(),
                edge_norm
            )
        else:
            # KGT
            spatial_features = self.graph_conv(
                batch_static_emb,
                batch_data.edge_index,
                batch_data.node_type.long(),
                batch_data.edge_type.long()
            )
            
        # 2. Temporal Attention
        # Retrieve history for current batch nodes
        history_state = dynamic_entity_emb.temporal
        
        batch_history = None
        batch_hist_times = None
        batch_mask = None
        
        if history_state.history is not None:
            # Indexing: [N, Window, D] -> [Batch, Window, D]
            batch_history = history_state.history[batch_nodes].to(device)
            batch_hist_times = history_state.timestamps[batch_nodes].to(device)
            batch_mask = history_state.mask[batch_nodes].to(device)
            
        current_timestamp = batch_data.timestamp
        
        # Apply Attention Mechanism
        # This fuses Spatial Features (Q) with History (K,V)
        new_node_states = self.temporal_attn(
            current_features=spatial_features,
            history_features=batch_history,
            history_times=batch_hist_times,
            current_time=current_timestamp,
            mask=batch_mask
        )
        
        # 3. Update Global State (CPU)
        # Update latest structural state
        updated_structural = dynamic_entity_emb.structural.clone()
        updated_structural[batch_nodes] = new_node_states.detach().cpu()
        
        # Update history window
        # We need to act on the *global* history tensors
        # But efficiently. For simplicity here:
        # We perform the shift logic for ALL nodes (or just batch nodes if we optimize)
        # Optimization: Only update batch nodes in the global tensor
        
        # Construct the new history specifically for batch nodes
        # We can reuse the update_history logic but applied to the batch slice
        # Then scatter back to global
        
        # Helper to get global slice
        global_hist = history_state.history if history_state.history is not None else \
            torch.zeros(self.num_nodes, self.window_size, self.structural_hid_dim)
        global_times = history_state.timestamps if history_state.timestamps is not None else \
            torch.zeros(self.num_nodes, self.window_size)
        global_mask = history_state.mask if history_state.mask is not None else \
            torch.zeros(self.num_nodes, self.window_size)
            
        # Slice batch
        batch_hist_slice = AttentionState(
            global_hist[batch_nodes].to(device),
            global_times[batch_nodes].to(device),
            global_mask[batch_nodes].to(device)
        )
        
        # Update batch slice
        new_batch_hist_slice = self.update_history(
            new_node_states.detach(), # Detach for storage
            current_timestamp,
            batch_hist_slice
        )
        
        # Scatter back to global (CPU)
        updated_history = global_hist.clone()
        updated_history[batch_nodes] = new_batch_hist_slice.history.cpu()
        
        updated_times = global_times.clone()
        updated_times[batch_nodes] = new_batch_hist_slice.timestamps.cpu()
        
        updated_mask = global_mask.clone()
        updated_mask[batch_nodes] = new_batch_hist_slice.mask.cpu()
        
        new_dynamic_entity_emb = MultiAspectEmbedding(
            structural=updated_structural,
            temporal=AttentionState(updated_history, updated_times, updated_mask)
        )
        
        # 4. Relation Updates (Simplified)
        # Just use simple GRU on relation embeddings
        # Assuming relation features are mean of entity features (simplified for now)
        # In a real impl, we'd replicate the scatter_mean logic from baseline
        # Here we just pass through for structural compatibility
        new_dynamic_relation_emb = dynamic_relation_emb # Placeholder
        
        return new_dynamic_entity_emb, new_dynamic_relation_emb


class KGTemporalAttention(nn.Module):
    """
    KGTransformer with Temporal Attention.
    """
    def __init__(self, num_entities, num_relations, args):
        super().__init__()
        self.args = args
        self.device = args.device
        
        # Static Embeddings
        self.static_entity_embeds = nn.Parameter(torch.randn(num_entities, args.static_entity_embed_dim))
        self.relation_embeds = nn.Parameter(torch.randn(num_relations, args.rel_embed_dim))
        
        # Dynamic State Storage
        # Structural: [N, D]
        # Temporal: History Window
        self.dynamic_entity_embeds = MultiAspectEmbedding(
            structural=torch.zeros(num_entities, args.structural_dynamic_entity_embed_dim),
            temporal=AttentionState() # Empty initially
        )
        
        self.dynamic_relation_embeds = torch.zeros(num_relations, args.rel_embed_dim)
        
        # Updater
        self.embedding_updater = EmbeddingUpdaterAttention(
            num_nodes=num_entities,
            in_dim=args.static_entity_embed_dim,
            structural_hid_dim=args.structural_dynamic_entity_embed_dim,
            temporal_hid_dim=args.temporal_dynamic_entity_embed_dim,
            num_rels=num_relations,
            rel_embed_dim=args.rel_embed_dim,
            graph_structural_conv='KGT',
            window_size=10, # Configurable
            num_attn_heads=4,
            dropout=args.dropout
        ).to(self.device)
        
        # Decoder (EdgeModel) - Reuse from baseline logic or simplified
        # For compatibility, we'll create a simple decoder here
        # that mimics the baseline's tail prediction
        self.decoder_head = nn.Sequential(
            nn.Linear(args.static_entity_embed_dim + args.structural_dynamic_entity_embed_dim + args.rel_embed_dim, 200),
            nn.Tanh(),
            nn.Linear(200, num_entities)
        )
        
    def forward(self, batch_data):
        # 1. Update Embeddings
        self.dynamic_entity_embeds, self.dynamic_relation_embeds = self.embedding_updater(
            batch_data,
            self.static_entity_embeds,
            self.dynamic_entity_embeds,
            self.dynamic_relation_embeds,
            self.device
        )
        
        # 2. Predict (Simplified Tail Prediction)
        # Combine Static + Dynamic
        batch_nodes = batch_data.node_id.cpu()
        static_emb = self.static_entity_embeds[batch_nodes].to(self.device)
        dynamic_emb = self.dynamic_entity_embeds.structural[batch_nodes].to(self.device)
        
        # We need embeddings for Heads and Relations in the batch edges
        # edge_index is local [2, B]
        heads_local = batch_data.edge_index[0]
        rels = batch_data.edge_type
        
        head_emb = torch.cat([static_emb[heads_local], dynamic_emb[heads_local]], dim=1)
        rel_emb = self.relation_embeds[rels]
        
        # Input: [Head, Rel] -> Output: Scores for all Tails
        decoder_input = torch.cat([head_emb, rel_emb], dim=1)
        scores = self.decoder_head(decoder_input)
        
        # Loss
        tails_global = batch_data.node_id[batch_data.edge_index[1].cpu()].to(self.device)
        loss = F.cross_entropy(scores, tails_global)
        
        return loss, scores

