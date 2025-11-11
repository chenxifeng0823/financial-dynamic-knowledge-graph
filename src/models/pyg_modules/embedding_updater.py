"""
Embedding Updater: Combines Graph Convolutions with RNN for temporal dynamics
Port of DGL's EmbeddingUpdater to PyG
"""

import torch
import torch.nn as nn
from torch_scatter import scatter_mean

from .rgcn import RGCN
from .graph_transformer import KGTransformer


class GraphStructuralRNNConv(nn.Module):
    """
    Graph Structural RNN Convolution
    
    Combines graph convolution (RGCN or KGTransformer) with RNN for structural dynamics.
    
    Args:
        graph_conv (str): Type of graph convolution ('RGCN' or 'KGT')
        num_gconv_layers (int): Number of graph convolution layers
        rnn (str): Type of RNN ('GRU' or 'RNN')
        num_rnn_layers (int): Number of RNN layers
        in_dim (int): Input dimension
        hid_dim (int): Hidden dimension
        num_nodes (int): Number of node types
        num_rels (int): Number of relation types
        rel_embed_dim (int): Relation embedding dimension
        add_entity_emb (bool): Whether to add entity embeddings to RNN input. Default: False
        dropout (float): Dropout rate. Default: 0.2
        num_node_types (int): Number of node types. Default: 1
        num_heads (int): Number of attention heads (for KGT). Default: 8
        activation (callable): Activation function. Default: None
        graph_name (str): Graph name (for specific configurations). Default: None
    """
    
    def __init__(
        self,
        graph_conv,
        num_gconv_layers,
        rnn,
        num_rnn_layers,
        in_dim,
        hid_dim,
        num_nodes,
        num_rels,
        rel_embed_dim,
        add_entity_emb=False,
        dropout=0.2,
        num_node_types=1,
        num_heads=8,
        activation=None,
        graph_name=None
    ):
        super().__init__()
        
        self.num_nodes = num_nodes
        self.num_node_types = num_node_types
        self.num_heads = num_heads
        self.num_rels = num_rels
        self.add_entity_emb = add_entity_emb
        
        # Node encoder layer (graph convolution)
        if graph_conv == 'RGCN':
            self.graph_conv = RGCN(
                in_dim, hid_dim, hid_dim,
                n_layers=num_gconv_layers,
                num_rels=self.num_rels,
                regularizer="bdd",
                num_bases=50 if graph_name == "GDELT" else 100,
                dropout=dropout,
                activation=activation,
                layer_norm=False
            )
        elif graph_conv == 'KGT':
            self.graph_conv = KGTransformer(
                in_dim, hid_dim, self.num_heads, hid_dim,
                n_layers=num_gconv_layers,
                num_nodes=self.num_node_types,
                num_rels=self.num_rels,
                layer_norm=True,
                dropout=dropout
            )
        else:
            raise ValueError(f"Invalid graph conv: {graph_conv}")
        
        # RNN input dimension
        structural_rnn_in_dim = hid_dim
        if self.add_entity_emb:
            structural_rnn_in_dim += hid_dim
        
        # RNN for temporal dynamics
        if rnn == "GRU":
            self.rnn_structural = nn.GRU(
                input_size=structural_rnn_in_dim,
                hidden_size=hid_dim,
                num_layers=num_rnn_layers,
                batch_first=True,
                dropout=0.0
            )
        elif rnn == "RNN":
            self.rnn_structural = nn.RNN(
                input_size=structural_rnn_in_dim,
                hidden_size=hid_dim,
                num_layers=num_rnn_layers,
                batch_first=True,
                dropout=0.0
            )
        else:
            raise ValueError(f"Invalid rnn: {rnn}")
        
        self.dropout = nn.Dropout(dropout)
        self.encoder_mode = graph_conv
    
    def forward(self, data, dynamic_entity_emb, static_entity_emb, batch_node_indices=None):
        """
        Forward pass: apply graph convolution + RNN
        
        Args:
            data: PyG Data object with edge_index, edge_type, node_type, etc.
            dynamic_entity_emb: Dynamic embeddings (RNN hidden states) of shape
                                (num_entities, num_rnn_layers, hid_dim)
            static_entity_emb: Static embeddings of shape (num_entities, in_dim)
            batch_node_indices: Indices of nodes to update (optional)
        
        Returns:
            Updated dynamic embeddings for batch nodes
        """
        device = data.edge_index.device
        
        if batch_node_indices is None:
            batch_node_indices = torch.arange(data.num_nodes, device=device)
        
        # Get static embeddings for nodes in the batch
        # Keep node_id on CPU for indexing, then move result to device
        batch_structural_static_entity_emb = static_entity_emb.structural[data.node_id.cpu()].to(device)
        
        # Apply graph convolution
        if self.encoder_mode == 'RGCN':
            # Compute edge normalization
            edge_norm = node_norm_to_edge_norm(data)
            conv_structural_static_emb = self.graph_conv(
                batch_structural_static_entity_emb,
                data.edge_index,
                data.edge_type.long(),
                edge_norm
            )
        elif self.encoder_mode == 'KGT':
            conv_structural_static_emb = self.graph_conv(
                batch_structural_static_entity_emb,
                data.edge_index,
                data.node_type.long(),
                data.edge_type.long()
            )
        else:
            raise ValueError(f"Unknown encoder mode: {self.encoder_mode}")
        
        # Prepare RNN input
        structural_rnn_input = [conv_structural_static_emb[batch_node_indices]]
        if self.add_entity_emb:
            structural_rnn_input.append(
                static_entity_emb.structural[data.node_id[batch_node_indices].cpu()].to(device)
            )
        structural_rnn_input = torch.cat(structural_rnn_input, dim=1).unsqueeze(1)
        
        # Get current RNN hidden state for batch nodes
        structural_dynamic = dynamic_entity_emb.structural[data.node_id[batch_node_indices].cpu()]
        structural_dynamic = structural_dynamic.to(device)
        
        # Update structural dynamics with RNN
        # Input: (batch, 1, rnn_in_dim), Hidden: (num_layers, batch, hidden_size)
        output, hn = self.rnn_structural(
            structural_rnn_input,
            structural_dynamic.transpose(0, 1).contiguous()
        )
        updated_structural_dynamic_entity_emb = hn.transpose(0, 1)
        
        return updated_structural_dynamic_entity_emb
    
    def extra_repr(self):
        return f"add_entity_emb={self.add_entity_emb}"


class GraphTemporalRNNConv(nn.Module):
    """
    Graph Temporal RNN Convolution
    
    Combines graph convolution with RNN for temporal dynamics,
    including time decay and bidirectional processing.
    
    Args:
        graph_conv (str): Type of graph convolution ('RGCN' or 'KGT')
        num_gconv_layers (int): Number of graph convolution layers
        rnn (str): Type of RNN ('GRU' or 'RNN')
        num_rnn_layers (int): Number of RNN layers
        in_dim (int): Input dimension
        hid_dim (int): Hidden dimension
        node_latest_event_time: Tensor tracking latest event times
        time_interval_transform: Function to transform time intervals
        num_nodes (int): Number of node types
        num_rels (int): Number of relation types
        num_node_types (int): Number of node types. Default: 1
        dropout (float): Dropout rate. Default: 0.2
        num_heads (int): Number of attention heads (for KGT). Default: 8
        activation (callable): Activation function. Default: None
        graph_name (str): Graph name (for specific configurations). Default: None
    """
    
    def __init__(
        self,
        graph_conv,
        num_gconv_layers,
        rnn,
        num_rnn_layers,
        in_dim,
        hid_dim,
        node_latest_event_time,
        time_interval_transform,
        num_nodes,
        num_rels,
        num_node_types=1,
        dropout=0.2,
        num_heads=8,
        activation=None,
        graph_name=None
    ):
        super().__init__()
        
        self.num_nodes = num_nodes
        self.num_node_types = num_node_types
        self.num_heads = num_heads
        self.num_rels = num_rels
        self.node_latest_event_time = node_latest_event_time
        self.time_interval_transform = time_interval_transform
        
        # Encoder module (graph convolution)
        self.encoder_mode = graph_conv
        if graph_conv == 'RGCN':
            self.graph_conv = RGCN(
                in_dim, hid_dim, hid_dim,
                n_layers=num_gconv_layers,
                num_rels=self.num_rels,
                regularizer="bdd",
                num_bases=50 if graph_name == "GDELT" else 100,
                dropout=dropout,
                activation=activation,
                layer_norm=False
            )
        elif graph_conv == 'KGT':
            self.graph_conv = KGTransformer(
                in_dim, hid_dim, self.num_heads, hid_dim,
                n_layers=num_gconv_layers,
                num_nodes=self.num_node_types,
                num_rels=self.num_rels,
                layer_norm=True,
                dropout=dropout
            )
        else:
            raise ValueError(f"Invalid graph conv: {graph_conv}")
        
        # RNN for temporal dynamics
        temporal_rnn_in_dim = hid_dim
        if rnn == "GRU":
            self.rnn_temporal = nn.GRU(
                input_size=temporal_rnn_in_dim,
                hidden_size=hid_dim,
                num_layers=num_rnn_layers,
                batch_first=True,
                dropout=0.0
            )
        elif rnn == "RNN":
            self.rnn_temporal = nn.RNN(
                input_size=temporal_rnn_in_dim,
                hidden_size=hid_dim,
                num_layers=num_rnn_layers,
                batch_first=True,
                dropout=0.0
            )
        else:
            raise ValueError(f"Invalid rnn: {rnn}")
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, data, dynamic_entity_emb, static_entity_emb, batch_node_indices=None):
        """
        Forward pass: apply graph convolution + RNN with time decay
        
        Processes both forward and reverse graphs for bidirectional temporal dynamics.
        
        Args:
            data: PyG Data object with edge_index, edge_type, timestamps, etc.
            dynamic_entity_emb: Dynamic embeddings (RNN hidden states) of shape
                                (num_entities, num_rnn_layers, hid_dim, 2)
                                Last dim: [0]=recipient, [1]=sender
            static_entity_emb: Static embeddings of shape (num_entities, in_dim)
            batch_node_indices: Indices of nodes to update (optional)
        
        Returns:
            Updated dynamic embeddings for batch nodes
        """
        device = data.edge_index.device
        
        if batch_node_indices is None:
            batch_node_indices = torch.arange(data.num_nodes, device=device)
        
        # Get static embeddings
        batch_temporal_static_entity_emb = static_entity_emb.temporal[data.node_id.cpu()].to(device)
        
        # Compute inter-event times and edge normalization
        batch_sparse_inter_event_times = get_sparse_inter_event_times(
            data, self.node_latest_event_time[..., 0]
        )
        edge_norm = (1 / self.time_interval_transform(batch_sparse_inter_event_times).clamp(min=1e-10)).clamp(max=10.0)
        
        # Forward graph convolution
        if self.encoder_mode == 'RGCN':
            batch_conv_temporal_static_emb = self.graph_conv(
                batch_temporal_static_entity_emb,
                data.edge_index,
                data.edge_type.long(),
                edge_norm
            )
        elif self.encoder_mode == 'KGT':
            batch_conv_temporal_static_emb = self.graph_conv(
                batch_temporal_static_entity_emb,
                data.edge_index,
                data.node_type.long(),
                data.edge_type.long(),
                norm=None
            )
        else:
            raise ValueError(f"Invalid encoder mode: {self.encoder_mode}")
        
        temporal_rnn_input_batch = batch_conv_temporal_static_emb[batch_node_indices].unsqueeze(1)
        
        # Reverse graph for bidirectional processing
        rev_edge_index = torch.stack([data.edge_index[1], data.edge_index[0]], dim=0)
        
        # Compute reverse inter-event times
        rev_batch_sparse_inter_event_times = get_sparse_inter_event_times_reverse(
            data, self.node_latest_event_time[..., 1]
        )
        rev_edge_norm = (1 / self.time_interval_transform(rev_batch_sparse_inter_event_times).clamp(min=1e-10)).clamp(max=10.0)
        
        # Reverse graph convolution
        if self.encoder_mode == 'RGCN':
            rev_batch_conv_temporal_static_emb = self.graph_conv(
                batch_temporal_static_entity_emb,
                rev_edge_index,
                data.edge_type.long(),
                rev_edge_norm
            )
        elif self.encoder_mode == 'KGT':
            rev_batch_conv_temporal_static_emb = self.graph_conv(
                batch_temporal_static_entity_emb,
                rev_edge_index,
                data.node_type.long(),
                data.edge_type.long()
            )
        else:
            raise ValueError(f"Invalid encoder mode: {self.encoder_mode}")
        
        temporal_rnn_input_rev_batch = rev_batch_conv_temporal_static_emb[batch_node_indices].unsqueeze(1)
        
        # Get current RNN hidden states
        temporal_dynamic = dynamic_entity_emb.temporal[data.node_id[batch_node_indices].cpu()].to(device)
        temporal_dynamic_batch = temporal_dynamic[..., 0]  # dynamics as recipient
        temporal_dynamic_rev_batch = temporal_dynamic[..., 1]  # dynamics as sender
        
        # Update temporal dynamics with RNN (forward)
        output, hn = self.rnn_temporal(
            temporal_rnn_input_batch,
            temporal_dynamic_batch.transpose(0, 1).contiguous()
        )
        updated_temporal_dynamic_batch = hn.transpose(0, 1)
        
        # Update temporal dynamics with RNN (reverse)
        output, hn = self.rnn_temporal(
            temporal_rnn_input_rev_batch,
            temporal_dynamic_rev_batch.transpose(0, 1).contiguous()
        )
        updated_temporal_dynamic_rev_batch = hn.transpose(0, 1)
        
        # Combine forward and reverse dynamics
        updated_temporal_dynamic_entity_emb = torch.cat([
            updated_temporal_dynamic_batch.unsqueeze(-1),
            updated_temporal_dynamic_rev_batch.unsqueeze(-1)
        ], dim=-1)
        
        return updated_temporal_dynamic_entity_emb


class RelationRNN(nn.Module):
    """
    Relation RNN for updating relation embeddings
    
    Aggregates entity embeddings by relation and updates relation hidden states.
    
    Args:
        rnn (str): Type of RNN ('GRU' or 'RNN')
        num_rnn_layers (int): Number of RNN layers
        rnn_in_dim (int): RNN input dimension
        rnn_hid_dim (int): RNN hidden dimension
        num_rels (int): Number of relations
        dropout (float): Dropout rate. Default: 0.0
    """
    
    def __init__(self, rnn, num_rnn_layers, rnn_in_dim, rnn_hid_dim, num_rels, dropout=0.0):
        super().__init__()
        
        if num_rnn_layers == 1:
            dropout = 0.0
        
        # RNN layer
        if rnn == "GRU":
            self.rnn_relation = nn.GRU(
                input_size=rnn_in_dim,
                hidden_size=rnn_hid_dim,
                num_layers=num_rnn_layers,
                batch_first=True,
                dropout=dropout
            )
        elif rnn == "RNN":
            self.rnn_relation = nn.RNN(
                input_size=rnn_in_dim,
                hidden_size=rnn_hid_dim,
                num_layers=num_rnn_layers,
                batch_first=True,
                dropout=dropout
            )
        else:
            raise ValueError(f"Invalid rnn: {rnn}")
    
    def forward(self, data, dynamic_relation_emb, static_entity_emb, device):
        """
        Forward pass: aggregate entities by relation and update relation RNN states
        
        Args:
            data: PyG Data object with edge_index, edge_type
            dynamic_relation_emb: Dynamic relation embeddings (RNN hidden states)
                                  Shape: (num_rels, num_rnn_layers, hid_dim, 2)
            static_entity_emb: Static entity embeddings
            device: Device to run computation on
        
        Returns:
            Updated dynamic relation embeddings for relations in the batch
        """
        edge_index = data.edge_index
        edge_type = data.edge_type.long()
        
        batch_src = edge_index[0]
        batch_dst = edge_index[1]
        
        batch_src_nid = data.node_id[batch_src.cpu().long()].long()
        batch_dst_nid = data.node_id[batch_dst.cpu().long()].long()
        
        # Aggregate entity embeddings by relation
        # Move edge_type to device for scatter_mean
        edge_type_device = edge_type.to(device)
        batch_src_emb_avg_by_rel = scatter_mean(
            static_entity_emb[batch_src_nid.cpu()].to(device).transpose(0, 1),
            edge_type_device
        ).transpose(0, 1)
        
        batch_dst_emb_avg_by_rel = scatter_mean(
            static_entity_emb[batch_dst_nid.cpu()].to(device).transpose(0, 1),
            edge_type_device
        ).transpose(0, 1)
        
        # Filter out non-existent relations
        batch_uniq_rel = torch.unique(edge_type, sorted=True).cpu()  # Keep on CPU for indexing
        batch_src_emb_avg_by_rel = batch_src_emb_avg_by_rel[batch_uniq_rel]
        batch_dst_emb_avg_by_rel = batch_dst_emb_avg_by_rel[batch_uniq_rel]
        
        # Get current RNN hidden states
        batch_dynamic_relation_emb = dynamic_relation_emb[batch_uniq_rel]
        batch_src_dynamic_relation_emb = batch_dynamic_relation_emb[..., 0].to(device)
        batch_dst_dynamic_relation_emb = batch_dynamic_relation_emb[..., 1].to(device)
        
        # Update source relation dynamics
        output, hn = self.rnn_relation(
            batch_src_emb_avg_by_rel.unsqueeze(1),
            batch_src_dynamic_relation_emb.transpose(0, 1).contiguous()
        )
        updated_batch_src_dynamic_relation_emb = hn.transpose(0, 1)
        
        # Update destination relation dynamics
        output, hn = self.rnn_relation(
            batch_dst_emb_avg_by_rel.unsqueeze(1),
            batch_dst_dynamic_relation_emb.transpose(0, 1).contiguous()
        )
        updated_batch_dst_dynamic_relation_emb = hn.transpose(0, 1)
        
        # Combine source and destination dynamics
        updated_batch_dynamic_relation_emb = torch.cat([
            updated_batch_src_dynamic_relation_emb.unsqueeze(-1),
            updated_batch_dst_dynamic_relation_emb.unsqueeze(-1)
        ], dim=-1)
        
        return updated_batch_dynamic_relation_emb


# Helper functions

def node_norm_to_edge_norm(data):
    """
    Compute edge normalization from node degrees (for RGCN).
    
    Args:
        data: PyG Data object with edge_index
    
    Returns:
        Edge normalization weights
    """
    edge_index = data.edge_index
    num_nodes = data.num_nodes
    
    # Compute node degrees
    row, col = edge_index
    deg = torch.zeros(num_nodes, device=edge_index.device)
    deg.scatter_add_(0, col, torch.ones(edge_index.size(1), device=edge_index.device))
    
    # Edge normalization: 1 / sqrt(deg[src] * deg[dst])
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    
    norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
    return norm


def get_sparse_inter_event_times(data, node_latest_event_time):
    """
    Compute inter-event times for edges (forward direction).
    
    Args:
        data: PyG Data object with edge_index, timestamps, node_id
        node_latest_event_time: Latest event time for each node pair
    
    Returns:
        Inter-event times for each edge
    """
    device = data.edge_index.device
    node_id = data.node_id.cpu()  # Keep on CPU for indexing
    batch_latest_event_time = node_latest_event_time[node_id]
    
    src, dst = data.edge_index
    batch_sparse_latest_event_times = batch_latest_event_time[dst.cpu().long(), node_id[src.cpu().long()]].to(device)
    
    return data.timestamps - batch_sparse_latest_event_times


def get_sparse_inter_event_times_reverse(data, node_latest_event_time):
    """
    Compute inter-event times for edges (reverse direction).
    
    Args:
        data: PyG Data object with edge_index, timestamps, node_id
        node_latest_event_time: Latest event time for each node pair (reverse)
    
    Returns:
        Inter-event times for each edge (reverse direction)
    """
    device = data.edge_index.device
    node_id = data.node_id.cpu()  # Keep on CPU for indexing
    batch_latest_event_time = node_latest_event_time[node_id]
    
    src, dst = data.edge_index
    # For reverse: swap src and dst
    batch_sparse_latest_event_times = batch_latest_event_time[src.cpu().long(), node_id[dst.cpu().long()]].to(device)
    
    return data.timestamps - batch_sparse_latest_event_times

