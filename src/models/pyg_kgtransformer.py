"""
Complete KGTransformer Model (PyG Implementation)
Port of DGL's DynamicKGEngine to PyG
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple

from .pyg_modules.embedding_updater import (
    GraphStructuralRNNConv,
    GraphTemporalRNNConv,
    RelationRNN
)
from .pyg_modules.rgcn import RGCN
from .pyg_modules.graph_transformer import KGTransformer


# Multi-aspect embedding structure
MultiAspectEmbedding = namedtuple('MultiAspectEmbedding', ['structural', 'temporal'], defaults=[None, None])


class ConfigArgs:
    """Configuration for KGTransformer"""
    def __init__(self):
        # Dataset
        self.graph = "FinDKG"
        self.num_node_types = 12  # FinDKG has 12 entity types
        
        # Embedding dimensions
        self.static_entity_embed_dim = 200
        self.structural_dynamic_entity_embed_dim = 200
        self.temporal_dynamic_entity_embed_dim = 200
        self.rel_embed_dim = 200
        
        # Model architecture
        self.num_gconv_layers = 2
        self.num_rnn_layers = 1
        self.num_attn_heads = 8
        self.dropout = 0.2
        
        # Graph convolution types
        self.embedding_updater_structural_gconv = 'KGT+RNN'  # or 'RGCN+RNN'
        self.embedding_updater_temporal_gconv = 'KGT+RNN'  # or 'RGCN+RNN'
        self.combiner_gconv = None
        
        # Combining mode
        self.static_dynamic_combine_mode = 'concat'
        
        # Activation
        self.embedding_updater_activation = F.tanh
        self.combiner_activation = F.tanh
        
        # Training (matching DGL train_DKG_run.py)
        self.lr = 0.0005  # DGL uses 0.0005
        self.weight_decay = 0.00001
        self.epochs = 150  # DGL trains for 150 epochs
        self.early_stop = True  # DGL uses early stopping
        self.patience = 10  # DGL patience
        self.early_stop_criterion = 'MRR'  # DGL criterion
        
        # Other
        self.inter_event_dtype = torch.float32
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.seed = 41  # DGL uses seed 41


def get_embedding(num_items, dims, zero_init=False):
    """
    Create embedding tensor.
    
    Args:
        num_items: Number of items (entities or relations)
        dims: Dimension(s) - int or list of ints
        zero_init: Whether to initialize with zeros (True) or Xavier (False)
    
    Returns:
        Embedding tensor
    """
    if isinstance(dims, int):
        shape = (num_items, dims)
    else:
        shape = tuple([num_items] + list(dims))
    
    emb = nn.Parameter(torch.Tensor(*shape))
    if zero_init:
        nn.init.zeros_(emb)
    else:
        nn.init.xavier_uniform_(emb)
    
    return emb


class TimeIntervalTransform:
    """Transform time intervals (log transform)"""
    def __init__(self, log_transform=True):
        self.log_transform = log_transform
    
    def __call__(self, intervals):
        if self.log_transform:
            return torch.log(intervals + 1)
        return intervals


class EmbeddingUpdater(nn.Module):
    """
    Embedding Updater: Updates dynamic entity and relation embeddings
    
    Combines:
    - GraphStructuralRNNConv: Graph conv + RNN for structural dynamics
    - GraphTemporalRNNConv: Graph conv + RNN for temporal dynamics
    - RelationRNN: RNN for relation dynamics
    
    Args:
        num_nodes (int): Number of entities
        in_dim (int): Static embedding dimension
        structural_hid_dim (int): Structural hidden dimension
        temporal_hid_dim (int): Temporal hidden dimension
        node_latest_event_time: Tensor tracking latest event times
        num_rels (int): Number of relations
        rel_embed_dim (int): Relation embedding dimension
        num_node_types (int): Number of node types
        graph_structural_conv (str): Type of structural graph conv ('KGT+RNN' or 'RGCN+RNN')
        graph_temporal_conv (str): Type of temporal graph conv ('KGT+RNN' or 'RGCN+RNN')
        num_gconv_layers (int): Number of graph conv layers
        num_rnn_layers (int): Number of RNN layers
        time_interval_transform: Time interval transformation function
        dropout (float): Dropout rate
        activation: Activation function
        graph_name (str): Graph name (for specific configurations)
    """
    
    def __init__(
        self,
        num_nodes,
        in_dim,
        structural_hid_dim,
        temporal_hid_dim,
        node_latest_event_time,
        num_rels,
        rel_embed_dim,
        num_node_types=1,
        graph_structural_conv='KGT+RNN',
        graph_temporal_conv='KGT+RNN',
        num_gconv_layers=2,
        num_rnn_layers=1,
        num_heads=8,
        time_interval_transform=None,
        dropout=0.0,
        activation=None,
        graph_name=None
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_rels = num_rels
        self.in_dim = in_dim
        self.structural_hid_dim = structural_hid_dim
        self.temporal_hid_dim = temporal_hid_dim
        self.node_latest_event_time = node_latest_event_time
        
        # Initialize structural graph conv + RNN
        if graph_structural_conv in ['KGT+GRU', 'KGT+RNN', 'RGCN+GRU', 'RGCN+RNN']:
            gconv, rnn = graph_structural_conv.split("+")
            self.graph_structural_conv = GraphStructuralRNNConv(
                gconv, num_gconv_layers, rnn, num_rnn_layers,
                in_dim, structural_hid_dim, num_nodes, num_rels, rel_embed_dim,
                num_node_types=num_node_types,
                num_heads=num_heads,
                dropout=dropout, activation=activation, graph_name=graph_name
            )
        elif graph_structural_conv is None:
            self.graph_structural_conv = None
        else:
            raise ValueError(f"Invalid graph structural conv: {graph_structural_conv}")
        
        # Initialize temporal graph conv + RNN
        if graph_temporal_conv in ['KGT+GRU', 'KGT+RNN', 'RGCN+GRU', 'RGCN+RNN']:
            gconv, rnn = graph_temporal_conv.split("+")
            self.graph_temporal_conv = GraphTemporalRNNConv(
                gconv, num_gconv_layers, rnn, num_rnn_layers,
                in_dim, temporal_hid_dim,
                node_latest_event_time, time_interval_transform,
                num_nodes, num_rels,
                num_node_types=num_node_types,
                num_heads=num_heads,
                dropout=dropout, activation=activation, graph_name=graph_name
            )
        elif graph_temporal_conv is None:
            self.graph_temporal_conv = None
        else:
            raise ValueError(f"Invalid graph temporal conv: {graph_temporal_conv}")
        
        # Initialize relation RNNs
        self.structural_relation_rnn = RelationRNN("RNN", num_rnn_layers, in_dim, rel_embed_dim, num_rels, dropout)
        self.temporal_relation_rnn = RelationRNN("RNN", num_rnn_layers, in_dim, rel_embed_dim, num_rels, dropout)
    
    def forward(self, batch_data, static_entity_emb, dynamic_entity_emb, dynamic_relation_emb, device, batch_node_indices=None):
        """
        Update embeddings for batch.
        
        Args:
            batch_data: PyG Data object for current batch
            static_entity_emb: MultiAspectEmbedding with static embeddings
            dynamic_entity_emb: MultiAspectEmbedding with RNN hidden states
            dynamic_relation_emb: MultiAspectEmbedding with relation RNN hidden states
            device: Device to run computation on
            batch_node_indices: Indices of nodes to update (optional)
        
        Returns:
            Tuple of (updated_dynamic_entity_emb, updated_dynamic_relation_emb)
        """
        # Ensure all dynamic embeddings are on CPU initially
        assert all([emb.device == torch.device('cpu') for emb in dynamic_entity_emb]), \
            [emb.device for emb in dynamic_entity_emb]
        
        batch_data = batch_data.to(device)
        if batch_node_indices is None:
            batch_node_indices = torch.arange(batch_data.num_nodes, device=device)
        
        # Update structural dynamic entity embeddings
        if self.graph_structural_conv is None:
            batch_structural_dynamic_entity_emb = None
        else:
            batch_structural_dynamic_entity_emb = self.graph_structural_conv(
                batch_data, dynamic_entity_emb, static_entity_emb, batch_node_indices
            )
        
        # Update temporal dynamic entity embeddings
        if self.graph_temporal_conv is None:
            batch_temporal_dynamic_entity_emb = None
        else:
            batch_temporal_dynamic_entity_emb = self.graph_temporal_conv(
                batch_data, dynamic_entity_emb, static_entity_emb, batch_node_indices
            )
        
        # Update structural dynamic relation embeddings
        batch_structural_dynamic_relation_emb = self.structural_relation_rnn.forward(
            batch_data, dynamic_relation_emb.structural, static_entity_emb.structural, device
        )
        
        # Update temporal dynamic relation embeddings
        batch_temporal_dynamic_relation_emb = self.temporal_relation_rnn.forward(
            batch_data, dynamic_relation_emb.temporal, static_entity_emb.temporal, device
        )
        
        # Update dynamic entity embeddings (move back to CPU)
        updated_structural = dynamic_entity_emb.structural
        if batch_structural_dynamic_entity_emb is not None:
            updated_structural = dynamic_entity_emb.structural.clone()
            updated_structural[batch_data.node_id[batch_node_indices].cpu().long()] = batch_structural_dynamic_entity_emb.cpu()
        
        updated_temporal = dynamic_entity_emb.temporal
        if batch_temporal_dynamic_entity_emb is not None:
            updated_temporal = dynamic_entity_emb.temporal.clone()
            updated_temporal[batch_data.node_id[batch_node_indices].cpu().long()] = batch_temporal_dynamic_entity_emb.cpu()
        
        updated_dynamic_entity_emb = MultiAspectEmbedding(structural=updated_structural, temporal=updated_temporal)
        
        # Update dynamic relation embeddings
        batch_uniq_rel = torch.unique(batch_data.edge_type, sorted=True).cpu().long()
        
        updated_structural = dynamic_relation_emb.structural
        if batch_structural_dynamic_relation_emb is not None:
            updated_structural = dynamic_relation_emb.structural.clone()
            updated_structural[batch_uniq_rel] = batch_structural_dynamic_relation_emb.cpu()
        
        updated_temporal = dynamic_relation_emb.temporal
        if batch_temporal_dynamic_relation_emb is not None:
            updated_temporal = dynamic_relation_emb.temporal.clone()
            updated_temporal[batch_uniq_rel] = batch_temporal_dynamic_relation_emb.cpu()
        
        updated_dynamic_relation_emb = MultiAspectEmbedding(structural=updated_structural, temporal=updated_temporal)
        
        # Detach to prevent backprop through time across batches
        updated_dynamic_entity_emb = MultiAspectEmbedding(
            structural=updated_dynamic_entity_emb.structural.detach(),
            temporal=updated_dynamic_entity_emb.temporal.detach()
        )
        updated_dynamic_relation_emb = MultiAspectEmbedding(
            structural=updated_dynamic_relation_emb.structural.detach(),
            temporal=updated_dynamic_relation_emb.temporal.detach()
        )
        
        return updated_dynamic_entity_emb, updated_dynamic_relation_emb


class Combiner(nn.Module):
    """
    Combines static and dynamic embeddings
    
    Args:
        static_emb_dim: Static embedding dimension
        dynamic_emb_dim: Dynamic embedding dimension
        static_dynamic_combine_mode: Mode for combining ('concat', 'static_only', 'dynamic_only')
        graph_conv: Optional graph convolution to apply before combining
        num_rels: Number of relations (if graph_conv is RGCN)
        dropout: Dropout rate
        activation: Activation function
        num_gconv_layers: Number of graph convolution layers
    """
    
    def __init__(
        self,
        static_emb_dim,
        dynamic_emb_dim,
        static_dynamic_combine_mode,
        graph_conv=None,
        num_rels=None,
        dropout=0.0,
        activation=None,
        num_gconv_layers=1
    ):
        super().__init__()
        self.static_emb_dim = static_emb_dim
        self.dynamic_emb_dim = dynamic_emb_dim
        self.static_dynamic_combine_mode = static_dynamic_combine_mode
        
        # Determine combined embedding dimension
        if static_dynamic_combine_mode == "concat":
            self.combined_emb_dim = static_emb_dim + dynamic_emb_dim
        elif static_dynamic_combine_mode == "static_only":
            self.combined_emb_dim = static_emb_dim
        elif static_dynamic_combine_mode == "dynamic_only":
            self.combined_emb_dim = dynamic_emb_dim
        else:
            raise ValueError(f"Invalid combiner mode: {static_dynamic_combine_mode}")
        
        # Optional graph convolution before combining
        self.graph_conv_static = None
        self.graph_conv_dynamic = None
        if graph_conv is not None and graph_conv == 'RGCN':
            if "static_only" not in static_dynamic_combine_mode:
                self.graph_conv_static = RGCN(
                    static_emb_dim, static_emb_dim, static_emb_dim,
                    n_layers=num_gconv_layers, num_rels=num_rels, regularizer="bdd",
                    num_bases=100, dropout=dropout, activation=activation
                )
            if "dynamic_only" not in static_dynamic_combine_mode:
                self.graph_conv_dynamic = RGCN(
                    dynamic_emb_dim, dynamic_emb_dim, dynamic_emb_dim,
                    n_layers=num_gconv_layers, num_rels=num_rels, regularizer="bdd",
                    num_bases=100, dropout=dropout, activation=activation
                )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, static_emb, dynamic_emb, data=None):
        """
        Combine static and dynamic embeddings.
        
        Args:
            static_emb: Static embeddings
            dynamic_emb: Dynamic embeddings
            data: PyG Data object (if graph_conv is used)
        
        Returns:
            Combined embeddings
        """
        # Apply optional graph convolution
        if self.graph_conv_static is not None and data is not None:
            from .pyg_modules.embedding_updater import node_norm_to_edge_norm
            edge_norm = node_norm_to_edge_norm(data)
            static_emb = self.graph_conv_static(static_emb, data.edge_index, data.edge_type, edge_norm)
        
        if self.graph_conv_dynamic is not None and data is not None:
            from .pyg_modules.embedding_updater import node_norm_to_edge_norm
            edge_norm = node_norm_to_edge_norm(data)
            dynamic_emb = self.graph_conv_dynamic(dynamic_emb, data.edge_index, data.edge_type, edge_norm)
        
        # Combine embeddings
        if self.static_dynamic_combine_mode == "concat":
            combined_emb = torch.cat([self.dropout(static_emb), dynamic_emb], dim=1)
        elif self.static_dynamic_combine_mode == "static_only":
            combined_emb = self.dropout(static_emb)
        elif self.static_dynamic_combine_mode == "dynamic_only":
            combined_emb = dynamic_emb
        
        return combined_emb


class GraphReadout(nn.Module):
    """
    Graph Readout Module - Computes graph-level embeddings
    
    Args:
        combiner: Combiner module
        readout_op: Readout operation ('max', 'mean', 'sum')
        readout_node_type: Which embeddings to use ('combined', 'static', 'dynamic')
    """
    def __init__(self, combiner, readout_op='max', readout_node_type='combined'):
        super().__init__()
        
        self.combiner = combiner
        self.readout_node_type = readout_node_type
        
        if readout_node_type == "combined":
            self.node_emb_dim = combiner.combined_emb_dim
        elif readout_node_type == "static":
            self.node_emb_dim = combiner.static_emb_dim
        elif readout_node_type == "dynamic":
            self.node_emb_dim = combiner.dynamic_emb_dim
        else:
            raise ValueError(f"Invalid type: {readout_node_type}")
        
        self.readout_op = readout_op
        if readout_op in ['max', 'min', 'mean', 'sum']:
            self.graph_emb_dim = self.node_emb_dim
        else:
            raise ValueError(f"Invalid readout: {readout_op}")
    
    def forward(self, combined_emb, static_emb, dynamic_emb, batch=None):
        """
        Compute graph-level embedding
        
        Args:
            combined_emb: Combined node embeddings [num_nodes, combined_dim]
            static_emb: Static node embeddings [num_nodes, static_dim]
            dynamic_emb: Dynamic node embeddings [num_nodes, dynamic_dim]
            batch: Batch assignment for nodes (None means single graph)
        
        Returns:
            graph_emb: Graph-level embedding [1, graph_emb_dim] or [batch_size, graph_emb_dim]
        """
        # Select which embeddings to use
        if self.readout_node_type == "combined":
            node_emb = combined_emb
        elif self.readout_node_type == "static":
            node_emb = static_emb
        elif self.readout_node_type == "dynamic":
            node_emb = dynamic_emb
        
        # Apply readout operation (single graph - aggregate all nodes)
        if self.readout_op == 'max':
            graph_emb = node_emb.max(dim=0, keepdim=True)[0]
        elif self.readout_op == 'mean':
            graph_emb = node_emb.mean(dim=0, keepdim=True)
        elif self.readout_op == 'sum':
            graph_emb = node_emb.sum(dim=0, keepdim=True)
        elif self.readout_op == 'min':
            graph_emb = node_emb.min(dim=0, keepdim=True)[0]
        
        return graph_emb


class EdgeModel(nn.Module):
    """
    Edge Model for link prediction with multi-task learning
    
    Predicts head, relation, and tail entities for temporal triplets.
    
    Args:
        num_entities: Number of entities
        num_rels: Number of relations
        rel_embed_dim: Relation embedding dimension
        combiner: Combiner module
        dropout: Dropout rate
        graph_readout_op: Graph readout operation
    """
    
    def __init__(self, num_entities, num_rels, rel_embed_dim, combiner, dropout=0.0, graph_readout_op='max'):
        super().__init__()
        
        self.num_entities = num_entities
        self.num_rels = num_rels
        self.rel_embed_dim = rel_embed_dim
        self.rel_embeds = get_embedding(num_rels, rel_embed_dim)
        self.combiner = combiner
        self.combined_emb_dim = combiner.combined_emb_dim
        
        # Graph readout module
        self.graph_readout = GraphReadout(combiner, readout_op=graph_readout_op, readout_node_type='combined')
        graph_emb_dim = self.graph_readout.graph_emb_dim
        
        # Head prediction: graph_emb -> head entity
        self.transform_head = nn.Sequential(
            nn.Linear(graph_emb_dim, 4 * graph_emb_dim),
            nn.Tanh(),
            nn.Linear(4 * graph_emb_dim, num_entities)
        )
        
        # Relation prediction: combined_emb + graph_emb -> relation
        node_graph_emb_dim = self.combined_emb_dim + graph_emb_dim
        self.transform_rel = nn.Sequential(
            nn.Linear(node_graph_emb_dim, node_graph_emb_dim),
            nn.Tanh(),
            nn.Linear(node_graph_emb_dim, num_rels)
        )
        
        # Tail prediction: combined_emb + graph_emb + rel_emb -> tail entity
        node_graph_rel_emb_dim = self.combined_emb_dim + graph_emb_dim + rel_embed_dim * 2
        self.transform_tail = nn.Sequential(
            nn.Linear(node_graph_rel_emb_dim, 2 * node_graph_rel_emb_dim),
            nn.Tanh(),
            nn.Linear(2 * node_graph_rel_emb_dim, num_entities)
        )
        
        self.dropout = nn.Dropout(dropout)
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, data, combined_emb, static_emb, dynamic_emb, dynamic_relation_emb, target_heads, target_rels, target_tails):
        """
        Forward pass for multi-task link prediction.
        
        Args:
            data: PyG Data object
            combined_emb: Combined entity embeddings [num_nodes, combined_dim]
            static_emb: Static entity embeddings [num_nodes, static_dim]
            dynamic_emb: Dynamic entity embeddings [num_nodes, dynamic_dim]
            dynamic_relation_emb: Dynamic relation embeddings
            target_heads: Target head entities (local indices)
            target_rels: Target relations
            target_tails: Target tail entities (local indices)
        
        Returns:
            Total loss and predictions (head_pred, rel_pred, tail_pred)
        """
        edge_index = data.edge_index
        edge_type = data.edge_type.long()
        
        # Compute graph-level embedding
        graph_emb = self.graph_readout(combined_emb, static_emb, dynamic_emb)
        
        # Get head embeddings
        edge_head_emb = combined_emb[edge_index[0]]
        
        # Get relation embeddings (static + dynamic)
        edge_static_rel_embeds = self.rel_embeds[edge_type]
        
        # Dynamic relation embeddings: already extracted last hidden state [:, -1, :, :]
        # Shape: [num_relations, rel_embed_dim, 2] where 2 is bidirectional
        # Select destination context (index 1 - relation-dest as in DGL)
        dynamic_rel_emb = dynamic_relation_emb[:, :, 1]  # [num_relations, rel_embed_dim]
        edge_dynamic_rel_embeds = dynamic_rel_emb[edge_type]
        
        edge_rel_embeds = torch.cat((edge_static_rel_embeds, edge_dynamic_rel_embeds), dim=1)
        
        # 1. Head prediction
        graph_emb_repeat = graph_emb.repeat(len(edge_head_emb), 1)
        head_emb = self.dropout(graph_emb_repeat)
        head_pred = self.transform_head(head_emb)
        
        # Map local head indices to global entity IDs
        global_head_ids = data.node_id[target_heads].long()
        log_prob_head = -self.criterion(head_pred, global_head_ids)
        
        # 2. Relation prediction
        rel_emb = torch.cat((edge_head_emb, graph_emb_repeat), dim=1)
        rel_emb = self.dropout(rel_emb)
        rel_pred = self.transform_rel(rel_emb)
        log_prob_rel = -self.criterion(rel_pred, target_rels.long())
        
        # 3. Tail prediction
        tail_emb = torch.cat((edge_head_emb, graph_emb_repeat, edge_rel_embeds), dim=1)
        tail_emb = self.dropout(tail_emb)
        tail_pred = self.transform_tail(tail_emb)
        
        # Map local tail indices to global entity IDs
        global_tail_ids = data.node_id[target_tails].long()
        log_prob_tail = -self.criterion(tail_pred, global_tail_ids)
        
        # Multi-task loss (same weights as DGL: tail + 0.2*rel + 0.1*head)
        log_prob = log_prob_tail + 0.2 * log_prob_rel + 0.1 * log_prob_head
        
        return log_prob, (head_pred, rel_pred, tail_pred)


class KGTransformerPyG(nn.Module):
    """
    Complete KGTransformer Model (PyG Implementation)
    
    Args:
        num_entities: Number of entities
        num_relations: Number of relations
        args: Configuration object with model hyperparameters
    """
    
    def __init__(self, num_entities, num_relations, args):
        super().__init__()
        
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.args = args
        self.device = args.device
        
        # Initialize time interval transform
        self.time_interval_transform = TimeIntervalTransform(log_transform=True)
        
        # Initialize node latest event time tracker
        self.node_latest_event_time = torch.zeros(
            num_entities, num_entities + 1, 2,
            dtype=args.inter_event_dtype
        )
        
        # Initialize embeddings
        self.static_entity_embeds, self.dynamic_entity_embeds, self.dynamic_relation_embeds = \
            self.init_embeddings()
        
        # Initialize embedding updater
        self.embedding_updater = EmbeddingUpdater(
            num_entities,
            args.static_entity_embed_dim,
            args.structural_dynamic_entity_embed_dim,
            args.temporal_dynamic_entity_embed_dim,
            self.node_latest_event_time,
            num_relations,
            args.rel_embed_dim,
            num_node_types=args.num_node_types,
            graph_structural_conv=args.embedding_updater_structural_gconv,
            graph_temporal_conv=args.embedding_updater_temporal_gconv,
            num_gconv_layers=args.num_gconv_layers,
            num_rnn_layers=args.num_rnn_layers,
            num_heads=args.num_attn_heads,
            time_interval_transform=self.time_interval_transform,
            dropout=args.dropout,
            activation=args.embedding_updater_activation,
            graph_name=args.graph
        ).to(self.device)
        
        # Initialize combiner
        self.combiner = Combiner(
            args.static_entity_embed_dim,
            args.structural_dynamic_entity_embed_dim,
            args.static_dynamic_combine_mode,
            args.combiner_gconv,
            num_relations,
            args.dropout,
            args.combiner_activation
        ).to(self.device)
        
        # Initialize edge model
        self.edge_model = EdgeModel(
            num_entities,
            num_relations,
            args.rel_embed_dim,
            self.combiner,
            dropout=args.dropout
        ).to(self.device)
    
    def init_embeddings(self):
        """Initialize static and dynamic embeddings"""
        args = self.args
        
        static_entity_embeds = MultiAspectEmbedding(
            structural=get_embedding(self.num_entities, args.static_entity_embed_dim, zero_init=False),
            temporal=get_embedding(self.num_entities, args.static_entity_embed_dim, zero_init=False),
        )
        
        dynamic_entity_embeds = MultiAspectEmbedding(
            structural=get_embedding(self.num_entities, [args.num_rnn_layers, args.structural_dynamic_entity_embed_dim], zero_init=True),
            temporal=get_embedding(self.num_entities, [args.num_rnn_layers, args.temporal_dynamic_entity_embed_dim, 2], zero_init=True),
        )
        
        dynamic_relation_embeds = MultiAspectEmbedding(
            structural=get_embedding(self.num_relations, [args.num_rnn_layers, args.rel_embed_dim, 2], zero_init=True),
            temporal=get_embedding(self.num_relations, [args.num_rnn_layers, args.rel_embed_dim, 2], zero_init=True),
        )
        
        return static_entity_embeds, dynamic_entity_embeds, dynamic_relation_embeds
    
    def forward(self, batch_data):
        """
        Forward pass: update embeddings and predict links
        
        Args:
            batch_data: PyG Data object for current batch
        
        Returns:
            Loss and predictions
        """
        # Update embeddings
        self.dynamic_entity_embeds, self.dynamic_relation_embeds = self.embedding_updater(
            batch_data,
            self.static_entity_embeds,
            self.dynamic_entity_embeds,
            self.dynamic_relation_embeds,
            self.device
        )
        
        # Get combined embeddings for nodes in batch
        batch_data = batch_data.to(self.device)
        static_structural = self.static_entity_embeds.structural[batch_data.node_id.cpu()].to(self.device)
        # Use last hidden state from RNN (matching DGL)
        dynamic_structural = self.dynamic_entity_embeds.structural[batch_data.node_id.cpu(), -1, :].to(self.device)
        
        # Combiner combines static_structural + dynamic_structural
        combined_emb = self.combiner(static_structural, dynamic_structural, batch_data)
        
        # For graph readout, use the SAME embeddings that went into the combiner
        # (NOT averaged with temporal - DGL uses structural embeddings only for graph readout)
        static_emb = static_structural
        dynamic_emb = dynamic_structural
        
        # Link prediction with multi-task learning
        target_heads = batch_data.edge_index[0]
        target_rels = batch_data.edge_type
        target_tails = batch_data.edge_index[1]
        
        # Extract relation embeddings (last hidden state from structural RNN)
        # Shape: [num_relations, embed_dim, 2] where 2 is bidirectional (sender/receiver)
        dynamic_rel_emb = self.dynamic_relation_embeds.structural[:, -1, :, :].to(self.device)
        
        log_prob, (head_pred, rel_pred, tail_pred) = self.edge_model(
            batch_data, 
            combined_emb, 
            static_emb,
            dynamic_emb,
            dynamic_rel_emb,  # Pass structural embeddings (last hidden state)
            target_heads,
            target_rels,
            target_tails
        )
        
        return log_prob, tail_pred

