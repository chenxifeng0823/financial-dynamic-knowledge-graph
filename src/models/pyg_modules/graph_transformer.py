"""
GraphTransformer: Multi-head attention for knowledge graphs
Port of DGL's KGTransformer to PyG
"""

import math
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax

from .typed_linear import TypedLinear


class GraphTransformerConv(MessagePassing):
    """
    Knowledge Graph Transformer Layer
    
    A single layer of the Graph Transformer that applies multi-head attention
    with relation-specific transformations.
    
    The layer computes:
    1. Query, Key, Value projections (type-specific)
    2. Multi-head attention with relation-specific transformations
    3. Message passing and aggregation
    4. Residual connection with learnable skip parameter
    5. Optional layer normalization
    
    Args:
        in_size (int): Input feature size
        hid_size (int): Hidden size (must be divisible by num_heads)
        num_heads (int): Number of attention heads
        num_ntypes (int): Number of node types
        num_etypes (int): Number of edge/relation types
        dropout (float): Dropout rate. Default: 0.2
        use_norm (bool): Whether to use layer normalization. Default: True
    
    Shape:
        - Input: (N, in_size)
        - Node types: (N,)
        - Edge index: (2, E)
        - Edge types: (E,)
        - Output: (N, hid_size)
    """
    
    def __init__(
        self,
        in_size,
        hid_size,
        num_heads,
        num_ntypes,
        num_etypes,
        dropout=0.2,
        use_norm=True,
    ):
        super().__init__(aggr='add', node_dim=0)
        
        self.in_size = in_size
        self.hid_size = hid_size
        self.num_heads = num_heads
        self.head_size = self.hid_size // self.num_heads
        assert self.hid_size % self.num_heads == 0, \
            f"hid_size ({hid_size}) must be divisible by num_heads ({num_heads})"
        self.sqrt_d = math.sqrt(self.head_size)
        self.use_norm = use_norm
        self.num_ntypes = num_ntypes
        self.num_etypes = num_etypes
        
        # Type-specific Q, K, V projections
        self.linear_k = TypedLinear(in_size, self.head_size * num_heads, num_ntypes)
        self.linear_q = TypedLinear(in_size, self.head_size * num_heads, num_ntypes)
        self.linear_v = TypedLinear(in_size, self.head_size * num_heads, num_ntypes)
        
        # Linear projection for output (A-Linear in the paper)
        self.activation = nn.SiLU()
        self.linear_a = TypedLinear(
            self.head_size * num_heads, self.head_size * num_heads, num_ntypes
        )
        
        # Relation-specific parameters for attention and message
        self.relation_pri = nn.ParameterList([
            nn.Parameter(torch.ones(num_etypes)) for _ in range(num_heads)
        ])
        self.relation_att = nn.ModuleList([
            TypedLinear(self.head_size, self.head_size, num_etypes)
            for _ in range(num_heads)
        ])
        self.relation_msg = nn.ModuleList([
            TypedLinear(self.head_size, self.head_size, num_etypes)
            for _ in range(num_heads)
        ])
        
        # Residual connection with learnable skip parameter
        self.skip = nn.Parameter(torch.ones(num_ntypes))
        
        # Dropout and layer norm
        self.drop = nn.Dropout(dropout)
        if use_norm:
            self.norm = nn.LayerNorm(self.head_size * num_heads)
        
        # Residual projection if input and output sizes don't match
        if in_size != self.head_size * num_heads:
            self.residual_w = nn.Parameter(
                torch.Tensor(in_size, self.head_size * num_heads)
            )
            nn.init.xavier_uniform_(self.residual_w)
        else:
            self.register_parameter('residual_w', None)
    
    def forward(self, x, edge_index, ntype, etype, norm=None, presorted=False):
        """
        Forward pass of GraphTransformer layer.
        
        Args:
            x (Tensor): Node features of shape (N, in_size)
            edge_index (Tensor): Edge indices of shape (2, E)
            ntype (Tensor): Node types of shape (N,)
            etype (Tensor): Edge types of shape (E,)
            norm (Tensor, optional): Edge normalization of shape (E,)
            presorted (bool): Whether types are presorted (for efficiency). Default: False
        
        Returns:
            Tensor: Updated node features of shape (N, hid_size)
        """
        self.presorted = presorted
        
        # Compute Q, K, V with type-specific transformations
        # Shape: (N, num_heads, head_size)
        k = self.linear_k(x, ntype, presorted).view(-1, self.num_heads, self.head_size)
        q = self.linear_q(x, ntype, presorted).view(-1, self.num_heads, self.head_size)
        v = self.linear_v(x, ntype, presorted).view(-1, self.num_heads, self.head_size)
        
        # Message passing with multi-head attention
        out = self.propagate(
            edge_index,
            k=k, q=q, v=v,
            etype=etype,
            norm=norm,
            size=None
        )
        
        # out shape: (N, num_heads, head_size) -> (N, hid_size)
        out = out.view(-1, self.num_heads * self.head_size)
        
        # Target-specific aggregation (A-Linear)
        out = self.drop(self.activation(self.linear_a(out, ntype, presorted)))
        
        # Residual connection with learnable skip parameter
        alpha = torch.sigmoid(self.skip[ntype]).unsqueeze(-1)
        if x.shape != out.shape:
            x_residual = x @ self.residual_w
        else:
            x_residual = x
        out = out * alpha + x_residual * (1 - alpha)
        
        # Layer normalization
        if self.use_norm:
            out = self.norm(out)
        
        return out
    
    def message(self, k_j, q_i, v_j, etype, norm, index):
        """
        Compute messages for each edge.
        
        This is called by PyG's propagate() method.
        
        Args:
            k_j: Key from source node (E, num_heads, head_size)
            q_i: Query from destination node (E, num_heads, head_size)
            v_j: Value from source node (E, num_heads, head_size)
            etype: Edge types (E,)
            norm: Edge normalization (E,) or None
            index: Destination node indices for softmax (E,)
        
        Returns:
            Tuple of (attention, message) both of shape (E, num_heads, head_size)
        """
        # Compute attention scores for each head
        a_list = []
        m_list = []
        
        # Unbind heads for processing
        k_heads = torch.unbind(k_j, dim=1)  # num_heads x (E, head_size)
        q_heads = torch.unbind(q_i, dim=1)
        v_heads = torch.unbind(v_j, dim=1)
        
        for i in range(self.num_heads):
            # Apply relation-specific attention transformation
            # kw: (E, head_size)
            kw = self.relation_att[i](k_heads[i], etype, self.presorted)
            
            # Compute attention score: (kw * q).sum(-1) * priority / sqrt(d)
            # Shape: (E,)
            attn_score = (kw * q_heads[i]).sum(-1) * self.relation_pri[i][etype] / self.sqrt_d
            a_list.append(attn_score)
            
            # Apply relation-specific message transformation
            m_head = self.relation_msg[i](v_heads[i], etype, self.presorted)
            
            # Apply edge normalization if provided
            if norm is not None:
                m_head = m_head * norm.unsqueeze(-1)
            
            m_list.append(m_head)
        
        # Stack attention scores and messages
        # Shape: (E, num_heads)
        a = torch.stack(a_list, dim=1)
        
        # Apply softmax to attention scores (group by destination node)
        # Shape: (E, num_heads)
        alpha = softmax(a, index)
        
        # Stack messages: (E, num_heads, head_size)
        m = torch.stack(m_list, dim=1)
        
        # Apply attention weights to messages
        # Shape: (E, num_heads, head_size)
        return alpha.unsqueeze(-1) * m
    
    def update(self, aggr_out):
        """
        Update step after aggregation (optional, currently just returns the aggregated output).
        """
        return aggr_out


class KGTransformer(nn.Module):
    """
    Multi-layer Knowledge Graph Transformer
    
    Stacks multiple GraphTransformerConv layers with a final feed-forward layer.
    
    Args:
        in_dim (int): Input dimension
        hid_dim (int): Hidden dimension
        num_heads (int): Number of attention heads
        out_dim (int): Output dimension
        n_layers (int): Number of transformer layers
        num_nodes (int): Number of node types
        num_rels (int): Number of relation types
        dropout (float): Dropout rate. Default: 0.2
        layer_norm (bool): Whether to use layer normalization. Default: True
        low_mem (bool): Low memory mode (unused, kept for compatibility). Default: False
    
    Shape:
        - Input: (N, in_dim)
        - Edge index: (2, E)
        - Node types: (N,)
        - Edge types: (E,)
        - Output: (N, out_dim)
    """
    
    def __init__(
        self,
        in_dim,
        hid_dim,
        num_heads,
        out_dim,
        n_layers,
        num_nodes,
        num_rels,
        dropout=0.2,
        layer_norm=True,
        low_mem=False
    ):
        super().__init__()
        self.in_dim = in_dim
        self.num_heads = num_heads
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.num_nodes = num_nodes  # number of node types
        self.num_rels = num_rels
        self.dropout = dropout
        self.layer_norm = layer_norm
        self.activation = nn.SiLU()
        self.feed_forward = nn.Linear(self.hid_dim, self.out_dim)
        self.low_mem = low_mem
        
        # Stack transformer layers
        self.n_layers = n_layers
        assert self.n_layers >= 1, f"Number of layers must be >= 1, got {self.n_layers}"
        
        self.layers = nn.ModuleList()
        if self.n_layers == 1:
            self.layers.append(GraphTransformerConv(
                self.in_dim, self.hid_dim, self.num_heads,
                num_ntypes=self.num_nodes,
                num_etypes=self.num_rels,
                dropout=self.dropout,
                use_norm=self.layer_norm
            ))
        else:
            # First layer: in_dim -> hid_dim
            self.layers.append(GraphTransformerConv(
                self.in_dim, self.hid_dim, self.num_heads,
                num_ntypes=self.num_nodes,
                num_etypes=self.num_rels,
                dropout=self.dropout,
                use_norm=self.layer_norm
            ))
            # Middle layers: hid_dim -> hid_dim
            for i in range(1, self.n_layers):
                self.layers.append(GraphTransformerConv(
                    self.hid_dim, self.hid_dim, self.num_heads,
                    num_ntypes=self.num_nodes,
                    num_etypes=self.num_rels,
                    dropout=self.dropout,
                    use_norm=self.layer_norm
                ))
        
        assert self.n_layers == len(self.layers), \
            f"Expected {self.n_layers} layers, got {len(self.layers)}"
    
    def forward(self, x, edge_index, ntype, etype, norm=None, use_norm=True):
        """
        Forward computation through all transformer layers.
        
        Args:
            x (Tensor): Node features of shape (N, in_dim)
            edge_index (Tensor): Edge indices of shape (2, E)
            ntype (Tensor): Node types of shape (N,)
            etype (Tensor): Edge types of shape (E,)
            norm (Tensor, optional): Edge normalization of shape (E,)
            use_norm (bool): Whether to use normalization (unused, kept for compatibility)
        
        Returns:
            Tensor: Output node features of shape (N, out_dim)
        """
        emb = x
        for layer in self.layers:
            emb = layer(emb, edge_index, ntype, etype, norm=norm)
        output_emb = self.feed_forward(emb)
        return output_emb
    
    def extra_repr(self):
        return (f'n_layers={self.n_layers}, in_dim={self.in_dim}, '
                f'hid_dim={self.hid_dim}, out_dim={self.out_dim}, '
                f'num_heads={self.num_heads}, num_nodes={self.num_nodes}, '
                f'num_rels={self.num_rels}')

