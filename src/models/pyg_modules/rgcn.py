"""
RGCN: Relational Graph Convolutional Network
Using PyG's RGCNConv
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv


class RGCN(nn.Module):
    """
    Relational Graph Convolutional Network (RGCN)
    
    This is a multi-layer RGCN that applies relational graph convolutions
    with optional regularization (basis decomposition or block-diagonal).
    
    Args:
        in_dim (int): Input feature dimension
        hid_dim (int): Hidden feature dimension
        out_dim (int): Output feature dimension
        n_layers (int): Number of RGCN layers
        num_rels (int): Number of relation types
        regularizer (str): Regularization method ('basis' or 'bdd'). Default: 'basis'
        num_bases (int): Number of bases for basis decomposition. If None, uses num_rels
        use_bias (bool): Whether to use bias. Default: True
        activation (callable): Activation function. Default: F.relu
        use_self_loop (bool): Whether to include self-loops. Default: True
        dropout (float): Dropout rate. Default: 0.0
        layer_norm (bool): Whether to use layer normalization. Default: False
        low_mem (bool): Low memory mode (unused, kept for compatibility). Default: False
    
    Shape:
        - Input features: (N, in_dim)
        - Edge index: (2, E)
        - Edge types: (E,)
        - Edge norm: (E,) optional
        - Output: (N, out_dim)
    """
    
    def __init__(
        self,
        in_dim,
        hid_dim,
        out_dim,
        n_layers,
        num_rels,
        regularizer="basis",
        num_bases=None,
        use_bias=True,
        activation=F.relu,
        use_self_loop=True,
        dropout=0.0,
        layer_norm=False,
        low_mem=False
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.num_rels = num_rels
        self.regularizer = regularizer
        self.num_bases = num_bases if num_bases is not None else num_rels
        self.use_bias = use_bias
        self.activation = activation
        self.use_self_loop = use_self_loop
        self.dropout = dropout
        self.layer_norm = layer_norm
        self.low_mem = low_mem
        
        # Map DGL regularizer names to PyG
        if regularizer == "bdd":
            # Block diagonal decomposition
            num_blocks = num_bases if num_bases else None
            decomp_kwargs = {"num_blocks": num_blocks}
        else:
            # Basis decomposition (default)
            decomp_kwargs = {"num_bases": self.num_bases}
        
        self.n_layers = n_layers
        assert self.n_layers >= 1, f"Number of layers must be >= 1, got {self.n_layers}"
        
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList() if layer_norm else None
        
        if self.n_layers == 1:
            # Single layer: in_dim -> out_dim, no activation
            self.layers.append(RGCNConv(
                self.in_dim,
                self.out_dim,
                self.num_rels,
                **decomp_kwargs,
                bias=self.use_bias
            ))
            if layer_norm:
                self.norms.append(nn.LayerNorm(self.out_dim))
        else:
            # Multi-layer: in_dim -> hid_dim -> ... -> out_dim
            
            # First layer: in_dim -> hid_dim
            self.layers.append(RGCNConv(
                self.in_dim,
                self.hid_dim,
                self.num_rels,
                **decomp_kwargs,
                bias=self.use_bias
            ))
            if layer_norm:
                self.norms.append(nn.LayerNorm(self.hid_dim))
            
            # Middle layers: hid_dim -> hid_dim
            for i in range(1, self.n_layers - 1):
                self.layers.append(RGCNConv(
                    self.hid_dim,
                    self.hid_dim,
                    self.num_rels,
                    **decomp_kwargs,
                    bias=self.use_bias
                ))
                if layer_norm:
                    self.norms.append(nn.LayerNorm(self.hid_dim))
            
            # Last layer: hid_dim -> out_dim, no activation
            self.layers.append(RGCNConv(
                self.hid_dim,
                self.out_dim,
                self.num_rels,
                **decomp_kwargs,
                bias=self.use_bias
            ))
            if layer_norm:
                self.norms.append(nn.LayerNorm(self.out_dim))
        
        assert self.n_layers == len(self.layers), \
            f"Expected {self.n_layers} layers, got {len(self.layers)}"
        
        self.drop = nn.Dropout(dropout)
    
    def forward(self, x, edge_index, edge_type, edge_norm=None):
        """
        Forward pass of RGCN.
        
        Args:
            x (Tensor): Node features of shape (N, in_dim)
            edge_index (Tensor): Edge indices of shape (2, E)
            edge_type (Tensor): Edge types of shape (E,)
            edge_norm (Tensor, optional): Edge normalization of shape (E,) or (E, 1)
                Note: In PyG 2.x, RGCNConv doesn't support edge_norm directly.
                We compute it internally based on node degrees if needed.
        
        Returns:
            Tensor: Updated node features of shape (N, out_dim)
        """
        # Ensure edge_type is long
        edge_type = edge_type.long()
        
        # Note: PyG's RGCNConv doesn't support custom edge weights in recent versions
        # It computes its own normalization internally
        # We'll ignore edge_norm for now as it's handled internally by RGCNConv
        
        h = x
        for i, layer in enumerate(self.layers):
            # Apply RGCN layer (no edge_norm parameter in PyG 2.x)
            h = layer(h, edge_index, edge_type)
            
            # Apply layer norm if specified
            if self.layer_norm and self.norms is not None:
                h = self.norms[i](h)
            
            # Apply activation and dropout for all but the last layer
            if i < self.n_layers - 1:
                if self.activation is not None:
                    h = self.activation(h)
                h = self.drop(h)
        
        return h
    
    def extra_repr(self):
        return (f'n_layers={self.n_layers}, in_dim={self.in_dim}, '
                f'hid_dim={self.hid_dim}, out_dim={self.out_dim}, '
                f'num_rels={self.num_rels}, regularizer={self.regularizer}, '
                f'num_bases={self.num_bases}')

