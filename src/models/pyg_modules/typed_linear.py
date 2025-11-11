"""
TypedLinear: Type-specific linear transformations
Equivalent to DGL's TypedLinear but for PyG
"""

import torch
import torch.nn as nn


class TypedLinear(nn.Module):
    """
    Linear transformation with separate weights for each type.
    
    This is equivalent to DGL's TypedLinear layer, which applies different
    linear transformations based on node/edge types.
    
    Args:
        in_features (int): Size of input features
        out_features (int): Size of output features
        num_types (int): Number of types (node types or edge types)
        bias (bool): If True, add bias term. Default: True
    
    Shape:
        - Input: (N, in_features) where N is the number of nodes/edges
        - Type indices: (N,) with values in [0, num_types-1]
        - Output: (N, out_features)
    
    Examples:
        >>> typed_linear = TypedLinear(64, 128, num_types=10)
        >>> x = torch.randn(100, 64)
        >>> types = torch.randint(0, 10, (100,))
        >>> out = typed_linear(x, types)  # (100, 128)
    """
    
    def __init__(self, in_features, out_features, num_types, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_types = num_types
        self.use_bias = bias
        
        # Option 1: Use a single large weight matrix for efficiency
        # Shape: (num_types, out_features, in_features)
        self.weight = nn.Parameter(torch.Tensor(num_types, out_features, in_features))
        
        if bias:
            self.bias = nn.Parameter(torch.Tensor(num_types, out_features))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters using Kaiming uniform initialization"""
        for i in range(self.num_types):
            nn.init.kaiming_uniform_(self.weight[i], a=2**0.5)
        if self.use_bias:
            nn.init.zeros_(self.bias)
    
    def forward(self, x, types, presorted=False):
        """
        Apply type-specific linear transformation.
        
        Args:
            x (Tensor): Input features of shape (N, in_features)
            types (Tensor): Type indices of shape (N,) with values in [0, num_types-1]
            presorted (bool): If True, assumes inputs are sorted by type (unused for now,
                            kept for compatibility with DGL API). Default: False
        
        Returns:
            Tensor: Output features of shape (N, out_features)
        """
        # Efficient implementation using gather/index_select
        # For each node, select the weight matrix corresponding to its type
        
        # Method 1: Loop-free implementation using advanced indexing
        # This is memory efficient and reasonably fast
        
        batch_size = x.size(0)
        device = x.device
        
        # Gather weights for each type: (N, out_features, in_features)
        type_weights = self.weight[types]  # (N, out_features, in_features)
        
        # Perform batched matrix multiplication
        # (N, out_features, in_features) @ (N, in_features, 1) -> (N, out_features, 1)
        out = torch.bmm(type_weights, x.unsqueeze(-1)).squeeze(-1)  # (N, out_features)
        
        if self.use_bias:
            type_biases = self.bias[types]  # (N, out_features)
            out = out + type_biases
        
        return out
    
    def extra_repr(self):
        return f'in_features={self.in_features}, out_features={self.out_features}, ' \
               f'num_types={self.num_types}, bias={self.use_bias}'


class FastTypedLinear(nn.Module):
    """
    Optimized version of TypedLinear for cases where types are presorted.
    
    When inputs are sorted by type, we can perform more efficient batched operations.
    This can be significantly faster for large graphs.
    
    Args:
        Same as TypedLinear
    """
    
    def __init__(self, in_features, out_features, num_types, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_types = num_types
        self.use_bias = bias
        
        # Use ModuleList of Linear layers for each type
        # This can be faster when types are presorted
        self.linears = nn.ModuleList([
            nn.Linear(in_features, out_features, bias=bias)
            for _ in range(num_types)
        ])
    
    def forward(self, x, types, presorted=False):
        """
        Apply type-specific linear transformation.
        
        If presorted=True, assumes x and types are sorted by type for better performance.
        """
        if presorted:
            # Optimized path for presorted inputs
            return self._forward_presorted(x, types)
        else:
            # General path
            return self._forward_general(x, types)
    
    def _forward_general(self, x, types):
        """General implementation (not presorted)"""
        outputs = []
        for i in range(self.num_types):
            mask = (types == i)
            if mask.any():
                outputs.append((mask, self.linears[i](x[mask])))
        
        # Reconstruct output in original order
        out = torch.zeros(x.size(0), self.out_features, device=x.device, dtype=x.dtype)
        for mask, result in outputs:
            out[mask] = result
        
        return out
    
    def _forward_presorted(self, x, types):
        """Optimized implementation for presorted inputs"""
        # Find boundaries between different types
        type_changes = torch.cat([
            torch.tensor([0], device=types.device),
            (types[1:] != types[:-1]).nonzero(as_tuple=True)[0] + 1,
            torch.tensor([len(types)], device=types.device)
        ])
        
        outputs = []
        for i in range(len(type_changes) - 1):
            start, end = type_changes[i].item(), type_changes[i + 1].item()
            type_id = types[start].item()
            outputs.append(self.linears[type_id](x[start:end]))
        
        return torch.cat(outputs, dim=0)
    
    def extra_repr(self):
        return f'in_features={self.in_features}, out_features={self.out_features}, ' \
               f'num_types={self.num_types}, bias={self.use_bias}'

