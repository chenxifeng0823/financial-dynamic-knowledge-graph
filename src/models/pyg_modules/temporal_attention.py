import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from ...utils.debug_utils import check_nan  # Debug utility

class TemporalPositionalEncoding(nn.Module):
    """
    Encodes the temporal position (time difference) into the embedding.
    Uses sinusoidal encoding based on the time difference from current time.
    """
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.d_model = d_model
        
        # Create constant 'pe' matrix with values dependent on pos and i
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)

    def forward(self, time_diffs):
        """
        Args:
            time_diffs: [Batch, Seq_Len] or [Batch, Seq_Len, 1] - Time difference from current time
        Returns:
            [Batch, Seq_Len, d_model]
        """
        # Clamp time diffs to range [0, max_len-1] for lookup
        indices = time_diffs.long().clamp(0, self.pe.size(0) - 1)
        if indices.dim() == 3:
            indices = indices.squeeze(-1)
            
        return self.pe[indices]


class TemporalAttentionLayer(nn.Module):
    """
    Scaled Dot-Product Attention over the temporal history of a node.
    """
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        assert self.head_dim * num_heads == d_model, "d_model must be divisible by num_heads"
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, query, history, history_times, current_time, mask=None):
        """
        Args:
            query: Current node state [Batch_Nodes, d_model] (Q)
            history: Historical node states [Batch_Nodes, Seq_Len, d_model] (K, V)
            history_times: Timestamps of history [Batch_Nodes, Seq_Len]
            current_time: Current timestamp [Batch_Nodes] or scalar
            mask: Optional mask for padding [Batch_Nodes, Seq_Len]
        
        Returns:
            Context vector [Batch_Nodes, d_model]
        """
        batch_size, seq_len, _ = history.size()
        
        # Reshape Query to [Batch, 1, d_model] for attention
        query = query.unsqueeze(1) 
        
        # Debug inputs
        if torch.isnan(query).any(): print("❌ Query has NaNs")
        if torch.isnan(history).any(): print("❌ History has NaNs")
        
        # Projections
        Q = self.q_proj(query).view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(history).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(history).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Debug Projections
        if torch.isnan(Q).any(): print(f"❌ Q has NaNs! Range: [{Q.min()}, {Q.max()}]")
        if torch.isnan(K).any(): print(f"❌ K has NaNs! Range: [{K.min()}, {K.max()}]")
        
        # Scaled Dot-Product Attention
        # Scores: [Batch, Heads, 1, Seq_Len]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Debug Scores
        if torch.isnan(scores).any(): 
            print(f"❌ Scores (pre-mask) has NaNs!")
        
        if mask is not None:
            # Mask should be [Batch, 1, 1, Seq_Len]
            mask = mask.unsqueeze(1).unsqueeze(1)
            
            # Check for empty history
            is_all_masked = (mask.sum(dim=-1) == 0) # [Batch, 1, 1]
            if is_all_masked.any():
                # print(f"⚠️ Found {is_all_masked.sum().item()} nodes with NO history (mask all 0)")
                pass
            
            scores = scores.masked_fill(mask == 0, float('-inf'))
            
            # Handle case where all history is masked (no history)
            if is_all_masked.any():
                # Set scores to 0.0 where mask is all 0, so softmax produces uniform distribution
                scores = scores.masked_fill(is_all_masked.unsqueeze(-1), 0.0)
        
        attn_weights = F.softmax(scores, dim=-1)
        
        # Debug Softmax Output
        if torch.isnan(attn_weights).any():
            print(f"❌ Softmax output has NaNs!")
            print(f"   Scores min/max: {scores.min()}/{scores.max()}")
            # Check specifically for the case that caused trouble
            nan_mask = torch.isnan(attn_weights)
            if nan_mask.any():
                idx = torch.where(nan_mask)[0][0]
                print(f"   Example NaN Score slice: {scores[idx, 0, 0, :]}")
        
        # If we had all-masked rows, attn_weights are uniform now.
        # We should zero them out if they are padding.
        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, 0.0)
            
        attn_weights = self.dropout(attn_weights)
        
        # Context: [Batch, Heads, 1, Head_Dim]
        context = torch.matmul(attn_weights, V)
        
        # Debug Context
        if torch.isnan(context).any(): print(f"❌ Context has NaNs!")
        
        # Reshape and Output
        context = context.transpose(1, 2).contiguous().view(batch_size, 1, self.d_model)
        output = self.out_proj(context).squeeze(1)
        
        # Residual Connection + Layer Norm
        # Note: We add to the *query* (current state)
        res = query.squeeze(1) + output
        output = self.layer_norm(res)
        
        if torch.isnan(output).any(): print(f"❌ Final output has NaNs!")
        
        return output

class GraphTemporalTransformerConv(nn.Module):
    """
    Replaces GraphTemporalRNNConv.
    Uses attention over a history window instead of an RNN state.
    """
    def __init__(self, 
                 in_dim, 
                 d_model, 
                 num_heads, 
                 window_size=10, 
                 dropout=0.1):
        super().__init__()
        self.window_size = window_size
        self.d_model = d_model
        
        # Input projection if dimensions don't match
        self.input_proj = nn.Linear(in_dim, d_model) if in_dim != d_model else nn.Identity()
        
        # Temporal Positional Encoding
        self.pos_encoder = TemporalPositionalEncoding(d_model)
        
        # Attention Layer
        self.attention = TemporalAttentionLayer(d_model, num_heads, dropout)
        
        # Output Update MLP (similar to GRU update gate)
        self.update_gate = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.Tanh()
        )

    def forward(self, current_features, history_features, history_times, current_time, mask=None):
        """
        Args:
            current_features: [Batch_Nodes, in_dim] - Features from spatial GNN
            history_features: [Batch_Nodes, Window_Size, d_model] - History bank
            history_times: [Batch_Nodes, Window_Size] - Timestamps of history
            current_time: Scalar or [Batch_Nodes]
        """
        # Project input
        curr_emb = self.input_proj(current_features) # [Batch, d_model]
        
        if history_features is None or history_features.size(1) == 0:
            # No history yet, just return current
            return curr_emb
            
        # Add positional encoding to history
        # Calculate time diff: current_time - history_time
        if isinstance(current_time, torch.Tensor):
            if current_time.dim() == 0:
                current_time = current_time.unsqueeze(0)
            if current_time.dim() == 1:
                current_time = current_time.unsqueeze(1) # [Batch, 1]
        
        time_diffs = current_time - history_times # [Batch, Window]
        pos_emb = self.pos_encoder(time_diffs)
        
        history_with_pos = history_features + pos_emb
        
        # Attend to history
        # context represents "relevant past information"
        context = self.attention(curr_emb, history_with_pos, history_times, current_time, mask)
        
        # Update current state using context (like a Gated Residual)
        # Combine [Current, Context] -> New State
        combined = torch.cat([curr_emb, context], dim=-1)
        new_state = self.update_gate(combined)
        
        return new_state

