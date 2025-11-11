# KGTransformer Architecture (PyG Implementation)

## Overview

This document provides a technical deep-dive into the KGTransformer architecture and its PyTorch Geometric (PyG) implementation.

## Table of Contents

1. [Model Architecture](#model-architecture)
2. [Core Components](#core-components)
3. [Implementation Details](#implementation-details)
4. [DGL vs PyG Differences](#dgl-vs-pyg-differences)
5. [Training & Evaluation](#training--evaluation)

---

## Model Architecture

### High-Level Structure

```
KGTransformerPyG
├── Static Embeddings (learnable)
│   ├── structural: [num_entities, 200]
│   └── temporal: [num_entities, 200]
│
├── Dynamic Embeddings (RNN hidden states)
│   ├── structural: [num_entities, num_rnn_layers, 200]
│   └── temporal: [num_entities, num_rnn_layers, 200, 2]
│
├── EmbeddingUpdater
│   ├── GraphStructuralRNNConv (Graph Conv + RNN)
│   ├── GraphTemporalRNNConv (Graph Conv + RNN + Time Decay)
│   └── RelationRNN (Relation embedding updates)
│
├── Combiner
│   └── Combines static + dynamic embeddings
│
└── EdgeModel
    └── Link prediction (head, relation, tail)
```

### Multi-Aspect Embeddings

The model maintains **two types of embeddings**:

1. **Static Embeddings**: Time-invariant entity/relation representations
   - Learned as standard embedding layers
   - Capture inherent properties of entities

2. **Dynamic Embeddings**: Time-evolving representations via RNN
   - **Structural**: Captures graph topology changes
   - **Temporal**: Captures temporal patterns with time decay
   - **Bidirectional**: Separate states for recipient/sender roles

---

## Core Components

### 1. TypedLinear

Type-specific linear transformations for heterogeneous graphs.

**Purpose**: Apply different weight matrices based on node/edge types.

**Implementation**:
```python
class TypedLinear(nn.Module):
    def __init__(self, in_features, out_features, num_types):
        self.weight = nn.Parameter(torch.Tensor(num_types, out_features, in_features))
    
    def forward(self, x, types):
        # Gather weights for each type: (N, out_features, in_features)
        type_weights = self.weight[types]
        # Batched matrix multiplication
        out = torch.bmm(type_weights, x.unsqueeze(-1)).squeeze(-1)
        return out
```

**Key Difference from DGL**: DGL uses optimized C++ backend; PyG uses pure PyTorch.

---

### 2. Graph Transformer

Multi-head attention mechanism for knowledge graphs with relation-specific transformations.

**Architecture**:
```
Input: Node features, Edge index, Node types, Edge types
  ↓
Q, K, V Projections (type-specific via TypedLinear)
  ↓
Multi-Head Attention (with relation-specific transformations)
  ↓
Message Passing & Aggregation
  ↓
Residual Connection + Layer Norm
  ↓
Output: Updated node features
```

**Key Features**:
- **Type-specific Q/K/V**: Different projections for different node types
- **Relation-specific attention**: `attention = (K @ W_rel @ Q^T) * priority / sqrt(d)`
- **Relation-specific messages**: `message = V @ W_rel`
- **Learnable skip connections**: Per-node-type residual weights

**Implementation**:
```python
class GraphTransformerConv(MessagePassing):
    def __init__(self, in_size, hid_size, num_heads, num_ntypes, num_etypes):
        self.linear_q = TypedLinear(in_size, hid_size * num_heads, num_ntypes)
        self.linear_k = TypedLinear(in_size, hid_size * num_heads, num_ntypes)
        self.linear_v = TypedLinear(in_size, hid_size * num_heads, num_ntypes)
        
        # Relation-specific transformations (per head)
        self.relation_att = nn.ModuleList([
            TypedLinear(head_size, head_size, num_etypes)
            for _ in range(num_heads)
        ])
        self.relation_msg = nn.ModuleList([
            TypedLinear(head_size, head_size, num_etypes)
            for _ in range(num_heads)
        ])
```

---

### 3. RGCN (Relational Graph Convolutional Network)

Alternative to Graph Transformer for graph convolutions.

**Purpose**: Aggregate neighbor information with relation-specific transformations.

**Implementation**: Uses PyG's built-in `RGCNConv` with basis decomposition.

```python
class RGCN(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, n_layers, num_rels, num_bases=100):
        self.layers = nn.ModuleList([
            RGCNConv(in_dim, hid_dim, num_rels, num_bases=num_bases),
            RGCNConv(hid_dim, out_dim, num_rels, num_bases=num_bases)
        ])
```

---

### 4. RNN-Based Temporal Encoders

#### GraphStructuralRNNConv

Captures structural dynamics (graph topology changes over time).

**Flow**:
```
Static Embeddings → Graph Conv (KGT/RGCN) → RNN → Updated Dynamic Embeddings
```

**Code**:
```python
# 1. Apply graph convolution on static embeddings
h = self.graph_conv(static_emb, edge_index, node_type, edge_type)

# 2. Feed to RNN with previous hidden state
output, hn = self.rnn(h.unsqueeze(1), dynamic_emb.transpose(0, 1))

# 3. Return updated hidden state
return hn.transpose(0, 1)
```

#### GraphTemporalRNNConv

Captures temporal dynamics with time decay.

**Key Innovation**: Uses inter-event times as edge weights.

**Flow**:
```
1. Compute inter-event times (Δt = current_time - last_event_time)
2. Transform: edge_weight = 1 / log(Δt + 1)
3. Apply graph conv with time-weighted edges
4. Process forward graph (as recipient)
5. Process reverse graph (as sender)
6. Update RNN states bidirectionally
```

**Code**:
```python
# Compute time decay
inter_event_times = data.timestamps - node_latest_event_time[dst, src]
edge_norm = (1 / log_transform(inter_event_times).clamp(min=1e-10)).clamp(max=10.0)

# Apply graph conv with time weighting
h = self.graph_conv(static_emb, edge_index, edge_type, edge_norm)

# Update RNN (separate for forward/reverse)
output, hn_fwd = self.rnn(h_fwd, dynamic_emb[..., 0])  # recipient
output, hn_rev = self.rnn(h_rev, dynamic_emb[..., 1])  # sender
```

---

## Implementation Details

### Data Flow

**Training Loop (Per Timestamp)**:
```python
for timestamp in range(num_timestamps):
    # 1. Get batch data for current timestamp
    batch_data = dataset[timestamp]  # PyG Data object
    
    # 2. Update embeddings
    dynamic_entity_emb, dynamic_relation_emb = embedding_updater(
        batch_data, static_emb, dynamic_entity_emb, dynamic_relation_emb
    )
    
    # 3. Combine static + dynamic
    combined_emb = combiner(static_emb, dynamic_entity_emb)
    
    # 4. Link prediction
    log_prob, tail_pred = edge_model(batch_data, combined_emb, dynamic_relation_emb)
    
    # 5. Compute loss and backprop
    loss = -log_prob
    loss.backward()
    optimizer.step()
```

### Memory Management

**Key Design Decision**: Keep dynamic embeddings on CPU, move to GPU only during computation.

**Rationale**:
- Dynamic embeddings are large: `[num_entities, num_rnn_layers, hidden_dim]`
- Only a subset of entities appear in each batch
- Moving entire tensor to GPU is wasteful

**Implementation**:
```python
# Embeddings stored on CPU
self.dynamic_entity_embeds = MultiAspectEmbedding(
    structural=torch.zeros(num_entities, num_rnn_layers, 200),  # CPU
    temporal=torch.zeros(num_entities, num_rnn_layers, 200, 2)  # CPU
)

# During forward pass: move only batch nodes to GPU
batch_dynamic = self.dynamic_entity_embeds.structural[batch_data.node_id.cpu()].to(device)
```

### Device Management

**Critical Pattern**: Always use CPU indices for indexing CPU tensors.

```python
# ✅ CORRECT
node_id_cpu = batch_data.node_id.cpu()
embeddings_cpu[node_id_cpu]  # Both on CPU

# ❌ WRONG
embeddings_cpu[batch_data.node_id]  # node_id on GPU, embeddings on CPU
```

### Preventing Backprop Through Time

**Problem**: RNN states are updated across batches, creating a computational graph spanning all timestamps.

**Solution**: Detach updated embeddings after each batch.

```python
# After updating embeddings
updated_dynamic_entity_emb = MultiAspectEmbedding(
    structural=updated_structural.detach(),  # Break gradient flow
    temporal=updated_temporal.detach()
)
```

---

## DGL vs PyG Differences

### API Comparison

| Aspect | DGL | PyG |
|--------|-----|-----|
| **Graph** | `dgl.DGLGraph` | `torch_geometric.data.Data` |
| **Edges** | `g.edges()` → (src, dst) | `data.edge_index` [2, E] |
| **Node Features** | `g.ndata['feat']` | `data.x` |
| **Edge Features** | `g.edata['feat']` | `data.edge_attr` |
| **Message Passing** | `g.update_all(msg_fn, reduce_fn)` | `MessagePassing.propagate()` |
| **Edge Softmax** | `dgl.nn.edge_softmax(g, scores)` | `torch_geometric.utils.softmax(scores, index)` |
| **TypedLinear** | `dgl.nn.TypedLinear` (C++) | Custom PyTorch implementation |
| **RGCN** | `dgl.nn.RelGraphConv` | `torch_geometric.nn.RGCNConv` |

### Key Implementation Differences

**1. Message Passing**

DGL:
```python
def message(edges):
    return {'m': edges.src['h'] * edges.data['w']}

g.apply_edges(message)
g.update_all(fn.copy_e('m', 'm'), fn.sum('m', 'h'))
```

PyG:
```python
class MyConv(MessagePassing):
    def message(self, x_j, edge_attr):
        return x_j * edge_attr
    
    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)
```

**2. Edge Softmax**

DGL:
```python
g.edata['a'] = attention_scores
g.edata['a'] = edge_softmax(g, g.edata['a'])
```

PyG:
```python
# Need to specify which node to group by (destination)
alpha = softmax(attention_scores, edge_index[1])
```

---

## Training & Evaluation

### Hyperparameters (FinDKG)

```python
config = {
    'static_entity_embed_dim': 200,
    'structural_dynamic_entity_embed_dim': 200,
    'temporal_dynamic_entity_embed_dim': 200,
    'rel_embed_dim': 200,
    'num_gconv_layers': 2,
    'num_rnn_layers': 1,
    'num_attn_heads': 8,
    'dropout': 0.2,
    'lr': 0.001,
    'weight_decay': 0.00001,
    'epochs': 100,
    'embedding_updater_gconv': 'KGT+RNN',  # or 'RGCN+RNN'
}
```

### Training Script

```bash
# Basic training
python train_kgt_pyg.py --device cuda --epochs 100 --save_model

# Custom hyperparameters
python train_kgt_pyg.py \
    --device cuda \
    --epochs 50 \
    --lr 0.0005 \
    --seed 42 \
    --save_model \
    --save_dir checkpoints
```

### Model Size

**Total Parameters**: ~31.9M for FinDKG
- Static embeddings: ~5.5M
- Dynamic embeddings: ~5.5M
- Graph Transformer: ~15M
- RNNs: ~3M
- Edge Model: ~2.9M

### Performance Expectations

**First Epoch Results** (FinDKG):
```
Initial Loss: 9.52
Final Loss: 7.39
Average Loss: 7.64
Validation Loss: 7.42
Test Loss: 8.44
```

Loss decreases steadily, indicating the model is learning temporal patterns.

---

## File Structure

```
src/models/
├── pyg_modules/
│   ├── typed_linear.py          # Type-specific linear layers
│   ├── rgcn.py                  # RGCN implementation
│   ├── graph_transformer.py     # Graph Transformer layers
│   └── embedding_updater.py     # RNN-based temporal encoders
└── pyg_kgtransformer.py         # Complete model

src/data_processing/
└── pyg_dataset.py               # PyG dataset and dataloaders

train_kgt_pyg.py                 # Training script
test_kgt_pyg.py                  # Test suite
```

---

## References

- **Original Paper**: FinDKG - Financial Dynamic Knowledge Graph
- **Original Code**: https://github.com/xiaohui-victor-li/FinDKG (DGL implementation)
- **PyTorch Geometric**: https://pytorch-geometric.readthedocs.io/

---

## Next Steps: Research Extension

### Goal: Replace RNN with Transformer for Temporal Modeling

**Current**: Graph Transformer (spatial) + RNN (temporal)  
**Target**: Graph Transformer (spatial) + Transformer (temporal)

**Implementation Plan**:
1. Create `GraphTemporalTransformerConv` to replace `GraphTemporalRNNConv`
2. Use `nn.TransformerEncoder` for temporal sequences
3. Design positional encoding for timestamps
4. Compare performance: RNN vs Transformer

**Expected Benefits**:
- Better long-range temporal dependencies
- Parallel processing of temporal sequences
- Attention visualization for interpretability

---

**Last Updated**: 2025-11-11  
**Status**: ✅ Implementation Complete & Tested

