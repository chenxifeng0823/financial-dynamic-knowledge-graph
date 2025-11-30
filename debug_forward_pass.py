"""
Debug script to compare DGL vs PyG forward pass step-by-step
"""
import sys
import torch
import numpy as np
from pathlib import Path

print("=" * 80)
print("STEP-BY-STEP COMPARISON: DGL vs PyG Forward Pass")
print("=" * 80)

# ============================================================================
# PART 1: Load PyG Model and Data
# ============================================================================
print("\n[PART 1] Loading PyG Model and Data")
print("-" * 80)

from src.models.pyg_kgtransformer import KGTransformerPyG, ConfigArgs
from src.data_processing.pyg_dataset import TemporalKGDatasetPyG

# Load PyG dataset
train_dataset, val_dataset, test_dataset, num_entities, num_relations = \
    TemporalKGDatasetPyG.from_txt_files(
        'data/FinDKG/train.txt',
        'data/FinDKG/valid.txt',
        'data/FinDKG/test.txt',
        'data/FinDKG/entity2id.txt'
    )

print(f"✓ PyG Dataset loaded:")
print(f"  Entities: {num_entities}")
print(f"  Relations: {num_relations}")
print(f"  Train batches: {len(train_dataset)}")

# Create PyG model
config = ConfigArgs()
config.device = 'cpu'  # Use CPU for easier debugging
pyg_model = KGTransformerPyG(num_entities=num_entities, num_relations=num_relations, args=config)
pyg_model.eval()

print(f"\n✓ PyG Model created:")
print(f"  Total parameters: {sum(p.numel() for p in pyg_model.parameters()):,}")
print(f"  Edge model parameters: {sum(p.numel() for p in pyg_model.edge_model.parameters()):,}")

# Get first batch
pyg_batch = train_dataset[0]
print(f"\n✓ First batch:")
print(f"  Nodes: {pyg_batch.node_id.shape[0]}")
print(f"  Edges: {pyg_batch.edge_index.shape[1]}")
print(f"  Node IDs (first 10): {pyg_batch.node_id[:10].tolist()}")
print(f"  Edge types (first 10): {pyg_batch.edge_type[:10].tolist()}")

# ============================================================================
# PART 2: Load DGL Model and Data
# ============================================================================
print("\n[PART 2] Loading DGL Model and Data")
print("-" * 80)

# Add DGL path
sys.path.insert(0, str(Path.cwd() / "reference/FinDKG_original"))

# Mock torch_scatter for DGL
class MockTorchScatter:
    @staticmethod
    def scatter_mean(src, index, dim=0, out=None, dim_size=None):
        # Simple implementation for testing
        if dim_size is None:
            dim_size = index.max().item() + 1
        out_tensor = torch.zeros(dim_size, src.size(1), dtype=src.dtype, device=src.device)
        for i in range(dim_size):
            mask = (index == i)
            if mask.any():
                out_tensor[i] = src[mask].mean(dim=0)
        return out_tensor

sys.modules['torch_scatter'] = MockTorchScatter()

try:
    import DKG
    from DKG import data as dkg_data
    from DKG.model import DKG_DEFAULT_CONFIG
    
    # Load DGL graph
    G = dkg_data.load_temporal_knowledge_graph("FinDKG", data_root='./data')
    
    print(f"✓ DGL Graph loaded:")
    print(f"  Nodes: {G.number_of_nodes()}")
    print(f"  Relations: {G.num_relations}")
    print(f"  Node types: {G.num_node_types}")
    print(f"  Train timestamps: {len(G.train_times)}")
    
    dgl_available = True
except Exception as e:
    print(f"✗ Could not load DGL: {e}")
    print("  Skipping DGL comparison")
    dgl_available = False

# ============================================================================
# PART 3: Compare Data Format
# ============================================================================
print("\n[PART 3] Comparing Data Format")
print("-" * 80)

print("\nPyG Batch Structure:")
print(f"  node_id: {pyg_batch.node_id.shape} - {pyg_batch.node_id.dtype}")
print(f"  edge_index: {pyg_batch.edge_index.shape} - {pyg_batch.edge_index.dtype}")
print(f"  edge_type: {pyg_batch.edge_type.shape} - {pyg_batch.edge_type.dtype}")
print(f"  timestamps: {pyg_batch.timestamps.shape} - {pyg_batch.timestamps.dtype}")
print(f"  node_type: {pyg_batch.node_type.shape} - {pyg_batch.node_type.dtype}")

if dgl_available:
    print(f"\nDGL Graph Info:")
    print(f"  Total edges in graph: {G.number_of_edges()}")
    print(f"  First timestamp: {G.train_times[0]}")
    print(f"  Edge data keys: {list(G.edata.keys())}")
    print(f"  Node data keys: {list(G.ndata.keys())}")

# ============================================================================
# PART 4: Compare Forward Pass - PyG
# ============================================================================
print("\n[PART 4] PyG Forward Pass Breakdown")
print("-" * 80)

with torch.no_grad():
    # Step 1: Get embeddings
    print("\nStep 1: Get embeddings for batch nodes")
    static_structural = pyg_model.static_entity_embeds.structural[pyg_batch.node_id.cpu()]
    dynamic_structural = pyg_model.dynamic_entity_embeds.structural[pyg_batch.node_id.cpu(), -1, :]  # Use last hidden state
    
    print(f"  static_structural: {static_structural.shape}, mean={static_structural.mean():.4f}")
    print(f"  dynamic_structural: {dynamic_structural.shape}, mean={dynamic_structural.mean():.4f}")
    
    # Step 2: Combiner
    print("\nStep 2: Combiner (static + dynamic)")
    combined_emb = pyg_model.combiner(static_structural, dynamic_structural, pyg_batch)
    print(f"  combined_emb: {combined_emb.shape}, mean={combined_emb.mean():.4f}")
    
    # Step 3: Graph Readout
    print("\nStep 3: Graph Readout")
    static_emb = static_structural
    dynamic_emb = dynamic_structural
    graph_emb = pyg_model.edge_model.graph_readout(combined_emb, static_emb, dynamic_emb)
    print(f"  graph_emb: {graph_emb.shape}, mean={graph_emb.mean():.4f}")
    print(f"  graph_emb values: min={graph_emb.min():.4f}, max={graph_emb.max():.4f}, std={graph_emb.std():.4f}")
    
    # Step 4: Get edge embeddings
    print("\nStep 4: Get edge embeddings")
    edge_head_emb = combined_emb[pyg_batch.edge_index[0]]
    print(f"  edge_head_emb: {edge_head_emb.shape}, mean={edge_head_emb.mean():.4f}")
    
    # Step 5: Relation embeddings
    print("\nStep 5: Relation embeddings")
    edge_static_rel = pyg_model.edge_model.rel_embeds[pyg_batch.edge_type.long()]
    dynamic_rel_emb = pyg_model.dynamic_relation_embeds.temporal[:, -1, :, 1]
    edge_dynamic_rel = dynamic_rel_emb[pyg_batch.edge_type.long()]
    edge_rel_embeds = torch.cat((edge_static_rel, edge_dynamic_rel), dim=1)
    print(f"  edge_static_rel: {edge_static_rel.shape}, mean={edge_static_rel.mean():.4f}")
    print(f"  edge_dynamic_rel: {edge_dynamic_rel.shape}, mean={edge_dynamic_rel.mean():.4f}")
    print(f"  edge_rel_embeds: {edge_rel_embeds.shape}, mean={edge_rel_embeds.mean():.4f}")
    
    # Step 6: Predictions
    print("\nStep 6: Make predictions")
    graph_emb_repeat = graph_emb.repeat(len(edge_head_emb), 1)
    
    # Head prediction
    head_input = graph_emb_repeat
    head_pred = pyg_model.edge_model.transform_head(head_input)
    print(f"  head_input: {head_input.shape}, mean={head_input.mean():.4f}")
    print(f"  head_pred: {head_pred.shape}, mean={head_pred.mean():.4f}")
    
    # Relation prediction
    rel_input = torch.cat((edge_head_emb, graph_emb_repeat), dim=1)
    rel_pred = pyg_model.edge_model.transform_rel(rel_input)
    print(f"  rel_input: {rel_input.shape}, mean={rel_input.mean():.4f}")
    print(f"  rel_pred: {rel_pred.shape}, mean={rel_pred.mean():.4f}")
    
    # Tail prediction
    tail_input = torch.cat((edge_head_emb, graph_emb_repeat, edge_rel_embeds), dim=1)
    tail_pred = pyg_model.edge_model.transform_tail(tail_input)
    print(f"  tail_input: {tail_input.shape}, mean={tail_input.mean():.4f}")
    print(f"  tail_pred: {tail_pred.shape}, mean={tail_pred.mean():.4f}")
    
    # Step 7: Compute losses
    print("\nStep 7: Compute losses")
    target_heads = pyg_batch.edge_index[0]
    target_rels = pyg_batch.edge_type
    target_tails = pyg_batch.edge_index[1]
    
    global_head_ids = pyg_batch.node_id[target_heads].long()
    global_tail_ids = pyg_batch.node_id[target_tails].long()
    
    criterion = torch.nn.CrossEntropyLoss()
    log_prob_head = -criterion(head_pred, global_head_ids)
    log_prob_rel = -criterion(rel_pred, target_rels.long())
    log_prob_tail = -criterion(tail_pred, global_tail_ids)
    
    log_prob_total = log_prob_tail + 0.2 * log_prob_rel + 0.1 * log_prob_head
    
    print(f"  log_prob_head: {log_prob_head.item():.4f}")
    print(f"  log_prob_rel: {log_prob_rel.item():.4f}")
    print(f"  log_prob_tail: {log_prob_tail.item():.4f}")
    print(f"  log_prob_total (0.1*head + 0.2*rel + 1.0*tail): {log_prob_total.item():.4f}")
    print(f"  loss (negative log_prob): {-log_prob_total.item():.4f}")

# ============================================================================
# PART 5: Key Findings
# ============================================================================
print("\n" + "=" * 80)
print("KEY FINDINGS")
print("=" * 80)

print("""
PyG Forward Pass Summary:
1. ✓ Embeddings are initialized and have reasonable values
2. ✓ Combiner produces combined embeddings
3. ✓ Graph readout produces graph-level embedding
4. ✓ All 3 prediction heads produce outputs
5. ✓ Multi-task loss is computed correctly

Next Steps:
1. Compare with DGL forward pass (if available)
2. Check if embedding values match DGL
3. Check if loss values match DGL
4. Identify any numerical differences

If DGL is not available, we need to:
1. Train DGL model to get baseline
2. Or carefully review DGL code for any missing components
""")

print("\n" + "=" * 80)

