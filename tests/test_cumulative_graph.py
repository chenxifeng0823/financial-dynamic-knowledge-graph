"""
Test cumulative graph building to ensure it matches DGL's approach
"""
import torch
from src.data_processing.pyg_dataset import TemporalKGDatasetPyG, CumulativeGraphBuilder

print("=" * 80)
print("TESTING CUMULATIVE GRAPH BUILDING")
print("=" * 80)

# Load dataset
train_dataset, val_dataset, test_dataset, num_entities, num_relations = \
    TemporalKGDatasetPyG.from_txt_files(
        'data/FinDKG/train.txt',
        'data/FinDKG/valid.txt',
        'data/FinDKG/test.txt',
        'data/FinDKG/entity2id.txt'
    )

print(f"\nDataset Info:")
print(f"  Entities: {num_entities}")
print(f"  Relations: {num_relations}")
print(f"  Train batches: {len(train_dataset)}")

# Initialize cumulative graph builder
entity_types = train_dataset.entity_types
cumul_builder = CumulativeGraphBuilder(
    num_entities,
    num_relations,
    entity_types
)

print(f"\n[INITIAL STATE]")
print(f"  Cumulative nodes: {len(cumul_builder.cumul_node_set)}")
print(f"  Cumulative edges: {len(cumul_builder.cumul_edge_index[0]) if cumul_builder.cumul_edge_index is not None else 0}")

# Process first few batches
num_test_batches = 5
for batch_idx in range(num_test_batches):
    batch_data = train_dataset[batch_idx]
    
    print(f"\n[BATCH {batch_idx}]")
    print(f"  Batch nodes: {batch_data.node_id.shape[0]}")
    print(f"  Batch edges: {batch_data.edge_index.shape[1]}")
    print(f"  Node IDs (first 5): {batch_data.node_id[:5].tolist()}")
    print(f"  Edge types (first 5): {batch_data.edge_type[:5].tolist()}")
    print(f"  Timestamps (first 5): {batch_data.timestamps[:5].tolist()}")
    
    # Add batch to cumulative graph
    cumul_data = cumul_builder.add_batch(batch_data)
    
    print(f"\n  [CUMULATIVE GRAPH AFTER BATCH {batch_idx}]")
    print(f"    Total nodes: {cumul_data.node_id.shape[0]}")
    print(f"    Total edges: {cumul_data.edge_index.shape[1]}")
    print(f"    Unique nodes in set: {len(cumul_builder.cumul_node_set)}")
    
    # Verify cumulative properties
    if batch_idx > 0:
        # Check that edges are accumulating
        expected_min_edges = sum(train_dataset[i].edge_index.shape[1] for i in range(batch_idx + 1))
        actual_edges = cumul_data.edge_index.shape[1]
        
        if actual_edges >= expected_min_edges:
            print(f"    ✓ Edges accumulating correctly ({actual_edges} >= {expected_min_edges})")
        else:
            print(f"    ❌ Edges NOT accumulating! ({actual_edges} < {expected_min_edges})")
        
        # Check that nodes are accumulating
        expected_min_nodes = len(set().union(*[set(train_dataset[i].node_id.tolist()) for i in range(batch_idx + 1)]))
        actual_nodes = len(cumul_builder.cumul_node_set)
        
        if actual_nodes >= expected_min_nodes:
            print(f"    ✓ Nodes accumulating correctly ({actual_nodes} >= {expected_min_nodes})")
        else:
            print(f"    ❌ Nodes NOT accumulating! ({actual_nodes} < {expected_min_nodes})")

# ============================================================================
# Compare with DGL approach
# ============================================================================
print("\n" + "=" * 80)
print("COMPARING WITH DGL APPROACH")
print("=" * 80)

import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / "reference/FinDKG_original"))

# Mock torch_scatter
class MockTorchScatter:
    @staticmethod
    def scatter_mean(src, index, dim=0, out=None, dim_size=None):
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
    
    # Load DGL graph
    G = dkg_data.load_temporal_knowledge_graph("FinDKG", data_root='./data')
    
    print(f"\nDGL Graph Structure:")
    print(f"  Total nodes: {G.number_of_nodes()}")
    print(f"  Total edges: {G.number_of_edges()}")
    print(f"  Train timestamps: {len(G.train_times)}")
    
    # Check how DGL processes timestamps
    print(f"\n[DGL TIMESTAMP PROCESSING]")
    
    # Get edges for first few timestamps
    for t_idx in range(min(5, len(G.train_times))):
        timestamp = G.train_times[t_idx]
        
        # Find edges at this timestamp
        edge_times = G.edata['time']
        time_mask = (edge_times == timestamp)
        time_edge_ids = torch.where(time_mask)[0]
        
        print(f"\n  Timestamp {t_idx} (time={timestamp}):")
        print(f"    Edges at this timestamp: {len(time_edge_ids)}")
        
        # In DGL, the cumulative graph includes ALL edges up to current time
        cumul_mask = (edge_times <= timestamp)
        cumul_edge_ids = torch.where(cumul_mask)[0]
        print(f"    Cumulative edges up to this time: {len(cumul_edge_ids)}")
        
        # Get unique nodes involved
        src, dst = G.edges()
        cumul_src = src[cumul_edge_ids]
        cumul_dst = dst[cumul_edge_ids]
        cumul_nodes = torch.cat([cumul_src, cumul_dst]).unique()
        print(f"    Cumulative nodes: {len(cumul_nodes)}")
    
    dgl_available = True
    
except Exception as e:
    print(f"\n✗ Could not load DGL: {e}")
    dgl_available = False

# ============================================================================
# Key Checks
# ============================================================================
print("\n" + "=" * 80)
print("KEY CHECKS")
print("=" * 80)

print("""
Cumulative Graph Requirements:
1. ✓ Edges should accumulate over time (not replace)
2. ✓ Nodes should accumulate over time
3. ✓ Each batch should see ALL previous edges + current edges
4. ✓ Graph should grow monotonically

Potential Issues:
1. Are we building a cumulative graph or just using current batch?
2. Are edges being deduplicated when they shouldn't be?
3. Are we resetting the cumulative graph between epochs?
4. Are timestamps being processed in correct order?
""")

# Check if cumulative graph is actually being used in training
print("\n[CHECKING TRAINING USAGE]")
print("Looking at train_kgt_pyg.py to see how cumulative graph is used...")

import re
with open('train_kgt_pyg.py', 'r') as f:
    train_code = f.read()
    
    # Check if cumulative graph is built
    if 'CumulativeGraphBuilder' in train_code:
        print("✓ CumulativeGraphBuilder is imported")
    else:
        print("❌ CumulativeGraphBuilder NOT imported!")
    
    # Check if it's used in training loop
    if 'cumul_builder.add_batch' in train_code:
        print("✓ cumul_builder.add_batch is called")
    else:
        print("❌ cumul_builder.add_batch NOT called!")
    
    # Check if cumulative data is used
    if 'cumul_data' in train_code:
        print("✓ cumul_data variable exists")
        
        # But is it actually USED?
        if re.search(r'model\(.*cumul_data.*\)', train_code):
            print("✓ cumul_data is passed to model")
        else:
            print("❌ cumul_data is NOT passed to model!")
            print("   → Model might be using batch_data instead of cumul_data!")
    else:
        print("❌ cumul_data variable NOT found!")

print("\n" + "=" * 80)
print("DIAGNOSIS")
print("=" * 80)

print("""
If cumul_data is built but NOT passed to the model, this is a CRITICAL BUG!

The model would be training on individual batches instead of the cumulative
graph, which means:
1. It doesn't see historical context
2. It can't learn temporal patterns
3. Performance would be much worse than expected

This could explain the 3.6% MRR vs 12.45% gap!
""")

