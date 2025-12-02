"""
Test script for KGTransformer PyG implementation
Verifies that all components work correctly
"""

import sys
import torch
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.data_processing.pyg_dataset import TemporalKGDatasetPyG, create_temporal_dataloaders
from src.models.pyg_modules.typed_linear import TypedLinear
from src.models.pyg_modules.rgcn import RGCN
from src.models.pyg_modules.graph_transformer import GraphTransformerConv, KGTransformer
from src.models.pyg_kgtransformer import KGTransformerPyG, ConfigArgs
import torch.nn.functional as F


def test_typed_linear():
    """Test TypedLinear layer"""
    print("\n" + "="*50)
    print("Testing TypedLinear")
    print("="*50)
    
    batch_size = 10
    in_features = 64
    out_features = 128
    num_types = 5
    
    typed_linear = TypedLinear(in_features, out_features, num_types)
    x = torch.randn(batch_size, in_features)
    types = torch.randint(0, num_types, (batch_size,))
    
    out = typed_linear(x, types)
    assert out.shape == (batch_size, out_features), f"Expected shape {(batch_size, out_features)}, got {out.shape}"
    
    print(f"✓ Input shape: {x.shape}")
    print(f"✓ Output shape: {out.shape}")
    print(f"✓ Types: {types.tolist()}")
    print("✓ TypedLinear test passed!")


def test_rgcn():
    """Test RGCN layer"""
    print("\n" + "="*50)
    print("Testing RGCN")
    print("="*50)
    
    num_nodes = 20
    num_edges = 50
    in_dim = 64
    hid_dim = 128
    out_dim = 64
    num_rels = 5
    
    rgcn = RGCN(in_dim, hid_dim, out_dim, n_layers=2, num_rels=num_rels)
    
    x = torch.randn(num_nodes, in_dim)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_type = torch.randint(0, num_rels, (num_edges,))
    
    out = rgcn(x, edge_index, edge_type)
    assert out.shape == (num_nodes, out_dim), f"Expected shape {(num_nodes, out_dim)}, got {out.shape}"
    
    print(f"✓ Input shape: {x.shape}")
    print(f"✓ Edge index shape: {edge_index.shape}")
    print(f"✓ Output shape: {out.shape}")
    print("✓ RGCN test passed!")


def test_graph_transformer():
    """Test GraphTransformer layer"""
    print("\n" + "="*50)
    print("Testing GraphTransformer")
    print("="*50)
    
    num_nodes = 20
    num_edges = 50
    in_dim = 64
    hid_dim = 128
    out_dim = 64
    num_heads = 8
    num_ntypes = 3
    num_etypes = 5
    
    kgt = KGTransformer(
        in_dim, hid_dim, num_heads, out_dim,
        n_layers=2,
        num_nodes=num_ntypes,
        num_rels=num_etypes
    )
    
    x = torch.randn(num_nodes, in_dim)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    ntype = torch.randint(0, num_ntypes, (num_nodes,))
    etype = torch.randint(0, num_etypes, (num_edges,))
    
    out = kgt(x, edge_index, ntype, etype)
    assert out.shape == (num_nodes, out_dim), f"Expected shape {(num_nodes, out_dim)}, got {out.shape}"
    
    print(f"✓ Input shape: {x.shape}")
    print(f"✓ Edge index shape: {edge_index.shape}")
    print(f"✓ Node types shape: {ntype.shape}")
    print(f"✓ Edge types shape: {etype.shape}")
    print(f"✓ Output shape: {out.shape}")
    print("✓ GraphTransformer test passed!")


def test_data_loading():
    """Test data loading"""
    print("\n" + "="*50)
    print("Testing Data Loading")
    print("="*50)
    
    data_root = "data/FinDKG"
    
    try:
        train_dataset, val_dataset, test_dataset, num_entities, num_relations = \
            TemporalKGDatasetPyG.from_txt_files(
                f"{data_root}/train.txt",
                f"{data_root}/valid.txt",
                f"{data_root}/test.txt",
                f"{data_root}/entity2id.txt"
            )
        
        print(f"✓ Num entities: {num_entities}")
        print(f"✓ Num relations: {num_relations}")
        print(f"✓ Train timestamps: {len(train_dataset)}")
        print(f"✓ Val timestamps: {len(val_dataset)}")
        print(f"✓ Test timestamps: {len(test_dataset)}")
        
        # Test getting a batch
        batch = train_dataset[0]
        print(f"✓ First batch nodes: {batch.num_nodes}")
        print(f"✓ First batch edges: {batch.edge_index.size(1)}")
        print(f"✓ First batch timestamp: {batch.timestamp}")
        
        print("✓ Data loading test passed!")
        return train_dataset, val_dataset, test_dataset, num_entities, num_relations
    
    except FileNotFoundError as e:
        print(f"⚠ Data files not found: {e}")
        print("  Skipping data loading test")
        return None, None, None, None, None


def test_model_instantiation(num_entities, num_relations):
    """Test model instantiation"""
    print("\n" + "="*50)
    print("Testing Model Instantiation")
    print("="*50)
    
    if num_entities is None or num_relations is None:
        print("  Skipping (no data loaded)")
        return None
    
    config = ConfigArgs()
    config.device = 'cpu'
    config.epochs = 1
    
    model = KGTransformerPyG(num_entities, num_relations, config)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Model created successfully")
    print(f"✓ Total parameters: {num_params:,}")
    print(f"✓ Device: {config.device}")
    
    print("✓ Model instantiation test passed!")
    return model


def test_forward_pass(model, train_dataset):
    """Test forward pass"""
    print("\n" + "="*50)
    print("Testing Forward Pass")
    print("="*50)
    
    if model is None or train_dataset is None:
        print("  Skipping (no model or data)")
        return
    
    # Get first batch
    batch_data = train_dataset[0]
    
    print(f"  Batch timestamp: {batch_data.timestamp}")
    print(f"  Batch nodes: {batch_data.num_nodes}")
    print(f"  Batch edges: {batch_data.edge_index.size(1)}")
    
    # Forward pass
    try:
        log_prob, tail_pred = model(batch_data)
        
        print(f"✓ Log prob shape: {log_prob.shape if hasattr(log_prob, 'shape') else 'scalar'}")
        print(f"✓ Tail pred shape: {tail_pred.shape}")
        print(f"✓ Log prob value: {log_prob.item() if hasattr(log_prob, 'item') else log_prob}")
        
        print("✓ Forward pass test passed!")
        return True
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_backward_pass(model, train_dataset):
    """Test backward pass"""
    print("\n" + "="*50)
    print("Testing Backward Pass")
    print("="*50)
    
    if model is None or train_dataset is None:
        print("  Skipping (no model or data)")
        return
    
    # Get first batch
    batch_data = train_dataset[0]
    
    # Forward pass
    try:
        log_prob, tail_pred = model(batch_data)
        loss = -log_prob
        
        print(f"  Loss: {loss.item()}")
        
        # Backward pass
        loss.backward()
        
        # Check gradients
        num_params_with_grad = sum(1 for p in model.parameters() if p.grad is not None)
        total_params = sum(1 for _ in model.parameters())
        
        print(f"✓ Backward pass completed")
        print(f"✓ Parameters with gradients: {num_params_with_grad}/{total_params}")
        
        print("✓ Backward pass test passed!")
        return True
    except Exception as e:
        print(f"✗ Backward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "="*70)
    print(" "*15 + "KGTransformer PyG Implementation Test")
    print("="*70)
    
    # Test individual components
    test_typed_linear()
    test_rgcn()
    test_graph_transformer()
    
    # Test data loading
    train_dataset, val_dataset, test_dataset, num_entities, num_relations = test_data_loading()
    
    # Test model
    model = test_model_instantiation(num_entities, num_relations)
    
    # Test forward and backward passes
    if model is not None and train_dataset is not None:
        forward_ok = test_forward_pass(model, train_dataset)
        if forward_ok:
            test_backward_pass(model, train_dataset)
    
    print("\n" + "="*70)
    print(" "*20 + "All Tests Completed!")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()

