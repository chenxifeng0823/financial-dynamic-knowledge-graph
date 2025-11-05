"""
Quick test script to verify the setup works.

Run this to check if:
1. Dataset loads correctly
2. Models can be instantiated
3. Forward pass works
4. Training loop runs
"""

import torch
from src.data_processing.dataset import create_dataloader
from src.models.transe import TransE
from src.models.distmult import DistMult
from src.models.complex import ComplEx


def test_data_loading():
    """Test data loading."""
    print("=" * 70)
    print("Testing Data Loading")
    print("=" * 70)
    
    data_path = "./data/FinDKG_repo/FinDKG_dataset/FinDKG"
    
    try:
        train_loader = create_dataloader(data_path, split='train', batch_size=32)
        print("✓ Train loader created")
        
        batch = next(iter(train_loader))
        print(f"✓ Batch loaded: {batch['positive'].shape}")
        print(f"✓ Negatives: {batch['negatives'].shape}")
        
        return train_loader
    except Exception as e:
        print(f"✗ Error: {e}")
        return None


def test_models(train_loader):
    """Test model instantiation and forward pass."""
    print("\n" + "=" * 70)
    print("Testing Models")
    print("=" * 70)
    
    if train_loader is None:
        print("✗ Skipping (no data loader)")
        return
    
    dataset = train_loader.dataset
    num_entities = dataset.num_entities
    num_relations = dataset.num_relations
    
    models = {
        'TransE': TransE(num_entities, num_relations, embedding_dim=50),
        'DistMult': DistMult(num_entities, num_relations, embedding_dim=50),
        'ComplEx': ComplEx(num_entities, num_relations, embedding_dim=50)
    }
    
    batch = next(iter(train_loader))
    
    for name, model in models.items():
        try:
            outputs = model(batch)
            print(f"✓ {name}: pos_scores {outputs['pos_scores'].shape}, "
                  f"neg_scores {outputs['neg_scores'].shape}")
        except Exception as e:
            print(f"✗ {name}: {e}")


def test_training_step():
    """Test one training step."""
    print("\n" + "=" * 70)
    print("Testing Training Step")
    print("=" * 70)
    
    data_path = "./data/FinDKG_repo/FinDKG_dataset/FinDKG"
    
    try:
        # Create small loader
        train_loader = create_dataloader(data_path, split='train', batch_size=32)
        dataset = train_loader.dataset
        
        # Create model
        model = TransE(dataset.num_entities, dataset.num_relations, embedding_dim=50)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Get batch
        batch = next(iter(train_loader))
        
        # Forward
        outputs = model(batch)
        
        # Loss
        pos_scores = outputs['pos_scores'].unsqueeze(1)
        neg_scores = outputs['neg_scores']
        loss = torch.clamp(1.0 - pos_scores + neg_scores, min=0).mean()
        
        print(f"✓ Loss computed: {loss.item():.4f}")
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print("✓ Backward pass successful")
        print("✓ All tests passed!")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()


def main():
    print("\n" + "=" * 70)
    print("Financial Dynamic Knowledge Graph - Setup Test")
    print("=" * 70 + "\n")
    
    # Test data loading
    train_loader = test_data_loading()
    
    # Test models
    test_models(train_loader)
    
    # Test training
    test_training_step()
    
    print("\n" + "=" * 70)
    print("Setup test completed!")
    print("=" * 70)
    print("\nNext step: Run full training with:")
    print("  python train.py --model transe --epochs 50")


if __name__ == '__main__':
    main()

