"""
Training script for KGTransformer (PyG implementation)
Matches original FinDKG hyperparameters
"""

import argparse
import os
import sys
import torch
import torch.nn.functional as F
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.data_processing.pyg_dataset import (
    TemporalKGDatasetPyG,
    create_temporal_dataloaders,
    CumulativeGraphBuilder
)
from src.models.pyg_kgtransformer import KGTransformerPyG, ConfigArgs


def train_epoch(model, train_loader, optimizer, epoch, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = 0
    
    # Initialize cumulative graph builder
    entity_types = train_loader.dataset.entity_types
    cumul_builder = CumulativeGraphBuilder(
        train_loader.dataset.num_entities,
        train_loader.dataset.num_relations,
        entity_types
    )
    
    for batch_idx, batch_data in enumerate(train_loader):
        optimizer.zero_grad()
        
        # Forward pass
        log_prob, tail_pred = model(batch_data)
        
        # Loss (negative log likelihood)
        loss = -log_prob
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        # Add batch to cumulative graph
        cumul_data = cumul_builder.add_batch(batch_data)
        
        if batch_idx % 10 == 0:
            print(f'  Batch [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f}')
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    print(f'Epoch {epoch}: Average Loss: {avg_loss:.4f}')
    
    return avg_loss


@torch.no_grad()
def evaluate(model, loader, device):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    # Initialize cumulative graph builder
    entity_types = loader.dataset.entity_types
    cumul_builder = CumulativeGraphBuilder(
        loader.dataset.num_entities,
        loader.dataset.num_relations,
        entity_types
    )
    
    for batch_data in loader:
        # Forward pass
        log_prob, tail_pred = model(batch_data)
        
        # Loss
        loss = -log_prob
        total_loss += loss.item()
        num_batches += 1
        
        # Add batch to cumulative graph
        cumul_data = cumul_builder.add_batch(batch_data)
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    return avg_loss


def main(args):
    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    print(f"Using device: {args.device}")
    
    # Load data
    data_root = Path(args.data_root)
    print(f"\nLoading data from {data_root}...")
    
    train_dataset, val_dataset, test_dataset, num_entities, num_relations = \
        TemporalKGDatasetPyG.from_txt_files(
            str(data_root / "train.txt"),
            str(data_root / "valid.txt"),
            str(data_root / "test.txt"),
            str(data_root / "entity2id.txt")
        )
    
    print(f"\nDataset statistics:")
    print(f"  Entities: {num_entities}")
    print(f"  Relations: {num_relations}")
    print(f"  Train timestamps: {len(train_dataset)}")
    print(f"  Val timestamps: {len(val_dataset)}")
    print(f"  Test timestamps: {len(test_dataset)}")
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_temporal_dataloaders(
        train_dataset, val_dataset, test_dataset
    )
    
    # Create config
    config = ConfigArgs()
    config.device = args.device
    config.epochs = args.epochs
    config.lr = args.lr
    
    # Initialize model
    print(f"\nInitializing KGTransformer...")
    print(f"  Graph conv: {config.embedding_updater_structural_gconv}")
    print(f"  Num layers: {config.num_gconv_layers}")
    print(f"  Num heads: {config.num_attn_heads}")
    print(f"  Embedding dim: {config.static_entity_embed_dim}")
    
    model = KGTransformerPyG(num_entities, num_relations, config)
    model = model.to(args.device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {num_params:,}")
    
    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay
    )
    
    # Training loop
    print(f"\nStarting training for {config.epochs} epochs...")
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(1, config.epochs + 1):
        print(f"\nEpoch {epoch}/{config.epochs}")
        print("-" * 50)
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, epoch, args.device)
        
        # Validate
        val_loss = evaluate(model, val_loader, args.device)
        print(f'Validation Loss: {val_loss:.4f}')
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save best model
            if args.save_model:
                save_path = Path(args.save_dir) / "kgt_pyg_best.pt"
                save_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'config': config
                }, save_path)
                print(f"  Model saved to {save_path}")
        else:
            patience_counter += 1
            if config.early_stop and patience_counter >= config.patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break
    
    # Test
    print(f"\nEvaluating on test set...")
    test_loss = evaluate(model, test_loader, args.device)
    print(f'Test Loss: {test_loss:.4f}')
    
    print("\nTraining completed!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train KGTransformer (PyG)')
    parser.add_argument('--data_root', type=str, default='data/FinDKG',
                        help='Path to data directory')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--seed', type=int, default=101,
                        help='Random seed for reproducibility')
    parser.add_argument('--save_model', action='store_true',
                        help='Save the best model')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                        help='Directory to save models')
    
    args = parser.parse_args()
    main(args)

