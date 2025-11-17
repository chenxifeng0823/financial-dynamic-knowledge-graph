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
    """Evaluate model (loss only)"""
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


@torch.no_grad()
def evaluate_with_rankings(model, loader, device, num_entities):
    """
    Evaluate model with ranking metrics (MRR, Recall@K).
    
    This implements the temporal sequential evaluation protocol:
    1. Process batches in temporal order
    2. For each batch, score all possible tail entities
    3. Compute rank of true tail entity
    4. Update dynamic embeddings for next batch
    """
    from src.evaluation.temporal_metrics import compute_ranking_metrics
    
    model.eval()
    all_ranks = []
    total_loss = 0
    num_batches = 0
    
    # Initialize cumulative graph builder
    entity_types = loader.dataset.entity_types
    cumul_builder = CumulativeGraphBuilder(
        loader.dataset.num_entities,
        loader.dataset.num_relations,
        entity_types
    )
    
    from tqdm import tqdm
    batch_tqdm = tqdm(loader, desc="Evaluating")
    
    for batch_data in batch_tqdm:
        # Forward pass
        log_prob, tail_pred = model(batch_data)  # tail_pred: [num_edges, num_entities]
        
        # Loss
        loss = -log_prob
        total_loss += loss.item()
        num_batches += 1
        
        # Compute ranks for this batch
        edge_index = batch_data.edge_index
        true_tails = edge_index[1]  # True tail entities
        
        # For each edge in the batch
        for i in range(tail_pred.size(0)):
            scores = tail_pred[i]  # Scores for all entities
            true_tail = true_tails[i].item()
            true_score = scores[true_tail]
            
            # Compute rank using optimistic ranking
            # rank = (# entities with higher score) + (# entities with equal score - 1) / 2 + 1
            num_higher = (scores > true_score).sum().item()
            num_equal = (scores == true_score).sum().item()
            rank = num_higher + (num_equal - 1.0) / 2.0 + 1.0
            
            all_ranks.append(rank)
        
        # Update progress bar
        if len(all_ranks) > 0:
            current_mrr = sum(1.0 / r for r in all_ranks) / len(all_ranks)
            batch_tqdm.set_postfix({
                'MRR': f'{current_mrr:.4f}',
                'Ranks': len(all_ranks)
            })
        
        # Add batch to cumulative graph (for next iteration)
        cumul_data = cumul_builder.add_batch(batch_data)
    
    # Compute metrics
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    metrics = compute_ranking_metrics(all_ranks, k_values=[1, 3, 10, 100])
    metrics['loss'] = avg_loss
    
    return metrics, all_ranks


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
    
    # Handle early stopping arguments
    if args.no_early_stop:
        config.early_stop = False
    if args.patience is not None:
        config.patience = args.patience
    
    print(f"\nTraining configuration:")
    print(f"  Epochs: {config.epochs}")
    print(f"  Learning rate: {config.lr}")
    print(f"  Early stopping: {'Enabled' if config.early_stop else 'Disabled'}")
    if config.early_stop:
        print(f"  Patience: {config.patience}")
    
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
    optimizer = torch.optim.AdamW(
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
    
    # Test with comprehensive evaluation
    print(f"\nEvaluating on test set...")
    if args.eval_rankings:
        print("Computing ranking metrics (MRR, Recall@K)...")
        test_metrics, test_ranks = evaluate_with_rankings(
            model, test_loader, args.device, num_entities
        )
        
        # Print results
        from src.evaluation.temporal_metrics import print_ranking_metrics, save_results_to_file
        print_ranking_metrics(test_metrics, phase="TEST")
        
        # Save results
        results_dir = Path('results')
        results_dir.mkdir(exist_ok=True)
        
        results_file = results_dir / f"kgt_pyg_test_results.txt"
        save_results_to_file(
            str(results_file),
            test_metrics,
            config={
                'model': 'KGTransformer_PyG',
                'num_entities': num_entities,
                'num_relations': num_relations,
                'embedding_dim': config.static_entity_embed_dim,
                'num_epochs_trained': epoch + 1,
                'seed': args.seed
            }
        )
        print(f"\nResults saved to: {results_file}")
        
        # Also save in FinDKG format for comparison
        findkg_file = results_dir / f"kgt_pyg_test_findkg_format.txt"
        with open(findkg_file, 'w') as f:
            f.write(f"{args.seed},{test_metrics['REC1']:.6f},{test_metrics['REC3']:.6f},"
                   f"{test_metrics['REC10']:.6f},{test_metrics['MRR']:.6f}\n")
        print(f"FinDKG format results saved to: {findkg_file}")
    else:
        test_loss = evaluate(model, test_loader, args.device)
        print(f'Test Loss: {test_loss:.4f}')
        print("\n(Use --eval_rankings flag to compute MRR and Recall@K metrics)")
    
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
    parser.add_argument('--eval_rankings', action='store_true',
                        help='Compute ranking metrics (MRR, Recall@K) on test set')
    parser.add_argument('--no_early_stop', action='store_true',
                        help='Disable early stopping (train for full epochs)')
    parser.add_argument('--patience', type=int, default=None,
                        help='Early stopping patience (overrides config default)')
    
    args = parser.parse_args()
    main(args)

