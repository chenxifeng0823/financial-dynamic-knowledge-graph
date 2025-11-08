"""
Main training script for knowledge graph embedding models.

Usage:
    python train.py --model transe --epochs 100 --batch_size 512
    python train.py --model distmult --embedding_dim 200
    python train.py --model complex --lr 0.01
"""

import argparse
import torch
from pathlib import Path

from src.data_processing.dataset import create_dataloader
from src.models.transe import TransE
from src.models.distmult import DistMult
from src.models.complex import ComplEx
from src.models.temporal_transe import TemporalTransE
from src.training import KGTrainer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train KG embedding models')
    
    # Data
    parser.add_argument('--data_path', type=str, 
                        default='./data/FinDKG_repo/FinDKG_dataset/FinDKG',
                        help='Path to dataset')
    
    # Model
    parser.add_argument('--model', type=str, default='transe',
                        choices=['transe', 'distmult', 'complex', 'temporal_transe'],
                        help='Model to train')
    parser.add_argument('--embedding_dim', type=int, default=100,
                        help='Embedding dimension')
    
    # Training
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--num_negatives', type=int, default=5,
                        help='Number of negative samples')
    parser.add_argument('--loss', type=str, default='margin',
                        choices=['margin', 'bce'],
                        help='Loss function')
    
    # TransE specific
    parser.add_argument('--margin', type=float, default=1.0,
                        help='Margin for TransE')
    parser.add_argument('--p_norm', type=int, default=2,
                        help='Norm for TransE (1 or 2)')
    
    # System
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'],
                        help='Device to use')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of dataloader workers')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                        help='Checkpoint directory')
    parser.add_argument('--save_every', type=int, default=10,
                        help='Save checkpoint every N epochs')
    
    return parser.parse_args()


def create_model(args, num_entities, num_relations, num_timestamps=None):
    """Create model based on arguments."""
    if args.model == 'transe':
        model = TransE(
            num_entities=num_entities,
            num_relations=num_relations,
            embedding_dim=args.embedding_dim,
            margin=args.margin,
            p_norm=args.p_norm
        )
    elif args.model == 'temporal_transe':
        if num_timestamps is None:
            raise ValueError("num_timestamps required for temporal models")
        model = TemporalTransE(
            num_entities=num_entities,
            num_relations=num_relations,
            num_timestamps=num_timestamps,
            embedding_dim=args.embedding_dim,
            time_dim=args.embedding_dim // 2,  # Half size for time embeddings
            margin=args.margin,
            p_norm=args.p_norm
        )
    elif args.model == 'distmult':
        model = DistMult(
            num_entities=num_entities,
            num_relations=num_relations,
            embedding_dim=args.embedding_dim
        )
    elif args.model == 'complex':
        model = ComplEx(
            num_entities=num_entities,
            num_relations=num_relations,
            embedding_dim=args.embedding_dim
        )
    else:
        raise ValueError(f"Unknown model: {args.model}")
    
    return model


def main():
    args = parse_args()
    
    # Device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print("=" * 70)
    print("Financial Dynamic Knowledge Graph - Training")
    print("=" * 70)
    print(f"Model: {args.model.upper()}")
    print(f"Device: {device}")
    print(f"Embedding dim: {args.embedding_dim}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Negative samples: {args.num_negatives}")
    print(f"Epochs: {args.epochs}")
    print("=" * 70)
    
    # Load data
    print("\nLoading data...")
    train_loader = create_dataloader(
        args.data_path,
        split='train',
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        num_negatives=args.num_negatives
    )
    
    valid_loader = create_dataloader(
        args.data_path,
        split='valid',
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        num_negatives=args.num_negatives
    )
    
    # Get dataset info
    dataset = train_loader.dataset
    num_entities = dataset.num_entities
    num_relations = dataset.num_relations
    
    # Get number of unique timestamps for temporal models
    num_timestamps = int(dataset.triplets[:, 3].max()) + 1
    
    print(f"Entities: {num_entities:,}")
    print(f"Relations: {num_relations}")
    print(f"Timestamps: {num_timestamps}")
    print(f"Training triplets: {len(dataset):,}")
    print(f"Training batches: {len(train_loader)}")
    
    # Create model
    print(f"\nInitializing {args.model.upper()} model...")
    model = create_model(args, num_entities, num_relations, num_timestamps)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Trainer
    trainer = KGTrainer(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        optimizer=optimizer,
        loss_fn=args.loss,
        device=device,
        checkpoint_dir=args.checkpoint_dir
    )
    
    # Train
    trainer.train(num_epochs=args.epochs, save_every=args.save_every)
    
    # Save final model
    final_path = Path(args.checkpoint_dir) / f'{args.model}_final.pth'
    torch.save(model.state_dict(), final_path)
    print(f"\nFinal model saved to: {final_path}")
    print("\n" + "=" * 70)
    print("Training completed successfully!")
    print("=" * 70)


if __name__ == '__main__':
    main()

