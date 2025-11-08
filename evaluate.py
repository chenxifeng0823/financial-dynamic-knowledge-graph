"""
Evaluation script for trained knowledge graph embedding models.

Usage:
    python evaluate.py --model transe --checkpoint checkpoints/best_model.pth
    python evaluate.py --model distmult --checkpoint checkpoints/distmult_final.pth --filtered
"""

import argparse
import torch
import numpy as np
from pathlib import Path

from src.data_processing.dataset import TemporalKGDataset
from src.models.transe import TransE
from src.models.distmult import DistMult
from src.models.complex import ComplEx
from src.models.temporal_transe import TemporalTransE
from src.evaluation import evaluate_model


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate KG embedding models')
    
    # Data
    parser.add_argument('--data_path', type=str,
                        default='./data/FinDKG_repo/FinDKG_dataset/FinDKG',
                        help='Path to dataset')
    parser.add_argument('--split', type=str, default='test',
                        choices=['valid', 'test'],
                        help='Which split to evaluate on')
    
    # Model
    parser.add_argument('--model', type=str, required=True,
                        choices=['transe', 'distmult', 'complex', 'temporal_transe'],
                        help='Model to evaluate')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--embedding_dim', type=int, default=100,
                        help='Embedding dimension')
    
    # Evaluation
    parser.add_argument('--batch_size', type=int, default=100,
                        help='Batch size for evaluation')
    parser.add_argument('--filtered', action='store_true',
                        help='Use filtered evaluation')
    
    # System
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'],
                        help='Device to use')
    
    return parser.parse_args()


def load_model(args, num_entities, num_relations, num_timestamps=None):
    """Load model from checkpoint."""
    if args.model == 'transe':
        model = TransE(
            num_entities=num_entities,
            num_relations=num_relations,
            embedding_dim=args.embedding_dim
        )
    elif args.model == 'temporal_transe':
        if num_timestamps is None:
            raise ValueError("num_timestamps required for temporal models")
        model = TemporalTransE(
            num_entities=num_entities,
            num_relations=num_relations,
            num_timestamps=num_timestamps,
            embedding_dim=args.embedding_dim,
            time_dim=args.embedding_dim // 2
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
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    return model


def main():
    args = parse_args()
    
    # Device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print("=" * 70)
    print("Knowledge Graph Model Evaluation")
    print("=" * 70)
    print(f"Model: {args.model.upper()}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Split: {args.split}")
    print(f"Device: {device}")
    print(f"Filtered: {args.filtered}")
    print("=" * 70)
    
    # Load data
    print("\nLoading data...")
    dataset = TemporalKGDataset(args.data_path, split=args.split)
    num_entities = dataset.num_entities
    num_relations = dataset.num_relations
    
    # Get num_timestamps for temporal models
    num_timestamps = int(dataset.triplets[:, 3].max()) + 1
    
    print(f"Entities: {num_entities:,}")
    print(f"Relations: {num_relations}")
    print(f"Timestamps: {num_timestamps}")
    print(f"Test triplets: {len(dataset):,}")
    
    # Load model
    print(f"\nLoading model from {args.checkpoint}...")
    model = load_model(args, num_entities, num_relations, num_timestamps)
    model = model.to(device)
    model.eval()
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Prepare data for evaluation
    test_triplets = dataset.triplets
    
    # Load all triplets for filtered evaluation
    all_triplets = None
    if args.filtered:
        print("\nLoading all triplets for filtered evaluation...")
        train_dataset = TemporalKGDataset(args.data_path, split='train')
        valid_dataset = TemporalKGDataset(args.data_path, split='valid')
        all_triplets = np.concatenate([
            train_dataset.triplets,
            valid_dataset.triplets,
            test_triplets
        ])
        print(f"Total valid triplets: {len(all_triplets):,}")
    
    # Evaluate
    metrics = evaluate_model(
        model=model,
        test_data=test_triplets,
        num_entities=num_entities,
        batch_size=args.batch_size,
        filtered=args.filtered,
        all_triplets=all_triplets
    )
    
    # Save results
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    
    results_file = results_dir / f"{args.model}_{args.split}_{'filtered' if args.filtered else 'raw'}.txt"
    with open(results_file, 'w') as f:
        f.write(f"Model: {args.model}\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Split: {args.split}\n")
        f.write(f"Filtered: {args.filtered}\n")
        f.write(f"\nMetrics:\n")
        for key, value in metrics.items():
            f.write(f"  {key}: {value:.4f}\n")
    
    print(f"\nResults saved to: {results_file}")
    print("\n" + "=" * 70)
    print("Evaluation completed!")
    print("=" * 70)


if __name__ == '__main__':
    main()

