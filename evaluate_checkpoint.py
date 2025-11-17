"""
Evaluate a trained KGTransformer checkpoint with comprehensive metrics.

Usage:
    python evaluate_checkpoint.py --checkpoint checkpoints/kgt_pyg_best.pt --split test
    python evaluate_checkpoint.py --checkpoint checkpoints/kgt_pyg_best.pt --split valid
"""

import argparse
import torch
from pathlib import Path
from torch_geometric.loader import DataLoader

from src.data_processing.pyg_dataset import TemporalKGDatasetPyG, CumulativeGraphBuilder
from src.models.pyg_kgtransformer import KGTransformerPyG
from src.evaluation.temporal_metrics import (
    compute_ranking_metrics,
    print_ranking_metrics,
    save_results_to_file
)
from tqdm import tqdm


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


def main():
    parser = argparse.ArgumentParser(description='Evaluate KGTransformer checkpoint')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint file')
    parser.add_argument('--split', type=str, default='test',
                        choices=['valid', 'test'],
                        help='Which split to evaluate on')
    parser.add_argument('--data_root', type=str, default='data/FinDKG',
                        help='Path to data directory')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size (temporal batching)')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("KGTransformer Checkpoint Evaluation")
    print("=" * 70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Split: {args.split}")
    print(f"Device: {args.device}")
    print("=" * 70)
    
    # Load checkpoint
    print("\nLoading checkpoint...")
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    
    if 'config' not in checkpoint:
        raise ValueError("Checkpoint does not contain config!")
    
    config = checkpoint['config']
    print(f"\nModel configuration:")
    print(f"  Embedding dim: {config.static_entity_embed_dim}")
    print(f"  GNN layers: {config.num_gconv_layers}")
    print(f"  RNN layers: {config.num_rnn_layers}")
    print(f"  Attention heads: {config.num_attn_heads}")
    
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
    
    # Select dataset
    eval_dataset = test_dataset if args.split == 'test' else val_dataset
    
    print(f"\nDataset statistics:")
    print(f"  Entities: {num_entities:,}")
    print(f"  Relations: {num_relations}")
    print(f"  {args.split.capitalize()} triplets: {len(eval_dataset):,}")
    
    # Create data loader
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # Create model
    print("\nCreating model...")
    model = KGTransformerPyG(
        num_entities=num_entities,
        num_relations=num_relations,
        args=config  # config is actually the args object
    ).to(args.device)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Evaluate
    print("\n" + "=" * 70)
    print(f"Starting {args.split.upper()} Evaluation")
    print("=" * 70)
    
    metrics, ranks = evaluate_with_rankings(
        model, eval_loader, args.device, num_entities
    )
    
    # Print results
    print_ranking_metrics(metrics, phase=args.split.upper())
    
    # Save results
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    
    checkpoint_name = Path(args.checkpoint).stem
    results_file = results_dir / f"{checkpoint_name}_{args.split}_results.txt"
    
    save_results_to_file(
        str(results_file),
        metrics,
        config={
            'checkpoint': args.checkpoint,
            'split': args.split,
            'num_entities': num_entities,
            'num_relations': num_relations,
            'num_evaluated': len(ranks)
        }
    )
    
    print(f"\nResults saved to: {results_file}")
    
    # Also save in FinDKG format
    findkg_file = results_dir / f"{checkpoint_name}_{args.split}_findkg_format.txt"
    with open(findkg_file, 'w') as f:
        f.write(f"0,{metrics['REC1']:.6f},{metrics['REC3']:.6f},"
               f"{metrics['REC10']:.6f},{metrics['MRR']:.6f}\n")
    
    print(f"FinDKG format results saved to: {findkg_file}")
    
    print("\n" + "=" * 70)
    print("Evaluation completed!")
    print("=" * 70)


if __name__ == '__main__':
    main()

