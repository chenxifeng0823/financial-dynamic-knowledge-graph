"""
Main Experiment Script for Financial Dynamic Knowledge Graph
Compares Multiple Models:
1. Static Baselines: TransE, DistMult, ComplEx
2. Simple Temporal: TemporalTransE
3. Deep Temporal: KGTransformer (RNN), Temporal Attention (Transformer)
"""

import argparse
import os
import sys
import torch
import wandb
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.data_processing.pyg_dataset import (
    TemporalKGDatasetPyG,
    create_temporal_dataloaders,
    CumulativeGraphBuilder
)
from src.models.pyg_kgtransformer import KGTransformerPyG, ConfigArgs as BaselineConfig
from src.models.kg_temporal_attention import KGTemporalAttention
from src.models.transe import TransE
from src.models.distmult import DistMult
from src.models.complex import ComplEx
from src.models.temporal_transe import TemporalTransE

def get_triplets_from_batch(batch_data, num_negatives=1, num_entities=None):
    """
    Extracts positive and negative triplets from a PyG batch for simple models.
    Simple models expect dictionary with 'positive' and 'negatives'.
    """
    # Positive: [Batch_Edges, 4] -> (h, r, t, time)
    heads = batch_data.node_id[batch_data.edge_index[0]]
    tails = batch_data.node_id[batch_data.edge_index[1]]
    rels = batch_data.edge_type
    times = batch_data.timestamps
    
    pos_triplets = torch.stack([heads, rels, tails, times], dim=1)
    
    # Negative Sampling (Random corruption of tails)
    # [Batch_Edges, Num_Negs, 4]
    batch_size = pos_triplets.size(0)
    neg_tails = torch.randint(0, num_entities, (batch_size, num_negatives), device=pos_triplets.device)
    
    neg_triplets = pos_triplets.unsqueeze(1).repeat(1, num_negatives, 1)
    neg_triplets[:, :, 2] = neg_tails # Replace tails
    
    return {
        'positive': pos_triplets,
        'negatives': neg_triplets
    }

def train_epoch(model, train_loader, optimizer, epoch, device, cumul_builder, model_type, num_entities):
    """Unified training epoch"""
    model.train()
    total_loss = 0
    num_batches = 0
    
    # Reset cumulative builder for new epoch (Only needed for deep temporal models)
    if model_type in ['RNN', 'Attention']:
        cumul_builder.reset()
    
    for batch_data in train_loader:
        batch_data = batch_data.to(device)
        optimizer.zero_grad()
        
        loss = 0
        
        # --- Deep Temporal Models (RNN, Attention) ---
        if model_type in ['RNN', 'Attention']:
            # Build cumulative graph
            cumul_data = cumul_builder.add_batch(batch_data)
            
            if model_type == 'RNN':
                # Baseline manual two-stage
                model.dynamic_entity_embeds, model.dynamic_relation_embeds = model.embedding_updater(
                    cumul_data,
                    model.static_entity_embeds,
                    model.dynamic_entity_embeds,
                    model.dynamic_relation_embeds,
                    device
                )
                log_prob, _ = model(batch_data)
                loss = -log_prob
                
            elif model_type == 'Attention':
                loss, _ = model(batch_data)
                
        # --- Simple/Static Models (TransE, DistMult, etc) ---
        else:
            # Prepare batch dict with negatives
            batch_dict = get_triplets_from_batch(batch_data, num_negatives=10, num_entities=num_entities)
            
            # Forward pass returns scores
            outputs = model(batch_dict)
            pos_scores = outputs['pos_scores']
            neg_scores = outputs['neg_scores']
            
            # Margin Loss or similar
            # TransE/TemporalTransE usually use Margin Ranking Loss
            # DistMult/ComplEx usually use Softplus/CrossEntropy
            
            if model_type in ['TransE', 'TemporalTransE']:
                # pos_scores are negative distances (higher is better)
                # We want pos > neg + margin
                # MarginRankingLoss: max(0, -y * (x1 - x2) + margin)
                # Here we manually compute: max(0, margin + neg - pos)
                margin = model.margin if hasattr(model, 'margin') else 1.0
                loss = torch.mean(torch.relu(margin + neg_scores - pos_scores.unsqueeze(1)))
            else:
                # Softplus Loss (Logistic Loss) for DistMult/ComplEx
                # log(1 + exp(-pos)) + log(1 + exp(neg))
                loss = torch.mean(torch.softplus(-pos_scores) + torch.softplus(neg_scores))
                
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
    return total_loss / num_batches if num_batches > 0 else 0

@torch.no_grad()
def evaluate(model, loader, device, cumul_builder, num_entities, model_type, phase="Val"):
    """Unified evaluation with MRR"""
    model.eval()
    all_ranks = []
    total_loss = 0
    num_batches = 0
    
    if model_type in ['RNN', 'Attention']:
        cumul_builder.reset()
    
    for batch_data in loader:
        batch_data = batch_data.to(device)
        
        # --- Deep Temporal Models ---
        if model_type in ['RNN', 'Attention']:
            cumul_data = cumul_builder.add_batch(batch_data)
            
            if model_type == 'RNN':
                model.dynamic_entity_embeds, model.dynamic_relation_embeds = model.embedding_updater(
                    cumul_data,
                    model.static_entity_embeds,
                    model.dynamic_entity_embeds,
                    model.dynamic_relation_embeds,
                    device
                )
                log_prob, tail_pred = model(batch_data)
                loss = -log_prob
                scores = tail_pred[2] # (head, rel, tail)
                
            elif model_type == 'Attention':
                loss, scores = model(batch_data)
        
        # --- Simple/Static Models ---
        else:
            # Loss calculation
            batch_dict = get_triplets_from_batch(batch_data, num_negatives=1, num_entities=num_entities)
            outputs = model(batch_dict)
            loss = 0 # Placeholder for evaluation loss
            
            # Ranking: Score all entities for each (h, r, ?)
            heads = batch_data.node_id[batch_data.edge_index[0]]
            rels = batch_data.edge_type
            
            # Simple models have a .predict() or .score() method
            # We need to score against ALL entities
            # scores shape: [Batch, Num_Entities]
            
            # For TemporalTransE, we need to pass time
            if model_type == 'TemporalTransE':
                # Create a batch of (h, r, all_t, time) is expensive
                # We usually just use the score function with broadcasting
                # But BaseKGModel.predict does exactly this
                 # But BaseKGModel.predict expects objects=None to mean "score all"
                 pass
            
            # Use predict() from BaseKGModel
            if model_type == 'TemporalTransE':
                # Pass time if supported (hacky check, ideally standardize API)
                # BaseKGModel.predict doesn't accept 'times' kwarg in signature, 
                # but TemporalTransE.score does.
                # We'll call score directly manually broadcasting
                
                # Expand h, r, time: [B, 1] -> [B, N]
                h_exp = heads.unsqueeze(1).expand(-1, num_entities)
                r_exp = rels.unsqueeze(1).expand(-1, num_entities)
                t_exp = batch_data.timestamps.unsqueeze(1).expand(-1, num_entities)
                all_ents = torch.arange(num_entities, device=device).unsqueeze(0).expand(heads.size(0), -1)
                
                scores = model.score(h_exp, r_exp, all_ents, t_exp)
                
            else:
                # TransE, DistMult, ComplEx
                scores = model.predict(heads, rels, objects=None) # [B, N]

        # --- Metric Calculation (Common) ---
        total_loss += loss.item() if isinstance(loss, torch.Tensor) else 0
        num_batches += 1
        
        # Calculate Ranks
        true_tails_global = batch_data.node_id[batch_data.edge_index[1].cpu()].to(device)
        
        for i in range(len(true_tails_global)):
            target_id = true_tails_global[i]
            target_score = scores[i, target_id]
            
            higher = (scores[i] > target_score).sum().item()
            equal = (scores[i] == target_score).sum().item()
            rank = higher + (equal - 1.0) / 2.0 + 1.0
            batch_ranks.append(rank)
            
        all_ranks.extend(batch_ranks)
        
    mrr = sum(1.0 / r for r in all_ranks) / len(all_ranks) if all_ranks else 0
    hits10 = sum(1.0 for r in all_ranks if r <= 10) / len(all_ranks) if all_ranks else 0
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    
    return {
        f"{phase}/Loss": avg_loss,
        f"{phase}/MRR": mrr,
        f"{phase}/Hits@10": hits10
    }

def run_experiment(args):
    wandb.init(
        project="findkg-model-comparison",
        name=f"{args.model_type}-seed{args.seed}",
        config=args
    )
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        
    device = args.device
    print(f"Running {args.model_type} experiment on {device} with seed {args.seed}")
    
    # Data Loading
    data_root = Path(args.data_root)
    train_dataset, val_dataset, test_dataset, num_entities, num_relations = \
        TemporalKGDatasetPyG.from_txt_files(
            str(data_root / "train.txt"),
            str(data_root / "valid.txt"),
            str(data_root / "test.txt"),
            str(data_root / "entity2id.txt")
        )
        
    train_loader, val_loader, test_loader = create_temporal_dataloaders(
        train_dataset, val_dataset, test_dataset
    )
    
    # Model Setup
    model = None
    
    if args.model_type == 'RNN':
        config = BaselineConfig()
        config.static_entity_embed_dim = args.embed_dim
        model = KGTransformerPyG(num_entities, num_relations, config).to(device)
        
    elif args.model_type == 'Attention':
        class AttnConfig:
            def __init__(self):
                self.device = device
                self.static_entity_embed_dim = args.embed_dim
                self.structural_dynamic_entity_embed_dim = args.embed_dim
                self.temporal_dynamic_entity_embed_dim = args.embed_dim
                self.rel_embed_dim = args.embed_dim
                self.dropout = 0.1
                self.window_size = args.window_size
        model = KGTemporalAttention(num_entities, num_relations, AttnConfig()).to(device)
        
    elif args.model_type == 'TransE':
        model = TransE(num_entities, num_relations, embedding_dim=args.embed_dim).to(device)
        
    elif args.model_type == 'DistMult':
        model = DistMult(num_entities, num_relations, embedding_dim=args.embed_dim).to(device)
        
    elif args.model_type == 'ComplEx':
        model = ComplEx(num_entities, num_relations, embedding_dim=args.embed_dim).to(device)
        
    elif args.model_type == 'TemporalTransE':
        # Need to know num_timestamps to init
        # We can scan dataset or just estimate safe upper bound
        all_times = np.concatenate([train_dataset.timestamps, val_dataset.timestamps, test_dataset.timestamps])
        num_timestamps = len(np.unique(all_times)) + 100 # Safety buffer for IDs if they are indices
        # Actually timestamps in FinDKG are floats/years?
        # The dataset class treats them as float timestamps.
        # TemporalTransE expects timestamp IDs (int).
        # We need to map float timestamps to IDs.
        # For this experiment, let's assume timestamps are mapped to 0..T indices by the dataset logic
        # Looking at dataset: "timestamps_array" are unique values.
        # In batch_data, timestamp is float.
        # We might need to map them on the fly or Pre-process.
        # For simplicity: Use len(all_times) and map valid times to indices.
        # HACK: Just use a large enough embedding size and cast time to int if it's year-like
        # Or better: Remap in the loop.
        
        # Proper mapping:
        unique_ts = np.unique(all_times)
        ts_map = {ts: i for i, ts in enumerate(unique_ts)}
        model = TemporalTransE(num_entities, num_relations, num_timestamps=len(unique_ts), embedding_dim=args.embed_dim).to(device)
        model.ts_map = ts_map # Attach map to model
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    # Cumulative Builders (Only for RNN/Attn)
    train_builder = CumulativeGraphBuilder(num_entities, num_relations, train_dataset.entity_types)
    val_builder = CumulativeGraphBuilder(num_entities, num_relations, val_dataset.entity_types)
    test_builder = CumulativeGraphBuilder(num_entities, num_relations, test_dataset.entity_types)
    
    # Training Loop
    best_val_mrr = 0
    
    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        
        # Hack for TemporalTransE time mapping
        if args.model_type == 'TemporalTransE':
            # Patch dataset/loader to use ID instead of float?
            # Easier: Patch the batch inside train_epoch
            # We'll do it in get_triplets_from_batch if we had passed the map
            pass
            
        train_loss = train_epoch(model, train_loader, optimizer, epoch, device, train_builder, args.model_type, num_entities)
        wandb.log({"Train/Loss": train_loss, "epoch": epoch})
        
        val_metrics = evaluate(model, val_loader, device, val_builder, num_entities, args.model_type, "Val")
        wandb.log(val_metrics)
        print(f"  Val MRR: {val_metrics['Val/MRR']:.4f}")
        
        if val_metrics['Val/MRR'] > best_val_mrr:
            best_val_mrr = val_metrics['Val/MRR']
            torch.save(model.state_dict(), os.path.join(wandb.run.dir, "best_model.pt"))
            
    # Test
    print("Loading best model for testing...")
    model.load_state_dict(torch.load(os.path.join(wandb.run.dir, "best_model.pt")))
    
    test_metrics = evaluate(model, test_loader, device, test_builder, num_entities, args.model_type, "Test")
    wandb.log(test_metrics)
    print(f"Test Results: {test_metrics}")
    
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, required=True, 
                        choices=['RNN', 'Attention', 'TransE', 'DistMult', 'ComplEx', 'TemporalTransE'])
    parser.add_argument('--data_root', type=str, default='data/FinDKG')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--seed', type=int, default=41)
    parser.add_argument('--embed_dim', type=int, default=200)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--window_size', type=int, default=10, help='History window for Attention model')
    
    args = parser.parse_args()
    run_experiment(args)
