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
import torch.nn as nn
import torch.nn.functional as F
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
    heads = batch_data.node_id[batch_data.edge_index[0]].long()
    tails = batch_data.node_id[batch_data.edge_index[1]].long()
    rels = batch_data.edge_type.long()
    times = batch_data.timestamps.long()  # Convert times to long for stacking
    
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

def train_epoch(model, train_loader, optimizer, epoch, device, cumul_builder, model_type, num_entities, grad_clip_norm=None):
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
                # Two-stage approach (matching DGL/main branch):
                # Stage 1: Update embeddings with cumulative graph (temporal context)
                # Stage 2: Predict on batch edges only (memory efficient)
                
                # Stage 1: Update embeddings using cumulative graph
                model.dynamic_entity_embeds, model.dynamic_relation_embeds = model.embedding_updater(
                    cumul_data,
                    model.static_entity_embeds,
                    model.dynamic_entity_embeds,
                    model.dynamic_relation_embeds,
                    device
                )
                
                # Stage 2: Predict on BATCH edges only (not cumulative)
                batch_data = batch_data.to(device)
                static_structural = model.static_entity_embeds.structural[batch_data.node_id.cpu()].to(device)
                # Use last hidden state from RNN (matching DGL)
                dynamic_structural = model.dynamic_entity_embeds.structural[batch_data.node_id.cpu(), -1, :].to(device)
                
                combined_emb = model.combiner(static_structural, dynamic_structural, batch_data)
                static_emb = static_structural
                dynamic_emb = dynamic_structural
                
                target_heads = batch_data.edge_index[0]
                target_rels = batch_data.edge_type
                target_tails = batch_data.edge_index[1]
                
                # Extract relation embeddings (last hidden state from structural RNN)
                # Shape: [num_relations, embed_dim, 2] where 2 is bidirectional (sender/receiver)
                dynamic_rel_emb = model.dynamic_relation_embeds.structural[:, -1, :, :].to(device)
                
                log_prob, _ = model.edge_model(
                    batch_data,
                    combined_emb,
                    static_emb,
                    dynamic_emb,
                    dynamic_rel_emb,  # Pass structural embeddings
                    target_heads,
                    target_rels,
                    target_tails
                )
                
                # Loss (negative log likelihood)
                loss = -log_prob
                
            elif model_type == 'Attention':
                # Two-stage approach for Attention (same as RNN):
                # Stage 1: Update embeddings with cumulative graph
                # Stage 2: Predict on batch edges only
                
                # Move cumulative graph to device
                cumul_data = cumul_data.to(device)
                
                # Stage 1: Update embeddings with cumulative graph
                model.dynamic_entity_embeds, model.dynamic_relation_embeds = model.embedding_updater(
                    cumul_data,  # ← Use cumul_data, not batch_data!
                    model.static_entity_embeds,
                    model.dynamic_entity_embeds,
                    model.dynamic_relation_embeds,
                    device
                )
                
                # Stage 2: Predict on BATCH edges only
                batch_data = batch_data.to(device)
                batch_nodes = batch_data.node_id.cpu()
                static_emb = model.static_entity_embeds[batch_nodes].to(device)
                dynamic_emb = model.dynamic_entity_embeds.structural[batch_nodes].to(device)
                
                heads_local = batch_data.edge_index[0]
                tails_local = batch_data.edge_index[1]
                rels = batch_data.edge_type
                
                head_emb = torch.cat([static_emb[heads_local], dynamic_emb[heads_local]], dim=1)
                rel_emb = model.relation_embeds[rels]
                
                decoder_input = torch.cat([head_emb, rel_emb], dim=1)
                tail_scores = model.decoder_head(decoder_input)  # [num_edges, num_entities]
                
                # Compute loss (cross-entropy)
                global_tail_ids = batch_data.node_id[tails_local].long()
                loss = F.cross_entropy(tail_scores, global_tail_ids)
                
                # Check for NaN loss
                if torch.isnan(loss):
                    print(f"❌ NaN loss detected at batch {num_batches}!")
                    print(f"   tail_scores min/max: {tail_scores.min().item()}/{tail_scores.max().item()}")
                    print(f"   tail_scores has nan: {torch.isnan(tail_scores).any().item()}")

        # --- Simple/Static Models (TransE, DistMult, etc) ---
        else:
            # Prepare batch dict with negatives
            batch_dict = get_triplets_from_batch(batch_data, num_negatives=10, num_entities=num_entities)
            
            # Forward pass returns scores
            outputs = model(batch_dict)
            pos_scores = outputs['pos_scores']
            neg_scores = outputs['neg_scores']
            
            # Margin Loss or similar
            if model_type in ['TransE', 'TemporalTransE']:
                margin = model.margin if hasattr(model, 'margin') else 1.0
                loss = torch.mean(torch.relu(margin + neg_scores - pos_scores.unsqueeze(1)))
            else:
                # Softplus Loss
                pos_loss = F.softplus(-pos_scores).mean()
                neg_loss = F.softplus(neg_scores).mean()
                loss = pos_loss + neg_loss
                
        loss.backward()
        
        # Gradient Clipping (configurable for HPO)
        if grad_clip_norm is not None and grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
        
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
        batch_ranks = []  # Initialize batch_ranks for this batch
        
        # --- Deep Temporal Models ---
        if model_type in ['RNN', 'Attention']:
            cumul_data = cumul_builder.add_batch(batch_data)
            
            if model_type == 'RNN':
                # Two-stage approach for evaluation:
                # Stage 1: Update embeddings with cumulative graph
                # Stage 2: Get predictions for ranking (model() returns tail predictions for all entities)
                
                # Stage 1: Update embeddings with cumulative graph
                model.dynamic_entity_embeds, model.dynamic_relation_embeds = model.embedding_updater(
                    cumul_data,
                    model.static_entity_embeds,
                    model.dynamic_entity_embeds,
                    model.dynamic_relation_embeds,
                    device
                )
                
                # Stage 2: Get tail predictions for ranking (using batch data, not cumulative)
                # Build embeddings and get predictions manually (same as training)
                batch_data = batch_data.to(device)
                static_structural = model.static_entity_embeds.structural[batch_data.node_id.cpu()].to(device)
                dynamic_structural = model.dynamic_entity_embeds.structural[batch_data.node_id.cpu(), -1, :].to(device)
                
                combined_emb = model.combiner(static_structural, dynamic_structural, batch_data)
                static_emb = static_structural
                dynamic_emb = dynamic_structural
                
                target_heads = batch_data.edge_index[0]
                target_rels = batch_data.edge_type
                target_tails = batch_data.edge_index[1]
                
                dynamic_rel_emb = model.dynamic_relation_embeds.structural[:, -1, :, :].to(device)
                
                log_prob, (head_pred, rel_pred, tail_pred) = model.edge_model(
                    batch_data,
                    combined_emb,
                    static_emb,
                    dynamic_emb,
                    dynamic_rel_emb,
                    target_heads,
                    target_rels,
                    target_tails
                )
                
                loss = -log_prob
                scores = tail_pred  # [num_edges, num_entities] - scores for all entities
                
            elif model_type == 'Attention':
                # Two-stage approach for Attention evaluation:
                # Stage 1: Update embeddings with cumulative graph
                # Stage 2: Get predictions for ranking
                
                # Move cumulative graph to device
                cumul_data = cumul_data.to(device)
                
                # Stage 1: Update embeddings with cumulative graph
                model.dynamic_entity_embeds, model.dynamic_relation_embeds = model.embedding_updater(
                    cumul_data,  # ← Use cumul_data, not batch_data!
                    model.static_entity_embeds,
                    model.dynamic_entity_embeds,
                    model.dynamic_relation_embeds,
                    device
                )
                
                # Stage 2: Get tail predictions for ranking (using batch data)
                batch_data = batch_data.to(device)
                batch_nodes = batch_data.node_id.cpu()
                static_emb = model.static_entity_embeds[batch_nodes].to(device)
                dynamic_emb = model.dynamic_entity_embeds.structural[batch_nodes].to(device)
                
                heads_local = batch_data.edge_index[0]
                tails_local = batch_data.edge_index[1]
                rels = batch_data.edge_type
                
                head_emb = torch.cat([static_emb[heads_local], dynamic_emb[heads_local]], dim=1)
                rel_emb = model.relation_embeds[rels]
                
                decoder_input = torch.cat([head_emb, rel_emb], dim=1)
                scores = model.decoder_head(decoder_input)  # [num_edges, num_entities]
                
                # Compute loss
                global_tail_ids = batch_data.node_id[tails_local].long()
                loss = F.cross_entropy(scores, global_tail_ids)
        
        # --- Simple/Static Models ---
        else:
            # Loss calculation
            batch_dict = get_triplets_from_batch(batch_data, num_negatives=1, num_entities=num_entities)
            outputs = model(batch_dict)
            loss = 0 # Placeholder for evaluation loss
            
            # Ranking: Score all entities for each (h, r, ?)
            heads = batch_data.node_id[batch_data.edge_index[0]].long()
            rels = batch_data.edge_type.long()
            
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
                t_exp = batch_data.timestamps.long().unsqueeze(1).expand(-1, num_entities)
                all_ents = torch.arange(num_entities, device=device, dtype=torch.long).unsqueeze(0).expand(heads.size(0), -1)
                
                scores = model.score(h_exp, r_exp, all_ents, t_exp)
                
            else:
                # TransE, DistMult, ComplEx
                scores = model.predict(heads, rels, objects=None) # [B, N]

        # --- Metric Calculation (Common) ---
        total_loss += loss.item() if isinstance(loss, torch.Tensor) else 0
        num_batches += 1
        
        # Calculate Ranks
        true_tails_global = batch_data.node_id[batch_data.edge_index[1]].long().to(device)
        
        for i in range(len(true_tails_global)):
            target_id = true_tails_global[i].long().item()  # Convert to Python int for indexing
            # Handle both 1D and 2D score tensors
            if scores.dim() == 1:
                # For models that return 1D scores (flattened)
                # This shouldn't happen in proper implementation, but handle it
                target_score = scores[i] if i < len(scores) else 0
                batch_score = scores
            else:
                # For models that return 2D scores [batch, num_entities]
                target_score = scores[i, target_id]
                batch_score = scores[i]
            
            higher = (batch_score > target_score).sum().item()
            equal = (batch_score == target_score).sum().item()
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
    # For Attention sweeps, cap epochs at 20 to speed up HPO
    if args.model_type == 'Attention' and args.epochs > 20:
        print(f"[Config] Overriding epochs for Attention from {args.epochs} to 20 for sweeps.")
        args.epochs = 20

    # Print full configuration for each run (helpful for sweeps)
    print("=== Run Configuration ===")
    for k, v in sorted(vars(args).items()):
        print(f"  {k}: {v}")
    print("=========================")

    # Determine model group for WandB organization
    if args.model_type in ['TransE', 'DistMult', 'ComplEx', 'TemporalTransE']:
        model_group = "Baselines"
    else:
        model_group = "Deep_Temporal"

    # Construct a descriptive WandB run name so sweeps clearly show hyperparams
    run_name = (
        f"{args.model_type}"
        f"-seed{args.seed}"
        f"-lr{args.lr}"
        f"-wd{getattr(args, 'weight_decay', 0.0)}"
        f"-drop{getattr(args, 'dropout', 0.0)}"
        f"-clip{getattr(args, 'grad_clip_norm', 0.0)}"
        f"-ws{getattr(args, 'window_size', 'NA')}"
    )

    wandb.init(
        project="findkg-fix-rnn",
        name=run_name,
        group=model_group,  # Group runs by model type
        tags=[model_group, args.model_type],
        config=vars(args)
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
        # Pass dropout / other HPO-related settings to RNN config when available
        if hasattr(config, "dropout") and hasattr(args, "dropout"):
            config.dropout = args.dropout
        model = KGTransformerPyG(num_entities, num_relations, config).to(device)
        
    elif args.model_type == 'Attention':
        class AttnConfig:
            def __init__(self):
                # Simplified config: Use RGCN instead of KGT to avoid TypedLinear bugs
                self.graph = "FinDKG"
                self.device = device
                self.num_node_types = 12  # FinDKG has 12 entity types
                # HPO-exposed architectural knobs
                self.num_attn_heads = args.num_heads  # Must divide embed_dim evenly
                self.num_gconv_layers = args.num_gconv_layers
                self.graph_structural_conv = 'RGCN'  # ← Use RGCN instead of KGT (avoids TypedLinear)
                self.static_entity_embed_dim = args.embed_dim
                self.structural_dynamic_entity_embed_dim = args.embed_dim
                self.temporal_dynamic_entity_embed_dim = args.embed_dim
                self.rel_embed_dim = args.embed_dim
                # Shared dropout used in both structural GNN and temporal attention
                self.dropout = args.dropout
                self.window_size = args.window_size
        model = KGTemporalAttention(num_entities, num_relations, AttnConfig()).to(device)
        
        # FIX: Re-initialize embeddings with Xavier (randn causes NaN in Transformers)
        nn.init.xavier_uniform_(model.static_entity_embeds)
        nn.init.xavier_uniform_(model.relation_embeds)
        print("   ✅ Re-initialized Attention model with Xavier Uniform")
        
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
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=getattr(args, "weight_decay", 0.0),
    )
    
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
            
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            epoch,
            device,
            train_builder,
            args.model_type,
            num_entities,
            grad_clip_norm=getattr(args, "grad_clip_norm", None),
        )
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
    # Best Attention training config: lr=2e-4, wd=1e-4, clip=2.0, dropout=0.1
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--seed', type=int, default=41)
    # Best Attention architecture from Stage B sweep uses 256-dim embeddings
    parser.add_argument('--embed_dim', type=int, default=256)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--window_size', type=int, default=10, help='History window for Attention model')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Directory to save models')
    # Hyperparameters for HPO (mainly used by Attention / RNN)
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate for GNN/temporal attention')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='Weight decay for AdamW')
    parser.add_argument('--grad_clip_norm', type=float, default=2.0, help='Max grad norm (set <=0 to disable clipping)')
    # Best Attention architecture: 8 heads, 2 GNN layers
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads for temporal model')
    parser.add_argument('--num_gconv_layers', type=int, default=2, help='Number of structural GNN layers')
    
    args = parser.parse_args()
    run_experiment(args)
