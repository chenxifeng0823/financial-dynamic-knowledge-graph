"""
Test if gradients flow to embeddings during training
"""
import torch
from src.models.pyg_kgtransformer import KGTransformerPyG, ConfigArgs
from src.data_processing.pyg_dataset import TemporalKGDatasetPyG

print("=" * 80)
print("TESTING GRADIENT FLOW TO EMBEDDINGS")
print("=" * 80)

# Load dataset
train_dataset, _, _, num_entities, num_relations = \
    TemporalKGDatasetPyG.from_txt_files(
        'data/FinDKG/train.txt',
        'data/FinDKG/valid.txt',
        'data/FinDKG/test.txt',
        'data/FinDKG/entity2id.txt'
    )

# Create model
config = ConfigArgs()
config.device = 'cpu'
model = KGTransformerPyG(num_entities=num_entities, num_relations=num_relations, args=config)

# Get first batch
batch_data = train_dataset[0]

# Record initial embedding values
initial_static_structural = model.static_entity_embeds.structural.clone()
initial_dynamic_structural = model.dynamic_entity_embeds.structural.clone()

print("\n[BEFORE TRAINING STEP]")
print(f"Static structural embedding[0]: {initial_static_structural[0, :5]}")
print(f"Dynamic structural embedding[0]: {initial_dynamic_structural[0, 0, :5]}")

# Forward pass
model.train()
log_prob, tail_pred = model(batch_data)
loss = -log_prob

print(f"\n[FORWARD PASS]")
print(f"Loss: {loss.item():.4f}")

# Backward pass
loss.backward()

print(f"\n[AFTER BACKWARD]")

# Check if static embeddings have gradients
if model.static_entity_embeds.structural.grad is not None:
    grad_norm = model.static_entity_embeds.structural.grad.norm().item()
    grad_mean = model.static_entity_embeds.structural.grad.mean().item()
    grad_max = model.static_entity_embeds.structural.grad.abs().max().item()
    print(f"✓ Static structural embeddings have gradients:")
    print(f"  Norm: {grad_norm:.6f}")
    print(f"  Mean: {grad_mean:.6f}")
    print(f"  Max: {grad_max:.6f}")
    print(f"  Grad[0, :5]: {model.static_entity_embeds.structural.grad[0, :5]}")
else:
    print(f"❌ Static structural embeddings have NO gradients!")

if model.static_entity_embeds.temporal.grad is not None:
    grad_norm = model.static_entity_embeds.temporal.grad.norm().item()
    print(f"✓ Static temporal embeddings have gradients (norm: {grad_norm:.6f})")
else:
    print(f"❌ Static temporal embeddings have NO gradients!")

# Check if dynamic embeddings have gradients  
if model.dynamic_entity_embeds.structural.grad is not None:
    grad_norm = model.dynamic_entity_embeds.structural.grad.norm().item()
    print(f"✓ Dynamic structural embeddings have gradients (norm: {grad_norm:.6f})")
else:
    print(f"❌ Dynamic structural embeddings have NO gradients!")

if model.dynamic_entity_embeds.temporal.grad is not None:
    grad_norm = model.dynamic_entity_embeds.temporal.grad.norm().item()
    print(f"✓ Dynamic temporal embeddings have gradients (norm: {grad_norm:.6f})")
else:
    print(f"❌ Dynamic temporal embeddings have NO gradients!")

# Simulate optimizer step
optimizer = torch.optim.AdamW([
    model.static_entity_embeds.structural,
    model.static_entity_embeds.temporal,
    model.dynamic_entity_embeds.structural,
    model.dynamic_entity_embeds.temporal,
    model.dynamic_relation_embeds.structural,
    model.dynamic_relation_embeds.temporal,
] + list(model.parameters()), lr=0.001)

optimizer.step()

print(f"\n[AFTER OPTIMIZER STEP]")
print(f"Static structural embedding[0]: {model.static_entity_embeds.structural[0, :5]}")
print(f"Dynamic structural embedding[0]: {model.dynamic_entity_embeds.structural[0, 0, :5]}")

# Check if embeddings changed
static_diff = (model.static_entity_embeds.structural - initial_static_structural).abs().max().item()
dynamic_diff = (model.dynamic_entity_embeds.structural - initial_dynamic_structural).abs().max().item()

print(f"\n[EMBEDDING CHANGES]")
print(f"Max change in static structural: {static_diff:.6f}")
print(f"Max change in dynamic structural: {dynamic_diff:.6f}")

if static_diff > 1e-6:
    print("✓ Static embeddings ARE being updated!")
else:
    print("❌ Static embeddings are NOT being updated!")

if dynamic_diff > 1e-6:
    print("✓ Dynamic embeddings ARE being updated!")
else:
    print("✓ Dynamic embeddings are zero (expected at start)")

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)

if model.static_entity_embeds.structural.grad is None:
    print("""
❌ CRITICAL PROBLEM: Static embeddings have NO gradients!

This means:
1. Gradients are not flowing back to the embeddings
2. The embeddings will never update during training
3. This explains why performance is stuck at 3.6%

Possible causes:
1. Embeddings are not in the optimizer's parameter list
2. Gradients are being detached somewhere
3. There's a .detach() call blocking gradients
""")
elif static_diff < 1e-6:
    print("""
❌ PROBLEM: Embeddings have gradients but are not updating!

This means:
1. Gradients are computed correctly
2. But optimizer is not updating the embeddings
3. Embeddings might not be in optimizer's parameter list
""")
else:
    print("""
✓ Embeddings are working correctly!

If this is the case, then the problem is elsewhere:
1. Maybe learning rate is too low
2. Maybe early stopping is too aggressive
3. Maybe there's an issue with the data or evaluation
""")

