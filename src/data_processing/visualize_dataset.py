"""
Simple visualization script for FinDKG dataset.

Run this after downloading the dataset to see visual statistics.
"""

import sys
from pathlib import Path
from collections import Counter

# Add parent directory to path
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root))

from src.data_processing.explore_dataset import FinDKGExplorer


def visualize_dataset():
    """Create simple text-based visualizations of the dataset."""
    
    # Initialize explorer
    explorer = FinDKGExplorer('./data/FinDKG_repo/FinDKG_dataset/FinDKG')
    
    # Load data
    print("Loading dataset...\n")
    entities = explorer.load_entities()
    relations = explorer.load_relations()
    train_triplets = explorer.load_triplets('train')
    
    # Entity type visualization
    print("\n" + "=" * 70)
    print("Entity Type Distribution (Bar Chart)")
    print("=" * 70)
    
    max_count = max(explorer.entity_types.values())
    for entity_type, count in sorted(explorer.entity_types.items(), 
                                    key=lambda x: x[1], reverse=True):
        bar_length = int((count / max_count) * 50)
        bar = '█' * bar_length
        percentage = (count / len(entities)) * 100
        print(f"{entity_type:15s} {bar:50s} {count:5,} ({percentage:5.2f}%)")
    
    # Relation distribution visualization
    print("\n" + "=" * 70)
    print("Relation Distribution in Training Set (Bar Chart)")
    print("=" * 70)
    
    relation_counts = Counter([t[1] for t in train_triplets])
    max_count = max(relation_counts.values())
    
    for rel_id, count in sorted(relation_counts.items(), 
                               key=lambda x: x[1], reverse=True):
        rel_name = relations[rel_id]
        bar_length = int((count / max_count) * 50)
        bar = '█' * bar_length
        percentage = (count / len(train_triplets)) * 100
        print(f"{rel_name:25s} {bar:50s} {count:6,} ({percentage:5.2f}%)")
    
    # Temporal distribution
    print("\n" + "=" * 70)
    print("Temporal Distribution (Timeline)")
    print("=" * 70)
    
    time_counts = Counter([t[3] for t in train_triplets])
    times = sorted(time_counts.keys())
    
    # Show every 10th time step
    print("\nTriplets per time step (showing every 10th step):")
    for i in range(0, len(times), 10):
        t = times[i]
        count = time_counts[t]
        bar_length = int((count / 2000) * 40)  # Scale to max ~2000
        bar = '█' * bar_length
        print(f"Time {t:3d}: {bar:40s} {count:4d}")
    
    # Most connected entities
    print("\n" + "=" * 70)
    print("Top 10 Most Connected Entities")
    print("=" * 70)
    
    entity_degree = Counter()
    for subj, rel, obj, time in train_triplets:
        entity_degree[subj] += 1
        entity_degree[obj] += 1
    
    print(f"\n{'Rank':<6} {'Entity Name':<40} {'Type':<12} {'Connections':>12}")
    print("-" * 70)
    
    for i, (entity_id, degree) in enumerate(entity_degree.most_common(10)):
        entity_name = entities[entity_id]['name']
        entity_type = entities[entity_id]['type']
        print(f"{i+1:<6} {entity_name:<40} {entity_type:<12} {degree:>12,}")
    
    print("\n" + "=" * 70)
    print("Visualization Complete!")
    print("=" * 70)


if __name__ == "__main__":
    visualize_dataset()

