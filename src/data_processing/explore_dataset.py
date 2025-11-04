"""
Dataset exploration script for FinDKG.

This script loads and analyzes the Financial Dynamic Knowledge Graph dataset,
providing statistics and insights about the temporal knowledge graph structure.
"""

import os
from pathlib import Path
from collections import Counter, defaultdict


class FinDKGExplorer:
    """Explorer for analyzing the FinDKG dataset."""
    
    def __init__(self, dataset_path="./data/FinDKG_repo/FinDKG_dataset/FinDKG"):
        """
        Initialize the dataset explorer.
        
        Args:
            dataset_path: Path to the FinDKG dataset directory
        """
        self.dataset_path = Path(dataset_path)
        self.entities = {}
        self.relations = {}
        self.entity_types = Counter()
        
    def load_entities(self):
        """Load entity mappings from entity2id.txt."""
        entity_file = self.dataset_path / "entity2id.txt"
        
        print("Loading entities...")
        with open(entity_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    entity_name, entity_id, entity_type = parts[0], int(parts[1]), parts[2]
                    self.entities[entity_id] = {
                        'name': entity_name,
                        'type': entity_type
                    }
                    self.entity_types[entity_type] += 1
        
        print(f"Loaded {len(self.entities)} entities")
        return self.entities
    
    def load_relations(self):
        """Load relation mappings from relation2id.txt."""
        relation_file = self.dataset_path / "relation2id.txt"
        
        print("Loading relations...")
        with open(relation_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    relation_name, relation_id = parts[0], int(parts[1])
                    self.relations[relation_id] = relation_name
        
        print(f"Loaded {len(self.relations)} relations")
        return self.relations
    
    def load_triplets(self, split='train'):
        """
        Load triplets from a specific split.
        
        Args:
            split: One of 'train', 'valid', or 'test'
        
        Returns:
            List of (subject, relation, object, time) tuples
        """
        triplet_file = self.dataset_path / f"{split}.txt"
        
        print(f"Loading {split} triplets...")
        triplets = []
        
        with open(triplet_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 4:
                    subject, relation, obj, time = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
                    triplets.append((subject, relation, obj, time))
        
        print(f"Loaded {len(triplets)} {split} triplets")
        return triplets
    
    def analyze_dataset(self):
        """Provide comprehensive analysis of the dataset."""
        print("\n" + "=" * 70)
        print("FinDKG Dataset Analysis")
        print("=" * 70)
        
        # Load all data
        self.load_entities()
        self.load_relations()
        
        train_triplets = self.load_triplets('train')
        valid_triplets = self.load_triplets('valid')
        test_triplets = self.load_triplets('test')
        
        # Basic statistics
        print("\n" + "-" * 70)
        print("Basic Statistics")
        print("-" * 70)
        print(f"Total Entities: {len(self.entities):,}")
        print(f"Total Relations: {len(self.relations)}")
        print(f"Training Triplets: {len(train_triplets):,}")
        print(f"Validation Triplets: {len(valid_triplets):,}")
        print(f"Test Triplets: {len(test_triplets):,}")
        print(f"Total Triplets: {len(train_triplets) + len(valid_triplets) + len(test_triplets):,}")
        
        # Entity types
        print("\n" + "-" * 70)
        print("Entity Types Distribution")
        print("-" * 70)
        for entity_type, count in sorted(self.entity_types.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(self.entities)) * 100
            print(f"  {entity_type:20s}: {count:6,} ({percentage:5.2f}%)")
        
        # Relation types
        print("\n" + "-" * 70)
        print("Relation Types")
        print("-" * 70)
        for rel_id, rel_name in sorted(self.relations.items()):
            print(f"  {rel_id:2d}: {rel_name}")
        
        # Temporal analysis
        print("\n" + "-" * 70)
        print("Temporal Distribution")
        print("-" * 70)
        
        all_triplets = train_triplets + valid_triplets + test_triplets
        time_stamps = [t[3] for t in all_triplets]
        
        print(f"  Time Range: {min(time_stamps)} - {max(time_stamps)}")
        print(f"  Total Time Steps: {max(time_stamps) - min(time_stamps) + 1}")
        
        # Triplets per time step
        time_counter = Counter(time_stamps)
        print(f"  Average Triplets per Time Step: {len(all_triplets) / len(time_counter):.2f}")
        print(f"  Max Triplets in a Time Step: {max(time_counter.values())}")
        print(f"  Min Triplets in a Time Step: {min(time_counter.values())}")
        
        # Relation distribution
        print("\n" + "-" * 70)
        print("Relation Distribution in Training Set")
        print("-" * 70)
        
        relation_counter = Counter([t[1] for t in train_triplets])
        for rel_id, count in sorted(relation_counter.items(), key=lambda x: x[1], reverse=True):
            rel_name = self.relations[rel_id]
            percentage = (count / len(train_triplets)) * 100
            print(f"  {rel_name:25s}: {count:7,} ({percentage:5.2f}%)")
        
        # Sample entities
        print("\n" + "-" * 70)
        print("Sample Entities (First 10)")
        print("-" * 70)
        for i in range(min(10, len(self.entities))):
            entity = self.entities[i]
            print(f"  ID {i:4d}: {entity['name']:40s} [{entity['type']}]")
        
        # Sample triplets
        print("\n" + "-" * 70)
        print("Sample Triplets (First 5 from Training Set)")
        print("-" * 70)
        for i, (subj, rel, obj, time) in enumerate(train_triplets[:5]):
            subj_name = self.entities[subj]['name']
            rel_name = self.relations[rel]
            obj_name = self.entities[obj]['name']
            print(f"  {i+1}. [{subj_name}] --{rel_name}--> [{obj_name}] at time {time}")
        
        print("\n" + "=" * 70)
        print("Analysis Complete!")
        print("=" * 70)


def main():
    """Main function to run dataset exploration."""
    explorer = FinDKGExplorer()
    explorer.analyze_dataset()


if __name__ == "__main__":
    main()

