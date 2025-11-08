"""
PyTorch Dataset for Temporal Knowledge Graph data.

This module provides Dataset classes for loading and batching temporal knowledge graph
triplets for model training.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional


class TemporalKGDataset(Dataset):
    """
    PyTorch Dataset for Temporal Knowledge Graph triplets.
    
    Each sample is a temporal triplet: (subject, relation, object, timestamp)
    """
    
    def __init__(
        self,
        data_path: str,
        split: str = 'train',
        num_entities: Optional[int] = None,
        num_relations: Optional[int] = None,
        num_negatives: int = 1
    ):
        """
        Initialize the dataset.
        
        Args:
            data_path: Path to the dataset directory
            split: One of 'train', 'valid', or 'test'
            num_entities: Total number of entities (for negative sampling)
            num_relations: Total number of relations
            num_negatives: Number of negative samples per positive triplet
        """
        self.data_path = Path(data_path)
        self.split = split
        self.num_negatives = num_negatives
        
        # Load triplets
        self.triplets = self._load_triplets(split)
        
        # Load statistics if not provided
        if num_entities is None or num_relations is None:
            num_entities, num_relations, _ = self._load_stats()
        
        self.num_entities = num_entities
        self.num_relations = num_relations
        
        print(f"Loaded {len(self.triplets)} {split} triplets")
        print(f"Entities: {self.num_entities}, Relations: {self.num_relations}")
    
    def _load_triplets(self, split: str) -> np.ndarray:
        """Load triplets from file."""
        triplet_file = self.data_path / f"{split}.txt"
        
        triplets = []
        with open(triplet_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 4:
                    subj, rel, obj, time = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
                    triplets.append([subj, rel, obj, time])
        
        return np.array(triplets, dtype=np.int64)
    
    def _load_stats(self) -> Tuple[int, int, int]:
        """Load dataset statistics."""
        stat_file = self.data_path / "stat.txt"
        
        with open(stat_file, 'r') as f:
            line = f.readline().strip().split('\t')
            num_entities = int(line[0])
            num_relations = int(line[1])
            num_timestamps = int(line[2])
        
        return num_entities, num_relations, num_timestamps
    
    def __len__(self) -> int:
        """Return the number of triplets."""
        return len(self.triplets)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single triplet with negative samples.
        
        Args:
            idx: Index of the triplet
            
        Returns:
            Dictionary containing:
                - 'positive': (subject, relation, object, time) positive triplet
                - 'negatives': List of negative triplets (corrupted entities)
        """
        # Get positive triplet
        pos_triplet = self.triplets[idx]
        subj, rel, obj, time = pos_triplet
        
        # Generate negative samples (corrupt object entities)
        # TODO: Implement more sophisticated negative sampling
        neg_objs = self._sample_negative_entities(subj, rel, obj, self.num_negatives)
        
        # Create negative triplets (same subject, relation, time; different object)
        neg_triplets = np.array([
            [subj, rel, neg_obj, time] for neg_obj in neg_objs
        ], dtype=np.int64)
        
        return {
            'positive': torch.from_numpy(pos_triplet),
            'negatives': torch.from_numpy(neg_triplets)
        }
    
    def _sample_negative_entities(
        self,
        subject: int,
        relation: int,
        true_object: int,
        num_samples: int
    ) -> List[int]:
        """
        Sample negative entities (simple random sampling).
        
        TODO: Implement filtered negative sampling that avoids sampling
        other valid triplets as negatives.
        
        Args:
            subject: Subject entity
            relation: Relation type
            true_object: True object entity (to avoid sampling it)
            num_samples: Number of negative samples
            
        Returns:
            List of negative entity IDs
        """
        negatives = []
        while len(negatives) < num_samples:
            neg_entity = np.random.randint(0, self.num_entities)
            # Avoid sampling the true object
            if neg_entity != true_object:
                negatives.append(neg_entity)
        
        return negatives


class TemporalKGBatchSampler:
    """
    Custom batch sampler that groups triplets by timestamp.
    
    This is useful for temporal models that process events in chronological order.
    """
    
    def __init__(self, dataset: TemporalKGDataset, batch_size: int, shuffle: bool = True):
        """
        Initialize the batch sampler.
        
        Args:
            dataset: TemporalKGDataset instance
            batch_size: Batch size
            shuffle: Whether to shuffle within each time step
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # Group triplets by timestamp
        self.time_to_indices = self._group_by_time()
    
    def _group_by_time(self) -> Dict[int, List[int]]:
        """Group triplet indices by their timestamps."""
        time_to_indices = {}
        
        for idx, triplet in enumerate(self.dataset.triplets):
            time = triplet[3]  # Timestamp is the 4th element
            if time not in time_to_indices:
                time_to_indices[time] = []
            time_to_indices[time].append(idx)
        
        return time_to_indices
    
    def __iter__(self):
        """Generate batches grouped by time."""
        timestamps = sorted(self.time_to_indices.keys())
        
        for time in timestamps:
            indices = self.time_to_indices[time]
            
            if self.shuffle:
                np.random.shuffle(indices)
            
            # Yield batches for this timestamp
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i + self.batch_size]
                yield batch
    
    def __len__(self):
        """Return the number of batches."""
        total_batches = 0
        for indices in self.time_to_indices.values():
            total_batches += (len(indices) + self.batch_size - 1) // self.batch_size
        return total_batches


def create_dataloader(
    data_path: str,
    split: str,
    batch_size: int = 512,
    num_workers: int = 0,
    shuffle: bool = True,
    temporal_batching: bool = False,
    num_negatives: int = 5
):
    """
    Create a DataLoader for temporal knowledge graph data.
    
    Args:
        data_path: Path to dataset directory
        split: One of 'train', 'valid', or 'test'
        batch_size: Batch size
        num_workers: Number of worker processes for data loading
        shuffle: Whether to shuffle data
        temporal_batching: Whether to group batches by timestamp
        num_negatives: Number of negative samples per positive triplet
        
    Returns:
        torch.utils.data.DataLoader instance
    """
    dataset = TemporalKGDataset(data_path, split=split, num_negatives=num_negatives)
    
    if temporal_batching:
        batch_sampler = TemporalKGBatchSampler(dataset, batch_size, shuffle=shuffle)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers
        )
    else:
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers
        )
    
    return dataloader


# Example usage
if __name__ == "__main__":
    # Test the dataset
    data_path = "./data/FinDKG_repo/FinDKG_dataset/FinDKG"
    
    print("=" * 70)
    print("Testing TemporalKGDataset")
    print("=" * 70)
    
    # Create dataset
    train_dataset = TemporalKGDataset(data_path, split='train', num_negatives=5)
    
    # Test single sample
    print("\nSample from dataset:")
    sample = train_dataset[0]
    print(f"Positive triplet: {sample['positive']}")
    print(f"Negative triplets shape: {sample['negatives'].shape}")
    print(f"First negative: {sample['negatives'][0]}")
    
    # Create dataloader
    print("\n" + "=" * 70)
    print("Testing DataLoader")
    print("=" * 70)
    
    train_loader = create_dataloader(
        data_path,
        split='train',
        batch_size=32,
        shuffle=True,
        temporal_batching=False
    )
    
    # Test iteration
    print(f"\nTotal batches: {len(train_loader)}")
    
    for i, batch in enumerate(train_loader):
        print(f"\nBatch {i}:")
        print(f"  Positive triplets shape: {batch['positive'].shape}")
        print(f"  Negative triplets shape: {batch['negatives'].shape}")
        
        if i >= 2:  # Only show first 3 batches
            break
    
    # Test temporal batching
    print("\n" + "=" * 70)
    print("Testing Temporal Batching")
    print("=" * 70)
    
    temporal_loader = create_dataloader(
        data_path,
        split='train',
        batch_size=32,
        shuffle=False,
        temporal_batching=True
    )
    
    print(f"\nTotal batches with temporal grouping: {len(temporal_loader)}")
    
    for i, batch in enumerate(temporal_loader):
        times = batch['positive'][:, 3]  # Extract timestamps
        print(f"Batch {i}: Time step = {times[0].item()}, Size = {len(times)}")
        
        if i >= 5:  # Show first 6 batches
            break
    
    print("\n" + "=" * 70)
    print("Dataset tests completed successfully!")
    print("=" * 70)

