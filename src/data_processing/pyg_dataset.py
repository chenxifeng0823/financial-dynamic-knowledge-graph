"""
PyG Dataset and DataLoader for Temporal Knowledge Graphs
Sequential temporal batching for KGTransformer
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data
from collections import namedtuple


# Multi-aspect embedding structure
MultiAspectEmbedding = namedtuple('MultiAspectEmbedding', ['structural', 'temporal'], defaults=[None, None])


class TemporalKGDatasetPyG(Dataset):
    """
    Temporal Knowledge Graph Dataset for PyG
    
    Processes temporal KG data sequentially, maintaining cumulative graph state.
    
    Args:
        triplets (numpy.ndarray): Array of shape (N, 4) with columns [head, rel, tail, time]
        num_entities (int): Number of entities
        num_relations (int): Number of relations
        entity_types (numpy.ndarray, optional): Node types for each entity
        timestamps_array (numpy.ndarray, optional): Array of unique timestamps to process
    """
    
    def __init__(self, triplets, num_entities, num_relations, entity_types=None, timestamps_array=None):
        self.triplets = triplets
        self.num_entities = num_entities
        self.num_relations = num_relations
        
        # Node types (if available)
        if entity_types is not None:
            self.entity_types = torch.from_numpy(entity_types).long()
        else:
            self.entity_types = torch.zeros(num_entities, dtype=torch.long)
        
        # Get unique timestamps
        if timestamps_array is not None:
            self.timestamps = timestamps_array
        else:
            self.timestamps = np.sort(np.unique(triplets[:, 3]))
        
        # Group triplets by timestamp
        self.timestamp_to_triplets = {}
        for t in self.timestamps:
            mask = triplets[:, 3] == t
            self.timestamp_to_triplets[t] = triplets[mask]
        
        print(f"TemporalKGDatasetPyG: {len(self.timestamps)} timestamps, "
              f"{len(triplets)} total triplets")
    
    def __len__(self):
        return len(self.timestamps)
    
    def __getitem__(self, idx):
        """
        Get batch for a specific timestamp.
        
        Returns:
            PyG Data object with:
                - edge_index: Edge indices [2, E]
                - edge_type: Relation types [E]
                - timestamps: Event times [E]
                - node_id: Global node IDs [N]
                - node_type: Node types [N]
                - num_nodes: Number of nodes in this graph
                - timestamp_idx: Index of this timestamp
        """
        timestamp = self.timestamps[idx]
        batch_triplets = self.timestamp_to_triplets[timestamp]
        
        # Extract heads, rels, tails, times
        heads = batch_triplets[:, 0].astype(np.int64)
        rels = batch_triplets[:, 1].astype(np.int64)
        tails = batch_triplets[:, 2].astype(np.int64)
        times = batch_triplets[:, 3]
        
        # Get unique nodes in this batch
        unique_nodes = np.unique(np.concatenate([heads, tails]))
        node_to_idx = {node: idx for idx, node in enumerate(unique_nodes)}
        
        # Map global node IDs to local IDs
        local_heads = np.array([node_to_idx[h] for h in heads])
        local_tails = np.array([node_to_idx[t] for t in tails])
        
        # Create PyG Data object
        edge_index = torch.from_numpy(np.stack([local_heads, local_tails], axis=0)).long()
        edge_type = torch.from_numpy(rels).long()
        timestamps_tensor = torch.from_numpy(times).float()
        
        node_id = torch.from_numpy(unique_nodes).long()
        node_type = self.entity_types[node_id]
        
        data = Data(
            edge_index=edge_index,
            edge_type=edge_type,
            timestamps=timestamps_tensor,
            node_id=node_id,
            node_type=node_type,
            num_nodes=len(unique_nodes),
            timestamp_idx=idx,
            timestamp=timestamp
        )
        
        return data
    
    @staticmethod
    def from_txt_files(train_file, val_file, test_file, entity_types_file=None):
        """
        Load dataset from text files (FinDKG format).
        
        Args:
            train_file (str): Path to train.txt
            val_file (str): Path to valid.txt
            test_file (str): Path to test.txt
            entity_types_file (str, optional): Path to entity2id.txt with node types
        
        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset, num_entities, num_relations)
        """
        # Load triplets
        train_data = np.loadtxt(train_file, dtype=int)[:, :4]  # head, rel, tail, time
        val_data = np.loadtxt(val_file, dtype=int)[:, :4]
        test_data = np.loadtxt(test_file, dtype=int)[:, :4]
        
        # Get statistics
        all_data = np.concatenate([train_data, val_data, test_data], axis=0)
        num_entities = int(all_data[:, [0, 2]].max()) + 1
        num_relations = int(all_data[:, 1].max()) + 1
        
        # Load entity types if available
        entity_types = None
        if entity_types_file is not None:
            # Format: entity_name \t entity_id \t type_name \t type_id
            # We need to parse it carefully due to string columns
            import pandas as pd
            entity_df = pd.read_csv(entity_types_file, sep='\t', header=None, 
                                   names=['entity_name', 'entity_id', 'type_name', 'type_id'])
            
            # Create entity_id -> type_id mapping
            entity_types = np.zeros(num_entities, dtype=int)
            entity_types[entity_df['entity_id'].values] = entity_df['type_id'].values
        
        # Get unique timestamps for each split
        train_times = np.sort(np.unique(train_data[:, 3]))
        val_times = np.sort(np.unique(val_data[:, 3]))
        test_times = np.sort(np.unique(test_data[:, 3]))
        
        # Create datasets
        train_dataset = TemporalKGDatasetPyG(
            train_data, num_entities, num_relations, entity_types, train_times
        )
        val_dataset = TemporalKGDatasetPyG(
            val_data, num_entities, num_relations, entity_types, val_times
        )
        test_dataset = TemporalKGDatasetPyG(
            test_data, num_entities, num_relations, entity_types, test_times
        )
        
        return train_dataset, val_dataset, test_dataset, num_entities, num_relations


class CumulativeGraphBuilder:
    """
    Builds cumulative graphs for temporal KG processing.
    
    Maintains the cumulative state of the graph up to the current timestamp,
    which is needed for link prediction tasks.
    """
    
    def __init__(self, num_entities, num_relations, entity_types):
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.entity_types = entity_types
        
        # Initialize cumulative graph (empty)
        self.cumul_edge_index = torch.empty((2, 0), dtype=torch.long)
        self.cumul_edge_type = torch.empty(0, dtype=torch.long)
        self.cumul_timestamps = torch.empty(0, dtype=torch.float)
        
        # Track all nodes that have appeared
        self.cumul_node_set = set()
    
    def add_batch(self, batch_data):
        """
        Add batch data to cumulative graph.
        
        Args:
            batch_data: PyG Data object for current timestamp
        
        Returns:
            Updated cumulative PyG Data object
        """
        # Convert local node IDs back to global IDs (keep on CPU)
        global_edge_index = torch.stack([
            batch_data.node_id[batch_data.edge_index[0].cpu()],
            batch_data.node_id[batch_data.edge_index[1].cpu()]
        ], dim=0).cpu()
        
        # Append to cumulative graph (all on CPU)
        self.cumul_edge_index = torch.cat([self.cumul_edge_index, global_edge_index], dim=1)
        self.cumul_edge_type = torch.cat([self.cumul_edge_type, batch_data.edge_type.cpu()], dim=0)
        self.cumul_timestamps = torch.cat([self.cumul_timestamps, batch_data.timestamps.cpu()], dim=0)
        
        # Update node set
        self.cumul_node_set.update(batch_data.node_id.cpu().tolist())
        
        # Create cumulative graph Data object
        cumul_nodes = torch.tensor(sorted(self.cumul_node_set), dtype=torch.long)
        cumul_node_types = self.entity_types[cumul_nodes]
        
        # Create mapping from global to local IDs
        global_to_local = {g.item(): l for l, g in enumerate(cumul_nodes)}
        
        # Map edge indices to local IDs
        local_edge_index = torch.stack([
            torch.tensor([global_to_local[g.item()] for g in self.cumul_edge_index[0]], dtype=torch.long),
            torch.tensor([global_to_local[g.item()] for g in self.cumul_edge_index[1]], dtype=torch.long)
        ], dim=0)
        
        cumul_data = Data(
            edge_index=local_edge_index,
            edge_type=self.cumul_edge_type,
            timestamps=self.cumul_timestamps,
            node_id=cumul_nodes,
            node_type=cumul_node_types,
            num_nodes=len(cumul_nodes)
        )
        
        return cumul_data
    
    def reset(self):
        """Reset the cumulative graph."""
        self.cumul_edge_index = torch.empty((2, 0), dtype=torch.long)
        self.cumul_edge_type = torch.empty(0, dtype=torch.long)
        self.cumul_timestamps = torch.empty(0, dtype=torch.float)
        self.cumul_node_set = set()


def create_temporal_dataloaders(
    train_dataset,
    val_dataset,
    test_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0
):
    """
    Create DataLoaders for temporal KG datasets.
    
    Note: For temporal KG, batch_size typically = 1 (one timestamp at a time),
    and shuffle = False (to maintain temporal ordering).
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset
        batch_size: Batch size (default: 1 for temporal ordering)
        shuffle: Whether to shuffle (default: False for temporal ordering)
        num_workers: Number of workers for data loading
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=lambda x: x[0] if batch_size == 1 else x
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # Always False for validation
        num_workers=num_workers,
        collate_fn=lambda x: x[0] if batch_size == 1 else x
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # Always False for testing
        num_workers=num_workers,
        collate_fn=lambda x: x[0] if batch_size == 1 else x
    )
    
    return train_loader, val_loader, test_loader


# Example usage
if __name__ == '__main__':
    # Load FinDKG dataset
    data_root = "data/FinDKG"
    
    train_dataset, val_dataset, test_dataset, num_entities, num_relations = \
        TemporalKGDatasetPyG.from_txt_files(
            f"{data_root}/train.txt",
            f"{data_root}/valid.txt",
            f"{data_root}/test.txt",
            f"{data_root}/entity2id.txt"
        )
    
    print(f"Num entities: {num_entities}")
    print(f"Num relations: {num_relations}")
    print(f"Train timestamps: {len(train_dataset)}")
    print(f"Val timestamps: {len(val_dataset)}")
    print(f"Test timestamps: {len(test_dataset)}")
    
    # Test data loading
    train_loader, val_loader, test_loader = create_temporal_dataloaders(
        train_dataset, val_dataset, test_dataset
    )
    
    # Get first batch
    batch = next(iter(train_loader))
    print(f"\nFirst batch:")
    print(f"  Nodes: {batch.num_nodes}")
    print(f"  Edges: {batch.edge_index.size(1)}")
    print(f"  Timestamp: {batch.timestamp}")

