"""
Trainer class for knowledge graph embedding models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from pathlib import Path
from .losses import margin_ranking_loss, bce_loss


class KGTrainer:
    """Trainer for knowledge graph embedding models."""
    
    def __init__(
        self,
        model,
        train_loader,
        valid_loader=None,
        optimizer=None,
        loss_fn='margin',
        device='cuda',
        checkpoint_dir='./checkpoints'
    ):
        """
        Initialize trainer.
        
        Args:
            model: Knowledge graph model
            train_loader: Training data loader
            valid_loader: Validation data loader (optional)
            optimizer: Optimizer (if None, uses Adam)
            loss_fn: Loss function ('margin' or 'bce')
            device: Device to train on
            checkpoint_dir: Directory to save checkpoints
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Optimizer
        if optimizer is None:
            self.optimizer = optim.Adam(model.parameters(), lr=0.001)
        else:
            self.optimizer = optimizer
        
        # Loss function
        if loss_fn == 'margin':
            self.loss_fn = margin_ranking_loss
        elif loss_fn == 'bce':
            self.loss_fn = bce_loss
        else:
            raise ValueError(f"Unknown loss function: {loss_fn}")
        
        self.best_valid_loss = float('inf')
        self.history = {'train_loss': [], 'valid_loss': []}
    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        
        for batch in tqdm(self.train_loader, desc='Training'):
            # Move to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Forward pass
            outputs = self.model(batch)
            
            # Compute loss
            loss = self.loss_fn(outputs['pos_scores'], outputs['neg_scores'])
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Normalize embeddings if TransE
            if hasattr(self.model, 'normalize_embeddings'):
                self.model.normalize_embeddings()
            
            total_loss += loss.item()
        
        return total_loss / len(self.train_loader)
    
    def validate(self):
        """Validate the model."""
        if self.valid_loader is None:
            return None
        
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in self.valid_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(batch)
                loss = self.loss_fn(outputs['pos_scores'], outputs['neg_scores'])
                total_loss += loss.item()
        
        return total_loss / len(self.valid_loader)
    
    def train(self, num_epochs, save_every=10):
        """
        Train the model for multiple epochs.
        
        Args:
            num_epochs: Number of epochs to train
            save_every: Save checkpoint every N epochs
        """
        print(f"Starting training for {num_epochs} epochs...")
        print("=" * 70)
        
        for epoch in range(1, num_epochs + 1):
            # Train
            train_loss = self.train_epoch()
            self.history['train_loss'].append(train_loss)
            
            # Validate
            if self.valid_loader is not None:
                valid_loss = self.validate()
                self.history['valid_loss'].append(valid_loss)
                print(f"Epoch {epoch}/{num_epochs} | Train Loss: {train_loss:.4f} | Valid Loss: {valid_loss:.4f}")
                
                # Save best model
                if valid_loss < self.best_valid_loss:
                    self.best_valid_loss = valid_loss
                    self.save_checkpoint(f'best_model.pth')
            else:
                print(f"Epoch {epoch}/{num_epochs} | Train Loss: {train_loss:.4f}")
            
            # Save periodic checkpoint
            if epoch % save_every == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pth')
        
        print("=" * 70)
        print("Training completed!")
    
    def save_checkpoint(self, filename):
        """Save model checkpoint."""
        checkpoint_path = self.checkpoint_dir / filename
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history
        }, checkpoint_path)
    
    def load_checkpoint(self, filename):
        """Load model checkpoint."""
        checkpoint_path = self.checkpoint_dir / filename
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint.get('history', {'train_loss': [], 'valid_loss': []})

