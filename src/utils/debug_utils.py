import torch
import sys

def check_nan(tensor, name, stop_on_nan=True):
    """Check tensor for NaN/Inf and print stats"""
    if tensor is None:
        return
        
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        print(f"\n❌ NaN/Inf detected in {name}!")
        print(f"   Shape: {tensor.shape}")
        print(f"   Min: {tensor.min().item()}")
        print(f"   Max: {tensor.max().item()}")
        print(f"   Mean: {tensor.mean().item()}")
        print(f"   NaN count: {torch.isnan(tensor).sum().item()}")
        print(f"   Inf count: {torch.isinf(tensor).sum().item()}")
        
        if stop_on_nan:
            raise ValueError(f"NaN detected in {name}")
    # else:
    #     # Optional: print range for healthy tensors
    #     # print(f"   {name}: range [{tensor.min().item():.4f}, {tensor.max().item():.4f}]")
