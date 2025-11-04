"""
Data loader for FinDKG dataset.

This script downloads and prepares the Financial Dynamic Knowledge Graph dataset
from the original FinDKG repository.
"""

import os
import subprocess
import sys
from pathlib import Path


class FinDKGDataLoader:
    """Loader for the FinDKG dataset."""
    
    def __init__(self, data_dir="./data"):
        """
        Initialize the data loader.
        
        Args:
            data_dir: Directory to store the dataset
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.repo_url = "https://github.com/xiaohui-victor-li/FinDKG.git"
        self.repo_dir = self.data_dir / "FinDKG_repo"
        
    def download_dataset(self):
        """
        Download the FinDKG dataset from the original repository.
        """
        print("=" * 60)
        print("Downloading FinDKG Dataset")
        print("=" * 60)
        
        if self.repo_dir.exists():
            print(f"Repository already exists at {self.repo_dir}")
            print("Updating repository...")
            try:
                subprocess.run(
                    ["git", "pull"],
                    cwd=self.repo_dir,
                    check=True,
                    capture_output=True
                )
                print("Repository updated successfully!")
            except subprocess.CalledProcessError as e:
                print(f"Failed to update repository: {e}")
                return False
        else:
            print(f"Cloning repository to {self.repo_dir}...")
            try:
                subprocess.run(
                    ["git", "clone", self.repo_url, str(self.repo_dir)],
                    check=True,
                    capture_output=True
                )
                print("Repository cloned successfully!")
            except subprocess.CalledProcessError as e:
                print(f"Failed to clone repository: {e}")
                return False
        
        return True
    
    def check_dataset(self):
        """
        Check if the dataset is available and show its structure.
        """
        print("\n" + "=" * 60)
        print("Dataset Structure")
        print("=" * 60)
        
        dataset_dir = self.repo_dir / "FinDKG_dataset"
        
        if not dataset_dir.exists():
            print(f"Dataset directory not found at {dataset_dir}")
            return False
        
        print(f"Dataset location: {dataset_dir}")
        print("\nContents:")
        
        # List all files in the dataset directory
        try:
            for item in sorted(dataset_dir.rglob("*")):
                if item.is_file():
                    rel_path = item.relative_to(dataset_dir)
                    size = item.stat().st_size
                    size_str = self._format_size(size)
                    print(f"  - {rel_path} ({size_str})")
        except Exception as e:
            print(f"Error reading dataset directory: {e}")
            return False
        
        return True
    
    def _format_size(self, size):
        """Format file size in human-readable format."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024.0:
                return f"{size:.2f} {unit}"
            size /= 1024.0
        return f"{size:.2f} TB"
    
    def load_dataset_info(self):
        """
        Load and display basic information about the dataset.
        """
        print("\n" + "=" * 60)
        print("Dataset Information")
        print("=" * 60)
        
        readme_path = self.repo_dir / "FinDKG_dataset" / "README.md"
        
        if readme_path.exists():
            print("\nDataset README:")
            print("-" * 60)
            with open(readme_path, 'r') as f:
                print(f.read())
        else:
            print("No README found in dataset directory")
            print("\nDataset is stored at:", self.repo_dir / "FinDKG_dataset")
        
        return True


def main():
    """Main function to download and check the dataset."""
    print("FinDKG Dataset Loader")
    print("=" * 60)
    
    loader = FinDKGDataLoader(data_dir="./data")
    
    # Download the dataset
    if not loader.download_dataset():
        print("\nFailed to download dataset. Please check your internet connection.")
        sys.exit(1)
    
    # Check dataset structure
    if not loader.check_dataset():
        print("\nFailed to verify dataset structure.")
        sys.exit(1)
    
    # Load dataset information
    loader.load_dataset_info()
    
    print("\n" + "=" * 60)
    print("Dataset download and verification complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Explore the dataset files in ./data/FinDKG_repo/FinDKG_dataset/")
    print("  2. Review the dataset documentation")
    print("  3. Start building data processing pipelines")


if __name__ == "__main__":
    main()

