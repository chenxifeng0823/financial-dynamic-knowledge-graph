# Financial Dynamic Knowledge Graph

This project implements a Dynamic Knowledge Graph system for financial data analysis, inspired by the FinDKG paper and extending it with new capabilities.

## Project Overview

This repository focuses on building and analyzing temporal knowledge graphs (TKGs) in the financial domain. Knowledge graphs capture relationships between entities, and temporal knowledge graphs add a time dimension to track how these relationships evolve over time.

### Key Objectives

1. Implement temporal knowledge graph processing for financial data
2. Build upon existing FinDKG research with new features and improvements
3. Develop models for temporal link prediction and anomaly detection in financial networks
4. Provide tools for financial knowledge graph analysis and visualization

## Project Structure

```
financial-dynamic-knowledge-graph/
├── data/                    # Dataset storage
├── src/                     # Source code
│   ├── models/             # Model implementations
│   ├── data_processing/    # Data loading and preprocessing
│   └── utils/              # Utility functions
├── notebooks/              # Jupyter notebooks for exploration
├── tests/                  # Unit tests
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Background

This project is inspired by the FinDKG work (see [original repository](https://github.com/xiaohui-victor-li/FinDKG)), which introduced:
- **FinDKG Dataset**: A financial temporal knowledge graph dataset
- **KGTransformer**: A model for temporal knowledge graph learning

## Setup

### Prerequisites

- Python 3.8+
- PyTorch 1.8+
- DGL (Deep Graph Library) 0.8+
- CUDA 11.0+ (optional, for GPU acceleration)

### Installation

```bash
# Clone the repository
git clone https://github.com/chenxifeng-web/financial-dynamic-knowledge-graph.git
cd financial-dynamic-knowledge-graph

# Install dependencies
pip install -r requirements.txt
```

## Dataset

The project will work with the FinDKG dataset, which contains financial entities and their temporal relationships. The dataset will be downloaded and prepared in the `data/` directory.

## Status

🚧 **Work in Progress** - This project is in early development.

## References

- Original FinDKG Repository: https://github.com/xiaohui-victor-li/FinDKG
- FinDKG Website: https://xiaohui-victor-li.github.io/FinDKG/

## License

This project is for research and educational purposes.

