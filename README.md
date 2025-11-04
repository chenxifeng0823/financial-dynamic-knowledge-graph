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

## Quick Start

Get started quickly with our [Quick Start Guide](QUICKSTART.md)!

### Prerequisites

- Python 3.8+
- PyTorch 1.8+ (for model training, optional for data exploration)
- DGL (Deep Graph Library) 0.8+ (for model training, optional for data exploration)
- CUDA 11.0+ (optional, for GPU acceleration)

### Installation

```bash
# Clone the repository
git clone https://github.com/chenxifeng-web/financial-dynamic-knowledge-graph.git
cd financial-dynamic-knowledge-graph

# Download the dataset
python src/data_processing/load_dataset.py

# Explore the dataset
python src/data_processing/explore_dataset.py

# Visualize the dataset
python src/data_processing/visualize_dataset.py
```

## Dataset

The project works with the **FinDKG dataset**, which contains financial entities and their temporal relationships:

- **13,645 entities** (companies, people, countries, financial instruments, etc.)
- **15 relation types** (Control, Impact, Operate_In, Invests_In, etc.)
- **144,062 temporal triplets** spanning 126 time steps
- Entity types include: PERSON, COMPANY, GPE (countries), CONCEPT, EVENT, SECTOR, etc.

Key entities include Donald Trump, United States, U.S. Federal Reserve, China, and major corporations.

See [DATASET_SUMMARY.md](data/DATASET_SUMMARY.md) for detailed statistics.

## Status

🚧 **Work in Progress** - This project is in early development.

## References

- Original FinDKG Repository: https://github.com/xiaohui-victor-li/FinDKG
- FinDKG Website: https://xiaohui-victor-li.github.io/FinDKG/

## License

This project is for research and educational purposes.

