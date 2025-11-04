# Quick Start Guide

This guide will help you get started with the Financial Dynamic Knowledge Graph project.

## Prerequisites

Make sure you have Python 3.8+ installed:

```bash
python --version
```

## Installation

1. **Clone the repository**

```bash
git clone https://github.com/chenxifeng-web/financial-dynamic-knowledge-graph.git
cd financial-dynamic-knowledge-graph
```

2. **Install dependencies** (optional, for basic data exploration)

```bash
pip install -r requirements.txt
```

For the basic data loading and exploration, you only need Python's standard library.

## Download and Explore the Dataset

### Step 1: Download the FinDKG Dataset

Run the data loader script to automatically download the dataset from the original repository:

```bash
python src/data_processing/load_dataset.py
```

This will:
- Clone the FinDKG repository
- Download the dataset (FinDKG_dataset folder)
- Display dataset structure and files
- Show the README from the dataset

### Step 2: Explore the Dataset

Run the exploration script to see detailed statistics:

```bash
python src/data_processing/explore_dataset.py
```

This will display:
- Total entities, relations, and triplets
- Entity type distribution
- Relation type distribution
- Temporal coverage
- Sample entities and triplets

### Step 3: Visualize the Dataset

Run the visualization script for graphical representations:

```bash
python src/data_processing/visualize_dataset.py
```

This will show:
- Entity type distribution (bar charts)
- Relation distribution (bar charts)
- Temporal distribution timeline
- Top 10 most connected entities

## Dataset Overview

The FinDKG dataset contains:
- **13,645 entities** (companies, people, countries, events, etc.)
- **15 relation types** (Control, Impact, Operate_In, etc.)
- **144,062 triplets** (subject-relation-object-time quadruples)
- **126 time steps** (weekly temporal resolution)

### Key Entities

- Donald Trump (8,727 connections)
- United States (7,528 connections)
- U.S. Federal Reserve (6,746 connections)
- China (3,882 connections)

### Key Relations

- Relate_To (27.55%)
- Control (22.06%)
- Operate_In (12.96%)
- Impact (10.97%)

## Next Steps

1. **Build Data Pipelines**: Create preprocessing pipelines for the temporal knowledge graph
2. **Implement Models**: Develop temporal link prediction models
3. **Run Experiments**: Train and evaluate models on the dataset
4. **Extend the Dataset**: Incorporate additional financial data sources
5. **Build Applications**: Create financial analysis tools using the knowledge graph

## Project Structure

```
financial-dynamic-knowledge-graph/
├── data/                           # Dataset storage
│   ├── DATASET_SUMMARY.md         # Detailed dataset statistics
│   └── FinDKG_repo/               # Downloaded original dataset
├── src/                           # Source code
│   ├── data_processing/           # Data loading and exploration
│   │   ├── load_dataset.py       # Dataset downloader
│   │   ├── explore_dataset.py    # Dataset explorer
│   │   └── visualize_dataset.py  # Dataset visualizer
│   ├── models/                    # Model implementations (coming soon)
│   └── utils/                     # Utility functions
├── notebooks/                     # Jupyter notebooks
├── tests/                         # Unit tests
├── requirements.txt               # Dependencies
└── README.md                      # Project documentation
```

## Troubleshooting

### Issue: Git clone fails

If the dataset download fails, you can manually clone the original repository:

```bash
cd data
git clone https://github.com/xiaohui-victor-li/FinDKG.git FinDKG_repo
cd ..
```

### Issue: Import errors

Make sure you run the scripts from the project root directory:

```bash
cd /path/to/financial-dynamic-knowledge-graph
python src/data_processing/explore_dataset.py
```

## Resources

- Original FinDKG Repository: https://github.com/xiaohui-victor-li/FinDKG
- FinDKG Website: https://xiaohui-victor-li.github.io/FinDKG/
- Project Repository: https://github.com/chenxifeng-web/financial-dynamic-knowledge-graph

## Contributing

This project is in early development. Contributions and suggestions are welcome!

