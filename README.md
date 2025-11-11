# Financial Dynamic Knowledge Graph

**🎯 KGTransformer Implementation with PyTorch Geometric**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1+-red.svg)](https://pytorch.org/)
[![PyG 2.7+](https://img.shields.io/badge/PyG-2.7+-green.svg)](https://pytorch-geometric.readthedocs.io/)

This project implements **KGTransformer** for temporal knowledge graph learning on financial data, with a focus on replacing RNN-based temporal modeling with Transformers.

## 🎯 Project Status

✅ **Phase 1: Complete** - KGTransformer baseline implemented and tested (Nov 2025)  
🚧 **Phase 2: In Progress** - Training and evaluation on FinDKG dataset  
⏳ **Phase 3: Planned** - Replace RNN with Transformer for temporal modeling

## 📊 What's Implemented

### ✅ Complete KGTransformer Architecture (PyG)

- **Graph Transformer** with multi-head attention and relation-specific transformations
- **RGCN** layers for relational graph convolutions
- **RNN-based temporal encoders** (structural + temporal with time decay)
- **Multi-aspect embeddings** (static + dynamic)
- **Link prediction** for temporal knowledge graphs

### ✅ Training Infrastructure

- Sequential temporal batching for time-evolving graphs
- Training script with original FinDKG hyperparameters
- Checkpoint saving and early stopping
- Comprehensive test suite

**Model Size**: 31.9M parameters  
**First Epoch Results**: Loss 9.52 → 7.39 (learning successfully!)

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/chenxifeng0823/financial-dynamic-knowledge-graph.git
cd financial-dynamic-knowledge-graph

# Create conda environment
conda create -n findkg python=3.10 -y
conda activate findkg

# Install PyTorch Geometric
pip install torch-geometric torch-scatter torch-sparse \
    -f https://data.pyg.org/whl/torch-2.1.0+cu118.html

# Install other dependencies
pip install pandas numpy scipy tqdm pyyaml
```

### 2. Download Dataset

```bash
# Download FinDKG dataset
python src/data_processing/load_dataset.py

# Explore dataset
python src/data_processing/explore_dataset.py
```

**Expected structure**:
```
data/FinDKG/
├── train.txt          # 119,549 temporal triplets (100 timestamps)
├── valid.txt          # 11,444 triplets (13 timestamps)
├── test.txt           # 13,069 triplets (13 timestamps)
├── entity2id.txt      # 13,645 entities with types
├── relation2id.txt    # 15 relation types
└── stat.txt           # Dataset statistics
```

### 3. Test Implementation

```bash
# Run all tests
python test_kgt_pyg.py
```

Expected output:
```
✓ TypedLinear test passed!
✓ RGCN test passed!
✓ GraphTransformer test passed!
✓ Data loading test passed!
✓ Model instantiation test passed!
✓ Forward pass test passed!
✓ Backward pass test passed!
```

### 4. Train Model

```bash
# Quick test (1 epoch)
python train_kgt_pyg.py --device cuda --epochs 1 --save_model

# Full training (100 epochs)
python train_kgt_pyg.py --device cuda --epochs 100 --save_model

# Custom hyperparameters
python train_kgt_pyg.py \
    --device cuda \
    --epochs 50 \
    --lr 0.0005 \
    --seed 42 \
    --save_model \
    --save_dir checkpoints
```

## 📁 Project Structure

```
financial-dynamic-knowledge-graph/
├── data/
│   └── FinDKG/                    # Dataset
│
├── reference/
│   └── FinDKG_original/           # Original DGL implementation
│
├── src/
│   ├── models/
│   │   ├── pyg_modules/           # Core PyG components
│   │   │   ├── typed_linear.py   # Type-specific linear layers
│   │   │   ├── rgcn.py            # RGCN implementation
│   │   │   ├── graph_transformer.py  # Graph Transformer
│   │   │   └── embedding_updater.py  # RNN temporal encoders
│   │   └── pyg_kgtransformer.py   # Complete model
│   │
│   └── data_processing/
│       └── pyg_dataset.py         # PyG dataset & dataloaders
│
├── docs/
│   ├── ARCHITECTURE.md            # Technical deep-dive
│   └── DATASET.md                 # Dataset information
│
├── train_kgt_pyg.py               # Training script
├── test_kgt_pyg.py                # Test suite
└── README.md                      # This file
```

## 🔬 Research Goal

**Replace RNN with Transformer for temporal modeling:**

```
Current Baseline:  Graph Transformer (spatial) + RNN (temporal)
Research Target:   Graph Transformer (spatial) + Transformer (temporal)
```

**Why?**
- Better capture long-range temporal dependencies
- Enable parallel processing of temporal sequences
- Provide interpretable attention over time

**Implementation Plan**:
1. Create `GraphTemporalTransformerConv` to replace `GraphTemporalRNNConv`
2. Design temporal positional encoding
3. Use `nn.TransformerEncoder` for sequence modeling
4. Compare performance: RNN vs Transformer

## 📊 Dataset: FinDKG

Financial Dynamic Knowledge Graph for temporal link prediction.

**Statistics**:
- **Entities**: 13,645 (companies, people, countries, financial instruments)
- **Relations**: 15 types (Control, Impact, Operate_In, Invests_In, etc.)
- **Triplets**: 144,062 temporal facts
- **Timestamps**: 126 time steps
- **Entity Types**: 12 categories (PERSON, COMPANY, GPE, ORG/REG, etc.)

**Key Entities**: Donald Trump, United States, China, U.S. Federal Reserve, major corporations

See [`docs/DATASET.md`](docs/DATASET.md) for detailed statistics.

## 🏗️ Model Architecture

```
KGTransformerPyG (31.9M parameters)
├── Static Embeddings (learnable)
│   ├── structural: [13645, 200]
│   └── temporal: [13645, 200]
│
├── Dynamic Embeddings (RNN states)
│   ├── structural: [13645, 1, 200]
│   └── temporal: [13645, 1, 200, 2]  # bidirectional
│
├── EmbeddingUpdater
│   ├── GraphStructuralRNNConv (KGT + RNN)
│   ├── GraphTemporalRNNConv (KGT + RNN + Time Decay)
│   └── RelationRNN
│
├── Combiner (static + dynamic)
│
└── EdgeModel (link prediction)
```

**Key Features**:
- Multi-head attention with relation-specific transformations
- Time decay modeling via inter-event times
- Bidirectional RNN for recipient/sender roles
- Type-specific linear layers for heterogeneous graphs

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for technical details.

## 📈 Training Results

**First Epoch** (FinDKG, 100 timestamps):
```
Initial Loss: 9.52
Final Loss: 7.39
Average Loss: 7.64
Validation Loss: 7.42
Test Loss: 8.44
```

Loss decreases steadily → Model is learning temporal patterns! ✅

## 🔧 Configuration

**Default Hyperparameters** (from original FinDKG paper):
```python
{
    'static_entity_embed_dim': 200,
    'structural_dynamic_entity_embed_dim': 200,
    'temporal_dynamic_entity_embed_dim': 200,
    'rel_embed_dim': 200,
    'num_gconv_layers': 2,
    'num_rnn_layers': 1,
    'num_attn_heads': 8,
    'dropout': 0.2,
    'lr': 0.001,
    'weight_decay': 0.00001,
    'epochs': 100,
}
```

## 📚 Documentation

- **[ARCHITECTURE.md](docs/ARCHITECTURE.md)** - Complete technical documentation
  - Model architecture details
  - Core components (TypedLinear, GraphTransformer, RGCN, RNN encoders)
  - Implementation details (data flow, memory management, device handling)
  - DGL vs PyG comparison
  - Training & evaluation guide

- **[DATASET.md](docs/DATASET.md)** - Dataset information
  - Statistics and structure
  - Entity and relation types
  - Temporal distribution
  - Data format

## 🧪 Testing

```bash
# Run all tests
python test_kgt_pyg.py

# Test specific components
python -c "from src.models.pyg_modules import TypedLinear; print('✓ Import successful')"

# Test data loading
python src/data_processing/pyg_dataset.py
```

## 🐛 Known Issues & Solutions

**Issue**: `RuntimeError: indices should be either on cpu or on the same device`  
**Solution**: All indexing of CPU tensors uses `.cpu()` on indices

**Issue**: `RuntimeError: Trying to backward through the graph a second time`  
**Solution**: Dynamic embeddings are detached after each batch to prevent BPTT across timestamps

**Issue**: `torch-scatter` import errors  
**Solution**: Install with correct PyTorch version: `pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.0+cu118.html`

## 🤝 Contributing

This is a research project. Feel free to:
- Open issues for bugs or questions
- Submit PRs for improvements
- Use the code for your own research (please cite)

## 📄 References

- **Original Paper**: FinDKG - Financial Dynamic Knowledge Graph
- **Original Code**: https://github.com/xiaohui-victor-li/FinDKG (DGL implementation)
- **PyTorch Geometric**: https://pytorch-geometric.readthedocs.io/
- **FinDKG Website**: https://xiaohui-victor-li.github.io/FinDKG/

## 📧 Contact

For questions or collaboration:
- GitHub Issues: https://github.com/chenxifeng0823/financial-dynamic-knowledge-graph/issues
- Email: [Your email if you want to add it]

## 📝 License

This project is for research and educational purposes.

---

**Last Updated**: 2025-11-11  
**Status**: ✅ Baseline Complete | 🚧 Training In Progress | ⏳ Research Extension Planned
