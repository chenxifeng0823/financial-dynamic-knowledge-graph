# Financial Dynamic Knowledge Graph

**🎯 Temporal Knowledge Graph Learning: RNN vs Transformer-based Temporal Attention**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1+-red.svg)](https://pytorch.org/)
[![PyG 2.7+](https://img.shields.io/badge/PyG-2.7+-green.svg)](https://pytorch-geometric.readthedocs.io/)

This project implements and compares **two deep temporal models** for knowledge graph learning on financial data:
1. **Recurrent Model (RNN)**: KGTransformer + GRU for temporal modeling
2. **Temporal Attention Model**: KGTransformer + Transformer-style attention over history windows

## 🎯 Project Status

✅ **Phase 1: Complete** - KGTransformer baseline implemented with PyG  
✅ **Phase 2: Complete** - Architecture aligned with DGL, all critical bugs fixed  
✅ **Phase 3: Complete** - Full training and evaluation on FinDKG dataset  
✅ **Phase 4: Complete** - Temporal Attention model implemented and optimized

## 🏆 Key Findings

| Model | Val MRR | Val Hits@10 | Notes |
|-------|---------|-------------|-------|
| **Temporal Attention** | **Higher** | **Higher** | Best after 3-stage HPO |
| Temporal RNN | Baseline | Baseline | Stable, easier to train |

**Main Result**: After proper hyperparameter tuning, **temporal attention outperforms RNN** on validation metrics, supporting our hypothesis that attention over historical states captures richer temporal dependencies.

📄 **Full Report**: See [`report.md`](report.md) for detailed analysis and methodology.

## 📊 What's Implemented

### Two Temporal Models

**1. Recurrent Model (KGTransformer + RNN)**
- GRU-based temporal state updates
- Per-entity hidden states maintained across time
- Simpler, more stable training

**2. Temporal Attention Model (KGTransformer + Transformer)**
- Multi-head attention over sliding history window
- Temporal positional encodings for time gaps
- More expressive but requires careful tuning

### Shared Components
- **Relational GNN** (RGCN / KGTransformer) for structural embeddings
- **Two-stage training**: cumulative graph for context, batch-level prediction
- **Multi-aspect embeddings**: static + structural + temporal
- **Link prediction decoder** with cross-entropy loss

### Best Configuration (Temporal Attention)
```
embed_dim = 256
num_heads = 8
num_gconv_layers = 2
window_size = 10
lr = 0.0002
dropout = 0.1
weight_decay = 0.0001
grad_clip_norm = 2.0
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/chenxifeng0823/financial-dynamic-knowledge-graph.git
cd financial-dynamic-knowledge-graph

# Create conda environment
conda create -n kg_env python=3.10 -y
conda activate kg_env

# Install PyTorch Geometric
pip install torch-geometric torch-scatter torch-sparse \
    -f https://data.pyg.org/whl/torch-2.1.0+cu118.html

# Install other dependencies
pip install -r requirements.txt
```

### 2. Train Models

```bash
# Train RNN model (baseline)
python main.py --model RNN --epochs 30 --lr 0.00005 --device cuda

# Train Temporal Attention model (best config)
python main.py --model Attention \
    --epochs 30 \
    --embed_dim 256 \
    --num_heads 8 \
    --num_gconv_layers 2 \
    --window_size 10 \
    --lr 0.0002 \
    --dropout 0.1 \
    --weight_decay 0.0001 \
    --grad_clip_norm 2.0 \
    --device cuda
```

### 3. Run All Models

```bash
# Official training run (all models, 30 epochs)
bash run_official_30epochs.sh

# Quick sanity check (2 epochs)
bash run_sanity_2epochs.sh
```

## 📁 Project Structure

```
financial-dynamic-knowledge-graph/
├── main.py                      # Main training script
├── report.md                    # Full project report (blog post format)
├── requirements.txt             # Python dependencies
│
├── src/
│   ├── models/
│   │   ├── kg_temporal_rnn.py           # RNN temporal model
│   │   ├── kg_temporal_attention.py     # Attention temporal model
│   │   └── pyg_modules/
│   │       ├── temporal_attention.py    # Temporal attention layer
│   │       ├── embedding_updater.py     # RNN embedding updater
│   │       └── ...
│   │
│   ├── data_processing/
│   │   └── pyg_dataset.py       # PyG dataset & dataloaders
│   │
│   └── evaluation/
│       └── metrics.py           # MRR, Hits@K metrics
│
├── data/FinDKG/                 # Dataset
├── sweeps/                      # WandB sweep configs
└── results/                     # Training outputs
```

## 🔬 Research Summary

### Hypothesis
> Replacing recurrent temporal updates with Transformer-style attention over each entity's recent history can capture richer temporal dependencies and improve predictive accuracy.

### Methodology
1. **Structural Update**: RGCN computes **zᵥ(τ)** for each entity at time τ
2. **Temporal Update**: 
   - RNN: **hᵥ(τ) = GRU(zᵥ(τ), hᵥ(τ−1))**
   - Attention: **hᵥ(τ) = αᵥ(τ) · Vᵥ(τ)** with multi-head attention over history
3. **Decoder**: Cross-entropy loss over candidate tails

### Three-Stage Hyperparameter Optimization
| Stage | Goal | Finding |
|-------|------|---------|
| A | Optimizer tuning | lr ≈ 0.0002, dropout 0.1, grad_clip 2.0 |
| B | Architecture sweep | 256-dim, 8 heads, 2 GNN layers, window 10 |
| C | Multi-seed confirmation | Configuration is stable across seeds |

### Results
- **Temporal Attention converges faster** and achieves **higher validation MRR/Hits@10**
- The benefit is **conditional on proper hyperparameter tuning**
- Without tuning, RNN can outperform Attention

## 📊 Dataset: FinDKG

Financial Dynamic Knowledge Graph for temporal link prediction.

- **Entities**: 13,645 (companies, people, countries, instruments)
- **Relations**: 15 types
- **Triplets**: 144,062 temporal facts
- **Timestamps**: 126 time steps
- **Task**: Predict future links *(h, r, ?, τ)* given history up to τ

## 📚 Documentation

- **[report.md](report.md)** - Full project report with methodology, results, and analysis
- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** - Technical architecture details
- **[docs/DATASET.md](docs/DATASET.md)** - Dataset information

## 📄 References

1. **Du, X. et al. (2024).** *FinDKG: Dynamic Knowledge Graphs with Large Language Models for Detecting Global Trends in Financial Markets.*
2. **Trivedi, R. et al. (2017).** *Know-Evolve: Deep Temporal Reasoning for Dynamic Knowledge Graphs.* ICML 2017.
3. **Trivedi, R. et al. (2019).** *DyRep: Learning Representations over Dynamic Graphs.* ICLR 2019.
4. **Xu, D. et al. (2020).** *Inductive Representation Learning on Temporal Graphs (TGAT).* ICLR 2020.
5. **Rossi, E. et al. (2020).** *Temporal Graph Networks for Deep Learning on Dynamic Graphs (TGN).* ICML 2020.
6. **Vaswani, A. et al. (2017).** *Attention Is All You Need.* NeurIPS 2017.

## 📧 Contact

- GitHub Issues: https://github.com/chenxifeng0823/financial-dynamic-knowledge-graph/issues

## 📝 License

This project is for research and educational purposes.

---

**Last Updated**: 2025-12-12  
**Status**: ✅ RNN Baseline | ✅ Temporal Attention | ✅ Hyperparameter Optimization | ✅ Report Complete
