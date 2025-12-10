---
title: Temporal Attention over Financial Knowledge Graphs
subtitle: From Static Embeddings to Heterogeneous Multi‑Aspect Temporal Reasoning
---

### Abstract

Many real‑world systems can be modeled as **heterogeneous temporal knowledge graphs**, where time‑stamped triplets (head, relation, tail, time) describe evolving interactions between diverse entities.  
Examples include users and items in recommender systems, accounts and transactions in fraud detection, entities and events in news or scientific corpora, and evolving knowledge bases such as Wikidata.  
Classical static knowledge graph embedding models and simple temporal extensions often assume that a single embedding per entity–relation pair is sufficient, ignoring how the *trajectory* of events and their ordering shape future predictions.  
Recent work on temporal KGs and dynamic graphs—such as **Know‑Evolve** (Trivedi et al., 2017), **DyRep** (Trivedi et al., 2019), **RE‑NET** (Jin et al., 2020), **TGAT** (Xu et al., 2020), and **TGN** (Rossi et al., 2020)—demonstrates that explicitly modeling time greatly improves performance, but also introduces new architectural and optimization challenges.  
In this project we systematically compare three classes of models for temporal link prediction on a large heterogeneous dynamic graph: (1) static baselines (TransE, DistMult, ComplEx), (2) a recurrent temporal graph model (KGTransformer with RNN‑style state updates), and (3) a temporal attention model that replaces recurrent updates with a Transformer‑style attention mechanism over each entity’s recent history.  
Our experiments show that static models underperform substantially once forecasting truly future links, that adding temporal structure via a recurrent model yields large gains, and that carefully tuned temporal attention can match or modestly surpass the RNN baseline while being more sensitive to hyperparameters.  
We provide a cleaned‑up codebase, detailed ablations, and practical lessons for training deep temporal models on large heterogeneous graphs.

### 1. Introduction

Our experiments are conducted on **FinDKG**, a large‑scale financial dynamic knowledge graph introduced by Du et al. (2024).  
FinDKG represents firms, sectors, indices, instruments and other financial entities as nodes; heterogeneous relations such as supply‑chain links, ownership, competition, co‑movement, and news‑based connections as edges; and associates each edge with a **time‑stamp** and, in many cases, additional attributes.  
The original FinDKG paper showed that this representation is powerful for downstream tasks such as risk propagation and counterparty analysis, but also highlighted that **temporal link prediction** on such a graph is unusually challenging.

There are several reasons why financial temporal KGs are hard:

- **High heterogeneity.** Multiple node and relation types interact (e.g., firm–firm trade, firm–index membership, firm–news, firm–sector), breaking many of the assumptions behind homogeneous temporal graph models such as TGAT (Xu et al., 2020) and TGN (Rossi et al., 2020).  
- **Multi‑aspect temporal signals.** Structural relations co‑evolve with prices, fundamentals, macro events and news; the same edge type can mean different things in different regimes.  
- **Non‑stationarity and regime shifts.** Shocks, crises and policy changes can abruptly change which patterns are predictive, making long‑range extrapolation difficult.  
- **Evaluation on truly future edges.** Following FinDKG, we predict links at later time intervals given only the past, rather than randomly masking edges within a single snapshot; this exposes weaknesses in purely static KG models such as TransE (Bordes et al., 2013), DistMult (Yang et al., 2015) and ComplEx (Trouillon et al., 2016).

Static KG embedding models and lightweight temporal variants (e.g., HyTE, Dasgupta et al., 2018; Know‑Evolve, Trivedi et al., 2017; RE‑NET, Jin et al., 2020) tend to compress each entity–relation pair into a small number of vectors and treat time as either a discrete index or an extra embedding dimension.  
On FinDKG, this proves insufficient: we find that **static and weakly temporal baselines underperform markedly** when asked to forecast future financial edges, because they discard the fine‑grained *trajectory* of interactions that precede each event.

To address this, the FinDKG benchmark includes a **deep temporal baseline** based on a KGTransformer with recurrent updates, in the spirit of DyRep (Trivedi et al., 2019) and other dynamic KG models.  
The model maintains a **per‑entity temporal state** that is updated as new edges arrive, and uses this state together with static and structural embeddings to predict future links.  
This recurrent baseline already demonstrates that **explicit temporal modeling is crucial**: it significantly outperforms the static models on temporal link prediction metrics such as mean reciprocal rank (MRR) and Hits@10.

In this report we explore a stronger hypothesis:

> On complex, heterogeneous financial graphs, **replacing recurrent temporal updates with a Transformer‑style attention mechanism** over each entity’s recent history can capture richer temporal dependencies and potentially improve predictive accuracy.

Concretely, we extend the FinDKG baseline with a **temporal attention module** that:

1. Uses a relational graph convolution (RGCN / KGTransformer) to compute structural node embeddings at each time step.  
2. Maintains, for each entity, a sliding window of its recent structural states and time‑stamps.  
3. Applies multi‑head scaled dot‑product attention (Vaswani et al., 2017) over this window, using temporal positional encodings to weight past states according to both content and time gaps.  
4. Feeds the resulting temporal embedding into a link‑prediction decoder together with static and structural embeddings.

This design aims to generalize the recurrent FinDKG model in the same way that Transformers generalize RNNs in sequence modeling: if attention focuses mostly on the latest state, it can imitate an RNN; if it learns to mix information across distant but relevant events, it can surpass the RNN on scenarios with long‑range or regime‑dependent dependencies.

However, building a stable temporal attention model for FinDKG is non‑trivial.  
We encounter and address practical issues around **two‑stage temporal training** (using cumulative graphs for state updates but only current batches for prediction), **heterogeneous relational structure**, **numerical stability of attention over sparse histories**, and **sensitivity to hyperparameters and architectural choices**.  
The remainder of this report details our modeling choices, debugging process, and a three‑stage optimization procedure that ultimately yields a competitive temporal attention model for FinDKG.

### 2. Models and Learning Objective

In this section we describe the core models we evaluate and the learning signal they receive.  
Throughout, a time‑stamped edge (event) is written as `(h, r, t, τ)` where `h` is the head entity, `r` the relation type, `t` the tail entity, and `τ` the time index.

#### 2.1. Static knowledge graph baselines

Static models ignore time `τ` and assign each entity `e` and relation `r` a time‑invariant embedding vector `e_e ∈ R^d` and `e_r ∈ R^d`.  
They define a score `s(h, r, t)` that should be high for true triplets and low for negatives, for example:

- **TransE** (Bordes et al., 2013):  
  `s(h, r, t) = - || e_h + e_r - e_t ||_2`
- **DistMult** (Yang et al., 2015):  
  `s(h, r, t) = < e_h, e_r, e_t >` (tri‑linear dot‑product)
- **ComplEx** (Trouillon et al., 2016): a complex‑valued extension of DistMult.

Given one positive triplet and a set of sampled negative tails `{t_j^-}`, our training objective for these baselines is a softplus ranking loss of the form  
`L_static = E[ log(1 + exp(-s(h, r, t^+))) + (1/K) * sum_j log(1 + exp(s(h, r, t_j^-))) ]`,  
where the expectation is over edges and negative samples.  
Gradients from this loss update only the static embeddings and (for ComplEx/DistMult) the relation parameters.

#### 2.2. Recurrent temporal model (FinDKG baseline)

The FinDKG baseline augments static embeddings with **structural** and **temporal** components that evolve over time.

- Each entity `v` has:
  - a static embedding `x_v ∈ R^d`,
  - a time‑dependent structural embedding `z_v^τ ∈ R^d` produced by a relational GNN, and
  - a temporal state (hidden state) `h_v^τ ∈ R^d` updated by an RNN.

##### 2.2.1. Structural update (graph convolution)

At each discrete time bucket `τ` we build a **cumulative graph** `G_≤τ` containing all edges with time `≤ τ`.  
Given entity features `{x_v}`, a relational GNN such as RGCN computes structural embeddings  
`z_v^τ = GNN_θ(v, G_≤τ, {x_u})`.  
Intuitively, `z_v^τ` summarizes the local multi‑hop neighborhood of `v` in the graph observed up to time `τ`.

##### 2.2.2. Temporal update (RNN)

The recurrent module consumes the structural sequence for each entity:  
`h_v^τ = GRU_φ(z_v^τ, h_v^{τ-1})`, with `h_v^0 = 0`.  
This hidden state is a compressed summary of the *trajectory* of structural embeddings for `v` up to time `τ`.

##### 2.2.3. Decoder and loss

For an edge `(h, r, t, τ)` in the **current batch** at time `τ`, we form a concatenated representation for the head:  
`u_h^τ = [x_h || h_h^τ]`, and obtain a relation embedding `r^τ` (static or temporal).  
A decoder `f_ψ` maps `(u_h^τ, r^τ)` to scores over all candidate tails: `s^τ = f_ψ(u_h^τ, r^τ) ∈ R^{|E|}`.  
We train with a cross‑entropy loss against the true tail index `t`:  
`L_RNN^τ = - log( exp(s_t^τ) / sum_{t'} exp(s_{t'}^τ) )`.  
The **two‑stage scheme** is: (1) update `h_v^τ` using the cumulative graph `G_≤τ` (temporal context), then (2) compute the decoder loss only on edges in the current batch (memory‑efficient prediction).

##### 2.2.4. Backpropagation path

For each batch at time step \(\tau\):

1. Forward pass computes `z^τ` via the GNN, then `h^τ` via the GRU, then scores `s^τ` via the decoder.  
2. The loss `L_RNN^τ` backpropagates gradients:
   - from the decoder into `u_h^τ` and relation parameters,  
   - from `u_h^τ` into the temporal state `h_h^τ` and static embeddings,  
   - from `h_h^τ` back through time via the GRU to earlier states `h_h^{τ-1}, h_h^{τ-2}, …` (truncated by the effective sequence length), and  
   - from `z_v^τ` through the GNN layers into static embeddings and GNN weights.  
3. Gradients from multiple time steps are accumulated before an optimizer step (AdamW), so earlier events influence both the GNN and RNN parameters via backpropagation through time.

#### 2.3. Temporal attention model

The temporal attention model keeps the same overall structure—static + structural + temporal components—but replaces the GRU with a **Transformer‑style attention layer** that looks back over a fixed‑length history window for each entity.

For each entity `v` at time `τ` we maintain a history buffer  
`H_v^τ = [ z_v^{τ-K}, …, z_v^{τ-1} ] ∈ R^{K × d}` and time indices `τ_v^τ = [τ-K, …, τ-1]`,  
where `K` is the window size (for example 10).  
The current structural embedding `z_v^τ` plays the role of the **query**, and the history acts as keys and values.

##### 2.3.1. Single‑head temporal attention

For clarity, consider a single attention head with parameters `W_Q, W_K, W_V ∈ R^{d × d_h}`.  
We compute

- `q_v^τ = W_Q z_v^τ` (query for the current state)  
- `K_v^τ = H_v^τ W_K` (keys for the history window)  
- `V_v^τ = H_v^τ W_V` (values for the history window)

To inject temporal information we add sinusoidal or learned encodings `φ(Δτ)` of time gaps `Δτ = τ - τ_v^τ` to the keys.  
Attention scores and weights are then  
`α_v^τ = softmax( q_v^τ (K_v^τ + φ(Δτ))^T / sqrt(d_h) + mask )`,  
where `mask` turns off empty history slots.  
The new temporal state is a weighted sum of values:  
`h_v^τ = α_v^τ V_v^τ ∈ R^{d_h}`.  
Multi‑head attention runs this computation in parallel for several heads and concatenates the resulting `h_v^τ` vectors, followed by a linear projection.

##### 2.3.2. Decoder and loss

As in the RNN model, we form a head representation `u_h^τ = [x_h || h_h^τ]`,  
concatenate with a relation embedding `r^τ`, and pass through a decoder `f_ψ` to obtain scores over candidate tails.  
The loss is the same cross‑entropy link‑prediction objective as for the RNN model, summed over time steps:  
`L_Attn = sum_τ L_Attn^τ`.

##### 2.3.3. Forward pass with two‑stage training

For each time bucket \(\tau\) we:

1. **Structural update.** Run the relational GNN on the cumulative graph \(G_{\le\tau}\) to obtain \(\mathbf{z}_v^\tau\) for entities touched in the current cumulative batch.  
2. **History lookup.** For each such entity, gather its history buffer \(H_v^\tau\) and time indices \(\boldsymbol{\tau}_v^\tau\) from the global `AttentionState`.  
3. **Temporal attention.** Compute \(\mathbf{h}_v^\tau\) via the attention equations above.  
4. **State write‑back.** Store the new structural state and history (sliding window) back into the global buffers (in our implementation these stored tensors are detached from the computation graph to control memory).  
5. **Prediction.** For edges in the *current batch only*, compute scores and cross‑entropy loss using \(\mathbf{h}_v^\tau\) and static embeddings.

##### 2.3.4. Backpropagation path

Given losses `{L_Attn^τ}` across time:

1. Gradients flow from the decoder into the temporal states `h_v^τ`, static embeddings, and relation embeddings.  
2. From `h_v^τ` they backpropagate into the attention weights `α_v^τ`, and further into:
   - the current structural embeddings `z_v^τ` (via `q_v^τ`), and  
   - the attention parameters `W_Q, W_K, W_V` and temporal encodings.  
3. Gradients on `z_v^τ` then propagate through the GNN layers into static embeddings and GNN weights.  
4. In our implementation, past history vectors stored in the sliding window are treated as constants (we detach them when writing into the buffer), so backpropagation does **not** traverse arbitrarily far back in time through attention; this is a deliberate design choice to stabilize training on long sequences while still allowing attention to *read* from historical context.

#### 2.4. Optimization and evaluation

All models are trained with **AdamW** using mini‑batches of time‑ordered edges.  
For temporal models we use the **two‑stage** scheme (cumulative update + batch‑level prediction) to balance temporal context and memory use, and apply gradient clipping by global norm to avoid exploding gradients.  
We evaluate using ranking metrics—mean reciprocal rank (MRR) and Hits@10—computed by scoring all candidate tails for each \((h, r, ?, \tau)\) query on the validation and test sets.

#### 2.5. Why hyperparameter search matters more for temporal attention

The temporal attention model is **strictly more flexible** than the recurrent baseline: in addition to optimizer choices (learning rate, weight decay, dropout, gradient clipping), its behavior depends heavily on architectural knobs such as embedding dimension, number of GNN layers, number of attention heads, and history window size.  
In practice this extra capacity made the model *much* more sensitive to initialization and hyperparameters—small changes in learning rate or architecture could move it from “barely learning” to clearly outperforming static baselines—so we designed a staged optimization procedure.

At a high level, we used three stages and retained only the best configuration from each:

| **Stage** | **Goal** | **Best configuration (Attention model)** | **Finding** |
|-----------|----------|-------------------------------------------|-------------|
| **Stage A – Training hyperparams** | Get a convergent training setup. | **`lr = 1.0`, `dropout = 0.1`, `grad_clip_norm = 5.0`, `window_size = 5`, `num_gconv_layers = 2`** (before correcting LR scale). | Showed that very large `lr` yields low but noisy validation MRR and motivated moving to a much smaller `lr ≈ 0.0002` with moderate clipping and dropout. |
| **Stage B – Architecture sweep** | Find a strong model shape with optimizer fixed. | **`embed_dim = 256`, `num_heads = 8`, `num_gconv_layers = 2`, `window_size = 10`, `lr = 0.0002`, `dropout = 0.1`, `weight_decay = 0.0001`, `grad_clip_norm = 2.0`, `epochs = 20`**. | Identified a 256‑dim, 2‑layer, 8‑head architecture with window 10 as the most effective and stable across runs. |
| **Stage C – Multi‑seed confirmation** | Check robustness of the best config. | **Same as Stage B**, evaluated with seeds 41, 42 and 43. | All three seeds produce very similar validation and test MRR / Hits@10, confirming that the chosen Attention configuration is stable rather than a lucky seed. |

These stages turned the temporal attention model from something that *theoretically* should beat static approaches but initially underperformed, into a **reliable and well‑understood baseline** that we can fairly compare against the recurrent FinDKG model in later sections.


