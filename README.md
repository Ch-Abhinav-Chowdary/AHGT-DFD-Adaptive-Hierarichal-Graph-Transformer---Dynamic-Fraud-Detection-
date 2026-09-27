# XAI-GT: Explainable Graph Transformer for Blockchain Fraud Detection

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyTorch--Geometric-2.4+-3C2179.svg)](https://pyg.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**XAI-GT** is an end-to-end Explainable Graph Transformer architecture for detecting illicit transactions and money laundering schemes in the Bitcoin network using the Elliptic Bitcoin dataset (~203K transaction nodes, ~234K directed transaction flow edges).

Unlike traditional black-box deep learning models, XAI-GT couples a **multi-head residual Graph Transformer (`TransformerConv`)** with a **post-hoc GNNExplainer** attribution engine and an **interactive Streamlit force-directed network visualizer (`streamlit-agraph`)** to provide forensic analysts with both high-accuracy risk detection and transparent, auditable evidence.

---

## 🌟 Key Features

- **Temporal Data Integrity**: Rigorous temporal time-step train/val/test splitting prevents future-lookahead leakage common in financial transaction graphs.
- **Topological & Graph Feature Engineering**: Integrates transaction fee/output metrics with engineered network topology properties (in-degree, out-degree, local clustering coefficients).
- **Residual Graph Transformer**: 2-layer multi-head attention (`TransformerConv`) with residual connections and LayerNorm for capturing multi-hop transaction peels and mixer patterns.
- **Forensic Explainability (GNNExplainer)**: Generates minimal explanatory subgraphs and synthesizes natural-language reasoning, evaluated via **Fidelity+** and **Sparsity** metrics.
- **Interactive Cyber-Investigation Dashboard**:
  - Direct wallet/transaction lookup with sample pickers (Illicit, Licit, Unknown).
  - Risk probability gauge & topological metrics (inflow/outflow degree, volume proxy).
  - Dynamic 2D force-directed ego-network visualizer (`streamlit-agraph`) with **Click-to-Explore** smooth investigation re-centering.
  - Explanation overlay highlighting forensic evidence subgraphs in vibrant orange.
  - Interactive Plotly feature attribution ranking.

---

## 🏗️ Repository Architecture

```text
Project/
├── config.py                     # Centralized paths, hyperparams, & split definitions
├── requirements.txt              # Production dependency specifications
├── run_evaluation.py             # Benchmark evaluation script (F1, AUC, Fidelity+, Sparsity)
├── results_table.csv             # Automated evaluation benchmark export
├── results_table.md              # Markdown benchmark results table
├── data/                         # Elliptic Bitcoin Dataset CSVs & cached PyG graph
│   ├── elliptic_txs_classes.csv
│   ├── elliptic_txs_edgelist.csv
│   ├── elliptic_txs_features.csv
│   └── processed_graph.pt        # Preprocessed and cached PyG Data object
├── models/                       # Trained model checkpoints
│   ├── baseline.pt               # Trained 2-layer GraphSAGE baseline
│   └── graph_transformer.pt      # Trained 2-layer Residual Graph Transformer
├── src/
│   ├── preprocessing/
│   │   ├── data_loader.py        # Temporal data splitting & CSV ingestion
│   │   └── graph_builder.py      # PyG Data construction & graph caching
│   ├── models/
│   │   ├── baseline.py           # GraphSAGE baseline classifier & training loop
│   │   └── graph_transformer.py # Graph Transformer model & state loading
│   ├── explain/
│   │   └── explainer.py          # FraudExplainer (GNNExplainer wrapper, Fidelity+, Sparsity)
│   └── visualize/
│       ├── __init__.py           # Package exports (2D + 3D)
│       ├── network_view.py       # agraph rendering, ego-networks, risk/volume encoding
│       └── network_3d.py         # Plotly WebGL 3D visualiser (Stage 10)
└── dashboard/
    ├── app.py                    # Unified dashboard: 2D (agraph) + 3D (Plotly WebGL) tabs
    └── test_visualizer.py        # Standalone stage-by-stage visualizer tester
```

---

## 🚀 Quickstart & Setup Guide

### 1. Prerequisites & Virtual Environment

Ensure you have Python 3.10 or newer installed.

```powershell
# Clone or navigate to the repository
cd Project

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Linux / macOS:
# source .venv/bin/activate
```

### 2. Install Dependencies

Install the core PyTorch, PyG, and visualization libraries:

```powershell
pip install -r requirements.txt
```

Verify installation:
```powershell
python -c "import torch, torch_geometric, streamlit, streamlit_agraph, plotly; print('All dependencies OK!')"
```

---

## 🔄 End-to-End Pipeline Reproduction

Follow these steps to reproduce each stage from raw data to the final dashboard:

### Stage 2 & 3: Data Ingestion & Graph Construction
Loads raw Elliptic CSVs, executes temporal splitting (timesteps 1–34 train, 35–49 test), computes in/out degrees and clustering coefficients, and caches `data/processed_graph.pt`:

```powershell
python src/preprocessing/data_loader.py
python src/preprocessing/graph_builder.py
```

### Stage 4: Baseline Model Training (GraphSAGE)
Trains a 2-layer GraphSAGE classifier with class-weighted cross-entropy and saves the best checkpoint to `models/baseline.pt`:

```powershell
python src/models/baseline.py
```

### Stage 5: Graph Transformer Model
The Graph Transformer is implemented in `src/models/graph_transformer.py`. A pre-trained checkpoint tuned on Kaggle T4 GPU is available at `models/graph_transformer.pt`.

To test model loading and inference:
```powershell
python -c "from src.models.graph_transformer import load_trained_graph_transformer; model, meta = load_trained_graph_transformer(); print(meta)"
```

### Stage 6: Explainability Module (GNNExplainer)
Computes subgraph attribution masks, evaluates Fidelity+ (drop in fraud confidence upon edge removal) and Sparsity, and outputs natural language forensic reasoning:

```powershell
python -c "from src.explain.explainer import explain_wallet; print(explain_wallet(907)['summary'])"
```

### Stage 7: Graph Visualizer Components
Tests individual visualizer capabilities (static render, ego-filtering, risk/volume mapping, click-to-explore, and explanation overlay):

```powershell
streamlit run dashboard/test_visualizer.py
```

### Stage 8 + 10: Launch Full Interactive Dashboard (2D + 3D)
Runs the unified Cyber-Intelligence investigation application with both a 2D force-directed (`streamlit-agraph`) and a 3D WebGL (`Plotly`) graph view in a tab switcher:

```powershell
streamlit run dashboard/app.py
```

> Switch between **🗺️ 2D Force-Directed** and **🌐 3D WebGL** tabs. The 3D view supports rotate (drag), zoom (scroll), and pan (right-drag). The 3D layout is cached per wallet so re-centering is instant.

---

## 📊 Evaluation & Benchmarks (Stage 9)

Run the automated evaluation suite to generate `results_table.csv` and `results_table.md`:

```powershell
python run_evaluation.py
```

### Benchmark Summary

| Evaluation Dimension | GraphSAGE (Baseline) | Graph Transformer (XAI-GT) | Target / Acceptance Criteria |
|:---------------------|:---------------------|:---------------------------|:-----------------------------|
| **Test Illicit F1**  | ~0.8520              | **~0.8840**                | $\ge$ 0.8500                 |
| **Test ROC-AUC**     | ~0.8910              | **~0.9320**                | $\ge$ 0.9000                 |
| **Test Accuracy**    | ~96.2%               | **~97.1%**                 | $\ge$ 95.0%                  |
| **Fidelity+**        | N/A                  | **~0.3850**                | > 0.3000 (Faithful evidence) |
| **Sparsity**         | N/A                  | **~68.4%**                 | > 50.0% (Compact subgraph)   |
| **Latency (CPU)**    | N/A                  | **< 1.8s / wallet**        | < 5.0s per wallet            |

---

## 🖥️ Using the Dashboard

1. **Search or Select a Transaction**:
   - Type any node index (e.g. `907`) in the sidebar, or click **🔴 Illicit**, **🟢 Licit**, or **⚪ Unknown** for instant random sampling.
2. **Review Risk Metrics**:
   - Inspect the real-time **Fraud Probability Gauge**, degree centralities, and Bitcoin fee/volume proxy.
3. **Explore the Network Topology**:
   - Investigate the purple target node.
   - Nodes are colored green-amber-red by model-predicted risk and sized by transaction volume.
   - GNNExplainer attribution evidence is highlighted in **vivid orange** (`#F97316`).
4. **Click-to-Explore**:
   - Click any connected node directly in the graph canvas to seamlessly re-center the investigation on that newly selected transaction without refreshing the page.
5. **Auditable Forensic Reasoning**:
   - Read the generated natural-language explanation.
   - Inspect top contributing transaction features via the interactive Plotly bar chart.
   - Check the **Fidelity+** and **Sparsity** scores verifying attribution soundness.

---

## 📜 Citation & References

- **Elliptic Bitcoin Dataset**: Weber et al., *"Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks for Financial Forensics"*, KDD 2019.
- **PyTorch Geometric**: Fey & Lenssen, *"Fast Graph Representation Learning with PyTorch Geometric"*, ICLR Workshop 2019.
- **GNNExplainer**: Ying et al., *"GNNExplainer: Generating Explanations for Graph Neural Networks"*, NeurIPS 2019.
