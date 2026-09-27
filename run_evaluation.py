"""Stage 9 – Comprehensive Model & Explainability Evaluation Runner.

Evaluates:
1. Baseline GraphSAGE (Stage 4) on Test Split (F1, AUC, Precision, Recall)
2. Graph Transformer (Stage 5) on Test Split (F1, AUC, Precision, Recall)
3. Explainability Evaluation (Stage 6) on Sampled Flagged Transactions (Fidelity+, Sparsity, Latency)

Outputs:
- results_table.csv
- results_table.md
- Formatted console output

Usage:
    python run_evaluation.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict, Any

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

import config
from src.explain.explainer import FraudExplainer
from src.models.baseline import GraphSAGEClassifier
from src.models.graph_transformer import load_trained_graph_transformer
from src.preprocessing.graph_builder import load_graph


def evaluate_graphsage_test(data: Any, device: torch.device) -> Dict[str, float]:
    """Evaluate trained GraphSAGE baseline on test split."""
    ckpt_path = config.BASELINE_CHECKPOINT
    if not ckpt_path.exists():
        print(f"[WARN] Baseline checkpoint not found at {ckpt_path}.")
        return {}

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    in_channels = ckpt.get("in_channels", 168)
    hidden_dim = ckpt.get("hidden_dim", config.HIDDEN_DIM)
    dropout = ckpt.get("dropout", config.DROPOUT)

    model = GraphSAGEClassifier(
        in_channels=in_channels,
        hidden_channels=hidden_dim,
        dropout=dropout,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    test_mask = data.test_mask
    x = data.x[:, :in_channels].to(device)
    edge_index = data.edge_index.to(device)
    y_test = data.y[test_mask].cpu().numpy()

    with torch.no_grad():
        out = model(x, edge_index)
        probs = F.softmax(out[test_mask], dim=-1)[:, 1].cpu().numpy()
        preds = (probs >= 0.5).astype(int)

    return {
        "Accuracy": float(accuracy_score(y_test, preds)),
        "Precision": float(precision_score(y_test, preds, pos_label=1, zero_division=0)),
        "Recall": float(recall_score(y_test, preds, pos_label=1, zero_division=0)),
        "F1 (Illicit)": float(f1_score(y_test, preds, pos_label=1, zero_division=0)),
        "ROC-AUC": float(roc_auc_score(y_test, probs)),
    }


def evaluate_graph_transformer_test(data: Any, device: torch.device) -> Dict[str, float]:
    """Evaluate trained Graph Transformer on test split."""
    model, meta = load_trained_graph_transformer(device=device)
    test_mask = data.test_mask
    in_ch = model.in_channels
    x = data.x[:, :in_ch].to(device)
    edge_index = data.edge_index.to(device)
    y_test = data.y[test_mask].cpu().numpy()

    best_thresh = meta.get("best_threshold", 0.5)

    with torch.no_grad():
        out = model(x, edge_index)
        probs = F.softmax(out[test_mask], dim=-1)[:, 1].cpu().numpy()
        preds = (probs >= best_thresh).astype(int)

    return {
        "Accuracy": float(accuracy_score(y_test, preds)),
        "Precision": float(precision_score(y_test, preds, pos_label=1, zero_division=0)),
        "Recall": float(recall_score(y_test, preds, pos_label=1, zero_division=0)),
        "F1 (Illicit)": float(f1_score(y_test, preds, pos_label=1, zero_division=0)),
        "ROC-AUC": float(roc_auc_score(y_test, probs)),
        "Threshold": float(best_thresh),
    }


def evaluate_gnn_explainer(
    data: Any,
    device: torch.device,
    num_samples: int = 30,
) -> Dict[str, float]:
    """Evaluate GNNExplainer fidelity, sparsity, and latency across test illicit wallets."""
    print(f"\n[Stage 9] Running GNNExplainer evaluation across {num_samples} flagged transactions …")
    test_illicit = torch.where((data.y == 1) & data.test_mask)[0].tolist()
    if not test_illicit:
        test_illicit = torch.where(data.y == 1)[0].tolist()

    np.random.seed(42)
    sample_nodes = np.random.choice(test_illicit, size=min(num_samples, len(test_illicit)), replace=False)

    explainer = FraudExplainer(device=device, epochs=40)

    fidelities = []
    sparsities = []
    latencies = []

    for i, nid in enumerate(sample_nodes, 1):
        res = explainer.explain_node(int(nid))
        fidelities.append(res["fidelity_plus"])
        sparsities.append(res["sparsity"])
        latencies.append(res["elapsed_sec"])
        if i % 10 == 0 or i == len(sample_nodes):
            print(f"  [{i}/{len(sample_nodes)}] Evaluated node {nid} · Fid+: {res['fidelity_plus']:.4f} · Sparsity: {res['sparsity']:.2%}")

    return {
        "Mean Fidelity+": float(np.mean(fidelities)),
        "Std Fidelity+": float(np.std(fidelities)),
        "Mean Sparsity": float(np.mean(sparsities)),
        "Std Sparsity": float(np.std(sparsities)),
        "Mean Latency (s)": float(np.mean(latencies)),
        "Samples Evaluated": len(sample_nodes),
    }


def main():
    print("=" * 70)
    print("      XAI-GT: Blockchain Fraud Detection & Explainability Evaluation")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load graph
    data = load_graph()
    test_count = int(data.test_mask.sum().item())
    print(f"Loaded graph with {data.num_nodes:,} nodes and {data.edge_index.shape[1]:,} edges.")
    print(f"Test split nodes: {test_count:,} (Illicit: {int(((data.y == 1) & data.test_mask).sum()):,})")

    # 1. Evaluate Baseline
    print("\n--- Evaluating Baseline (GraphSAGE) ---")
    sage_metrics = evaluate_graphsage_test(data, device)
    for k, v in sage_metrics.items():
        print(f"  {k:15}: {v:.4f}")

    # 2. Evaluate Graph Transformer
    print("\n--- Evaluating Graph Transformer (XAI-GT) ---")
    gt_metrics = evaluate_graph_transformer_test(data, device)
    for k, v in gt_metrics.items():
        print(f"  {k:15}: {v:.4f}")

    # 3. Evaluate Explainability
    print("\n--- Evaluating Explainability (GNNExplainer) ---")
    exp_metrics = evaluate_gnn_explainer(data, device, num_samples=30)
    for k, v in exp_metrics.items():
        print(f"  {k:20}: {v:.4f}" if isinstance(v, float) else f"  {k:20}: {v}")

    # Build Comparison Table
    records = [
        {
            "Metric": "Test Illicit F1-Score",
            "GraphSAGE (Baseline)": f"{sage_metrics.get('F1 (Illicit)', 0.0):.4f}",
            "Graph Transformer (XAI-GT)": f"{gt_metrics.get('F1 (Illicit)', 0.0):.4f}",
            "Target / Expected": ">= 0.8500",
        },
        {
            "Metric": "Test ROC-AUC",
            "GraphSAGE (Baseline)": f"{sage_metrics.get('ROC-AUC', 0.0):.4f}",
            "Graph Transformer (XAI-GT)": f"{gt_metrics.get('ROC-AUC', 0.0):.4f}",
            "Target / Expected": ">= 0.9000",
        },
        {
            "Metric": "Test Precision (Illicit)",
            "GraphSAGE (Baseline)": f"{sage_metrics.get('Precision', 0.0):.4f}",
            "Graph Transformer (XAI-GT)": f"{gt_metrics.get('Precision', 0.0):.4f}",
            "Target / Expected": "High (Low False Alarm)",
        },
        {
            "Metric": "Test Recall (Illicit)",
            "GraphSAGE (Baseline)": f"{sage_metrics.get('Recall', 0.0):.4f}",
            "Graph Transformer (XAI-GT)": f"{gt_metrics.get('Recall', 0.0):.4f}",
            "Target / Expected": "High (Fraud Capture)",
        },
        {
            "Metric": "Test Accuracy",
            "GraphSAGE (Baseline)": f"{sage_metrics.get('Accuracy', 0.0):.4f}",
            "Graph Transformer (XAI-GT)": f"{gt_metrics.get('Accuracy', 0.0):.4f}",
            "Target / Expected": ">= 0.9500",
        },
        {
            "Metric": "Fidelity+ (Impact on Fraud Prob)",
            "GraphSAGE (Baseline)": "N/A",
            "Graph Transformer (XAI-GT)": f"{exp_metrics['Mean Fidelity+']:.4f} +/- {exp_metrics['Std Fidelity+']:.4f}",
            "Target / Expected": "> 0.3000 (Faithful)",
        },
        {
            "Metric": "Sparsity (% Irrelevant Edges Pruned)",
            "GraphSAGE (Baseline)": "N/A",
            "Graph Transformer (XAI-GT)": f"{exp_metrics['Mean Sparsity']:.1%}",
            "Target / Expected": "> 50.0% (Compact)",
        },
        {
            "Metric": "Attribution Latency (CPU)",
            "GraphSAGE (Baseline)": "N/A",
            "Graph Transformer (XAI-GT)": f"{exp_metrics['Mean Latency (s)']:.2f}s / wallet",
            "Target / Expected": "< 5.0s (Real-time)",
        },
    ]

    df_results = pd.DataFrame(records)

    # Save to CSV
    csv_path = PROJECT_ROOT / "results_table.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"\n[Saved] CSV results table to: {csv_path}")

    # Save to Markdown
    md_content = "# XAI-GT Experimental Results & Evaluation\n\n"
    md_content += "Performance evaluation on the Elliptic Bitcoin Transaction Dataset (temporal test split).\n\n"
    
    # Pure-python markdown table generator
    headers = list(df_results.columns)
    md_table = "| " + " | ".join(headers) + " |\n"
    md_table += "| " + " | ".join(["---"] * len(headers)) + " |\n"
    for _, row in df_results.iterrows():
        md_table += "| " + " | ".join(str(val) for val in row.values) + " |\n"
    md_content += md_table
    md_content += "\n\n### Evaluation Notes:\n"
    md_content += "- **Temporal Split**: Test evaluation strictly uses time-step splits (timesteps 35–49) to simulate true forward-in-time deployment.\n"
    md_content += "- **Fidelity+**: Measures average probability drop when explanatory edges identified by GNNExplainer are removed.\n"
    md_content += "- **Sparsity**: Proportion of edges pruned to isolate the core forensic pattern.\n"
    md_content += "- **Latency**: Average CPU inference & attribution time per investigated transaction.\n"

    md_path = PROJECT_ROOT / "results_table.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"[Saved] Markdown results table to: {md_path}")

    print("\n" + "=" * 70)
    print(md_content)
    print("=" * 70)


if __name__ == "__main__":
    main()
