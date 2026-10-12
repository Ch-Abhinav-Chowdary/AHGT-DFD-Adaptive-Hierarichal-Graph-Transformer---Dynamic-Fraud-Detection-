"""Inference and prediction engine for XAI-GT.

Provides:
- predict_custom_transaction: evaluates a custom or simulated transaction
  with specified topological and transactional attributes.
- predict_node_transaction: evaluates an existing node from the Elliptic graph.
- ARCHETYPE_PRESETS: pre-configured behavioral fraud and normal patterns.
- generate_forensic_reasoning: synthesizes human-auditable AML explanations.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.utils import k_hop_subgraph


# ---------------------------------------------------------------------------
# Archetype Presets for Scenario Simulation
# ---------------------------------------------------------------------------
ARCHETYPE_PRESETS: Dict[str, Dict[str, Any]] = {
    "🚨 Peeling Chain Laundering": {
        "description": "Rapid serial dispersion where a single input is split into dozens of small outputs to obfuscate tracking.",
        "in_degree": 1,
        "out_degree": 42,
        "clustering_coef": 0.02,
        "volume_proxy": 2.8,
        "fee_proxy": 1.9,
        "time_step": 36,
        "neighbor_risk": "high",
        "expected_label": "Illicit / Laundering",
    },
    "🌪️ Mixer / Tumbler Cycle": {
        "description": "High-clustering cyclic aggregation where funds circulate through interconnected pooling addresses.",
        "in_degree": 28,
        "out_degree": 32,
        "clustering_coef": 0.68,
        "volume_proxy": 1.4,
        "fee_proxy": 2.5,
        "time_step": 38,
        "neighbor_risk": "high",
        "expected_label": "Illicit / Mixer",
    },
    "🛒 Legitimate Retail Payment": {
        "description": "Standard point-of-sale or merchant transaction with 1-2 inputs and 1-2 outputs to low-risk counter-parties.",
        "in_degree": 2,
        "out_degree": 2,
        "clustering_coef": 0.00,
        "volume_proxy": -0.4,
        "fee_proxy": -0.2,
        "time_step": 40,
        "neighbor_risk": "low",
        "expected_label": "Licit / Clean",
    },
    "🐋 Whale Cold Storage Vault": {
        "description": "Infrequent, very high-value transfer between cold storage custody addresses with minimal connectivity.",
        "in_degree": 1,
        "out_degree": 2,
        "clustering_coef": 0.00,
        "volume_proxy": 4.5,
        "fee_proxy": 0.8,
        "time_step": 42,
        "neighbor_risk": "low",
        "expected_label": "Licit / High-Net-Worth",
    },
    "🔄 Exchange Deposit Consolidator": {
        "description": "High fan-in aggregation where dozens of customer deposits are swept into a single exchange omnibus wallet.",
        "in_degree": 55,
        "out_degree": 1,
        "clustering_coef": 0.05,
        "volume_proxy": 1.2,
        "fee_proxy": 0.4,
        "time_step": 44,
        "neighbor_risk": "mixed",
        "expected_label": "Licit / Exchange Infrastructure",
    },
    "⚡ Structuring / Smurfing Pattern": {
        "description": "Multiple below-threshold inputs consolidated and immediately fanned out to avoid regulatory detection thresholds.",
        "in_degree": 18,
        "out_degree": 16,
        "clustering_coef": 0.25,
        "volume_proxy": 0.1,
        "fee_proxy": 1.8,
        "time_step": 39,
        "neighbor_risk": "high",
        "expected_label": "Suspicious / Smurfing",
    },
}


def build_synthetic_ego_graph(
    in_degree: int,
    out_degree: int,
    clustering_coef: float,
    volume_proxy: float,
    fee_proxy: float,
    time_step: int,
    neighbor_risk: str = "low",
    base_features: Optional[np.ndarray] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Constructs a synthetic ego-graph centered at node 0 with realistic
    topological connections and feature distributions matching Elliptic.
    """
    total_neighbors = max(1, in_degree + out_degree)
    num_nodes = 1 + total_neighbors

    # Node 0 is the focal transaction
    edges = []
    curr_id = 1

    # Inflow edges (incoming to node 0)
    in_neighbors = []
    for _ in range(max(1, in_degree)):
        edges.append((curr_id, 0))
        in_neighbors.append(curr_id)
        curr_id += 1

    # Outflow edges (outgoing from node 0)
    out_neighbors = []
    for _ in range(max(1, out_degree)):
        edges.append((0, curr_id))
        out_neighbors.append(curr_id)
        curr_id += 1

    # Add clustering triangles if clustering_coef > 0
    if clustering_coef > 0.05 and len(in_neighbors) > 1 and len(out_neighbors) > 1:
        num_triangles = int(np.clip(clustering_coef * min(len(in_neighbors), len(out_neighbors)), 1, 10))
        for t in range(num_triangles):
            src_n = in_neighbors[t % len(in_neighbors)]
            dst_n = out_neighbors[t % len(out_neighbors)]
            edges.append((src_n, dst_n))

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    # Construct node features X (num_nodes, 168)
    X = np.zeros((num_nodes, 168), dtype=np.float32)

    # Base feature template (standard normal baseline if none provided)
    if base_features is not None:
        X[:] = base_features[:168]
    else:
        X[:] = np.random.normal(0.0, 0.3, size=(num_nodes, 168)).astype(np.float32)

    # Set focal node (node 0)
    X[0, 0] = float(volume_proxy)      # volume / fee proxy
    X[0, 1] = float(fee_proxy)         # fee proxy
    X[0, 2] = float(time_step) / 49.0  # normalized time-step

    # Set degree features at 165, 166, 167 (standard Elliptic indices)
    # Degrees in Elliptic are normalized by max (~500)
    X[0, 165] = float(np.clip(in_degree / 100.0, 0.0, 1.0))
    X[0, 166] = float(np.clip(out_degree / 100.0, 0.0, 1.0))
    X[0, 167] = float(np.clip(clustering_coef, 0.0, 1.0))

    # Configure neighbor node features according to risk bias
    risk_factor = 1.8 if neighbor_risk == "high" else (-0.6 if neighbor_risk == "low" else 0.4)
    for nid in range(1, num_nodes):
        X[nid, 0] = float(volume_proxy * 0.7 + np.random.normal(0, 0.2))
        X[nid, 1] = float(fee_proxy * 0.8 + np.random.normal(0, 0.2))
        X[nid, 165] = float(np.clip(np.random.uniform(0.02, 0.3) + risk_factor * 0.1, 0, 1))
        X[nid, 166] = float(np.clip(np.random.uniform(0.02, 0.3) + risk_factor * 0.1, 0, 1))
        X[nid, 167] = float(np.clip(clustering_coef * 0.8, 0, 1))
        # Neighbor local feature bias
        X[nid, 3:30] += risk_factor * 0.4

    return torch.tensor(X, dtype=torch.float32), edge_index


def predict_custom_transaction(
    model: torch.nn.Module,
    baseline_model: Optional[torch.nn.Module],
    in_degree: int,
    out_degree: int,
    clustering_coef: float,
    volume_proxy: float,
    fee_proxy: float,
    time_step: int = 42,
    neighbor_risk: str = "low",
    threshold: float = 0.50,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Runs forward pass for both Graph Transformer and GraphSAGE on a
    simulated transaction and returns calibrated predictions and forensic metrics.
    """
    device = device or torch.device("cpu")
    model = model.to(device).eval()

    X, edge_index = build_synthetic_ego_graph(
        in_degree=in_degree,
        out_degree=out_degree,
        clustering_coef=clustering_coef,
        volume_proxy=volume_proxy,
        fee_proxy=fee_proxy,
        time_step=time_step,
        neighbor_risk=neighbor_risk,
    )
    X = X.to(device)
    edge_index = edge_index.to(device)

    # Disable explain mode masks if any exist on the model
    for module in model.modules():
        if hasattr(module, "explain"):
            module.explain = False
        if hasattr(module, "edge_mask"):
            module.edge_mask = None

    # Format input for Graph Transformer
    gt_in_ch = getattr(model, "in_channels", 165)
    X_gt = X[:, :gt_in_ch] if X.shape[1] >= gt_in_ch else F.pad(X, (0, gt_in_ch - X.shape[1]))

    with torch.no_grad():
        gt_logits = model(X_gt, edge_index)
        gt_probs = F.softmax(gt_logits, dim=-1)
        gt_fraud_risk = float(gt_probs[0, 1].item())

        baseline_fraud_risk = None
        if baseline_model is not None:
            baseline_model = baseline_model.to(device).eval()
            b_in_ch = getattr(baseline_model, "in_channels", 168)
            X_b = X[:, :b_in_ch] if X.shape[1] >= b_in_ch else F.pad(X, (0, b_in_ch - X.shape[1]))
            b_logits = baseline_model(X_b, edge_index)
            b_probs = F.softmax(b_logits, dim=-1)
            baseline_fraud_risk = float(b_probs[0, 1].item())

    # Decision outcome
    is_fraud = gt_fraud_risk >= threshold
    if gt_fraud_risk >= 0.70:
        verdict = "HIGH RISK / FRAUD (ILLICIT)"
        badge_type = "fraud"
    elif gt_fraud_risk >= 0.40:
        verdict = "SUSPICIOUS / ELEVATED RISK"
        badge_type = "suspicious"
    else:
        verdict = "LOW RISK / CLEAN (LICIT)"
        badge_type = "clean"

    # Synthesize plain-language forensic reasoning
    reasoning = generate_forensic_reasoning(
        fraud_risk=gt_fraud_risk,
        in_degree=in_degree,
        out_degree=out_degree,
        clustering_coef=clustering_coef,
        volume_proxy=volume_proxy,
        neighbor_risk=neighbor_risk,
    )

    # Key driving factors
    feature_impacts = [
        {"feature": "Outflow Dispersion (Out-Degree)", "value": f"{out_degree} destinations", "impact": "High" if out_degree > 20 else "Low"},
        {"feature": "Clustering Coefficient", "value": f"{clustering_coef:.2f}", "impact": "High" if clustering_coef > 0.4 else "Low"},
        {"feature": "Transaction Volume Proxy", "value": f"{volume_proxy:+.2f} σ", "impact": "High" if abs(volume_proxy) > 2.0 else "Normal"},
        {"feature": "Inflow Consolidation (In-Degree)", "value": f"{in_degree} inputs", "impact": "High" if in_degree > 25 else "Low"},
        {"feature": "Counterparty Risk Bias", "value": neighbor_risk.capitalize(), "impact": "High" if neighbor_risk == "high" else "Low"},
    ]

    return {
        "verdict": verdict,
        "badge_type": badge_type,
        "is_fraud": is_fraud,
        "fraud_risk": gt_fraud_risk,
        "clean_prob": 1.0 - gt_fraud_risk,
        "baseline_fraud_risk": baseline_fraud_risk,
        "confidence": max(gt_fraud_risk, 1.0 - gt_fraud_risk),
        "threshold": threshold,
        "reasoning": reasoning,
        "feature_impacts": feature_impacts,
        "num_nodes": X.shape[0],
        "num_edges": edge_index.shape[1],
    }


def predict_node_transaction(
    node_id: int,
    graph_data: Any,
    model: torch.nn.Module,
    baseline_model: Optional[torch.nn.Module] = None,
    threshold: float = 0.50,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Runs full forward pass on an existing node from the Elliptic graph
    using its exact 2-hop k-hop induced neighborhood subgraph.
    """
    device = device or torch.device("cpu")
    model = model.to(device).eval()

    # Clear explain masks
    for module in model.modules():
        if hasattr(module, "explain"):
            module.explain = False
        if hasattr(module, "edge_mask"):
            module.edge_mask = None

    x_full = graph_data.x.to(device)
    ei_full = graph_data.edge_index.to(device)

    target_tensor = torch.tensor([node_id], dtype=torch.long, device=device)
    subset, sub_ei, mapping, _ = k_hop_subgraph(
        node_idx=target_tensor,
        num_hops=2,
        edge_index=ei_full,
        relabel_nodes=True,
    )

    sub_x_raw = x_full[subset]
    local_pos = mapping[0].item()

    # Model input
    gt_in_ch = getattr(model, "in_channels", 165)
    sub_x_gt = sub_x_raw[:, :gt_in_ch] if sub_x_raw.shape[1] >= gt_in_ch else F.pad(sub_x_raw, (0, gt_in_ch - sub_x_raw.shape[1]))

    with torch.no_grad():
        gt_logits = model(sub_x_gt, sub_ei)
        gt_probs = F.softmax(gt_logits, dim=-1)
        gt_fraud_risk = float(gt_probs[local_pos, 1].item())

        baseline_fraud_risk = None
        if baseline_model is not None:
            baseline_model = baseline_model.to(device).eval()
            b_in_ch = getattr(baseline_model, "in_channels", 168)
            sub_x_b = sub_x_raw[:, :b_in_ch] if sub_x_raw.shape[1] >= b_in_ch else F.pad(sub_x_raw, (0, b_in_ch - sub_x_raw.shape[1]))
            b_logits = baseline_model(sub_x_b, sub_ei)
            b_probs = F.softmax(b_logits, dim=-1)
            baseline_fraud_risk = float(b_probs[local_pos, 1].item())

    # Ground truth
    gt_label = int(graph_data.y[node_id].item()) if node_id < len(graph_data.y) else -1
    label_map = {1: "Verified Illicit", 0: "Verified Licit", -1: "Unlabelled / Unknown"}
    ground_truth_str = label_map.get(gt_label, "Unknown")

    if gt_fraud_risk >= 0.70:
        verdict = "HIGH RISK / FRAUD (ILLICIT)"
        badge_type = "fraud"
    elif gt_fraud_risk >= 0.40:
        verdict = "SUSPICIOUS / ELEVATED RISK"
        badge_type = "suspicious"
    else:
        verdict = "LOW RISK / CLEAN (LICIT)"
        badge_type = "clean"

    # Topological stats
    raw_vol = float(graph_data.x[node_id, 0].item())
    in_deg = int(graph_data.x[node_id, 165].item() * 100) if graph_data.x.shape[1] > 165 else 0
    out_deg = int(graph_data.x[node_id, 166].item() * 100) if graph_data.x.shape[1] > 166 else 0
    clust = float(graph_data.x[node_id, 167].item()) if graph_data.x.shape[1] > 167 else 0.0

    reasoning = generate_forensic_reasoning(
        fraud_risk=gt_fraud_risk,
        in_degree=in_deg,
        out_degree=out_deg,
        clustering_coef=clust,
        volume_proxy=raw_vol,
        neighbor_risk="mixed" if gt_fraud_risk >= 0.4 else "low",
    )

    return {
        "node_id": node_id,
        "verdict": verdict,
        "badge_type": badge_type,
        "is_fraud": gt_fraud_risk >= threshold,
        "fraud_risk": gt_fraud_risk,
        "clean_prob": 1.0 - gt_fraud_risk,
        "baseline_fraud_risk": baseline_fraud_risk,
        "confidence": max(gt_fraud_risk, 1.0 - gt_fraud_risk),
        "ground_truth": ground_truth_str,
        "ground_truth_code": gt_label,
        "threshold": threshold,
        "subgraph_nodes": len(subset),
        "subgraph_edges": sub_ei.shape[1],
        "reasoning": reasoning,
        "volume_proxy": raw_vol,
        "in_degree": in_deg,
        "out_degree": out_deg,
        "clustering_coef": clust,
    }


def generate_forensic_reasoning(
    fraud_risk: float,
    in_degree: int,
    out_degree: int,
    clustering_coef: float,
    volume_proxy: float,
    neighbor_risk: str,
) -> str:
    """Synthesizes human-auditable AML narrative based on model findings."""
    deductions = []

    if fraud_risk >= 0.70:
        deductions.append(f"Model identified severe illicit indicators with a fraud probability of {fraud_risk:.1%}.")
        if out_degree >= 25:
            deductions.append(f"Extensive fan-out dispersion ({out_degree} outbound outputs) strongly mirrors a rapid peeling chain laundering scheme.")
        if clustering_coef >= 0.35:
            deductions.append(f"High local clustering coefficient ({clustering_coef:.2f}) indicates cyclic mixer pooling behavior.")
        if neighbor_risk == "high":
            deductions.append("The 2-hop ego-neighborhood is predominantly composed of known illicit or high-risk counterparties.")
        if volume_proxy > 2.0:
            deductions.append(f"Abnormally high transaction volume ({volume_proxy:+.2f} σ above mean) amplifies financial severity.")
        action = "Recommended Action: Immediate compliance freeze & Suspicious Activity Report (SAR/STR) filing."
    elif fraud_risk >= 0.40:
        deductions.append(f"Elevated risk indicators detected with a fraud probability of {fraud_risk:.1%}.")
        deductions.append("Transaction exhibits borderline topological properties requiring manual forensic investigation.")
        if out_degree > 10 or in_degree > 15:
            deductions.append(f"Disproportionate degree connectivity (In: {in_degree}, Out: {out_degree}) deviates from typical retail patterns.")
        action = "Recommended Action: Route to Level 2 AML analyst for Enhanced Due Diligence (EDD)."
    else:
        deductions.append(f"Low risk indicators verified with clean confidence of {(1.0 - fraud_risk):.1%}.")
        deductions.append(f"Transaction topology (In: {in_degree}, Out: {out_degree}, Clustering: {clustering_coef:.2f}) aligns with standard licit blockchain behavior.")
        action = "Recommended Action: Approved for automated settlement clearance."

    return " ".join(deductions) + " " + action
