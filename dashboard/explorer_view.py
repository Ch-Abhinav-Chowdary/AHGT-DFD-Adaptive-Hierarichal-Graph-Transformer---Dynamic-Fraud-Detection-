"""Forensic Network Explorer View Component for XAI-GT Dashboard.

Provides interactive 2D (agraph) + 3D (Plotly WebGL) ego-network visualizer,
GNNExplainer attribution overlay, and neighborhood transaction inspector.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Set, Tuple
import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch

from src.visualize.network_view import (
    get_ego_network_with_scores,
    render_graph,
    score_nodes,
)
from src.visualize.network_3d import (
    layout_graph_3d,
    render_graph_3d,
)
from src.visualize.embedding_viz import build_embedding_figure


def render_explorer_view(
    data: Any,
    G: nx.DiGraph,
    model: torch.nn.Module,
    explainer: Any,
    hops: int = 2,
    max_nodes: int = 120,
    physics: bool = True,
    canvas_height: int = 600,
    show_explanation: bool = True,
    top_k_feat: int = 5,
) -> None:
    """Renders the Forensic Ego-Network Explorer."""
    curr_wallet = int(st.session_state["wallet_id"])
    ground_truth_y = int(data.y[curr_wallet].item()) if curr_wallet < len(data.y) else -1

    # Fetch ego-network with model risk scores and normalized volume
    node_list, edge_list = get_ego_network_with_scores(
        wallet_id=curr_wallet,
        G=G,
        graph_data=data,
        model=model,
        hops=hops,
        max_nodes=max_nodes,
        device=torch.device("cpu"),
    )

    # Focal node risk score & volume
    focal_node_dict = next((n for n in node_list if n["id"] == curr_wallet), None)
    risk_score = focal_node_dict.get("risk_score", 0.5) if focal_node_dict else 0.5
    raw_volume_proxy = float(data.x[curr_wallet, 0].item())

    # Status categorization
    if risk_score >= 0.70:
        risk_label = "HIGH RISK / FRAUD"
        badge_cls = "badge-fraud"
    elif risk_score >= 0.35:
        risk_label = "SUSPICIOUS ACTIVITY"
        badge_cls = "badge-suspicious"
    else:
        risk_label = "CLEAN / LOW RISK"
        badge_cls = "badge-clean"

    gt_text = {1: "Verified Illicit", 0: "Verified Licit", -1: "Unlabelled"}.get(ground_truth_y, "Unknown")
    gt_cls = {1: "badge-fraud", 0: "badge-clean", -1: "badge-unknown"}.get(ground_truth_y, "badge-unknown")

    # Top Banner
    st.markdown(
        f"""
    <div class="investigation-header">
        <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:10px;">
            <div>
                <span style="font-size:1.4rem; font-weight:700; letter-spacing:-0.02em;">
                    Transaction #{curr_wallet:,}
                </span>
                <span style="margin-left: 1rem;">
                    <span class="status-badge {badge_cls}">{risk_label}</span>
                    <span class="status-badge {gt_cls}">Ground Truth: {gt_text}</span>
                </span>
            </div>
            <div style="color:#94A3B8; font-size:0.85rem;">
                Ego Neighborhood: <b>{len(node_list)}</b> nodes · <b>{len(edge_list)}</b> transactions
            </div>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Key Metrics Row with Plotly Gauge
    col_gauge, col_m1, col_m2, col_m3 = st.columns([1.5, 1, 1, 1])

    with col_gauge:
        gauge_fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=risk_score * 100,
                number={"suffix": "%", "font": {"size": 28, "color": "#F8FAFC"}},
                title={"text": "Fraud Probability", "font": {"size": 14, "color": "#94A3B8"}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#475569"},
                    "bar": {"color": "#EF4444" if risk_score >= 0.7 else ("#EAB308" if risk_score >= 0.35 else "#22C55E")},
                    "bgcolor": "rgba(30, 41, 59, 0.4)",
                    "borderwidth": 1,
                    "bordercolor": "rgba(255, 255, 255, 0.1)",
                    "steps": [
                        {"range": [0, 35], "color": "rgba(34, 197, 94, 0.15)"},
                        {"range": [35, 70], "color": "rgba(234, 179, 8, 0.15)"},
                        {"range": [70, 100], "color": "rgba(239, 68, 68, 0.15)"},
                    ],
                },
            )
        )
        gauge_fig.update_layout(
            height=180,
            margin={"l": 20, "r": 20, "t": 40, "b": 10},
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(gauge_fig, use_container_width=True)

    with col_m1:
        in_deg = int(G.in_degree(curr_wallet)) if curr_wallet in G else 0
        st.metric("Inflow Transactions", f"{in_deg} in", help="Direct incoming payments into this transaction.")
        st.metric("Model Confidence", f"{max(risk_score, 1 - risk_score):.1%}")

    with col_m2:
        out_deg = int(G.out_degree(curr_wallet)) if curr_wallet in G else 0
        st.metric("Outflow Transactions", f"{out_deg} out", help="Dispersion / peeling chains exiting this transaction.")
        st.metric("Degree Centrality", f"{in_deg + out_deg}")

    with col_m3:
        st.metric("Volume Proxy (Std)", f"{raw_volume_proxy:.2f} σ", help="Normalized Bitcoin fee/output volume feature #1.")
        sub_illicit = sum(1 for n in node_list if n.get("risk_score", 0) >= 0.70)
        st.metric("Ego Fraud Neighbours", f"{sub_illicit} flagged")

    # Explainability Computation
    explanation_res = None
    highlight_nodes: Set[int] = set()
    highlight_edges: Set[Tuple[int, int]] = set()

    if not show_explanation:
        st.info("💡 **GNNExplainer Overlay is OFF** — toggle it on in the sidebar under **🔬 Explainability** to see forensic attribution evidence.")

    elif explainer is None:
        st.warning(
            "⚠️ **Explainability engine could not be initialized.** "
            "This usually means the Graph Transformer checkpoint (`models/graph_transformer.pt`) "
            "failed to load. Check the sidebar model info section."
        )

    else:
        with st.spinner("🧠 GNNExplainer evaluating neighborhood topology & feature masks …"):
            try:
                explanation_res = explainer.explain_node(curr_wallet, top_k_features=top_k_feat)
                highlight_nodes = set(explanation_res["subgraph_nodes"])
                highlight_edges = {(int(e["src"]), int(e["dst"])) for e in explanation_res["subgraph_edges"]}

                # Ensure all explanation nodes are present in node_list
                existing_ids = {n["id"] for n in node_list}
                missing_nodes = [nid for nid in highlight_nodes if nid not in existing_ids]
                if missing_nodes:
                    for nid in missing_nodes:
                        lbl = int(data.y[nid].item()) if nid < len(data.y) else -1
                        node_list.append({"id": nid, "label": lbl})
                    if model is not None:
                        extra_scores = score_nodes(missing_nodes, data, model, device=torch.device("cpu"))
                        for n in node_list:
                            if n["id"] in extra_scores:
                                n["risk_score"] = extra_scores[n["id"]].get("risk_score")
                                n["volume"] = extra_scores[n["id"]].get("volume")

                # Ensure all explanation edges are present in edge_list
                existing_edges = {(e["src"], e["dst"]) for e in edge_list}
                for e in explanation_res["subgraph_edges"]:
                    s, d = int(e["src"]), int(e["dst"])
                    if (s, d) not in existing_edges and (d, s) not in existing_edges:
                        edge_list.append({"src": s, "dst": d, "weight": e.get("weight", 0.9)})

            except Exception as exc:
                import traceback
                st.error(
                    f"**GNNExplainer error** for node `{curr_wallet}`:\n\n"
                    f"```\n{traceback.format_exc()}\n```"
                )

    # Main Network Visualization
    st.markdown("### 🕸️ **Interactive Transaction Graph**")
    tab_2d, tab_3d, tab_emb = st.tabs([
        "🗺️  2D Force-Directed",
        "🌐  3D WebGL",
        "🔮  Embedding Space",
    ])

    with tab_2d:
        st.caption(
            "Purple node is the investigated wallet. Orange nodes & thick edges represent "
            "GNNExplainer evidence. **Click any visible node to navigate its neighbourhood.**"
        )
        clicked_node = render_graph(
            node_list=node_list,
            edge_list=edge_list,
            target_id=curr_wallet,
            highlight_nodes=highlight_nodes,
            highlight_edges=highlight_edges,
            max_nodes=max_nodes,
            height=canvas_height,
            physics=physics,
            show_legend=False,
        )
        if clicked_node is not None:
            try:
                clicked_id = int(clicked_node)
                if clicked_id != curr_wallet:
                    st.session_state["wallet_id"] = clicked_id
                    st.rerun()
            except (ValueError, TypeError):
                pass

    with tab_3d:
        st.caption(
            "Plotly WebGL 3D ego-network. Drag to rotate · Scroll to zoom · Right-click drag to pan. "
            "Orange nodes represent GNNExplainer attribution evidence."
        )
        pos_3d = layout_graph_3d(node_list, edge_list, seed=42)
        fig_3d = render_graph_3d(
            node_list=node_list,
            edge_list=edge_list,
            cached_pos=pos_3d,
            target_id=curr_wallet,
            highlight_nodes=highlight_nodes,
            highlight_edges=highlight_edges,
            height=canvas_height,
        )
        st.plotly_chart(fig_3d, use_container_width=True)

    with tab_emb:
        st.caption(
            "2D projection of the Graph Transformer's penultimate hidden layer across "
            "a stratified sample of ~1,500 transactions (PCA + t-SNE)."
        )
        if model is None:
            st.info("Load the Graph Transformer checkpoint to view embedding space.")
        else:
            fig_emb = build_embedding_figure(
                model=model,
                graph_data=data,
                highlight_node=curr_wallet,
            )
            st.plotly_chart(fig_emb, use_container_width=True)

    # Explainability Insights Panel
    if show_explanation and explanation_res is not None:
        st.markdown("---")
        st.markdown("### 🔍 **Why Flagged? — AI Forensic Reasoning**")
        st.markdown(
            f"""
        <div class="explanation-box">
            <b>Forensic Summary:</b> {explanation_res["summary"]}
        </div>
        """,
            unsafe_allow_html=True,
        )

        exp_col1, exp_col2 = st.columns([1.4, 1])
        with exp_col1:
            st.markdown("#### 📊 **Key Driving Feature Attributions**")
            feats = explanation_res["top_features"]
            feat_df = pd.DataFrame(feats)
            feat_df = feat_df.sort_values(by="importance", ascending=True)

            fig_feat = px.bar(
                feat_df,
                x="importance",
                y="name",
                orientation="h",
                color="importance",
                color_continuous_scale=["#38BDF8", "#F97316", "#EF4444"],
                labels={"importance": "Attribution Weight", "name": "Feature"},
            )
            fig_feat.update_layout(
                height=280,
                margin={"l": 10, "r": 20, "t": 20, "b": 20},
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                coloraxis_showscale=False,
                xaxis={"gridcolor": "rgba(255,255,255,0.08)"},
                yaxis={"gridcolor": "rgba(255,255,255,0.08)"},
                font={"color": "#CBD5E1"},
            )
            st.plotly_chart(fig_feat, use_container_width=True)

        with exp_col2:
            st.markdown("#### 🔬 **Sub-graph Evidence Metrics**")
            sub_c1, sub_c2 = st.columns(2)
            sub_c1.metric("Fidelity+ (Impact)", f"{explanation_res['fidelity_plus']:.4f}")
            sub_c2.metric("Sparsity", f"{explanation_res['sparsity']:.1%}")
            st.markdown(
                f"""
            - **Explanation Subgraph:** {len(highlight_nodes)} nodes, {len(explanation_res['subgraph_edges'])} edges
            - **Attribution Latency:** `{explanation_res['elapsed_sec']:.2f}` seconds on CPU
            - **Highlight Code:** Identified edges rendered in **#F97316 orange** on the map above.
            """
            )

    # Immediate Neighborhood Table Inspector
    st.markdown("---")
    with st.expander(f"📋 **Immediate Neighborhood Transactions ({len(node_list)} nodes)**", expanded=False):
        table_records = []
        for n in node_list:
            nid = n["id"]
            lbl_val = n.get("label", -1)
            lbl_str = {1: "Illicit", 0: "Licit", -1: "Unknown"}.get(lbl_val, "?")
            r_score = n.get("risk_score")
            r_str = f"{r_score:.1%}" if r_score is not None else "N/A"
            vol = n.get("volume")
            vol_str = f"{vol:.2f}" if vol is not None else "N/A"
            is_evidence = "⭐ Evidence" if nid in highlight_nodes else ("🎯 Focal" if nid == curr_wallet else "")

            table_records.append({
                "Wallet / Node ID": nid,
                "Role": is_evidence,
                "Ground Truth": lbl_str,
                "Risk Score": r_str,
                "Normalized Volume": vol_str,
            })

        nb_df = pd.DataFrame(table_records)
        st.dataframe(nb_df, use_container_width=True, hide_index=True)
