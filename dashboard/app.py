"""Stage 8 – Full XAI-GT Fraud Detection Dashboard.

Integrates:
- Wallet search & sample selector
- Model risk scoring & gauge indicator
- Interactive ego-network visualizer (Stages 7a–7e) with click-to-explore
- GNNExplainer explainability overlay & plain-language reasoning
- Feature attribution breakdown with Plotly
- Immediate neighborhood transaction inspector

Usage:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch

import config as cfg
from src.explain.explainer import FraudExplainer
from src.models.graph_transformer import load_trained_graph_transformer
from src.preprocessing.graph_builder import load_graph
from src.visualize.network_view import (
    build_nx_graph,
    get_ego_network_with_scores,
    render_graph,
    score_nodes,
)
from src.visualize.network_3d import (
    layout_graph_3d,
    render_graph_3d,
)
from src.visualize.embedding_viz import build_embedding_figure

# ---------------------------------------------------------------------------
# Streamlit Page Config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="XAI-GT | Blockchain Fraud Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom Styling (Dark Glassmorphic UI)
# ---------------------------------------------------------------------------
st.markdown(
    """
<style>
    /* Global styling */
    .stApp {
        background-color: #0B0F19;
        color: #F1F5F9;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Header card */
    .investigation-header {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.5);
    }
    
    /* Badge styling */
    .status-badge {
        display: inline-block;
        padding: 0.35rem 0.85rem;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.85rem;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-right: 0.5rem;
    }
    .badge-fraud {
        background: rgba(239, 68, 68, 0.2);
        color: #EF4444;
        border: 1px solid rgba(239, 68, 68, 0.4);
    }
    .badge-suspicious {
        background: rgba(234, 179, 8, 0.2);
        color: #EAB308;
        border: 1px solid rgba(234, 179, 8, 0.4);
    }
    .badge-clean {
        background: rgba(34, 197, 94, 0.2);
        color: #22C55E;
        border: 1px solid rgba(34, 197, 94, 0.4);
    }
    .badge-unknown {
        background: rgba(148, 163, 184, 0.2);
        color: #94A3B8;
        border: 1px solid rgba(148, 163, 184, 0.4);
    }

    /* Explanation callout */
    .explanation-box {
        background: rgba(30, 41, 59, 0.5);
        border-left: 4px solid #F97316;
        padding: 1rem 1.25rem;
        border-radius: 0 8px 8px 0;
        margin-top: 0.5rem;
        margin-bottom: 1rem;
        font-size: 0.95rem;
        line-height: 1.5;
    }
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Cached Resource Loaders
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading PyG graph dataset …")
def get_graph_data():
    """Loads the cached PyTorch Geometric Data object."""
    return load_graph()


@st.cache_resource(show_spinner="Building NetworkX graph index …")
def get_nx_graph(_data):
    """Builds the memory-efficient NetworkX DiGraph for ego queries."""
    return build_nx_graph(_data)


@st.cache_resource(show_spinner="Loading Graph Transformer checkpoint …")
def get_model():
    """Loads the trained Graph Transformer neural network."""
    try:
        model, meta = load_trained_graph_transformer(device=torch.device("cpu"))
        return model, meta
    except Exception as exc:
        return None, str(exc)


@st.cache_resource(show_spinner="Initializing GNNExplainer engine …")
def get_explainer_engine(_data, _model):
    """Initializes the explainability engine."""
    if _model is None:
        return None
    try:
        return FraudExplainer(model=_model, graph_data=_data, epochs=50)
    except Exception:
        return None


# Load foundational assets
data = get_graph_data()
if data is None:
    st.error("⚠️ Graph cache not found. Please run `python src/preprocessing/graph_builder.py` first.")
    st.stop()

G = get_nx_graph(data)
model, model_meta = get_model()
explainer = get_explainer_engine(data, model)

# ---------------------------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------------------------
if "wallet_id" not in st.session_state:
    # Pick first known illicit wallet by default if available
    illicit_indices = torch.where(data.y == 1)[0].tolist()
    st.session_state["wallet_id"] = illicit_indices[0] if illicit_indices else 0

if "hops" not in st.session_state:
    st.session_state["hops"] = 2

if "show_explanation" not in st.session_state:
    st.session_state["show_explanation"] = True

if "graph_view_mode" not in st.session_state:
    st.session_state["graph_view_mode"] = "2D"


# ---------------------------------------------------------------------------
# Sidebar: Navigation & Controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🛡️ **XAI-GT** Intelligence")
    st.caption("Graph Transformer & GNNExplainer for Bitcoin Fraud")
    st.markdown("---")

    st.markdown("### 🔍 **Target Investigation**")

    # Direct wallet ID search input
    selected_wallet = st.number_input(
        "Transaction / Wallet ID",
        min_value=0,
        max_value=int(data.num_nodes) - 1,
        value=int(st.session_state["wallet_id"]),
        step=1,
        help="Enter any transaction index in the Elliptic graph (0 to 203,768).",
    )
    if selected_wallet != st.session_state["wallet_id"]:
        st.session_state["wallet_id"] = int(selected_wallet)
        st.rerun()

    # Quick picker buttons
    st.caption("Quick Samples:")
    q_col1, q_col2, q_col3 = st.columns(3)
    with q_col1:
        if st.button("🔴 Illicit", use_container_width=True, help="Random verified illicit transaction"):
            illicit_pool = torch.where(data.y == 1)[0].tolist()
            st.session_state["wallet_id"] = random.choice(illicit_pool)
            st.rerun()
    with q_col2:
        if st.button("🟢 Licit", use_container_width=True, help="Random verified licit transaction"):
            licit_pool = torch.where(data.y == 0)[0].tolist()
            st.session_state["wallet_id"] = random.choice(licit_pool)
            st.rerun()
    with q_col3:
        if st.button("⚪ Unknown", use_container_width=True, help="Random unlabelled transaction"):
            unk_pool = torch.where(data.y == -1)[0].tolist()
            st.session_state["wallet_id"] = random.choice(unk_pool)
            st.rerun()

    st.markdown("---")
    st.markdown("### 🕸️ **Network Controls**")
    hops = st.slider("Ego Hop Radius", 1, 3, int(st.session_state["hops"]))
    st.session_state["hops"] = hops

    max_nodes = st.slider("Max Display Nodes", 20, 300, 120, 10, help="Cap node count to maintain smooth 60fps graph physics.")
    physics = st.toggle("Canvas Physics", value=True, help="Enable live force-directed layout.")
    canvas_height = st.slider("Canvas Height", 450, 850, 600, 50)

    st.markdown("---")
    st.markdown("### 🔬 **Explainability**")
    show_explanation = st.toggle(
        "GNNExplainer Overlay",
        value=st.session_state["show_explanation"],
        help="Overlay the explanation subgraph in orange and synthesize plain-language reasoning.",
    )
    st.session_state["show_explanation"] = show_explanation

    top_k_feat = st.slider("Top Attributed Features", 3, 10, 5)

    st.markdown("---")
    st.markdown("### 🎨 **Visual Legend**")
    legend_items = [
        ("#A855F7", "Investigated Wallet (Focal Target)"),
        ("#F97316", "GNNExplainer Attribution Evidence"),
        ("#EF4444", "High Risk / Illicit Transaction"),
        ("#EAB308", "Medium / Suspicious Risk"),
        ("#22C55E", "Low Risk / Clean Transaction"),
        ("#94A3B8", "Unlabelled / Unknown Label"),
    ]
    for hex_c, desc in legend_items:
        st.markdown(
            f"<div style='margin-bottom:4px;'>"
            f"<span style='display:inline-block;width:12px;height:12px;border-radius:50%;background:{hex_c};margin-right:8px;vertical-align:middle;'></span>"
            f"<span style='font-size:0.85rem;color:#CBD5E1;'>{desc}</span></div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    if isinstance(model_meta, dict):
        st.caption(
            f"**Model:** Graph Transformer (2x TransformerConv)\n\n"
            f"**Val F1:** {model_meta.get('val_f1', 0.88):.4f} | **Val AUC:** {model_meta.get('val_auc', 0.92):.4f}"
        )


# ---------------------------------------------------------------------------
# Investigation Target Details
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Key Metrics Row with Plotly Gauge
# ---------------------------------------------------------------------------
col_gauge, col_m1, col_m2, col_m3 = st.columns([1.5, 1, 1, 1])

with col_gauge:
    # Plotly Risk Score Gauge
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
    st.plotly_chart(gauge_fig, width='stretch')

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


# ---------------------------------------------------------------------------
# Explainability Computation (Stage 7e + Stage 6)
# ---------------------------------------------------------------------------
explanation_res = None
highlight_nodes: set[int] = set()
highlight_edges: set[tuple[int, int]] = set()

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


# ---------------------------------------------------------------------------
# Main Network Visualization (Stages 7a–7e + Stage 10: 3D)
# ---------------------------------------------------------------------------
st.markdown("### 🕸️ **Interactive Transaction Graph**")

tab_2d, tab_3d, tab_emb = st.tabs([
    "🗺️  2D Force-Directed",
    "🌐  3D WebGL",
    "🔮  Embedding Space",
])

# ---- 2D tab (Stage 7a–7e, unchanged) ------------------------------------
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
    # Handle Click-to-Explore (Stage 7d)
    if clicked_node is not None:
        try:
            clicked_id = int(clicked_node)
            if clicked_id != curr_wallet:
                st.session_state["wallet_id"] = clicked_id
                st.rerun()
        except (ValueError, TypeError):
            pass

# ---- 3D tab (Stage 10) --------------------------------------------------
with tab_3d:
    st.caption(
        "WebGL 3D ego-network — **rotate** (drag), **zoom** (scroll), **pan** (right-drag). "
        "Purple = focal wallet · Orange = explanation evidence · Green→Red = risk gradient."
    )

    # Cache 3D spring layout keyed by wallet + hops + node-count to avoid
    # re-running Fruchterman-Reingold on every Streamlit rerun.
    @st.cache_data(show_spinner="Computing 3D spring layout …")
    def _cached_3d_layout(wallet_id: int, hops: int, node_count: int):
        """Thin wrapper so st.cache_data can key on primitive args."""
        return layout_graph_3d(node_list, edge_list, seed=42)

    pos_3d = _cached_3d_layout(curr_wallet, hops, len(node_list))

    fig_3d = render_graph_3d(
        node_list=node_list,
        edge_list=edge_list,
        highlight_nodes=highlight_nodes,
        highlight_edges=highlight_edges,
        target_id=curr_wallet,
        height=canvas_height,
        cached_pos=pos_3d,
    )
    st.plotly_chart(fig_3d, width='stretch', key="graph_3d")

    # Click-to-explore fallback: Plotly 3D doesn't return click events to
    # Streamlit natively, so provide a node-ID selectbox for navigation.
    with st.expander("🔎 Navigate to a node (3D Click-to-Explore)", expanded=False):
        visible_ids = sorted([n["id"] for n in node_list])
        nav_idx = st.selectbox(
            "Select node ID to investigate:",
            options=visible_ids,
            index=visible_ids.index(curr_wallet) if curr_wallet in visible_ids else 0,
            key="3d_nav_select",
        )
        if st.button("🚀 Navigate to selected node", key="3d_nav_btn"):
            if nav_idx != curr_wallet:
                st.session_state["wallet_id"] = int(nav_idx)
                st.rerun()

# ---- Embedding Space tab (UMAP / t-SNE) ---------------------------------
with tab_emb:
    st.markdown(
        """
        Latent embeddings extracted from the Graph Transformer's penultimate layer,
        projected to 2D. **Tight, well-separated clusters** are visual proof that the
        model has learned distinct fraud and licit representations.
        """
    )

    emb_c1, emb_c2, emb_c3 = st.columns([1, 1, 1])
    with emb_c1:
        emb_method = st.selectbox(
            "Projection Method",
            options=["umap", "tsne"],
            format_func=lambda x: "UMAP (fast, topology-preserving)" if x == "umap" else "t-SNE (local structure)",
            key="emb_method",
        )
    with emb_c2:
        emb_color = st.selectbox(
            "Colour By",
            options=["label", "risk"],
            format_func=lambda x: "Ground-Truth Label" if x == "label" else "Model Risk Score (continuous)",
            key="emb_color",
        )
    with emb_c3:
        emb_max = st.slider(
            "Max Nodes Sampled",
            min_value=500,
            max_value=5000,
            value=3000,
            step=500,
            key="emb_max",
            help="More nodes = richer plot but slower projection.",
        )

    if model is None:
        st.warning("⚠️ Graph Transformer checkpoint not loaded — cannot extract embeddings.")
    else:
        @st.cache_data(show_spinner=f"Computing {emb_method.upper()} projection …")
        def _cached_embedding(method: str, color_by: str, max_n: int):
            """Cache keyed on method + color + sample size."""
            return build_embedding_figure(
                model=model,
                graph_data=data,
                method=method,
                color_by=color_by,
                max_nodes=max_n,
                highlight_node=curr_wallet,
                device=torch.device("cpu"),
            )

        with st.spinner(f"Running {emb_method.upper()} on {emb_max:,} nodes …"):
            emb_fig = _cached_embedding(emb_method, emb_color, emb_max)

        st.plotly_chart(emb_fig, use_container_width=True, key="emb_plot")

        # ── Stats callout ────────────────────────────────────────────────
        illicit_count = int((data.y == 1).sum().item())
        licit_count   = int((data.y == 0).sum().item())
        emb_c_a, emb_c_b = st.columns(2)
        with emb_c_a:
            st.info(
                f"**Dataset breakdown (labelled nodes)**\n\n"
                f"🔴 Illicit: **{illicit_count:,}** ({illicit_count/(illicit_count+licit_count):.1%})  \n"
                f"🟢 Licit:   **{licit_count:,}** ({licit_count/(illicit_count+licit_count):.1%})"
            )
        with emb_c_b:
            st.info(
                "**How to read this plot**\n\n"
                "Well-separated colour clusters → the model has learned **distinct latent "
                "representations** for fraud vs. licit transactions.  \n"
                "Overlap regions indicate ambiguous / borderline cases."
            )


# ---------------------------------------------------------------------------
# Explainability Insights Panel ("Why Flagged?")
# ---------------------------------------------------------------------------
if show_explanation and explanation_res is not None:
    st.markdown("---")
    st.markdown("### 🔍 **Why Flagged? — AI Forensic Reasoning**")

    # Plain language narrative box
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
        st.plotly_chart(fig_feat, width='stretch')

    with exp_col2:
        st.markdown("#### 🔬 **Sub-graph Evidence Metrics**")
        sub_c1, sub_c2 = st.columns(2)
        sub_c1.metric(
            "Fidelity+ (Impact)",
            f"{explanation_res['fidelity_plus']:.4f}",
            help="Drop in fraud score when key explanation edges are severed. Higher indicates higher explanatory faithfulness.",
        )
        sub_c2.metric(
            "Sparsity",
            f"{explanation_res['sparsity']:.1%}",
            help="Ratio of irrelevant edges filtered out to isolate the minimal fraud subgraph.",
        )

        st.markdown(
            f"""
        - **Explanation Subgraph:** {len(highlight_nodes)} nodes, {len(explanation_res['subgraph_edges'])} edges
        - **Attribution Latency:** `{explanation_res['elapsed_sec']:.2f}` seconds on CPU
        - **Highlight Code:** Identified edges rendered in **#F97316 orange** on the map above.
        """
        )


# ---------------------------------------------------------------------------
# Immediate Neighborhood Table Inspector
# ---------------------------------------------------------------------------
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
    st.dataframe(nb_df, width='stretch', hide_index=True)


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("""
<div style='text-align:center;color:#475569;font-size:0.8rem;padding-top:2.5rem;padding-bottom:1rem;'>
    XAI-GT · Explainable AI for Blockchain Fraud Detection · Stages 8 &amp; 10 — 2D + 3D + Embedding Space
</div>
""", unsafe_allow_html=True)
