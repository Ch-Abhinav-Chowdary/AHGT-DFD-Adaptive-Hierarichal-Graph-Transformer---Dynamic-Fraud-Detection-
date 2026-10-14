"""Stage 8 & 10 – Full XAI-GT Fraud Intelligence Dashboard.

Integrates:
- ⚡ Live Transaction Fraud Predictor & Scenario Simulator (New Feature)
- 🔍 Forensic Network Explorer (2D agraph + 3D WebGL + GNNExplainer)
- 📂 Batch Transaction Screener & Compliance Audit Log
- 📊 Model Arena & Comparative Benchmarks

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

# ---------------------------------------------------------------------------
# Cloud startup: auto-download data from Kaggle if not present locally
# ---------------------------------------------------------------------------
import download_data as _dd
_dd.ensure_data_ready(streamlit_ui=True)


import networkx as nx
import streamlit as st
import torch

import config as cfg
from src.explain.explainer import FraudExplainer
try:
    from src.models.baseline import load_trained_baseline
except ImportError:
    import importlib
    import src.models.baseline
    importlib.reload(src.models.baseline)
    from src.models.baseline import load_trained_baseline
from src.models.graph_transformer import load_trained_graph_transformer
from src.preprocessing.graph_builder import load_graph
from src.visualize.network_view import build_nx_graph

import importlib

import src.visualize.network_3d
importlib.reload(src.visualize.network_3d)

import dashboard.explorer_view
importlib.reload(dashboard.explorer_view)
from dashboard.explorer_view import render_explorer_view

import dashboard.predictor_view
importlib.reload(dashboard.predictor_view)
from dashboard.predictor_view import render_predictor_view

from dashboard.batch_view import render_batch_view
from dashboard.benchmarks_view import render_benchmarks_view

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


@st.cache_resource(show_spinner="Loading GraphSAGE baseline checkpoint …")
def get_baseline_model():
    """Loads the trained GraphSAGE baseline neural network."""
    try:
        b_model, b_meta = load_trained_baseline(device=torch.device("cpu"))
        return b_model, b_meta
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
baseline_model, baseline_meta = get_baseline_model()
explainer = get_explainer_engine(data, model)

# ---------------------------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------------------------
if "wallet_id" not in st.session_state:
    illicit_indices = torch.where(data.y == 1)[0].tolist()
    st.session_state["wallet_id"] = illicit_indices[0] if illicit_indices else 0

if "hops" not in st.session_state:
    st.session_state["hops"] = 2

if "show_explanation" not in st.session_state:
    st.session_state["show_explanation"] = True

if "graph_view_mode" not in st.session_state:
    st.session_state["graph_view_mode"] = "2D"


# ---------------------------------------------------------------------------
# Sidebar: Portal Navigation
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🛡️ **XAI-GT** Portal")
    st.caption("Explainable Graph AI for Blockchain Fraud Intelligence")
    st.markdown("---")

    nav_portal = st.radio(
        "Navigation Portal",
        [
            "⚡ Live Fraud Predictor",
            "🔍 Forensic Network Explorer",
            "📂 Batch Transaction Screener",
            "📊 Model Arena & Benchmarks",
        ],
        index=0,
    )
    st.markdown("---")

    # Dynamic Sidebar controls depending on view
    if nav_portal == "🔍 Forensic Network Explorer":
        st.markdown("### 🔍 **Target Investigation**")

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
            f"**Graph Transformer:** Val F1: {model_meta.get('val_f1', 0.88):.4f} · AUC: {model_meta.get('val_auc', 0.92):.4f}\n\n"
            f"**Baseline GraphSAGE:** Val F1: {baseline_meta.get('val_f1', 0.65) if isinstance(baseline_meta, dict) else 0.65:.4f}"
        )


# ---------------------------------------------------------------------------
# Multi-Page Portal Routing
# ---------------------------------------------------------------------------
if nav_portal == "⚡ Live Fraud Predictor":
    render_predictor_view(data, model, baseline_model, G)

elif nav_portal == "🔍 Forensic Network Explorer":
    render_explorer_view(
        data=data,
        G=G,
        model=model,
        explainer=explainer,
        hops=st.session_state.get("hops", 2),
        max_nodes=max_nodes if "max_nodes" in locals() else 120,
        physics=physics if "physics" in locals() else True,
        canvas_height=canvas_height if "canvas_height" in locals() else 600,
        show_explanation=st.session_state.get("show_explanation", True),
        top_k_feat=top_k_feat if "top_k_feat" in locals() else 5,
    )

elif nav_portal == "📂 Batch Transaction Screener":
    render_batch_view(data, model, baseline_model)

elif nav_portal == "📊 Model Arena & Benchmarks":
    render_benchmarks_view(model=model, data=data, metadata=model_meta if isinstance(model_meta, dict) else None)


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("""
<div style='text-align:center;color:#475569;font-size:0.8rem;padding-top:2.5rem;padding-bottom:1rem;'>
    XAI-GT · Explainable AI for Blockchain Fraud Detection · Live Inference &amp; Forensic Intelligence Suite
</div>
""", unsafe_allow_html=True)
