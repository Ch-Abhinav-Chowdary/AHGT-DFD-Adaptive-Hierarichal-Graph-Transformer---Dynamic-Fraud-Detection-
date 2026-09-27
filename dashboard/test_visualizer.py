"""Stage 7a/7b/7c/7d/7e – Graph visualiser test page.

Usage
-----
    streamlit run dashboard/test_visualizer.py

Stage 7d: clicking any visible node re-centers the ego-network on that
wallet within the same Streamlit session (no full page reload).
Stage 7e: toggling 'Show Explanation' runs GNNExplainer and overlays the
explanation subgraph in orange on top of the ego-network.
"""

import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import streamlit as st

import config as cfg
from src.explain.explainer import FraudExplainer
from src.models.graph_transformer import load_trained_graph_transformer
from src.visualize.network_view import (
    build_nx_graph,
    get_ego_network,
    get_ego_network_with_scores,
    render_graph,
    score_nodes,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="XAI-GT | Graph Visualiser",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%); color: #E2E8F0; }
[data-testid="stSidebar"] { background: #1E293B; border-right: 1px solid #334155; }
[data-testid="metric-container"] {
    background: #1E293B; border: 1px solid #334155;
    border-radius: 12px; padding: 1rem 1.2rem;
}
[data-testid="stMetricLabel"] { color: #94A3B8; font-size: 0.78rem; }
[data-testid="stMetricValue"] { color: #F1F5F9; font-weight: 600; }
hr { border-color: #334155; }
.stCaption, small { color: #64748B; }
h1, h2, h3 { color: #F1F5F9; }
.stCheckbox label, .stSlider label, .stRadio label,
.stNumberInput label, .stToggle label { color: #CBD5E1; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
if "active_wallet" not in st.session_state:
    st.session_state["active_wallet"] = 0
if "mode" not in st.session_state:
    st.session_state["mode"] = "🎲 Random overview"

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown("""
<div style="padding:1.5rem 0 0.5rem 0;">
  <h1 style="font-size:1.8rem;font-weight:700;margin:0;">
    🕸️ XAI-GT · Transaction Graph Visualiser
  </h1>
  <p style="color:#64748B;margin-top:0.3rem;font-size:0.92rem;">
    Stages 7a · 7b · 7c · 7d &nbsp;|&nbsp; Elliptic Bitcoin Dataset
  </p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading graph cache …")
def load_graph():
    p = cfg.GRAPH_CACHE_FILE
    if not p.exists():
        return None
    return torch.load(p, weights_only=False, map_location="cpu")

@st.cache_resource(show_spinner="Building NetworkX topology …")
def load_nx(_data):
    return build_nx_graph(_data)

@st.cache_resource(show_spinner="Loading Graph Transformer …")
def load_model():
    try:
        model, meta = load_trained_graph_transformer(device=torch.device("cpu"))
        return model, meta
    except Exception as e:
        return None, str(e)

@st.cache_resource(show_spinner="Loading GNNExplainer …")
def load_explainer(_data):
    """Singleton FraudExplainer – reuses loaded model + graph."""
    try:
        return FraudExplainer(epochs=60)
    except Exception as e:
        return None

data = load_graph()
if data is None:
    st.error("⚠️ Graph cache not found. Run `python src/preprocessing/graph_builder.py` first.")
    st.stop()

G                 = load_nx(data)
model, model_meta = load_model()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Mode")
    mode = st.radio(
        "Display mode",
        ["🎲 Random overview", "🔎 Ego-network (wallet ID)"],
        index=["🎲 Random overview", "🔎 Ego-network (wallet ID)"].index(
            st.session_state["mode"]
        ),
    )
    st.session_state["mode"] = mode

    st.markdown("---")
    st.markdown("### 🛠️ Options")

    if mode == "🔎 Ego-network (wallet ID)":
        # Manual wallet input — stays in sync with session state
        manual_id = st.number_input(
            "Wallet / Node ID",
            min_value=0,
            max_value=int(data.num_nodes) - 1,
            value=int(st.session_state["active_wallet"]),
            step=1,
            key="wallet_input",
        )
        # Manual input always wins over session state when user types
        st.session_state["active_wallet"] = int(manual_id)

        hops = st.slider("Hop radius", 1, 3, 2)

        if st.button("🎯 Pick random illicit wallet"):
            illicit = torch.where(data.y == 1)[0].tolist()
            st.session_state["active_wallet"] = random.choice(illicit)
            st.rerun()

        st.markdown("---")
        use_risk = st.toggle(
            "🎨 Risk & Volume encoding (Stage 7c)",
            value=(model is not None),
            disabled=(model is None),
        )
        if model is None:
            st.caption(f"⚠️ Model unavailable: {model_meta}")

        # Stage 7e: explanation overlay
        use_explain = st.toggle(
            "🔬 Show Explanation overlay (Stage 7e)",
            value=False,
            help="Runs GNNExplainer and highlights the explanation subgraph in orange.",
        )

        max_ego = st.slider("Max ego nodes", 20, cfg.MAX_VIS_NODES, 150, 10)

    else:
        hops     = 2
        use_risk = False
        max_rand = st.slider("Max random nodes", 20, cfg.MAX_VIS_NODES, 100, 10)
        rand_seed = st.number_input("Random seed", value=42, min_value=0, step=1)

    physics_on = st.toggle("Physics simulation", value=True)
    canvas_h   = st.slider("Canvas height (px)", 400, 900, 620, 50)

    st.markdown("---")
    st.markdown("### 🎨 Legend")
    for colour, label in [
        ("#EF4444", "Illicit / High risk"),
        ("#EAB308", "Moderate risk"),
        ("#22C55E", "Licit / Low risk"),
        ("#94A3B8", "Unknown"),
        ("#A855F7", "Focal wallet"),
        ("#F97316", "Explanation (Stage 7e)"),
    ]:
        st.markdown(
            f"<span style='display:inline-block;width:12px;height:12px;"
            f"background:{colour};border-radius:50%;vertical-align:middle;"
            f"margin-right:6px'></span>{label}",
            unsafe_allow_html=True,
        )
    st.markdown("---")
    st.caption("Stages 7a · 7b · 7c · 7d")

# ---------------------------------------------------------------------------
# Dataset stats
# ---------------------------------------------------------------------------
st.markdown("#### 📊 Dataset")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Nodes", f"{data.num_nodes:,}")
c2.metric("Total Edges", f"{data.edge_index.shape[1]:,}")
c3.metric("Illicit",     f"{int((data.y==1).sum()):,}")
c4.metric("Licit",       f"{int((data.y==0).sum()):,}")
c5.metric("Unknown",     f"{int((data.y==-1).sum()):,}")
st.markdown("---")

# ---------------------------------------------------------------------------
# Model metadata strip (Stage 7c)
# ---------------------------------------------------------------------------
if model is not None and mode == "🔎 Ego-network (wallet ID)" and use_risk:
    st.markdown("#### 🤖 Model")
    m1, m2, m3 = st.columns(3)
    m1.metric("Val F1",    f"{model_meta.get('val_f1', 0):.4f}")
    m2.metric("Val AUC",   f"{model_meta.get('val_auc', 0):.4f}")
    m3.metric("Threshold", f"{model_meta.get('best_threshold', 0.5):.2f}")
    st.markdown("---")

# ---------------------------------------------------------------------------
# Graph display
# ---------------------------------------------------------------------------
if mode == "🔎 Ego-network (wallet ID)":
    wallet_id = int(st.session_state["active_wallet"])
    st.markdown(f"#### 🔎 Ego-Network — Wallet `{wallet_id}` · {hops}-hop")

    try:
        if use_risk and model is not None:
            node_list, edge_list = get_ego_network_with_scores(
                wallet_id=wallet_id,
                G=G,
                graph_data=data,
                model=model,
                hops=hops,
                max_nodes=max_ego,
                device=torch.device("cpu"),
            )
            encoding_note = "Colour = risk score (green→red) · Size = transaction volume"
        else:
            node_list, edge_list = get_ego_network(
                wallet_id=wallet_id,
                G=G,
                graph_data=data,
                hops=hops,
                max_nodes=max_ego,
            )
            encoding_note = "Colour = ground-truth label · Enable Risk & Volume for model scores"

        wallet_label = {1: "🔴 ILLICIT", 0: "🟢 LICIT", -1: "⚪ UNKNOWN"}.get(
            int(data.y[wallet_id].item()), "?"
        )
        ic1, ic2, ic3 = st.columns(3)
        ic1.metric("Focal Wallet", str(wallet_id))
        ic2.metric("Label",        wallet_label)
        ic3.metric("Ego Nodes",    f"{len(node_list):,}")
        st.caption(f"{encoding_note}. Purple = focal wallet. **Click any node to re-center.**")

        # ── Stage 7e: explanation overlay ───────────────────────────────────
        highlight_nodes: set[int] = set()
        highlight_edges: set[tuple[int, int]] = set()
        if use_explain:
            explainer = load_explainer(data)
            if explainer is not None:
                with st.spinner("Running GNNExplainer …"):
                    try:
                        exp = explainer.explain_node(wallet_id)
                        highlight_nodes = set(exp["subgraph_nodes"])
                        highlight_edges = {
                            (int(e["src"]), int(e["dst"])) for e in exp["subgraph_edges"]
                        }

                        # Ensure all explanation nodes are present in node_list so they are visible
                        existing_node_ids = {n["id"] for n in node_list}
                        missing_nodes = [nid for nid in highlight_nodes if nid not in existing_node_ids]
                        if missing_nodes:
                            for nid in missing_nodes:
                                lbl = int(data.y[nid].item()) if nid < len(data.y) else -1
                                node_list.append({"id": nid, "label": lbl})
                            if use_risk and model is not None:
                                extra_scores = score_nodes(missing_nodes, data, model, device=torch.device("cpu"))
                                for n in node_list:
                                    if n["id"] in extra_scores:
                                        n["risk_score"] = extra_scores[n["id"]].get("risk_score")
                                        n["volume"] = extra_scores[n["id"]].get("volume")

                        # Ensure all explanation edges are present in edge_list
                        existing_edges = {(e["src"], e["dst"]) for e in edge_list}
                        for e in exp["subgraph_edges"]:
                            s, d = int(e["src"]), int(e["dst"])
                            if (s, d) not in existing_edges and (d, s) not in existing_edges:
                                edge_list.append({"src": s, "dst": d, "weight": e.get("weight", 0.9)})

                        # Show explanation panel
                        with st.expander("🔍 Why flagged? — Explanation details", expanded=True):
                            ec1, ec2, ec3 = st.columns(3)
                            ec1.metric("Risk Score",  f"{exp['risk_score']:.1%}")
                            ec2.metric("Fidelity+",   f"{exp['fidelity_plus']:.4f}")
                            ec3.metric("Sparsity",    f"{exp['sparsity']:.4f}")
                            st.caption(exp["summary"])
                            st.markdown("**Top contributing features:**")
                            for feat in exp["top_features"]:
                                st.markdown(
                                    f"- `{feat['name']}` — importance: **{feat['importance']:.4f}**"
                                )
                            st.caption(
                                f"Explanation subgraph: {len(highlight_nodes)} nodes "
                                f"(highlighted orange) · {len(exp['subgraph_edges'])} edges "
                                f"· computed in {exp['elapsed_sec']:.2f}s"
                            )
                    except Exception as exc:
                        st.warning(f"Explainer error: {exc}")
            else:
                st.warning("Explainer could not be loaded (model checkpoint required).")

        # ── Stage 7d: capture click → update session state → rerun ──────────
        clicked = render_graph(
            node_list=node_list,
            edge_list=edge_list,
            target_id=wallet_id,
            highlight_nodes=highlight_nodes,
            highlight_edges=highlight_edges,
            max_nodes=max_ego,
            height=canvas_h,
            physics=physics_on,
            show_legend=False,
        )

        if clicked is not None:
            clicked_id = int(clicked)
            if clicked_id != wallet_id:
                # Re-center: update session state and trigger rerun
                st.session_state["active_wallet"] = clicked_id
                st.rerun()
            else:
                # Clicked the focal node itself — show info
                r = next(
                    (n.get("risk_score") for n in node_list if n["id"] == clicked_id),
                    None,
                )
                risk_str = f" · Risk: **{r:.1%}**" if r is not None else ""
                lbl_str  = {1: "ILLICIT", 0: "LICIT", -1: "unknown"}.get(
                    int(data.y[clicked_id].item()), "?"
                )
                st.info(f"🖱️ Focal node **{clicked_id}** · Label: **{lbl_str}**{risk_str}")

    except ValueError as exc:
        st.error(str(exc))

else:
    # ── Random overview (Stage 7a) ───────────────────────────────────────────
    st.markdown("#### 🎲 Random Graph Overview")
    y  = data.y.cpu().numpy()
    ei = data.edge_index.cpu().numpy()

    random.seed(int(rand_seed))
    sampled_ids = random.sample(list(range(data.num_nodes)), min(max_rand, data.num_nodes))
    sampled_set = set(sampled_ids)

    node_list = [{"id": int(i), "label": int(y[i])} for i in sampled_ids]
    edge_list = [
        {"src": int(ei[0, j]), "dst": int(ei[1, j])}
        for j in range(ei.shape[1])
        if int(ei[0, j]) in sampled_set and int(ei[1, j]) in sampled_set
    ]

    st.caption(
        f"Random sample of **{len(node_list)}** nodes · **{len(edge_list)}** edges. "
        "Click a node then switch to Ego-network mode to explore it."
    )

    clicked = render_graph(
        node_list=node_list,
        edge_list=edge_list,
        max_nodes=max_rand,
        height=canvas_h,
        physics=physics_on,
        show_legend=False,
    )

    # Stage 7d groundwork: clicking in overview pre-loads wallet into state
    if clicked is not None:
        st.session_state["active_wallet"] = int(clicked)
        st.info(
            f"🖱️ Node **{clicked}** selected. "
            "Switch to **Ego-network mode** in the sidebar to explore its neighbourhood."
        )

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("""
<div style='text-align:center;color:#334155;font-size:0.78rem;padding-top:2rem;'>
    XAI-GT · Blockchain Fraud Detection · Stages 7a / 7b / 7c / 7d / 7e
</div>
""", unsafe_allow_html=True)
