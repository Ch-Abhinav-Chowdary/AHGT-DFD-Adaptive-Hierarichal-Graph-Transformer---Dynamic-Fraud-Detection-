"""Stage 7a – Static Transaction-Graph Visualiser (streamlit-agraph).

Public API
----------
render_graph(node_list, edge_list, *, highlight_nodes=None, config_override=None)
    Converts node/edge data into agraph primitives and renders them inside
    the calling Streamlit page.  Samples at most MAX_VIS_NODES nodes to
    prevent layout chaos.

Data contracts
--------------
node_list : list[dict]  -- each dict has at minimum {"id": int}
    Optional keys consumed by this module:
        "label"      : int  -- 0 = licit, 1 = illicit, -1 = unknown
        "risk_score" : float (0–1)
        "volume"     : float (raw transaction volume, used for size scaling)

edge_list : list[dict]  -- each dict has at minimum {"src": int, "dst": int}
    Optional keys:
        "weight" : float -- edge importance / transaction amount (0–1)

highlight_nodes : set[int] | None
    Node IDs to render with a distinct highlight style (explanation overlay
    prepared for Stage 7e).
"""

from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
from streamlit_agraph import Edge, Node, agraph
from streamlit_agraph import Config as AGraphConfig

import config as cfg

# ---------------------------------------------------------------------------
# Visual constants
# ---------------------------------------------------------------------------

# Node colours (hex)
COLOUR_ILLICIT   = "#EF4444"   # vivid red
COLOUR_LICIT     = "#22C55E"   # vivid green
COLOUR_UNKNOWN   = "#94A3B8"   # slate-grey
COLOUR_HIGHLIGHT = "#F97316"   # orange  (explanation overlay – Stage 7e)
COLOUR_TARGET    = "#A855F7"   # purple  (the focal wallet)

# Node size range (pixels)
NODE_SIZE_MIN = 10
NODE_SIZE_MAX = 40
NODE_SIZE_DEFAULT = 18

# Edge width range
EDGE_WIDTH_MIN = 1
EDGE_WIDTH_MAX = 6
EDGE_WIDTH_DEFAULT = 1.5

# Max nodes sampled for a random overview
MAX_DISPLAY_NODES: int = getattr(cfg, "MAX_VIS_NODES", 200)


# ---------------------------------------------------------------------------
# Colour / size helpers
# ---------------------------------------------------------------------------

def _node_colour(node: dict, highlight_nodes: set[int]) -> str:
    """Return hex colour for a node dict."""
    nid = node["id"]
    if nid in highlight_nodes:
        return COLOUR_HIGHLIGHT
    label = node.get("label", -1)
    risk  = node.get("risk_score")
    if risk is not None:
        # Interpolate green→red based on risk score
        return _risk_to_hex(float(risk))
    if label == 1:
        return COLOUR_ILLICIT
    if label == 0:
        return COLOUR_LICIT
    return COLOUR_UNKNOWN


def _risk_to_hex(risk: float) -> str:
    """
    Map a risk score in [0, 1] to a hex colour blending green → amber → red.
    Uses a two-segment linear interpolation:
        0.0 → #22C55E (green)
        0.5 → #EAB308 (amber)
        1.0 → #EF4444 (red)
    """
    risk = max(0.0, min(1.0, risk))
    if risk <= 0.5:
        t = risk / 0.5
        r = int(34  + t * (234 - 34))
        g = int(197 + t * (179 - 197))
        b = int(94  + t * (8   - 94))
    else:
        t = (risk - 0.5) / 0.5
        r = int(234 + t * (239 - 234))
        g = int(179 + t * (68  - 179))
        b = int(8   + t * (68  - 8))
    return f"#{r:02X}{g:02X}{b:02X}"


def _node_size(node: dict) -> int:
    """Scale node size by volume if present, else return default."""
    vol = node.get("volume")
    if vol is None:
        return NODE_SIZE_DEFAULT
    # Caller should pass already-normalised volume in [0, 1]
    vol = max(0.0, min(1.0, float(vol)))
    return int(NODE_SIZE_MIN + vol * (NODE_SIZE_MAX - NODE_SIZE_MIN))


def _edge_width(edge: dict) -> float:
    """Scale edge width by importance weight if present."""
    w = edge.get("weight")
    if w is None:
        return EDGE_WIDTH_DEFAULT
    w = max(0.0, min(1.0, float(w)))
    return EDGE_WIDTH_MIN + w * (EDGE_WIDTH_MAX - EDGE_WIDTH_MIN)


# ---------------------------------------------------------------------------
# agraph object builders
# ---------------------------------------------------------------------------

def _build_agraph_nodes(
    node_list: list[dict],
    highlight_nodes: set[int],
    target_id: int | None,
) -> list[Node]:
    """Convert raw node dicts into agraph Node objects."""
    nodes = []
    for n in node_list:
        nid   = n["id"]
        color = COLOUR_TARGET if nid == target_id else _node_colour(n, highlight_nodes)
        size  = NODE_SIZE_MAX if nid == target_id else _node_size(n)

        # Short label shown on canvas
        label = str(nid)

        # Tooltip shown on hover
        risk  = n.get("risk_score")
        lbl   = n.get("label", -1)
        lbl_str = {1: "ILLICIT", 0: "LICIT", -1: "unknown"}.get(lbl, "?")
        title = f"Node: {nid}\nLabel: {lbl_str}"
        if risk is not None:
            title += f"\nRisk: {risk:.1%}"

        nodes.append(
            Node(
                id=str(nid),
                label=label,
                size=size,
                color=color,
                title=title,
                font={"size": 10, "color": "#E2E8F0"},
            )
        )
    return nodes


def _build_agraph_edges(
    edge_list: list[dict],
    highlight_edges: set[tuple[int, int]] | None = None,
) -> list[Edge]:
    """Convert raw edge dicts into agraph Edge objects."""
    highlight_edges = highlight_edges or set()
    edges = []
    for e in edge_list:
        src, dst = int(e["src"]), int(e["dst"])
        is_highlight = (src, dst) in highlight_edges or (dst, src) in highlight_edges
        width = max(3.5, _edge_width(e) * 1.8) if is_highlight else _edge_width(e)
        color = COLOUR_HIGHLIGHT if is_highlight else "#64748B"
        edges.append(
            Edge(
                source=str(src),
                target=str(dst),
                width=width,
                color=color,
                arrows="to",
            )
        )
    return edges


# ---------------------------------------------------------------------------
# agraph layout config
# ---------------------------------------------------------------------------

def _make_agraph_config(height: int = 600, physics: bool = True) -> AGraphConfig:
    """Build a sensible agraph Config for fraud-graph rendering."""
    return AGraphConfig(
        width="100%",
        height=height,
        directed=True,
        physics=physics,
        hierarchical=False,
        # Barnes-Hut repulsion for cleaner layouts
        solver="barnesHut",
        nodeHighlightBehavior=True,
        highlightColor="#F97316",
        collapsible=False,
        node={"labelProperty": "label"},
        link={"labelProperty": "label", "renderLabel": False},
        # Pass extra vis.js network options via the kwargs below
        **{
            "barnesHut": {
                "gravitationalConstant": -5000,
                "centralGravity": 0.3,
                "springLength": 120,
                "springConstant": 0.04,
                "damping": 0.09,
            }
        },
    )


# ---------------------------------------------------------------------------
# Sampling helper (Stage 7a: random overview when no ego is selected)
# ---------------------------------------------------------------------------

def _sample_nodes(
    node_list: list[dict],
    edge_list: list[dict],
    max_nodes: int,
    seed: int = 42,
    keep_ids: set[int] | None = None,
) -> tuple[list[dict], list[dict]]:
    """
    Randomly sample up to *max_nodes* nodes and retain only edges whose
    both endpoints appear in the sampled set.
    """
    if len(node_list) <= max_nodes:
        return node_list, edge_list

    keep_ids = keep_ids or set()
    random.seed(seed)
    always_keep = [n for n in node_list if n["id"] in keep_ids]
    others = [n for n in node_list if n["id"] not in keep_ids]
    needed = max_nodes - len(always_keep)
    if needed > 0 and len(others) > 0:
        sampled_others = random.sample(others, min(needed, len(others)))
        sampled = always_keep + sampled_others
    else:
        sampled = always_keep[:max_nodes]

    sampled_ids = {n["id"] for n in sampled}
    filtered_edges = [
        e for e in edge_list
        if e["src"] in sampled_ids and e["dst"] in sampled_ids
    ]
    return sampled, filtered_edges


# ---------------------------------------------------------------------------
# Public render function
# ---------------------------------------------------------------------------

def render_graph(
    node_list: list[dict],
    edge_list: list[dict],
    *,
    highlight_nodes: set[int] | None = None,
    highlight_edges: set[tuple[int, int]] | None = None,
    target_id: int | None = None,
    max_nodes: int = MAX_DISPLAY_NODES,
    height: int = 600,
    physics: bool = True,
    show_legend: bool = True,
    config_override: AGraphConfig | None = None,
) -> Any:
    """
    Render a transaction sub-graph inside the current Streamlit page.

    Parameters
    ----------
    node_list : list[dict]
        Node records.  Required key: "id".
        Optional: "label" (0/1/-1), "risk_score" (float), "volume" (float 0-1).
    edge_list : list[dict]
        Edge records.  Required keys: "src", "dst".
        Optional: "weight" (float 0-1).
    highlight_nodes : set[int], optional
        IDs to colour orange (explanation overlay – Stage 7e).
    highlight_edges : set[tuple[int, int]], optional
        Edges (src, dst) to highlight in orange with thicker strokes.
    target_id : int, optional
        The focal wallet; rendered purple and larger than peers.
    max_nodes : int
        Hard cap on displayed nodes.  Excess nodes are randomly sampled away.
    height : int
        Canvas height in pixels.
    physics : bool
        Enable/disable live physics simulation.
    show_legend : bool
        Render the colour legend below the graph.
    config_override : AGraphConfig, optional
        Supply a custom agraph Config to replace the default.

    Returns
    -------
    clicked_node_id : str | None
        The node ID that was most recently clicked by the user, or None.
    """
    highlight_nodes = highlight_nodes or set()
    highlight_edges = highlight_edges or set()

    # Always preserve focal target and explanation nodes if sampling
    keep_ids = set(highlight_nodes)
    if target_id is not None:
        keep_ids.add(target_id)

    # Sample if too many nodes
    display_nodes, display_edges = _sample_nodes(
        node_list, edge_list, max_nodes, keep_ids=keep_ids
    )

    agraph_nodes = _build_agraph_nodes(display_nodes, highlight_nodes, target_id)
    agraph_edges = _build_agraph_edges(display_edges, highlight_edges)

    graph_cfg = config_override or _make_agraph_config(height=height, physics=physics)

    # ---- Info strip --------------------------------------------------------
    n_total = len(node_list)
    e_total = len(edge_list)
    n_shown = len(display_nodes)
    e_shown = len(display_edges)

    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Nodes (total)", f"{n_total:,}")
    col_b.metric("Edges (total)", f"{e_total:,}")
    col_c.metric("Nodes (shown)", f"{n_shown:,}")
    col_d.metric("Edges (shown)", f"{e_shown:,}")

    if n_shown < n_total:
        st.caption(
            f"⚠️ Graph sampled to {n_shown} of {n_total} nodes for readability. "
            "Use wallet search (Stage 7b) to explore specific ego-networks."
        )

    # ---- Render ─────────────────────────────────────────────────────────────
    clicked = agraph(nodes=agraph_nodes, edges=agraph_edges, config=graph_cfg)

    # ---- Legend ─────────────────────────────────────────────────────────────
    if show_legend:
        _render_legend()

    return clicked


# ---------------------------------------------------------------------------
# Legend helper
# ---------------------------------------------------------------------------

def _render_legend() -> None:
    """Render a compact colour legend using Streamlit columns."""
    st.markdown("---")
    st.caption("**Node colour legend**")
    items = [
        (COLOUR_ILLICIT,   "Illicit (fraud)"),
        (COLOUR_LICIT,     "Licit (clean)"),
        (COLOUR_UNKNOWN,   "Unknown"),
        (COLOUR_TARGET,    "Selected wallet"),
        (COLOUR_HIGHLIGHT, "Explanation subgraph"),
    ]
    cols = st.columns(len(items))
    for col, (color, label) in zip(cols, items):
        col.markdown(
            f"<span style='display:inline-block;width:14px;height:14px;"
            f"background:{color};border-radius:50%;vertical-align:middle;"
            f"margin-right:6px'></span>{label}",
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Stage 7b – Ego-network helpers
# ---------------------------------------------------------------------------

import networkx as nx
import torch
from torch_geometric.data import Data as _PyGData


def build_nx_graph(graph_data: _PyGData) -> nx.DiGraph:
    """
    Convert a cached PyG Data object into a lightweight NetworkX DiGraph.

    Only node IDs and edge topology are stored (no feature tensors) so the
    graph stays memory-efficient for ego-network queries.

    Cache this at the call-site with ``@st.cache_resource`` — it is built
    only once per session.
    """
    G = nx.DiGraph()
    n = int(graph_data.num_nodes)
    G.add_nodes_from(range(n))
    ei = graph_data.edge_index.cpu().numpy()
    edges = list(zip(ei[0].tolist(), ei[1].tolist()))
    G.add_edges_from(edges)
    return G


def get_ego_network(
    wallet_id: int,
    G: nx.DiGraph,
    graph_data: _PyGData,
    hops: int = 2,
    max_nodes: int = MAX_DISPLAY_NODES,
) -> tuple[list[dict], list[dict]]:
    """
    Extract the N-hop ego-network around *wallet_id* and return it in the
    ``render_graph()`` dict format.

    Parameters
    ----------
    wallet_id : int
        The focal node (PyG integer index).
    G : nx.DiGraph
        Full directed graph built by ``build_nx_graph()``.
    graph_data : torch_geometric.data.Data
        Cached PyG Data object (for labels).
    hops : int
        Neighbourhood radius (default 2).
    max_nodes : int
        Hard cap — if the ego subgraph exceeds this, randomly trim outer
        nodes while always keeping the focal node and its 1-hop neighbours.

    Returns
    -------
    (node_list, edge_list) : tuple[list[dict], list[dict]]
        Ready to pass directly to ``render_graph()``.
    """
    if wallet_id not in G:
        raise ValueError(
            f"wallet_id {wallet_id} not found in graph ({G.number_of_nodes()} nodes)."
        )

    # Undirected projection for symmetric N-hop neighbourhood
    G_und = G.to_undirected(as_view=True)
    ego = nx.ego_graph(G_und, wallet_id, radius=hops)
    ego_nodes = list(ego.nodes())

    # Trim if over cap — keep 1-hop neighbours + focal node intact
    if len(ego_nodes) > max_nodes:
        one_hop = set(nx.ego_graph(G_und, wallet_id, radius=1).nodes())
        one_hop.add(wallet_id)
        outer = [n for n in ego_nodes if n not in one_hop]
        random.shuffle(outer)
        keep = list(one_hop) + outer[: max_nodes - len(one_hop)]
        ego_nodes = keep

    ego_set = set(ego_nodes)
    y = graph_data.y.cpu().numpy()

    node_list: list[dict] = [
        {"id": int(nid), "label": int(y[nid]) if nid < len(y) else -1}
        for nid in ego_nodes
    ]

    # Directed edges inside the ego subgraph (from original directed graph)
    ei = graph_data.edge_index.cpu().numpy()
    edge_list: list[dict] = [
        {"src": int(ei[0, j]), "dst": int(ei[1, j])}
        for j in range(ei.shape[1])
        if int(ei[0, j]) in ego_set and int(ei[1, j]) in ego_set
    ]

    return node_list, edge_list


def convert_pyg_to_display(
    graph_data: _PyGData,
    node_ids: list[int] | None = None,
) -> tuple[list[dict], list[dict]]:
    """
    Convert a PyG Data object (or subset of nodes) to render_graph() dicts.

    Parameters
    ----------
    graph_data : Data
        Full graph object.
    node_ids : list[int] | None
        If given, restrict to these node indices.  Otherwise all nodes.

    Returns
    -------
    (node_list, edge_list)
    """
    y = graph_data.y.cpu().numpy()
    ei = graph_data.edge_index.cpu().numpy()

    if node_ids is None:
        node_ids = list(range(graph_data.num_nodes))

    node_set = set(node_ids)
    node_list = [{"id": int(nid), "label": int(y[nid])} for nid in node_ids]
    edge_list = [
        {"src": int(ei[0, j]), "dst": int(ei[1, j])}
        for j in range(ei.shape[1])
        if int(ei[0, j]) in node_set and int(ei[1, j]) in node_set
    ]
    return node_list, edge_list


# ---------------------------------------------------------------------------
# Stage 7c – Risk score & volume inference helpers
# ---------------------------------------------------------------------------

import torch.nn.functional as _F
from torch_geometric.utils import k_hop_subgraph as _k_hop


# Feature index used as a proxy for "transaction output volume" in Elliptic.
# Features 0-9 are local transaction fee/output value metrics; index 0 is
# the strongest single-feature proxy for total BTC moved.
_VOLUME_FEATURE_IDX = 0


def _disable_explain_mode(model: "torch.nn.Module") -> None:
    """
    Reset GNNExplainer state on all MessagePassing layers.

    GNNExplainer sets ``_explain = True`` and stores an ``edge_mask`` on
    every conv layer during attribution.  If the model is then used for
    normal inference on a *different* subgraph, PyG's ``explain_message``
    asserts that the edge count matches the stored mask and raises an
    AssertionError.  This helper clears those flags so ordinary forward
    passes are unaffected.
    """
    for module in model.modules():
        # PyG ≥ 2.0 MessagePassing flags
        if hasattr(module, "_explain"):
            module._explain = False
        if hasattr(module, "explain"):
            module.explain = False
        # Remove any stale edge / node masks attached by GNNExplainer
        if hasattr(module, "edge_mask"):
            module.edge_mask = None
        if hasattr(module, "node_mask"):
            module.node_mask = None


def score_nodes(
    node_ids: list[int],
    graph_data: "_PyGData",
    model: "torch.nn.Module",
    device: "torch.device | None" = None,
    batch_size: int = 512,
) -> dict[int, dict]:
    """
    Run the trained Graph Transformer on a list of node IDs and return per-node
    risk scores and normalised volume values.

    Uses 2-hop k_hop_subgraph extraction per batch to stay CPU-friendly.

    Parameters
    ----------
    node_ids : list[int]
        Node indices to score.
    graph_data : torch_geometric.data.Data
        Full cached graph (CPU).
    model : torch.nn.Module
        Trained GraphTransformerClassifier in eval mode.
    device : torch.device, optional
        Inference device (defaults to CPU).
    batch_size : int
        How many nodes to process per subgraph extraction call.

    Returns
    -------
    dict mapping node_id -> {"risk_score": float, "volume": float}
        volume is normalised to [0, 1] within the returned set.
    """
    import torch as _torch

    device = device or _torch.device("cpu")
    model = model.to(device).eval()

    # Clear any GNNExplainer masks left from a previous explain_node() call.
    # Without this, PyG's explain_message() asserts edge-count == mask-size
    # and raises an AssertionError when the scoring subgraph differs in size.
    _disable_explain_mode(model)
    x_full = graph_data.x.to(device)
    ei_full = graph_data.edge_index.to(device)

    # Trim features to match model's expected in_channels
    in_ch = getattr(model, "in_channels", x_full.shape[1])
    if x_full.shape[1] > in_ch:
        x_full = x_full[:, :in_ch]

    results: dict[int, dict] = {}

    with _torch.no_grad():
        for start in range(0, len(node_ids), batch_size):
            batch_ids = node_ids[start : start + batch_size]
            batch_tensor = _torch.tensor(batch_ids, dtype=_torch.long, device=device)

            # Extract 2-hop subgraph for the whole batch at once
            subset, sub_ei, mapping, _ = _k_hop(
                node_idx=batch_tensor,
                num_hops=2,
                edge_index=ei_full,
                relabel_nodes=True,
            )
            sub_x = x_full[subset]

            logits = model(sub_x, sub_ei)          # (|subset|, 2)
            probs  = _F.softmax(logits, dim=-1)    # (|subset|, 2)

            for local_pos, global_id in zip(mapping.tolist(), batch_ids):
                risk  = float(probs[local_pos, 1].item())
                # raw volume from the full (untrimmed) feature tensor
                raw_vol = float(graph_data.x[global_id, _VOLUME_FEATURE_IDX].item())
                results[global_id] = {"risk_score": risk, "volume_raw": raw_vol}

    # Normalise volume to [0, 1] across the scored set
    vols = [v["volume_raw"] for v in results.values()]
    v_min, v_max = min(vols), max(vols)
    v_range = v_max - v_min if v_max > v_min else 1.0
    for nid in results:
        results[nid]["volume"] = (results[nid]["volume_raw"] - v_min) / v_range
        del results[nid]["volume_raw"]

    return results


def get_ego_network_with_scores(
    wallet_id: int,
    G: nx.DiGraph,
    graph_data: "_PyGData",
    model: "torch.nn.Module | None" = None,
    hops: int = 2,
    max_nodes: int = MAX_DISPLAY_NODES,
    device: "torch.device | None" = None,
) -> tuple[list[dict], list[dict]]:
    """
    Stage 7c wrapper: extracts ego-network AND attaches model risk scores +
    normalised volume to every node dict.

    Falls back gracefully to label-only colouring when *model* is None
    (e.g. checkpoint not yet available).

    Returns
    -------
    (node_list, edge_list)
        node dicts include "risk_score" and "volume" when model is provided.
    """
    node_list, edge_list = get_ego_network(
        wallet_id=wallet_id,
        G=G,
        graph_data=graph_data,
        hops=hops,
        max_nodes=max_nodes,
    )

    if model is not None:
        ids = [n["id"] for n in node_list]
        scores = score_nodes(ids, graph_data, model, device=device)
        for n in node_list:
            info = scores.get(n["id"], {})
            n["risk_score"] = info.get("risk_score")
            n["volume"]     = info.get("volume")

    return node_list, edge_list
