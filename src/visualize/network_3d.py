"""Stage 10 - 3D Transaction-Graph Visualiser (Plotly WebGL).

Public API
----------
render_graph_3d(node_list, edge_list, *, highlight_nodes=None,
                highlight_edges=None, target_id=None, height=650,
                physics_seed=42) -> go.Figure
    Builds a Plotly 3D scatter + line-trace figure from the same
    node_list / edge_list dicts used by render_graph() in network_view.py.
    Returns the figure so the caller can pass it to st.plotly_chart().

layout_graph_3d(node_list, edge_list, *, seed=42) -> dict[int, tuple[float,float,float]]
    Compute a 3-D spring layout using NetworkX and return a mapping of
    node_id -> (x, y, z).  Exposed so callers can cache the layout
    separately from the render step.

Data contracts  (identical to network_view.py)
--------------
node_list : list[dict]  - required key: "id" (int)
    Optional: "risk_score" (float 0-1), "volume" (float 0-1), "label" (0/1/-1)

edge_list : list[dict]  - required keys: "src" (int), "dst" (int)
    Optional: "weight" (float 0-1)

highlight_nodes : set[int]  - orange evidence overlay
highlight_edges : set[tuple[int,int]]  - orange thick edge overlay
target_id : int  - the focal wallet (purple, oversized)
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import networkx as nx
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Visual constants (matched to network_view.py palette)
# ---------------------------------------------------------------------------

COLOUR_ILLICIT    = "#EF4444"
COLOUR_LICIT      = "#22C55E"
COLOUR_UNKNOWN    = "#94A3B8"
COLOUR_HIGHLIGHT  = "#F97316"
COLOUR_TARGET     = "#A855F7"
COLOUR_EDGE_DEF   = "#334155"
COLOUR_EDGE_HI    = "#F97316"

NODE_SIZE_MIN     = 5
NODE_SIZE_DEFAULT = 9
NODE_SIZE_MAX     = 24
NODE_SIZE_TARGET  = 28

EDGE_WIDTH_DEFAULT = 1.0
EDGE_WIDTH_HI      = 4.0


# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------

def _risk_to_rgb(risk: float) -> tuple[int, int, int]:
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
    return r, g, b


def _risk_to_css(risk: float) -> str:
    r, g, b = _risk_to_rgb(risk)
    return f"rgb({r},{g},{b})"


def _node_colour_3d(node: dict, highlight_nodes: set[int], target_id) -> str:
    nid = node["id"]
    if nid == target_id:
        return COLOUR_TARGET
    if nid in highlight_nodes:
        return COLOUR_HIGHLIGHT
    risk = node.get("risk_score")
    if risk is not None:
        return _risk_to_css(float(risk))
    lbl = node.get("label", -1)
    if lbl == 1:
        return COLOUR_ILLICIT
    if lbl == 0:
        return COLOUR_LICIT
    return COLOUR_UNKNOWN


def _node_size_3d(node: dict, target_id) -> int:
    if node["id"] == target_id:
        return NODE_SIZE_TARGET
    vol = node.get("volume")
    if vol is None:
        return NODE_SIZE_DEFAULT
    vol = max(0.0, min(1.0, float(vol)))
    return int(NODE_SIZE_MIN + vol * (NODE_SIZE_MAX - NODE_SIZE_MIN))


# ---------------------------------------------------------------------------
# 3-D spring layout
# ---------------------------------------------------------------------------

def layout_graph_3d(
    node_list: list,
    edge_list: list,
    *,
    seed: int = 42,
) -> dict:
    """
    Compute a 3-D spring layout for the given node/edge lists.
    Returns dict mapping node_id -> (x, y, z).
    """
    G = nx.Graph()
    node_ids = [n["id"] for n in node_list]
    G.add_nodes_from(node_ids)
    for e in edge_list:
        G.add_edge(int(e["src"]), int(e["dst"]))

    if G.number_of_nodes() == 0:
        return {}

    k = 1.5 / math.sqrt(max(G.number_of_nodes(), 1))
    pos = nx.spring_layout(G, dim=3, seed=seed, k=k, iterations=60)

    return {
        nid: (float(pos[nid][0]), float(pos[nid][1]), float(pos[nid][2]))
        for nid in node_ids
        if nid in pos
    }


# ---------------------------------------------------------------------------
# Plotly trace builders
# ---------------------------------------------------------------------------

def _build_edge_traces(
    edge_list: list,
    pos: dict,
    highlight_edges: set,
) -> list:
    def_x, def_y, def_z = [], [], []
    hi_x,  hi_y,  hi_z  = [], [], []

    for e in edge_list:
        src, dst = int(e["src"]), int(e["dst"])
        if src not in pos or dst not in pos:
            continue
        xs, ys, zs = pos[src]
        xd, yd, zd = pos[dst]

        is_hi = (src, dst) in highlight_edges or (dst, src) in highlight_edges
        if is_hi:
            hi_x  += [xs, xd, None]
            hi_y  += [ys, yd, None]
            hi_z  += [zs, zd, None]
        else:
            def_x += [xs, xd, None]
            def_y += [ys, yd, None]
            def_z += [zs, zd, None]

    traces = []

    if def_x:
        traces.append(go.Scatter3d(
            x=def_x, y=def_y, z=def_z,
            mode="lines",
            name="Transactions",
            line=dict(color=COLOUR_EDGE_DEF, width=EDGE_WIDTH_DEFAULT),
            hoverinfo="none",
            showlegend=False,
        ))

    if hi_x:
        traces.append(go.Scatter3d(
            x=hi_x, y=hi_y, z=hi_z,
            mode="lines",
            name="Evidence Edges",
            line=dict(color=COLOUR_EDGE_HI, width=EDGE_WIDTH_HI),
            hoverinfo="none",
            showlegend=True,
        ))

    return traces


def _build_node_traces(
    node_list: list,
    pos: dict,
    highlight_nodes: set,
    target_id,
) -> list:
    groups = {
        "target":   {"x": [], "y": [], "z": [], "text": [], "colour": [], "size": []},
        "evidence": {"x": [], "y": [], "z": [], "text": [], "colour": [], "size": []},
        "normal":   {"x": [], "y": [], "z": [], "text": [], "colour": [], "size": []},
    }

    for n in node_list:
        nid = n["id"]
        if nid not in pos:
            continue
        x, y, z = pos[nid]

        colour = _node_colour_3d(n, highlight_nodes, target_id)
        size   = _node_size_3d(n, target_id)

        risk  = n.get("risk_score")
        lbl   = n.get("label", -1)
        lbl_s = {1: "ILLICIT", 0: "LICIT", -1: "unknown"}.get(lbl, "?")
        tip   = f"Node: {nid}<br>Label: {lbl_s}"
        if risk is not None:
            tip += f"<br>Risk: {risk:.1%}"
        if nid == target_id:
            tip += "<br><b>FOCAL WALLET</b>"
        if nid in highlight_nodes:
            tip += "<br><b>Explanation Evidence</b>"

        if nid == target_id:
            grp = "target"
        elif nid in highlight_nodes:
            grp = "evidence"
        else:
            grp = "normal"

        groups[grp]["x"].append(x)
        groups[grp]["y"].append(y)
        groups[grp]["z"].append(z)
        groups[grp]["text"].append(tip)
        groups[grp]["colour"].append(colour)
        groups[grp]["size"].append(size)

    legend_cfg = {
        "target":   ("Investigated Wallet", True),
        "evidence": ("Explanation Evidence", True),
        "normal":   ("Transactions",         False),
    }

    traces = []
    for grp_key, (grp_name, show_legend) in legend_cfg.items():
        g = groups[grp_key]
        if not g["x"]:
            continue
        traces.append(go.Scatter3d(
            x=g["x"], y=g["y"], z=g["z"],
            mode="markers",
            name=grp_name,
            marker=dict(
                size=g["size"],
                color=g["colour"],
                opacity=0.92,
                line=dict(width=0.5, color="rgba(255,255,255,0.15)"),
            ),
            text=g["text"],
            hovertemplate="%{text}<extra></extra>",
            showlegend=show_legend,
        ))

    return traces


# ---------------------------------------------------------------------------
# Main public render function
# ---------------------------------------------------------------------------

def render_graph_3d(
    node_list: list,
    edge_list: list,
    *,
    highlight_nodes=None,
    highlight_edges=None,
    target_id=None,
    height: int = 650,
    physics_seed: int = 42,
    cached_pos=None,
):
    """
    Build a Plotly 3D scatter figure visualising the transaction ego-network.

    Parameters
    ----------
    node_list, edge_list : same format as network_view.render_graph()
    highlight_nodes : set[int]   - GNNExplainer attribution nodes (orange)
    highlight_edges : set[tuple] - GNNExplainer attribution edges (orange + thick)
    target_id : int              - Focal wallet (purple, oversized)
    height : int                 - Figure height in pixels
    physics_seed : int           - Seed for reproducible spring layout
    cached_pos : dict, optional  - Pre-computed positions to skip re-layout

    Returns
    -------
    go.Figure  - pass to st.plotly_chart(fig, use_container_width=True)
    """
    highlight_nodes = highlight_nodes or set()
    highlight_edges = highlight_edges or set()

    pos = cached_pos if cached_pos is not None else layout_graph_3d(
        node_list, edge_list, seed=physics_seed
    )

    edge_traces = _build_edge_traces(edge_list, pos, highlight_edges)
    node_traces = _build_node_traces(node_list, pos, highlight_nodes, target_id)

    all_traces = edge_traces + node_traces

    scene = dict(
        bgcolor="rgb(11, 15, 25)",
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False,
                   backgroundcolor="rgb(11, 15, 25)", title=""),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False,
                   backgroundcolor="rgb(11, 15, 25)", title=""),
        zaxis=dict(showticklabels=False, showgrid=False, zeroline=False,
                   backgroundcolor="rgb(11, 15, 25)", title=""),
        camera=dict(eye=dict(x=1.4, y=1.4, z=1.0), up=dict(x=0, y=0, z=1)),
        aspectmode="cube",
    )

    layout = go.Layout(
        height=height,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgb(11, 15, 25)",
        plot_bgcolor="rgb(11, 15, 25)",
        scene=scene,
        legend=dict(
            font=dict(color="#CBD5E1", size=12),
            bgcolor="rgba(15, 23, 42, 0.7)",
            bordercolor="rgba(255,255,255,0.08)",
            borderwidth=1,
            x=0.01,
            y=0.99,
        ),
        uirevision="3d-graph",
    )

    return go.Figure(data=all_traces, layout=layout)
