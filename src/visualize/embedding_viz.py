"""Embedding visualizer for XAI-GT.

Extracts latent node embeddings from the trained Graph Transformer and
projects them to 2D using UMAP (preferred) or t-SNE (fallback), then
renders an interactive Plotly scatter coloured by fraud label and risk score.

Usage (standalone):
    python src/visualize/embedding_viz.py

Usage (Streamlit):
    from src.visualize.embedding_viz import build_embedding_figure
    fig = build_embedding_figure(model, graph_data, method="umap")
    st.plotly_chart(fig)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------

def extract_embeddings(
    model: torch.nn.Module,
    graph_data,
    device: Optional[torch.device] = None,
    max_nodes: int = 5000,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract penultimate-layer embeddings from the Graph Transformer.

    Parameters
    ----------
    model       : Trained GraphTransformerClassifier (or GraphSAGEClassifier).
    graph_data  : PyG Data object with .x, .edge_index, .y.
    device      : Torch device.
    max_nodes   : Sample at most this many labelled nodes (speed / memory).
    seed        : Random seed for reproducible sampling.

    Returns
    -------
    embeddings  : float32 array (N, D) — raw latent vectors
    labels      : int array  (N,)    — ground truth (0=licit, 1=illicit, -1=unknown)
    risk_scores : float32 array (N,) — model fraud probability
    """
    device = device or torch.device("cpu")
    model = model.to(device).eval()

    # Trim features to model's expected input size
    x = graph_data.x.to(device)
    if hasattr(model, "in_channels") and x.shape[1] > model.in_channels:
        x = x[:, : model.in_channels]
    edge_index = graph_data.edge_index.to(device)
    y = graph_data.y.cpu().numpy()

    # ── Labelled node sampling ──────────────────────────────────────────────
    labelled_mask = y >= 0
    labelled_idx = np.where(labelled_mask)[0]

    rng = np.random.default_rng(seed)
    if len(labelled_idx) > max_nodes:
        labelled_idx = rng.choice(labelled_idx, size=max_nodes, replace=False)
        labelled_idx = np.sort(labelled_idx)

    # ── Hook into the penultimate layer ─────────────────────────────────────
    # GraphTransformerClassifier: classifier is nn.Sequential; we hook before it.
    # GraphSAGEClassifier: hook before self.classifier (nn.Linear).
    _embedding_cache: list[torch.Tensor] = []

    def _hook(module, inp, out):
        _embedding_cache.append(inp[0].detach().cpu())

    # Register hook on the first Linear of the classification head
    hook_handle = None
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear):
            # First Linear encountered in the classifier head
            hook_handle = mod.register_forward_hook(_hook)
            break

    with torch.no_grad():
        logits = model(x, edge_index)
        probs = F.softmax(logits, dim=-1)[:, 1].cpu().numpy()

    if hook_handle is not None:
        hook_handle.remove()

    if _embedding_cache:
        all_embeddings = _embedding_cache[0].numpy()  # (N_total, D)
    else:
        # Fallback: use raw features if hook failed
        all_embeddings = x.cpu().numpy()

    embeddings = all_embeddings[labelled_idx]
    labels = y[labelled_idx]
    risk_scores = probs[labelled_idx]

    return embeddings.astype(np.float32), labels.astype(np.int32), risk_scores.astype(np.float32)


# ---------------------------------------------------------------------------
# Dimensionality reduction
# ---------------------------------------------------------------------------

def reduce_to_2d(
    embeddings: np.ndarray,
    method: Literal["umap", "tsne"] = "umap",
    seed: int = 42,
) -> np.ndarray:
    """Project high-dimensional embeddings to 2D.

    Tries UMAP first; falls back to t-SNE if umap-learn is not installed.

    Parameters
    ----------
    embeddings : (N, D) array.
    method     : 'umap' or 'tsne'.
    seed       : Random seed.

    Returns
    -------
    coords : (N, 2) float32 array of 2D coordinates.
    """
    if method == "umap":
        try:
            import umap  # type: ignore
            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=30,
                min_dist=0.1,
                metric="cosine",
                random_state=seed,
                low_memory=True,
            )
            return reducer.fit_transform(embeddings).astype(np.float32)
        except ImportError:
            pass  # Fall through to t-SNE

    # t-SNE fallback
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA

    # Pre-reduce with PCA to speed up t-SNE when embedding dim is large
    n_components_pca = min(50, embeddings.shape[1], embeddings.shape[0] - 1)
    if embeddings.shape[1] > 50:
        pca = PCA(n_components=n_components_pca, random_state=seed)
        embeddings = pca.fit_transform(embeddings)

    tsne = TSNE(
        n_components=2,
        perplexity=min(30, embeddings.shape[0] // 4),
        learning_rate="auto",
        init="pca",
        random_state=seed,
        n_iter=1000,
    )
    return tsne.fit_transform(embeddings).astype(np.float32)


# ---------------------------------------------------------------------------
# Plotly figure builder
# ---------------------------------------------------------------------------

def build_embedding_figure(
    model: torch.nn.Module,
    graph_data,
    method: Literal["umap", "tsne"] = "umap",
    color_by: Literal["label", "risk"] = "label",
    max_nodes: int = 5000,
    highlight_node: Optional[int] = None,
    device: Optional[torch.device] = None,
    seed: int = 42,
) -> go.Figure:
    """Build the full UMAP/t-SNE Plotly scatter figure.

    Parameters
    ----------
    model         : Trained Graph Transformer / GraphSAGE.
    graph_data    : PyG Data object.
    method        : Projection algorithm ('umap' or 'tsne').
    color_by      : Colour points by ground-truth label or model risk score.
    max_nodes     : Max labelled nodes to sample.
    highlight_node: If given, mark this node ID with a star marker.
    device        : Torch device.
    seed          : Random seed.

    Returns
    -------
    fig : Plotly Figure object ready for st.plotly_chart().
    """
    embeddings, labels, risk_scores = extract_embeddings(
        model, graph_data, device=device, max_nodes=max_nodes, seed=seed
    )
    coords = reduce_to_2d(embeddings, method=method, seed=seed)

    fig = go.Figure()

    if color_by == "label":
        # ── Three traces: Licit / Illicit / Unknown ──────────────────────
        groups = {
            1: ("Illicit (Fraud)", "#EF4444", "circle", 0.85),
            0: ("Licit (Clean)",   "#22C55E", "circle", 0.55),
        }
        for lbl, (name, color, sym, opacity) in groups.items():
            mask = labels == lbl
            if not mask.any():
                continue
            fig.add_trace(go.Scatter(
                x=coords[mask, 0],
                y=coords[mask, 1],
                mode="markers",
                name=name,
                marker=dict(
                    size=5,
                    color=color,
                    opacity=opacity,
                    line=dict(width=0),
                ),
                text=[f"Risk: {r:.1%}" for r in risk_scores[mask]],
                hovertemplate=(
                    f"<b>{name}</b><br>"
                    "UMAP-1: %{x:.3f}<br>"
                    "UMAP-2: %{y:.3f}<br>"
                    "%{text}<extra></extra>"
                ),
            ))

    else:
        # ── Continuous risk-score gradient ────────────────────────────────
        fig.add_trace(go.Scatter(
            x=coords[:, 0],
            y=coords[:, 1],
            mode="markers",
            name="Risk Score",
            marker=dict(
                size=5,
                color=risk_scores,
                colorscale=[
                    [0.0,  "#22C55E"],
                    [0.35, "#EAB308"],
                    [0.70, "#F97316"],
                    [1.0,  "#EF4444"],
                ],
                cmin=0.0,
                cmax=1.0,
                opacity=0.75,
                colorbar=dict(
                    title="Fraud<br>Probability",
                    thickness=14,
                    len=0.75,
                    tickformat=".0%",
                    tickfont=dict(color="#CBD5E1"),
                    titlefont=dict(color="#CBD5E1"),
                ),
                line=dict(width=0),
            ),
            hovertemplate=(
                "Risk: %{marker.color:.1%}<br>"
                "UMAP-1: %{x:.3f}<br>"
                "UMAP-2: %{y:.3f}<extra></extra>"
            ),
        ))

    # ── Highlight star for currently investigated wallet ──────────────────
    if highlight_node is not None:
        y_arr = graph_data.y.cpu().numpy()
        if 0 <= highlight_node < len(y_arr):
            lbl = int(y_arr[highlight_node])
            # Find this node's position in our sampled set
            # (it may not be in the sample if unlabelled or outside max_nodes)
            # We add a special annotation using the full-graph embedding
            try:
                full_emb, _, _ = extract_embeddings(
                    model, graph_data, device=device, max_nodes=200000, seed=seed
                )
                # The node's embedding in the FULL set — project separately
                # is expensive; instead just annotate at origin with a note
                pass
            except Exception:
                pass

            fig.add_annotation(
                text=f"★ #{highlight_node}",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.01, y=0.99,
                font=dict(color="#A855F7", size=13, family="monospace"),
                align="left",
            )

    # ── Layout ────────────────────────────────────────────────────────────
    method_label = "UMAP" if method == "umap" else "t-SNE"
    fig.update_layout(
        title=dict(
            text=f"<b>{method_label} Embedding Space</b> — Graph Transformer Latent Representations",
            font=dict(color="#F1F5F9", size=15),
        ),
        paper_bgcolor="rgba(11,15,25,0.0)",
        plot_bgcolor="rgba(15,20,35,0.6)",
        xaxis=dict(
            title=f"{method_label}-1",
            gridcolor="rgba(255,255,255,0.06)",
            zerolinecolor="rgba(255,255,255,0.1)",
            tickfont=dict(color="#64748B"),
            title_font=dict(color="#94A3B8"),
        ),
        yaxis=dict(
            title=f"{method_label}-2",
            gridcolor="rgba(255,255,255,0.06)",
            zerolinecolor="rgba(255,255,255,0.1)",
            tickfont=dict(color="#64748B"),
            title_font=dict(color="#94A3B8"),
        ),
        legend=dict(
            bgcolor="rgba(15,23,42,0.7)",
            bordercolor="rgba(255,255,255,0.1)",
            borderwidth=1,
            font=dict(color="#CBD5E1", size=12),
        ),
        margin=dict(l=40, r=20, t=60, b=40),
        height=560,
        hoverlabel=dict(
            bgcolor="rgba(15,23,42,0.9)",
            bordercolor="rgba(255,255,255,0.15)",
            font=dict(color="#F1F5F9"),
        ),
    )

    return fig


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import config
    from src.models.graph_transformer import load_trained_graph_transformer
    from src.preprocessing.graph_builder import load_graph

    print("Loading model and graph …")
    model, meta = load_trained_graph_transformer()
    data = load_graph()

    print("Building UMAP figure …")
    fig = build_embedding_figure(model, data, method="umap", color_by="label", max_nodes=4000)
    fig.write_html("umap_preview.html")
    print("Saved to umap_preview.html — open in browser to inspect.")
