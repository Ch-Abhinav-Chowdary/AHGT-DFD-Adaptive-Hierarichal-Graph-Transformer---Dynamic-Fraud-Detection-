"""Build and cache the PyTorch Geometric graph from the Elliptic dataset.

Stage 3 deliverables
--------------------
* build_graph()   -- constructs a PyG Data object with:
    - node feature matrix  (164 raw features + 3 computed: in-degree,
                            out-degree, clustering coefficient)
    - edge_index           (from elliptic_txs_edgelist.csv)
    - y                    (binary fraud labels; -1 for unlabelled nodes)
  Nodes are only those present in the LABELLED splits (train + val + test).
  Computed features are derived via NetworkX on the induced subgraph.

* cache_graph()   -- saves the Data object with torch.save().
* load_graph()    -- loads the cached object (fast re-use in later stages).

Acceptance criteria (printed at runtime):
    print(graph_data) shows ~203 K nodes, ~234 K edges, feature_dim = 167.

Run directly:
    python src/preprocessing/graph_builder.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.preprocessing.data_loader import (
    FEATURE_COLS,
    load_elliptic,
    load_splits,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

COMPUTED_FEATURE_COLS = ["in_degree", "out_degree", "clustering_coef"]
ALL_FEATURE_COLS = FEATURE_COLS + COMPUTED_FEATURE_COLS   # 165 + 3 = 168


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_node_index(
    all_tx_ids: pd.Series,
) -> dict[int, int]:
    """
    Create a stable txId -> consecutive integer mapping for ALL nodes
    (labelled and unknown alike) so that the full edge set is preserved.
    """
    unique_ids = all_tx_ids.drop_duplicates()
    return {tx_id: idx for idx, tx_id in enumerate(unique_ids)}


def _build_networkx_graph(
    edgelist: pd.DataFrame,
    node_index: dict[int, int],
) -> nx.DiGraph:
    """
    Build a directed NetworkX graph restricted to the labelled node set.

    Only edges whose BOTH endpoints exist in node_index are kept.
    """
    G = nx.DiGraph()
    G.add_nodes_from(range(len(node_index)))

    src_col, dst_col = "txId1", "txId2"
    src_mapped = edgelist[src_col].map(node_index)
    dst_mapped = edgelist[dst_col].map(node_index)

    # Keep only edges where both nodes are in our labelled set
    mask = src_mapped.notna() & dst_mapped.notna()
    edges = list(zip(src_mapped[mask].astype(int), dst_mapped[mask].astype(int)))
    G.add_edges_from(edges)

    return G


def _compute_graph_features(
    G: nx.DiGraph,
    num_nodes: int,
) -> np.ndarray:
    """
    Compute per-node structural features using NetworkX.

    Returns an (num_nodes, 3) float32 array:
        col 0 -- in-degree  (normalised by max)
        col 1 -- out-degree (normalised by max)
        col 2 -- local clustering coefficient (on undirected projection)
    """
    print("[Stage 3]   Computing in/out-degrees ...")
    in_deg  = np.array([G.in_degree(n)  for n in range(num_nodes)], dtype=np.float32)
    out_deg = np.array([G.out_degree(n) for n in range(num_nodes)], dtype=np.float32)

    # Normalise degrees to [0, 1]
    in_max  = in_deg.max()  if in_deg.max()  > 0 else 1.0
    out_max = out_deg.max() if out_deg.max() > 0 else 1.0
    in_deg  = in_deg  / in_max
    out_deg = out_deg / out_max

    print("[Stage 3]   Computing clustering coefficients (undirected projection) ...")
    G_und = G.to_undirected()
    clust = nx.clustering(G_und)
    clust_arr = np.array([clust.get(n, 0.0) for n in range(num_nodes)], dtype=np.float32)

    return np.stack([in_deg, out_deg, clust_arr], axis=1)   # (N, 3)


def _build_feature_matrix(
    train_df:   pd.DataFrame,
    val_df:     pd.DataFrame,
    test_df:    pd.DataFrame,
    all_df:     pd.DataFrame,
    node_index: dict[int, int],
    computed:   np.ndarray,
) -> np.ndarray:
    """
    Assemble the full (N, 168) node feature matrix.

    For unlabelled nodes their raw features come from all_df (the full
    features CSV). Labelled nodes override with normalised features from
    the split DataFrames so normalisation is consistent.
    """
    num_nodes = len(node_index)
    num_raw   = len(FEATURE_COLS)
    X = np.zeros((num_nodes, num_raw), dtype=np.float32)

    # Fill all nodes with raw (unnormalised) features first
    if all_df is not None:
        all_idxs = all_df[config.TX_ID_COL].map(node_index)
        valid    = all_idxs.notna()
        X[all_idxs[valid].astype(int).values] = (
            all_df.loc[valid, FEATURE_COLS].values.astype(np.float32)
        )

    # Overwrite labelled nodes with normalised features
    for df in (train_df, val_df, test_df):
        idxs = df[config.TX_ID_COL].map(node_index).values
        X[idxs] = df[FEATURE_COLS].values.astype(np.float32)

    # Concatenate computed structural features
    X_full = np.concatenate([X, computed], axis=1)   # (N, 168)
    return X_full


def _build_label_vector(
    train_df:   pd.DataFrame,
    val_df:     pd.DataFrame,
    test_df:    pd.DataFrame,
    node_index: dict[int, int],
) -> np.ndarray:
    """
    Build label vector y of shape (N,).

    Values: 0 = licit, 1 = illicit, -1 = unlabelled.
    """
    num_nodes = len(node_index)
    y = np.full(num_nodes, -1, dtype=np.int64)

    for df in (train_df, val_df, test_df):
        idxs   = df[config.TX_ID_COL].map(node_index).values
        labels = df["label"].values.astype(np.int64)
        y[idxs] = labels

    return y


def _build_split_masks(
    train_df:   pd.DataFrame,
    val_df:     pd.DataFrame,
    test_df:    pd.DataFrame,
    node_index: dict[int, int],
    num_nodes:  int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Boolean masks for train / val / test nodes."""
    def _mask(df: pd.DataFrame) -> torch.Tensor:
        mask = torch.zeros(num_nodes, dtype=torch.bool)
        idxs = df[config.TX_ID_COL].map(node_index).dropna().astype(int).values
        mask[idxs] = True
        return mask

    return _mask(train_df), _mask(val_df), _mask(test_df)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_graph(
    train_df:   pd.DataFrame | None = None,
    val_df:     pd.DataFrame | None = None,
    test_df:    pd.DataFrame | None = None,
    edgelist:   pd.DataFrame | None = None,
) -> Data:
    """
    Build a PyTorch Geometric Data object for the Elliptic dataset.

    Includes ALL 203K nodes (labelled + unknown) so the full 234K-edge
    graph structure is preserved.  Unknown nodes get y=-1 and are excluded
    from train/val/test masks -- they act as message-passing neighbours
    only.  This is standard semi-supervised GNN practice.

    Returns
    -------
    graph_data : torch_geometric.data.Data
        .x            -- float32 tensor (N, 168)
        .edge_index   -- int64  tensor (2, E)   (~234K edges)
        .y            -- int64  tensor (N,)   {0, 1, -1}
        .train_mask   -- bool   tensor (N,)
        .val_mask     -- bool   tensor (N,)
        .test_mask    -- bool   tensor (N,)
        .num_nodes    -- int    (~203K)
    """
    # --- Load splits if not provided ---
    if train_df is None:
        processed_dir = config.PROCESSED_DATA_DIR
        if (processed_dir / "train.parquet").exists():
            print("[Stage 3] Loading pre-processed splits from data/processed/ ...")
            train_df, val_df, test_df, edgelist = load_splits()
        else:
            print("[Stage 3] Processed splits not found -- running Stage 2 pipeline ...")
            train_df, val_df, test_df, edgelist = load_elliptic()

    assert edgelist is not None, "edgelist must be provided or loadable from splits."

    # --- Load ALL node IDs (including unknown) for full edge coverage ---
    # We read the features CSV and normalize the features for all nodes (including unknowns)
    # using StandardScaler fit on the training split to fix the zero-features bug.
    all_df: pd.DataFrame | None = None
    if config.ELLIPTIC_FEATURES_FILE.exists():
        print("[Stage 3] Loading full features (labelled + unknown) ...")
        all_df = pd.read_csv(
            config.ELLIPTIC_FEATURES_FILE,
            header=None,
        )
        all_df.columns = [config.TX_ID_COL, config.TIME_STEP_COL] + FEATURE_COLS
        
        # Fit scaler on raw training features
        train_ids = set(train_df[config.TX_ID_COL])
        raw_train = all_df[all_df[config.TX_ID_COL].isin(train_ids)]
        
        print("[Stage 3] Fitting StandardScaler on training features ...")
        scaler = StandardScaler()
        scaler.fit(raw_train[FEATURE_COLS])
        
        print("[Stage 3] Normalizing all node features ...")
        all_df[FEATURE_COLS] = scaler.transform(all_df[FEATURE_COLS])

        all_tx_ids = pd.Series(np.concatenate([
            all_df[config.TX_ID_COL].values,
            train_df[config.TX_ID_COL].values,
            val_df[config.TX_ID_COL].values,
            test_df[config.TX_ID_COL].values,
        ]))
    else:
        print("[Stage 3] Features CSV not found; falling back to labelled-only graph.")
        all_tx_ids = pd.Series(np.concatenate([
            train_df[config.TX_ID_COL].values,
            val_df[config.TX_ID_COL].values,
            test_df[config.TX_ID_COL].values,
        ]))

    print(f"[Stage 3] Train: {train_df.shape} | Val: {val_df.shape} | Test: {test_df.shape}")

    # --- Node index (ALL nodes) ---
    print("[Stage 3] Building node index (all nodes) ...")
    node_index = _build_node_index(all_tx_ids)
    num_nodes  = len(node_index)
    print(f"[Stage 3] Total nodes (incl. unknown): {num_nodes:,}")

    # --- NetworkX graph (for structural features) ---
    print("[Stage 3] Building NetworkX graph ...")
    G = _build_networkx_graph(edgelist, node_index)
    num_edges_nx = G.number_of_edges()
    print(f"[Stage 3] NetworkX graph: {G.number_of_nodes():,} nodes, {num_edges_nx:,} edges")

    # --- Computed features ---
    computed = _compute_graph_features(G, num_nodes)   # (N, 3)

    # --- Feature matrix ---
    print("[Stage 3] Assembling feature matrix ...")
    X = _build_feature_matrix(train_df, val_df, test_df, all_df, node_index, computed)
    print(f"[Stage 3] Feature matrix shape: {X.shape}  (expected N x 168)")

    # --- Labels (y=-1 for unlabelled) ---
    y = _build_label_vector(train_df, val_df, test_df, node_index)

    # --- Edge index (PyG format: shape [2, E]) ---
    print("[Stage 3] Building edge_index tensor ...")
    src_col, dst_col = "txId1", "txId2"
    src_mapped = edgelist[src_col].map(node_index)
    dst_mapped = edgelist[dst_col].map(node_index)
    mask = src_mapped.notna() & dst_mapped.notna()
    src_arr = src_mapped[mask].astype(int).values
    dst_arr = dst_mapped[mask].astype(int).values
    edge_index = torch.tensor(
        np.stack([src_arr, dst_arr], axis=0), dtype=torch.long
    )   # shape (2, E)
    print(f"[Stage 3] edge_index shape: {edge_index.shape}  (expected 2 x ~234K)")

    # --- Split masks ---
    train_mask, val_mask, test_mask = _build_split_masks(
        train_df, val_df, test_df, node_index, num_nodes
    )

    # --- Assemble PyG Data object ---
    graph_data = Data(
        x          = torch.tensor(X,  dtype=torch.float),
        edge_index = edge_index,
        y          = torch.tensor(y,  dtype=torch.long),
        train_mask = train_mask,
        val_mask   = val_mask,
        test_mask  = test_mask,
        num_nodes  = num_nodes,
    )

    return graph_data


def cache_graph(graph_data: Data, path: Path | None = None) -> None:
    """Save the graph Data object to disk with torch.save()."""
    path = path or config.GRAPH_CACHE_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(graph_data, path)
    print(f"[Stage 3] Graph cached to {path}")


def load_graph(path: Path | None = None) -> Data:
    """Load a cached graph Data object."""
    path = path or config.GRAPH_CACHE_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"Graph cache not found at {path}. Run graph_builder.py first."
        )
    return torch.load(path, weights_only=False)


# ---------------------------------------------------------------------------
# Stage 3 runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    graph_data = build_graph()

    print("\n=== Stage 3 Results ===")
    print(graph_data)
    print(f"  Nodes          : {graph_data.num_nodes:,}")
    print(f"  Edges          : {graph_data.edge_index.shape[1]:,}")
    print(f"  Feature dim    : {graph_data.x.shape[1]}")
    print(f"  Train nodes    : {graph_data.train_mask.sum().item():,}")
    print(f"  Val nodes      : {graph_data.val_mask.sum().item():,}")
    print(f"  Test nodes     : {graph_data.test_mask.sum().item():,}")
    print(f"  Illicit nodes  : {(graph_data.y == 1).sum().item():,}")
    print(f"  Licit nodes    : {(graph_data.y == 0).sum().item():,}")
    print(f"  Unlabelled     : {(graph_data.y == -1).sum().item():,}")

    cache_graph(graph_data)
    print("\n[Stage 3] [DONE] Complete -- graph_data.pt saved to data/processed/")
