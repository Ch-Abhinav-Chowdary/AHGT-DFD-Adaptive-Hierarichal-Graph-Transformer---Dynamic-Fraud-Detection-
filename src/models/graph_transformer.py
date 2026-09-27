"""Graph Transformer model definition and loading utilities."""

from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv

import config


class GraphTransformerClassifier(nn.Module):
    """
    2-Layer Residual Graph Transformer for node classification.
    Uses multi-head attention (TransformerConv) with residual skip connections
    and LayerNorm to capture transaction topology while retaining local feature power.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        num_classes: int = 2,
        heads: int = 4,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.dropout = dropout
        self.heads = heads
        head_dim = hidden_channels // heads

        # Layer 1: in_channels -> hidden_channels
        self.conv1 = TransformerConv(
            in_channels,
            head_dim,
            heads=heads,
            dropout=dropout,
            concat=True,
            beta=True,
        )
        self.skip1 = nn.Linear(in_channels, hidden_channels)
        self.norm1 = nn.LayerNorm(hidden_channels)

        # Layer 2: hidden_channels -> hidden_channels
        self.conv2 = TransformerConv(
            hidden_channels,
            head_dim,
            heads=heads,
            dropout=dropout,
            concat=True,
            beta=True,
        )
        self.norm2 = nn.LayerNorm(hidden_channels)

        # Classifier MLP Head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

        self.attention_weights = {}

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        return_attention_weights: bool = False,
    ) -> torch.Tensor:
        if x.shape[1] > self.in_channels:
            x = x[:, :self.in_channels]

        # Layer 1
        if return_attention_weights:
            out1, (e_idx1, attn1) = self.conv1(x, edge_index, return_attention_weights=True)
            self.attention_weights["layer1"] = (e_idx1, attn1.detach())
        else:
            out1 = self.conv1(x, edge_index)

        h1 = self.norm1(out1 + self.skip1(x))
        h1 = F.elu(h1)
        h1 = F.dropout(h1, p=self.dropout, training=self.training)

        # Layer 2
        if return_attention_weights:
            out2, (e_idx2, attn2) = self.conv2(h1, edge_index, return_attention_weights=True)
            self.attention_weights["layer2"] = (e_idx2, attn2.detach())
        else:
            out2 = self.conv2(h1, edge_index)

        h2 = self.norm2(out2 + h1)
        h2 = F.elu(h2)
        h2 = F.dropout(h2, p=self.dropout, training=self.training)

        return self.classifier(h2)


def load_trained_graph_transformer(
    checkpoint_path: Optional[Path] = None,
    device: Optional[torch.device] = None,
) -> Tuple[GraphTransformerClassifier, dict]:
    """
    Load the trained Graph Transformer checkpoint from disk.

    Returns
    -------
    model : GraphTransformerClassifier in eval mode on device
    metadata : dict with training metadata (epoch, val_f1, best_threshold, etc.)
    """
    checkpoint_path = checkpoint_path or config.GRAPH_TRANSFORMER_CHECKPOINT
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {checkpoint_path}. Ensure graph_transformer.pt is placed in models/."
        )

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    in_channels = ckpt.get("in_channels", 168)
    hidden_dim = ckpt.get("hidden_dim", 128)
    heads = ckpt.get("heads", 4)
    dropout = ckpt.get("dropout", 0.2)

    model = GraphTransformerClassifier(
        in_channels=in_channels,
        hidden_channels=hidden_dim,
        heads=heads,
        dropout=dropout,
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    metadata = {
        "epoch": ckpt.get("epoch", 0),
        "val_f1": ckpt.get("val_f1", 0.0),
        "val_auc": ckpt.get("val_auc", 0.0),
        "best_threshold": ckpt.get("best_threshold", 0.5),
        "in_channels": in_channels,
        "hidden_dim": hidden_dim,
        "heads": heads,
    }

    return model, metadata
