"""
Graph Neural Network models for ETA prediction.

Two architectures implemented with PyTorch Geometric:

1. GraphSAGE (inductive)
   - 3-layer SAGEConv with mean aggregation
   - Node features: node2vec + handcrafted hub stats
   - Edge features appended at final regression head
   - Inductive → can handle new hubs without retraining

2. Graph Attention Network (GAT)
   - 2-layer GATConv with 8 attention heads
   - Attention weights are saved for interpretability
   - Visualising which neighbours influence a hub's delay most

Task: edge-level regression — predict actual_time for each corridor.
Edge prediction = concat(h_src, h_dst, edge_features) → MLP → actual_time
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

log = logging.getLogger(__name__)

MODEL_DIR = Path("data/processed/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

try:
    from torch_geometric.nn import SAGEConv, GATConv
    from torch_geometric.data import Data
    _HAS_PYG = True
except ImportError:
    _HAS_PYG = False
    log.warning("PyTorch Geometric not installed. GNN models unavailable.")


# ── Helper: within-15% metric ─────────────────────────────────────────────────

def within_15pct(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    ratio = torch.abs(y_pred - y_true) / (y_true.abs() + 1e-8)
    return float((ratio <= 0.15).float().mean().item() * 100)


# ── PyG Data builder ──────────────────────────────────────────────────────────

def build_pyg_data(
    node_features: np.ndarray,          # (N, node_feat_dim)
    node_index: list[str],              # ordered list of node names
    edge_df,                            # DataFrame with source, destination, actual_time, edge features
    edge_feature_cols: list[str],
) -> "Data":
    """Convert node/edge arrays to a PyTorch Geometric Data object."""
    assert _HAS_PYG, "PyTorch Geometric required."

    name_to_idx = {name: i for i, name in enumerate(node_index)}

    srcs, dsts, edge_feats, targets = [], [], [], []
    for _, row in edge_df.iterrows():
        s = name_to_idx.get(row["source"])
        d = name_to_idx.get(row["destination"])
        if s is None or d is None:
            continue
        srcs.append(s)
        dsts.append(d)
        feat = [float(row.get(c, 0.0)) for c in edge_feature_cols]
        edge_feats.append(feat)
        targets.append(float(row.get("actual_time", 0.0)))

    data = Data(
        x=torch.tensor(node_features, dtype=torch.float),
        edge_index=torch.tensor([srcs, dsts], dtype=torch.long),
        edge_attr=torch.tensor(edge_feats, dtype=torch.float),
        y=torch.tensor(targets, dtype=torch.float),
    )
    return data


# ── GraphSAGE ─────────────────────────────────────────────────────────────────

class GraphSAGEModel(nn.Module):
    def __init__(self, node_feat_dim: int, edge_feat_dim: int, hidden: int = 128, out: int = 64):
        super().__init__()
        self.conv1 = SAGEConv(node_feat_dim, hidden)
        self.conv2 = SAGEConv(hidden, hidden)
        self.conv3 = SAGEConv(hidden, out)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.bn2 = nn.BatchNorm1d(hidden)
        # Edge regression head: concat(h_src, h_dst) + edge_features → actual_time
        self.head = nn.Sequential(
            nn.Linear(out * 2 + edge_feat_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x, edge_index, edge_attr):
        h = F.relu(self.bn1(self.conv1(x, edge_index)))
        h = F.relu(self.bn2(self.conv2(h, edge_index)))
        h = self.conv3(h, edge_index)

        src_idx, dst_idx = edge_index
        edge_emb = torch.cat([h[src_idx], h[dst_idx], edge_attr], dim=-1)
        return self.head(edge_emb).squeeze(-1)


# ── Graph Attention Network ───────────────────────────────────────────────────

class GATModel(nn.Module):
    def __init__(self, node_feat_dim: int, edge_feat_dim: int, hidden: int = 64, heads: int = 8, out: int = 64):
        super().__init__()
        self.gat1 = GATConv(node_feat_dim, hidden, heads=heads, dropout=0.2, concat=True)
        self.gat2 = GATConv(hidden * heads, out, heads=1, dropout=0.2, concat=False)
        self.head = nn.Sequential(
            nn.Linear(out * 2 + edge_feat_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.attention_weights = None  # stored after forward for visualisation

    def forward(self, x, edge_index, edge_attr):
        h, attn = self.gat1(x, edge_index, return_attention_weights=True)
        self.attention_weights = attn  # (edge_index, alpha)
        h = F.elu(h)
        h = self.gat2(h, edge_index)

        src_idx, dst_idx = edge_index
        edge_emb = torch.cat([h[src_idx], h[dst_idx], edge_attr], dim=-1)
        return self.head(edge_emb).squeeze(-1)


# ── Training loop ─────────────────────────────────────────────────────────────

def train_gnn(
    model: nn.Module,
    data: "Data",
    epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    val_split: float = 0.2,
    model_name: str = "gnn",
    device: str = "cpu",
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else device)
    model = model.to(device)
    data = data.to(device)

    n_edges = data.edge_index.shape[1]
    perm = torch.randperm(n_edges, device=device)
    val_size = int(n_edges * val_split)
    val_mask  = torch.zeros(n_edges, dtype=torch.bool, device=device)
    val_mask[perm[:val_size]] = True
    train_mask = ~val_mask

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_mae = float("inf")
    best_state = None
    history = {"train_mae": [], "val_mae": [], "val_within15": []}

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        pred = model(data.x, data.edge_index, data.edge_attr)
        loss = F.l1_loss(pred[train_mask], data.y[train_mask])
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(data.x, data.edge_index, data.edge_attr)
            val_mae = F.l1_loss(val_pred[val_mask], data.y[val_mask]).item()
            w15 = within_15pct(data.y[val_mask], val_pred[val_mask])

        history["train_mae"].append(float(loss.item()))
        history["val_mae"].append(val_mae)
        history["val_within15"].append(w15)

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0:
            log.info(f"[{model_name}] Epoch {epoch:3d}/{epochs}  train_MAE={loss.item():.2f}  val_MAE={val_mae:.2f}  within15%={w15:.1f}%")

    model.load_state_dict(best_state)
    torch.save(model.state_dict(), MODEL_DIR / f"{model_name}.pt")
    log.info(f"Best val MAE: {best_val_mae:.3f}  |  Model saved → {MODEL_DIR / model_name}.pt")

    model.eval()
    with torch.no_grad():
        final_pred = model(data.x, data.edge_index, data.edge_attr)
        test_mae = F.l1_loss(final_pred[val_mask], data.y[val_mask]).item()
        test_w15 = within_15pct(data.y[val_mask], final_pred[val_mask])

    return {
        "model": model_name,
        "MAE": round(test_mae, 3),
        "within_15pct": round(test_w15, 2),
        "history": history,
    }


def run(node_features, node_index, edge_df, edge_feature_cols, epochs=100):
    if not _HAS_PYG:
        log.error("PyTorch Geometric required.")
        return []

    data = build_pyg_data(node_features, node_index, edge_df, edge_feature_cols)
    node_dim = data.x.shape[1]
    edge_dim = data.edge_attr.shape[1]

    results = []

    sage = GraphSAGEModel(node_dim, edge_dim)
    results.append(train_gnn(sage, data, epochs=epochs, model_name="GraphSAGE"))

    gat = GATModel(node_dim, edge_dim)
    results.append(train_gnn(gat, data, epochs=epochs, model_name="GAT"))

    return results
