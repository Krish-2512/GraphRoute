"""
Spatio-Temporal Graph Neural Network (ST-GNN) for Multi-Hop Logistics Routes.

Integrates Spatial Graph Message Passing (GraphSAGE) over hub topology
with Temporal Sequential Modeling (BiLSTM) across consecutive trip hops.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.models.gnn_layers import SAGENativeConv, calculate_metrics

log = logging.getLogger(__name__)
MODEL_DIR = Path("data/processed/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)


class SpatioTemporalGNN(nn.Module):
    """
    Spatio-Temporal Graph Neural Network architecture:
    1. Spatial Node/Edge Encoder: 2-layer GraphSAGE embedding network
    2. Temporal Sequence Layer: Bi-directional LSTM across multi-hop trajectory legs
    3. Trajectory Aggregation & Regression Head -> Predicts Full-Trip Actual Travel Time
    """
    def __init__(
        self,
        node_in_dim: int,
        edge_in_dim: int,
        spatial_dim: int = 64,
        temporal_hidden: int = 64,
        num_lstm_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        # Spatial Graph Encoder
        self.conv1 = SAGENativeConv(node_in_dim, spatial_dim)
        self.bn1 = nn.BatchNorm1d(spatial_dim)
        self.conv2 = SAGENativeConv(spatial_dim, spatial_dim)
        self.bn2 = nn.BatchNorm1d(spatial_dim)

        # Hop Feature Fusion (Spatial Hub Embeddings + En-Route Features)
        self.hop_proj = nn.Sequential(
            nn.Linear(spatial_dim * 2 + edge_in_dim, spatial_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Temporal Sequence Recurrence
        self.bilstm = nn.LSTM(
            input_size=spatial_dim,
            hidden_size=temporal_hidden,
            num_layers=num_lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_lstm_layers > 1 else 0.0,
        )

        # Final Trajectory ETA Head
        self.head = nn.Sequential(
            nn.Linear(temporal_hidden * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(
        self,
        node_features: torch.Tensor,
        graph_edge_index: torch.Tensor,
        trajectory_src_indices: torch.Tensor,   # (Batch, Max_Hops)
        trajectory_dst_indices: torch.Tensor,   # (Batch, Max_Hops)
        trajectory_edge_attrs: torch.Tensor,    # (Batch, Max_Hops, edge_dim)
        trajectory_mask: torch.Tensor,          # (Batch, Max_Hops)
    ) -> torch.Tensor:
        # 1. Spatial Message Passing across full supply network
        h = F.relu(self.bn1(self.conv1(node_features, graph_edge_index)))
        h = F.relu(self.bn2(self.conv2(h, graph_edge_index)))  # (N, spatial_dim)

        B, T = trajectory_src_indices.shape

        # 2. Gather spatial embeddings for each hop in batch
        src_emb = h[trajectory_src_indices]  # (B, T, spatial_dim)
        dst_emb = h[trajectory_dst_indices]  # (B, T, spatial_dim)

        hop_features = torch.cat([src_emb, dst_emb, trajectory_edge_attrs], dim=-1)
        hop_encoded = self.hop_proj(hop_features)  # (B, T, spatial_dim)

        # Zero out padding hops
        hop_encoded = hop_encoded * trajectory_mask.unsqueeze(-1)

        # 3. Temporal Sequence Processing
        lstm_out, _ = self.bilstm(hop_encoded)  # (B, T, 2 * temporal_hidden)

        # Masked average pooling over valid hops in trajectory
        mask_expanded = trajectory_mask.unsqueeze(-1)
        sum_pooled = (lstm_out * mask_expanded).sum(dim=1)
        lengths = trajectory_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        route_repr = sum_pooled / lengths  # (B, 2 * temporal_hidden)

        # 4. Regression Output
        out = self.head(route_repr).squeeze(-1)
        return out


class TrajectoryDataset(Dataset):
    def __init__(self, trajectories: List[Dict[str, Any]], max_hops: int = 6, edge_dim: int = 5):
        self.trajectories = trajectories
        self.max_hops = max_hops
        self.edge_dim = edge_dim

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        item = self.trajectories[idx]
        hops = item["hops"]
        num_hops = min(len(hops), self.max_hops)

        src_indices = np.zeros(self.max_hops, dtype=np.int64)
        dst_indices = np.zeros(self.max_hops, dtype=np.int64)
        edge_attrs = np.zeros((self.max_hops, self.edge_dim), dtype=np.float32)
        mask = np.zeros(self.max_hops, dtype=np.float32)

        for t in range(num_hops):
            h = hops[t]
            src_indices[t] = h["src_idx"]
            dst_indices[t] = h["dst_idx"]
            edge_attrs[t] = h["edge_attr"]
            mask[t] = 1.0

        return (
            torch.tensor(src_indices, dtype=torch.long),
            torch.tensor(dst_indices, dtype=torch.long),
            torch.tensor(edge_attrs, dtype=torch.float),
            torch.tensor(mask, dtype=torch.float),
            torch.tensor(item["actual_time"], dtype=torch.float),
        )


def build_trajectories_from_df(
    df: pd.DataFrame,
    node_to_idx: Dict[str, int],
    edge_cols: List[str],
    max_hops: int = 6,
) -> List[Dict[str, Any]]:
    """Construct multi-leg trip sequences from grouped DataFrame records."""
    trajectories = []
    uuid_col = next((c for c in df.columns if "uuid" in c.lower() or "trip_id" in c.lower()), None)

    if uuid_col:
        grouped = df.groupby(uuid_col)
    else:
        # Fallback: Treat each 1-hop trip as a length-1 trajectory
        grouped = [(i, df.iloc[[i]]) for i in range(len(df))]

    for _, grp in grouped:
        hops = []
        for _, row in grp.iterrows():
            src = row.get("source_name")
            dst = row.get("destination_name")
            if src not in node_to_idx or dst not in node_to_idx:
                continue
            e_feat = [float(row.get(c, 0.0)) for c in edge_cols]
            hops.append({
                "src_idx": node_to_idx[src],
                "dst_idx": node_to_idx[dst],
                "edge_attr": np.array(e_feat, dtype=np.float32),
            })
        if hops:
            total_actual = float(grp["actual_time"].sum()) if "actual_time" in grp else float(grp["actual_time"].iloc[0])
            trajectories.append({"hops": hops[:max_hops], "actual_time": total_actual})
    return trajectories


def train_st_gnn(
    model: SpatioTemporalGNN,
    node_features: torch.Tensor,
    graph_edge_index: torch.Tensor,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 50,
    lr: float = 1e-3,
    device: str = "cpu",
) -> Dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else device)
    model = model.to(device)
    node_features = node_features.to(device)
    graph_edge_index = graph_edge_index.to(device)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    best_val_mae = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for src_idx, dst_idx, e_attr, mask, y in train_loader:
            src_idx, dst_idx = src_idx.to(device), dst_idx.to(device)
            e_attr, mask, y = e_attr.to(device), mask.to(device), y.to(device)

            optimizer.zero_grad()
            pred = model(node_features, graph_edge_index, src_idx, dst_idx, e_attr, mask)
            loss = F.l1_loss(pred, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_preds, val_trues = [], []
        with torch.no_grad():
            for src_idx, dst_idx, e_attr, mask, y in val_loader:
                src_idx, dst_idx = src_idx.to(device), dst_idx.to(device)
                e_attr, mask = e_attr.to(device), mask.to(device)
                pred = model(node_features, graph_edge_index, src_idx, dst_idx, e_attr, mask)
                val_preds.extend(pred.cpu().numpy().tolist())
                val_trues.extend(y.numpy().tolist())

        val_mae = float(np.mean(np.abs(np.array(val_preds) - np.array(val_trues))))
        scheduler.step(val_mae)

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == epochs:
            log.info(f"[ST-GNN] Epoch {epoch}/{epochs} | Train MAE: {np.mean(train_losses):.2f} | Val MAE: {val_mae:.2f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), MODEL_DIR / "st_gnn.pt")

    metrics = calculate_metrics(np.array(val_trues), np.array(val_preds), model_name="ST-GNN (Spatio-Temporal)")
    return metrics
