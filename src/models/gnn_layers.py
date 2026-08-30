"""
Native PyTorch Graph Neural Network Layers and Architectures.

Provides PyTorch implementations of GraphSAGE and Graph Attention Networks (GAT)
without requiring torch_geometric C++ extensions or external binary wheels.
"""

import logging
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

log = logging.getLogger(__name__)
MODEL_DIR = Path("data/processed/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Native GraphSAGE Convolution Layer
# ---------------------------------------------------------------------------

class SAGENativeConv(nn.Module):
    """
    Inductive GraphSAGE Convolution with Mean Aggregation:
      h_v = ReLU( W_self * h_v + W_neigh * Mean_{u in N(v)}(h_u) + b )
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear_self = nn.Linear(in_features, out_features, bias=False)
        self.linear_neigh = nn.Linear(in_features, out_features, bias=False)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear_self.weight)
        nn.init.xavier_uniform_(self.linear_neigh.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, num_nodes: Optional[int] = None) -> torch.Tensor:
        """
        x: (N, in_features)
        edge_index: (2, E) where edge_index[0] = source, edge_index[1] = destination
        """
        N = x.size(0) if num_nodes is None else num_nodes
        src, dst = edge_index[0], edge_index[1]

        # Aggregate incoming neighbor features to destination nodes using scatter mean
        neigh_sum = torch.zeros(N, self.in_features, device=x.device, dtype=x.dtype)
        neigh_count = torch.zeros(N, 1, device=x.device, dtype=x.dtype)

        neigh_sum.index_add_(0, dst, x[src])
        neigh_count.index_add_(0, dst, torch.ones((src.size(0), 1), device=x.device, dtype=x.dtype))

        neigh_mean = neigh_sum / (neigh_count + 1e-8)

        out = self.linear_self(x) + self.linear_neigh(neigh_mean)
        if self.bias is not None:
            out = out + self.bias
        return out


# ---------------------------------------------------------------------------
# 2. Native Graph Attention Convolution (GAT) Layer
# ---------------------------------------------------------------------------

class GATNativeConv(nn.Module):
    """
    Multi-Head Graph Attention Network Layer with LeakyReLU and Softmax coefficients.
    """
    def __init__(self, in_features: int, out_features: int, heads: int = 4,
                 dropout: float = 0.1, negative_slope: float = 0.2):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.heads = heads
        self.dropout = nn.Dropout(dropout)
        self.negative_slope = negative_slope

        self.W = nn.Linear(in_features, heads * out_features, bias=False)
        self.a_src = nn.Parameter(torch.empty(1, heads, out_features))
        self.a_dst = nn.Parameter(torch.empty(1, heads, out_features))
        self.bias = nn.Parameter(torch.zeros(heads * out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        N = x.size(0)
        src, dst = edge_index[0], edge_index[1]

        # Linear projection: (N, H, F_out)
        h = self.W(x).view(N, self.heads, self.out_features)

        # Attention scores: e_{ij} = LeakyReLU( a_src^T h_i + a_dst^T h_j )
        alpha_src = (h * self.a_src).sum(dim=-1)  # (N, H)
        alpha_dst = (h * self.a_dst).sum(dim=-1)  # (N, H)

        edge_attn = alpha_src[src] + alpha_dst[dst]  # (E, H)
        edge_attn = F.leaky_relu(edge_attn, self.negative_slope)

        # Softmax over incoming edges per destination node
        exp_attn = torch.exp(edge_attn - edge_attn.max())
        sum_exp = torch.zeros(N, self.heads, device=x.device)
        sum_exp.index_add_(0, dst, exp_attn)
        alpha = exp_attn / (sum_exp[dst] + 1e-8)  # (E, H)
        alpha_drop = self.dropout(alpha)

        # Weighted message aggregation: (E, H, F_out)
        weighted_msg = h[src] * alpha_drop.unsqueeze(-1)
        out = torch.zeros(N, self.heads, self.out_features, device=x.device)
        out.index_add_(0, dst, weighted_msg)

        out = out.view(N, self.heads * self.out_features) + self.bias
        return out, alpha


# ---------------------------------------------------------------------------
# 3. Complete Deep Learning Models for Corridor ETA Regression
# ---------------------------------------------------------------------------

class GraphSAGEETAModel(nn.Module):
    """
    Inductive GraphSAGE Network for Corridor ETA Regression.
    Aggregates node embeddings -> Edge Concat Head -> Output actual_time.
    """
    def __init__(self, node_in_dim: int, edge_in_dim: int, hidden_dim: int = 128, out_dim: int = 64):
        super().__init__()
        self.conv1 = SAGENativeConv(node_in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = SAGENativeConv(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.conv3 = SAGENativeConv(hidden_dim, out_dim)

        self.edge_mlp = nn.Sequential(
            nn.Linear(out_dim * 2 + edge_in_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.bn1(self.conv1(x, edge_index)))
        h = F.relu(self.bn2(self.conv2(h, edge_index)))
        h = self.conv3(h, edge_index)

        src, dst = edge_index[0], edge_index[1]
        edge_repr = torch.cat([h[src], h[dst], edge_attr], dim=-1)
        return self.edge_mlp(edge_repr).squeeze(-1)


class GATETAModel(nn.Module):
    """
    Interpretable Graph Attention Network for ETA Regression.
    """
    def __init__(self, node_in_dim: int, edge_in_dim: int, hidden_dim: int = 64, heads: int = 4, out_dim: int = 64):
        super().__init__()
        self.gat1 = GATNativeConv(node_in_dim, hidden_dim, heads=heads, dropout=0.2)
        self.gat2 = GATNativeConv(hidden_dim * heads, out_dim, heads=1, dropout=0.2)

        self.edge_mlp = nn.Sequential(
            nn.Linear(out_dim * 2 + edge_in_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.last_attention_weights = None

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        h, attn = self.gat1(x, edge_index)
        self.last_attention_weights = attn
        h = F.elu(h)
        h, _ = self.gat2(h, edge_index)

        src, dst = edge_index[0], edge_index[1]
        edge_repr = torch.cat([h[src], h[dst], edge_attr], dim=-1)
        return self.edge_mlp(edge_repr).squeeze(-1)


# ---------------------------------------------------------------------------
# 4. Evaluation and Training Utilities
# ---------------------------------------------------------------------------

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, model_name: str = "") -> Dict[str, Any]:
    mae = float(np.mean(np.abs(y_pred - y_true)))
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    within_15 = float(np.mean(np.abs(y_pred - y_true) / (np.abs(y_true) + 1e-6) <= 0.15) * 100.0)
    mape = float(np.mean(np.abs(y_pred - y_true) / (np.abs(y_true) + 1e-6)) * 100.0)
    return {
        "model": model_name,
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "within_15pct": round(within_15, 2),
        "MAPE": round(mape, 2),
    }


def train_gnn_model(
    model: nn.Module,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    y: torch.Tensor,
    epochs: int = 80,
    lr: float = 1e-3,
    val_split: float = 0.2,
    model_name: str = "GraphSAGE",
    device: str = "cpu",
) -> Dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else device)
    model = model.to(device)
    x = x.to(device)
    edge_index = edge_index.to(device)
    edge_attr = edge_attr.to(device)
    y = y.to(device)

    num_edges = edge_index.size(1)
    perm = torch.randperm(num_edges, device=device)
    val_size = int(num_edges * val_split)
    val_idx = perm[:val_size]
    train_idx = perm[val_size:]

    y_mean = y[train_idx].mean()
    y_std = y[train_idx].std() + 1e-6
    y_norm = (y - y_mean) / y_std

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_mae = float("inf")
    best_state = None
    history = {"train_loss": [], "val_mae": []}

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        preds = model(x, edge_index, edge_attr)
        loss = F.l1_loss(preds[train_idx], y_norm[train_idx])
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            preds_norm = model(x, edge_index, edge_attr)
            val_preds_unscaled = preds_norm[val_idx] * y_std + y_mean
            val_mae = F.l1_loss(val_preds_unscaled, y[val_idx]).item()

        history["train_loss"].append(float(loss.item()))
        history["val_mae"].append(val_mae)

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 20 == 0 or epoch == epochs:
            log.info(f"[{model_name}] Epoch {epoch}/{epochs} | Train Loss (norm): {loss.item():.3f} | Val MAE: {val_mae:.2f} min")

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save({
        "state_dict": model.state_dict(),
        "y_mean": float(y_mean.cpu().item()),
        "y_std": float(y_std.cpu().item()),
    }, MODEL_DIR / f"{model_name}.pt")

    model.eval()
    with torch.no_grad():
        final_norm = model(x, edge_index, edge_attr)[val_idx]
        final_preds = (final_norm * y_std + y_mean).cpu().numpy()
        final_trues = y[val_idx].cpu().numpy()

    metrics = calculate_metrics(final_trues, final_preds, model_name=model_name)
    metrics["history"] = history
    return metrics


def run(features_path: str = "data/processed/features.parquet", epochs: int = 60) -> List[Dict[str, Any]]:
    import sys
    import json
    root_dir = Path(__file__).resolve().parent.parent.parent
    if str(root_dir) not in sys.path:
        sys.path.insert(0, str(root_dir))

    log.info("Preparing data tensors for PyTorch Graph Neural Networks...")
    df = pd.read_parquet(features_path)
    
    # Unique nodes
    nodes = sorted(list(set(df["source_name"].dropna().unique()).union(set(df["destination_name"].dropna().unique()))))
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    num_nodes = len(nodes)

    # Node features: [out_degree, in_degree, betweenness, pagerank, dwell]
    centrality_df = pd.read_csv("data/processed/hub_centrality.csv") if Path("data/processed/hub_centrality.csv").exists() else pd.DataFrame()
    cent_dict = centrality_df.set_index("hub").to_dict("index") if not centrality_df.empty else {}

    node_feats = []
    for n in nodes:
        c_info = cent_dict.get(n, {})
        node_feats.append([
            float(c_info.get("out_degree", 1)),
            float(c_info.get("in_degree", 1)),
            float(c_info.get("betweenness_centrality", 0.0)),
            float(c_info.get("pagerank", 0.01)),
            float(c_info.get("avg_dwell_min", 20.0)),
        ])
    x = torch.tensor(node_feats, dtype=torch.float)
    # Standardize node features
    x = (x - x.mean(dim=0, keepdim=True)) / (x.std(dim=0, keepdim=True) + 1e-6)

    # Edge features & targets: Sample unique corridor/trip records
    edge_df = df.dropna(subset=["source_name", "destination_name", "actual_time"]).copy()
    edge_cols = ["osrm_time", "osrm_distance", "is_ftl", "time_of_day_enc", "dwell_time_proxy"]
    available_edge_cols = [c for c in edge_cols if c in edge_df.columns]

    src_idx = [node_to_idx[s] for s in edge_df["source_name"]]
    dst_idx = [node_to_idx[d] for d in edge_df["destination_name"]]
    edge_index = torch.tensor([src_idx, dst_idx], dtype=torch.long)

    e_feats = edge_df[available_edge_cols].fillna(0).values
    edge_attr = torch.tensor(e_feats, dtype=torch.float)
    # Standardize edge features
    edge_attr = (edge_attr - edge_attr.mean(dim=0, keepdim=True)) / (edge_attr.std(dim=0, keepdim=True) + 1e-6)

    y = torch.tensor(edge_df["actual_time"].values, dtype=torch.float)

    node_in_dim = x.size(1)
    edge_in_dim = edge_attr.size(1)

    results = []
    log.info(f"Training GraphSAGE on {num_nodes} nodes, {edge_index.size(1)} trip edges...")
    sage = GraphSAGEETAModel(node_in_dim=node_in_dim, edge_in_dim=edge_in_dim, hidden_dim=128, out_dim=64)
    sage_res = train_gnn_model(sage, x, edge_index, edge_attr, y, epochs=epochs, model_name="GraphSAGE")
    results.append(sage_res)

    log.info("Training Graph Attention Network (GAT)...")
    gat = GATETAModel(node_in_dim=node_in_dim, edge_in_dim=edge_in_dim, hidden_dim=64, heads=4, out_dim=64)
    gat_res = train_gnn_model(gat, x, edge_index, edge_attr, y, epochs=epochs, model_name="GAT")
    results.append(gat_res)

    out_path = Path("data/processed/gnn_benchmark_results.json")
    clean_results = [{k: v for k, v in r.items() if k != "history"} for r in results]
    with open(out_path, "w") as f:
        json.dump(clean_results, f, indent=2)
    log.info(f"GNN benchmark saved → {out_path}")
    return results


if __name__ == "__main__":
    run(epochs=50)

