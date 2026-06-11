"""
LSTM-based sequential ETA prediction for multi-hop delivery routes.

Each route = sequence of segment feature vectors:
  [osrm_time_i, osrm_distance_i, is_ftl_i, time_of_day_enc_i, dwell_proxy_i,
   delay_ratio_history_i, hub_betweenness_i]

LSTM captures: "if hub_1 is congested, hub_2 downstream will also be delayed"
(temporal/sequential dependency across hops — something XGBoost misses).

Target: cumulative actual_time for the full route.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

log = logging.getLogger(__name__)

MODEL_DIR = Path("data/processed/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

SEQ_FEATURES = [
    "osrm_time",
    "osrm_distance",
    "is_ftl",
    "time_of_day_enc",
    "dwell_time_proxy",
    "corridor_mean_delay",
    "src_betweenness",
]
TARGET = "actual_time"
MAX_HOPS = 5   # pad/truncate all sequences to this length


# ── Dataset ────────────────────────────────────────────────────────────────────

class RouteSequenceDataset(Dataset):
    """
    Groups trips by route_uuid (if available) or creates synthetic 2-hop sequences
    by pairing consecutive legs that share a hub.
    """

    def __init__(self, df: pd.DataFrame, max_hops: int = MAX_HOPS):
        self.sequences = []
        self.targets = []
        self.max_hops = max_hops
        self._build(df)

    def _build(self, df: pd.DataFrame):
        feat_cols = [c for c in SEQ_FEATURES if c in df.columns]

        # If route_uuid exists, group real multi-hop sequences
        uuid_col = next((c for c in df.columns if "uuid" in c.lower()), None)
        if uuid_col:
            for _, grp in df.groupby(uuid_col):
                grp = grp.sort_values("time_of_day_enc") if "time_of_day_enc" in grp else grp
                feats = grp[feat_cols].fillna(0).values
                target = float(grp[TARGET].sum()) if TARGET in grp else 0.0
                self.sequences.append(feats)
                self.targets.append(target)
        else:
            # Synthetic 2-hop: pair each trip with its downstream leg (same destination→source)
            dest_map = df.set_index("source_name")[feat_cols + [TARGET]].to_dict("index") \
                       if "source_name" in df.columns else {}
            for _, row in df.iterrows():
                feats = [row.get(c, 0.0) for c in feat_cols]
                target = float(row.get(TARGET, 0.0))
                # Try to find a downstream leg
                dst = row.get("destination_name", "")
                if dst in dest_map:
                    downstream = [dest_map[dst].get(c, 0.0) for c in feat_cols]
                    seq = np.array([feats, downstream], dtype=np.float32)
                    target += float(dest_map[dst].get(TARGET, 0.0))
                else:
                    seq = np.array([feats], dtype=np.float32)
                self.sequences.append(seq)
                self.targets.append(target)

    def _pad(self, seq: np.ndarray) -> np.ndarray:
        L, F = seq.shape
        if L >= self.max_hops:
            return seq[:self.max_hops]
        pad = np.zeros((self.max_hops - L, F), dtype=np.float32)
        return np.vstack([seq, pad])

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self._pad(np.array(self.sequences[idx], dtype=np.float32))
        return torch.tensor(seq), torch.tensor(self.targets[idx], dtype=torch.float)


# ── Model ──────────────────────────────────────────────────────────────────────

class ETALSTMModel(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 128, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        h_last = h_n[-1]  # last layer hidden state
        return self.head(h_last).squeeze(-1)


# ── Training ───────────────────────────────────────────────────────────────────

def train(
    df: pd.DataFrame,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    hidden: int = 128,
    num_layers: int = 2,
) -> dict:
    feat_cols = [c for c in SEQ_FEATURES if c in df.columns]
    input_dim = len(feat_cols)
    if input_dim == 0:
        log.error("No sequence features available.")
        return {}

    # Normalise
    means = df[feat_cols].mean()
    stds  = df[feat_cols].std().replace(0, 1)
    df_norm = df.copy()
    df_norm[feat_cols] = (df[feat_cols] - means) / stds
    target_mean = float(df[TARGET].mean())
    target_std  = float(df[TARGET].std()) or 1.0
    df_norm[TARGET] = (df[TARGET] - target_mean) / target_std

    dataset = RouteSequenceDataset(df_norm)
    n_val = max(1, int(len(dataset) * 0.2))
    n_train = len(dataset) - n_val
    train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size)

    model = ETALSTMModel(input_dim=input_dim, hidden=hidden, num_layers=num_layers)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_val_mae = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = F.l1_loss(pred, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_preds, val_trues = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                val_preds.append(model(X_batch))
                val_trues.append(y_batch)
        val_pred = torch.cat(val_preds) * target_std + target_mean
        val_true = torch.cat(val_trues) * target_std + target_mean
        val_mae = F.l1_loss(val_pred, val_true).item()
        scheduler.step(val_mae)

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0:
            log.info(f"[LSTM] Epoch {epoch:3d}/{epochs}  train_MAE={np.mean(train_losses)*target_std:.2f}  val_MAE={val_mae:.2f}")

    model.load_state_dict(best_state)
    torch.save({"state_dict": model.state_dict(), "target_mean": target_mean,
                "target_std": target_std, "feat_means": means.to_dict(),
                "feat_stds": stds.to_dict()},
               MODEL_DIR / "lstm_eta.pt")

    # Final metrics
    model.eval()
    val_pred_np = val_pred.numpy()
    val_true_np = val_true.numpy()
    w15 = float(np.mean(np.abs(val_pred_np - val_true_np) / (val_true_np + 1e-8) <= 0.15) * 100)

    log.info(f"[LSTM] Best val MAE={best_val_mae:.3f}  within15%={w15:.1f}%")
    return {"model": "LSTM", "MAE": round(best_val_mae, 3), "within_15pct": round(w15, 2)}


if __name__ == "__main__":
    # Quick smoke test with synthetic data
    rng = np.random.default_rng(42)
    n = 2000
    sample_df = pd.DataFrame({
        "source_name":       [f"Hub_{i%20}" for i in range(n)],
        "destination_name":  [f"Hub_{(i+1)%20}" for i in range(n)],
        "osrm_time":         rng.uniform(60, 900, n),
        "osrm_distance":     rng.uniform(50, 1500, n),
        "is_ftl":            rng.integers(0, 2, n),
        "time_of_day_enc":   rng.integers(0, 4, n),
        "dwell_time_proxy":  rng.uniform(0, 30, n),
        "corridor_mean_delay": rng.uniform(1.0, 1.8, n),
        "src_betweenness":   rng.uniform(0, 0.5, n),
        "actual_time":       rng.uniform(80, 1200, n),
    })
    result = train(sample_df, epochs=20)
    print(result)
