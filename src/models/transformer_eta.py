"""
Temporal Transformer for time-of-day-aware ETA prediction.

Advantages over LSTM:
  * Positional encoding = time-of-day + day-of-week (not just hop index)
  * Multi-head self-attention captures long-range dependencies across hops
  * Parallelisable training (faster than LSTM on GPUs)

Architecture:
  Input: sequence of hop feature vectors (same as LSTM)
  Positional encoding: sinusoidal over hop index + time-of-day embedding
  Transformer encoder: 4 layers, 8 heads, d_model=128
  Regression head: CLS token → MLP → actual_time
"""

import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

from src.models.lstm_eta import RouteSequenceDataset, SEQ_FEATURES, TARGET

log = logging.getLogger(__name__)

MODEL_DIR = Path("data/processed/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ── Positional Encoding ────────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 20, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# ── Time-of-day Embedding ──────────────────────────────────────────────────────

class TimeEmbedding(nn.Module):
    """Learns a separate embedding for each time-of-day bucket (0–3)."""
    def __init__(self, d_model: int, n_bins: int = 4):
        super().__init__()
        self.emb = nn.Embedding(n_bins + 1, d_model, padding_idx=n_bins)

    def forward(self, time_of_day_idx: torch.LongTensor) -> torch.Tensor:
        return self.emb(time_of_day_idx.clamp(0, self.emb.num_embeddings - 2))


# ── Model ──────────────────────────────────────────────────────────────────────

class ETATransformerModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_seq_len: int = 10,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_seq_len, dropout=dropout)
        self.time_emb = TimeEmbedding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-LayerNorm for training stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # CLS token aggregation
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor, time_idx: torch.Tensor | None = None) -> torch.Tensor:
        B, L, _ = x.shape
        h = self.input_proj(x)                   # (B, L, d_model)
        h = self.pos_enc(h)

        if time_idx is not None:
            h = h + self.time_emb(time_idx)      # (B, L, d_model)

        cls = self.cls_token.expand(B, -1, -1)   # (B, 1, d_model)
        h = torch.cat([cls, h], dim=1)           # (B, L+1, d_model)

        h = self.transformer(h)
        cls_out = h[:, 0]                        # CLS representation
        return self.head(cls_out).squeeze(-1)


# ── Training ───────────────────────────────────────────────────────────────────

def train(
    df: pd.DataFrame,
    epochs: int = 60,
    batch_size: int = 64,
    lr: float = 3e-4,
    d_model: int = 128,
    nhead: int = 8,
    num_layers: int = 4,
) -> dict:
    feat_cols = [c for c in SEQ_FEATURES if c in df.columns]
    input_dim = len(feat_cols)
    if input_dim == 0:
        log.error("No sequence features found.")
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
    n_val   = max(1, int(len(dataset) * 0.2))
    n_train = len(dataset) - n_val
    train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size)

    model = ETATransformerModel(input_dim=input_dim, d_model=d_model, nhead=nhead,
                                 num_encoder_layers=num_layers)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = OneCycleLR(optimizer, max_lr=lr, epochs=epochs,
                           steps_per_epoch=len(train_loader))

    best_val_mae = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            # Extract time_of_day from feature index 3 (time_of_day_enc)
            tod_idx = None
            if "time_of_day_enc" in feat_cols:
                ti = feat_cols.index("time_of_day_enc")
                tod_idx = X_batch[:, :, ti].long().clamp(0, 3)
            pred = model(X_batch, time_idx=tod_idx)
            loss = F.l1_loss(pred, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            train_losses.append(loss.item())

        model.eval()
        val_preds, val_trues = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                tod_idx = None
                if "time_of_day_enc" in feat_cols:
                    ti = feat_cols.index("time_of_day_enc")
                    tod_idx = X_batch[:, :, ti].long().clamp(0, 3)
                val_preds.append(model(X_batch, time_idx=tod_idx))
                val_trues.append(y_batch)

        val_pred = torch.cat(val_preds) * target_std + target_mean
        val_true = torch.cat(val_trues) * target_std + target_mean
        val_mae  = F.l1_loss(val_pred, val_true).item()

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0:
            log.info(f"[Transformer] Epoch {epoch:3d}/{epochs}  "
                     f"train_MAE={np.mean(train_losses)*target_std:.2f}  val_MAE={val_mae:.2f}")

    model.load_state_dict(best_state)
    torch.save({"state_dict": model.state_dict(), "target_mean": target_mean,
                "target_std": target_std, "feat_means": means.to_dict(),
                "feat_stds": stds.to_dict()},
               MODEL_DIR / "transformer_eta.pt")

    val_pred_np = val_pred.detach().numpy()
    val_true_np = val_true.detach().numpy()
    w15 = float(np.mean(np.abs(val_pred_np - val_true_np) / (val_true_np + 1e-8) <= 0.15) * 100)

    log.info(f"[Transformer] Best val MAE={best_val_mae:.3f}  within15%={w15:.1f}%")
    return {"model": "Transformer", "MAE": round(best_val_mae, 3), "within_15pct": round(w15, 2)}


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    n = 2000
    sample_df = pd.DataFrame({
        "source_name":         [f"Hub_{i%20}" for i in range(n)],
        "destination_name":    [f"Hub_{(i+1)%20}" for i in range(n)],
        "osrm_time":           rng.uniform(60, 900, n),
        "osrm_distance":       rng.uniform(50, 1500, n),
        "is_ftl":              rng.integers(0, 2, n),
        "time_of_day_enc":     rng.integers(0, 4, n),
        "dwell_time_proxy":    rng.uniform(0, 30, n),
        "corridor_mean_delay": rng.uniform(1.0, 1.8, n),
        "src_betweenness":     rng.uniform(0, 0.5, n),
        "actual_time":         rng.uniform(80, 1200, n),
    })
    result = train(sample_df, epochs=20)
    print(result)
