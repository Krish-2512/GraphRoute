"""
Semantic route embeddings using sentence-transformers.

Each corridor is described as a natural-language string:
  "Delhi Gateway Hub to Mumbai Fulfillment Center via FTL in the morning"

A 384-dim embedding from all-MiniLM-L6-v2 is produced per corridor.
These embeddings are used as edge features in GraphSAGE / GAT.

Key insight: corridors with similar semantic descriptions share embedding
space — enabling zero-shot delay estimation for new or rare routes.
"""

import logging
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

MODEL_NAME = "all-MiniLM-L6-v2"
EMBED_DIM = 384


@lru_cache(maxsize=1)
def _load_model():
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(MODEL_NAME)
        log.info(f"SentenceTransformer loaded: {MODEL_NAME}")
        return model
    except ImportError:
        log.warning("sentence-transformers not installed. Using zero embeddings.")
        return None


def _build_route_string(
    src: str,
    dst: str,
    route_type: str = "",
    time_of_day: str = "",
    src_city: str = "",
    dst_city: str = "",
) -> str:
    src_label = src_city if src_city else src
    dst_label = dst_city if dst_city else dst
    parts = [f"{src_label} to {dst_label}"]
    if route_type:
        parts.append(f"via {route_type}")
    if time_of_day:
        parts.append(f"in the {time_of_day}")
    return " ".join(parts)


def embed_routes(route_strings: list[str], batch_size: int = 256) -> np.ndarray:
    model = _load_model()
    if model is None:
        return np.zeros((len(route_strings), EMBED_DIM), dtype=np.float32)
    embeddings = model.encode(
        route_strings,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    return embeddings.astype(np.float32)


def build_corridor_embeddings(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Returns:
      corridor_df - one row per unique (source_name, destination_name, route_type, time_of_day)
      embeddings  - shape (n_corridors, EMBED_DIM)
    """
    group_cols = ["source_name", "destination_name", "route_type", "time_of_day"]
    available = [c for c in group_cols if c in df.columns]

    corridor_df = df[available].drop_duplicates().reset_index(drop=True)

    route_strings = []
    for _, row in corridor_df.iterrows():
        rs = _build_route_string(
            src=row.get("source_name", ""),
            dst=row.get("destination_name", ""),
            route_type=row.get("route_type", ""),
            time_of_day=row.get("time_of_day", ""),
            src_city=row.get("src_city", ""),
            dst_city=row.get("dst_city", ""),
        )
        route_strings.append(rs)

    log.info(f"Embedding {len(route_strings)} unique corridor descriptions …")
    embeddings = embed_routes(route_strings)
    log.info(f"Embeddings shape: {embeddings.shape}")
    return corridor_df, embeddings


def save_embeddings(
    corridor_df: pd.DataFrame,
    embeddings: np.ndarray,
    out_dir: str | Path = "data/processed",
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    corridor_df.to_parquet(out_dir / "corridor_index.parquet", index=False)
    np.save(out_dir / "corridor_embeddings.npy", embeddings)
    log.info(f"Saved corridor_index.parquet + corridor_embeddings.npy → {out_dir}")


def load_embeddings(
    out_dir: str | Path = "data/processed",
) -> tuple[pd.DataFrame, np.ndarray]:
    out_dir = Path(out_dir)
    corridor_df = pd.read_parquet(out_dir / "corridor_index.parquet")
    embeddings = np.load(out_dir / "corridor_embeddings.npy")
    return corridor_df, embeddings


def get_embedding_for_corridor(
    src: str,
    dst: str,
    route_type: str,
    corridor_df: pd.DataFrame,
    embeddings: np.ndarray,
) -> Optional[np.ndarray]:
    mask = (
        (corridor_df["source_name"] == src) &
        (corridor_df["destination_name"] == dst) &
        (corridor_df["route_type"] == route_type)
    )
    idx = corridor_df[mask].index
    if len(idx) == 0:
        return None
    return embeddings[idx[0]]


if __name__ == "__main__":
    sample_routes = [
        "Delhi Gateway Hub to Mumbai Fulfillment Center via FTL in the morning",
        "Bengaluru Whitefield Hub to Chennai Sorting Center via Carting in the evening",
        "Kolkata Dankuni Hub to Bhubaneswar Hub via Carting at night",
    ]
    embs = embed_routes(sample_routes)
    print(f"Shape: {embs.shape}")
    from numpy.linalg import norm
    cos = np.dot(embs[0], embs[1]) / (norm(embs[0]) * norm(embs[1]))
    print(f"Cosine similarity (Delhi-Mumbai vs Bengaluru-Chennai): {cos:.4f}")
