"""
node2vec graph embeddings for the logistics network.

Random-walk based embeddings capture each hub's structural position:
  - Central relay hubs → high-connectivity embedding region
  - Leaf last-mile hubs → peripheral region
  - Bottleneck hubs → isolated cluster

The 128-dim embeddings are used as node features in GraphSAGE / GAT.
"""

import logging
import pickle
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

EMBED_DIM = 128
WALK_LENGTH = 30
NUM_WALKS = 200
P = 1.0   # return parameter (controls BFS-like exploration)
Q = 0.5   # in-out parameter < 1 → DFS-like, explores farther neighbourhoods


def train_node2vec(
    G: nx.DiGraph,
    dimensions: int = EMBED_DIM,
    walk_length: int = WALK_LENGTH,
    num_walks: int = NUM_WALKS,
    p: float = P,
    q: float = Q,
    workers: int = 4,
    seed: int = 42,
) -> dict:
    """
    Returns
    -------
    embeddings : dict  {node_name: np.ndarray(dimensions,)}
    """
    try:
        from node2vec import Node2Vec
    except ImportError:
        log.error("node2vec not installed. Run: pip install node2vec")
        return {}

    log.info(f"Training node2vec: dim={dimensions}, walks={num_walks}, length={walk_length} …")

    # node2vec expects undirected graph for walks
    G_und = G.to_undirected()

    node2vec_model = Node2Vec(
        G_und,
        dimensions=dimensions,
        walk_length=walk_length,
        num_walks=num_walks,
        p=p,
        q=q,
        workers=workers,
        seed=seed,
        quiet=True,
    )
    model = node2vec_model.fit(window=10, min_count=1, batch_words=4)

    embeddings = {node: model.wv[str(node)] for node in G.nodes() if str(node) in model.wv}
    log.info(f"node2vec complete: {len(embeddings)} node embeddings (dim={dimensions})")
    return embeddings


def embeddings_to_df(embeddings: dict, G: nx.DiGraph) -> pd.DataFrame:
    rows = []
    for node, vec in embeddings.items():
        row = {"hub": node}
        row.update({f"n2v_{i}": float(v) for i, v in enumerate(vec)})
        node_data = G.nodes[node]
        row["city"] = node_data.get("city")
        row["hub_type"] = node_data.get("hub_type")
        rows.append(row)
    return pd.DataFrame(rows)


def save_embeddings(embeddings: dict, out_dir: str | Path = "data/processed") -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "node2vec_embeddings.pkl", "wb") as f:
        pickle.dump(embeddings, f)

    # Also save as numpy matrix for PyTorch Geometric
    nodes = list(embeddings.keys())
    matrix = np.stack([embeddings[n] for n in nodes])
    np.save(out_dir / "node2vec_matrix.npy", matrix)
    pd.Series(nodes).to_csv(out_dir / "node2vec_node_index.csv", index=False, header=False)
    log.info(f"node2vec embeddings saved → {out_dir}")


def load_embeddings(out_dir: str | Path = "data/processed") -> dict:
    out_dir = Path(out_dir)
    with open(out_dir / "node2vec_embeddings.pkl", "rb") as f:
        return pickle.load(f)


def get_node_feature_matrix(
    G: nx.DiGraph,
    embeddings: dict,
    extra_features: list[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """
    Combine node2vec embeddings with handcrafted node features.

    Returns
    -------
    X : np.ndarray (n_nodes, embed_dim + n_extra)
    node_order : list of node names (row order)
    """
    node_order = list(G.nodes())
    rows = []
    for node in node_order:
        n2v = embeddings.get(node, np.zeros(EMBED_DIM))
        node_data = G.nodes[node]
        handcrafted = [
            float(node_data.get("avg_dwell", 0.0)),
            float(G.in_degree(node)),
            float(G.out_degree(node)),
            float(node_data.get("trip_volume", 0)),
        ]
        rows.append(np.concatenate([n2v, handcrafted]))
    X = np.array(rows, dtype=np.float32)
    return X, node_order


if __name__ == "__main__":
    from src.graph.builder import load_graph
    G = load_graph()
    embeddings = train_node2vec(G)
    save_embeddings(embeddings)
    X, node_order = get_node_feature_matrix(G, embeddings)
    print(f"Node feature matrix: {X.shape}")
    print(f"First node '{node_order[0]}': embedding norm = {np.linalg.norm(X[0]):.4f}")
