"""
Graph analytics: centrality measures, bottleneck detection, community detection.

Produces the ranked bottleneck table that feeds the strategy memo and dashboard.
"""

import logging
from pathlib import Path
from typing import Optional

import networkx as nx
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

try:
    import community as community_louvain  # python-louvain
    _HAS_LOUVAIN = True
except ImportError:
    _HAS_LOUVAIN = False
    log.warning("python-louvain not installed. Community detection will be skipped.")

CHRONIC_DELAY_THRESHOLD = 1.20  # edges with median_delay_ratio > this are "chronic"


def compute_centrality(G: nx.DiGraph) -> pd.DataFrame:
    """Compute multiple centrality metrics and return a node-level DataFrame."""
    log.info("Computing centrality metrics …")

    # Betweenness centrality — most important for bottleneck identification
    bc = nx.betweenness_centrality(G, weight="weight", normalized=True)

    # PageRank — systemically critical nodes (even low-degree can be critical)
    pr = nx.pagerank(G, weight="weight")

    # In/out degree
    in_deg  = dict(G.in_degree())
    out_deg = dict(G.out_degree())

    # Clustering coefficient (undirected proxy)
    G_undirected = G.to_undirected()
    cc = nx.clustering(G_undirected)

    rows = []
    for node in G.nodes():
        node_data = G.nodes[node]
        rows.append({
            "hub": node,
            "city": node_data.get("city"),
            "hub_type": node_data.get("hub_type"),
            "avg_dwell_min": node_data.get("avg_dwell", 0.0),
            "trip_volume": node_data.get("trip_volume", 0),
            "betweenness_centrality": bc.get(node, 0.0),
            "pagerank": pr.get(node, 0.0),
            "in_degree": in_deg.get(node, 0),
            "out_degree": out_deg.get(node, 0),
            "clustering_coeff": cc.get(node, 0.0),
        })

    df = pd.DataFrame(rows).sort_values("betweenness_centrality", ascending=False)
    return df


def compute_sla_breach_contribution(
    G: nx.DiGraph,
    centrality_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Estimate each hub's SLA breach contribution:
      contribution = betweenness_centrality × avg_outgoing_delay_ratio × trip_volume_share
    """
    total_volume = centrality_df["trip_volume"].sum() or 1

    contributions = []
    for _, row in centrality_df.iterrows():
        hub = row["hub"]
        # Average delay ratio of outgoing edges
        out_edges = list(G.out_edges(hub, data=True))
        if out_edges:
            avg_out_delay = np.mean([d.get("median_delay_ratio", 1.0) for _, _, d in out_edges])
        else:
            avg_out_delay = 1.0

        vol_share = row["trip_volume"] / total_volume
        score = row["betweenness_centrality"] * (avg_out_delay - 1.0) * (1 + vol_share)
        contributions.append(score)

    centrality_df = centrality_df.copy()
    centrality_df["sla_breach_score"] = contributions
    total_score = centrality_df["sla_breach_score"].sum() or 1
    centrality_df["sla_breach_pct"] = (
        centrality_df["sla_breach_score"] / total_score * 100
    ).round(2)
    return centrality_df.sort_values("sla_breach_score", ascending=False)


def get_chronic_corridors(G: nx.DiGraph) -> pd.DataFrame:
    """Corridors where actual time exceeds OSRM by > 20%."""
    rows = []
    for src, dst, data in G.edges(data=True):
        if data.get("median_delay_ratio", 1.0) > CHRONIC_DELAY_THRESHOLD:
            rows.append({
                "source": src,
                "destination": dst,
                "route_type": data.get("route_type"),
                "median_delay_ratio": data.get("median_delay_ratio"),
                "volume": data.get("volume", 0),
                "osrm_distance": data.get("osrm_distance"),
                "is_chronic": True,
            })
    df = pd.DataFrame(rows).sort_values("median_delay_ratio", ascending=False)
    log.info(f"Chronic corridors (>{CHRONIC_DELAY_THRESHOLD:.0%} over OSRM): {len(df)}")
    return df


def detect_communities(G: nx.DiGraph) -> dict:
    """Louvain community detection on undirected projection."""
    if not _HAS_LOUVAIN:
        return {}
    G_und = G.to_undirected()
    partition = community_louvain.best_partition(G_und)
    n_communities = len(set(partition.values()))
    log.info(f"Louvain detected {n_communities} communities.")
    return partition


def shortest_path_delay(G: nx.DiGraph, src: str, dst: str) -> Optional[float]:
    """Cumulative delay ratio along the shortest (by weight) path."""
    try:
        path = nx.shortest_path(G, src, dst, weight="weight")
        total = sum(G[path[i]][path[i+1]]["median_delay_ratio"] for i in range(len(path)-1))
        return total
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None


def bottleneck_report(G: nx.DiGraph, top_n: int = 10) -> pd.DataFrame:
    """Full bottleneck analysis pipeline → returns ranked hub table."""
    centrality_df = compute_centrality(G)
    centrality_df = compute_sla_breach_contribution(G, centrality_df)
    return centrality_df.head(top_n)


def save_reports(centrality_df: pd.DataFrame, chronic_df: pd.DataFrame, out_dir: str | Path = "data/processed") -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    centrality_df.to_csv(out_dir / "hub_centrality.csv", index=False)
    chronic_df.to_csv(out_dir / "chronic_corridors.csv", index=False)
    log.info(f"Analytics reports saved → {out_dir}")


def run(G: Optional[nx.DiGraph] = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    import sys
    root_dir = Path(__file__).resolve().parent.parent.parent
    if str(root_dir) not in sys.path:
        sys.path.insert(0, str(root_dir))
    from src.graph.builder import load_graph
    if G is None:
        G = load_graph()

    centrality_df = compute_centrality(G)
    centrality_df = compute_sla_breach_contribution(G, centrality_df)
    chronic_df = get_chronic_corridors(G)
    save_reports(centrality_df, chronic_df)
    return centrality_df, chronic_df


if __name__ == "__main__":
    c_df, ch_df = run()
    print("Top 5 Bottlenecks:\n", c_df[["hub", "city", "betweenness_centrality", "pagerank", "sla_breach_pct"]].head(5))
    print(f"\nTotal Chronic Corridors: {len(ch_df)}")

