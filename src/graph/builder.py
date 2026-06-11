"""
Directed weighted graph construction from Delhivery trip data.

Nodes  = unique facilities (hubs, DCs, fulfillment centers)
Edges  = corridors between facilities
Weights = median(delay_ratio) per corridor, stratified by route_type + time_of_day

The graph is stored as a NetworkX DiGraph and serialised to GraphML + pickle.
"""

import logging
import pickle
from pathlib import Path
from typing import Optional

import networkx as nx
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

GRAPH_DIR = Path("data/processed/graphs")
GRAPH_DIR.mkdir(parents=True, exist_ok=True)


def build_graph(df: pd.DataFrame, embed_map: Optional[dict] = None) -> nx.DiGraph:
    """
    Parameters
    ----------
    df : cleaned + feature-engineered DataFrame
    embed_map : optional dict mapping (src, dst, route_type) → np.ndarray embedding

    Returns
    -------
    G : nx.DiGraph with node and edge attributes
    """
    G = nx.DiGraph()

    # --- Node attributes ---
    node_cols = {
        "source_name": "src_city",
        "destination_name": "dst_city",
    }
    for center_col, city_col in node_cols.items():
        if center_col not in df.columns:
            continue
        for center, grp in df.groupby(center_col):
            city = grp[city_col].iloc[0] if city_col in grp.columns else None
            hub_type_col = f"{'src' if center_col == 'source_name' else 'dst'}_hub_type"
            hub_type = grp[hub_type_col].iloc[0] if hub_type_col in grp.columns else "unknown"
            if not G.has_node(center):
                G.add_node(
                    center,
                    city=city,
                    hub_type=hub_type,
                    avg_dwell=float(grp["dwell_time_proxy"].mean()) if "dwell_time_proxy" in grp else 0.0,
                    trip_volume=len(grp),
                )

    # --- Edge attributes ---
    group_keys = ["source_name", "destination_name", "route_type"]
    available_keys = [k for k in group_keys if k in df.columns]

    for keys, grp in df.groupby(available_keys):
        src, dst = keys[0], keys[1]
        route_type = keys[2] if len(keys) > 2 else "unknown"

        delay_ratios = grp["delay_ratio"].dropna() if "delay_ratio" in grp else pd.Series([1.0])
        median_delay = float(delay_ratios.median())
        std_delay = float(delay_ratios.std()) if len(delay_ratios) > 1 else 0.0
        volume = len(grp)

        dist_col = next((c for c in grp.columns if "distance" in c), None)
        osrm_dist = float(grp[dist_col].median()) if dist_col else 0.0

        osrm_time = float(grp["osrm_time"].median()) if "osrm_time" in grp else 0.0
        actual_time = float(grp["actual_time"].median()) if "actual_time" in grp else 0.0

        embed = None
        if embed_map:
            embed = embed_map.get((src, dst, route_type))

        edge_attrs = dict(
            route_type=route_type,
            median_delay_ratio=median_delay,
            std_delay_ratio=std_delay,
            volume=volume,
            osrm_distance=osrm_dist,
            osrm_time=osrm_time,
            actual_time=actual_time,
            is_chronic=median_delay > 1.20,
            weight=median_delay,  # used by NetworkX algorithms
        )
        if embed is not None:
            edge_attrs["embedding"] = embed.tolist()

        if G.has_edge(src, dst):
            # Keep higher-volume version
            if volume > G[src][dst].get("volume", 0):
                G[src][dst].update(edge_attrs)
        else:
            G.add_edge(src, dst, **edge_attrs)

    log.info(
        f"Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges  "
        f"({nx.is_strongly_connected(G) = })"
    )
    return G


def save_graph(G: nx.DiGraph, name: str = "logistics_graph") -> None:
    pickle_path = GRAPH_DIR / f"{name}.pkl"
    with open(pickle_path, "wb") as f:
        pickle.dump(G, f)
    graphml_path = GRAPH_DIR / f"{name}.graphml"
    # GraphML cannot store list-type attributes (embeddings); strip them first
    G_export = G.copy()
    for _, _, data in G_export.edges(data=True):
        data.pop("embedding", None)
    nx.write_graphml(G_export, str(graphml_path))
    log.info(f"Graph saved → {pickle_path} + {graphml_path}")


def load_graph(name: str = "logistics_graph") -> nx.DiGraph:
    pickle_path = GRAPH_DIR / f"{name}.pkl"
    with open(pickle_path, "rb") as f:
        G = pickle.load(f)
    log.info(f"Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


def graph_to_dataframe(G: nx.DiGraph) -> pd.DataFrame:
    """Flatten edges to a DataFrame for downstream ML feature engineering."""
    rows = []
    for src, dst, data in G.edges(data=True):
        row = {"source": src, "destination": dst}
        row.update({k: v for k, v in data.items() if k != "embedding"})
        src_data = G.nodes[src]
        dst_data = G.nodes[dst]
        row["src_city"] = src_data.get("city")
        row["dst_city"] = dst_data.get("city")
        row["src_hub_type"] = src_data.get("hub_type")
        row["dst_hub_type"] = dst_data.get("hub_type")
        row["src_avg_dwell"] = src_data.get("avg_dwell", 0.0)
        row["dst_avg_dwell"] = dst_data.get("avg_dwell", 0.0)
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    sample = pd.DataFrame({
        "source_name": ["Delhi_Hub", "Delhi_Hub", "Mumbai_Hub"],
        "destination_name": ["Mumbai_Hub", "Pune_Hub", "Chennai_Hub"],
        "route_type": ["FTL", "Carting", "FTL"],
        "delay_ratio": [1.3, 1.1, 1.5],
        "dwell_time_proxy": [15, 8, 20],
        "osrm_time": [600, 180, 900],
        "actual_time": [780, 198, 1350],
        "osrm_distance": [1400, 150, 1330],
        "src_city": ["Delhi", "Delhi", "Mumbai"],
        "dst_city": ["Mumbai", "Pune", "Chennai"],
    })
    G = build_graph(sample)
    print(nx.info(G))
    print(graph_to_dataframe(G))
