"""
Two-level hierarchical graph:
  Level 1 — City super-graph  (nodes = cities, edges = inter-city corridors)
  Level 2 — Facility sub-graph (one per city, intra-city facilities)

The multi-city extension goes beyond the base problem (flat single-city graph)
by enabling:
  * Inter-city delay comparison and ranking
  * Cross-city transfer learning (shared GNN encoder)
  * City-pair SLA breach attribution
"""

import logging
from pathlib import Path
from typing import Optional

import networkx as nx
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

GRAPH_DIR = Path("data/processed/graphs")
GRAPH_DIR.mkdir(parents=True, exist_ok=True)


def build_city_super_graph(df: pd.DataFrame) -> nx.DiGraph:
    """
    Aggregate trips at city level → directed city-to-city graph.

    Edge weight = median delay_ratio across all corridors between those cities.
    """
    src_col = "source_city" if "source_city" in df.columns else "src_city"
    dst_col = "dest_city"   if "dest_city"   in df.columns else "dst_city"

    if src_col not in df.columns or dst_col not in df.columns:
        log.warning("City columns not found. Run city_extender.tag_cities() first.")
        return nx.DiGraph()

    G_city = nx.DiGraph()
    grouped = df.groupby([src_col, dst_col])

    for (src_city, dst_city), grp in grouped:
        if src_city == dst_city or not isinstance(src_city, str) or not isinstance(dst_city, str):
            continue
        median_delay = float(grp["delay_ratio"].median()) if "delay_ratio" in grp else 1.0
        volume = len(grp)
        osrm_dist = float(grp["osrm_distance"].median()) if "osrm_distance" in grp else 0.0
        routes = grp["route_type"].value_counts().to_dict() if "route_type" in grp else {}

        G_city.add_edge(
            src_city, dst_city,
            median_delay_ratio=median_delay,
            volume=volume,
            osrm_distance=osrm_dist,
            route_type_mix=routes,
            weight=median_delay,
            is_chronic=median_delay > 1.20,
        )

    # Node attributes
    for city in G_city.nodes():
        city_trips = df[(df[src_col] == city) | (df[dst_col] == city)]
        G_city.nodes[city]["total_volume"] = len(city_trips)
        G_city.nodes[city]["avg_delay"] = float(city_trips["delay_ratio"].mean()) if "delay_ratio" in city_trips else 1.0

    log.info(f"City super-graph: {G_city.number_of_nodes()} cities, {G_city.number_of_edges()} corridors")
    return G_city


def build_city_subgraphs(
    full_graph: nx.DiGraph,
    df: pd.DataFrame,
) -> dict[str, nx.DiGraph]:
    """
    Extract one intra-city subgraph per city from the full facility-level graph.

    Returns dict {city_name: nx.DiGraph}
    """
    city_col = "src_city"
    if city_col not in df.columns:
        log.warning("src_city column missing. Run address_parser.enrich_dataframe() first.")
        return {}

    subgraphs = {}
    cities = df[city_col].dropna().unique()
    for city in cities:
        city_hubs = set(
            df[df[city_col] == city]["source_name"].tolist() +
            df[df["dst_city"] == city]["destination_name"].tolist()
            if "dst_city" in df.columns else
            df[df[city_col] == city]["source_name"].tolist()
        )
        nodes_in_graph = [n for n in city_hubs if full_graph.has_node(n)]
        if len(nodes_in_graph) < 2:
            continue
        sg = full_graph.subgraph(nodes_in_graph).copy()
        subgraphs[city] = sg
        log.debug(f"  {city}: {sg.number_of_nodes()} hubs, {sg.number_of_edges()} edges")

    log.info(f"Built {len(subgraphs)} city subgraphs: {sorted(subgraphs.keys())}")
    return subgraphs


def city_delay_ranking(G_city: nx.DiGraph) -> pd.DataFrame:
    """Rank cities by average outgoing delay ratio (worst first)."""
    rows = []
    for city in G_city.nodes():
        out_edges = list(G_city.out_edges(city, data=True))
        if out_edges:
            avg_delay = np.mean([d.get("median_delay_ratio", 1.0) for _, _, d in out_edges])
            total_vol = sum(d.get("volume", 0) for _, _, d in out_edges)
            n_chronic  = sum(1 for _, _, d in out_edges if d.get("is_chronic"))
        else:
            avg_delay, total_vol, n_chronic = 1.0, 0, 0
        rows.append({
            "city": city,
            "avg_outgoing_delay_ratio": round(avg_delay, 4),
            "total_outgoing_volume": total_vol,
            "chronic_corridors": n_chronic,
            "total_volume": G_city.nodes[city].get("total_volume", 0),
        })
    return pd.DataFrame(rows).sort_values("avg_outgoing_delay_ratio", ascending=False)


def intercity_bottleneck_corridors(G_city: nx.DiGraph, top_n: int = 20) -> pd.DataFrame:
    """Worst inter-city corridors ranked by delay ratio × volume."""
    rows = []
    for src, dst, data in G_city.edges(data=True):
        rows.append({
            "source_city": src,
            "destination_city": dst,
            "median_delay_ratio": data.get("median_delay_ratio", 1.0),
            "volume": data.get("volume", 0),
            "osrm_distance": data.get("osrm_distance", 0),
            "is_chronic": data.get("is_chronic", False),
            "impact_score": data.get("median_delay_ratio", 1.0) * data.get("volume", 0),
        })
    df = pd.DataFrame(rows).sort_values("impact_score", ascending=False)
    return df.head(top_n)


def save_hierarchical(
    G_city: nx.DiGraph,
    subgraphs: dict,
    out_dir: str | Path = "data/processed/graphs",
) -> None:
    import pickle
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "city_super_graph.pkl", "wb") as f:
        pickle.dump(G_city, f)
    with open(out_dir / "city_subgraphs.pkl", "wb") as f:
        pickle.dump(subgraphs, f)
    log.info(f"Hierarchical graphs saved → {out_dir}")


if __name__ == "__main__":
    from src.data.city_extender import run as build_multicity
    df = build_multicity()
    G_city = build_city_super_graph(df)
    ranking = city_delay_ranking(G_city)
    print("City delay ranking:")
    print(ranking.head(10))
    bottlenecks = intercity_bottleneck_corridors(G_city)
    print("\nTop inter-city bottleneck corridors:")
    print(bottlenecks.head(5))
