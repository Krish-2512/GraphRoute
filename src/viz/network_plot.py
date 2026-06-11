"""
Interactive network visualization utilities.

1. PyVis graph — browser-based interactive graph with hover tooltips
2. Folium map  — India map with hub markers colored by bottleneck risk
3. Plotly charts — delay heatmap, city ranking, corridor bar charts
"""

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

CITY_COORDS = {
    "Delhi": (28.6139, 77.2090), "Mumbai": (19.0760, 72.8777),
    "Bengaluru": (12.9716, 77.5946), "Hyderabad": (17.3850, 78.4867),
    "Chennai": (13.0827, 80.2707), "Kolkata": (22.5726, 88.3639),
    "Pune": (18.5204, 73.8567), "Ahmedabad": (23.0225, 72.5714),
    "Jaipur": (26.9124, 75.7873), "Lucknow": (26.8467, 80.9462),
    "Chandigarh": (30.7333, 76.7794), "Kochi": (9.9312, 76.2673),
    "Bhubaneswar": (20.2961, 85.8245), "Indore": (22.7196, 75.8577),
    "Nagpur": (21.1458, 79.0882),
}


def build_pyvis_graph(
    G,
    centrality_df: Optional[pd.DataFrame] = None,
    output_path: str = "reports/network.html",
) -> str:
    try:
        from pyvis.network import Network
    except ImportError:
        log.warning("pyvis not installed.")
        return ""

    net = Network(height="700px", width="100%", directed=True, bgcolor="#0e1117", font_color="white")
    net.set_options(json.dumps({
        "physics": {"barnesHut": {"gravitationalConstant": -5000}},
        "edges": {"smooth": {"type": "curvedCW"}},
    }))

    # Node colors by bottleneck risk
    if centrality_df is not None:
        risk_map = dict(zip(centrality_df["hub"], centrality_df["sla_breach_pct"]))
        max_risk = centrality_df["sla_breach_pct"].max() or 1
    else:
        risk_map, max_risk = {}, 1

    def _risk_color(score: float) -> str:
        ratio = min(score / max_risk, 1.0)
        r = int(255 * ratio)
        g = int(255 * (1 - ratio))
        return f"#{r:02x}{g:02x}33"

    for node, data in G.nodes(data=True):
        risk = risk_map.get(node, 0)
        net.add_node(
            node,
            label=node[:20],
            title=f"City: {data.get('city', 'N/A')}<br>Hub: {data.get('hub_type', 'N/A')}<br>SLA breach score: {risk:.2f}%",
            color=_risk_color(risk),
            size=10 + risk * 2,
        )

    for src, dst, data in G.edges(data=True):
        dr = data.get("median_delay_ratio", 1.0)
        color = "#ff4444" if dr > 1.2 else "#44ff44"
        net.add_edge(
            src, dst,
            title=f"Delay ratio: {dr:.2f}<br>Route: {data.get('route_type', 'N/A')}",
            color=color,
            width=max(1, dr * 2),
        )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    net.save_graph(output_path)
    log.info(f"PyVis graph saved → {output_path}")
    return output_path


def build_folium_map(
    centrality_df: pd.DataFrame,
    city_delay_df: Optional[pd.DataFrame] = None,
) -> "folium.Map":
    try:
        import folium
        from folium.plugins import HeatMap
    except ImportError:
        log.warning("folium not installed.")
        return None

    m = folium.Map(location=[22.0, 79.0], zoom_start=5, tiles="CartoDB dark_matter")

    max_sla = centrality_df["sla_breach_pct"].max() or 1

    for _, row in centrality_df.iterrows():
        city = row.get("city")
        if not city or city not in CITY_COORDS:
            continue
        lat, lon = CITY_COORDS[city]
        risk = row["sla_breach_pct"]
        radius = 8 + risk * 2
        color = "red" if risk > max_sla * 0.6 else "orange" if risk > max_sla * 0.3 else "green"
        folium.CircleMarker(
            location=[lat, lon],
            radius=radius,
            color=color,
            fill=True,
            fill_opacity=0.7,
            popup=folium.Popup(
                f"<b>{row['hub']}</b><br>"
                f"City: {city}<br>"
                f"SLA breach: {risk:.1f}%<br>"
                f"Betweenness: {row['betweenness_centrality']:.4f}",
                max_width=250,
            ),
        ).add_to(m)

    # Heatmap overlay
    heat_data = [
        [CITY_COORDS[r["city"]][0], CITY_COORDS[r["city"]][1], r["sla_breach_pct"]]
        for _, r in centrality_df.iterrows()
        if r.get("city") in CITY_COORDS
    ]
    if heat_data:
        HeatMap(heat_data, radius=40, blur=25, min_opacity=0.3).add_to(m)

    return m


def city_delay_bar_chart(city_delay_df: pd.DataFrame):
    try:
        import plotly.express as px
    except ImportError:
        return None
    fig = px.bar(
        city_delay_df.head(15),
        x="city",
        y="avg_outgoing_delay_ratio",
        color="avg_outgoing_delay_ratio",
        color_continuous_scale="RdYlGn_r",
        title="City-level Average Outgoing Delay Ratio (worst first)",
        labels={"avg_outgoing_delay_ratio": "Delay Ratio", "city": "City"},
    )
    fig.update_layout(template="plotly_dark", showlegend=False)
    return fig


def model_benchmark_chart(results_df: pd.DataFrame):
    try:
        import plotly.graph_objects as go
    except ImportError:
        return None
    fig = go.Figure()
    fig.add_trace(go.Bar(name="MAE (min)", x=results_df["model"], y=results_df["MAE"],
                         marker_color="#4f90ff"))
    fig.add_trace(go.Bar(name="% within 15%", x=results_df["model"], y=results_df["within_15pct"],
                         marker_color="#44cc88"))
    fig.update_layout(
        barmode="group",
        title="Model Benchmark: MAE vs. Within-15% Accuracy",
        template="plotly_dark",
        yaxis_title="Value",
    )
    return fig
