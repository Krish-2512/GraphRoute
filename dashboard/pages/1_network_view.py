"""
Page 1 — Interactive Logistics Network Map

Shows the directed graph of hubs overlaid on an India map.
Nodes are colored by bottleneck risk (red = high, green = low).
Edges colored red if chronic (delay_ratio > 1.2).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Network View", layout="wide")
st.title("🗺 Logistics Network View")
st.markdown("Interactive hub-and-corridor map. Node size ∝ bottleneck risk. Red edges = chronic delay corridors.")

# ── Data loading ───────────────────────────────────────────────────────────────

@st.cache_data
def load_centrality():
    try:
        return pd.read_csv("data/processed/hub_centrality.csv")
    except FileNotFoundError:
        # Synthetic demo data
        cities = ["Delhi", "Mumbai", "Bengaluru", "Hyderabad", "Chennai",
                  "Kolkata", "Pune", "Ahmedabad", "Jaipur", "Lucknow"]
        rng = np.random.default_rng(42)
        n = 40
        return pd.DataFrame({
            "hub": [f"{cities[i%10]}_Hub_{i}" for i in range(n)],
            "city": [cities[i%10] for i in range(n)],
            "hub_type": rng.choice(["gateway_hub", "fulfillment_center", "last_mile_hub", "sorting_center"], n),
            "betweenness_centrality": rng.uniform(0, 0.4, n),
            "pagerank": rng.uniform(0.01, 0.08, n),
            "in_degree": rng.integers(1, 10, n),
            "out_degree": rng.integers(1, 10, n),
            "avg_dwell_min": rng.uniform(5, 45, n),
            "sla_breach_pct": rng.uniform(0.5, 20, n),
            "trip_volume": rng.integers(50, 2000, n),
        })

@st.cache_data
def load_chronic():
    try:
        return pd.read_csv("data/processed/chronic_corridors.csv")
    except FileNotFoundError:
        cities = ["Delhi", "Mumbai", "Bengaluru", "Hyderabad", "Chennai", "Kolkata"]
        rng = np.random.default_rng(42)
        n = 20
        return pd.DataFrame({
            "source": [f"{cities[i%6]}_Hub_{i}" for i in range(n)],
            "destination": [f"{cities[(i+1)%6]}_Hub_{(i+2)%6}" for i in range(n)],
            "route_type": rng.choice(["FTL", "Carting"], n),
            "median_delay_ratio": rng.uniform(1.2, 2.5, n),
            "volume": rng.integers(10, 500, n),
            "osrm_distance": rng.uniform(100, 1500, n),
        })

centrality_df = load_centrality()
chronic_df = load_chronic()

# ── Filters ────────────────────────────────────────────────────────────────────

col1, col2, col3 = st.columns(3)
with col1:
    cities = sorted(centrality_df["city"].dropna().unique())
    selected_cities = st.multiselect("Filter by City", cities, default=cities[:5])
with col2:
    hub_types = sorted(centrality_df["hub_type"].dropna().unique())
    selected_hub_types = st.multiselect("Hub Type", hub_types, default=hub_types)
with col3:
    min_risk = st.slider("Min SLA Breach %", 0.0, float(centrality_df["sla_breach_pct"].max()), 0.0)

filtered = centrality_df[
    centrality_df["city"].isin(selected_cities) &
    centrality_df["hub_type"].isin(selected_hub_types) &
    (centrality_df["sla_breach_pct"] >= min_risk)
]

st.markdown(f"**{len(filtered)} hubs** matching filters | **{len(chronic_df)} chronic corridors** total")

# ── Folium Map ─────────────────────────────────────────────────────────────────

try:
    import folium
    from streamlit_folium import st_folium

    CITY_COORDS = {
        "Delhi": (28.61, 77.21), "Mumbai": (19.08, 72.88), "Bengaluru": (12.97, 77.59),
        "Hyderabad": (17.39, 78.49), "Chennai": (13.08, 80.27), "Kolkata": (22.57, 88.36),
        "Pune": (18.52, 73.86), "Ahmedabad": (23.02, 72.57), "Jaipur": (26.91, 75.79),
        "Lucknow": (26.85, 80.95), "Chandigarh": (30.73, 76.78), "Kochi": (9.93, 76.27),
        "Bhubaneswar": (20.30, 85.82), "Indore": (22.72, 75.86), "Nagpur": (21.15, 79.09),
    }

    m = folium.Map(location=[22.0, 79.0], zoom_start=5, tiles="CartoDB dark_matter")
    max_risk = filtered["sla_breach_pct"].max() or 1

    for _, row in filtered.iterrows():
        city = row.get("city")
        if city not in CITY_COORDS:
            continue
        lat, lon = CITY_COORDS[city]
        # Add jitter so multiple hubs in same city don't overlap
        lat += np.random.uniform(-0.3, 0.3)
        lon += np.random.uniform(-0.3, 0.3)
        risk_ratio = row["sla_breach_pct"] / max_risk
        color = "red" if risk_ratio > 0.6 else "orange" if risk_ratio > 0.3 else "green"
        folium.CircleMarker(
            location=[lat, lon],
            radius=5 + row["sla_breach_pct"] * 0.8,
            color=color, fill=True, fill_opacity=0.75,
            popup=folium.Popup(
                f"<b>{row['hub']}</b><br>"
                f"Type: {row['hub_type']}<br>"
                f"SLA breach: {row['sla_breach_pct']:.1f}%<br>"
                f"Betweenness: {row['betweenness_centrality']:.4f}<br>"
                f"Dwell: {row['avg_dwell_min']:.1f} min",
                max_width=250,
            ),
        ).add_to(m)

    st_folium(m, height=550, use_container_width=True)

except ImportError:
    st.warning("Install `folium` and `streamlit-folium` for the interactive map.")
    st.dataframe(filtered[["hub", "city", "hub_type", "sla_breach_pct", "betweenness_centrality"]].head(20))

# ── Chronic corridors table ────────────────────────────────────────────────────

st.markdown("#### Chronic Delay Corridors (delay > 20% over OSRM)")
chronic_display = chronic_df.copy()
chronic_display["delay_severity"] = chronic_display["median_delay_ratio"].apply(
    lambda x: "🔴 Severe" if x > 2.0 else "🟠 High" if x > 1.5 else "🟡 Moderate"
)
st.dataframe(
    chronic_display[["source", "destination", "route_type", "median_delay_ratio", "volume", "delay_severity"]]
    .sort_values("median_delay_ratio", ascending=False),
    use_container_width=True,
)
