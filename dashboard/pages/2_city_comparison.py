"""
Page 2 — Multi-City Delay Comparison

Choropleth + ranked table: which cities have the worst average delay,
highest SLA breach rates, and most chronic inter-city corridors.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="City Comparison", layout="wide")
st.title("🏙 Multi-City Delay Comparison")

# ── Demo / real data ───────────────────────────────────────────────────────────

CITY_COORDS = {
    "Delhi": (28.61, 77.21), "Mumbai": (19.08, 72.88), "Bengaluru": (12.97, 77.59),
    "Hyderabad": (17.39, 78.49), "Chennai": (13.08, 80.27), "Kolkata": (22.57, 88.36),
    "Pune": (18.52, 73.86), "Ahmedabad": (23.02, 72.57), "Jaipur": (26.91, 75.79),
    "Lucknow": (26.85, 80.95), "Chandigarh": (30.73, 76.78), "Kochi": (9.93, 76.27),
    "Bhubaneswar": (20.30, 85.82), "Indore": (22.72, 75.86), "Nagpur": (21.15, 79.09),
}


@st.cache_data
def load_city_data():
    try:
        from src.graph.hierarchical import city_delay_ranking
        import pickle
        with open("data/processed/graphs/city_super_graph.pkl", "rb") as f:
            G_city = pickle.load(f)
        return city_delay_ranking(G_city)
    except Exception:
        rng = np.random.default_rng(42)
        cities = list(CITY_COORDS.keys())
        return pd.DataFrame({
            "city": cities,
            "avg_outgoing_delay_ratio": rng.uniform(1.05, 1.65, len(cities)),
            "total_outgoing_volume":    rng.integers(200, 5000, len(cities)),
            "chronic_corridors":        rng.integers(0, 15, len(cities)),
            "total_volume":             rng.integers(500, 10000, len(cities)),
        }).sort_values("avg_outgoing_delay_ratio", ascending=False)


city_df = load_city_data()
city_df["lat"] = city_df["city"].map(lambda c: CITY_COORDS.get(c, (22, 79))[0])
city_df["lon"] = city_df["city"].map(lambda c: CITY_COORDS.get(c, (22, 79))[1])
city_df["delay_pct_over_osrm"] = ((city_df["avg_outgoing_delay_ratio"] - 1) * 100).round(1)

# ── KPI row ────────────────────────────────────────────────────────────────────

col1, col2, col3 = st.columns(3)
worst = city_df.iloc[0]
best  = city_df.iloc[-1]
with col1:
    st.metric("Worst City (Avg Delay)", worst["city"],
              delta=f"+{worst['delay_pct_over_osrm']:.1f}% over OSRM", delta_color="inverse")
with col2:
    st.metric("Best City (Avg Delay)", best["city"],
              delta=f"+{best['delay_pct_over_osrm']:.1f}% over OSRM", delta_color="normal")
with col3:
    st.metric("Cities Analysed", len(city_df))

st.markdown("---")

# ── Map + Bar chart side by side ───────────────────────────────────────────────

col_map, col_bar = st.columns([1.2, 1])

with col_map:
    st.markdown("#### Delay Intensity Map")
    fig_map = px.scatter_mapbox(
        city_df,
        lat="lat", lon="lon",
        size="avg_outgoing_delay_ratio",
        color="avg_outgoing_delay_ratio",
        color_continuous_scale="RdYlGn_r",
        hover_name="city",
        hover_data={"delay_pct_over_osrm": True, "chronic_corridors": True,
                    "total_outgoing_volume": True, "lat": False, "lon": False},
        mapbox_style="carto-darkmatter",
        zoom=3.8,
        center={"lat": 22.0, "lon": 79.0},
        title="City Avg Delay Ratio (larger = worse)",
        size_max=35,
    )
    fig_map.update_layout(height=420, margin={"r": 0, "t": 30, "l": 0, "b": 0})
    st.plotly_chart(fig_map, use_container_width=True)

with col_bar:
    st.markdown("#### City Delay Ranking")
    fig_bar = px.bar(
        city_df.sort_values("avg_outgoing_delay_ratio", ascending=True),
        x="avg_outgoing_delay_ratio",
        y="city",
        orientation="h",
        color="avg_outgoing_delay_ratio",
        color_continuous_scale="RdYlGn_r",
        labels={"avg_outgoing_delay_ratio": "Avg Delay Ratio", "city": ""},
        title="Outgoing Delay Ratio by City",
    )
    fig_bar.add_vline(x=1.2, line_dash="dash", line_color="red",
                      annotation_text="Chronic threshold (1.20)", annotation_position="top right")
    fig_bar.update_layout(height=420, template="plotly_dark", showlegend=False)
    st.plotly_chart(fig_bar, use_container_width=True)

# ── Volume vs Delay scatter ────────────────────────────────────────────────────

st.markdown("#### Volume vs. Delay — Prioritisation Quadrant")
st.markdown("Top-right quadrant = high volume + high delay → highest intervention priority.")

fig_scatter = px.scatter(
    city_df,
    x="total_outgoing_volume",
    y="avg_outgoing_delay_ratio",
    text="city",
    size="chronic_corridors",
    color="avg_outgoing_delay_ratio",
    color_continuous_scale="RdYlGn_r",
    labels={"total_outgoing_volume": "Total Trip Volume", "avg_outgoing_delay_ratio": "Avg Delay Ratio"},
    title="Volume vs. Delay (size = chronic corridor count)",
)
fig_scatter.add_hline(y=1.20, line_dash="dot", line_color="red", annotation_text="Chronic threshold")
fig_scatter.update_traces(textposition="top center")
fig_scatter.update_layout(template="plotly_dark", height=420)
st.plotly_chart(fig_scatter, use_container_width=True)

# ── Sortable table ─────────────────────────────────────────────────────────────

st.markdown("#### City Metrics Table")
display_cols = ["city", "avg_outgoing_delay_ratio", "delay_pct_over_osrm",
                "total_outgoing_volume", "chronic_corridors"]
st.dataframe(
    city_df[display_cols].rename(columns={
        "avg_outgoing_delay_ratio": "Avg Delay Ratio",
        "delay_pct_over_osrm": "% Over OSRM",
        "total_outgoing_volume": "Total Volume",
        "chronic_corridors": "Chronic Corridors",
    }),
    use_container_width=True,
)
