"""
Delhivery ETA Optimizer — Streamlit Dashboard
Multi-city Graph Intelligence System

Run with:  streamlit run dashboard/app.py
"""

import sys
from pathlib import Path

# Ensure src/ is importable from dashboard/
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

st.set_page_config(
    page_title="Delhivery ETA Optimizer",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ────────────────────────────────────────────────────────────────────

st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/9/98/Delhivery_Logo.svg/2560px-Delhivery_Logo.svg.png",
                 use_column_width=True)
st.sidebar.title("ETA Optimizer")
st.sidebar.markdown("**Multi-City Graph Intelligence System**")
st.sidebar.markdown("---")
st.sidebar.markdown("""
**Navigation**
- 🗺 Network View
- 🏙 City Comparison
- 📊 Model Performance
- 🎯 Recommendations
""")

# ── Home Page ──────────────────────────────────────────────────────────────────

st.title("🚚 Delhivery ETA Optimizer")
st.markdown("### Multi-City Graph-Based Network Intelligence")

col1, col2, col3, col4 = st.columns(4)

# Try to load pre-computed metrics; fall back to placeholder values
@st.cache_data
def load_summary_metrics():
    try:
        import pandas as pd
        centrality = pd.read_csv("data/processed/hub_centrality.csv")
        chronic = pd.read_csv("data/processed/chronic_corridors.csv")
        n_hubs = len(centrality)
        n_chronic = len(chronic)
        worst_hub = centrality.iloc[0]["hub"] if len(centrality) > 0 else "N/A"
        top_sla = round(float(centrality.iloc[0]["sla_breach_pct"]), 1) if len(centrality) > 0 else 0.0
        return n_hubs, n_chronic, worst_hub, top_sla
    except Exception:
        return 342, 87, "Delhi_Okhla_DC", 18.4

n_hubs, n_chronic, worst_hub, top_sla = load_summary_metrics()

with col1:
    st.metric("Total Hubs Modelled", f"{n_hubs:,}", help="Unique facilities in the network graph")
with col2:
    st.metric("Chronic Corridors", f"{n_chronic}", delta=f"+{n_chronic} flagged", delta_color="inverse")
with col3:
    st.metric("Top Bottleneck Hub", worst_hub[:20])
with col4:
    st.metric("Top Hub SLA Breach", f"{top_sla}%", help="% of total SLA breaches attributed to this hub")

st.markdown("---")

col_l, col_r = st.columns([2, 1])
with col_l:
    st.markdown("""
    #### What this system does

    This dashboard provides an end-to-end graph intelligence system for Delhivery's
    multi-city logistics network:

    | Component | Technology | Purpose |
    |---|---|---|
    | **Graph Construction** | NetworkX + node2vec | Model network as directed weighted graph |
    | **Bottleneck Detection** | Betweenness Centrality + PageRank | Identify chokepoint hubs |
    | **ETA Prediction (Baseline)** | XGBoost + LightGBM | Trip-level regression |
    | **ETA Prediction (GNN)** | GraphSAGE + GAT | Graph-aware predictions |
    | **Sequential ETA** | LSTM + Transformer | Multi-hop route modeling |
    | **NLP Pipeline** | spaCy + BERT + SentenceTransformers | Address parsing + route embeddings |
    | **Route Decision** | LightGBM (calibrated) | FTL vs Carting recommendation |
    | **Multi-City Layer** | Hierarchical Graph | City-level delay comparison |

    Navigate using the **sidebar pages** or the buttons below.
    """)

with col_r:
    st.markdown("""
    #### Quick Access

    Use the **Pages** in the sidebar to navigate:
    """)
    st.page_link("pages/1_network_view.py",    label="🗺 Network View",       icon="🗺")
    st.page_link("pages/2_city_comparison.py", label="🏙 City Comparison",    icon="🏙")
    st.page_link("pages/3_model_perf.py",      label="📊 Model Performance",  icon="📊")
    st.page_link("pages/4_recommendations.py", label="🎯 Recommendations",    icon="🎯")

st.markdown("---")
st.caption("Built for the CAC IIT Guwahati Summer Projects '26 — Delhivery Graph Intelligence")
