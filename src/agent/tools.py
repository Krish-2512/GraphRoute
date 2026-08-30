"""
Supply Chain Network Operations Tools for LangChain / Agentic AI Copilot.

Provides structured tools for:
1. Hub health and bottleneck inspection
2. What-if capacity and latency simulation
3. FTL vs Carting route recommendation
4. Operations Incident & Policy Memo generation
"""

import json
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List
import pandas as pd
import networkx as nx

root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.graph.simulator import NetworkSimulator


def _load_graph_data():
    centrality_df = None
    if Path("data/processed/hub_centrality.csv").exists():
        centrality_df = pd.read_csv("data/processed/hub_centrality.csv")
    
    chronic_df = None
    if Path("data/processed/chronic_corridors.csv").exists():
        chronic_df = pd.read_csv("data/processed/chronic_corridors.csv")

    G = None
    graph_pkl = Path("data/processed/graphs/logistics_graph.pkl")
    if graph_pkl.exists():
        import pickle
        with open(graph_pkl, "rb") as f:
            G = pickle.load(f)
    else:
        # Fallback minimal graph
        G = nx.DiGraph()
        G.add_node("Delhi_Okhla_Phase2_DC", city="Delhi", hub_type="fulfillment_center", avg_dwell=38.2, trip_volume=4820)
        G.add_node("Mumbai_Bhiwandi_GH", city="Mumbai", hub_type="gateway_hub", avg_dwell=29.5, trip_volume=3950)
        G.add_node("Bengaluru_Whitefield_GH", city="Bengaluru", hub_type="gateway_hub", avg_dwell=24.1, trip_volume=3310)
        G.add_edge("Delhi_Okhla_Phase2_DC", "Mumbai_Bhiwandi_GH", volume=450, median_delay_ratio=1.45, osrm_time=600, route_type="FTL")

    return G, centrality_df, chronic_df


class HubHealthTool:
    """Inspects structural risk, dwell time, and SLA breach contribution of any hub."""
    name = "query_hub_health"
    description = "Queries the health, betweenness centrality, dwell time, and SLA breach risk of a logistics hub."

    def run(self, hub_name: str) -> str:
        G, centrality_df, _ = _load_graph_data()
        if centrality_df is not None and not centrality_df.empty:
            matches = centrality_df[centrality_df["hub"].str.contains(hub_name, case=False, na=False)]
            if not matches.empty:
                row = matches.iloc[0]
                res = {
                    "hub": row.get("hub"),
                    "city": row.get("city"),
                    "hub_type": row.get("hub_type"),
                    "sla_breach_contribution_pct": f"{row.get('sla_breach_pct', 0):.2f}%",
                    "betweenness_centrality": f"{row.get('betweenness_centrality', 0):.4f}",
                    "pagerank": f"{row.get('pagerank', 0):.4f}",
                    "avg_dwell_time_minutes": f"{row.get('avg_dwell_min', 0):.1f} min",
                    "monthly_trip_volume": int(row.get("trip_volume", 0)),
                    "status": "CRITICAL BOTTLENECK" if row.get("sla_breach_pct", 0) > 10 else "MODERATE RISK" if row.get("sla_breach_pct", 0) > 5 else "HEALTHY",
                }
                return json.dumps(res, indent=2)

        if G and G.has_node(hub_name):
            data = G.nodes[hub_name]
            return json.dumps({
                "hub": hub_name,
                "city": data.get("city", "Unknown"),
                "hub_type": data.get("hub_type", "Hub"),
                "avg_dwell_time_minutes": f"{data.get('avg_dwell', 20.0):.1f} min",
                "outbound_corridors": len(list(G.out_edges(hub_name))),
            }, indent=2)

        return json.dumps({"error": f"Hub '{hub_name}' not found in active logistics network."})


class WhatIfSimulationTool:
    """Simulates the network-wide latency and revenue impact of upgrading a hub's capacity."""
    name = "simulate_hub_upgrade"
    description = "Simulates the network-wide SLA breach reduction and financial ROI (₹ Lakhs) from expanding a hub's capacity."

    def run(self, hub_name: str, capacity_boost_pct: float = 25.0, capex_lakhs: float = 65.0) -> str:
        G, centrality_df, _ = _load_graph_data()
        simulator = NetworkSimulator(G, centrality_df)
        try:
            res = simulator.simulate_hub_upgrade(
                hub_name=hub_name,
                capacity_boost_pct=float(capacity_boost_pct),
                capex_lakhs=float(capex_lakhs),
            )
            return json.dumps(res, indent=2)
        except Exception as e:
            return json.dumps({"error": f"Simulation failed: {str(e)}"})


class RouteAdvisorTool:
    """Evaluates time vs cost trade-off to recommend FTL vs Carting for a shipment corridor."""
    name = "recommend_route_type"
    description = "Recommends whether FTL or Carting should be used for a given corridor and estimates time saved vs cost premium."

    def run(self, distance_km: float, time_of_day: str = "morning", historical_delay_ratio: float = 1.25) -> str:
        dist = float(distance_km)
        dr = float(historical_delay_ratio)
        tod = time_of_day.lower()

        ftl_score = (
            0.35 * min(dist / 1000.0, 1.0) +
            0.30 * min((dr - 1.0) / 1.0, 1.0) +
            (0.15 if tod in ["night", "afternoon"] else 0.05) +
            (0.20 if dist > 500 else 0.05)
        )
        recommendation = "FTL" if ftl_score >= 0.50 else "Carting"
        base_time_hours = dist / 55.0
        time_saving_min = max(0.0, (dr - 1.0) * base_time_hours * 60.0 * 0.35 * ftl_score)
        cost_premium_pct = 30.0 if recommendation == "FTL" else 0.0

        return json.dumps({
            "recommended_mode": recommendation,
            "ftl_suitability_score": f"{ftl_score*100:.1f}%",
            "distance_km": dist,
            "projected_time_saving_minutes": round(time_saving_min, 1),
            "cost_premium_pct": f"{cost_premium_pct:.0f}%",
            "rationale": (
                f"FTL recommended due to long-haul distance ({dist:.0f}km) and elevated corridor delay ({dr:.2f}x). "
                f"Direct point-to-point loading bypasses transshipment sorting, recovering ~{time_saving_min:.0f} mins."
                if recommendation == "FTL" else
                f"Carting recommended for distance ({dist:.0f}km). Transshipment overhead is economically justified by a 30% lower freight cost."
            )
        }, indent=2)


class IncidentMemoTool:
    """Drafts an executive Operations Strategy Memo for a selected hub."""
    name = "generate_incident_memo"
    description = "Generates an executive memo detailing top bottleneck hubs, interventions, and financial recovery."

    def run(self, hub_name: str) -> str:
        health_info = json.loads(HubHealthTool().run(hub_name))
        sim_info = json.loads(WhatIfSimulationTool().run(hub_name, capacity_boost_pct=30.0))

        memo = f"""# Operations Strategy Action Memo

**Target Hub:** {hub_name} ({health_info.get('city', 'All-India')} - {health_info.get('hub_type', 'Hub')})
**Current Health Status:** {health_info.get('status', 'EVALUATION')}
**SLA Breach Contribution:** {health_info.get('sla_breach_contribution_pct', 'N/A')}
**Average Facility Dwell Time:** {health_info.get('avg_dwell_time_minutes', 'N/A')}

---

### Key Operational Findings
1. Hub `{hub_name}` exhibits a high betweenness centrality of **{health_info.get('betweenness_centrality', 'N/A')}**, causing transit dwell ripples across **{sim_info.get('affected_outbound_corridors', 'N/A')}** downstream corridors.
2. Inbound and sorting congestion impacts an estimated **{sim_info.get('monthly_trip_volume_impacted', 'N/A'):,}** shipments monthly.

### Recommended Intervention Plan
- **Action:** Expand dock throughput and add automated sorting lane.
- **Estimated CAPEX:** ₹{sim_info.get('estimated_capex_lakhs', 65)} Lakhs
- **Monthly SLA Breaches Prevented:** ~{sim_info.get('monthly_breaches_avoided', 0):.0f} breaches
- **Monthly Revenue Recovered:** ₹{sim_info.get('monthly_revenue_recovered_lakhs', 0):.2f} Lakhs/month
- **Payback Horizon:** {sim_info.get('payback_period_months', 0):.1f} months
"""
        return memo


class SmartRerouteTool:
    """Computes dynamic optimal paths bypassing bottleneck hubs."""
    name = "find_alternate_route"
    description = "Finds optimal green bypass route for a source and destination to avoid congested chokepoint hubs."

    def run(self, source_hub: str, dest_hub: str) -> str:
        from src.graph.rerouter import SmartRerouter
        G, centrality_df, _ = _load_graph_data()
        rerouter = SmartRerouter(G, centrality_df)
        res = rerouter.find_alternate_routes(source_hub, dest_hub)
        return json.dumps(res, indent=2)


def get_all_tools():
    return [HubHealthTool(), WhatIfSimulationTool(), RouteAdvisorTool(), IncidentMemoTool(), SmartRerouteTool()]

