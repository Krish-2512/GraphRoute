"""
What-If Network Latency & Capacity Simulation Engine.

Simulates network-wide ripple effects when a hub's processing capacity is upgraded
or when dwell times drop. Quantifies:
1. Downstream corridor delay reduction
2. Total SLA breaches prevented per month
3. Financial ROI (Revenue recovered in ₹ Lakhs & Payback period)
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
import networkx as nx
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

REVENUE_AT_RISK_PER_BREACH = 850.0  # ₹ penalty + re-delivery cost per breach
AVG_CAPEX_PER_LANE_LAKHS = 45.0      # ₹ Lakhs typical cost for facility automation lane


class NetworkSimulator:
    def __init__(self, G: nx.DiGraph, centrality_df: Optional[pd.DataFrame] = None):
        self.G = G
        self.centrality_df = centrality_df

    @classmethod
    def from_files(cls, graph_path: str = "data/processed/graphs/logistics_graph.pkl",
                   centrality_path: str = "data/processed/hub_centrality.csv"):
        import pickle
        with open(graph_path, "rb") as f:
            G = pickle.load(f)
        centrality_df = None
        if Path(centrality_path).exists():
            centrality_df = pd.read_csv(centrality_path)
        return cls(G, centrality_df)

    def simulate_hub_upgrade(
        self,
        hub_name: str,
        capacity_boost_pct: float = 25.0,
        dwell_reduction_pct: Optional[float] = None,
        capex_lakhs: float = 65.0,
    ) -> Dict[str, Any]:
        """
        Simulate an upgrade to a bottleneck hub.
        
        Parameters:
        -----------
        hub_name: Target facility name
        capacity_boost_pct: Percentage increase in sorting/dock capacity (e.g. 25%)
        dwell_reduction_pct: Estimated % reduction in dwell time (defaults to 0.8 * capacity_boost)
        capex_lakhs: Estimated capital expenditure in ₹ Lakhs
        
        Returns:
        --------
        Dictionary containing empirical before/after metrics and ROI calculations.
        """
        if not self.G.has_node(hub_name):
            matching = [n for n in self.G.nodes if hub_name.lower() in n.lower()]
            if not matching:
                raise ValueError(f"Hub '{hub_name}' not found in logistics graph.")
            hub_name = matching[0]

        if dwell_reduction_pct is None:
            dwell_reduction_pct = capacity_boost_pct * 0.8  # 80% elasticity

        node_data = self.G.nodes[hub_name]
        current_dwell = node_data.get("avg_dwell", 25.0)
        new_dwell = current_dwell * (1.0 - dwell_reduction_pct / 100.0)
        dwell_saved_min = current_dwell - new_dwell

        out_edges = list(self.G.out_edges(hub_name, data=True))
        n_out_corridors = len(out_edges)
        total_monthly_volume = sum(d.get("volume", 50) for _, _, d in out_edges)

        corridor_details: List[Dict[str, Any]] = []
        total_breaches_baseline = 0.0
        total_breaches_simulated = 0.0
        total_transit_hours_saved_monthly = 0.0

        for u, v, data in out_edges:
            vol = data.get("volume", 50)
            base_delay_ratio = data.get("median_delay_ratio", 1.25)
            osrm_time = data.get("osrm_time", 180.0)
            
            base_actual_time = osrm_time * base_delay_ratio
            base_is_breach = base_delay_ratio > 1.20
            base_breaches = vol * (0.85 if base_is_breach else 0.15)
            total_breaches_baseline += base_breaches

            effective_time_saved = min(dwell_saved_min * 0.75, base_actual_time - osrm_time) if base_actual_time > osrm_time else 0.0
            new_actual_time = max(osrm_time, base_actual_time - effective_time_saved)
            new_delay_ratio = new_actual_time / (osrm_time + 1e-6)

            new_is_breach = new_delay_ratio > 1.20
            sim_breaches = vol * (0.45 if new_is_breach else 0.08)
            total_breaches_simulated += sim_breaches

            hours_saved = (effective_time_saved / 60.0) * vol
            total_transit_hours_saved_monthly += hours_saved

            corridor_details.append({
                "destination": v,
                "route_type": data.get("route_type", "Carting"),
                "monthly_volume": vol,
                "base_delay_ratio": round(base_delay_ratio, 3),
                "simulated_delay_ratio": round(new_delay_ratio, 3),
                "time_saved_min_per_trip": round(effective_time_saved, 1),
                "breaches_prevented": round(base_breaches - sim_breaches, 1),
            })

        corridor_details.sort(key=lambda x: x["breaches_prevented"], reverse=True)

        monthly_breaches_avoided = max(0.0, total_breaches_baseline - total_breaches_simulated)
        monthly_revenue_recovered_inr = monthly_breaches_avoided * REVENUE_AT_RISK_PER_BREACH
        annual_revenue_recovered_lakhs = (monthly_revenue_recovered_inr * 12) / 100000.0

        payback_months = (capex_lakhs / (monthly_revenue_recovered_inr / 100000.0)) if monthly_revenue_recovered_inr > 0 else 999.0

        return {
            "hub_name": hub_name,
            "city": node_data.get("city", "Unknown"),
            "hub_type": node_data.get("hub_type", "hub"),
            "capacity_boost_pct": capacity_boost_pct,
            "current_dwell_min": round(current_dwell, 1),
            "simulated_dwell_min": round(new_dwell, 1),
            "dwell_saved_min": round(dwell_saved_min, 1),
            "affected_outbound_corridors": n_out_corridors,
            "monthly_trip_volume_impacted": int(total_monthly_volume),
            "monthly_breaches_baseline": round(total_breaches_baseline, 1),
            "monthly_breaches_simulated": round(total_breaches_simulated, 1),
            "monthly_breaches_avoided": round(monthly_breaches_avoided, 1),
            "monthly_revenue_recovered_lakhs": round(monthly_revenue_recovered_inr / 100000.0, 2),
            "annual_revenue_recovered_lakhs": round(annual_revenue_recovered_lakhs, 2),
            "estimated_capex_lakhs": capex_lakhs,
            "payback_period_months": round(payback_months, 1),
            "total_transit_hours_saved_monthly": round(total_transit_hours_saved_monthly, 1),
            "top_benefiting_corridors": corridor_details[:5],
        }


if __name__ == "__main__":
    G = nx.DiGraph()
    G.add_node("Delhi_Okhla_DC", city="Delhi", hub_type="fulfillment_center", avg_dwell=38.0)
    G.add_node("Mumbai_Bhiwandi_GH", city="Mumbai", hub_type="gateway_hub", avg_dwell=30.0)
    G.add_edge("Delhi_Okhla_DC", "Mumbai_Bhiwandi_GH", volume=450, median_delay_ratio=1.45, osrm_time=600, route_type="FTL")
    sim = NetworkSimulator(G)
    res = sim.simulate_hub_upgrade("Delhi_Okhla_DC", capacity_boost_pct=30.0)
    print("Simulation result:", res)
