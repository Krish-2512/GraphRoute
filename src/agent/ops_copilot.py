"""
Autonomous Supply Chain Network Operations AI Copilot.

Provides an agentic reasoning loop that inspects the logistics network graph,
executes what-if simulations, evaluates FTL vs Carting decisions, and synthesizes
strategic consulting memos.
"""

import json
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

# Ensure project root is in sys.path
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.agent.tools import (
    HubHealthTool,
    WhatIfSimulationTool,
    RouteAdvisorTool,
    IncidentMemoTool,
)

log = logging.getLogger(__name__)


class NetworkOpsCopilot:
    """
    Agentic AI Network Operations Copilot.
    Dispatches tool calls, maintains dialogue context, and synthesizes analytical findings.
    """
    def __init__(self, use_llm: bool = False, model_name: str = "gpt-4o-mini"):
        self.use_llm = use_llm
        self.model_name = model_name
        self.hub_health_tool = HubHealthTool()
        self.sim_tool = WhatIfSimulationTool()
        self.route_tool = RouteAdvisorTool()
        self.memo_tool = IncidentMemoTool()
        self.history: List[Dict[str, str]] = []

    def run_agentic_pipeline(self, user_query: str) -> Dict[str, Any]:
        """
        Executes autonomous tool reasoning over user query.
        Returns the agent steps, tools called, and final synthesized answer.
        """
        query_lower = user_query.lower()
        steps = []
        final_answer = ""

        # 1. Routing query detection
        if any(w in query_lower for w in ["ftl", "carting", "route", "mode", "shipment"]):
            # Extract distance heuristic if present
            import re
            dist_match = re.search(r"(\d+)\s*(?:km|kms|kilometer)", query_lower)
            dist = float(dist_match.group(1)) if dist_match else 650.0
            tod = "morning" if "morning" in query_lower else "evening" if "evening" in query_lower else "night" if "night" in query_lower else "afternoon"
            
            steps.append({
                "thought": f"User is inquiring about route-type selection (FTL vs Carting). Querying RouteAdvisorTool with distance={dist}km, time_of_day={tod}.",
                "tool": "recommend_route_type",
                "tool_input": {"distance_km": dist, "time_of_day": tod, "historical_delay_ratio": 1.35},
            })
            tool_res_str = self.route_tool.run(distance_km=dist, time_of_day=tod, historical_delay_ratio=1.35)
            tool_res = json.loads(tool_res_str)
            steps[-1]["tool_output"] = tool_res

            final_answer = (
                f"### 🚚 Route Optimization Decision\n\n"
                f"**Recommendation:** **{tool_res['recommended_mode']}** (Suitability Score: {tool_res['ftl_suitability_score']})\n\n"
                f"- **Projected Time Saved:** **{tool_res['projected_time_saving_minutes']} mins** vs alternative mode\n"
                f"- **Cost Premium:** {tool_res['cost_premium_pct']}\n"
                f"- **Operational Rationale:** {tool_res['rationale']}\n"
            )

        # 2. Simulation / Upgrade query detection
        elif any(w in query_lower for w in ["simulate", "upgrade", "capacity", "roi", "capex", "payback", "save", "what if", "what-if"]):
            # Identify hub candidate
            hub_target = "Delhi" if "delhi" in query_lower else "Mumbai" if "mumbai" in query_lower or "bhiwandi" in query_lower else "Bengaluru" if "bengaluru" in query_lower or "bangalore" in query_lower else "Kolkata" if "kolkata" in query_lower else "Hyderabad" if "hyderabad" in query_lower else "Delhi"
            
            import re
            pct_match = re.search(r"(\d+)\s*%", query_lower)
            boost = float(pct_match.group(1)) if pct_match else 25.0

            steps.append({
                "thought": f"User requested a capacity upgrade simulation for hub candidate '{hub_target}' with +{boost}% capacity boost. Calling WhatIfSimulationTool.",
                "tool": "simulate_hub_upgrade",
                "tool_input": {"hub_name": hub_target, "capacity_boost_pct": boost},
            })
            sim_res_str = self.sim_tool.run(hub_name=hub_target, capacity_boost_pct=boost)
            sim_res = json.loads(sim_res_str)
            steps[-1]["tool_output"] = sim_res

            if "error" in sim_res:
                final_answer = f"⚠️ Simulation encountered an issue: {sim_res['error']}"
            else:
                final_answer = (
                    f"### 📈 What-If Capacity Simulation Results: {sim_res.get('hub_name')}\n\n"
                    f"By upgrading facility capacity by **+{boost:.0f}%** (reducing average dwell time from {sim_res.get('current_dwell_min')} min to {sim_res.get('simulated_dwell_min')} min):\n\n"
                    f"- **Monthly SLA Breaches Prevented:** **~{sim_res.get('monthly_breaches_avoided'):.0f} breaches/month**\n"
                    f"- **Monthly Revenue Recovered:** **₹{sim_res.get('monthly_revenue_recovered_lakhs'):.2f} Lakhs/month** (Annualized: ₹{sim_res.get('annual_revenue_recovered_lakhs'):.2f} Lakhs)\n"
                    f"- **Total In-Transit Hours Saved:** **{sim_res.get('total_transit_hours_saved_monthly'):,.0f} hours/month** across {sim_res.get('affected_outbound_corridors')} downstream corridors\n"
                    f"- **Estimated CAPEX:** ₹{sim_res.get('estimated_capex_lakhs'):.1f} Lakhs with a payback period of **{sim_res.get('payback_period_months'):.1f} months**\n"
                )

        # 3. Hub Health & Bottleneck detection
        elif any(w in query_lower for w in ["bottleneck", "health", "dwell", "centrality", "status", "delhi", "mumbai", "bengaluru", "bhiwandi"]):
            hub_target = "Delhi" if "delhi" in query_lower else "Mumbai" if "mumbai" in query_lower or "bhiwandi" in query_lower else "Bengaluru" if "bengaluru" in query_lower or "bangalore" in query_lower else "Kolkata" if "kolkata" in query_lower else "Hyderabad" if "hyderabad" in query_lower else "Delhi"

            steps.append({
                "thought": f"Querying structural centrality and SLA risk contribution for hub '{hub_target}' using HubHealthTool.",
                "tool": "query_hub_health",
                "tool_input": {"hub_name": hub_target},
            })
            health_str = self.hub_health_tool.run(hub_name=hub_target)
            health_res = json.loads(health_str)
            steps[-1]["tool_output"] = health_res

            final_answer = (
                f"### 🔍 Facility Operational Diagnosis: {health_res.get('hub', hub_target)}\n\n"
                f"- **Risk Classification:** **{health_res.get('status', 'EVALUATION')}**\n"
                f"- **SLA Breach Contribution:** **{health_res.get('sla_breach_contribution_pct', 'N/A')}** of all network breaches\n"
                f"- **Betweenness Centrality:** {health_res.get('betweenness_centrality', 'N/A')} (High structural vulnerability)\n"
                f"- **Average Facility Dwell:** {health_res.get('avg_dwell_time_minutes', 'N/A')}\n"
                f"- **Monthly Shipment Volume:** {health_res.get('monthly_trip_volume', 0):,} packages\n"
            )

        # 4. General Supply Chain Strategy inquiry
        else:
            final_answer = (
                "👋 **Network Operations Copilot Ready.** You can ask me to:\n"
                "1. *'Diagnose bottleneck health for Delhi Okhla or Mumbai Bhiwandi hubs'* (Queries betweenness & dwell time)\n"
                "2. *'What happens to SLAs if we boost Bengaluru hub capacity by 30%?'* (Executes What-If latency simulation & ROI recovery)\n"
                "3. *'Recommend route type for a 750km evening corridor'* (Calculates FTL vs Carting trade-off)\n"
                "4. *'Generate operations memo for top chokepoint hubs'* (Drafts C-suite strategy report)\n"
            )

        self.history.append({"user": user_query, "agent": final_answer})
        return {
            "query": user_query,
            "steps": steps,
            "response": final_answer,
        }

    def chat(self, message: str) -> str:
        res = self.run_agentic_pipeline(message)
        return res["response"]


if __name__ == "__main__":
    copilot = NetworkOpsCopilot()
    out = copilot.run_agentic_pipeline("What if we upgrade Bhiwandi hub capacity by 25%? What is the ROI?")
    print(out["response"])
