"""
Complete Live System Demonstration Script for GraphRoute.
"""

import sys
import json
from pathlib import Path

root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.agent.tools import HubHealthTool, WhatIfSimulationTool, RouteAdvisorTool, IncidentMemoTool
from src.agent.ops_copilot import NetworkOpsCopilot

def run_all_demos():
    print("================================================================")
    print("DEMO 1: HUB HEALTH & BOTTLENECK AUDIT (Real Data Output)")
    print("================================================================")
    health_tool = HubHealthTool()
    res1 = health_tool.run("Gurgaon")
    print(res1)

    print("\n================================================================")
    print("DEMO 2: WHAT-IF LATENCY & CAPACITY SIMULATION (Real Data Output)")
    print("================================================================")
    sim_tool = WhatIfSimulationTool()
    res2 = sim_tool.run("Kolkata_Dankuni_HB", capacity_boost_pct=30.0, capex_lakhs=45.0)
    print(res2)

    print("\n================================================================")
    print("DEMO 3: FTL VS CARTING DECISION ENGINE (Real Cost-Delay Tradeoff)")
    print("================================================================")
    route_tool = RouteAdvisorTool()
    res3 = route_tool.run(distance_km=850.0, time_of_day="evening", historical_delay_ratio=1.45)
    print(res3)

    print("\n================================================================")
    print("DEMO 4: AUTONOMOUS AI OPS COPILOT AGENTIC REASONING")
    print("================================================================")
    copilot = NetworkOpsCopilot()
    query = "What happens to network SLAs if we upgrade Bangalore hub by 25%?"
    res4 = copilot.run_agentic_pipeline(query)
    print("USER QUERY:", query)
    for i, step in enumerate(res4.get("steps", []), 1):
        print(f"\n[Agent Thought {i}]: {step.get('thought')}")
        print(f"[Tool Invoked {i}]: {step.get('tool')}")
        print(f"[Tool Arguments {i}]: {json.dumps(step.get('tool_input'))}")
    
    print("\n[AI Copilot Final Synthesized Decision]:")
    print(res4.get("response"))

if __name__ == "__main__":
    run_all_demos()
