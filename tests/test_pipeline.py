"""
Unit and Integration Tests for GraphRoute Logistics System.
"""

import unittest
import torch
import networkx as nx
import pandas as pd
import numpy as np

from src.nlp.address_parser import parse_hub_name
from src.graph.builder import build_graph
from src.graph.analytics import compute_centrality, compute_sla_breach_contribution
from src.graph.simulator import NetworkSimulator
from src.models.gnn_layers import GraphSAGEETAModel, GATETAModel
from src.agent.tools import HubHealthTool, WhatIfSimulationTool, RouteAdvisorTool
from src.agent.ops_copilot import NetworkOpsCopilot


class TestGraphRoutePipeline(unittest.TestCase):

    def test_address_parser(self):
        res = parse_hub_name("Bengaluru_Whitefield_Gateway_Hub")
        self.assertEqual(res["city"], "Bengaluru")
        self.assertEqual(res["state"], "Karnataka")
        self.assertEqual(res["hub_type"], "gateway_hub")

    def test_graph_construction(self):
        sample = pd.DataFrame({
            "source_name": ["Delhi_Hub", "Mumbai_Hub"],
            "destination_name": ["Mumbai_Hub", "Chennai_Hub"],
            "route_type": ["FTL", "Carting"],
            "delay_ratio": [1.3, 1.1],
            "dwell_time_proxy": [15.0, 8.0],
            "osrm_time": [600.0, 900.0],
            "actual_time": [780.0, 990.0],
            "osrm_distance": [1400.0, 1330.0],
        })
        G = build_graph(sample)
        self.assertEqual(G.number_of_nodes(), 3)
        self.assertEqual(G.number_of_edges(), 2)

    def test_whatif_simulator(self):
        G = nx.DiGraph()
        G.add_node("Delhi_Okhla_DC", city="Delhi", hub_type="fulfillment_center", avg_dwell=38.0)
        G.add_node("Mumbai_Bhiwandi_GH", city="Mumbai", hub_type="gateway_hub", avg_dwell=30.0)
        G.add_edge("Delhi_Okhla_DC", "Mumbai_Bhiwandi_GH", volume=450, median_delay_ratio=1.45, osrm_time=600, route_type="FTL")

        sim = NetworkSimulator(G)
        res = sim.simulate_hub_upgrade("Delhi_Okhla_DC", capacity_boost_pct=30.0)
        self.assertIn("monthly_revenue_recovered_lakhs", res)
        self.assertGreater(res["monthly_breaches_avoided"], 0)
        self.assertLess(res["simulated_dwell_min"], 38.0)

    def test_gnn_forward_pass(self):
        x = torch.randn(4, 5)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
        edge_attr = torch.randn(3, 4)

        sage = GraphSAGEETAModel(node_in_dim=5, edge_in_dim=4, hidden_dim=16, out_dim=8)
        out_sage = sage(x, edge_index, edge_attr)
        self.assertEqual(out_sage.shape, torch.Size([3]))

        gat = GATETAModel(node_in_dim=5, edge_in_dim=4, hidden_dim=8, heads=2, out_dim=8)
        out_gat = gat(x, edge_index, edge_attr)
        self.assertEqual(out_gat.shape, torch.Size([3]))

    def test_agent_tools(self):
        route_tool = RouteAdvisorTool()
        res_route = route_tool.run(distance_km=850.0, time_of_day="morning", historical_delay_ratio=1.4)
        self.assertIn("recommended_mode", res_route)

    def test_ops_copilot_reasoning(self):
        copilot = NetworkOpsCopilot()
        res = copilot.run_agentic_pipeline("Should we use FTL for 800km trip?")
        self.assertIn("response", res)
        self.assertTrue(len(res["steps"]) > 0)


if __name__ == "__main__":
    unittest.main()
