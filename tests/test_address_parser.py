"""Unit tests for the NLP address parser."""

import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.nlp.address_parser import parse_hub_name, KNOWN_CITIES


class TestAddressParser(unittest.TestCase):

    def test_city_extraction(self):
        cases = [
            ("Bengaluru_Whitefield_Gateway_Hub", "Bengaluru"),
            ("Delhi_Okhla_Phase2_DC", "Delhi"),
            ("Mumbai (Bhiwandi) - Fulfillment Center", "Mumbai"),
            ("Chennai_Ambattur_Sorting_SC", "Chennai"),
            ("Hyderabad_Shamshabad_LM", "Hyderabad"),
            ("Kolkata_Dankuni_Transit_Hub", "Kolkata"),
            ("Pune_Chakan_GH", "Pune"),
        ]
        for name, expected_city in cases:
            res = parse_hub_name(name)
            self.assertEqual(res["city"], expected_city)

    def test_hub_type_extraction(self):
        cases = [
            ("Delhi_Okhla_FC", "fulfillment_center"),
            ("Mumbai_Bhiwandi_Gateway_Hub", "gateway_hub"),
            ("Chennai_LM_Hub", "last_mile_hub"),
            ("Kolkata_Sorting_SC", "sorting_center"),
        ]
        for name, expected_type in cases:
            res = parse_hub_name(name)
            self.assertEqual(res["hub_type"], expected_type)

    def test_null_and_empty(self):
        res_null = parse_hub_name(None)
        self.assertIsNone(res_null["city"])
        self.assertEqual(res_null["hub_type"], "unknown")

        res_empty = parse_hub_name("")
        self.assertIsNone(res_empty["city"])

    def test_success_rate_over_90pct(self):
        test_names = [
            "Bengaluru_Whitefield_GH", "Delhi_Okhla_DC",
            "Mumbai_Bhiwandi_FC", "Chennai_Ambattur_SC",
            "Hyderabad_Shamshabad_LM", "Kolkata_Dankuni_TH",
            "Pune_Chakan_GH", "Jaipur_Sitapura_WH",
            "Lucknow_Amausi_FC", "Nagpur_Butibori_SC",
        ]
        results = [parse_hub_name(n) for n in test_names]
        success_rate = sum(1 for r in results if r["city"] is not None) / len(results)
        self.assertGreaterEqual(success_rate, 0.90)


if __name__ == "__main__":
    unittest.main()
