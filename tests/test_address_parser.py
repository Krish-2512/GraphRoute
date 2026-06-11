"""Unit tests for the NLP address parser."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from src.nlp.address_parser import parse_hub_name, KNOWN_CITIES


@pytest.mark.parametrize("name, expected_city", [
    ("Bengaluru_Whitefield_Gateway_Hub", "Bengaluru"),
    ("Delhi_Okhla_Phase2_DC", "Delhi"),
    ("Mumbai (Bhiwandi) - Fulfillment Center", "Mumbai"),
    ("Chennai_Ambattur_Sorting_SC", "Chennai"),
    ("Hyderabad_Shamshabad_LM", "Hyderabad"),
    ("Kolkata_Dankuni_Transit_Hub", "Kolkata"),
    ("Pune_Chakan_GH", "Pune"),
])
def test_city_extraction(name, expected_city):
    result = parse_hub_name(name)
    assert result["city"] == expected_city, f"Expected {expected_city}, got {result['city']} for '{name}'"


@pytest.mark.parametrize("name, expected_hub_type", [
    ("Delhi_Okhla_FC", "fulfillment_center"),
    ("Mumbai_Bhiwandi_Gateway_Hub", "gateway_hub"),
    ("Chennai_LM_Hub", "last_mile_hub"),
    ("Kolkata_Sorting_SC", "sorting_center"),
])
def test_hub_type_extraction(name, expected_hub_type):
    result = parse_hub_name(name)
    assert result["hub_type"] == expected_hub_type, f"Expected {expected_hub_type}, got {result['hub_type']} for '{name}'"


def test_null_input():
    result = parse_hub_name(None)
    assert result["city"] is None
    assert result["hub_type"] == "unknown"


def test_empty_string():
    result = parse_hub_name("")
    assert result["city"] is None


def test_success_rate_over_90pct():
    test_names = [
        "Bengaluru_Whitefield_GH", "Delhi_Okhla_DC",
        "Mumbai_Bhiwandi_FC", "Chennai_Ambattur_SC",
        "Hyderabad_Shamshabad_LM", "Kolkata_Dankuni_TH",
        "Pune_Chakan_GH", "Jaipur_Sitapura_WH",
        "Lucknow_Amausi_FC", "Nagpur_Butibori_SC",
    ]
    results = [parse_hub_name(n) for n in test_names]
    success_rate = sum(1 for r in results if r["city"] is not None) / len(results)
    assert success_rate >= 0.90, f"City extraction success rate {success_rate:.2%} < 90%"
