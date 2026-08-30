"""
NLP-based address/hub-name parser.

Extracts structured fields (city, state, hub_type, hub_id) from raw
Delhivery center name strings such as:
  "Bengaluru_Whitefield_Gateway_Hub"
  "Delhi_Okhla_Phase2_DC"
  "Mumbai (Bhiwandi) - Fulfillment Center"

Pipeline:
  1. Regex normalization + tokenization
  2. spaCy NER for GPE (geo-political entity) detection
  3. Keyword matching for hub_type classification
  4. Fuzzy fallback against known city list
"""

import re
import difflib
import logging
from functools import lru_cache
from pathlib import Path
from typing import Optional

import pandas as pd

log = logging.getLogger(__name__)

KNOWN_CITIES = [
    "Delhi", "Mumbai", "Bengaluru", "Bangalore", "Hyderabad", "Chennai",
    "Kolkata", "Pune", "Ahmedabad", "Jaipur", "Lucknow", "Chandigarh",
    "Kochi", "Bhubaneswar", "Indore", "Nagpur", "Surat", "Patna",
    "Bhopal", "Agra", "Varanasi", "Gurgaon", "Noida", "Faridabad",
    "Ghaziabad", "Meerut", "Rajkot", "Vadodara", "Coimbatore",
    "Visakhapatnam", "Vijayawada", "Nashik", "Aurangabad", "Amritsar",
    "Ludhiana", "Jodhpur", "Udaipur", "Raipur", "Ranchi", "Guwahati",
    "Bhiwandi", "Kundli", "Manesar",
]

STATE_KEYWORDS = {
    "Karnataka": ["bengaluru", "bangalore", "mysuru", "hubli"],
    "Maharashtra": ["mumbai", "pune", "nashik", "nagpur", "bhiwandi", "aurangabad"],
    "Delhi": ["delhi", "new delhi", "noida", "gurgaon", "faridabad", "ghaziabad"],
    "Telangana": ["hyderabad", "warangal"],
    "Tamil Nadu": ["chennai", "coimbatore", "madurai"],
    "West Bengal": ["kolkata", "howrah"],
    "Gujarat": ["ahmedabad", "surat", "rajkot", "vadodara"],
    "Rajasthan": ["jaipur", "jodhpur", "udaipur"],
    "Uttar Pradesh": ["lucknow", "agra", "varanasi", "meerut", "noida"],
    "Punjab": ["chandigarh", "ludhiana", "amritsar"],
    "Kerala": ["kochi", "kozhikode"],
    "Odisha": ["bhubaneswar"],
    "Madhya Pradesh": ["indore", "bhopal"],
    "Haryana": ["gurgaon", "faridabad", "manesar", "kundli"],
    "Assam": ["guwahati"],
}

HUB_TYPE_KEYWORDS = {
    "gateway_hub":          ["gateway", "gw", "gh"],
    "fulfillment_center":   ["fulfillment", "fc", "dc", "distribution"],
    "last_mile_hub":        ["lm", "last_mile", "delivery", "dlv"],
    "sorting_center":       ["sort", "sorting", "sc"],
    "pickup_point":         ["pickup", "pp", "collection"],
    "transit_hub":          ["transit", "th", "relay"],
    "intercity_hub":        ["ic_hub", "intercity", "ic"],
}


def _normalize(name: str) -> str:
    name = name.replace("_", " ").replace("-", " ").replace("(", " ").replace(")", " ")
    return re.sub(r"\s+", " ", name).strip()


def _infer_state(city: str) -> Optional[str]:
    c_lower = city.lower()
    for state, keywords in STATE_KEYWORDS.items():
        if any(kw in c_lower for kw in keywords):
            return state
    return None


def _classify_hub_type(tokens: list[str]) -> str:
    joined = " ".join(tokens).lower()
    for hub_type, keywords in HUB_TYPE_KEYWORDS.items():
        if any(kw in joined for kw in keywords):
            return hub_type
    return "unknown"


@lru_cache(maxsize=1)
def _load_spacy():
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        log.info("spaCy model loaded: en_core_web_sm")
        return nlp
    except Exception as e:
        log.warning(f"spaCy unavailable ({e}). Falling back to regex+fuzzy only.")
        return None


def _spacy_city(text: str, nlp) -> Optional[str]:
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ in ("GPE", "LOC"):
            matches = difflib.get_close_matches(ent.text, KNOWN_CITIES, n=1, cutoff=0.7)
            if matches:
                return matches[0]
    return None


def parse_hub_name(name: str) -> dict:
    if not isinstance(name, str) or not name.strip():
        return {"raw": name, "city": None, "state": None, "hub_type": "unknown", "hub_id": None}

    normalized = _normalize(name)
    tokens = normalized.split()

    # Try spaCy first
    city = None
    nlp = _load_spacy()
    if nlp:
        city = _spacy_city(normalized, nlp)

    # Fuzzy fallback
    if city is None:
        for token in tokens:
            if len(token) >= 4:
                matches = difflib.get_close_matches(token.title(), KNOWN_CITIES, n=1, cutoff=0.75)
                if matches:
                    city = matches[0]
                    break

    state = _infer_state(city) if city else None
    hub_type = _classify_hub_type(tokens)

    # hub_id: last numeric-looking token, else None
    hub_id = next((t for t in reversed(tokens) if t.isdigit() or re.match(r"ph\d", t.lower())), None)

    return {
        "raw": name,
        "city": city,
        "state": state,
        "hub_type": hub_type,
        "hub_id": hub_id,
    }


def parse_series(names: pd.Series) -> pd.DataFrame:
    results = names.apply(parse_hub_name)
    return pd.DataFrame(list(results))


def enrich_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Add parsed fields for source and destination center names."""
    for prefix, col in [("src", "source_name"), ("dst", "destination_name")]:
        if col in df.columns:
            parsed = parse_series(df[col])
            parsed.columns = [f"{prefix}_{c}" if c != "raw" else c for c in parsed.columns]
            parsed.drop(columns=["raw"], errors="ignore", inplace=True)
            df = pd.concat([df.reset_index(drop=True), parsed.reset_index(drop=True)], axis=1)
    return df


if __name__ == "__main__":
    samples = [
        "Bengaluru_Whitefield_Gateway_Hub",
        "Delhi_Okhla_Phase2_DC",
        "Mumbai (Bhiwandi) - Fulfillment Center",
        "Chennai_Ambattur_Sorting_SC",
        "Hyderabad_Shamshabad_LM",
        "Kolkata_Dankuni_Transit_Hub",
    ]
    for s in samples:
        print(parse_hub_name(s))
