"""
Multi-city extension: extracts city tags from hub names and generates a
synthetic inter-city corridor layer covering 15 major Indian cities.

The real dataset's source_name / destination_name strings already encode
city information (e.g. "Bengaluru_Whitefield_Gateway"). This module:
  1. Parses city from hub name strings.
  2. Groups facilities by city.
  3. Generates synthetic inter-city trip records so the graph has both
     intra-city and inter-city edges.
"""

import re
import random
import numpy as np
import pandas as pd
from pathlib import Path
import logging

log = logging.getLogger(__name__)

CITY_AUGMENTED_DIR = Path("data/city_augmented")
CITY_AUGMENTED_DIR.mkdir(parents=True, exist_ok=True)

MAJOR_CITIES = [
    "Delhi", "Mumbai", "Bengaluru", "Hyderabad", "Chennai",
    "Kolkata", "Pune", "Ahmedabad", "Jaipur", "Lucknow",
    "Chandigarh", "Kochi", "Bhubaneswar", "Indore", "Nagpur",
]

# Approximate inter-city OSRM distances (km) as a symmetric lookup.
# Values are rough road distances; delay ratios are calibrated synthetically.
CITY_COORDS = {
    "Delhi":       (28.6, 77.2),  "Mumbai":      (19.1, 72.9),
    "Bengaluru":   (12.9, 77.6),  "Hyderabad":   (17.4, 78.5),
    "Chennai":     (13.1, 80.3),  "Kolkata":      (22.6, 88.4),
    "Pune":        (18.5, 73.9),  "Ahmedabad":   (23.0, 72.6),
    "Jaipur":      (26.9, 75.8),  "Lucknow":     (26.8, 80.9),
    "Chandigarh":  (30.7, 76.8),  "Kochi":       (9.9, 76.3),
    "Bhubaneswar": (20.3, 85.8),  "Indore":      (22.7, 75.9),
    "Nagpur":      (21.1, 79.1),
}

# Haversine distance in km
def _haversine(lat1, lon1, lat2, lon2):
    R = 6371
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


_CITY_PATTERNS = [
    r"^([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)",        # "Bengaluru Whitefield..."
    r"^([A-Za-z]+)[-_]",                          # "Delhi-" or "Delhi_"
    r"([A-Z][a-z]{2,})",                           # first title-case word
]

def extract_city(name: str) -> str | None:
    if not isinstance(name, str):
        return None
    for pattern in _CITY_PATTERNS:
        m = re.search(pattern, name.strip())
        if m:
            candidate = m.group(1).strip().title()
            # Fuzzy match against known cities
            for city in MAJOR_CITIES:
                if city.lower() in candidate.lower() or candidate.lower() in city.lower():
                    return city
    # fallback: first token
    token = re.split(r"[\s_\-]", name.strip())[0].title()
    return token if len(token) > 2 else None


def tag_cities(df: pd.DataFrame) -> pd.DataFrame:
    if "source_name" in df.columns:
        df["source_city"] = df["source_name"].apply(extract_city)
    if "destination_name" in df.columns:
        df["dest_city"] = df["destination_name"].apply(extract_city)
    known = set(MAJOR_CITIES)
    if "source_city" in df.columns:
        df["source_city_known"] = df["source_city"].isin(known)
    if "dest_city" in df.columns:
        df["dest_city_known"] = df["dest_city"].isin(known)
    return df


def generate_intercity_trips(n_per_pair: int = 30, seed: int = 42) -> pd.DataFrame:
    """
    Synthesize trip records for every pair of major cities.
    Delay ratios are drawn from calibrated distributions:
      - Short haul (<400 km): lower delay, higher variance (more last-mile issues)
      - Long haul (>400 km): higher base delay (more highway variability)
    """
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    records = []
    cities = MAJOR_CITIES
    for i, src in enumerate(cities):
        for j, dst in enumerate(cities):
            if src == dst:
                continue
            lat1, lon1 = CITY_COORDS[src]
            lat2, lon2 = CITY_COORDS[dst]
            dist_km = _haversine(lat1, lon1, lat2, lon2)
            # OSRM time: assume avg 55 km/h on highways
            osrm_minutes = (dist_km / 55) * 60
            # Delay calibration
            if dist_km < 400:
                delay_mean, delay_std = 1.15, 0.18
            else:
                delay_mean, delay_std = 1.28, 0.22
            for _ in range(n_per_pair):
                route_type = rng.choice(["FTL", "Carting"], weights=[0.4, 0.6])
                tod = rng.choice(["morning", "afternoon", "evening", "night"])
                tod_factor = {"morning": 1.05, "afternoon": 1.10, "evening": 1.20, "night": 0.95}[tod]
                delay_ratio = float(np_rng.normal(delay_mean * tod_factor, delay_std))
                delay_ratio = max(0.6, min(4.5, delay_ratio))
                actual_minutes = osrm_minutes * delay_ratio
                records.append({
                    "source_name": f"{src}_IC_Hub",
                    "destination_name": f"{dst}_IC_Hub",
                    "source_city": src,
                    "dest_city": dst,
                    "route_type": route_type,
                    "time_of_day": tod,
                    "osrm_time": round(osrm_minutes, 2),
                    "osrm_distance": round(dist_km, 2),
                    "actual_time": round(actual_minutes, 2),
                    "delay_ratio": round(delay_ratio, 4),
                    "is_intercity": True,
                    "is_ftl": int(route_type == "FTL"),
                })
    df = pd.DataFrame(records)
    log.info(f"Generated {len(df)} synthetic inter-city trips across {len(cities)} cities.")
    return df


def run(clean_path: str | Path | None = None) -> pd.DataFrame:
    if clean_path is None:
        clean_path = Path("data/processed/delhivery_clean.parquet")

    df_real = pd.read_parquet(clean_path)
    df_real = tag_cities(df_real)
    df_real["is_intercity"] = False

    df_synth = generate_intercity_trips()

    combined = pd.concat([df_real, df_synth], ignore_index=True)
    out = CITY_AUGMENTED_DIR / "multi_city_trips.parquet"
    combined.to_parquet(out, index=False)
    log.info(f"Multi-city dataset saved → {out}  ({len(combined)} rows)")
    return combined


if __name__ == "__main__":
    run()
