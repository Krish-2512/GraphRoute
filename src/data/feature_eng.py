"""
Feature engineering on top of the cleaned Delhivery dataset.

Produces a feature matrix ready for ML/DL model training.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

log = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")


def compute_delay_ratio(df: pd.DataFrame) -> pd.DataFrame:
    df["delay_ratio"] = df["actual_time"] / df["osrm_time"]
    df["delay_minutes"] = df["actual_time"] - df["osrm_time"]
    return df


def compute_dwell_time(df: pd.DataFrame) -> pd.DataFrame:
    """Proxy for hub processing/dwell time using segment-level columns."""
    seg_actual = next((c for c in df.columns if "segment_actual" in c), None)
    seg_osrm   = next((c for c in df.columns if "segment_osrm_time" in c), None)
    if seg_actual and seg_osrm:
        df["dwell_time_proxy"] = (df[seg_actual] - df[seg_osrm]).clip(lower=0)
    else:
        df["dwell_time_proxy"] = 0.0
    return df


def compute_speed_efficiency(df: pd.DataFrame) -> pd.DataFrame:
    dist_col = next((c for c in df.columns if "distance" in c and "osrm" in c), None)
    if dist_col and "actual_time" in df.columns:
        df["speed_efficiency"] = df[dist_col] / (df["actual_time"].clip(lower=1))
    return df


def encode_route_type(df: pd.DataFrame) -> pd.DataFrame:
    if "route_type" in df.columns:
        df["is_ftl"] = (df["route_type"].str.upper() == "FTL").astype(int)
    return df


def encode_time_of_day(df: pd.DataFrame) -> pd.DataFrame:
    if "time_of_day" in df.columns:
        mapping = {"night": 0, "morning": 1, "afternoon": 2, "evening": 3}
        df["time_of_day_enc"] = df["time_of_day"].astype(str).str.lower().map(mapping).fillna(0).astype(int)
    return df


def compute_corridor_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Historical per-corridor aggregates used as lookup features."""
    if "source_name" not in df.columns or "destination_name" not in df.columns:
        return df
    stats = (
        df.groupby(["source_name", "destination_name"])
        .agg(
            corridor_mean_delay=("delay_ratio", "mean"),
            corridor_std_delay=("delay_ratio", "std"),
            corridor_volume=("delay_ratio", "count"),
        )
        .reset_index()
    )
    df = df.merge(stats, on=["source_name", "destination_name"], how="left")
    return df


def compute_hub_degree(df: pd.DataFrame) -> pd.DataFrame:
    """Approximate out-degree and in-degree of each hub from trip counts."""
    if "source_name" not in df.columns or "destination_name" not in df.columns:
        return df
    out_deg = df.groupby("source_name").size().rename("source_out_degree")
    in_deg  = df.groupby("destination_name").size().rename("dest_in_degree")
    df = df.join(out_deg, on="source_name").join(in_deg, on="destination_name")
    return df


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    df = compute_delay_ratio(df)
    df = compute_dwell_time(df)
    df = compute_speed_efficiency(df)
    df = encode_route_type(df)
    df = encode_time_of_day(df)
    df = compute_corridor_stats(df)
    df = compute_hub_degree(df)
    log.info(f"Feature matrix ready: {df.shape[1]} columns")
    return df


FEATURE_COLS = [
    "osrm_time",
    "osrm_distance",
    "is_ftl",
    "time_of_day_enc",
    "dwell_time_proxy",
    "speed_efficiency",
    "corridor_mean_delay",
    "corridor_std_delay",
    "corridor_volume",
    "source_out_degree",
    "dest_in_degree",
]
TARGET_COL = "actual_time"


def get_xy(df: pd.DataFrame):
    available = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available].fillna(0)
    y = df[TARGET_COL]
    return X, y


def save(df: pd.DataFrame, name: str = "features.parquet") -> Path:
    out = PROCESSED_DIR / name
    df.to_parquet(out, index=False)
    log.info(f"Saved features → {out}")
    return out


def run(clean_path: str | Path | None = None) -> pd.DataFrame:
    if clean_path is None:
        clean_path = PROCESSED_DIR / "delhivery_clean.parquet"
    df = pd.read_parquet(clean_path)
    df = build_feature_matrix(df)
    save(df)
    return df


if __name__ == "__main__":
    run()
