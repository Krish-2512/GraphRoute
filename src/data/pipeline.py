"""
Data ingestion, cleaning, and merging pipeline for the Delhivery dataset.

Expected raw file: data/raw/delhivery_data.csv
Kaggle dataset: https://www.kaggle.com/datasets/himanshurana001/delhivery-delivery-dataset
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

DELAY_RATIO_LOWER = 0.5   # trips where actual < 50% OSRM are likely data errors
DELAY_RATIO_UPPER = 5.0   # extreme outliers (> 5x OSRM)
CHRONIC_THRESHOLD = 1.20  # corridor is "chronic" if median delay ratio > 20% over OSRM
CHRONIC_MIN_TRIPS  = 5    # minimum trips to qualify a corridor as chronic


def load_raw(path: str | Path | None = None) -> pd.DataFrame:
    if path is None:
        candidates = list(RAW_DIR.glob("*.csv"))
        if not candidates:
            raise FileNotFoundError(
                f"No CSV found in {RAW_DIR}. Download the Delhivery dataset from Kaggle "
                "and place it in data/raw/."
            )
        path = candidates[0]
    log.info(f"Loading raw data from {path}")
    df = pd.read_csv(path, low_memory=False)
    log.info(f"Raw shape: {df.shape}")
    return df


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = (
        df.columns.str.strip()
                  .str.lower()
                  .str.replace(r"[ \-]", "_", regex=True)
    )
    return df


def parse_datetimes(df: pd.DataFrame) -> pd.DataFrame:
    # Only parse columns that are actual timestamps, not numeric duration columns.
    # Columns like actual_time / osrm_time / segment_*_time are minutes (float),
    # not datetime strings — converting them breaks remove_outliers.
    TIMESTAMP_KEYWORDS = ("creation", "start", "end", "date", "timestamp", "cutoff")
    SKIP_KEYWORDS = ("actual_time", "osrm_time", "segment_actual", "segment_osrm")

    timestamp_cols = [
        c for c in df.columns
        if any(kw in c for kw in TIMESTAMP_KEYWORDS)
        and not any(skip in c for skip in SKIP_KEYWORDS)
    ]
    for col in timestamp_cols:
        try:
            df[col] = pd.to_datetime(df[col], errors="coerce")
        except Exception:
            pass
    return df


def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    if "actual_time" in df.columns and "osrm_time" in df.columns:
        df = df[df["osrm_time"] > 0].copy()
        df["delay_ratio_raw"] = df["actual_time"] / df["osrm_time"]
        before = len(df)
        df = df[
            (df["delay_ratio_raw"] >= DELAY_RATIO_LOWER) &
            (df["delay_ratio_raw"] <= DELAY_RATIO_UPPER)
        ]
        log.info(f"Outlier removal: {before - len(df)} rows dropped ({before} → {len(df)})")
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    trip_col = next(
        (c for c in df.columns if "trip_creation" in c or "od_start" in c), None
    )
    if trip_col and pd.api.types.is_datetime64_any_dtype(df[trip_col]):
        df["hour"] = df[trip_col].dt.hour
        df["day_of_week"] = df[trip_col].dt.dayofweek
        df["month"] = df[trip_col].dt.month
        df["time_of_day"] = pd.cut(
            df["hour"],
            bins=[-1, 6, 12, 18, 23],
            labels=["night", "morning", "afternoon", "evening"],
        )
        log.info("Time-of-day features added.")
    return df


def flag_chronic_corridors(df: pd.DataFrame) -> pd.DataFrame:
    if "source_name" not in df.columns or "destination_name" not in df.columns:
        return df
    if "delay_ratio_raw" not in df.columns:
        return df
    corridor_stats = (
        df.groupby(["source_name", "destination_name"])["delay_ratio_raw"]
        .agg(["median", "count"])
        .reset_index()
        .rename(columns={"median": "corridor_median_delay", "count": "corridor_trip_count"})
    )
    corridor_stats["is_chronic_corridor"] = (
        (corridor_stats["corridor_median_delay"] > CHRONIC_THRESHOLD) &
        (corridor_stats["corridor_trip_count"] >= CHRONIC_MIN_TRIPS)
    )
    df = df.merge(corridor_stats, on=["source_name", "destination_name"], how="left")
    n_chronic = corridor_stats["is_chronic_corridor"].sum()
    log.info(f"Chronic corridors flagged: {n_chronic} / {len(corridor_stats)}")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = standardize_columns(df)
    df = parse_datetimes(df)
    df.drop_duplicates(inplace=True)
    df.dropna(subset=["actual_time", "osrm_time"], inplace=True)
    df = remove_outliers(df)
    df = add_time_features(df)
    df = flag_chronic_corridors(df)
    return df


def save(df: pd.DataFrame, name: str = "delhivery_clean.parquet") -> Path:
    out = PROCESSED_DIR / name
    df.to_parquet(out, index=False)
    log.info(f"Saved cleaned data → {out}  ({len(df)} rows, {df.shape[1]} cols)")
    return out


def run(raw_path: str | Path | None = None) -> pd.DataFrame:
    df = load_raw(raw_path)
    df = clean(df)
    save(df)
    return df


if __name__ == "__main__":
    run()
