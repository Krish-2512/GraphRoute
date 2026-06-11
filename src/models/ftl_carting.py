"""
ML-backed FTL vs Carting decision framework.

Given a new trip (distance, time_of_day, corridor delay profile, source hub
graph position), recommend whether FTL or Carting will yield better on-time
delivery — and quantify the time-cost trade-off.

Model: LightGBM binary classifier with calibrated probabilities.
Decision boundary: P(FTL) > threshold → recommend FTL.

Outputs:
  * route_type_recommendation  ("FTL" | "Carting")
  * ftl_probability            (0–1)
  * expected_time_saving_min   (positive = FTL faster)
  * cost_premium_pct           (FTL typically costs 20–40% more)
  * decision_reason            (plain-text explanation)
"""

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import LabelEncoder

log = logging.getLogger(__name__)

MODEL_DIR = Path("data/processed/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# FTL is typically 20–40% more expensive per km
FTL_COST_PREMIUM_PCT = 30.0
# FTL recommendation threshold — lower = more aggressive FTL recommendation
FTL_THRESHOLD = 0.55

FEATURES = [
    "osrm_distance",
    "time_of_day_enc",
    "corridor_mean_delay",
    "corridor_volume",
    "src_betweenness",
    "src_pagerank",
    "dwell_time_proxy",
    "is_intercity",
]


def _label_encode(df: pd.DataFrame) -> pd.Series:
    if "route_type" not in df.columns:
        raise ValueError("route_type column missing.")
    return (df["route_type"].str.upper() == "FTL").astype(int)


def build_training_set(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    available = [c for c in FEATURES if c in df.columns]
    X = df[available].fillna(0)
    y = _label_encode(df)
    return X, y


def train(df: pd.DataFrame) -> object:
    try:
        import lightgbm as lgb
    except ImportError:
        log.error("lightgbm not installed.")
        return None

    X, y = build_training_set(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    base_model = lgb.LGBMClassifier(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
        verbose=-1,
    )
    model = CalibratedClassifierCV(base_model, cv=5, method="isotonic")
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= FTL_THRESHOLD).astype(int)

    auc = roc_auc_score(y_test, y_prob)
    report = classification_report(y_test, y_pred, target_names=["Carting", "FTL"])
    log.info(f"FTL Classifier — AUC: {auc:.4f}\n{report}")

    path = MODEL_DIR / "ftl_carting_model.pkl"
    with open(path, "wb") as f:
        pickle.dump({"model": model, "features": [c for c in FEATURES if c in df.columns]}, f)
    log.info(f"FTL/Carting model saved → {path}")
    return model


def load_model() -> dict:
    path = MODEL_DIR / "ftl_carting_model.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)


def _time_saving_estimate(
    dist_km: float,
    time_of_day: str,
    corridor_delay: float,
    is_ftl_prob: float,
) -> float:
    """
    Heuristic: FTL saves time mainly on long hauls with high base delay.
    Returns estimated minutes saved (positive = FTL faster).
    """
    base_time_hr = dist_km / 55.0
    delay_overhead = base_time_hr * 60 * (corridor_delay - 1.0)
    # FTL reduces delay by ~35% on average (direct loading, no transshipment)
    ftl_reduction = delay_overhead * 0.35 * is_ftl_prob
    # Evening/morning FTL advantage is smaller (traffic affects both)
    if time_of_day in ("morning", "evening"):
        ftl_reduction *= 0.75
    return round(ftl_reduction, 1)


def _decision_reason(prob: float, dist_km: float, time_saving: float) -> str:
    if prob >= FTL_THRESHOLD:
        if dist_km > 600:
            return (
                f"Long-haul corridor ({dist_km:.0f} km). FTL removes transshipment overhead "
                f"and is projected to save ~{time_saving:.0f} min vs Carting."
            )
        return (
            f"High delay-ratio corridor. FTL's direct routing is projected to "
            f"save ~{time_saving:.0f} min (P(FTL better)={prob:.2f})."
        )
    else:
        return (
            f"Carting is preferred. Short haul or low delay corridor — transshipment "
            f"overhead is justified by {FTL_COST_PREMIUM_PCT:.0f}% lower cost vs FTL."
        )


def predict_route_type(
    trips: pd.DataFrame,
    model_bundle: dict | None = None,
) -> pd.DataFrame:
    """
    Parameters
    ----------
    trips : DataFrame with at minimum [osrm_distance, time_of_day_enc,
            corridor_mean_delay, dwell_time_proxy]
    model_bundle : loaded model dict (loads from disk if None)

    Returns
    -------
    DataFrame with added columns:
      route_type_recommendation, ftl_probability,
      expected_time_saving_min, cost_premium_pct, decision_reason
    """
    if model_bundle is None:
        model_bundle = load_model()

    model   = model_bundle["model"]
    feat_cols = model_bundle["features"]
    available = [c for c in feat_cols if c in trips.columns]
    X = trips[available].fillna(0)

    probs = model.predict_proba(X)[:, 1]
    recommendations = np.where(probs >= FTL_THRESHOLD, "FTL", "Carting")

    tod_map_rev = {0: "night", 1: "morning", 2: "afternoon", 3: "evening"}
    results = trips.copy()
    results["ftl_probability"] = probs.round(4)
    results["route_type_recommendation"] = recommendations
    results["cost_premium_pct"] = np.where(probs >= FTL_THRESHOLD, FTL_COST_PREMIUM_PCT, 0.0)

    time_savings = []
    reasons = []
    for i, row in results.iterrows():
        tod = tod_map_rev.get(int(row.get("time_of_day_enc", 0)), "morning")
        ts = _time_saving_estimate(
            dist_km=row.get("osrm_distance", 300),
            time_of_day=tod,
            corridor_delay=row.get("corridor_mean_delay", 1.1),
            is_ftl_prob=float(probs[results.index.get_loc(i)]),
        )
        time_savings.append(ts)
        reasons.append(_decision_reason(
            prob=float(probs[results.index.get_loc(i)]),
            dist_km=row.get("osrm_distance", 300),
            time_saving=ts,
        ))
    results["expected_time_saving_min"] = time_savings
    results["decision_reason"] = reasons
    return results


if __name__ == "__main__":
    sample = pd.DataFrame({
        "osrm_distance": [120, 800, 350, 1200],
        "time_of_day_enc": [1, 3, 2, 0],
        "corridor_mean_delay": [1.05, 1.45, 1.15, 1.60],
        "corridor_volume": [200, 50, 150, 30],
        "src_betweenness": [0.1, 0.4, 0.2, 0.05],
        "src_pagerank": [0.02, 0.08, 0.03, 0.01],
        "dwell_time_proxy": [5, 25, 10, 40],
        "is_intercity": [0, 1, 0, 1],
        "route_type": ["Carting", "FTL", "Carting", "FTL"],
    })
    model = train(sample)
    if model:
        results = predict_route_type(sample)
        print(results[["route_type_recommendation", "ftl_probability", "expected_time_saving_min", "decision_reason"]])
