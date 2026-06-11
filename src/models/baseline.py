"""
ML baseline ETA prediction models: XGBoost + LightGBM.

Two variants:
  1. Baseline — trip-level features only (no graph awareness)
  2. Graph-enhanced baseline — adds centrality + corridor stats as features

Metrics reported:
  * MAE (primary)
  * RMSE
  * % trips with predicted ETA within 15% of actual (business metric)
  * MAPE
"""

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error

log = logging.getLogger(__name__)

MODEL_DIR = Path("data/processed/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ── Feature sets ──────────────────────────────────────────────────────────────

BASE_FEATURES = [
    "osrm_time",
    "osrm_distance",
    "is_ftl",
    "time_of_day_enc",
    "dwell_time_proxy",
]

GRAPH_FEATURES = [
    "corridor_mean_delay",
    "corridor_std_delay",
    "corridor_volume",
    "source_out_degree",
    "dest_in_degree",
    # filled in from analytics.py
    "src_betweenness",
    "dst_betweenness",
    "src_pagerank",
    "dst_pagerank",
    "src_avg_dwell",
]

TARGET = "actual_time"


def _within_15pct(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_pred - y_true) / (y_true + 1e-8) <= 0.15) * 100)


def evaluate(y_true, y_pred, label: str = "") -> dict:
    mae   = mean_absolute_error(y_true, y_pred)
    rmse  = np.sqrt(mean_squared_error(y_true, y_pred))
    w15   = _within_15pct(np.array(y_true), np.array(y_pred))
    mape  = float(np.mean(np.abs((np.array(y_true) - np.array(y_pred)) / (np.array(y_true) + 1e-8))) * 100)
    metrics = {"model": label, "MAE": round(mae, 3), "RMSE": round(rmse, 3),
               "within_15pct": round(w15, 2), "MAPE": round(mape, 2)}
    log.info(f"[{label}] MAE={mae:.2f}  RMSE={rmse:.2f}  within15%={w15:.1f}%  MAPE={mape:.1f}%")
    return metrics


def add_graph_features(df: pd.DataFrame, centrality_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Merge hub-level centrality into trip dataframe."""
    if centrality_df is None:
        try:
            centrality_df = pd.read_csv("data/processed/hub_centrality.csv")
        except FileNotFoundError:
            log.warning("hub_centrality.csv not found; graph features will be zeros.")
            for col in ["src_betweenness", "dst_betweenness", "src_pagerank", "dst_pagerank", "src_avg_dwell"]:
                df[col] = 0.0
            return df

    c = centrality_df[["hub", "betweenness_centrality", "pagerank", "avg_dwell_min"]].rename(
        columns={"hub": "source_name",
                 "betweenness_centrality": "src_betweenness",
                 "pagerank": "src_pagerank",
                 "avg_dwell_min": "src_avg_dwell"}
    )
    df = df.merge(c, on="source_name", how="left")

    c_dst = centrality_df[["hub", "betweenness_centrality", "pagerank"]].rename(
        columns={"hub": "destination_name",
                 "betweenness_centrality": "dst_betweenness",
                 "pagerank": "dst_pagerank"}
    )
    df = df.merge(c_dst, on="destination_name", how="left")
    df[["src_betweenness", "dst_betweenness", "src_pagerank", "dst_pagerank", "src_avg_dwell"]] = \
        df[["src_betweenness", "dst_betweenness", "src_pagerank", "dst_pagerank", "src_avg_dwell"]].fillna(0)
    return df


def train_xgboost(X_train, y_train, X_val, y_val, **kwargs):
    try:
        import xgboost as xgb
    except ImportError:
        log.error("xgboost not installed.")
        return None
    params = dict(
        n_estimators=800,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=50,
        eval_metric="mae",
    )
    params.update(kwargs)
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model


def train_lightgbm(X_train, y_train, X_val, y_val, **kwargs):
    try:
        import lightgbm as lgb
    except ImportError:
        log.error("lightgbm not installed.")
        return None
    params = dict(
        n_estimators=800,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        early_stopping_round=50,
        verbose=-1,
    )
    params.update(kwargs)
    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )
    return model


def shap_importance(model, X_val: pd.DataFrame, model_name: str = "") -> pd.DataFrame:
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_val)
        importance = pd.DataFrame({
            "feature": X_val.columns,
            "mean_abs_shap": np.abs(shap_values).mean(axis=0),
        }).sort_values("mean_abs_shap", ascending=False)
        out = Path("data/processed") / f"shap_{model_name}.csv"
        importance.to_csv(out, index=False)
        log.info(f"SHAP importance saved → {out}")
        return importance
    except ImportError:
        log.warning("shap not installed.")
        return pd.DataFrame()


def run(df_path: str | Path | None = None, use_graph_features: bool = True) -> list[dict]:
    if df_path is None:
        df_path = "data/processed/features.parquet"
    df = pd.read_parquet(df_path)

    if use_graph_features:
        df = add_graph_features(df)

    feature_cols = BASE_FEATURES + (GRAPH_FEATURES if use_graph_features else [])
    available = [c for c in feature_cols if c in df.columns]
    X = df[available].fillna(0).values
    y = df[TARGET].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=42)
    X_df_val = pd.DataFrame(X_val, columns=available)

    results = []

    # XGBoost
    xgb_model = train_xgboost(X_tr, y_tr, X_val, y_val)
    if xgb_model:
        label = "XGBoost+Graph" if use_graph_features else "XGBoost_Baseline"
        results.append(evaluate(y_test, xgb_model.predict(X_test), label=label))
        shap_importance(xgb_model, X_df_val, model_name=label)
        with open(MODEL_DIR / f"{label}.pkl", "wb") as f:
            pickle.dump(xgb_model, f)

    # LightGBM
    lgb_model = train_lightgbm(X_tr, y_tr, X_val, y_val)
    if lgb_model:
        label = "LightGBM+Graph" if use_graph_features else "LightGBM_Baseline"
        results.append(evaluate(y_test, lgb_model.predict(X_test), label=label))
        with open(MODEL_DIR / f"{label}.pkl", "wb") as f:
            pickle.dump(lgb_model, f)

    return results


if __name__ == "__main__":
    r1 = run(use_graph_features=False)
    r2 = run(use_graph_features=True)
    summary = pd.DataFrame(r1 + r2)
    print(summary.to_string(index=False))
