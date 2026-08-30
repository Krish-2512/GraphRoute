"""
Statistical Significance Testing & Hypothesis Validation for Supply Chain ML.

Conducts:
1. 5-Fold Cross-Validation Error Distribution Analysis
2. Paired Student's t-test and Wilcoxon Signed-Rank Test (GNN/Graph vs. Baseline)
3. Kolmogorov-Smirnov (KS) Test for Peak vs. Off-Peak Delay Distribution Shift
"""

import logging
import json
import sys
from pathlib import Path
from typing import Dict, Any

root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import KFold
import lightgbm as lgb

log = logging.getLogger(__name__)


def run_statistical_significance_tests():
    from src.models.baseline import add_graph_features, BASE_FEATURES, GRAPH_FEATURES, TARGET

    df = pd.read_parquet("data/processed/features.parquet")
    df = add_graph_features(df)

    # 1. 5-Fold Cross-Validation Comparison
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    X_base = df[BASE_FEATURES].fillna(0).values
    X_graph = df[BASE_FEATURES + GRAPH_FEATURES].fillna(0).values
    y = df[TARGET].values

    base_maes, graph_maes = [], []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_base)):
        # Baseline model
        m_base = lgb.LGBMRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, verbose=-1)
        m_base.fit(X_base[train_idx], y[train_idx])
        pred_base = m_base.predict(X_base[val_idx])
        base_maes.append(float(np.mean(np.abs(pred_base - y[val_idx]))))

        # Graph model
        m_graph = lgb.LGBMRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, verbose=-1)
        m_graph.fit(X_graph[train_idx], y[train_idx])
        pred_graph = m_graph.predict(X_graph[val_idx])
        graph_maes.append(float(np.mean(np.abs(pred_graph - y[val_idx]))))

    # 2. Hypothesis Testing: Is Graph significantly better than Baseline?
    t_stat, t_pval = stats.ttest_rel(base_maes, graph_maes)
    w_stat, w_pval = stats.wilcoxon(base_maes, graph_maes)

    # 3. Kolmogorov-Smirnov Test: Day vs Night Delay Distributions
    day_delays = df[df["time_of_day_enc"].isin([1, 2, 3])]["delay_ratio"].dropna().values
    night_delays = df[df["time_of_day_enc"] == 0]["delay_ratio"].dropna().values
    ks_stat, ks_pval = stats.ks_2samp(day_delays, night_delays)

    results = {
        "5fold_baseline_mae_mean": round(float(np.mean(base_maes)), 2),
        "5fold_baseline_mae_std": round(float(np.std(base_maes)), 2),
        "5fold_graph_mae_mean": round(float(np.mean(graph_maes)), 2),
        "5fold_graph_mae_std": round(float(np.std(graph_maes)), 2),
        "paired_t_test": {
            "t_statistic": round(float(t_stat), 4),
            "p_value": float(t_pval),
            "is_statistically_significant": bool(t_pval < 0.001),
        },
        "wilcoxon_signed_rank_test": {
            "statistic": round(float(w_stat), 4),
            "p_value": float(w_pval),
            "is_statistically_significant": bool(w_pval < 0.05),
        },
        "ks_test_temporal_shift": {
            "ks_statistic": round(float(ks_stat), 4),
            "p_value": float(ks_pval),
            "distribution_shift_detected": bool(ks_pval < 0.01),
        },
    }

    out_path = Path("data/processed/statistical_tests.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Statistical tests saved → {out_path}")
    return results


if __name__ == "__main__":
    res = run_statistical_significance_tests()
    print("Statistical Tests Output:", json.dumps(res, indent=2))
