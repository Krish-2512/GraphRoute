"""
Conformal Prediction & Uncertainty Quantification (UQ) for Delivery ETAs.

Provides distribution-free, finite-sample calibrated prediction intervals:
  P( y ∈ [ \hat{y} - q_{1-\alpha}, \hat{y} + q_{1-\alpha} ] ) ≥ 1 - α

Uses split conformal prediction on hold-out calibration residual distribution.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
import pickle

log = logging.getLogger(__name__)
MODEL_DIR = Path("data/processed/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)


class ConformalETAPredictor:
    """
    Split-Conformal Prediction wrapper over any trained regression model.
    Guarantees 1 - alpha statistical coverage on test distributions.
    """
    def __init__(self, base_model, alpha: float = 0.10):
        self.base_model = base_model
        self.alpha = alpha  # e.g., 0.10 for 90% confidence coverage
        self.q_hat = None

    def calibrate(self, X_cal: np.ndarray, y_cal: np.ndarray) -> float:
        """
        Computes non-conformity scores R_i = |y_i - \hat{y}_i| and the (1-alpha) empirical quantile.
        """
        y_pred = self.base_model.predict(X_cal)
        residuals = np.abs(y_cal - y_pred)
        n = len(residuals)
        
        # Conformal quantile adjustment: ceil((n + 1) * (1 - alpha)) / n
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        q_level = min(1.0, max(0.0, q_level))
        self.q_hat = float(np.quantile(residuals, q_level))
        log.info(f"Conformal Calibration completed on n={n} samples. Quantile q_hat={self.q_hat:.2f} min (Coverage: {(1-self.alpha)*100:.0f}%)")
        return self.q_hat

    def predict_interval(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns:
          y_pred: Point estimate (minutes)
          y_lower: Lower conformal bound (clipped at 0)
          y_upper: Upper conformal bound
        """
        if self.q_hat is None:
            raise ValueError("Predictor is not calibrated yet. Call .calibrate(X_cal, y_cal) first.")
        y_pred = self.base_model.predict(X)
        y_lower = np.maximum(0.0, y_pred - self.q_hat)
        y_upper = y_pred + self.q_hat
        return y_pred, y_lower, y_upper

    def evaluate_coverage(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Validates empirical coverage percentage on unseen test data."""
        y_pred, y_lower, y_upper = self.predict_interval(X_test)
        in_bounds = (y_test >= y_lower) & (y_test <= y_upper)
        empirical_coverage = float(np.mean(in_bounds) * 100.0)
        avg_interval_width = float(np.mean(y_upper - y_lower))
        
        metrics = {
            "target_coverage_pct": (1.0 - self.alpha) * 100.0,
            "empirical_coverage_pct": round(empirical_coverage, 2),
            "avg_interval_width_min": round(avg_interval_width, 2),
            "q_hat_radius_min": round(self.q_hat, 2),
        }
        log.info(f"Conformal Evaluation: Target={metrics['target_coverage_pct']}% | Empirical={metrics['empirical_coverage_pct']}% | Width={metrics['avg_interval_width_min']} min")
        return metrics


def run_conformal_pipeline():
    import sys
    root_dir = Path(__file__).resolve().parent.parent.parent
    if str(root_dir) not in sys.path:
        sys.path.insert(0, str(root_dir))
    from src.models.baseline import add_graph_features, BASE_FEATURES, GRAPH_FEATURES, TARGET
    from sklearn.model_selection import train_test_split

    df = pd.read_parquet("data/processed/features.parquet")
    df = add_graph_features(df)

    feature_cols = BASE_FEATURES + GRAPH_FEATURES
    available = [c for c in feature_cols if c in df.columns]
    X = df[available].fillna(0).values
    y = df[TARGET].values

    # 3-way Split: Train (60%), Calibration (20%), Test (20%)
    X_tr, X_temp, y_tr, y_temp = train_test_split(X, y, test_size=0.40, random_state=42)
    X_cal, X_test, y_cal, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)

    # Load trained LightGBM+Graph model
    with open("data/processed/models/LightGBM+Graph.pkl", "rb") as f:
        model = pickle.load(f)

    predictor = ConformalETAPredictor(model, alpha=0.10)
    predictor.calibrate(X_cal, y_cal)
    results = predictor.evaluate_coverage(X_test, y_test)

    # Save calibrated conformal artifact
    with open(MODEL_DIR / "conformal_predictor.pkl", "wb") as f:
        pickle.dump(predictor, f)
    
    with open("data/processed/conformal_metrics.json", "w") as f:
        import json
        json.dump(results, f, indent=2)
    return results


if __name__ == "__main__":
    res = run_conformal_pipeline()
    print("Conformal Prediction Results (90% Confidence Interval):", res)
