"""Model metrics visualisation utilities."""

import logging
from pathlib import Path
import pandas as pd

log = logging.getLogger(__name__)


def benchmark_comparison_chart(results_df: pd.DataFrame, save_path: str | Path | None = None):
    try:
        import plotly.graph_objects as go
    except ImportError:
        return None

    cat_colors = {
        "Baseline": "#5588ff", "ML+Graph": "#55ccff",
        "GNN": "#ff8844", "DL-Sequential": "#44ee88",
    }
    cat_col = "category" if "category" in results_df.columns else None

    fig = go.Figure()
    for metric, label, offset in [("MAE", "MAE (min)", -0.2), ("within_15pct", "Within 15%", 0.2)]:
        if metric not in results_df.columns:
            continue
        for cat, grp in (results_df.groupby(cat_col) if cat_col else [("All", results_df)]):
            fig.add_trace(go.Bar(
                name=f"{cat} — {label}",
                x=grp["model"],
                y=grp[metric],
                marker_color=cat_colors.get(cat, "#aaaaaa"),
                legendgroup=cat,
            ))

    fig.update_layout(
        barmode="group",
        title="Model Benchmark: MAE vs Within-15% Accuracy",
        template="plotly_dark",
        yaxis_title="Value",
        xaxis_tickangle=-20,
        height=450,
    )

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(save_path))
        log.info(f"Benchmark chart saved → {save_path}")

    return fig


def shap_bar_chart(shap_df: pd.DataFrame, top_n: int = 15):
    try:
        import plotly.express as px
    except ImportError:
        return None
    fig = px.bar(
        shap_df.head(top_n),
        x="mean_abs_shap",
        y="feature",
        orientation="h",
        color="mean_abs_shap",
        color_continuous_scale="Blues",
        title=f"Top-{top_n} Features by Mean |SHAP| Value",
    )
    fig.update_layout(template="plotly_dark", height=400, yaxis_autorange="reversed")
    return fig
