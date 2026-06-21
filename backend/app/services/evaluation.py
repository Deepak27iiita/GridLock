import json
from pathlib import Path

import pandas as pd

from backend.app.config import PROCESSED_DIR
from backend.app.services.data_loader import load_metrics


def compile_evaluation_report(
    silhouette: float,
    pcis_metrics: dict,
    forecast_metrics: dict,
    violations: pd.DataFrame,
) -> dict:
    report = {
        "clustering_silhouette": round(float(silhouette), 4),
        "forecast_mape": round(float(forecast_metrics.get("forecast_mape", 0)), 2),
        "forecast_rmse": round(float(forecast_metrics.get("forecast_rmse", 0)), 3),
        "pcis_r2": round(float(pcis_metrics.get("pcis_r2", 0)), 4),
        "pcis_mae": round(float(pcis_metrics.get("pcis_mae", 0)), 4),
        "validation_records": int(len(violations)),
        "model_grade": _grade(silhouette, pcis_metrics, forecast_metrics),
    }
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    (PROCESSED_DIR / "pipeline_metrics.json").write_text(json.dumps(report, indent=2))
    return report


def _grade(silhouette: float, pcis_metrics: dict, forecast_metrics: dict) -> str:
    score = 0
    if silhouette >= 0.50:
        score += 2
    elif silhouette >= 0.35:
        score += 1
    if pcis_metrics.get("pcis_r2", 0) >= 0.75:
        score += 2
    elif pcis_metrics.get("pcis_r2", 0) >= 0.55:
        score += 1
    if forecast_metrics.get("forecast_mape", 100) <= 18:
        score += 2
    elif forecast_metrics.get("forecast_mape", 100) <= 28:
        score += 1
    if score >= 5:
        return "A+ Production Ready"
    if score >= 3:
        return "A Pilot Ready"
    return "B Prototype"


def get_model_metrics() -> dict:
    return load_metrics()
