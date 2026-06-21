import json
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from backend.app.config import BASE_DELAY_PER_VIOLATION, MODELS_DIR, PROCESSED_DIR
from backend.app.services.storage import load_frame, save_frame
from backend.app.services.vectors import impact_blend, risk_labels, rule_pcis_frame, spread_scores


def train_pcis_model(cells: pd.DataFrame, hotspots: pd.DataFrame) -> dict:
    frame = hotspots.copy()
    max_v = float(frame["violation_count"].max() or 1)
    frame["log_violations"] = np.log1p(frame["violation_count"]) / np.log1p(max_v)
    frame["rule_pcis"] = rule_pcis_frame(frame)
    frame["impact_target"] = impact_blend(frame, max_v)

    features = [
        "violation_density",
        "severity_score",
        "junction_proximity",
        "temporal_congestion",
        "spillover_risk",
        "log_violations",
        "rule_pcis",
    ]
    x = frame[features].to_numpy()
    y = frame["impact_target"].to_numpy()

    regressor = LGBMRegressor(
        n_estimators=250,
        learning_rate=0.05,
        max_depth=5,
        n_jobs=-1,
        random_state=42,
        verbose=-1,
    )

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    regressor.fit(x_train, y_train)
    raw_pred = regressor.predict(x_test)
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(raw_pred, y_test)
    calibrated = calibrator.predict(raw_pred)

    metrics = {
        "pcis_r2": float(r2_score(y_test, calibrated)),
        "pcis_mae": float(mean_absolute_error(y_test, calibrated)),
    }

    frame["pcis_raw"] = regressor.predict(frame[features].to_numpy())
    frame["pcis_ml"] = np.clip(calibrator.predict(frame["pcis_raw"]), 0, 1)
    blended = 0.45 * frame["pcis_ml"] + 0.55 * frame["impact_target"]
    frame["pcis"] = spread_scores(blended)
    frame["estimated_delay_min_per_hour"] = (
        frame["pcis"] * np.sqrt(frame["violation_count"]) * BASE_DELAY_PER_VIOLATION * 1.15
    )
    frame["risk_level"] = risk_labels(frame["pcis"])

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "regressor": regressor,
            "calibrator": calibrator,
            "features": features,
            "importance": dict(zip(features, regressor.feature_importances_.tolist())),
        },
        MODELS_DIR / "pcis.joblib",
    )
    save_frame(frame, "hotspots_scored")

    cells_scored = cells.copy()
    density_pct = cells_scored["violation_count"].rank(pct=True, method="average")
    cells_scored["pcis"] = spread_scores(
        pd.Series(0.55 * rule_pcis_frame(cells_scored) + 0.45 * density_pct.to_numpy()),
        power=0.7,
    )
    save_frame(cells_scored, "cells_scored")
    (PROCESSED_DIR / "pcis_metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def load_scored_hotspots() -> pd.DataFrame:
    return load_frame("hotspots_scored")


def load_scored_cells() -> pd.DataFrame:
    cells = load_frame("cells_scored")
    if cells.empty:
        cells = load_frame("cells")
    if not cells.empty and "pcis" not in cells.columns:
        cells = cells.copy()
        cells["pcis"] = rule_pcis_frame(cells)
    return cells


@lru_cache(maxsize=1)
def _load_pcis_bundle() -> dict:
    return joblib.load(MODELS_DIR / "pcis.joblib")


def get_feature_importance() -> list[dict]:
    if not (MODELS_DIR / "pcis.joblib").exists():
        return []
    importance = _load_pcis_bundle().get("importance", {})
    labels = {
        "violation_density": "Violation Density",
        "severity_score": "Violation Severity",
        "junction_proximity": "Junction Proximity",
        "temporal_congestion": "Peak Hour Congestion",
        "spillover_risk": "Lane Spillover Risk",
        "log_violations": "Violation Volume",
        "rule_pcis": "Rule-based PCIS",
    }
    rows = [{"feature": labels.get(k, k), "importance": round(float(v), 4)} for k, v in importance.items()]
    return sorted(rows, key=lambda item: item["importance"], reverse=True)
