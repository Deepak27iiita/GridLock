import json

import numpy as np
import pandas as pd

from backend.app.config import PROCESSED_DIR
from backend.app.services.pcis import load_scored_cells, load_scored_hotspots
from backend.app.services.storage import load_frame


def load_violations() -> pd.DataFrame:
    return load_frame("violations")


def load_metrics() -> dict:
    metrics = {}
    for name in ("pcis_metrics.json", "forecast_metrics.json", "pipeline_metrics.json"):
        path = PROCESSED_DIR / name
        if path.exists():
            metrics.update(json.loads(path.read_text()))
    return metrics


def build_analytics_summary() -> dict:
    violations = load_violations()
    hotspots = load_scored_hotspots()
    if violations.empty:
        return {}

    hour_counts = violations["hour_ist"].value_counts()
    station_counts = violations["police_station"].value_counts()
    approved = int((violations["validation_status"] == "approved").sum())

    pcis = hotspots["pcis"].to_numpy() if not hotspots.empty else np.array([])
    delay = float(hotspots["estimated_delay_min_per_hour"].sum() * 16) if len(pcis) else 0.0

    return {
        "total_violations": int(len(violations)),
        "approved_violations": approved,
        "unique_hotspots": int(len(hotspots)),
        "avg_pcis": round(float(pcis.mean()), 3) if len(pcis) else 0.0,
        "critical_hotspots": int((pcis >= 0.72).sum()) if len(pcis) else 0,
        "severe_hotspots": int((pcis >= 0.85).sum()) if len(pcis) else 0,
        "top_police_station": str(station_counts.index[0]),
        "peak_hour_ist": int(hour_counts.idxmax()),
        "estimated_city_delay_min_per_day": round(delay, 1),
    }


def build_hourly_pattern() -> list[dict]:
    violations = load_violations()
    if violations.empty:
        return []
    hourly = violations["hour_ist"].value_counts().sort_index()
    peak = float(hourly.max() or 1)
    return [
        {"hour": int(h), "violations": int(c), "intensity": round(float(c / peak), 3)}
        for h, c in hourly.items()
    ]


def build_top_junctions(limit: int = 12) -> list[dict]:
    violations = load_violations()
    if violations.empty:
        return []
    junctions = violations.loc[violations["is_junction"] == 1, "junction_label"]
    counts = junctions.value_counts().head(limit)
    return [{"junction": str(name), "violations": int(count)} for name, count in counts.items()]


def build_heatmap(limit: int = 800) -> list[dict]:
    cells = load_scored_cells()
    if cells.empty:
        return []
    top = cells.nlargest(limit, "pcis")
    return [
        {
            "h3_index": h3,
            "latitude": round(float(lat), 6),
            "longitude": round(float(lon), 6),
            "pcis": round(float(pcis), 3),
            "violation_count": int(count),
        }
        for h3, lat, lon, pcis, count in zip(
            top["h3_index"].to_numpy(),
            top["latitude"].to_numpy(),
            top["longitude"].to_numpy(),
            top["pcis"].to_numpy(),
            top["violation_count"].to_numpy(),
        )
    ]
