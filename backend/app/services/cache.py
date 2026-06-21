import json

import numpy as np

from backend.app.config import PROCESSED_DIR
from backend.app.services.data_loader import (
    build_analytics_summary,
    build_heatmap,
    build_hourly_pattern,
    build_top_junctions,
)
from backend.app.services.evaluation import get_model_metrics
from backend.app.services.pcis import get_feature_importance, load_scored_hotspots
from backend.app.services.recommender import build_enforcement_plan
from backend.app.services.storage import clear_frame_cache, load_frame


class DataCache:
    """In-memory cache for API responses — avoids repeated parquet reads."""

    _loaded = False

    @classmethod
    def warm(cls) -> None:
        clear_frame_cache()
        for name in ("violations", "cells", "cells_scored", "hotspots", "hotspots_scored"):
            load_frame(name)
        cls._loaded = True

    @classmethod
    def is_ready(cls) -> bool:
        return cls._loaded and not load_frame("violations").empty

    @classmethod
    def dashboard_payload(
        cls,
        *,
        heatmap_limit: int = 700,
        hotspot_limit: int = 20,
        officers: int = 6,
    ) -> dict:
        if not cls.is_ready():
            cls.warm()
        summary = build_analytics_summary()
        if not summary:
            return {}

        metrics = get_model_metrics()
        hotspots_df = (
            load_scored_hotspots()
            .sort_values("pcis", ascending=False)
            .head(hotspot_limit)
            .round({"pcis": 3, "latitude": 6, "longitude": 6, "estimated_delay_min_per_hour": 2})
        )
        hotspots = hotspots_df.to_dict("records")

        return {
            "summary": summary,
            "metrics": metrics,
            "heatmap": build_heatmap(limit=heatmap_limit),
            "hotspots": hotspots,
            "hourly": build_hourly_pattern(),
            "junctions": build_top_junctions(limit=8),
            "importance": get_feature_importance(),
            "plan": build_enforcement_plan(officers=officers),
        }


def load_metrics_json() -> dict:
    metrics = {}
    for name in ("pcis_metrics.json", "forecast_metrics.json", "pipeline_metrics.json"):
        path = PROCESSED_DIR / name
        if path.exists():
            metrics.update(json.loads(path.read_text()))
    return metrics
