"""Fast PCIS rescore — uses existing processed data, no full retrain."""

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.config import BASE_DELAY_PER_VIOLATION, PROCESSED_DIR  # noqa: E402
from backend.app.services.cache import DataCache  # noqa: E402
from backend.app.services.storage import clear_frame_cache, load_frame, save_frame  # noqa: E402
from backend.app.services.vectors import impact_blend, risk_labels, spread_scores  # noqa: E402


def main() -> None:
    hotspots = load_frame("hotspots")
    cells = load_frame("cells")
    if hotspots.empty:
        raise FileNotFoundError("Run scripts/train_pipeline.py first.")

    hotspots = hotspots.copy()
    hotspots["pcis"] = spread_scores(pd.Series(impact_blend(hotspots)))
    hotspots["estimated_delay_min_per_hour"] = (
        hotspots["pcis"] * (hotspots["violation_count"] ** 0.5) * BASE_DELAY_PER_VIOLATION * 1.15
    )
    hotspots["risk_level"] = risk_labels(hotspots["pcis"])
    save_frame(hotspots, "hotspots_scored")

    if not cells.empty:
        cells = cells.copy()
        blend = 0.55 * cells["violation_density"] + 0.45 * cells["violation_count"].rank(pct=True)
        cells["pcis"] = spread_scores(blend, power=0.7)
        save_frame(cells, "cells_scored")

    clear_frame_cache()
    DataCache.warm()

    summary = {
        "severe_hotspots": int((hotspots["pcis"] >= 0.85).sum()),
        "critical_hotspots": int((hotspots["pcis"] >= 0.72).sum()),
        "top_pcis": round(float(hotspots["pcis"].max()), 3),
        "avg_pcis": round(float(hotspots["pcis"].mean()), 3),
    }
    (PROCESSED_DIR / "rescore_summary.json").write_text(json.dumps(summary, indent=2))
    print("Rescored PCIS successfully:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
