import json
from pathlib import Path

import pandas as pd

from backend.app.config import PROCESSED_DIR
from backend.app.services.pcis import load_scored_hotspots


def _window_for_hotspot(row: pd.Series) -> str:
    if row["temporal_congestion"] >= 0.65:
        return "08:30-11:00 IST"
    if row["temporal_congestion"] >= 0.45:
        return "17:00-20:00 IST"
    return "10:00-13:00 IST"


def _officers_needed(pcis: float, violation_count: int) -> int:
    if pcis >= 0.8 or violation_count >= 500:
        return 3
    if pcis >= 0.65 or violation_count >= 250:
        return 2
    return 1


def build_enforcement_plan(officers: int = 6, tow_trucks: int = 2) -> list[dict]:
    hotspots = load_scored_hotspots()
    if hotspots.empty:
        return []

    ranked = hotspots.sort_values("pcis", ascending=False).head(min(officers, len(hotspots)))
    plan = []
    tow_left = tow_trucks

    for rank, (_, row) in enumerate(ranked.iterrows(), start=1):
        need_tow = bool(row["spillover_risk"] >= 0.55 and tow_left > 0)
        if need_tow:
            tow_left -= 1
        officers_needed = _officers_needed(row["pcis"], row["violation_count"])
        delay_saved = float(row["estimated_delay_min_per_hour"] * (0.55 + row["pcis"] * 0.35))

        plan.append(
            {
                "rank": rank,
                "hotspot_id": row["hotspot_id"],
                "name": row["name"],
                "latitude": float(row["latitude"]),
                "longitude": float(row["longitude"]),
                "pcis": round(float(row["pcis"]), 3),
                "recommended_window": _window_for_hotspot(row),
                "officers_needed": officers_needed,
                "tow_truck": need_tow,
                "estimated_delay_saved_min": round(delay_saved, 1),
                "rationale": (
                    f"{row['dominant_violation']} cluster with junction impact "
                    f"{row['junction_proximity']:.0%} and spillover risk {row['spillover_risk']:.0%}."
                ),
            }
        )

    (PROCESSED_DIR / "enforcement_plan.json").write_text(json.dumps(plan, indent=2))
    return plan


def simulate_deployment(extra_teams: int = 1, duration_hours: int = 2) -> dict:
    hotspots = load_scored_hotspots().sort_values("pcis", ascending=False).head(10)
    if hotspots.empty:
        return {}

    base_delay = float(hotspots["estimated_delay_min_per_hour"].sum())
    efficiency = min(0.25 + extra_teams * 0.12, 0.72)
    reduced = base_delay * efficiency * duration_hours
    new_avg_pcis = float(hotspots["pcis"].mean() * (1 - efficiency * 0.45))

    return {
        "extra_teams": extra_teams,
        "duration_hours": duration_hours,
        "baseline_delay_min_per_hour": round(base_delay, 1),
        "projected_delay_reduction_min": round(reduced, 1),
        "projected_avg_pcis_after": round(new_avg_pcis, 3),
        "top_zones_covered": hotspots["name"].head(5).tolist(),
    }
