from fastapi import APIRouter, HTTPException, Query

from backend.app.models.schemas import (
    AnalyticsSummary,
    EnforcementRecommendation,
    ForecastPoint,
    HeatmapCell,
    HotspotSummary,
    ModelMetrics,
)
from backend.app.services.cache import DataCache
from backend.app.services.data_loader import (
    build_analytics_summary,
    build_heatmap,
    build_hourly_pattern,
    build_top_junctions,
    load_violations,
)
from backend.app.services.evaluation import get_model_metrics
from backend.app.services.forecast import predict_hotspot_forecast
from backend.app.services.pcis import get_feature_importance, load_scored_hotspots
from backend.app.services.recommender import build_enforcement_plan, simulate_deployment

router = APIRouter(prefix="/api/v1", tags=["gridlock"])


def _hotspot_records(df):
    return [
        HotspotSummary(
            hotspot_id=str(r["hotspot_id"]),
            name=str(r["name"]),
            latitude=float(r["latitude"]),
            longitude=float(r["longitude"]),
            pcis=round(float(r["pcis"]), 3),
            violation_count=int(r["violation_count"]),
            dominant_violation=str(r["dominant_violation"]),
            police_station=str(r["police_station"]),
            junction_name=r.get("junction_name"),
            estimated_delay_min_per_hour=round(float(r["estimated_delay_min_per_hour"]), 2),
            risk_level=str(r["risk_level"]),
        )
        for r in df.to_dict("records")
    ]


@router.get("/dashboard/init")
def dashboard_init(
    heatmap_limit: int = Query(default=700, ge=50, le=2000),
    hotspot_limit: int = Query(default=20, ge=1, le=100),
    officers: int = Query(default=6, ge=1, le=20),
):
    payload = DataCache.dashboard_payload(
        heatmap_limit=heatmap_limit,
        hotspot_limit=hotspot_limit,
        officers=officers,
    )
    if not payload:
        raise HTTPException(status_code=503, detail="Run scripts/train_pipeline.py first.")
    return payload


@router.get("/health")
def health():
    violations = load_violations()
    return {
        "status": "ok",
        "service": "GridLock Parking Intelligence",
        "records_loaded": len(violations),
    }


@router.get("/analytics/summary", response_model=AnalyticsSummary)
def analytics_summary():
    summary = build_analytics_summary()
    if not summary:
        raise HTTPException(status_code=503, detail="Run scripts/train_pipeline.py first.")
    return summary


@router.get("/metrics", response_model=ModelMetrics)
def model_metrics():
    metrics = get_model_metrics()
    if not metrics:
        raise HTTPException(status_code=503, detail="Pipeline metrics unavailable.")
    return ModelMetrics(
        clustering_silhouette=metrics.get("clustering_silhouette", 0),
        forecast_mape=metrics.get("forecast_mape", 0),
        forecast_rmse=metrics.get("forecast_rmse", 0),
        pcis_r2=metrics.get("pcis_r2", 0),
        pcis_mae=metrics.get("pcis_mae", 0),
        validation_records=metrics.get("validation_records", 0),
        model_grade=metrics.get("model_grade"),
    )


@router.get("/heatmap", response_model=list[HeatmapCell])
def heatmap(limit: int = Query(default=600, ge=50, le=2000)):
    return build_heatmap(limit=limit)


@router.get("/hotspots", response_model=list[HotspotSummary])
def hotspots(min_pcis: float = Query(default=0.0, ge=0, le=1), limit: int = Query(default=25, ge=1, le=100)):
    df = load_scored_hotspots()
    if df.empty:
        raise HTTPException(status_code=503, detail="Run scripts/train_pipeline.py first.")
    df = df[df["pcis"] >= min_pcis].sort_values("pcis", ascending=False).head(limit)
    return _hotspot_records(df)


@router.get("/hotspots/{hotspot_id}/forecast", response_model=list[ForecastPoint])
def hotspot_forecast(hotspot_id: str):
    df = load_scored_hotspots()
    match = df[df["hotspot_id"] == hotspot_id]
    if match.empty:
        raise HTTPException(status_code=404, detail="Hotspot not found.")
    row = match.iloc[0]
    return predict_hotspot_forecast(hotspot_id, row["name"])


@router.get("/enforcement/plan", response_model=list[EnforcementRecommendation])
def enforcement_plan(officers: int = Query(default=6, ge=1, le=20)):
    return build_enforcement_plan(officers=officers)


@router.get("/simulate")
def simulate(extra_teams: int = Query(default=1, ge=1, le=10), duration_hours: int = Query(default=2, ge=1, le=8)):
    return simulate_deployment(extra_teams=extra_teams, duration_hours=duration_hours)


@router.get("/violations/top-junctions")
def top_junctions(limit: int = Query(default=10, ge=1, le=30)):
    return build_top_junctions(limit=limit)


@router.get("/analytics/hourly-pattern")
def hourly_pattern():
    data = build_hourly_pattern()
    if not data:
        raise HTTPException(status_code=503, detail="Processed data unavailable.")
    return data


@router.get("/analytics/feature-importance")
def feature_importance():
    data = get_feature_importance()
    if not data:
        raise HTTPException(status_code=503, detail="PCIS model unavailable.")
    return data
