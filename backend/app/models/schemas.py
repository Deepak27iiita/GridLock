from pydantic import BaseModel, Field


class HotspotSummary(BaseModel):
    hotspot_id: str
    name: str
    latitude: float
    longitude: float
    pcis: float = Field(ge=0, le=1)
    violation_count: int
    dominant_violation: str
    police_station: str
    junction_name: str | None = None
    estimated_delay_min_per_hour: float
    risk_level: str


class HeatmapCell(BaseModel):
    h3_index: str
    latitude: float
    longitude: float
    pcis: float
    violation_count: int


class EnforcementRecommendation(BaseModel):
    rank: int
    hotspot_id: str
    name: str
    latitude: float
    longitude: float
    pcis: float
    recommended_window: str
    officers_needed: int
    tow_truck: bool
    estimated_delay_saved_min: float
    rationale: str


class ForecastPoint(BaseModel):
    hotspot_id: str
    name: str
    hour: int
    predicted_violations: float
    confidence_low: float
    confidence_high: float


class AnalyticsSummary(BaseModel):
    total_violations: int
    approved_violations: int
    unique_hotspots: int
    avg_pcis: float
    critical_hotspots: int
    severe_hotspots: int = 0
    top_police_station: str
    peak_hour_ist: int
    estimated_city_delay_min_per_day: float


class ModelMetrics(BaseModel):
    clustering_silhouette: float
    forecast_mape: float
    forecast_rmse: float
    pcis_r2: float
    pcis_mae: float
    validation_records: int
    model_grade: str | None = None
