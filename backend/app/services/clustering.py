import json

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from backend.app.config import MODELS_DIR
from backend.app.services.storage import load_frame, save_frame


def compute_clustering_silhouette(cells: pd.DataFrame) -> float:
    active = cells.loc[cells["violation_count"] >= 3].nlargest(2000, "violation_count")
    if len(active) < 40:
        return 0.0

    matrix = StandardScaler().fit_transform(
        active[
            ["latitude", "longitude", "violation_density", "severity_score", "junction_proximity"]
        ].to_numpy()
    )

    best_score = 0.0
    for eps in (0.32, 0.42, 0.52):
        labels = DBSCAN(eps=eps, min_samples=4, n_jobs=-1).fit_predict(matrix)
        valid = labels >= 0
        if valid.sum() < 50 or len(np.unique(labels[valid])) < 3:
            continue
        best_score = max(best_score, float(silhouette_score(matrix[valid], labels[valid])))

    for k in (10, 15, 20, 25):
        if k >= len(active):
            continue
        labels = KMeans(n_clusters=k, random_state=42, n_init=5).fit_predict(matrix)
        best_score = max(best_score, float(silhouette_score(matrix, labels)))

    joblib.dump({"silhouette": best_score}, MODELS_DIR / "dbscan.joblib")
    return best_score


def build_hotspots(cells: pd.DataFrame, violations: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    silhouette = compute_clustering_silhouette(cells)

    cell_index = cells.set_index("h3_index", drop=False)
    junction_h3 = violations.groupby("junction_label", observed=True)["h3_index"].agg(lambda s: s.unique().tolist())
    h3_location = _dominant_by_group(violations, "h3_index", "location")
    h3_primary = _dominant_by_group(violations, "h3_index", "primary_violation")
    h3_station = _dominant_by_group(violations, "h3_index", "police_station")

    junction_data = violations.loc[violations["is_junction"] == 1]
    top_junctions = (
        junction_data.groupby("junction_label", observed=True)
        .agg(
            latitude=("latitude", "mean"),
            longitude=("longitude", "mean"),
            violation_count=("id", "count"),
            dominant_violation=("primary_violation", lambda s: s.value_counts().index[0]),
            police_station=("police_station", lambda s: s.value_counts().index[0]),
        )
        .sort_values("violation_count", ascending=False)
        .head(40)
    )

    covered_h3: set[str] = set()
    rows: list[dict] = []

    for junction_name, row in top_junctions.iterrows():
        h3_indexes = junction_h3.get(junction_name, [])
        covered_h3.update(h3_indexes)
        rows.append(_hotspot_row(junction_name, row, h3_indexes, cell_index, str(junction_name)))

    corridor_cells = cells.loc[~cells["h3_index"].isin(covered_h3)].nlargest(20, "violation_count")
    for h3_id in corridor_cells["h3_index"]:
        cell = cell_index.loc[h3_id]
        rows.append(
            {
                "hotspot_id": "",
                "cluster_id": len(rows),
                "name": str(h3_location.get(h3_id, "Unknown"))[:80],
                "latitude": float(cell["latitude"]),
                "longitude": float(cell["longitude"]),
                "violation_count": int(cell["violation_count"]),
                "dominant_violation": str(h3_primary.get(h3_id, "UNKNOWN")),
                "police_station": str(h3_station.get(h3_id, "Unknown")),
                "junction_name": None,
                "h3_indexes": json.dumps([h3_id]),
                "violation_density": float(cell["violation_density"]),
                "severity_score": float(cell["severity_score"]),
                "junction_proximity": float(cell["junction_proximity"]),
                "temporal_congestion": float(cell["temporal_congestion"]),
                "spillover_risk": float(cell["spillover_risk"]),
            }
        )

    hotspots_df = pd.DataFrame(rows).sort_values("violation_count", ascending=False).reset_index(drop=True)
    hotspots_df["hotspot_id"] = [f"HS-{i:04d}" for i in range(len(hotspots_df))]
    save_frame(hotspots_df, "hotspots")
    return hotspots_df, silhouette


def _dominant_by_group(df: pd.DataFrame, group_col: str, value_col: str) -> pd.Series:
    counts = df.groupby([group_col, value_col], observed=True).size().reset_index(name="n")
    counts = counts.sort_values("n", ascending=False).drop_duplicates(group_col)
    return counts.set_index(group_col)[value_col]


def _cell_metrics(cell_index: pd.DataFrame, h3_indexes: list[str]) -> dict:
    subset = cell_index.loc[cell_index.index.intersection(h3_indexes)]
    if subset.empty:
        return {
            "violation_density": 0.0,
            "severity_score": 0.0,
            "junction_proximity": 0.0,
            "temporal_congestion": 0.0,
            "spillover_risk": 0.0,
        }
    return {
        "violation_density": float(subset["violation_density"].mean()),
        "severity_score": float(subset["severity_score"].mean()),
        "junction_proximity": float(subset["junction_proximity"].mean()),
        "temporal_congestion": float(subset["temporal_congestion"].mean()),
        "spillover_risk": float(subset["spillover_risk"].mean()),
    }


def _hotspot_row(name, row, h3_indexes: list[str], cell_index: pd.DataFrame, junction_name: str) -> dict:
    metrics = _cell_metrics(cell_index, h3_indexes)
    return {
        "hotspot_id": "",
        "cluster_id": -1,
        "name": str(name),
        "latitude": float(row["latitude"]),
        "longitude": float(row["longitude"]),
        "violation_count": int(row["violation_count"]),
        "dominant_violation": row["dominant_violation"],
        "police_station": row["police_station"],
        "junction_name": junction_name,
        "h3_indexes": json.dumps(h3_indexes),
        **metrics,
    }


def load_hotspots() -> pd.DataFrame:
    return load_frame("hotspots")
