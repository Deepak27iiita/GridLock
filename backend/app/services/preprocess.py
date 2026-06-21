import json
import re

import h3
import numpy as np
import pandas as pd

from backend.app.config import H3_RESOLUTION, VEHICLE_SIZE_FACTOR, VIOLATION_WEIGHTS
from backend.app.services.storage import save_frame

_PRIMARY_VIOLATION_RE = re.compile(r'\["([^"]+)"')
_VEHICLE_MAP = {k: v for k, v in VEHICLE_SIZE_FACTOR.items()}
_WEIGHT_MAP = {k: v for k, v in VIOLATION_WEIGHTS.items()}

_READ_DTYPES = {
    "id": "string",
    "vehicle_type": "string",
    "validation_status": "string",
    "police_station": "string",
    "junction_name": "string",
    "vehicle_number": "string",
    "location": "string",
    "violation_type": "string",
}


def load_raw_dataset(path) -> pd.DataFrame:
    usecols = list(_READ_DTYPES.keys()) + ["latitude", "longitude", "created_datetime"]
    chunks = []
    for chunk in pd.read_csv(path, usecols=usecols, dtype=_READ_DTYPES, engine="python", on_bad_lines="skip", chunksize=25000):
        # Filter out rejected rows early to save RAM
        chunk = chunk[chunk["validation_status"].fillna("") != "rejected"]
        chunks.append(chunk)
    return pd.concat(chunks, ignore_index=True)


def _assign_h3_unique(latitudes: np.ndarray, longitudes: np.ndarray) -> np.ndarray:
    res = H3_RESOLUTION
    return np.fromiter(
        (h3.latlng_to_cell(float(lat), float(lon), res) for lat, lon in zip(latitudes, longitudes)),
        dtype=object,
        count=len(latitudes),
    )


def _dominant_by_group(df: pd.DataFrame, group_col: str, value_col: str) -> pd.Series:
    counts = df.groupby([group_col, value_col], observed=True).size().reset_index(name="n")
    counts = counts.sort_values("n", ascending=False).drop_duplicates(group_col)
    return counts.set_index(group_col)[value_col]


def preprocess_violations(df: pd.DataFrame) -> pd.DataFrame:
    # Filter is already applied in chunks during load_raw_dataset; avoid .copy() to save memory

    df["primary_violation"] = (
        df["violation_type"].str.extract(_PRIMARY_VIOLATION_RE, expand=False).fillna("UNKNOWN")
    )
    df["severity_weight"] = df["primary_violation"].map(_WEIGHT_MAP).fillna(0.5).astype(np.float32)
    df["vehicle_factor"] = df["vehicle_type"].str.upper().map(_VEHICLE_MAP).fillna(1.0).astype(np.float32)
    df["composite_severity"] = df["severity_weight"] * df["vehicle_factor"]

    df["created_dt"] = pd.to_datetime(df["created_datetime"], format="mixed", utc=True)
    hour_float = df["created_dt"].dt.hour + df["created_dt"].dt.minute / 60.0 + 5.5
    df["hour_ist"] = (hour_float % 24).astype(np.int8)
    df["day_of_week"] = df["created_dt"].dt.dayofweek.astype(np.int8)
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(np.int8)
    df["month"] = df["created_dt"].dt.month.astype(np.int8)

    df["is_junction"] = (
        df["junction_name"].notna() & (df["junction_name"] != "No Junction")
    ).astype(np.int8)
    df["junction_label"] = df["junction_name"].where(df["is_junction"] == 1)

    coord_key = df["latitude"].round(6).astype(str) + "|" + df["longitude"].round(6).astype(str)
    unique_idx = coord_key.drop_duplicates().index
    h3_for_unique = _assign_h3_unique(
        df.loc[unique_idx, "latitude"].to_numpy(),
        df.loc[unique_idx, "longitude"].to_numpy(),
    )
    h3_lookup = dict(zip(coord_key.loc[unique_idx], h3_for_unique))
    df["h3_index"] = coord_key.map(h3_lookup)

    vehicle_key = df["vehicle_number"].astype(str).where(
        df["vehicle_number"].notna() & (df["vehicle_number"].astype(str).str.len() > 4),
        df["id"].astype(str),
    )
    df = df.sort_values("created_dt", kind="mergesort")
    df["dedupe_key"] = (
        vehicle_key
        + "_"
        + df["h3_index"].astype(str)
        + "_"
        + df["created_dt"].dt.floor("15min").astype(str)
    )
    df = df.drop_duplicates(subset=["dedupe_key"], keep="first").reset_index(drop=True)
    return df


def build_cell_features(violations: pd.DataFrame) -> pd.DataFrame:
    hour = violations["hour_ist"].to_numpy()
    violations = violations.copy()
    violations["_peak"] = (((hour >= 8) & (hour <= 11)) | ((hour >= 17) & (hour <= 20))).astype(np.int8)

    grouped = violations.groupby("h3_index", sort=False).agg(
        latitude=("latitude", "mean"),
        longitude=("longitude", "mean"),
        violation_count=("id", "count"),
        avg_severity=("composite_severity", "mean"),
        junction_ratio=("is_junction", "mean"),
        peak_hour_share=("_peak", "mean"),
    )

    grouped["dominant_violation"] = _dominant_by_group(violations, "h3_index", "primary_violation")
    grouped["top_station"] = _dominant_by_group(violations, "h3_index", "police_station")
    junction_only = violations.dropna(subset=["junction_label"])
    if not junction_only.empty:
        grouped["top_junction"] = _dominant_by_group(junction_only, "h3_index", "junction_label")
    else:
        grouped["top_junction"] = None

    grouped = grouped.reset_index()
    max_count = grouped["violation_count"].max() or 1
    grouped["violation_density"] = grouped["violation_count"] / max_count
    grouped["spillover_risk"] = np.clip(
        grouped["violation_density"] * (0.6 + grouped["avg_severity"] * 0.4),
        0,
        1,
    )
    grouped["temporal_congestion"] = np.clip(
        grouped["peak_hour_share"] * (0.5 + grouped["junction_ratio"] * 0.5),
        0,
        1,
    )
    grouped["junction_proximity"] = np.clip(
        grouped["junction_ratio"] * 0.7 + grouped["violation_density"] * 0.3,
        0,
        1,
    )
    grouped["severity_score"] = np.clip(grouped["avg_severity"], 0, 1.5) / 1.5
    return grouped


def save_processed(violations: pd.DataFrame, cells: pd.DataFrame) -> None:
    slim = violations[
        [
            "id",
            "latitude",
            "longitude",
            "h3_index",
            "primary_violation",
            "composite_severity",
            "hour_ist",
            "day_of_week",
            "month",
            "is_junction",
            "junction_label",
            "police_station",
            "validation_status",
            "vehicle_type",
            "location",
        ]
    ]
    save_frame(slim, "violations")
    save_frame(cells, "cells")
