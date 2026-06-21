import numpy as np
import pandas as pd

from backend.app.config import PCIS_WEIGHTS


def rule_pcis_frame(frame: pd.DataFrame) -> np.ndarray:
    return np.clip(
        PCIS_WEIGHTS["violation_density"] * frame["violation_density"].to_numpy()
        + PCIS_WEIGHTS["severity_score"] * frame["severity_score"].to_numpy()
        + PCIS_WEIGHTS["junction_proximity"] * frame["junction_proximity"].to_numpy()
        + PCIS_WEIGHTS["temporal_congestion"] * frame["temporal_congestion"].to_numpy()
        + PCIS_WEIGHTS["spillover_risk"] * frame["spillover_risk"].to_numpy(),
        0,
        1,
    )


def spread_scores(scores: pd.Series, power: float = 0.68) -> np.ndarray:
    ranked = scores.rank(pct=True, method="average")
    return np.clip(np.power(ranked.to_numpy(dtype=np.float64), power), 0, 1)


def risk_labels(pcis: pd.Series | np.ndarray) -> np.ndarray:
    values = pcis.to_numpy() if isinstance(pcis, pd.Series) else pcis
    return np.select(
        [values >= 0.85, values >= 0.72, values >= 0.55],
        ["Severe", "Critical", "High"],
        default="Moderate",
    )


def impact_blend(frame: pd.DataFrame, max_violations: float | None = None) -> np.ndarray:
    max_v = max_violations or float(frame["violation_count"].max() or 1)
    log_v = np.log1p(frame["violation_count"].to_numpy()) / np.log1p(max_v)
    return np.clip(
        frame["violation_density"].to_numpy() * 0.28
        + frame["severity_score"].to_numpy() * 0.20
        + frame["junction_proximity"].to_numpy() * 0.22
        + frame["temporal_congestion"].to_numpy() * 0.15
        + frame["spillover_risk"].to_numpy() * 0.10
        + log_v * 0.15,
        0,
        1,
    )
