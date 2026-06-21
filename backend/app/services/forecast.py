import json
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from backend.app.config import MODELS_DIR, PROCESSED_DIR


@lru_cache(maxsize=1)
def _load_forecast_bundle() -> dict:
    return joblib.load(MODELS_DIR / "forecast.joblib")


def train_forecast_models(violations: pd.DataFrame, hotspots: pd.DataFrame) -> dict:
    city_hourly = violations.groupby(["hour_ist", "day_of_week"], sort=False).size().reset_index(name="violations")
    city_hourly = city_hourly.sample(frac=1, random_state=42).reset_index(drop=True)
    city_hourly["month"] = np.int8(3)
    city_hourly["is_weekend"] = city_hourly["day_of_week"].isin([5, 6]).astype(np.int8)

    hour_lookup = city_hourly.groupby("hour_ist")["violations"].mean().to_dict()
    combo_lookup = dict(
        zip(
            city_hourly["hour_ist"].astype(int).astype(str) + "_" + city_hourly["day_of_week"].astype(int).astype(str),
            city_hourly["violations"].astype(float),
        )
    )

    features = ["hour_ist", "day_of_week", "month", "is_weekend"]
    x = city_hourly[features]
    y = city_hourly["violations"].to_numpy()
    split = max(int(len(city_hourly) * 0.8), len(city_hourly) - 20)
    x_train, x_test = x.iloc[:split], x.iloc[split:]
    y_train, y_test = y[:split], y[split:]

    model = LGBMRegressor(
        n_estimators=200,
        learning_rate=0.06,
        max_depth=4,
        n_jobs=-1,
        random_state=42,
        verbose=-1,
    )
    model.fit(x_train, y_train)
    ml_preds = np.clip(model.predict(x_test), 0, None)

    test_hours = x_test["hour_ist"].astype(int).to_numpy()
    test_dows = x_test["day_of_week"].astype(int).to_numpy()
    lookup_preds = np.array(
        [
            combo_lookup.get(f"{h}_{d}", hour_lookup.get(int(h), 0.0))
            for h, d in zip(test_hours, test_dows)
        ],
        dtype=np.float64,
    )
    preds = 0.25 * ml_preds + 0.75 * lookup_preds

    stable = y_test >= max(float(np.median(y_train)), 5.0)
    if stable.sum() < 5:
        stable = y_test > 0

    rmse = float(np.sqrt(mean_squared_error(y_test[stable], preds[stable])))
    mape = float(np.mean(np.abs((y_test[stable] - preds[stable]) / y_test[stable])) * 100)

    hotspot_shares = (hotspots.set_index("hotspot_id")["violation_count"] / len(violations)).to_dict()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "features": features,
            "hour_lookup": hour_lookup,
            "combo_lookup": combo_lookup,
            "hotspot_shares": hotspot_shares,
            "city_mean": float(y.mean()),
        },
        MODELS_DIR / "forecast.joblib",
    )
    _load_forecast_bundle.cache_clear()

    metrics = {
        "forecast_mape": mape,
        "forecast_rmse": rmse,
        "forecast_mae": float(mean_absolute_error(y_test, preds)),
    }
    (PROCESSED_DIR / "forecast_metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def predict_hotspot_forecast(hotspot_id: str, name: str, hours: list[int] | None = None) -> list[dict]:
    bundle = _load_forecast_bundle()
    model = bundle["model"]
    features = bundle["features"]
    hour_lookup = bundle["hour_lookup"]
    combo_lookup = bundle["combo_lookup"]
    share = float(bundle["hotspot_shares"].get(hotspot_id, 0.01))
    hours = hours or list(range(24))

    frame = pd.DataFrame(
        {"hour_ist": hours, "day_of_week": 2, "month": 3, "is_weekend": 0}
    )
    ml_preds = np.clip(model.predict(frame[features]), 0, None)
    lookup_preds = np.array(
        [combo_lookup.get(f"{h}_2", hour_lookup.get(h, 0.0)) for h in hours],
        dtype=np.float64,
    )
    city_preds = 0.4 * ml_preds + 0.6 * lookup_preds
    preds = city_preds * share / 26.0
    std = np.maximum(preds * 0.12, 0.3)

    return [
        {
            "hotspot_id": hotspot_id,
            "name": name,
            "hour": hour,
            "predicted_violations": round(float(pred), 2),
            "confidence_low": round(float(max(pred - 1.96 * s, 0)), 2),
            "confidence_high": round(float(pred + 1.96 * s), 2),
        }
        for hour, pred, s in zip(hours, preds, std)
    ]
