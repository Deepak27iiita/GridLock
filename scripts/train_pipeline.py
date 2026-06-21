import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.config import DATASET_PATH, PROCESSED_DIR  # noqa: E402
from backend.app.services.cache import DataCache  # noqa: E402
from backend.app.services.clustering import build_hotspots  # noqa: E402
from backend.app.services.evaluation import compile_evaluation_report  # noqa: E402
from backend.app.services.forecast import train_forecast_models  # noqa: E402
from backend.app.services.pcis import train_pcis_model  # noqa: E402
from backend.app.services.preprocess import (  # noqa: E402
    build_cell_features,
    load_raw_dataset,
    preprocess_violations,
    save_processed,
)
from backend.app.services.recommender import build_enforcement_plan  # noqa: E402
from backend.app.services.storage import clear_frame_cache  # noqa: E402


def _step(label: str, fn):
    t0 = time.perf_counter()
    result = fn()
    print(f"      done in {time.perf_counter() - t0:.1f}s")
    return result


def main() -> None:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    started = time.perf_counter()
    print("GridLock Pipeline — Optimized ML Training")
    print("=" * 50)

    print("[1/6] Loading raw dataset...")
    raw = _step("", lambda: load_raw_dataset(DATASET_PATH))
    print(f"      Loaded {len(raw):,} records")

    print("[2/6] Preprocessing & feature engineering...")
    violations = preprocess_violations(raw)
    del raw
    cells = build_cell_features(violations)
    save_processed(violations, cells)
    print(f"      Clean records: {len(violations):,} | H3 cells: {len(cells):,}")

    print("[3/6] Hotspot detection...")
    hotspots, silhouette = _step("", lambda: build_hotspots(cells, violations))
    print(f"      Hotspots: {len(hotspots)} | Silhouette: {silhouette:.4f}")

    print("[4/6] PCIS model...")
    pcis_metrics = _step("", lambda: train_pcis_model(cells, hotspots))
    print(f"      R²: {pcis_metrics['pcis_r2']:.4f} | MAE: {pcis_metrics['pcis_mae']:.4f}")

    print("[5/6] Forecast model...")
    forecast_metrics = _step("", lambda: train_forecast_models(violations, hotspots))
    print(
        f"      MAPE: {forecast_metrics.get('forecast_mape', 0):.2f}% | "
        f"RMSE: {forecast_metrics.get('forecast_rmse', 0):.3f}"
    )

    print("[6/6] Enforcement plan & evaluation...")
    plan = build_enforcement_plan(officers=8, tow_trucks=3)
    report = compile_evaluation_report(silhouette, pcis_metrics, forecast_metrics, violations)
    print(f"      Model grade: {report['model_grade']} | Targets: {len(plan)}")

    clear_frame_cache()

    elapsed = time.perf_counter() - started
    summary = {"elapsed_seconds": round(elapsed, 1), "violations": len(violations), "hotspots": len(hotspots), **report}
    (PROCESSED_DIR / "pipeline_summary.json").write_text(json.dumps(summary, indent=2))
    print("=" * 50)
    print(f"Pipeline complete in {elapsed:.1f}s")
    print(f"Outputs: {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
