from pathlib import Path

import pandas as pd

from backend.app.config import PROCESSED_DIR

_parquet_cache: dict[str, tuple[float, pd.DataFrame]] = {}


def _frame_path(name: str) -> Path:
    return PROCESSED_DIR / f"{name}.parquet"


def _mtime(path: Path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


def save_frame(df: pd.DataFrame, name: str) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    path = _frame_path(name)
    df.to_parquet(path, index=False, compression="snappy")
    _parquet_cache.pop(name, None)


def load_frame(name: str, *, use_cache: bool = True) -> pd.DataFrame:
    path = _frame_path(name)
    if not path.exists():
        return pd.DataFrame()

    mtime = _mtime(path)
    if use_cache:
        cached = _parquet_cache.get(name)
        if cached and cached[0] == mtime:
            return cached[1]

    df = pd.read_parquet(path)
    if use_cache:
        _parquet_cache[name] = (mtime, df)
    return df


def clear_frame_cache() -> None:
    _parquet_cache.clear()
