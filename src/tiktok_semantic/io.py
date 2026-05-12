from __future__ import annotations

import json
import math
import pickle
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


PKL_FILES = {
    "dtaPosts": "dtaPosts.pkl",
    "dtaAuthors": "dtaAuthors.pkl",
    "dtaAuthorPosts": "dtaAuthorPosts.pkl",
    "dtaAuthorFavorites": "dtaAuthorFavorites.pkl",
    "dtaAuthorFollowers": "dtaAuthorFollowers.pkl",
    "dtaAuthorFollowings": "dtaAuthorFollowings.pkl",
}


def read_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg


def extract_zip(zip_path: str | Path, dest_dir: str | Path, overwrite: bool = False) -> Path:
    zip_path = Path(zip_path)
    dest_dir = Path(dest_dir)
    marker = dest_dir / ".extracted"
    if marker.exists() and not overwrite:
        return dest_dir
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest_dir)
    marker.write_text(zip_path.name, encoding="utf-8")
    return dest_dir


def resolve_data_root(raw_dir: str | Path) -> Path:
    raw_dir = Path(raw_dir)
    if (raw_dir / "dtaPosts.pkl").exists():
        return raw_dir
    nested = raw_dir / "Data_Sample"
    if (nested / "dtaPosts.pkl").exists():
        return nested
    return raw_dir


def load_pickle(path: str | Path) -> Any:
    with Path(path).open("rb") as f:
        return pickle.load(f)


def load_raw_pickles(raw_dir: str | Path) -> dict[str, Any]:
    root = resolve_data_root(raw_dir)
    missing = [fname for fname in PKL_FILES.values() if not (root / fname).exists()]
    if missing:
        raise FileNotFoundError(f"Missing pickle files in {root}: {missing}")
    return {name: load_pickle(root / fname) for name, fname in PKL_FILES.items()}


def make_json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    return value


def prepare_for_parquet(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.select_dtypes(include=["object"]).columns:
        series = out[col]
        has_nested = series.apply(lambda x: isinstance(x, (dict, list, tuple, np.ndarray))).any()
        if has_nested:
            out[col] = series.apply(
                lambda v: json.dumps(make_json_safe(v), ensure_ascii=False)
                if isinstance(v, (dict, list, tuple, np.ndarray))
                else v
            )
        else:
            types = {type(v) for v in series.dropna()}
            if len(types) > 1:
                out[col] = series.apply(lambda v: None if pd.isna(v) else str(v))
    return out


def write_table(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".csv":
        df.to_csv(path, index=False)
    elif path.suffix == ".parquet":
        prepare_for_parquet(df).to_parquet(path, index=False)
    else:
        raise ValueError(f"Unsupported output format: {path.suffix}")
    return path


def write_manifest(rows: list[dict[str, Any]], path: str | Path) -> pd.DataFrame:
    manifest = pd.DataFrame(rows).sort_values("table").reset_index(drop=True)
    write_table(manifest, path)
    return manifest
