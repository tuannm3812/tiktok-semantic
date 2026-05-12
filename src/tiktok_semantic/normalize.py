from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd


def dict_to_frame(
    data: Any,
    key_name: str | None = None,
    record_name: str = "record",
    unify_video_id: bool = True,
) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        df = data.copy()
    elif isinstance(data, list):
        df = pd.DataFrame(data)
    elif isinstance(data, dict):
        rows = []
        for key, value in data.items():
            if isinstance(value, dict):
                row = value.copy()
                row[key_name or "key"] = key
            else:
                row = {key_name or "key": key, record_name: value}
            rows.append(row)
        df = pd.DataFrame(rows)
    else:
        df = pd.DataFrame([{record_name: data}])

    if unify_video_id and "video_id" not in df.columns:
        for candidate in ("video_id", "post_id", "aweme_id", "id"):
            if candidate in df.columns:
                df["video_id"] = df[candidate]
                break
    for col in ("video_id", "author_id", "id"):
        if col in df.columns:
            df[col] = df[col].astype("string")
    return df


def safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def ensure_video_id(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "video_id" not in out.columns:
        for candidate in ("post_id", "aweme_id", "id"):
            if candidate in out.columns:
                out["video_id"] = out[candidate]
                break
    if "video_id" in out.columns:
        out["video_id"] = out["video_id"].astype("string")
    return out


def dedupe_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if (out.columns == "video_id").sum() > 1:
        out["video_id"] = out.loc[:, out.columns == "video_id"].bfill(axis=1).iloc[:, 0]
    return out.loc[:, ~out.columns.duplicated()]


def scalar_columns(df: pd.DataFrame) -> list[str]:
    cols = []
    for col in df.columns:
        non_null = df[col].dropna()
        first_type = type(non_null.iloc[0]) if len(non_null) else None
        if first_type not in (dict, list, tuple, np.ndarray, pd.Series, pd.DataFrame):
            cols.append(col)
    return cols


def flatten_dict_column(
    df: pd.DataFrame, id_cols: list[str], column: str, prefix: str
) -> pd.DataFrame:
    base = df[id_cols].copy()
    if column not in df.columns:
        return base
    nested = pd.json_normalize([safe_dict(x) for x in df[column].tolist()])
    nested.index = df.index
    nested.columns = [f"{prefix}{c}" for c in nested.columns]
    return pd.concat([base.reset_index(drop=True), nested.reset_index(drop=True)], axis=1)


def explode_list_of_dicts(
    df: pd.DataFrame,
    list_col: str,
    parent_cols: list[str],
    prefix: str | None = None,
) -> pd.DataFrame:
    base = ensure_video_id(df)
    if list_col not in base.columns:
        return pd.DataFrame(columns=parent_cols)
    keep = [c for c in parent_cols if c in base.columns]
    tmp = base[keep + [list_col]].explode(list_col, ignore_index=True)
    tmp = tmp[tmp[list_col].apply(lambda x: isinstance(x, dict))].copy()
    if tmp.empty:
        return pd.DataFrame(columns=keep)
    nested = pd.json_normalize(tmp[list_col].tolist())
    if prefix:
        nested.columns = [f"{prefix}{c}" for c in nested.columns]
    return pd.concat([tmp[keep].reset_index(drop=True), nested.reset_index(drop=True)], axis=1)


def explode_record_dicts(df: pd.DataFrame, key_col: str, record_col: str = "record") -> pd.DataFrame:
    if record_col not in df.columns:
        return pd.DataFrame(columns=[key_col])
    exploded = df[[key_col, record_col]].explode(record_col, ignore_index=True)
    exploded = exploded[exploded[record_col].apply(lambda x: isinstance(x, dict))].copy()
    if exploded.empty:
        return pd.DataFrame(columns=[key_col])
    nested = pd.json_normalize(exploded[record_col].tolist())
    out = pd.concat([exploded[[key_col]].reset_index(drop=True), nested.reset_index(drop=True)], axis=1)
    return ensure_video_id(out)


def extract_hashtags(title: str | float | None) -> list[str]:
    if not isinstance(title, str):
        return []
    return [tag.lower() for tag in re.findall(r"#\w+", title)]


def add_time_features(df: pd.DataFrame, create_col: str = "create_time") -> pd.DataFrame:
    out = df.copy()
    if create_col in out.columns:
        out["ts"] = pd.to_datetime(out[create_col], unit="s", utc=True, errors="coerce")
        out["date"] = out["ts"].dt.date.astype("string")
        out["month"] = out["ts"].dt.strftime("%Y-%m")
        out["dow"] = out["ts"].dt.day_name()
        out["hour"] = out["ts"].dt.hour.astype("Int32")
    return out


def coerce_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0)
    return out
