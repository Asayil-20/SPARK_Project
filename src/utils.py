"""Reusable utility helpers for the SPARK backend."""

from __future__ import annotations

import json
import logging
import math
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from config import CSV_ENCODING, DEFAULT_LOG_LEVEL


def setup_logging(level: str | int = DEFAULT_LOG_LEVEL) -> logging.Logger:
    numeric_level = getattr(logging, str(level).upper(), logging.INFO)
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(
            level=numeric_level,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            handlers=[logging.StreamHandler(sys.stdout)],
        )
    else:
        root.setLevel(numeric_level)
    return logging.getLogger("spark")


def ensure_dir(path: Path | str) -> Path:
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def resolve_input_csv(data_dir: Path | str, expected_name: str, required: bool = False) -> Path | None:
    data_dir = Path(data_dir)
    exact_path = data_dir / expected_name
    if exact_path.exists():
        return exact_path

    expected_stem = Path(expected_name).stem.lower()
    candidates: list[Path] = []
    for candidate in data_dir.glob("*.csv"):
        stem = candidate.stem.lower().strip()
        normalized_stem = re.sub(r"\s*\(\d+\)$", "", stem).strip()
        normalized_stem = normalized_stem.replace(" ", "")
        target = expected_stem.replace(" ", "")
        if normalized_stem == target or stem.startswith(expected_stem):
            candidates.append(candidate)

    if candidates:
        return sorted(candidates)[0]

    if required:
        raise FileNotFoundError(
            f"Required file not found: expected {expected_name} in {data_dir}. "
            f"Available CSVs: {[p.name for p in sorted(data_dir.glob('*.csv'))]}"
        )
    return None


def safe_read_csv(path: Path | str | None, required: bool = False) -> pd.DataFrame:
    if path is None:
        if required:
            raise FileNotFoundError("Required CSV path is missing.")
        return pd.DataFrame()

    path_obj = Path(path)
    if not path_obj.exists():
        if required:
            raise FileNotFoundError(f"Required file not found: {path_obj}")
        logging.getLogger("spark").warning("Optional file not found: %s", path_obj)
        return pd.DataFrame()

    try:
        return pd.read_csv(path_obj)
    except pd.errors.EmptyDataError:
        logging.getLogger("spark").warning("CSV is empty: %s", path_obj)
        return pd.DataFrame()
    except Exception as exc:
        if required:
            raise RuntimeError(f"Failed to read required CSV {path_obj}: {exc}") from exc
        logging.getLogger("spark").warning("Failed to read optional CSV %s: %s", path_obj, exc)
        return pd.DataFrame()


def safe_write_csv(df: pd.DataFrame, path: Path | str) -> Path:
    path_obj = Path(path)
    ensure_dir(path_obj.parent)
    df.to_csv(path_obj, index=False, encoding=CSV_ENCODING)
    return path_obj


def save_json(obj: dict[str, Any] | list[Any], path: Path | str) -> Path:
    path_obj = Path(path)
    ensure_dir(path_obj.parent)
    with path_obj.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, ensure_ascii=False, indent=2, default=str)
    return path_obj


def load_json(path: Path | str) -> dict[str, Any]:
    path_obj = Path(path)
    if not path_obj.exists():
        return {}
    with path_obj.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    renamed = {}
    for col in df.columns:
        col_norm = str(col).strip().lower()
        col_norm = col_norm.replace("/", "_").replace("-", "_")
        col_norm = re.sub(r"[^\w\s]", "_", col_norm)
        col_norm = re.sub(r"\s+", "_", col_norm)
        col_norm = re.sub(r"_+", "_", col_norm).strip("_")
        renamed[col] = col_norm
    return df.rename(columns=renamed).copy()


def find_column(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    df_cols = set(df.columns)
    for candidate in candidates:
        if candidate in df_cols:
            return candidate
    return None


def add_missing_columns(df: pd.DataFrame, defaults: dict[str, Any]) -> pd.DataFrame:
    out = df.copy()
    for col, default in defaults.items():
        if col not in out.columns:
            out[col] = default
    return out


_ARABIC_DIACRITICS_RE = re.compile(
    "["
    "\u0610-\u061A"
    "\u064B-\u065F"
    "\u0670"
    "\u06D6-\u06ED"
    "\u0640"
    "]"
)


def normalize_arabic_text(text: Any) -> str:
    if pd.isna(text):
        return ""
    value = str(text).strip()
    value = unicodedata.normalize("NFKC", value)
    value = _ARABIC_DIACRITICS_RE.sub("", value)
    replacements = {
        "أ": "ا",
        "إ": "ا",
        "آ": "ا",
        "ى": "ي",
        "ة": "ه",
        "ؤ": "و",
        "ئ": "ي",
    }
    for src, dst in replacements.items():
        value = value.replace(src, dst)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def safe_divide(numerator: Any, denominator: Any) -> float:
    try:
        if denominator in [0, None] or (isinstance(denominator, float) and math.isnan(denominator)):
            return float("nan")
        return float(numerator) / float(denominator)
    except Exception:
        return float("nan")


def parse_datetime_safe(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=False)


def clean_numeric(series: pd.Series) -> pd.Series:
    if series.dtype.kind in {"i", "u", "f"}:
        return pd.to_numeric(series, errors="coerce")
    cleaned = (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace(r"[^\d\.\-]", "", regex=True)
        .replace({"": np.nan, "nan": np.nan, "None": np.nan})
    )
    return pd.to_numeric(cleaned, errors="coerce")


def dataframe_summary(df: pd.DataFrame) -> dict[str, Any]:
    return {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "null_counts": {k: int(v) for k, v in df.isna().sum().to_dict().items()},
        "duplicate_rows": int(df.duplicated().sum()),
    }


def validate_required_columns(df: pd.DataFrame, required_columns: Iterable[str], dataset_name: str) -> None:
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(
            f"{dataset_name} is missing required columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )


def normalize_text_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = out[col].map(normalize_arabic_text)
    return out


def ensure_str_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = out[col].astype(str).str.strip()
    return out


def parse_date_arg(value: str | None) -> pd.Timestamp | None:
    if not value:
        return None
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        raise ValueError(f"Invalid date value: {value}")
    return pd.Timestamp(ts).normalize()


def filter_df_by_date_range(df: pd.DataFrame, dt_col: str, start_date: pd.Timestamp | None, end_date: pd.Timestamp | None) -> pd.DataFrame:
    if df.empty or dt_col not in df.columns or (start_date is None and end_date is None):
        return df.copy()
    temp = df.copy()
    temp[dt_col] = pd.to_datetime(temp[dt_col], errors="coerce")
    mask = temp[dt_col].notna()
    if start_date is not None:
        mask &= temp[dt_col] >= pd.Timestamp(start_date)
    if end_date is not None:
        mask &= temp[dt_col] < (pd.Timestamp(end_date) + pd.Timedelta(days=1))
    return temp[mask].copy()


def coalesce_columns(df: pd.DataFrame, target: str, candidates: Iterable[str], default: Any = None) -> pd.DataFrame:
    out = df.copy()
    if target not in out.columns:
        out[target] = default
    for candidate in candidates:
        if candidate in out.columns:
            out[target] = out[target].where(out[target].notna() & (out[target] != ""), out[candidate])
    if default is not None:
        out[target] = out[target].fillna(default)
    return out


def infer_max_timestamp(*dfs_and_cols: tuple[pd.DataFrame, str]) -> pd.Timestamp | None:
    values: list[pd.Timestamp] = []
    for df, col in dfs_and_cols:
        if df.empty or col not in df.columns:
            continue
        temp = pd.to_datetime(df[col], errors="coerce")
        if temp.notna().any():
            values.append(pd.Timestamp(temp.max()).normalize())
    return max(values) if values else None
