"""
Shared utilities for logging, type detection, formatting, and ML helpers.

Pure-Python helpers live here so ``core`` modules and the Streamlit UI can
reuse the same logic without duplication.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RANDOM_STATE: int = 42
LOG_DIR: Path = Path("logs")
LOG_FILE: Path = LOG_DIR / "app.log"

# Target / encoding heuristics
CLASSIFICATION_UNIQUE_THRESHOLD: int = 20
ONE_HOT_MAX_CATEGORIES: int = 10
TARGET_ENCODING_MIN_CATEGORIES: int = 11
ID_COLUMN_UNIQUE_RATIO: float = 0.90

EncodingStrategy = Literal["label", "onehot", "target"]


class ProblemType(str, Enum):
    """Supported supervised learning problem types."""

    REGRESSION = "regression"
    CLASSIFICATION = "classification"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class ColumnGroups:
    """Grouped column names by semantic type."""

    numeric: tuple[str, ...]
    categorical: tuple[str, ...]
    datetime: tuple[str, ...]
    boolean: tuple[str, ...]
    other: tuple[str, ...]

    @property
    def all_feature_candidates(self) -> tuple[str, ...]:
        return self.numeric + self.categorical + self.boolean + self.datetime


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def setup_logging(
    name: str = "data_analytics_ml",
    level: int = logging.INFO,
    log_file: Path | None = LOG_FILE,
) -> logging.Logger:
    """
    Configure application logging with console and rotating file handlers.

    Args:
        name: Logger name.
        level: Logging level.
        log_file: Optional path for persistent logs.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=2_000_000,
            backupCount=3,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the application namespace."""
    setup_logging()
    return logging.getLogger(f"data_analytics_ml.{name}")


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------
def format_integer(value: int | float) -> str:
    """Format integers with thousands separators."""
    return f"{int(value):,}"


def format_memory(size_bytes: int | float) -> str:
    """Human-readable memory size."""
    size = float(size_bytes)
    if size >= 1024 ** 2:
        return f"{size / (1024 ** 2):.2f} MB"
    if size >= 1024:
        return f"{size / 1024:.1f} KB"
    return f"{size:.0f} B"


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Serialize a dataframe to UTF-8 CSV bytes for download widgets."""
    return df.to_csv(index=False).encode("utf-8")


def normalize_column_name(name: str) -> str:
    """
    Normalize a single column name: strip, lowercase, replace spaces/symbols.

    Examples:
        "Store ID" -> "store_id"
        "Holiday/Promotion" -> "holiday_promotion"
    """
    cleaned = str(name).strip().lower()
    cleaned = re.sub(r"[^\w]+", "_", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned)
    return cleaned.strip("_")


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with normalized column names."""
    renamed = df.copy()
    renamed.columns = [normalize_column_name(col) for col in renamed.columns]
    return renamed


# ---------------------------------------------------------------------------
# Column typing
# ---------------------------------------------------------------------------
def classify_columns(df: pd.DataFrame) -> ColumnGroups:
    """
    Classify dataframe columns into numeric, categorical, datetime, and boolean.

    Args:
        df: Input dataframe.

    Returns:
        ColumnGroups with column name tuples per type.
    """
    numeric = df.select_dtypes(include=np.number).columns.tolist()
    boolean = df.select_dtypes(include=["bool"]).columns.tolist()
    datetime_cols = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()
    categorical = df.select_dtypes(include=["object", "category", "string"]).columns.tolist()

    known = set(numeric) | set(boolean) | set(datetime_cols) | set(categorical)
    other = [col for col in df.columns if col not in known]

    return ColumnGroups(
        numeric=tuple(str(c) for c in numeric),
        categorical=tuple(str(c) for c in categorical),
        datetime=tuple(str(c) for c in datetime_cols),
        boolean=tuple(str(c) for c in boolean),
        other=tuple(str(c) for c in other),
    )


def is_likely_id_column(series: pd.Series) -> bool:
    """
    Heuristic: column is probably an identifier, not a useful ML feature.

    High cardinality relative to row count suggests an ID column.
    """
    non_null = series.dropna()
    if non_null.empty:
        return False
    unique_ratio = non_null.nunique() / len(non_null)
    return unique_ratio >= ID_COLUMN_UNIQUE_RATIO


def suggest_target_column(df: pd.DataFrame) -> str | None:
    """
    Suggest a likely target column using naming patterns and dtype heuristics.

    Prefers common target names, then numeric columns that are not IDs.

    Args:
        df: Input dataframe.

    Returns:
        Suggested column name, or ``None`` if no reasonable guess exists.
    """
    preferred_names = (
        "target",
        "label",
        "y",
        "units sold",
        "price",
        "sales",
        "revenue",
        "churn",
        "default",
        "class",
        "outcome",
    )
    lower_map = {str(col).lower(): str(col) for col in df.columns}

    for name in preferred_names:
        if name in lower_map:
            return lower_map[name]

    groups = classify_columns(df)
    candidates: list[str] = []

    for column in groups.numeric:
        series = df[column]
        if is_likely_id_column(series):
            continue
        candidates.append(column)

    for column in groups.categorical:
        if df[column].nunique(dropna=True) <= CLASSIFICATION_UNIQUE_THRESHOLD:
            candidates.append(column)

    return candidates[0] if candidates else None


def detect_problem_type(target: pd.Series) -> ProblemType:
    """
    Infer regression vs classification from target dtype and cardinality.

    Rules:
        - Boolean or object/category -> classification
        - Numeric with <= 20 unique values -> classification
        - Numeric with > 20 unique values -> regression

    Args:
        target: Target variable series.

    Raises:
        ValueError: If target has no non-null values.
    """
    clean = target.dropna()
    if clean.empty:
        raise ValueError("Target column has no non-null values.")

    if clean.dtype == bool or clean.dtype.name in {"object", "category", "string"}:
        unique_count = clean.nunique()
        if unique_count <= CLASSIFICATION_UNIQUE_THRESHOLD:
            return ProblemType.CLASSIFICATION
        # High-cardinality text targets are unusual; default to classification.
        return ProblemType.CLASSIFICATION

    if pd.api.types.is_numeric_dtype(clean):
        unique_count = clean.nunique()
        if unique_count <= CLASSIFICATION_UNIQUE_THRESHOLD:
            return ProblemType.CLASSIFICATION
        return ProblemType.REGRESSION

    return ProblemType.CLASSIFICATION


def choose_encoding_strategy(unique_count: int) -> EncodingStrategy:
    """
    Pick an encoding strategy based on categorical cardinality.

    - <= 10 unique values: one-hot encoding
    - 11–50 unique values: target encoding (optional in pipeline)
    - > 50 or default fallback: label encoding
    """
    if unique_count <= ONE_HOT_MAX_CATEGORIES:
        return "onehot"
    if unique_count <= 50:
        return "target"
    return "label"


def get_regression_metrics() -> tuple[str, ...]:
    """Metric names used for regression model comparison."""
    return ("MAE", "RMSE", "R2")


def get_classification_metrics() -> tuple[str, ...]:
    """Metric names used for classification model comparison."""
    return ("Accuracy", "Precision", "Recall", "F1", "ROC AUC")


def dataframe_memory_bytes(df: pd.DataFrame) -> int:
    """Return deep memory usage of a dataframe in bytes."""
    return int(df.memory_usage(deep=True).sum())


def reduce_dataframe_memory(df: pd.DataFrame, copy: bool = True) -> pd.DataFrame:
    """
    Downcast numeric columns to reduce memory footprint.

    Args:
        df: Input dataframe.
        copy: When True, operate on a copy to avoid mutating the original.

    Returns:
        Memory-optimized dataframe.
    """
    result = df.copy() if copy else df

    for column in result.select_dtypes(include=["int"]).columns:
        result[column] = pd.to_numeric(result[column], downcast="integer")

    for column in result.select_dtypes(include=["float"]).columns:
        result[column] = pd.to_numeric(result[column], downcast="float")

    for column in result.select_dtypes(include=["object"]).columns:
        num_unique = result[column].nunique(dropna=False)
        num_total = len(result[column])
        if num_unique / max(num_total, 1) < 0.5:
            result[column] = result[column].astype("category")

    return result
