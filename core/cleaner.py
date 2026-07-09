"""
Automatic data cleaning: missing values, duplicates, outliers, and dates.

Reuses logic from the original notebook pipeline, generalized for any CSV.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from core.loader import detect_date_columns
from core.utils import classify_columns, get_logger, normalize_column_name, normalize_column_names

logger = get_logger("cleaner")

OutlierStrategy = Literal["cap", "remove", "skip"]

DATETIME_FEATURES: tuple[str, ...] = (
    "year",
    "month",
    "quarter",
    "week",
    "day",
    "dayofweek",
    "weekend",
)


@dataclass
class OutlierSummary:
    """Outlier statistics for a single numeric column."""

    column: str
    outlier_count: int
    lower_bound: float
    upper_bound: float


@dataclass
class CleaningReport:
    """Audit trail produced by the cleaning pipeline."""

    rows_before: int
    rows_after: int
    columns_before: int
    columns_after: int
    duplicates_removed: int
    missing_values_before: int
    missing_values_after: int
    empty_columns_removed: list[str] = field(default_factory=list)
    date_columns_parsed: list[str] = field(default_factory=list)
    datetime_features_added: list[str] = field(default_factory=list)
    outlier_summaries: list[OutlierSummary] = field(default_factory=list)
    outlier_rows_removed: int = 0
    actions: list[str] = field(default_factory=list)

    @property
    def rows_removed(self) -> int:
        return self.rows_before - self.rows_after

    def as_metrics(self) -> dict[str, str]:
        """Format key results for Streamlit metrics."""
        return {
            "Rows Before": f"{self.rows_before:,}",
            "Rows After": f"{self.rows_after:,}",
            "Duplicates Removed": str(self.duplicates_removed),
            "Missing Fixed": str(self.missing_values_before - self.missing_values_after),
            "Outlier Columns": str(len(self.outlier_summaries)),
        }


def trim_whitespace(df: pd.DataFrame) -> pd.DataFrame:
    """Strip leading/trailing spaces from string columns."""
    result = df.copy()
    for column in result.select_dtypes(include=["object", "string"]).columns:
        result[column] = result[column].astype(str).str.strip()
        result[column] = result[column].replace({"nan": np.nan, "None": np.nan, "": np.nan})
    return result


def remove_empty_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Drop columns that are entirely null."""
    empty_cols = [str(col) for col in df.columns if df[col].isnull().all()]
    if empty_cols:
        df = df.drop(columns=empty_cols)
    return df, empty_cols


def remove_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Remove duplicate rows and return count removed."""
    before = len(df)
    cleaned = df.drop_duplicates().reset_index(drop=True)
    return cleaned, before - len(cleaned)


def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing values.

    - Numeric columns: median
    - Categorical / object columns: mode (fallback to ``Unknown``)
    - Datetime columns: forward-fill then backward-fill
    """
    result = df.copy()
    groups = classify_columns(result)

    for column in groups.numeric:
        if result[column].isnull().any():
            result[column] = result[column].fillna(result[column].median())

    for column in groups.categorical:
        if result[column].isnull().any():
            mode = result[column].mode(dropna=True)
            fill_value = mode.iloc[0] if not mode.empty else "Unknown"
            result[column] = result[column].fillna(fill_value)

    for column in groups.datetime:
        if result[column].isnull().any():
            result[column] = result[column].ffill().bfill()

    for column in groups.other:
        if result[column].isnull().any():
            result[column] = result[column].fillna("Unknown")

    return result


def parse_date_columns(df: pd.DataFrame, date_columns: list[str] | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Parse detected or provided date columns to datetime dtype."""
    result = df.copy()
    columns = date_columns if date_columns is not None else detect_date_columns(result)
    parsed: list[str] = []

    for column in columns:
        if column not in result.columns:
            continue
        converted = pd.to_datetime(result[column], errors="coerce")
        if converted.notna().sum() > 0:
            result[column] = converted
            parsed.append(column)

    return result, parsed


def extract_datetime_features(df: pd.DataFrame, date_columns: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """
    Create calendar features from datetime columns.

    Adds: year, month, quarter, week, day, dayofweek, weekend
    (prefixed by column name when multiple date columns exist).
    """
    result = df.copy()
    added: list[str] = []

    for column in date_columns:
        if column not in result.columns:
            continue
        if not pd.api.types.is_datetime64_any_dtype(result[column]):
            continue

        prefix = "" if len(date_columns) == 1 else f"{column}_"
        feature_map = {
            f"{prefix}year": result[column].dt.year,
            f"{prefix}month": result[column].dt.month,
            f"{prefix}quarter": result[column].dt.quarter,
            f"{prefix}week": result[column].dt.isocalendar().week.astype(int),
            f"{prefix}day": result[column].dt.day,
            f"{prefix}dayofweek": result[column].dt.dayofweek,
            f"{prefix}weekend": (result[column].dt.dayofweek >= 5).astype(int),
        }

        for feature_name, feature_values in feature_map.items():
            if feature_name not in result.columns:
                result[feature_name] = feature_values
                added.append(feature_name)

    return result, added


def compute_iqr_bounds(series: pd.Series) -> tuple[float, float]:
    """Return lower and upper IQR fences for a numeric series."""
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = float(q1 - 1.5 * iqr)
    upper = float(q3 + 1.5 * iqr)
    return lower, upper


def detect_outliers_iqr(df: pd.DataFrame, numeric_columns: list[str] | None = None) -> list[OutlierSummary]:
    """Detect outliers using the 1.5×IQR rule on numeric columns."""
    columns = numeric_columns or classify_columns(df).numeric
    summaries: list[OutlierSummary] = []

    for column in columns:
        if column not in df.columns:
            continue
        series = pd.to_numeric(df[column], errors="coerce").dropna()
        if series.empty:
            continue

        lower, upper = compute_iqr_bounds(series)
        mask = (df[column] < lower) | (df[column] > upper)
        count = int(mask.sum())
        if count > 0:
            summaries.append(
                OutlierSummary(
                    column=str(column),
                    outlier_count=count,
                    lower_bound=lower,
                    upper_bound=upper,
                )
            )

    return summaries


def cap_outliers_iqr(df: pd.DataFrame, numeric_columns: list[str] | None = None) -> tuple[pd.DataFrame, list[OutlierSummary]]:
    """Cap numeric outliers at IQR fence values."""
    result = df.copy()
    summaries = detect_outliers_iqr(result, numeric_columns)

    for item in summaries:
        result[item.column] = result[item.column].clip(item.lower_bound, item.upper_bound)

    return result, summaries


def remove_outlier_rows(df: pd.DataFrame, numeric_columns: list[str] | None = None) -> tuple[pd.DataFrame, list[OutlierSummary], int]:
    """Remove rows that contain IQR outliers in any numeric column."""
    summaries = detect_outliers_iqr(df, numeric_columns)
    if not summaries:
        return df.copy(), summaries, 0

    outlier_mask = pd.Series(False, index=df.index)
    for item in summaries:
        outlier_mask |= (df[item.column] < item.lower_bound) | (df[item.column] > item.upper_bound)

    removed = int(outlier_mask.sum())
    cleaned = df.loc[~outlier_mask].reset_index(drop=True)
    return cleaned, summaries, removed


def treat_outliers(
    df: pd.DataFrame,
    strategy: OutlierStrategy = "cap",
) -> tuple[pd.DataFrame, list[OutlierSummary], int]:
    """
    Apply outlier treatment strategy.

    Args:
        df: Input dataframe.
        strategy: ``cap``, ``remove``, or ``skip``.

    Returns:
        Tuple of (cleaned dataframe, outlier summaries, rows removed).
    """
    if strategy == "skip":
        return df.copy(), [], 0

    numeric_columns = list(classify_columns(df).numeric)
    if strategy == "cap":
        capped, summaries = cap_outliers_iqr(df, numeric_columns)
        return capped, summaries, 0

    return remove_outlier_rows(df, numeric_columns)


def clean_dataframe(
    df: pd.DataFrame,
    *,
    outlier_strategy: OutlierStrategy = "cap",
    date_columns: list[str] | None = None,
    normalize_names: bool = True,
    extract_dates: bool = True,
) -> tuple[pd.DataFrame, CleaningReport]:
    """
    Run the full automatic cleaning pipeline.

    Steps:
        1. Normalize column names
        2. Trim whitespace
        3. Remove empty columns
        4. Remove duplicates
        5. Parse date columns
        6. Extract datetime features
        7. Fill missing values
        8. Treat outliers (IQR cap / remove / skip)

    Args:
        df: Raw input dataframe.
        outlier_strategy: How to handle numeric outliers.
        date_columns: Optional explicit date column names.
        normalize_names: Whether to normalize column names.
        extract_dates: Whether to add calendar features from dates.

    Returns:
        Cleaned dataframe and a ``CleaningReport`` audit trail.
    """
    actions: list[str] = []
    rows_before = len(df)
    cols_before = df.shape[1]
    missing_before = int(df.isnull().sum().sum())

    result = df.copy()

    if normalize_names:
        result = normalize_column_names(result)
        actions.append("Normalized column names")

    result = trim_whitespace(result)
    actions.append("Trimmed whitespace in text columns")

    result, empty_cols = remove_empty_columns(result)
    if empty_cols:
        actions.append(f"Removed empty columns: {', '.join(empty_cols)}")

    result, dupes_removed = remove_duplicates(result)
    if dupes_removed:
        actions.append(f"Removed {dupes_removed:,} duplicate rows")

    # Re-detect dates after renaming
    explicit_dates = date_columns
    if explicit_dates:
        explicit_dates = [normalize_column_name(c) for c in explicit_dates if normalize_column_name(c) in result.columns]

    result, parsed_dates = parse_date_columns(result, explicit_dates)
    if parsed_dates:
        actions.append(f"Parsed date columns: {', '.join(parsed_dates)}")

    datetime_features: list[str] = []
    if extract_dates and parsed_dates:
        result, datetime_features = extract_datetime_features(result, parsed_dates)
        if datetime_features:
            actions.append(f"Added datetime features: {', '.join(datetime_features)}")

    result = fill_missing_values(result)
    missing_after_fill = int(result.isnull().sum().sum())
    if missing_before > missing_after_fill:
        actions.append(
            f"Filled {missing_before - missing_after_fill:,} missing values "
            "(median / mode / ffill)"
        )

    outlier_summaries: list[OutlierSummary] = []
    outlier_rows_removed = 0
    if outlier_strategy != "skip":
        result, outlier_summaries, outlier_rows_removed = treat_outliers(result, outlier_strategy)
        if outlier_summaries:
            actions.append(
                f"Applied outlier strategy '{outlier_strategy}' "
                f"on {len(outlier_summaries)} numeric column(s)"
            )
        if outlier_rows_removed:
            actions.append(f"Removed {outlier_rows_removed:,} outlier rows")

    report = CleaningReport(
        rows_before=rows_before,
        rows_after=len(result),
        columns_before=cols_before,
        columns_after=result.shape[1],
        duplicates_removed=dupes_removed,
        missing_values_before=missing_before,
        missing_values_after=int(result.isnull().sum().sum()),
        empty_columns_removed=empty_cols,
        date_columns_parsed=parsed_dates,
        datetime_features_added=datetime_features,
        outlier_summaries=outlier_summaries,
        outlier_rows_removed=outlier_rows_removed,
        actions=actions,
    )

    logger.info(
        "Cleaning complete: rows %s→%s, cols %s→%s, outliers=%s",
        report.rows_before,
        report.rows_after,
        report.columns_before,
        report.columns_after,
        len(report.outlier_summaries),
    )

    return result, report
