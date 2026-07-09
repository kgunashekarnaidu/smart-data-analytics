"""
CSV loading, validation, and dataset inspection utilities.

Handles file ingestion from Streamlit uploads or local paths and produces
structured summaries for the upload / preview UI.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import BinaryIO, TextIO, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

FileSource = Union[str, Path, BinaryIO, TextIO, BytesIO]


class LoaderError(Exception):
    """Raised when a dataset cannot be loaded or fails validation."""


@dataclass(frozen=True)
class DatasetSummary:
    """High-level metrics for a loaded dataframe."""

    filename: str
    row_count: int
    column_count: int
    memory_bytes: int
    missing_values: int
    duplicate_rows: int
    numeric_columns: int
    categorical_columns: int
    datetime_columns: int

    @property
    def memory_kb(self) -> float:
        return self.memory_bytes / 1024

    @property
    def memory_mb(self) -> float:
        return self.memory_bytes / (1024 * 1024)

    def as_metrics(self) -> dict[str, str]:
        """Format summary values for Streamlit metric widgets."""
        memory = (
            f"{self.memory_mb:.2f} MB"
            if self.memory_mb >= 1
            else f"{self.memory_kb:.1f} KB"
        )
        return {
            "Rows": f"{self.row_count:,}",
            "Columns": str(self.column_count),
            "Missing Values": str(self.missing_values),
            "Duplicate Rows": str(self.duplicate_rows),
            "Memory": memory,
        }


@dataclass
class LoadResult:
    """Container returned after a successful CSV load."""

    dataframe: pd.DataFrame
    summary: DatasetSummary
    dtype_report: pd.DataFrame = field(repr=False)
    missing_report: pd.DataFrame = field(repr=False)
    date_columns: list[str] = field(default_factory=list)


def _normalize_filename(source: FileSource, explicit_name: str | None = None) -> str:
    if explicit_name:
        return explicit_name
    if isinstance(source, (str, Path)):
        return Path(source).name
    name = getattr(source, "name", None)
    return str(name) if name else "uploaded.csv"


def _read_csv(source: FileSource, encoding: str, sep: str) -> pd.DataFrame:
    """Read CSV from path or file-like object."""
    read_kwargs = {"encoding": encoding, "sep": sep, "low_memory": False}

    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise LoaderError(f"File not found: {path}")
        if path.stat().st_size == 0:
            raise LoaderError("The CSV file is empty.")
        logger.info("Loading CSV from path: %s", path)
        return pd.read_csv(path, **read_kwargs)

    if hasattr(source, "seek"):
        source.seek(0)

    logger.info("Loading CSV from file upload: %s", _normalize_filename(source))
    return pd.read_csv(source, **read_kwargs)


def validate_dataframe(df: pd.DataFrame, filename: str) -> None:
    """
    Validate that a dataframe is usable for downstream analytics.

    Raises:
        LoaderError: If the dataframe fails any validation rule.
    """
    if df is None:
        raise LoaderError("No dataframe was produced from the CSV file.")

    if df.empty:
        raise LoaderError(f"'{filename}' was read but contains no rows.")

    if df.shape[1] == 0:
        raise LoaderError(f"'{filename}' contains no columns.")

    if df.columns.duplicated().any():
        duplicated = df.columns[df.columns.duplicated()].tolist()
        raise LoaderError(
            f"Duplicate column names detected: {duplicated}. "
            "Please fix the CSV and upload again."
        )


def detect_date_columns(df: pd.DataFrame, sample_size: int = 50) -> list[str]:
    """
    Detect object columns that can be parsed as datetimes.

    Args:
        df: Input dataframe.
        sample_size: Number of non-null values to test per column.

    Returns:
        List of column names likely representing dates.
    """
    candidates: list[str] = []
    for column in df.select_dtypes(include=["object", "string"]).columns:
        sample = df[column].dropna().astype(str).head(sample_size)
        if sample.empty:
            continue
        try:
            parsed = pd.to_datetime(sample, errors="coerce")
            if parsed.notna().mean() >= 0.8:
                candidates.append(str(column))
        except (ValueError, TypeError, OverflowError):
            continue
    return candidates


def build_dtype_report(df: pd.DataFrame) -> pd.DataFrame:
    """Build a per-column dtype and cardinality report."""
    return pd.DataFrame(
        {
            "Column": df.columns.astype(str),
            "Dtype": df.dtypes.astype(str).values,
            "Non-Null": df.notnull().sum().values,
            "Null Count": df.isnull().sum().values,
            "Unique Values": df.nunique().values,
            "Sample Value": [
                str(df[col].dropna().iloc[0]) if df[col].notna().any() else ""
                for col in df.columns
            ],
        }
    )


def build_missing_report(df: pd.DataFrame) -> pd.DataFrame:
    """Build a missing-value report sorted by severity."""
    report = (
        df.isnull()
        .sum()
        .reset_index(name="Missing Count")
        .rename(columns={"index": "Column"})
    )
    if report.empty:
        return report

    report["Missing %"] = (report["Missing Count"] / len(df) * 100).round(2)
    report = report[report["Missing Count"] > 0].sort_values(
        "Missing Count", ascending=False
    )
    return report.reset_index(drop=True)


def build_summary(df: pd.DataFrame, filename: str) -> DatasetSummary:
    """Compute dataset-level summary statistics."""
    numeric_cols = df.select_dtypes(include=np.number).columns
    categorical_cols = df.select_dtypes(include=["object", "category", "string"]).columns
    datetime_cols = df.select_dtypes(include=["datetime", "datetimetz"]).columns

    return DatasetSummary(
        filename=filename,
        row_count=int(len(df)),
        column_count=int(df.shape[1]),
        memory_bytes=int(df.memory_usage(deep=True).sum()),
        missing_values=int(df.isnull().sum().sum()),
        duplicate_rows=int(df.duplicated().sum()),
        numeric_columns=len(numeric_cols),
        categorical_columns=len(categorical_cols),
        datetime_columns=len(datetime_cols),
    )


def load_csv(
    source: FileSource,
    *,
    filename: str | None = None,
    encoding: str = "utf-8",
    sep: str = ",",
) -> LoadResult:
    """
    Load, validate, and summarize a CSV dataset.

    Args:
        source: File path or file-like object (e.g. Streamlit ``UploadedFile``).
        filename: Optional display name when ``source`` is a buffer.
        encoding: Text encoding for the CSV reader.
        sep: Column delimiter.

    Returns:
        LoadResult with dataframe and inspection artifacts.

    Raises:
        LoaderError: On read failure or validation failure.
    """
    resolved_name = _normalize_filename(source, filename)

    try:
        df = _read_csv(source, encoding=encoding, sep=sep)
    except pd.errors.EmptyDataError as exc:
        raise LoaderError("The CSV file contains no data.") from exc
    except pd.errors.ParserError as exc:
        raise LoaderError(f"Could not parse CSV: {exc}") from exc
    except UnicodeDecodeError as exc:
        raise LoaderError(
            f"Encoding error while reading '{resolved_name}'. "
            "Try saving the file as UTF-8 and upload again."
        ) from exc
    except OSError as exc:
        raise LoaderError(f"Failed to read file: {exc}") from exc

    validate_dataframe(df, resolved_name)

    date_columns = detect_date_columns(df)
    summary = build_summary(df, resolved_name)
    dtype_report = build_dtype_report(df)
    missing_report = build_missing_report(df)

    logger.info(
        "Loaded '%s': rows=%s cols=%s missing=%s duplicates=%s",
        resolved_name,
        summary.row_count,
        summary.column_count,
        summary.missing_values,
        summary.duplicate_rows,
    )

    return LoadResult(
        dataframe=df,
        summary=summary,
        dtype_report=dtype_report,
        missing_report=missing_report,
        date_columns=date_columns,
    )


def load_csv_with_fallback_encoding(source: FileSource, filename: str | None = None) -> LoadResult:
    """
    Attempt UTF-8 first, then fall back to latin-1 for legacy CSV exports.

    Args:
        source: File path or file-like object.
        filename: Optional display name.

    Returns:
        LoadResult from the first successful encoding attempt.
    """
    try:
        return load_csv(source, filename=filename, encoding="utf-8")
    except LoaderError as utf8_error:
        if "Encoding error" not in str(utf8_error):
            raise
        logger.warning("UTF-8 failed for %s; retrying with latin-1", filename)
        return load_csv(source, filename=filename, encoding="latin-1")


def preview_dataframe(df: pd.DataFrame, n: int = 20, tail: bool = False) -> pd.DataFrame:
    """Return the first or last ``n`` rows for UI preview."""
    if tail:
        return df.tail(n)
    return df.head(n)
