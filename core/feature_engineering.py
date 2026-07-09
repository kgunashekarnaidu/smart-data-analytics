"""
Feature engineering: datetime features and categorical encoding.

Builds model-ready features from cleaned data with cardinality-aware
encoding (label, one-hot, or target encoding).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from core.cleaner import extract_datetime_features, parse_date_columns
from core.loader import detect_date_columns
from core.utils import (
    EncodingStrategy,
    choose_encoding_strategy,
    classify_columns,
    get_logger,
    is_likely_id_column,
)

logger = get_logger("feature_engineering")


@dataclass
class EncodingDetail:
    """Encoding applied to a single categorical column."""

    column: str
    strategy: EncodingStrategy
    unique_count: int
    output_columns: list[str] = field(default_factory=list)


@dataclass
class FeatureEngineeringReport:
    """Summary of feature engineering steps."""

    target_column: str
    raw_feature_count: int
    final_feature_count: int
    datetime_columns: list[str] = field(default_factory=list)
    datetime_features_added: list[str] = field(default_factory=list)
    id_columns_excluded: list[str] = field(default_factory=list)
    dropped_columns: list[str] = field(default_factory=list)
    encoding_details: list[EncodingDetail] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)

    def encoding_summary_df(self) -> pd.DataFrame:
        """Tabular encoding plan for UI display."""
        if not self.encoding_details:
            return pd.DataFrame(columns=["Column", "Strategy", "Unique Values", "Output Columns"])

        return pd.DataFrame(
            [
                {
                    "Column": detail.column,
                    "Strategy": detail.strategy,
                    "Unique Values": detail.unique_count,
                    "Output Columns": len(detail.output_columns),
                }
                for detail in self.encoding_details
            ]
        )


@dataclass
class FeatureEngineeringResult:
    """Output artifacts from feature engineering."""

    features: pd.DataFrame
    target: pd.Series
    feature_names: list[str]
    label_encoders: dict[str, LabelEncoder] = field(default_factory=dict)
    target_encoding_maps: dict[str, dict[str, float]] = field(default_factory=dict)
    onehot_columns: dict[str, list[str]] = field(default_factory=dict)
    report: FeatureEngineeringReport | None = None

    @property
    def processed_dataframe(self) -> pd.DataFrame:
        """Features joined with target for downstream UI display."""
        output = self.features.copy()
        output[self.report.target_column if self.report else "target"] = self.target.values
        return output


def get_datetime_columns(df: pd.DataFrame) -> list[str]:
    """Return parsed datetime columns plus detectable date-like object columns."""
    parsed = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()
    detected = detect_date_columns(df)
    combined: list[str] = []
    for column in parsed + detected:
        if column not in combined:
            combined.append(str(column))
    return combined


def plan_encodings(df: pd.DataFrame, categorical_columns: list[str]) -> dict[str, EncodingStrategy]:
    """Decide encoding strategy per categorical column based on cardinality."""
    plans: dict[str, EncodingStrategy] = {}
    for column in categorical_columns:
        if column not in df.columns:
            continue
        unique_count = int(df[column].nunique(dropna=True))
        plans[column] = choose_encoding_strategy(unique_count)
    return plans


def _apply_label_encoding(
    df: pd.DataFrame,
    column: str,
    encoders: dict[str, LabelEncoder],
) -> tuple[pd.DataFrame, list[str]]:
    result = df.copy()
    encoder = LabelEncoder()
    values = result[column].astype(str)
    result[column] = encoder.fit_transform(values)
    encoders[column] = encoder
    return result, [column]


def _apply_one_hot_encoding(df: pd.DataFrame, column: str) -> tuple[pd.DataFrame, list[str]]:
    dummies = pd.get_dummies(df[column].astype(str), prefix=column, dtype=int)
    result = pd.concat([df.drop(columns=[column]), dummies], axis=1)
    return result, dummies.columns.tolist()


def _apply_target_encoding(
    df: pd.DataFrame,
    column: str,
    target: pd.Series,
    smoothing: float = 10.0,
) -> tuple[pd.DataFrame, list[str], dict[str, float]]:
    """Mean target encoding with smoothing to reduce overfitting on small groups."""
    result = df.copy()
    target_numeric = pd.to_numeric(target, errors="coerce")
    temp = pd.DataFrame({column: df[column].astype(str), "_target": target_numeric})
    global_mean = float(temp["_target"].mean())

    stats = temp.groupby(column)["_target"].agg(["mean", "count"])
    encoded_map = {
        str(idx): float((count * mean + smoothing * global_mean) / (count + smoothing))
        for idx, (mean, count) in stats.iterrows()
    }

    encoded_col = f"{column}_target_enc"
    result[encoded_col] = result[column].astype(str).map(encoded_map).fillna(global_mean)
    result = result.drop(columns=[column])
    return result, [encoded_col], encoded_map


def encode_categorical_columns(
    df: pd.DataFrame,
    categorical_columns: list[str],
    target: pd.Series,
    *,
    use_target_encoding: bool = True,
    encoders: dict[str, LabelEncoder] | None = None,
) -> tuple[pd.DataFrame, list[EncodingDetail], dict[str, LabelEncoder], dict[str, dict[str, float]], dict[str, list[str]]]:
    """
    Encode categorical columns using cardinality-based strategies.

    Args:
        df: Feature dataframe (without target).
        categorical_columns: Columns to encode.
        target: Target series aligned with ``df``.
        use_target_encoding: Allow target encoding for medium-cardinality columns.
        encoders: Optional existing label encoder store.

    Returns:
        Encoded dataframe, encoding details, label encoders, target maps, one-hot column lists.
    """
    result = df.copy()
    label_encoders: dict[str, LabelEncoder] = encoders or {}
    target_maps: dict[str, dict[str, float]] = {}
    onehot_columns: dict[str, list[str]] = {}
    details: list[EncodingDetail] = []

    plans = plan_encodings(result, categorical_columns)

    for column, strategy in plans.items():
        if column not in result.columns:
            continue

        unique_count = int(result[column].nunique(dropna=True))

        if strategy == "target" and not use_target_encoding:
            strategy = "label"

        if strategy == "onehot":
            result, output_cols = _apply_one_hot_encoding(result, column)
            onehot_columns[column] = output_cols
        elif strategy == "target":
            result, output_cols, encoding_map = _apply_target_encoding(result, column, target)
            target_maps[column] = encoding_map
        else:
            result, output_cols = _apply_label_encoding(result, column, label_encoders)

        details.append(
            EncodingDetail(
                column=column,
                strategy=strategy,
                unique_count=unique_count,
                output_columns=output_cols,
            )
        )

    return result, details, label_encoders, target_maps, onehot_columns


def select_feature_columns(
    df: pd.DataFrame,
    target_column: str,
    *,
    exclude_ids: bool = True,
) -> tuple[list[str], list[str], list[str]]:
    """
    Choose usable feature columns excluding target, raw dates, and IDs.

    Returns:
        feature_columns, datetime_columns, excluded_id_columns
    """
    datetime_columns = get_datetime_columns(df)
    excluded_ids: list[str] = []
    feature_columns: list[str] = []

    for column in df.columns:
        if column == target_column:
            continue
        if column in datetime_columns and pd.api.types.is_datetime64_any_dtype(df[column]):
            continue
        if exclude_ids and is_likely_id_column(df[column]):
            excluded_ids.append(column)
            continue
        feature_columns.append(column)

    return feature_columns, datetime_columns, excluded_ids


def engineer_features(
    df: pd.DataFrame,
    target_column: str,
    *,
    use_target_encoding: bool = True,
    add_datetime_features: bool = True,
    exclude_ids: bool = True,
) -> FeatureEngineeringResult:
    """
    Transform cleaned data into encoded, model-ready features.

    Steps:
        1. Parse any remaining date columns
        2. Extract calendar features (if enabled)
        3. Drop raw datetime columns and ID-like columns from features
        4. Encode categorical columns (label / one-hot / target)

    Args:
        df: Cleaned dataframe.
        target_column: Name of prediction target.
        use_target_encoding: Enable target encoding for medium-cardinality columns.
        add_datetime_features: Extract year/month/quarter/week/day/weekend features.
        exclude_ids: Exclude high-cardinality ID columns from features.

    Returns:
        FeatureEngineeringResult with features, target, encoders, and report.
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataframe.")

    actions: list[str] = []
    working = df.copy()
    target = working[target_column].copy()

    # Ensure dates are parsed
    date_cols = get_datetime_columns(working)
    working, parsed_dates = parse_date_columns(working, date_cols)
    if parsed_dates:
        actions.append(f"Parsed datetime columns: {', '.join(parsed_dates)}")

    datetime_features: list[str] = []
    if add_datetime_features and parsed_dates:
        working, datetime_features = extract_datetime_features(working, parsed_dates)
        if datetime_features:
            actions.append(f"Added calendar features: {', '.join(datetime_features)}")

    feature_columns, datetime_columns, excluded_ids = select_feature_columns(
        working,
        target_column,
        exclude_ids=exclude_ids,
    )

    dropped_columns = [
        col for col in working.columns if col not in feature_columns and col != target_column
    ]

    features = working[feature_columns].copy()
    groups = classify_columns(features)
    categorical_columns = list(groups.categorical)

    encoded_features, encoding_details, label_encoders, target_maps, onehot_columns = (
        encode_categorical_columns(
            features,
            categorical_columns,
            target,
            use_target_encoding=use_target_encoding,
        )
    )

    if encoding_details:
        actions.append(f"Encoded {len(encoding_details)} categorical column(s)")

    feature_names = encoded_features.columns.tolist()
    report = FeatureEngineeringReport(
        target_column=target_column,
        raw_feature_count=len(feature_columns),
        final_feature_count=len(feature_names),
        datetime_columns=parsed_dates,
        datetime_features_added=datetime_features,
        id_columns_excluded=excluded_ids,
        dropped_columns=dropped_columns,
        encoding_details=encoding_details,
        actions=actions,
    )

    logger.info(
        "Feature engineering complete: %s raw features -> %s encoded features",
        report.raw_feature_count,
        report.final_feature_count,
    )

    return FeatureEngineeringResult(
        features=encoded_features,
        target=target,
        feature_names=feature_names,
        label_encoders=label_encoders,
        target_encoding_maps=target_maps,
        onehot_columns=onehot_columns,
        report=report,
    )


def preview_encoding_plan(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """Preview encoding strategies before running the full pipeline."""
    feature_columns, _, excluded_ids = select_feature_columns(df, target_column)
    features = df[feature_columns]
    categorical = classify_columns(features).categorical
    plans = plan_encodings(features, list(categorical))

    rows = [
        {
            "Column": column,
            "Unique Values": int(df[column].nunique(dropna=True)),
            "Planned Strategy": plans.get(column, "numeric"),
        }
        for column in feature_columns
        if column in plans
    ]

    if excluded_ids:
        for column in excluded_ids:
            rows.append(
                {
                    "Column": column,
                    "Unique Values": int(df[column].nunique(dropna=True)),
                    "Planned Strategy": "excluded (id-like)",
                }
            )

    return pd.DataFrame(rows)
