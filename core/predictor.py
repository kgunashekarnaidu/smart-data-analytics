"""
Inference utilities: transform new rows and generate predictions.

Applies the same feature-engineering and scaling steps used during training,
using persisted encoders, maps, and the fitted preprocessor.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from core.cleaner import extract_datetime_features, parse_date_columns
from core.feature_engineering import FeatureEngineeringReport
from core.preprocessor import model_requires_scaling
from core.utils import ProblemType, get_logger, normalize_column_names

logger = get_logger("predictor")


@dataclass
class PredictionConfig:
    """Artifacts required to score new observations."""

    model: Any
    model_name: str
    problem_type: ProblemType
    target_column: str
    feature_names: list[str]
    input_columns: list[str]
    label_encoders: dict[str, LabelEncoder]
    target_encoding_maps: dict[str, dict[str, float]]
    onehot_columns: dict[str, list[str]]
    fe_report: FeatureEngineeringReport | None
    preprocessor: Any
    target_encoder: LabelEncoder | None = None


@dataclass
class PredictionResult:
    """Output of a batch or single-row prediction run."""

    predictions: np.ndarray
    predictions_df: pd.DataFrame
    feature_matrix: pd.DataFrame


def get_input_feature_columns(
    df: pd.DataFrame,
    target_column: str,
    fe_report: FeatureEngineeringReport | None,
) -> list[str]:
    """
    Return raw cleaned-dataset columns needed before encoding.

    These are the fields shown on the manual prediction form.
    """
    if fe_report is None:
        return [col for col in df.columns if col != target_column]

    excluded = set(fe_report.id_columns_excluded)
    raw_dates = set(fe_report.datetime_columns) - set(fe_report.datetime_features_added)

    columns: list[str] = []
    for column in df.columns:
        if column == target_column:
            continue
        if column in excluded:
            continue
        if column in raw_dates:
            continue
        columns.append(column)
    return columns


def _apply_label_encoding_inference(
    df: pd.DataFrame,
    column: str,
    encoder: LabelEncoder,
) -> pd.DataFrame:
    result = df.copy()
    values = result[column].astype(str)
    class_map = {str(cls): idx for idx, cls in enumerate(encoder.classes_)}
    fallback = int(class_map[str(encoder.classes_[0])])
    result[column] = values.map(class_map).fillna(fallback).astype(int)
    return result


def _apply_one_hot_inference(
    df: pd.DataFrame,
    column: str,
    expected_columns: list[str],
) -> pd.DataFrame:
    dummies = pd.get_dummies(df[column].astype(str), prefix=column, dtype=int)
    for expected in expected_columns:
        if expected not in dummies.columns:
            dummies[expected] = 0
    dummies = dummies.reindex(columns=expected_columns, fill_value=0)
    return pd.concat([df.drop(columns=[column]), dummies], axis=1)


def _apply_target_encoding_inference(
    df: pd.DataFrame,
    column: str,
    encoding_map: dict[str, float],
) -> pd.DataFrame:
    result = df.copy()
    encoded_col = f"{column}_target_enc"
    global_mean = float(np.mean(list(encoding_map.values()))) if encoding_map else 0.0
    result[encoded_col] = result[column].astype(str).map(encoding_map).fillna(global_mean)
    return result.drop(columns=[column])


def transform_rows_for_prediction(
    df: pd.DataFrame,
    config: PredictionConfig,
) -> pd.DataFrame:
    """
    Convert raw cleaned-dataset rows into encoded model features.

    Args:
        df: Rows containing ``config.input_columns`` (target optional).
        config: Fitted encoders, maps, and feature order from training.

    Returns:
        DataFrame with columns aligned to ``config.feature_names``.
    """
    if config.fe_report is None:
        raise ValueError("Feature engineering report is required for prediction.")

    working = normalize_column_names(df.copy())
    fe_report = config.fe_report

    for column in config.input_columns:
        if column not in working.columns:
            working[column] = np.nan

    features = working[config.input_columns].copy()

    date_cols = fe_report.datetime_columns
    if date_cols:
        features, _ = parse_date_columns(features, date_cols)
        if fe_report.datetime_features_added:
            features, _ = extract_datetime_features(features, date_cols)

    for detail in fe_report.encoding_details:
        column = detail.column
        if column not in features.columns:
            continue

        if detail.strategy == "onehot":
            expected = config.onehot_columns.get(column, detail.output_columns)
            features = _apply_one_hot_inference(features, column, expected)
        elif detail.strategy == "target":
            encoding_map = config.target_encoding_maps.get(column, {})
            features = _apply_target_encoding_inference(features, column, encoding_map)
        else:
            encoder = config.label_encoders.get(column)
            if encoder is None:
                continue
            features = _apply_label_encoding_inference(features, column, encoder)

    numeric_features = features.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    aligned = pd.DataFrame(
        {name: numeric_features[name] if name in numeric_features.columns else 0.0 for name in config.feature_names},
        index=features.index,
    )
    return aligned


def _prepare_model_matrix(
    features: pd.DataFrame,
    config: PredictionConfig,
) -> np.ndarray:
    matrix = features[config.feature_names].to_numpy(dtype=float)
    if config.preprocessor is not None and model_requires_scaling(config.model_name):
        matrix = config.preprocessor.transform(features)
    return np.asarray(matrix, dtype=float)


def predict_from_features(
    features: pd.DataFrame,
    config: PredictionConfig,
) -> np.ndarray:
    """Generate predictions from an already-encoded feature matrix."""
    matrix = _prepare_model_matrix(features, config)
    predictions = config.model.predict(matrix)

    if config.target_encoder is not None:
        return config.target_encoder.inverse_transform(predictions.astype(int))
    return predictions


def predict_dataframe(
    df: pd.DataFrame,
    config: PredictionConfig,
    *,
    actual_column: str | None = None,
) -> PredictionResult:
    """
    End-to-end prediction on raw cleaned-dataset rows.

    Args:
        df: Input rows (same schema as cleaned training data).
        config: Prediction artifacts from training.
        actual_column: Optional target column for comparison output.

    Returns:
        PredictionResult with arrays and a display-ready dataframe.
    """
    features = transform_rows_for_prediction(df, config)
    predictions = predict_from_features(features, config)

    output = df.copy()
    target_col = config.target_column
    prediction_col = f"predicted_{target_col}"

    output[prediction_col] = predictions

    if actual_column and actual_column in df.columns:
        output[f"actual_{target_col}"] = df[actual_column].values

    logger.info("Generated %s predictions with model '%s'", len(predictions), config.model_name)

    return PredictionResult(
        predictions=predictions,
        predictions_df=output,
        feature_matrix=features,
    )


def predict_test_split(
    *,
    config: PredictionConfig,
    X_test: np.ndarray,
    X_test_raw: np.ndarray,
    y_test: np.ndarray,
) -> PredictionResult:
    """Score the hold-out test set from preprocessing."""
    use_scaled = model_requires_scaling(config.model_name)
    matrix = X_test if use_scaled else X_test_raw
    raw_predictions = config.model.predict(matrix)

    if config.target_encoder is not None:
        predictions = config.target_encoder.inverse_transform(raw_predictions.astype(int))
    else:
        predictions = raw_predictions

    prediction_col = f"predicted_{config.target_column}"
    output = pd.DataFrame(
        {
            f"actual_{config.target_column}": y_test,
            prediction_col: predictions,
        }
    )

    return PredictionResult(
        predictions=np.asarray(predictions),
        predictions_df=output,
        feature_matrix=pd.DataFrame(matrix, columns=config.feature_names),
    )


def build_prediction_config(
    *,
    model: Any,
    model_name: str,
    problem_type: ProblemType | str,
    target_column: str,
    feature_names: list[str],
    input_columns: list[str],
    label_encoders: dict[str, LabelEncoder] | None = None,
    target_encoding_maps: dict[str, dict[str, float]] | None = None,
    onehot_columns: dict[str, list[str]] | None = None,
    fe_report: FeatureEngineeringReport | None = None,
    preprocessor: Any = None,
    target_encoder: LabelEncoder | None = None,
) -> PredictionConfig:
    """Construct a ``PredictionConfig`` from session or artifact fields."""
    if isinstance(problem_type, str):
        problem_type = ProblemType(problem_type)

    return PredictionConfig(
        model=model,
        model_name=model_name,
        problem_type=problem_type,
        target_column=target_column,
        feature_names=feature_names,
        input_columns=input_columns,
        label_encoders=label_encoders or {},
        target_encoding_maps=target_encoding_maps or {},
        onehot_columns=onehot_columns or {},
        fe_report=fe_report,
        preprocessor=preprocessor,
        target_encoder=target_encoder,
    )


def load_prediction_config(artifact_path: str | Path) -> PredictionConfig:
    """Load a ``PredictionConfig`` from a saved Joblib artifact."""
    payload = joblib.load(artifact_path)
    fe_report = payload.get("fe_report")

    return build_prediction_config(
        model=payload["model"],
        model_name=payload["model_name"],
        problem_type=payload["problem_type"],
        target_column=payload.get("target_column") or "target",
        feature_names=payload["feature_names"],
        input_columns=payload.get("input_columns") or payload["feature_names"],
        label_encoders=payload.get("label_encoders") or {},
        target_encoding_maps=payload.get("target_encoding_maps") or {},
        onehot_columns=payload.get("onehot_columns") or {},
        fe_report=fe_report,
        preprocessor=payload.get("preprocessor"),
        target_encoder=payload.get("target_encoder"),
    )
