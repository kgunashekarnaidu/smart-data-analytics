"""
Scikit-learn preprocessing: scaling, train-test split, and pipelines.

Works on feature-engineered numeric matrices and prepares data for model
training in Step 9. Tree-based models can use unscaled features; linear
models use scaled features.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from core.feature_engineering import FeatureEngineeringResult
from core.utils import RANDOM_STATE, ProblemType, detect_problem_type, get_logger

logger = get_logger("preprocessor")

ScalerType = Literal["standard", "minmax", "robust", "none"]

TREE_MODEL_KEYWORDS: tuple[str, ...] = (
    "decision tree",
    "random forest",
    "gradient boosting",
    "xgboost",
    "extra trees",
    "hist gradient",
)


@dataclass
class PreprocessReport:
    """Summary of preprocessing and data splitting."""

    target_column: str
    problem_type: str
    scaler_type: ScalerType
    scaling_applied: bool
    train_rows: int
    test_rows: int
    feature_count: int
    test_size: float
    random_state: int
    actions: list[str] = field(default_factory=list)


@dataclass
class PreprocessResult:
    """Train/test splits and fitted preprocessing artifacts."""

    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    X_train_raw: np.ndarray
    X_test_raw: np.ndarray
    feature_names: list[str]
    preprocessor: ColumnTransformer | None
    report: PreprocessReport
    target_column: str

    @property
    def train_test_dataframe(self) -> pd.DataFrame:
        """Combined preview of training labels for UI display."""
        return pd.DataFrame(
            {
                "split": ["train"] * len(self.y_train) + ["test"] * len(self.y_test),
                "target": np.concatenate([self.y_train, self.y_test]),
            }
        )


class IdentityScaler(TransformerMixin):
    """No-op scaler for tree-based models or when scaling is disabled."""

    def fit(self, X: np.ndarray, y: Any = None) -> IdentityScaler:
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=float)

    def fit_transform(self, X: np.ndarray, y: Any = None) -> np.ndarray:
        return self.transform(X)


def model_requires_scaling(model_name: str) -> bool:
    """Return True when a model family benefits from feature scaling."""
    lowered = model_name.lower()
    return not any(keyword in lowered for keyword in TREE_MODEL_KEYWORDS)


def get_scaler(scaler_type: ScalerType) -> Any:
    """Return a sklearn-compatible scaler instance."""
    scalers = {
        "standard": StandardScaler(),
        "minmax": MinMaxScaler(),
        "robust": RobustScaler(),
        "none": IdentityScaler(),
    }
    if scaler_type not in scalers:
        raise ValueError(f"Unsupported scaler type: {scaler_type}")
    return scalers[scaler_type]


def build_preprocessor(
    feature_names: list[str],
    scaler_type: ScalerType = "standard",
) -> ColumnTransformer | None:
    """
    Build a ``ColumnTransformer`` that scales all engineered features.

    Args:
        feature_names: Ordered feature column names.
        scaler_type: Scaling strategy or ``none`` for passthrough.

    Returns:
        Fitted-ready preprocessor, or ``None`` when there are no features.
    """
    if not feature_names:
        raise ValueError("At least one feature column is required for preprocessing.")

    if scaler_type == "none":
        return None

    scaler = get_scaler(scaler_type)
    return ColumnTransformer(
        transformers=[("features", scaler, feature_names)],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_model_pipeline(
    preprocessor: ColumnTransformer | None,
    model: Any,
) -> Pipeline:
    """
    Combine preprocessing and an estimator into a single sklearn ``Pipeline``.

    Args:
        preprocessor: Optional column transformer (may be ``None``).
        model: Scikit-learn compatible estimator.

    Returns:
        sklearn Pipeline ready for ``fit`` / ``predict``.
    """
    steps: list[tuple[str, Any]] = []
    if preprocessor is not None:
        steps.append(("preprocessor", preprocessor))
    steps.append(("model", model))
    return Pipeline(steps=steps)


def _to_numpy_features(features: pd.DataFrame, feature_names: list[str]) -> np.ndarray:
    matrix = features[feature_names].copy()
    return matrix.apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(dtype=float)


def _to_numpy_target(target: pd.Series, problem_type: ProblemType) -> np.ndarray:
    if problem_type == ProblemType.CLASSIFICATION:
        return target.astype(str).to_numpy()
    return pd.to_numeric(target, errors="coerce").to_numpy(dtype=float)


def preprocess_dataset(
    features: pd.DataFrame,
    target: pd.Series,
    target_column: str,
    *,
    scaler_type: ScalerType = "standard",
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
) -> PreprocessResult:
    """
    Split data and optionally fit a scaling preprocessor on training features.

    Always stores both raw and scaled train/test matrices so Step 9 can
    route tree models to raw data and linear models to scaled data.

    Args:
        features: Engineered feature dataframe.
        target: Target series aligned with ``features``.
        target_column: Target column name for reporting.
        scaler_type: Scaling strategy applied to training data.
        test_size: Hold-out fraction for testing.
        random_state: Reproducible split seed.

    Returns:
        PreprocessResult with arrays, fitted preprocessor, and report.
    """
    feature_names = features.columns.tolist()
    problem = detect_problem_type(target)

    X = _to_numpy_features(features, feature_names)
    y = _to_numpy_target(target, problem)

    if np.isnan(y).any() and problem == ProblemType.REGRESSION:
        valid_mask = ~np.isnan(y)
        X = X[valid_mask]
        y = y[valid_mask]
        features = features.loc[valid_mask].reset_index(drop=True)
        target = target.loc[valid_mask].reset_index(drop=True)

    stratify = y if problem == ProblemType.CLASSIFICATION and len(np.unique(y)) > 1 else None

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    actions = [
        f"Train/test split: {len(y_train):,} train / {len(y_test):,} test "
        f"({test_size:.0%} hold-out)",
    ]

    preprocessor = build_preprocessor(feature_names, scaler_type)
    scaling_applied = preprocessor is not None

    if preprocessor is not None:
        X_train = preprocessor.fit_transform(
            pd.DataFrame(X_train_raw, columns=feature_names)
        )
        X_test = preprocessor.transform(
            pd.DataFrame(X_test_raw, columns=feature_names)
        )
        actions.append(f"Applied {scaler_type} scaling on {len(feature_names)} features")
    else:
        X_train = X_train_raw
        X_test = X_test_raw
        actions.append("Scaling skipped (raw features preserved for tree-based models)")

    report = PreprocessReport(
        target_column=target_column,
        problem_type=problem.value,
        scaler_type=scaler_type,
        scaling_applied=scaling_applied,
        train_rows=len(y_train),
        test_rows=len(y_test),
        feature_count=len(feature_names),
        test_size=test_size,
        random_state=random_state,
        actions=actions,
    )

    logger.info(
        "Preprocessing complete: train=%s test=%s scaler=%s features=%s",
        report.train_rows,
        report.test_rows,
        scaler_type,
        report.feature_count,
    )

    return PreprocessResult(
        X_train=np.asarray(X_train, dtype=float),
        X_test=np.asarray(X_test, dtype=float),
        y_train=y_train,
        y_test=y_test,
        X_train_raw=np.asarray(X_train_raw, dtype=float),
        X_test_raw=np.asarray(X_test_raw, dtype=float),
        feature_names=feature_names,
        preprocessor=preprocessor,
        report=report,
        target_column=target_column,
    )


def preprocess_from_feature_result(
    fe_result: FeatureEngineeringResult,
    *,
    scaler_type: ScalerType = "standard",
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
) -> PreprocessResult:
    """Convenience wrapper around ``preprocess_dataset`` for Step 6 output."""
    target_column = (
        fe_result.report.target_column if fe_result.report else "target"
    )
    return preprocess_dataset(
        fe_result.features,
        fe_result.target,
        target_column,
        scaler_type=scaler_type,
        test_size=test_size,
        random_state=random_state,
    )


def recommend_scaler(problem_type: ProblemType) -> ScalerType:
    """Suggest a default scaler based on problem type."""
    if problem_type == ProblemType.CLASSIFICATION:
        return "robust"
    return "standard"
