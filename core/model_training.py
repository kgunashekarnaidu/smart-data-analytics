"""
Model training, comparison, cross-validation, and artifact persistence.

Trains multiple scikit-learn and XGBoost models, selects the best performer,
and saves the winning pipeline with Joblib.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor

from core.preprocessor import model_requires_scaling
from core.utils import (
    RANDOM_STATE,
    ProblemType,
    get_classification_metrics,
    get_logger,
    get_regression_metrics,
)

logger = get_logger("model_training")

ARTIFACTS_DIR = Path("artifacts")
DEFAULT_CV_FOLDS = 5


@dataclass
class ModelEvaluation:
    """Evaluation results for a single trained model."""

    name: str
    metrics: dict[str, float]
    cv_mean: float
    cv_std: float
    estimator: Any


@dataclass
class TrainingResult:
    """Output of the full model comparison workflow."""

    problem_type: ProblemType
    results_df: pd.DataFrame
    models: dict[str, Any]
    best_model_name: str
    best_model: Any
    best_metrics: dict[str, float]
    feature_names: list[str]
    target_encoder: LabelEncoder | None = None
    feature_importance: pd.DataFrame | None = None
    artifact_path: str | None = None
    evaluations: list[ModelEvaluation] = field(default_factory=list)


def _encode_classification_targets(
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, LabelEncoder]:
    encoder = LabelEncoder()
    y_train_enc = encoder.fit_transform(y_train.astype(str))
    y_test_enc = encoder.transform(y_test.astype(str))
    return y_train_enc, y_test_enc, encoder


def _select_features(
    model_name: str,
    X_scaled: np.ndarray,
    X_raw: np.ndarray,
) -> np.ndarray:
    """Route models to scaled or raw feature matrices."""
    return X_scaled if model_requires_scaling(model_name) else X_raw


def get_regression_estimators() -> dict[str, Any]:
    """Return default regression model zoo."""
    return {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(max_depth=8, random_state=RANDOM_STATE),
        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            random_state=RANDOM_STATE,
        ),
        "XGBoost": XGBRegressor(
            n_estimators=100,
            random_state=RANDOM_STATE,
            verbosity=0,
            n_jobs=-1,
        ),
    }


def get_classification_estimators() -> dict[str, Any]:
    """Return default classification model zoo."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
        ),
        "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
        "XGBoost": XGBClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE,
            verbosity=0,
            n_jobs=-1,
            eval_metric="logloss",
        ),
    }


def evaluate_regression(model: Any, X_test: np.ndarray, y_test: np.ndarray) -> dict[str, float]:
    """Compute regression metrics on hold-out data."""
    predictions = model.predict(X_test)
    return {
        "MAE": float(mean_absolute_error(y_test, predictions)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, predictions))),
        "R2": float(r2_score(y_test, predictions)),
    }


def evaluate_classification(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, float]:
    """Compute classification metrics on hold-out data."""
    predictions = model.predict(X_test)
    metrics: dict[str, float] = {
        "Accuracy": float(accuracy_score(y_test, predictions)),
        "Precision": float(
            precision_score(y_test, predictions, average="weighted", zero_division=0)
        ),
        "Recall": float(
            recall_score(y_test, predictions, average="weighted", zero_division=0)
        ),
        "F1": float(f1_score(y_test, predictions, average="weighted", zero_division=0)),
    }

    try:
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(X_test)
            n_classes = len(np.unique(y_test))
            if n_classes == 2:
                metrics["ROC AUC"] = float(roc_auc_score(y_test, probabilities[:, 1]))
            elif n_classes > 2:
                metrics["ROC AUC"] = float(
                    roc_auc_score(
                        y_test,
                        probabilities,
                        multi_class="ovr",
                        average="weighted",
                    )
                )
    except ValueError:
        metrics["ROC AUC"] = float("nan")

    return metrics


def extract_feature_importance(
    model: Any,
    feature_names: list[str],
) -> pd.DataFrame | None:
    """Extract feature importance from tree-based models when available."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        coef = model.coef_
        importances = np.abs(coef).mean(axis=0) if coef.ndim > 1 else np.abs(coef)
    else:
        return None

    importance_df = pd.DataFrame(
        {"Feature": feature_names, "Importance": importances}
    ).sort_values("Importance", ascending=False)
    return importance_df.reset_index(drop=True)


def _rank_results(results_df: pd.DataFrame, problem_type: ProblemType) -> pd.DataFrame:
    """Sort models best-first depending on task type."""
    if problem_type == ProblemType.REGRESSION:
        return results_df.sort_values(["R2", "RMSE"], ascending=[False, True]).reset_index(
            drop=True
        )
    return results_df.sort_values(
        ["F1", "Accuracy", "ROC AUC"], ascending=[False, False, False]
    ).reset_index(drop=True)


def train_all_models(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    X_train_raw: np.ndarray,
    X_test_raw: np.ndarray,
    *,
    problem_type: ProblemType,
    feature_names: list[str],
    cv_folds: int = DEFAULT_CV_FOLDS,
) -> TrainingResult:
    """
    Train and compare all models for regression or classification.

    Args:
        X_train: Scaled training features.
        X_test: Scaled test features.
        y_train: Training target.
        y_test: Test target.
        X_train_raw: Unscaled training features for tree models.
        X_test_raw: Unscaled test features for tree models.
        problem_type: Regression or classification task.
        feature_names: Ordered feature names.
        cv_folds: Cross-validation fold count.

    Returns:
        TrainingResult with leaderboard, fitted models, and best model.
    """
    target_encoder: LabelEncoder | None = None
    if problem_type == ProblemType.CLASSIFICATION:
        y_train_fit, y_test_fit, target_encoder = _encode_classification_targets(
            y_train, y_test
        )
        estimators = get_classification_estimators()
        scoring = "f1_weighted"
    else:
        y_train_fit = np.asarray(y_train, dtype=float)
        y_test_fit = np.asarray(y_test, dtype=float)
        estimators = get_regression_estimators()
        scoring = "r2"

    evaluations: list[ModelEvaluation] = []
    fitted_models: dict[str, Any] = {}

    for name, estimator in estimators.items():
        X_tr = _select_features(name, X_train, X_train_raw)
        X_te = _select_features(name, X_test, X_test_raw)

        logger.info("Training model: %s", name)
        model = estimator
        model.fit(X_tr, y_train_fit)

        cv_scores = cross_val_score(
            model,
            X_tr,
            y_train_fit,
            cv=cv_folds,
            scoring=scoring,
            n_jobs=-1,
        )

        if problem_type == ProblemType.REGRESSION:
            metrics = evaluate_regression(model, X_te, y_test_fit)
        else:
            metrics = evaluate_classification(model, X_te, y_test_fit)

        metrics["CV Mean"] = float(cv_scores.mean())
        metrics["CV Std"] = float(cv_scores.std())

        evaluations.append(
            ModelEvaluation(
                name=name,
                metrics=metrics,
                cv_mean=float(cv_scores.mean()),
                cv_std=float(cv_scores.std()),
                estimator=model,
            )
        )
        fitted_models[name] = model

    rows = [{"Model": ev.name, **ev.metrics} for ev in evaluations]
    results_df = pd.DataFrame(rows)
    results_df = _rank_results(results_df, problem_type)

    best_model_name = results_df.iloc[0]["Model"]
    best_model = fitted_models[best_model_name]
    best_metrics = {
        col: float(results_df.iloc[0][col])
        for col in results_df.columns
        if col != "Model"
    }

    importance = extract_feature_importance(best_model, feature_names)

    logger.info("Best model: %s", best_model_name)

    return TrainingResult(
        problem_type=problem_type,
        results_df=results_df,
        models=fitted_models,
        best_model_name=best_model_name,
        best_model=best_model,
        best_metrics=best_metrics,
        feature_names=feature_names,
        target_encoder=target_encoder,
        feature_importance=importance,
        evaluations=evaluations,
    )


def save_training_artifacts(
    result: TrainingResult,
    *,
    preprocessor: Any = None,
    label_encoders: dict[str, Any] | None = None,
    target_encoding_maps: dict[str, dict[str, float]] | None = None,
    target_column: str | None = None,
    onehot_columns: dict[str, list[str]] | None = None,
    fe_report: Any = None,
    input_columns: list[str] | None = None,
    output_dir: Path = ARTIFACTS_DIR,
) -> str:
    """
    Persist best model and metadata with Joblib.

    Returns:
        Path to saved artifact file as string.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = output_dir / "best_model.joblib"

    payload = {
        "model": result.best_model,
        "model_name": result.best_model_name,
        "problem_type": result.problem_type.value,
        "feature_names": result.feature_names,
        "target_column": target_column,
        "target_encoder": result.target_encoder,
        "preprocessor": preprocessor,
        "label_encoders": label_encoders or {},
        "target_encoding_maps": target_encoding_maps or {},
        "onehot_columns": onehot_columns or {},
        "fe_report": fe_report,
        "input_columns": input_columns or result.feature_names,
        "metrics": result.best_metrics,
    }

    joblib.dump(payload, artifact_path)
    logger.info("Saved model artifact to %s", artifact_path)
    return str(artifact_path)


def predict_with_best_model(
    result: TrainingResult,
    features: np.ndarray,
) -> np.ndarray:
    """Generate predictions using the best model and correct feature scaling."""
    X = _select_features(result.best_model_name, features, features)
    predictions = result.best_model.predict(X)

    if result.target_encoder is not None:
        return result.target_encoder.inverse_transform(predictions.astype(int))
    return predictions
