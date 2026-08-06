"""PostgreSQL data bridge for Grafana integration.

Writes pipeline DataFrames to PostgreSQL so Grafana can query live data.
Uses SQLAlchemy for ORM-free DataFrame persistence and psycopg2 as the driver.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

try:
    from core.utils import get_logger
    logger = get_logger("data_bridge")
except ImportError:
    logger = logging.getLogger("data_bridge")
    logger.setLevel(logging.INFO)

# Connection defaults
DEFAULT_HOST = "localhost"
DEFAULT_PORT = 5432
DEFAULT_USER = "postgres"
DEFAULT_DB = "dataml_pro"

_engines: dict[str, Engine] = {}

def get_engine(password: str, host=DEFAULT_HOST, port=DEFAULT_PORT, user=DEFAULT_USER, database=DEFAULT_DB) -> Engine:
    """
    Creates SQLAlchemy engine with postgresql+psycopg2:// URL.
    Supports sslmode=require for cloud PostgreSQL providers (Neon, Supabase, Render).
    """
    engine_key = f"{host}:{port}:{user}:{database}"
    if engine_key not in _engines:
        ssl_param = "?sslmode=require" if host not in ("localhost", "127.0.0.1") else ""
        url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}{ssl_param}"
        engine = create_engine(url, pool_pre_ping=True)
        _engines[engine_key] = engine
    return _engines[engine_key]

def ensure_database(password: str, host=DEFAULT_HOST, port=DEFAULT_PORT, user=DEFAULT_USER, database=DEFAULT_DB) -> None:
    """
    Connects to PostgreSQL and creates the target database if it doesn't exist.
    Catches permission errors for cloud databases with pre-created databases.
    """
    try:
        ssl_param = "?sslmode=require" if host not in ("localhost", "127.0.0.1") else ""
        default_db = "postgres" if database != "neondb" else "neondb"
        url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{default_db}{ssl_param}"
        engine = create_engine(url, isolation_level="AUTOCOMMIT")
        with engine.connect() as conn:
            res = conn.execute(text(f"SELECT 1 FROM pg_database WHERE datname = '{database}'"))
            if not res.scalar():
                try:
                    conn.execute(text(f"CREATE DATABASE {database}"))
                    logger.info(f"Database {database} created.")
                except Exception as db_err:
                    logger.warning(f"Could not create database {database}: {db_err}")
            else:
                logger.info(f"Database {database} already exists.")
    except Exception as e:
        logger.warning(f"Database check warning for {database}: {e}")
        logger.error(f"Failed to ensure database {database}: {e}")

def save_dataframe(df: pd.DataFrame, table_name: str, engine: Engine, if_exists: str = 'replace') -> int:
    """
    Writes DataFrame to PostgreSQL using df.to_sql().
    Normalizes column names to lowercase to prevent PostgreSQL case-sensitivity issues.
    
    Args:
        df (pd.DataFrame): DataFrame to save
        table_name (str): Destination table name
        engine (Engine): SQLAlchemy engine
        if_exists (str): Behavior when table exists ('fail', 'replace', 'append')
        
    Returns:
        int: Number of rows written
    """
    if df is None or df.empty:
        logger.info(f"DataFrame for {table_name} is empty. Skipping.")
        return 0
    try:
        clean_df = df.copy()
        clean_df.columns = [str(c).strip().lower() for c in clean_df.columns]
        rows = clean_df.to_sql(table_name, con=engine, if_exists=if_exists, index=False, chunksize=5000)
        logger.info(f"Successfully saved {rows} rows to {table_name}")
        return rows if rows is not None else len(clean_df)
    except Exception as e:
        logger.error(f"Error saving DataFrame to {table_name}: {e}")
        return 0

def save_pipeline_metadata(metadata: dict[str, Any], engine: Engine) -> None:
    """
    Writes key-value metadata to pipeline_metadata table.
    
    Args:
        metadata (dict[str, Any]): Metadata dictionary
        engine (Engine): SQLAlchemy engine
    """
    try:
        now = datetime.now(timezone.utc)
        records = [{"key": k, "value": str(v), "updated_at": now} for k, v in metadata.items()]
        
        df = pd.DataFrame(records)
        save_dataframe(df, "pipeline_metadata", engine, if_exists="replace")
    except Exception as e:
        logger.error(f"Error saving pipeline metadata: {e}")


# ---------------------------------------------------------------------------
# Advanced computed-statistics helpers (for Grafana advanced panels)
# ---------------------------------------------------------------------------

def _compute_column_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-column descriptive statistics for all numeric columns.
    
    Returns a long-format DataFrame with columns:
    column_name, mean, median, std, min, max, skewness, kurtosis, null_count, null_pct
    """
    num_cols = df.select_dtypes(include="number").columns.tolist()
    if not num_cols:
        return pd.DataFrame()

    records = []
    for col in num_cols:
        series = df[col]
        records.append({
            "column_name": col,
            "mean": float(series.mean()) if not series.isna().all() else None,
            "median": float(series.median()) if not series.isna().all() else None,
            "std": float(series.std()) if not series.isna().all() else None,
            "min_val": float(series.min()) if not series.isna().all() else None,
            "max_val": float(series.max()) if not series.isna().all() else None,
            "skewness": float(series.skew()) if not series.isna().all() else None,
            "kurtosis": float(series.kurtosis()) if not series.isna().all() else None,
            "null_count": int(series.isna().sum()),
            "null_pct": round(float(series.isna().mean()) * 100, 2),
        })
    return pd.DataFrame(records)


def _compute_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute pairwise Pearson correlations in long format.
    
    Returns DataFrame with columns: col1, col2, correlation
    """
    num_df = df.select_dtypes(include="number")
    if num_df.shape[1] < 2:
        return pd.DataFrame()

    # Limit to top 20 columns to keep size reasonable
    if num_df.shape[1] > 20:
        num_df = num_df.iloc[:, :20]

    corr = num_df.corr()
    records = []
    cols = corr.columns.tolist()
    for i, c1 in enumerate(cols):
        for c2 in cols[i:]:
            records.append({
                "col1": c1,
                "col2": c2,
                "correlation": round(float(corr.loc[c1, c2]), 4),
            })
    return pd.DataFrame(records)


def _compute_missing_data_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-column missing data counts and percentages.
    
    Returns DataFrame with columns: column_name, missing_count, missing_pct, dtype
    """
    records = []
    for col in df.columns:
        mc = int(df[col].isna().sum())
        records.append({
            "column_name": col,
            "missing_count": mc,
            "missing_pct": round(float(mc / len(df)) * 100, 2) if len(df) > 0 else 0.0,
            "dtype": str(df[col].dtype),
        })
    return pd.DataFrame(records)


def _normalize_predictions_df(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensures predictions_df has standardized 'actual' and 'predicted' columns
    in addition to any target-prefixed column names (e.g., actual_Weekly_Sales).
    """
    if predictions_df is None or predictions_df.empty:
        return predictions_df
        
    df = predictions_df.copy()
    cols_map = {str(c).strip().lower(): c for c in df.columns}
    
    actual_col = None
    if "actual" in cols_map:
        actual_col = cols_map["actual"]
    else:
        for norm, orig in cols_map.items():
            if norm.startswith("actual_") or norm in ("y_true", "y_test", "target", "label"):
                actual_col = orig
                break

    pred_col = None
    if "predicted" in cols_map:
        pred_col = cols_map["predicted"]
    elif "prediction" in cols_map:
        pred_col = cols_map["prediction"]
    else:
        for norm, orig in cols_map.items():
            if norm.startswith("predicted_") or norm.startswith("prediction_") or norm in ("y_pred", "pred"):
                pred_col = orig
                break

    if actual_col and "actual" not in [str(c).lower() for c in df.columns]:
        df["actual"] = pd.to_numeric(df[actual_col], errors="coerce")
    if pred_col and "predicted" not in [str(c).lower() for c in df.columns]:
        df["predicted"] = pd.to_numeric(df[pred_col], errors="coerce")

    return df


def _compute_prediction_residuals(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute residuals (actual - predicted) with row index.
    
    Expects a DataFrame with 'actual' and 'predicted' columns (case-insensitive).
    Returns DataFrame with: row_idx, actual, predicted, residual, abs_residual
    """
    df = _normalize_predictions_df(predictions_df)
    if df is None or df.empty:
        return pd.DataFrame()

    cols_map = {str(c).strip().lower(): c for c in df.columns}
    if "actual" not in cols_map or "predicted" not in cols_map:
        return pd.DataFrame()

    actual_col = cols_map["actual"]
    pred_col = cols_map["predicted"]

    out = pd.DataFrame()
    out["actual"] = pd.to_numeric(df[actual_col], errors="coerce")
    out["predicted"] = pd.to_numeric(df[pred_col], errors="coerce")
    out = out.dropna(subset=["actual", "predicted"]).copy()

    if out.empty:
        return pd.DataFrame()

    out["row_idx"] = range(1, len(out) + 1)
    out["residual"] = out["actual"] - out["predicted"]
    out["abs_residual"] = out["residual"].abs()
    return out[["row_idx", "actual", "predicted", "residual", "abs_residual"]]


def _compute_data_distribution(df: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    """
    Histogram bin counts for each numeric column.
    
    Returns DataFrame with: column_name, bin_start, bin_end, bin_label, count
    """
    import numpy as _np

    num_cols = df.select_dtypes(include="number").columns.tolist()
    if not num_cols:
        return pd.DataFrame()

    # Limit to top 15 columns
    num_cols = num_cols[:15]
    records = []
    for col in num_cols:
        series = df[col].dropna()
        if series.empty:
            continue
        counts, edges = _np.histogram(series, bins=n_bins)
        for i in range(len(counts)):
            records.append({
                "column_name": str(col).lower(),
                "bin_start": round(float(edges[i]), 4),
                "bin_end": round(float(edges[i + 1]), 4),
                "bin_label": f"{edges[i]:.2f} – {edges[i+1]:.2f}",
                "count": int(counts[i]),
            })
    return pd.DataFrame(records)


def _build_model_metrics_detail(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Reshape model_results into a long-format metrics table for grouped bar charts.
    
    Returns DataFrame with: model, metric, value
    """
    if results_df is None or results_df.empty:
        return pd.DataFrame()

    cols_map = {str(c).strip().lower(): c for c in results_df.columns}
    model_orig_col = None
    for candidate in ("model", "model_name", "name"):
        if candidate in cols_map:
            model_orig_col = cols_map[candidate]
            break

    if not model_orig_col:
        return pd.DataFrame()

    metric_cols = [
        orig for norm, orig in cols_map.items()
        if norm not in ("model", "model_name", "name", "type", "problem_type")
    ]
    if not metric_cols:
        return pd.DataFrame()

    records = []
    for _, row in results_df.iterrows():
        for m in metric_cols:
            val = row.get(m)
            if val is not None and pd.notna(val):
                try:
                    records.append({
                        "model": str(row[model_orig_col]),
                        "metric": str(m).strip().lower(),
                        "value": round(float(val), 4),
                    })
                except (ValueError, TypeError):
                    pass
    return pd.DataFrame(records)


def sync_pipeline_to_grafana(
    session_state: dict[str, Any],
    password: str,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    user: str = DEFAULT_USER,
    database: str = DEFAULT_DB,
) -> dict[str, bool]:
    """
    Main function called from Streamlit UI.
    Syncs all available pipeline stages, computed statistics, and metadata
    to PostgreSQL for Grafana consumption.
    
    Args:
        session_state (dict[str, Any]): Streamlit session state dict
        password (str): PostgreSQL password
        host (str): PostgreSQL host
        port (int): PostgreSQL port
        user (str): PostgreSQL user
        database (str): PostgreSQL database name
        
    Returns:
        dict[str, bool]: Dictionary indicating which tables were successfully synced
    """
    result = {}
    ensure_database(password, host=host, port=port, user=user, database=database)
    
    try:
        engine = get_engine(password, host=host, port=port, user=user, database=database)
    except Exception as e:
        logger.error(f"Failed to get engine for sync: {e}")
        return result

    # ── 1. Direct DataFrame tables ────────────────────────────────────────
    tables_to_sync = {
        "raw_data": "raw_df",
        "cleaned_data": "cleaned_df",
        "processed_data": "processed_df",
        "predictions": "predictions_df",
        "model_results": "results_df",
        "feature_importance": "feature_importance"
    }

    for table_name, state_key in tables_to_sync.items():
        if state_key in session_state and session_state[state_key] is not None:
            df = session_state[state_key]
            if isinstance(df, pd.DataFrame):
                if state_key == "predictions_df":
                    df = _normalize_predictions_df(df)
                saved = save_dataframe(df, table_name, engine)
                result[table_name] = saved > 0
            else:
                try:
                    df = pd.DataFrame(df)
                    if state_key == "predictions_df":
                        df = _normalize_predictions_df(df)
                    saved = save_dataframe(df, table_name, engine)
                    result[table_name] = saved > 0
                except Exception as e:
                    logger.error(f"Could not convert {state_key} to DataFrame: {e}")
                    result[table_name] = False
        else:
            result[table_name] = False

    # ── 2. Computed statistics tables (for advanced Grafana panels) ────────
    # Use cleaned_data if available, otherwise fall back to raw_data
    analysis_df = None
    for key in ("cleaned_df", "raw_df"):
        if key in session_state and isinstance(session_state[key], pd.DataFrame):
            analysis_df = session_state[key]
            break

    if analysis_df is not None and not analysis_df.empty:
        # Column statistics
        try:
            col_stats = _compute_column_statistics(analysis_df)
            if not col_stats.empty:
                saved = save_dataframe(col_stats, "column_statistics", engine)
                result["column_statistics"] = saved > 0
            else:
                result["column_statistics"] = False
        except Exception as e:
            logger.error(f"Error computing column_statistics: {e}")
            result["column_statistics"] = False

        # Correlation matrix
        try:
            corr_df = _compute_correlation_matrix(analysis_df)
            if not corr_df.empty:
                saved = save_dataframe(corr_df, "correlation_matrix", engine)
                result["correlation_matrix"] = saved > 0
            else:
                result["correlation_matrix"] = False
        except Exception as e:
            logger.error(f"Error computing correlation_matrix: {e}")
            result["correlation_matrix"] = False

        # Missing data summary
        try:
            missing_df = _compute_missing_data_summary(analysis_df)
            if not missing_df.empty:
                saved = save_dataframe(missing_df, "missing_data_summary", engine)
                result["missing_data_summary"] = saved > 0
            else:
                result["missing_data_summary"] = False
        except Exception as e:
            logger.error(f"Error computing missing_data_summary: {e}")
            result["missing_data_summary"] = False

        # Data distribution (histogram bins)
        try:
            dist_df = _compute_data_distribution(analysis_df)
            if not dist_df.empty:
                saved = save_dataframe(dist_df, "data_distribution", engine)
                result["data_distribution"] = saved > 0
            else:
                result["data_distribution"] = False
        except Exception as e:
            logger.error(f"Error computing data_distribution: {e}")
            result["data_distribution"] = False

    # Prediction residuals
    pred_key = "predictions_df"
    if pred_key in session_state and isinstance(session_state[pred_key], pd.DataFrame):
        try:
            resid_df = _compute_prediction_residuals(session_state[pred_key])
            if not resid_df.empty:
                saved = save_dataframe(resid_df, "prediction_residuals", engine)
                result["prediction_residuals"] = saved > 0
            else:
                result["prediction_residuals"] = False
        except Exception as e:
            logger.error(f"Error computing prediction_residuals: {e}")
            result["prediction_residuals"] = False

    # Model metrics detail (long-format for grouped bar charts)
    res_key = "results_df"
    if res_key in session_state and isinstance(session_state[res_key], pd.DataFrame):
        try:
            metrics_df = _build_model_metrics_detail(session_state[res_key])
            if not metrics_df.empty:
                saved = save_dataframe(metrics_df, "model_metrics_detail", engine)
                result["model_metrics_detail"] = saved > 0
            else:
                result["model_metrics_detail"] = False
        except Exception as e:
            logger.error(f"Error computing model_metrics_detail: {e}")
            result["model_metrics_detail"] = False

    # ── 3. Pipeline metadata ──────────────────────────────────────────────
    metadata = {}
    if "raw_df" in session_state and isinstance(session_state["raw_df"], pd.DataFrame):
        df = session_state["raw_df"]
        metadata["total_rows"] = len(df)
        metadata["total_columns"] = len(df.columns)
        missing_pct = float((df.isnull().sum().sum() / df.size) * 100) if df.size > 0 else 0.0
        dup_pct = float((df.duplicated().sum() / len(df)) * 100) if len(df) > 0 else 0.0
        metadata["missing_percentage"] = missing_pct
        metadata["duplicate_percentage"] = dup_pct
        metadata["numeric_count"] = len(df.select_dtypes(include="number").columns)
        metadata["categorical_count"] = len(df.select_dtypes(exclude="number").columns)

        # Data quality score: 100 − (missing% + duplicate%)  clamped to [0, 100]
        quality_score = max(0.0, min(100.0, 100.0 - missing_pct - dup_pct))
        metadata["data_quality_score"] = round(quality_score, 2)

        # Pipeline stage flags (1 / 0)
        for stage, key in [("has_raw", "raw_df"), ("has_cleaned", "cleaned_df"),
                           ("has_processed", "processed_df"), ("has_predictions", "predictions_df"),
                           ("has_model_results", "results_df")]:
            metadata[stage] = 1 if (key in session_state and isinstance(session_state.get(key), pd.DataFrame)) else 0
    
    for key in ["target_column", "best_model", "dataset_name"]:
        if key in session_state:
            metadata[key] = session_state[key]
            
    if metadata:
        save_pipeline_metadata(metadata, engine)
        result["pipeline_metadata"] = True
    else:
        result["pipeline_metadata"] = False
        
    return result

def test_connection(password: str, host=DEFAULT_HOST, port=DEFAULT_PORT, user=DEFAULT_USER) -> tuple[bool, str]:
    """
    Tests if PostgreSQL is reachable.
    Supports sslmode=require for cloud PostgreSQL hosts.
    """
    try:
        ssl_param = "?sslmode=require" if host not in ("localhost", "127.0.0.1") else ""
        default_db = "neondb" if "neon" in host.lower() else "postgres"
        url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{default_db}{ssl_param}"
        engine = create_engine(url)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True, "Connection successful"
    except Exception as e:
        logger.error(f"Connection test failed: {e}")
        return False, str(e)
