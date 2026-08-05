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
    Uses pool_pre_ping=True for connection health checks.
    Caches engine in a module-level dict to avoid recreating.
    
    Args:
        password (str): PostgreSQL password
        host (str): Database host
        port (int): Database port
        user (str): Database user
        database (str): Database name
        
    Returns:
        Engine: SQLAlchemy Engine instance
    """
    engine_key = f"{host}:{port}:{user}:{database}"
    if engine_key not in _engines:
        url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}"
        engine = create_engine(url, pool_pre_ping=True)
        _engines[engine_key] = engine
    return _engines[engine_key]

def ensure_database(password: str, host=DEFAULT_HOST, port=DEFAULT_PORT, user=DEFAULT_USER, database=DEFAULT_DB) -> None:
    """
    Connects to the default postgres database and creates the target database if it doesn't exist.
    
    Args:
        password (str): PostgreSQL password
        host (str): Database host
        port (int): Database port
        user (str): Database user
        database (str): Database name to create
    """
    try:
        url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/postgres"
        engine = create_engine(url, isolation_level="AUTOCOMMIT")
        with engine.connect() as conn:
            res = conn.execute(text(f"SELECT 1 FROM pg_database WHERE datname = '{database}'"))
            if not res.scalar():
                conn.execute(text(f"CREATE DATABASE {database}"))
                logger.info(f"Database {database} created.")
            else:
                logger.info(f"Database {database} already exists.")
    except Exception as e:
        logger.error(f"Failed to ensure database {database}: {e}")

def save_dataframe(df: pd.DataFrame, table_name: str, engine: Engine, if_exists: str = 'replace') -> int:
    """
    Writes DataFrame to PostgreSQL using df.to_sql().
    
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
        rows = df.to_sql(table_name, con=engine, if_exists=if_exists, index=False, chunksize=5000)
        logger.info(f"Successfully saved {rows} rows to {table_name}")
        return rows if rows is not None else len(df)
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

def sync_pipeline_to_grafana(session_state: dict[str, Any], password: str) -> dict[str, bool]:
    """
    Main function called from Streamlit UI.
    Syncs all available pipeline stages and metadata to PostgreSQL.
    
    Args:
        session_state (dict[str, Any]): Streamlit session state dict
        password (str): PostgreSQL password
        
    Returns:
        dict[str, bool]: Dictionary indicating which tables were successfully synced
    """
    result = {}
    ensure_database(password)
    
    try:
        engine = get_engine(password)
    except Exception as e:
        logger.error(f"Failed to get engine for sync: {e}")
        return result

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
                saved = save_dataframe(df, table_name, engine)
                result[table_name] = saved > 0
            else:
                try:
                    df = pd.DataFrame(df)
                    saved = save_dataframe(df, table_name, engine)
                    result[table_name] = saved > 0
                except Exception as e:
                    logger.error(f"Could not convert {state_key} to DataFrame: {e}")
                    result[table_name] = False
        else:
            result[table_name] = False

    # Extract metadata
    metadata = {}
    if "raw_df" in session_state and isinstance(session_state["raw_df"], pd.DataFrame):
        df = session_state["raw_df"]
        metadata["total_rows"] = len(df)
        metadata["total_columns"] = len(df.columns)
        metadata["missing_percentage"] = float((df.isnull().sum().sum() / df.size) * 100) if df.size > 0 else 0.0
        metadata["duplicate_percentage"] = float((df.duplicated().sum() / len(df)) * 100) if len(df) > 0 else 0.0
        metadata["numeric_count"] = len(df.select_dtypes(include="number").columns)
        metadata["categorical_count"] = len(df.select_dtypes(exclude="number").columns)
    
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
    
    Args:
        password (str): PostgreSQL password
        host (str): Database host
        port (int): Database port
        user (str): Database user
        
    Returns:
        tuple[bool, str]: Success boolean and message string
    """
    try:
        url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/postgres"
        engine = create_engine(url)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True, "Connection successful"
    except Exception as e:
        logger.error(f"Connection test failed: {e}")
        return False, str(e)
