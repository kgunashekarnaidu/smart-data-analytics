"""
Data Analytics & ML Application — Streamlit entry point.

Provides sidebar navigation, shared session state, and page routing.
Business logic lives in ``core/`` (added in later steps); this module
focuses on layout, navigation, and wiring UI pages together.
"""

from __future__ import annotations

from pathlib import Path

from typing import Any, Callable

import pandas as pd
import streamlit as st

import plotly.express as px

from core.cleaner import CleaningReport, clean_dataframe, detect_outliers_iqr
from core.feature_engineering import (
    FeatureEngineeringReport,
    engineer_features,
    preview_encoding_plan,
)
from core.model_training import TrainingResult, save_training_artifacts, train_all_models
from core.predictor import (
    PredictionConfig,
    build_prediction_config,
    get_input_feature_columns,
    predict_dataframe,
    predict_test_split,
)
from core.preprocessor import PreprocessReport, preprocess_from_feature_result
from core.visualizer import (
    build_eda_figures,
    get_categorical_columns,
    get_numeric_columns,
    plot_bar,
    plot_box,
    plot_correlation_heatmap,
    plot_count,
    plot_histogram,
    plot_line,
    plot_missing_bar,
    plot_missing_heatmap,
    plot_pie,
    plot_scatter,
    plot_violin,
)
from core.loader import (
    LoaderError,
    build_dtype_report,
    load_csv_with_fallback_encoding,
    preview_dataframe,
)
from core.themes import (
    ThemeName,
    apply_plotly_theme,
    build_app_css,
    get_theme,
    home_hero_html,
    home_info_card_html,
    set_active_theme,
    sidebar_brand_html,
    style_bar_trace,
    pipeline_progress_html,
)
from core.utils import (
    ProblemType,
    detect_problem_type,
    get_logger,
    is_likely_id_column,
    setup_logging,
    suggest_target_column,
    to_csv_bytes,
)
from core.data_bridge import sync_pipeline_to_grafana, test_connection as test_pg_connection
import streamlit.components.v1 as components

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
setup_logging()
logger = get_logger("app")

# ---------------------------------------------------------------------------
# Page configuration (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="DataML Pro — Analytics & ML",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
SESSION_DEFAULTS: dict[str, Any] = {
    "raw_df": None,
    "cleaned_df": None,
    "processed_df": None,
    "target_col": None,
    "problem_type": None,
    "feature_cols": [],
    "best_model_name": None,
    "best_model": None,
    "models": {},
    "results_df": None,
    "predictions_df": None,
    "pipeline": None,
    "date_col": None,
    "outlier_strategy": "cap",
    "dataset_name": None,
    "load_result": None,
    "cleaning_report": None,
    "fe_report": None,
    "label_encoders": {},
    "target_encoding_maps": {},
    "onehot_columns": {},
    "preprocess_report": None,
    "preprocessor": None,
    "X_train": None,
    "X_test": None,
    "y_train": None,
    "y_test": None,
    "X_train_raw": None,
    "X_test_raw": None,
    "scaler_type": "standard",
    "test_size": 0.2,
    "training_result": None,
    "feature_importance": None,
    "model_artifact_path": None,
    "input_columns": [],
    "ui_theme": "dark",
    "grafana_synced": False,
    "grafana_sync_result": None,
    "pg_host": "localhost",
    "pg_port": 5432,
    "pg_user": "postgres",
    "pg_password": "admin123",
    "pg_database": "dataml_pro",
}


def init_session_state() -> None:
    """Initialize all session-state keys with safe defaults."""
    import os
    for key, default in SESSION_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = default

    # Overrides for Cloud Deployment (Streamlit Secrets / ENV vars)
    # st.secrets throws if no secrets.toml exists, so wrap in try/except
    def _get_secret(key: str) -> str | None:
        """Safely read a Streamlit secret, returning None if unavailable."""
        try:
            if key in st.secrets:
                return str(st.secrets[key])
        except Exception:
            pass
        return None

    if "PGHOST" in os.environ:
        st.session_state.pg_host = os.environ["PGHOST"]
    else:
        val = _get_secret("PGHOST")
        if val:
            st.session_state.pg_host = val

    if "PGUSER" in os.environ:
        st.session_state.pg_user = os.environ["PGUSER"]
    else:
        val = _get_secret("PGUSER")
        if val:
            st.session_state.pg_user = val

    if "PGPASSWORD" in os.environ:
        st.session_state.pg_password = os.environ["PGPASSWORD"]
    else:
        val = _get_secret("PGPASSWORD")
        if val:
            st.session_state.pg_password = val

    if "PGDATABASE" in os.environ:
        st.session_state.pg_database = os.environ["PGDATABASE"]
    else:
        val = _get_secret("PGDATABASE")
        if val:
            st.session_state.pg_database = val


def reset_pipeline_state() -> None:
    """Clear downstream state after a new upload."""
    keys_to_reset = [
        "cleaned_df",
        "processed_df",
        "target_col",
        "problem_type",
        "feature_cols",
        "best_model_name",
        "best_model",
        "models",
        "results_df",
        "predictions_df",
        "pipeline",
        "date_col",
        "load_result",
        "cleaning_report",
        "fe_report",
        "label_encoders",
        "target_encoding_maps",
        "onehot_columns",
        "preprocess_report",
        "preprocessor",
        "X_train",
        "X_test",
        "y_train",
        "y_test",
        "X_train_raw",
        "X_test_raw",
        "training_result",
        "feature_importance",
        "model_artifact_path",
        "input_columns",
    ]
    for key in keys_to_reset:
        st.session_state[key] = SESSION_DEFAULTS[key]


def active_dataframe() -> pd.DataFrame | None:
    """Return the most processed dataframe available."""
    for key in ("processed_df", "cleaned_df", "raw_df"):
        df = st.session_state.get(key)
        if df is not None:
            return df
    return None


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------
def section_header(icon: str, title: str) -> None:
    """Render a styled section heading."""
    st.markdown(
        f'<div class="section-header">{icon} {title}</div>',
        unsafe_allow_html=True,
    )


def metric_row(metrics: dict[str, str]) -> None:
    """Render a row of Streamlit metrics."""
    columns = st.columns(len(metrics))
    for column, (label, value) in zip(columns, metrics.items()):
        column.metric(label, value)


def require_dataset(message: str = "Please upload a CSV dataset first.") -> pd.DataFrame | None:
    """Show a warning and return None when no dataset is loaded."""
    df = st.session_state.raw_df
    if df is None:
        st.warning(f"📂 {message}")
        return None
    return df


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------
NAV_ITEMS: dict[str, str] = {
    "🏠  Home": "home",
    "📂  Upload Dataset": "upload",
    "🧹  Cleaning": "cleaning",
    "📊  EDA": "eda",
    "📈  Visualizations": "visualizations",
    "📊  Grafana Dashboard": "grafana",
    "⚙️  Preprocessing": "preprocessing",
    "🤖  Machine Learning": "ml",
    "🔮  Predictions": "predictions",
    "⬇️  Download": "download",
}


def current_theme():
    """Return active theme tokens from session state."""
    return get_theme(st.session_state.ui_theme)


def render_theme_styles() -> None:
    """Inject CSS and sync Plotly theme with user preference."""
    theme = set_active_theme(st.session_state.ui_theme)
    st.markdown(build_app_css(theme), unsafe_allow_html=True)


def render_theme_selector() -> None:
    """Let the user switch between dark and light appearance."""
    theme_labels = {"dark": "🌙 Dark", "light": "☀️ Light"}
    options: list[ThemeName] = ["dark", "light"]
    if st.session_state.ui_theme not in options:
        st.session_state.ui_theme = "dark"

    st.radio(
        "Appearance",
        options=options,
        format_func=lambda value: theme_labels[value],
        horizontal=True,
        key="ui_theme",
    )


def get_pipeline_steps() -> list[tuple[str, bool]]:
    """Return ordered pipeline steps and whether each is complete."""
    return [
        ("Upload", st.session_state.raw_df is not None),
        ("Clean", st.session_state.cleaned_df is not None),
        ("Process", st.session_state.processed_df is not None),
        ("Train", st.session_state.best_model_name is not None),
        ("Predict", st.session_state.predictions_df is not None),
    ]


def render_pipeline_progress_bar() -> None:
    """Show a fillable progress bar that tracks pipeline completion."""
    theme = current_theme()
    steps = get_pipeline_steps()
    st.markdown(pipeline_progress_html(theme, steps), unsafe_allow_html=True)


def render_sidebar_status() -> None:
    """Render pipeline status in the sidebar (always expanded when visible)."""
    theme = current_theme()
    with st.sidebar:
        st.markdown(sidebar_brand_html(theme), unsafe_allow_html=True)
        render_theme_selector()
        st.markdown("---")
        st.markdown("**📋 Pipeline Status**")

        if st.session_state.raw_df is not None:
            df = active_dataframe()
            assert df is not None
            st.success(f"✅ {st.session_state.dataset_name or 'uploaded.csv'}")
            st.caption(f"{df.shape[0]:,} rows × {df.shape[1]} columns")
        else:
            st.info("No dataset loaded.")

        status_checks = [
            ("Cleaned", st.session_state.cleaned_df is not None),
            ("Processed", st.session_state.processed_df is not None),
            ("Model trained", st.session_state.best_model_name is not None),
            ("Predictions", st.session_state.predictions_df is not None),
        ]
        for label, done in status_checks:
            st.markdown(f"{'✅' if done else '⬜'} {label}")

        if st.session_state.best_model_name:
            st.markdown("---")
            st.markdown("**🏆 Best Model**")
            st.success(st.session_state.best_model_name)

        st.markdown("---")
        st.caption("Tip: use the **Navigation** menu at the top of the page to switch sections.")


def render_top_navigation() -> str:
    """Primary navigation bar — always visible in the main content area."""
    render_pipeline_progress_bar()

    labels = list(NAV_ITEMS.keys())

    if "current_page_label" not in st.session_state:
        st.session_state.current_page_label = labels[0]

    selected_label = st.selectbox(
        "Navigation",
        labels,
        index=labels.index(st.session_state.current_page_label),
        key="top_nav_select",
        label_visibility="visible",
    )

    st.session_state.current_page_label = selected_label
    return NAV_ITEMS[selected_label]


# ---------------------------------------------------------------------------
# Pages (will move to pages/ module in later steps)
# ---------------------------------------------------------------------------
def page_home() -> None:
    """Landing page with project overview and quick-start guide."""
    theme = current_theme()
    st.markdown(home_hero_html(theme), unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            home_info_card_html(
                theme,
                "🎯 What this app does",
                """<p>Turn your notebook workflow into a professional dashboard.
                Upload <b>any CSV</b>, run automatic cleaning and EDA,
                train multiple ML models, compare results, and download outputs.</p>""",
            ),
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            home_info_card_html(
                theme,
                "🚀 Quick start",
                """<ol>
                    <li>Upload your CSV in <b>Upload Dataset</b></li>
                    <li>Run cleaning & preprocessing</li>
                    <li>Explore charts in EDA / Visualizations</li>
                    <li>Train models & pick the best one</li>
                    <li>Predict & download results</li>
                </ol>""",
            ),
            unsafe_allow_html=True,
        )

    section_header("📋", "Workflow Progress")
    steps = get_pipeline_steps()
    step_labels = {
        "Upload": "Upload CSV",
        "Clean": "Clean data",
        "Process": "Preprocess features",
        "Train": "Train ML model",
        "Predict": "Generate predictions",
    }
    progress_cols = st.columns(len(steps))
    for col, (label, done) in zip(progress_cols, steps):
        col.metric(step_labels.get(label, label), "Done" if done else "Pending")

    st.info("Use the **Navigation** dropdown at the top and choose **Upload Dataset** to begin.")


def page_upload() -> None:
    """CSV upload, validation, and dataset preview."""
    section_header("📂", "Upload Dataset")

    uploaded = st.file_uploader(
        "Drop your CSV file here",
        type=["csv"],
        help="Any CSV dataset. Column names and types are detected automatically.",
        key="upload_dataset",
    )


    if uploaded is not None:
        try:
            with st.spinner("Reading CSV…"):
                result = load_csv_with_fallback_encoding(
                    uploaded,
                    filename=uploaded.name,
                )
                reset_pipeline_state()
                st.session_state.raw_df = result.dataframe
                st.session_state.load_result = result
                st.session_state.dataset_name = result.summary.filename
                st.session_state.date_col = (
                    result.date_columns[0] if result.date_columns else None
                )
                logger.info(
                    "Uploaded dataset: %s shape=%s",
                    result.summary.filename,
                    result.dataframe.shape,
                )
            st.success(
                f"✅ **{result.summary.filename}** loaded — "
                f"{result.summary.row_count:,} rows × {result.summary.column_count} columns"
            )
        except LoaderError as exc:
            st.error(f"❌ {exc}")
            logger.exception("CSV upload failed")
            return

    df = st.session_state.raw_df
    if df is None:
        st.warning("📂 Upload a CSV file to get started.")
        return

    result = st.session_state.load_result
    if result is not None:
        metric_row(result.summary.as_metrics())
    else:
        metric_row(
            {
                "Rows": f"{df.shape[0]:,}",
                "Columns": str(df.shape[1]),
                "Missing Values": str(int(df.isnull().sum().sum())),
                "Duplicate Rows": str(int(df.duplicated().sum())),
                "Memory": f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB",
            }
        )

    if st.session_state.date_col:
        st.info(f"🗓️ Likely date column detected: **{st.session_state.date_col}**")

    st.markdown("---")
    tab_preview, tab_types, tab_missing, tab_stats = st.tabs(
        ["Preview", "Data Types", "Missing Values", "Summary Statistics"]
    )

    with tab_preview:
        st.dataframe(preview_dataframe(df, n=20), use_container_width=True)

    with tab_types:
        dtype_df = result.dtype_report if result is not None else build_dtype_report(df)
        st.dataframe(dtype_df, use_container_width=True)

    with tab_missing:
        missing_df = result.missing_report if result is not None else None
        if missing_df is not None and not missing_df.empty:
            st.dataframe(missing_df, use_container_width=True)
        elif df.isnull().sum().sum() == 0:
            st.success("✅ No missing values detected.")
        else:
            st.dataframe(
                df.isnull().sum().reset_index(name="Missing Count"),
                use_container_width=True,
            )

    with tab_stats:
        st.dataframe(df.describe(include="all").T, use_container_width=True)

def page_cleaning() -> None:
    """Data cleaning page powered by ``core/cleaner.py``."""
    section_header("🧹", "Data Cleaning")
    raw_df = require_dataset()
    if raw_df is None:
        return

    st.markdown(
        """
        Automatic cleaning will:
        - Normalize column names and trim whitespace
        - Remove empty columns and duplicate rows
        - Parse dates and add calendar features (year, month, quarter, week, day, weekend)
        - Fill missing values (median for numeric, mode for categorical)
        - Detect and treat outliers using the IQR rule
        """
    )

    strategy = st.selectbox(
        "Outlier treatment strategy",
        ["cap", "remove", "skip"],
        format_func=lambda x: {
            "cap": "Cap outliers (IQR bounds)",
            "remove": "Remove outlier rows",
            "skip": "Skip outlier treatment",
        }[x],
        index=["cap", "remove", "skip"].index(st.session_state.outlier_strategy),
    )
    st.session_state.outlier_strategy = strategy

    # Preview outliers on raw data before cleaning
    with st.expander("🔍 Preview outliers on raw data (IQR)", expanded=False):
        outlier_preview = detect_outliers_iqr(raw_df)
        if outlier_preview:
            preview_df = pd.DataFrame(
                [
                    {
                        "Column": o.column,
                        "Outlier Rows": o.outlier_count,
                        "Lower Bound": round(o.lower_bound, 2),
                        "Upper Bound": round(o.upper_bound, 2),
                    }
                    for o in outlier_preview
                ]
            )
            st.dataframe(preview_df, use_container_width=True)
        else:
            st.success("No IQR outliers detected in numeric columns.")

    if st.button("🚀 Run Automatic Cleaning", use_container_width=True):
        try:
            with st.spinner("Cleaning dataset…"):
                date_cols: list[str] | None = None
                if st.session_state.date_col:
                    date_cols = [st.session_state.date_col]
                elif st.session_state.load_result and st.session_state.load_result.date_columns:
                    date_cols = list(st.session_state.load_result.date_columns)

                cleaned, report = clean_dataframe(
                    raw_df,
                    outlier_strategy=strategy,  # type: ignore[arg-type]
                    date_columns=date_cols,
                )

                # Reset downstream pipeline but keep new cleaned data
                reset_pipeline_state()
                st.session_state.cleaned_df = cleaned
                st.session_state.cleaning_report = report
                if report.date_columns_parsed:
                    st.session_state.date_col = report.date_columns_parsed[0]

                logger.info("Cleaning finished: %s", report.actions)
            st.success("✅ Data cleaning complete!")
            st.rerun()
        except Exception as exc:
            st.error(f"❌ Cleaning failed: {exc}")
            logger.exception("Cleaning failed")

    report: CleaningReport | None = st.session_state.cleaning_report
    cleaned_df = st.session_state.cleaned_df

    if cleaned_df is not None and report is not None:
        st.markdown("---")
        st.markdown("### 📋 Cleaning Summary")
        metric_row(report.as_metrics())

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Before**")
            st.caption(f"{report.rows_before:,} rows × {report.columns_before} columns")
            st.caption(f"Missing values: {report.missing_values_before:,}")
        with col2:
            st.markdown("**After**")
            st.caption(f"{report.rows_after:,} rows × {report.columns_after} columns")
            st.caption(f"Missing values: {report.missing_values_after:,}")

        with st.expander("📝 Actions performed", expanded=True):
            for action in report.actions:
                st.markdown(f"- {action}")

        if report.outlier_summaries:
            st.markdown("### 🎯 Outlier treatment")
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Column": o.column,
                            "Outliers Found": o.outlier_count,
                            "Lower (IQR)": round(o.lower_bound, 2),
                            "Upper (IQR)": round(o.upper_bound, 2),
                        }
                        for o in report.outlier_summaries
                    ]
                ),
                use_container_width=True,
            )

        st.markdown("### 🧽 Cleaned dataset preview")
        st.dataframe(cleaned_df.head(20), use_container_width=True)

        st.download_button(
            label="⬇️ Download Cleaned Dataset",
            data=to_csv_bytes(cleaned_df),
            file_name="cleaned_dataset.csv",
            mime="text/csv",
            use_container_width=True,
        )


def page_eda() -> None:
    """Exploratory data analysis with interactive Plotly charts."""
    section_header("📊", "Exploratory Data Analysis")
    df = active_dataframe()
    if df is None:
        require_dataset("Upload and optionally clean a dataset first.")
        return

    num_cols = get_numeric_columns(df)
    cat_cols = get_categorical_columns(df)

    tab_summary, tab_missing, tab_corr, tab_auto = st.tabs(
        ["Summary", "Missing Values", "Correlation", "Auto Dashboard"]
    )

    with tab_summary:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Numeric summary")
            if num_cols:
                st.dataframe(df[num_cols].describe().T, use_container_width=True)
            else:
                st.info("No numeric columns found.")
        with col2:
            st.subheader("Categorical overview")
            if cat_cols:
                selected = st.selectbox("Column", cat_cols, key="eda_cat_col")
                st.dataframe(
                    df[selected].value_counts().head(15).reset_index(),
                    use_container_width=True,
                )
            else:
                st.info("No categorical columns found.")

    with tab_missing:
        st.plotly_chart(plot_missing_bar(df), width="stretch")
        if df.isnull().sum().sum() > 0:
            st.plotly_chart(plot_missing_heatmap(df), width="stretch")
        else:
            st.success("✅ No missing values in the dataset.")

    with tab_corr:
        if len(num_cols) >= 2:
            max_cols = min(30, len(num_cols))
            default_cols = min(15, len(num_cols))
            if max_cols <= 5:
                # Too few columns for a slider — just use all of them
                top_n = max_cols
            else:
                top_n = st.slider("Max columns in heatmap", 2, max_cols, default_cols)
            st.plotly_chart(plot_correlation_heatmap(df, num_cols[:top_n]), width="stretch")
        else:
            st.info("Need at least 2 numeric columns for a correlation heatmap.")

    with tab_auto:
        st.caption("Automatically generated charts based on your dataset structure.")
        eda_figures = build_eda_figures(df)
        for name, figure in eda_figures.items():
            st.plotly_chart(figure, width="stretch", key=f"eda_auto_{name}")


def page_visualizations() -> None:
    """Interactive Plotly visualization studio."""
    section_header("📈", "Visualizations")
    df = active_dataframe()
    if df is None:
        require_dataset("Upload a dataset to generate charts.")
        return

    num_cols = get_numeric_columns(df)
    cat_cols = get_categorical_columns(df)
    all_cols = df.columns.tolist()

    t_hist, t_box, t_scatter, t_line, t_bar, t_pie, t_violin, t_count = st.tabs(
        ["Histogram", "Box Plot", "Scatter", "Line", "Bar", "Pie", "Violin", "Count"]
    )

    with t_hist:
        if not num_cols:
            st.warning("No numeric columns available.")
        else:
            col = st.selectbox("Column", num_cols, key="viz_hist_col")
            bins = st.slider("Bins", 10, 80, 30, key="viz_hist_bins")
            st.plotly_chart(plot_histogram(df, col, bins=bins), width="stretch")

    with t_box:
        if not num_cols:
            st.warning("No numeric columns available.")
        else:
            col = st.selectbox("Numeric column", num_cols, key="viz_box_col")
            group = st.selectbox(
                "Group by (optional)",
                ["None"] + cat_cols,
                key="viz_box_group",
            )
            group_col = None if group == "None" else group
            st.plotly_chart(plot_box(df, col, group_col), width="stretch")

    with t_scatter:
        if len(num_cols) < 2:
            st.warning("Need at least 2 numeric columns.")
        else:
            x_col = st.selectbox("X axis", num_cols, key="viz_sc_x")
            y_col = st.selectbox("Y axis", num_cols[::-1], key="viz_sc_y")
            color = st.selectbox("Color (optional)", ["None"] + cat_cols, key="viz_sc_color")
            trend = st.checkbox("Show trendline", value=False, key="viz_sc_trend")
            color_col = None if color == "None" else color
            st.plotly_chart(
                plot_scatter(df, x_col, y_col, color_col, show_trendline=trend),
                width="stretch",
            )

    with t_line:
        x_default = "month" if "month" in all_cols else ("year" if "year" in all_cols else all_cols[0])
        x_col = st.selectbox("X axis", all_cols, index=all_cols.index(x_default) if x_default in all_cols else 0, key="viz_line_x")
        y_col = st.selectbox("Y axis", num_cols if num_cols else all_cols, key="viz_line_y")
        group = st.selectbox("Group by (optional)", ["None"] + cat_cols, key="viz_line_group")
        group_col = None if group == "None" else group
        st.plotly_chart(plot_line(df, x_col, y_col, group_col), width="stretch")

    with t_bar:
        if not cat_cols or not num_cols:
            st.warning("Need at least one categorical and one numeric column.")
        else:
            x_col = st.selectbox("Category (X)", cat_cols, key="viz_bar_x")
            y_col = st.selectbox("Value (Y)", num_cols, key="viz_bar_y")
            agg = st.selectbox("Aggregation", ["mean", "sum", "count", "max", "min"], key="viz_bar_agg")
            st.plotly_chart(plot_bar(df, x_col, y_col, agg), width="stretch")

    with t_pie:
        if not cat_cols:
            st.warning("No categorical columns available.")
        else:
            cat_col = st.selectbox("Category", cat_cols, key="viz_pie_cat")
            val = st.selectbox("Value column (optional)", ["Count"] + num_cols, key="viz_pie_val")
            val_col = None if val == "Count" else val
            st.plotly_chart(plot_pie(df, cat_col, val_col), width="stretch")

    with t_violin:
        if not num_cols:
            st.warning("No numeric columns available.")
        else:
            col = st.selectbox("Numeric column", num_cols, key="viz_violin_col")
            group = st.selectbox("Group by (optional)", ["None"] + cat_cols, key="viz_violin_group")
            group_col = None if group == "None" else group
            st.plotly_chart(plot_violin(df, col, group_col), width="stretch")

    with t_count:
        if not cat_cols:
            st.warning("No categorical columns available.")
        else:
            col = st.selectbox("Categorical column", cat_cols, key="viz_count_col")
            st.plotly_chart(plot_count(df, col), width="stretch")


def page_preprocessing() -> None:
    """Feature engineering and encoding via ``core/feature_engineering.py``."""
    section_header("⚙️", "Preprocessing & Feature Engineering")
    df = st.session_state.cleaned_df
    if df is None:
        st.warning("🧹 Please run data cleaning first.")
        return

    suggested = suggest_target_column(df)
    column_options = df.columns.tolist()
    default_index = column_options.index(suggested) if suggested in column_options else 0

    target = st.selectbox(
        "Select target column (what to predict)",
        column_options,
        index=default_index,
        help="Auto-suggested based on column names and data types.",
    )
    st.session_state.target_col = target

    try:
        problem = detect_problem_type(df[target])
        st.session_state.problem_type = problem.value
        label = "Classification" if problem == ProblemType.CLASSIFICATION else "Regression"
        st.info(f"Detected problem type: **{label}**")
    except ValueError as exc:
        st.warning(str(exc))

    if is_likely_id_column(df[target]):
        st.error(
            f"**`{target}` looks like an ID column** (too many unique values). "
            "IDs are not good prediction targets. Choose something meaningful instead — "
            "e.g. price, rating, sales, category, or sold quantity."
        )

    use_target_encoding = st.checkbox(
        "Enable target encoding for medium-cardinality columns (11–50 unique values)",
        value=True,
    )

    col_scale, col_split = st.columns(2)
    with col_scale:
        scaler_options = {
            "standard": "StandardScaler (default for regression)",
            "minmax": "MinMaxScaler (bounded 0–1)",
            "robust": "RobustScaler (handles outliers)",
            "none": "None (best for tree-based models)",
        }
        default_scaler = st.session_state.scaler_type
        scaler_type = st.selectbox(
            "Feature scaling",
            list(scaler_options.keys()),
            index=list(scaler_options.keys()).index(default_scaler),
            format_func=lambda x: scaler_options[x],
        )
        st.session_state.scaler_type = scaler_type

    with col_split:
        test_size = st.slider(
            "Test size (%)",
            min_value=10,
            max_value=40,
            value=int(st.session_state.test_size * 100),
        ) / 100
        st.session_state.test_size = test_size

    with st.expander("📋 Preview encoding plan", expanded=False):
        plan_df = preview_encoding_plan(df, target)
        if plan_df.empty:
            st.info("No categorical columns require encoding.")
        else:
            st.dataframe(plan_df, use_container_width=True)

    if st.button("⚙️ Run Full Preprocessing Pipeline", use_container_width=True):
        if is_likely_id_column(df[target]):
            st.error("Please select a valid target column before preprocessing.")
            return
        try:
            with st.spinner("Engineering features and preparing train/test split…"):
                fe_result = engineer_features(
                    df,
                    target,
                    use_target_encoding=use_target_encoding,
                )
                preprocess_result = preprocess_from_feature_result(
                    fe_result,
                    scaler_type=scaler_type,  # type: ignore[arg-type]
                    test_size=test_size,
                )

                st.session_state.processed_df = fe_result.processed_dataframe
                st.session_state.feature_cols = fe_result.feature_names
                st.session_state.fe_report = fe_result.report
                st.session_state.label_encoders = fe_result.label_encoders
                st.session_state.target_encoding_maps = fe_result.target_encoding_maps
                st.session_state.onehot_columns = fe_result.onehot_columns
                st.session_state.input_columns = get_input_feature_columns(
                    df, target, fe_result.report
                )

                st.session_state.preprocess_report = preprocess_result.report
                st.session_state.preprocessor = preprocess_result.preprocessor
                st.session_state.X_train = preprocess_result.X_train
                st.session_state.X_test = preprocess_result.X_test
                st.session_state.y_train = preprocess_result.y_train
                st.session_state.y_test = preprocess_result.y_test
                st.session_state.X_train_raw = preprocess_result.X_train_raw
                st.session_state.X_test_raw = preprocess_result.X_test_raw

                logger.info(
                    "Preprocessing pipeline: %s features, train=%s test=%s",
                    len(fe_result.feature_names),
                    len(preprocess_result.y_train),
                    len(preprocess_result.y_test),
                )
            st.success("✅ Preprocessing pipeline complete!")
            st.rerun()
        except Exception as exc:
            st.error(f"❌ Preprocessing failed: {exc}")
            logger.exception("Preprocessing failed")

    fe_report: FeatureEngineeringReport | None = st.session_state.fe_report
    preprocess_report: PreprocessReport | None = st.session_state.preprocess_report

    if st.session_state.processed_df is not None and fe_report is not None:
        st.markdown("---")
        st.markdown("### 📊 Feature Engineering Summary")
        metric_row(
            {
                "Raw Features": str(fe_report.raw_feature_count),
                "Final Features": str(fe_report.final_feature_count),
                "Encoded Columns": str(len(fe_report.encoding_details)),
                "IDs Excluded": str(len(fe_report.id_columns_excluded)),
                "Calendar Features": str(len(fe_report.datetime_features_added)),
            }
        )

        with st.expander("📝 Actions performed", expanded=True):
            for action in fe_report.actions:
                st.markdown(f"- {action}")

        if fe_report.encoding_details:
            st.markdown("**Encoding details**")
            st.dataframe(fe_report.encoding_summary_df(), use_container_width=True)

        if fe_report.id_columns_excluded:
            st.caption(
                "Excluded ID-like columns: "
                + ", ".join(f"`{c}`" for c in fe_report.id_columns_excluded)
            )

        st.success(f"✅ Target: **{st.session_state.target_col}**")
        st.dataframe(st.session_state.processed_df.head(10), use_container_width=True)

    if preprocess_report is not None:
        st.markdown("---")
        st.markdown("### 🔧 Scaling & Train/Test Split")
        metric_row(
            {
                "Train Rows": f"{preprocess_report.train_rows:,}",
                "Test Rows": f"{preprocess_report.test_rows:,}",
                "Features": str(preprocess_report.feature_count),
                "Scaler": preprocess_report.scaler_type,
                "Problem": preprocess_report.problem_type.title(),
            }
        )
        with st.expander("📝 Preprocessing actions", expanded=True):
            for action in preprocess_report.actions:
                st.markdown(f"- {action}")
        st.caption(
            "Tree models (Random Forest, XGBoost) use raw features. "
            "Linear models use scaled features when scaling is enabled."
        )


def page_ml() -> None:
    """Model training, comparison, and leaderboard."""
    section_header("🤖", "Machine Learning")

    if st.session_state.X_train is None or st.session_state.y_train is None:
        st.warning("⚙️ Please complete the full preprocessing pipeline first.")
        return

    target = st.session_state.target_col
    problem_type = ProblemType(st.session_state.problem_type or "regression")

    st.markdown(
        f"**Target column:** `{target}`  \n"
        f"**Problem type:** `{problem_type.value.title()}`  \n"
        f"**Train rows:** {len(st.session_state.y_train):,}  \n"
        f"**Test rows:** {len(st.session_state.y_test):,}  \n"
        f"**Features:** {len(st.session_state.feature_cols)}"
    )

    if target and st.session_state.processed_df is not None:
        if is_likely_id_column(st.session_state.processed_df[target]):
            st.error(
                f"`{target}` looks like an ID column — training will not work well. "
                "Go back to **Preprocessing** and pick a real target "
                "(price, rating, sales, etc.)."
            )

    # ── Training options ────────────────────────────────────────────────────
    cv_folds = 3  # default; overridden by the slider below when expanded
    with st.expander("⚙️ Training options", expanded=False):
        cv_folds = st.slider(
            "Cross-validation folds",
            min_value=2,
            max_value=5,
            value=3,
            help=(
                "More folds = more reliable CV scores but slower training. "
                "3 folds is a good balance of speed and accuracy."
            ),
        )
        st.caption(
            "ℹ️ Models trained: Linear/Logistic Regression, Decision Tree, "
            "Random Forest, Gradient Boosting (fast: 50 trees + early stopping), XGBoost"
        )

    if st.button("🚀 Train All Models", use_container_width=True):
        try:
            total_models = 5
            progress_bar = st.progress(0.0)
            status_box = st.empty()

            def _on_model_progress(idx: int, total: int, model_name: str) -> None:
                frac = idx / total
                progress_bar.progress(frac)
                status_box.info(
                    f"⏳ Training model {idx + 1}/{total}: **{model_name}**…"
                )

            status_box.info("⏳ Starting model training…")

            training_result = train_all_models(
                st.session_state.X_train,
                st.session_state.X_test,
                st.session_state.y_train,
                st.session_state.y_test,
                st.session_state.X_train_raw,
                st.session_state.X_test_raw,
                problem_type=problem_type,
                feature_names=st.session_state.feature_cols,
                cv_folds=cv_folds,
                progress_callback=_on_model_progress,
            )

            artifact_path = save_training_artifacts(
                training_result,
                preprocessor=st.session_state.preprocessor,
                label_encoders=st.session_state.label_encoders,
                target_encoding_maps=st.session_state.target_encoding_maps,
                target_column=target,
                onehot_columns=st.session_state.onehot_columns,
                fe_report=st.session_state.fe_report,
                input_columns=st.session_state.input_columns,
            )

            st.session_state.training_result = training_result
            st.session_state.results_df = training_result.results_df
            st.session_state.models = training_result.models
            st.session_state.best_model_name = training_result.best_model_name
            st.session_state.best_model = training_result.best_model
            st.session_state.pipeline = training_result.best_model
            st.session_state.feature_importance = training_result.feature_importance
            st.session_state.model_artifact_path = artifact_path

            progress_bar.progress(1.0)
            status_box.success(f"✅ Best model: **{training_result.best_model_name}**")
            st.rerun()
        except Exception as exc:
            st.error(f"❌ Training failed: {exc}")
            logger.exception("Model training failed")

    if st.session_state.results_df is not None:
        results_df = st.session_state.results_df
        best = results_df.iloc[0]

        st.markdown("---")
        st.markdown("### 🏆 Model Leaderboard")
        primary_metric = "R2" if problem_type == ProblemType.REGRESSION else "F1"
        st.markdown(
            f"""
            <div class='success-banner'>
                Best Model: {st.session_state.best_model_name}
                &nbsp;|&nbsp; {primary_metric} = {best.get(primary_metric, 0):.4f}
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.dataframe(
            results_df.style.highlight_max(
                subset=[c for c in results_df.columns if c != "Model"],
                color=current_theme().table_highlight,
            ),
            use_container_width=True,
        )

        chart_metric = primary_metric
        if chart_metric in results_df.columns:
            theme = current_theme()
            fig = px.bar(
                results_df,
                x="Model",
                y=chart_metric,
                color="Model",
                text=chart_metric,
                title=f"Model Comparison — {chart_metric}",
                color_discrete_sequence=list(theme.chart_palette),
            )
            fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
            fig.update_layout(showlegend=False)
            style_bar_trace(fig, opacity=0.92)
            apply_plotly_theme(fig)
            st.plotly_chart(fig, width="stretch")

        # Actual vs predicted for best model
        training_result: TrainingResult | None = st.session_state.training_result
        if training_result is not None:
            from core.preprocessor import model_requires_scaling

            use_scaled = model_requires_scaling(training_result.best_model_name)
            X_eval = st.session_state.X_test if use_scaled else st.session_state.X_test_raw
            y_true = st.session_state.y_test
            y_pred = training_result.best_model.predict(X_eval)

            if training_result.target_encoder is not None:
                y_pred = training_result.target_encoder.inverse_transform(
                    y_pred.astype(int)
                )

            n_points = min(150, len(y_true))
            comparison = pd.DataFrame({"Actual": y_true[:n_points], "Predicted": y_pred[:n_points]})
            st.markdown("### 📈 Actual vs Predicted (best model)")
            st.line_chart(comparison)

        importance = st.session_state.feature_importance
        if importance is not None and not importance.empty:
            st.markdown("### 🎯 Feature Importance (best model)")
            theme = current_theme()
            top_importance = importance.head(15)
            fig_imp = px.bar(
                top_importance,
                x="Importance",
                y="Feature",
                orientation="h",
                title="Top 15 Features",
                color_discrete_sequence=[theme.chart_primary],
            )
            style_bar_trace(fig_imp)
            fig_imp.update_layout(yaxis={"categoryorder": "total ascending"}, showlegend=False)
            apply_plotly_theme(fig_imp)
            st.plotly_chart(fig_imp, width="stretch")

        if st.session_state.model_artifact_path:
            st.caption(f"Model saved to `{st.session_state.model_artifact_path}`")


def _get_prediction_config() -> PredictionConfig | None:
    """Build prediction config from current session state."""
    if st.session_state.best_model is None:
        return None

    training_result: TrainingResult | None = st.session_state.training_result
    target_encoder = training_result.target_encoder if training_result else None

    return build_prediction_config(
        model=st.session_state.best_model,
        model_name=st.session_state.best_model_name or "model",
        problem_type=st.session_state.problem_type or "regression",
        target_column=st.session_state.target_col or "target",
        feature_names=st.session_state.feature_cols,
        input_columns=st.session_state.input_columns,
        label_encoders=st.session_state.label_encoders,
        target_encoding_maps=st.session_state.target_encoding_maps,
        onehot_columns=st.session_state.onehot_columns,
        fe_report=st.session_state.fe_report,
        preprocessor=st.session_state.preprocessor,
        target_encoder=target_encoder,
    )


def page_predictions() -> None:
    """Single-row, bulk, and test-set predictions."""
    section_header("🔮", "Predictions")
    if st.session_state.best_model is None:
        st.warning("🤖 Train a model first on the Machine Learning page.")
        return

    config = _get_prediction_config()
    if config is None:
        st.warning("Prediction configuration is incomplete. Re-run preprocessing and training.")
        return

    target = st.session_state.target_col
    model_name = st.session_state.best_model_name
    st.markdown(
        f"**Model:** `{model_name}`  \n"
        f"**Target:** `{target}`  \n"
        f"**Input fields:** {len(config.input_columns)} raw columns → "
        f"{len(config.feature_names)} encoded features"
    )

    tab_manual, tab_bulk, tab_test = st.tabs(
        ["Manual input", "Upload CSV", "Test set"]
    )

    with tab_manual:
        cleaned = st.session_state.cleaned_df
        if cleaned is None:
            st.warning("Cleaned dataset not available. Re-upload and clean your data.")
        else:
            st.markdown("Enter values for each feature column, then submit to predict.")

            with st.form("manual_prediction_form"):
                values: dict[str, Any] = {}
                columns = st.columns(2)
                for idx, column in enumerate(config.input_columns):
                    with columns[idx % 2]:
                        series = cleaned[column]
                        if pd.api.types.is_datetime64_any_dtype(series):
                            default = series.dropna().iloc[0].date() if series.notna().any() else None
                            values[column] = st.date_input(column, value=default)
                        elif pd.api.types.is_numeric_dtype(series):
                            default = float(series.median()) if series.notna().any() else 0.0
                            values[column] = st.number_input(column, value=default)
                        else:
                            options = series.dropna().astype(str).unique().tolist()
                            default = options[0] if options else ""
                            values[column] = st.selectbox(column, options=options or [""], index=0)

                submitted = st.form_submit_button("Predict", use_container_width=True)

            if submitted:
                try:
                    from datetime import date

                    row_values = {}
                    for column, value in values.items():
                        if isinstance(value, date):
                            row_values[column] = pd.Timestamp(value)
                        else:
                            row_values[column] = value
                    row_df = pd.DataFrame([row_values])
                    result = predict_dataframe(row_df, config)
                    prediction = result.predictions[0]
                    st.session_state.predictions_df = result.predictions_df

                    st.success(f"**Predicted {target}:** `{prediction}`")
                    st.dataframe(result.predictions_df, use_container_width=True)
                except Exception as exc:
                    st.error(f"Prediction failed: {exc}")
                    logger.exception("Manual prediction failed")

    with tab_bulk:
        st.markdown(
            "Upload a CSV with the same columns as your cleaned dataset "
            f"(target `{target}` optional)."
        )
        bulk = st.file_uploader("Upload CSV for bulk prediction", type=["csv"], key="bulk_predict_csv")

        if bulk is not None:
            try:
                from core.loader import load_csv_with_fallback_encoding

                upload_result = load_csv_with_fallback_encoding(bulk)
                bulk_df = upload_result.dataframe
                st.caption(f"Loaded {len(bulk_df):,} rows × {bulk_df.shape[1]} columns")

                actual_col = target if target in bulk_df.columns else None
                if st.button("Run bulk prediction", use_container_width=True, key="bulk_predict_btn"):
                    with st.spinner("Generating predictions…"):
                        result = predict_dataframe(
                            bulk_df,
                            config,
                            actual_column=actual_col,
                        )
                        st.session_state.predictions_df = result.predictions_df

                    pred_col = f"predicted_{target}"
                    st.success(f"Generated {len(result.predictions):,} predictions.")
                    st.dataframe(result.predictions_df.head(20), use_container_width=True)

                    if actual_col:
                        from sklearn.metrics import mean_absolute_error, r2_score

                        actual = pd.to_numeric(result.predictions_df[f"actual_{target}"], errors="coerce")
                        predicted = pd.to_numeric(result.predictions_df[pred_col], errors="coerce")
                        valid = actual.notna() & predicted.notna()
                        if valid.any() and st.session_state.problem_type == "regression":
                            st.info(
                                f"Bulk evaluation — MAE: {mean_absolute_error(actual[valid], predicted[valid]):.4f}, "
                                f"R²: {r2_score(actual[valid], predicted[valid]):.4f}"
                            )
            except Exception as exc:
                st.error(f"Could not load CSV: {exc}")

    with tab_test:
        st.markdown("Score the hold-out test split from preprocessing.")
        if st.session_state.X_test is None or st.session_state.y_test is None:
            st.warning("Test split not available. Re-run preprocessing.")
        elif st.button("Predict on test set", use_container_width=True, key="test_predict_btn"):
            try:
                result = predict_test_split(
                    config=config,
                    X_test=st.session_state.X_test,
                    X_test_raw=st.session_state.X_test_raw,
                    y_test=st.session_state.y_test,
                )
                st.session_state.predictions_df = result.predictions_df

                st.success(f"Generated {len(result.predictions):,} test predictions.")
                st.dataframe(result.predictions_df.head(20), use_container_width=True)

                if st.session_state.problem_type == "regression":
                    from sklearn.metrics import mean_absolute_error, r2_score

                    actual = pd.to_numeric(result.predictions_df[f"actual_{target}"], errors="coerce")
                    predicted = pd.to_numeric(result.predictions_df[f"predicted_{target}"], errors="coerce")
                    st.info(
                        f"Test set — MAE: {mean_absolute_error(actual, predicted):.4f}, "
                        f"R²: {r2_score(actual, predicted):.4f}"
                    )
            except Exception as exc:
                st.error(f"Test prediction failed: {exc}")
                logger.exception("Test set prediction failed")

    if st.session_state.predictions_df is not None:
        st.markdown("---")
        st.markdown("### Latest predictions")
        st.dataframe(st.session_state.predictions_df.head(50), use_container_width=True)
        st.caption("Download the full results from the **Download Results** page.")


def render_native_grafana_dashboard() -> None:
    """Render a native Plotly/Streamlit interactive analytics dashboard matching Grafana panels."""
    theme = current_theme()
    df = active_dataframe()

    if df is None:
        st.info("📂 Please upload a CSV dataset on the **Upload Dataset** page to unlock live analytics charts.")
        return

    st.markdown("### 🏠 Executive Summary & Pipeline Metrics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Rows", f"{len(df):,}")
    with col2:
        st.metric("Total Columns", f"{len(df.columns)}")
    with col3:
        missing_count = int(df.isnull().sum().sum())
        st.metric("Missing Values", f"{missing_count:,}")
    with col4:
        best_model = st.session_state.get("best_model_name") or "Not Trained"
        st.metric("Best Model", best_model)

    st.markdown("---")

    results_df = st.session_state.get("results_df")
    if results_df is not None and not results_df.empty:
        st.markdown("### 🤖 Model Analytics & Leaderboard")
        col_m1, col_m2 = st.columns([1, 1])
        with col_m1:
            problem_type = st.session_state.get("problem_type") or "regression"
            primary_metric = "R2" if problem_type == "regression" else "F1"
            if primary_metric in results_df.columns:
                fig = px.bar(
                    results_df,
                    x="Model",
                    y=primary_metric,
                    color="Model",
                    text=primary_metric,
                    title=f"Model Performance ({primary_metric})",
                    color_discrete_sequence=list(theme.chart_palette),
                )
                fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
                fig.update_layout(showlegend=False)
                style_bar_trace(fig, opacity=0.92)
                apply_plotly_theme(fig)
                st.plotly_chart(fig, use_container_width=True)
        with col_m2:
            st.markdown("**🏆 Leaderboard Data Table**")
            st.dataframe(results_df, use_container_width=True)
        st.markdown("---")

    st.markdown("### 📊 Feature & Data Analysis")
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        importance = st.session_state.get("feature_importance")
        if importance is not None and not importance.empty:
            st.markdown("**🎯 Top 15 Feature Importances**")
            top_imp = importance.head(15)
            fig_imp = px.bar(
                top_imp,
                x="Importance",
                y="Feature",
                orientation="h",
                color_discrete_sequence=[theme.chart_primary],
            )
            style_bar_trace(fig_imp)
            fig_imp.update_layout(yaxis={"categoryorder": "total ascending"}, showlegend=False)
            apply_plotly_theme(fig_imp)
            st.plotly_chart(fig_imp, use_container_width=True)
        else:
            st.info("💡 Train models on the **Machine Learning** page to view feature importances.")
    with col_f2:
        num_cols = get_numeric_columns(df)
        if len(num_cols) >= 2:
            st.markdown("**🔥 Correlation Heatmap**")
            max_c = min(12, len(num_cols))
            fig_corr = plot_correlation_heatmap(df, num_cols[:max_c])
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.info("💡 Need at least 2 numeric columns for a correlation heatmap.")

    training_result = st.session_state.get("training_result")
    if training_result is not None and st.session_state.get("X_test") is not None:
        st.markdown("---")
        st.markdown("### 📈 Predictions & Error Analysis")
        from core.preprocessor import model_requires_scaling
        use_scaled = model_requires_scaling(training_result.best_model_name)
        X_eval = st.session_state.X_test if use_scaled else st.session_state.X_test_raw
        y_true = st.session_state.y_test
        y_pred = training_result.best_model.predict(X_eval)
        if training_result.target_encoder is not None:
            y_pred = training_result.target_encoder.inverse_transform(y_pred.astype(int))

        n_pts = min(150, len(y_true))
        comp_df = pd.DataFrame({"Actual": y_true[:n_pts], "Predicted": y_pred[:n_pts]})
        fig_pred = px.line(comp_df, title="Actual vs Predicted (Test Sample)")
        apply_plotly_theme(fig_pred)
        st.plotly_chart(fig_pred, use_container_width=True)


def page_grafana() -> None:
    """Grafana dashboard integration with live PostgreSQL sync and native fallback."""
    section_header("📊", "Grafana & Live Analytics Dashboard")

    theme = current_theme()

    # --- Connection & Sync Controls ---
    st.markdown(
        """
        Explore live analytics using our **built-in interactive dashboard** or sync your data to **Grafana** via PostgreSQL.
        """
    )

    col_status, col_actions = st.columns([1, 1])

    with col_status:
        st.markdown("### 🔌 PostgreSQL Connection")
        
        with st.expander("⚙️ Connection Settings", expanded=False):
            cfg_col1, cfg_col2 = st.columns(2)
            with cfg_col1:
                host_input = st.text_input("Host", value=st.session_state.pg_host, key="pg_host_input")
                user_input = st.text_input("User", value=st.session_state.pg_user, key="pg_user_input")
            with cfg_col2:
                port_input = st.number_input("Port", value=int(st.session_state.pg_port), min_value=1, max_value=65535, key="pg_port_input")
                password_input = st.text_input("Password", value=st.session_state.pg_password, type="password", key="pg_password_input")
            
            st.session_state.pg_host = host_input
            st.session_state.pg_port = port_input
            st.session_state.pg_user = user_input
            st.session_state.pg_password = password_input

        host = st.session_state.pg_host
        port = int(st.session_state.pg_port)
        user = st.session_state.pg_user
        password = st.session_state.pg_password

        success, msg = test_pg_connection(password, host=host, port=port, user=user)
        if success:
            st.success(f"✅ {msg} ({user}@{host}:{port})")
        else:
            st.info(
                f"ℹ️ PostgreSQL Status: {msg} ({user}@{host}:{port})\n\n"
                "*(PostgreSQL sync is optional. The **✨ Live Analytics (Built-in)** tab below works everywhere!)*"
            )

    with col_actions:
        st.markdown("### 🔄 Data Sync")
        df = st.session_state.raw_df
        if df is None:
            st.info("📂 Upload a dataset first to enable PostgreSQL sync.")
        else:
            st.caption(
                f"Dataset: **{st.session_state.dataset_name or 'uploaded.csv'}** "
                f"({df.shape[0]:,} rows × {df.shape[1]} cols)"
            )
            if st.button("🚀 Sync Data to Grafana / PostgreSQL", use_container_width=True, key="grafana_sync_btn"):
                try:
                    with st.spinner("Syncing pipeline data to PostgreSQL…"):
                        result = sync_pipeline_to_grafana(
                            dict(st.session_state),
                            password=password,
                            host=host,
                            port=port,
                            user=user,
                        )
                        st.session_state.grafana_synced = True
                        st.session_state.grafana_sync_result = result

                    synced_tables = [t for t, ok in result.items() if ok]
                    skipped = [t for t, ok in result.items() if not ok]
                    st.success(
                        f"✅ Synced {len(synced_tables)} tables: "
                        + ", ".join(f"`{t}`" for t in synced_tables)
                    )
                    if skipped:
                        st.caption(
                            "Skipped (no data yet): "
                            + ", ".join(f"`{t}`" for t in skipped)
                        )
                except Exception as exc:
                    st.error(f"❌ Sync failed: {exc}")
                    logger.exception("Grafana sync failed")

    # --- Sync status ---
    if st.session_state.grafana_sync_result:
        st.markdown("---")
        result = st.session_state.grafana_sync_result
        status_cols = st.columns(len(result))
        for col, (table, ok) in zip(status_cols, result.items()):
            col.metric(table, "✅ Synced" if ok else "⬜ Empty")

    st.markdown("---")

    # --- Analytics Tabs ---
    tab_native, tab_full, tab_panels, tab_direct = st.tabs(
        ["✨ Live Analytics (Built-in)", "📊 Embedded Grafana", "📌 Individual Grafana Panels", "🌐 Open Grafana Directly"]
    )

    with tab_native:
        render_native_grafana_dashboard()

    with tab_full:
        default_grafana = "http://localhost:3000"
        try:
            if hasattr(st, "secrets") and "GRAFANA_URL" in st.secrets:
                default_grafana = st.secrets["GRAFANA_URL"]
        except Exception:
            pass

        grafana_url = st.text_input("Grafana Server URL", value=default_grafana, key="grafana_embed_url")
        dashboard_url = f"{grafana_url}/d/dataml-analytics?kiosk&refresh=10s"

        st.warning(
            "💡 **Note on Browser Iframe Security & Grafana:**\n\n"
            "- If you are accessing this app over **HTTPS** (e.g. Streamlit Cloud), web browsers block embedding `http://localhost:3000` (mixed content rule), showing a blank iframe.\n"
            "- To view embedded Grafana, run the app locally (`streamlit run app.py`) or configure an HTTPS Grafana server URL above.\n"
            "- Otherwise, use the **✨ Live Analytics (Built-in)** tab for instant charts!"
        )

        components.iframe(
            src=dashboard_url,
            height=1200,
            scrolling=True,
        )

    with tab_panels:
        grafana_url = st.session_state.get("grafana_embed_url", default_grafana)
        st.warning(
            "💡 **Note for Online / Mentor Viewers:**\n\n"
            "- If accessing this app online via **Streamlit Cloud**, `http://localhost:3000` attempts to connect to **your local machine**'s port 3000.\n"
            "- If Grafana is not running locally on your computer, browsers display `localhost refused to connect`.\n"
            "- For instant interactive visualizations without running local Grafana, please use the **✨ Live Analytics (Built-in)** tab!"
        )
        st.caption("Individual Grafana panels (requires running local/cloud Grafana instance).")

        solo_base = f"{grafana_url}/d-solo/dataml-analytics?orgId=1&kiosk&refresh=10s"

        panel_categories = {
            "🏠 Executive Summary": {
                "📊 Pipeline Key Metrics": 1,
                "🎯 Best Model Score": 2,
                "💎 Data Quality Score": 9,
                "📋 Pipeline Stage Progress": 10,
            },
            "🤖 Model Analytics": {
                "📊 Model Comparison": 3,
                "📋 Model Metrics Detail": 11,
                "📉 Prediction Error Distribution": 12,
                "🏆 Model Leaderboard": 8,
            },
            "📊 Feature & Data Analysis": {
                "🎯 Feature Importance": 5,
                "🍩 Data Composition": 4,
                "🔥 Correlation Heatmap": 13,
                "📋 Column Statistics": 14,
            },
            "🔍 Data Quality & Distributions": {
                "⚠️ Missing Data by Column": 15,
                "📊 Data Distribution": 16,
            },
            "🎯 Predictions Deep-Dive": {
                "📈 Actual vs Predicted": 6,
                "🎯 Residual vs Predicted": 17,
                "📉 Prediction Confidence Band": 18,
            },
        }

        category = st.selectbox(
            "Panel Category",
            list(panel_categories.keys()),
            key="grafana_panel_category",
        )

        panels = panel_categories[category]
        panel_col1, panel_col2 = st.columns(2)

        for idx, (label, panel_id) in enumerate(panels.items()):
            with (panel_col1 if idx % 2 == 0 else panel_col2):
                st.markdown(f"**{label}**")
                panel_height = 450 if panel_id in (13, 14, 18) else 350
                components.iframe(
                    src=f"{solo_base}&panelId={panel_id}",
                    height=panel_height,
                    scrolling=True,
                )

    with tab_direct:
        grafana_url = st.session_state.get("grafana_embed_url", default_grafana)
        st.markdown(
            f"""
            ### 🌐 Open Grafana Directly

            Access the full Grafana interface for advanced editing, custom queries, and more panels.

            **URL:** [{grafana_url}]({grafana_url})

            **Default login:** `admin` / `admin` (first-time only)

            **Tips:**
            - Create custom panels with SQL queries against your data tables
            - Available tables: `raw_data`, `cleaned_data`, `processed_data`, `predictions`, `model_results`, `feature_importance`, `pipeline_metadata`, `column_statistics`, `correlation_matrix`, `missing_data_summary`, `model_metrics_detail`, `prediction_residuals`, `data_distribution`
            - Use Grafana's built-in alerting to monitor metric thresholds
            """
        )
        if st.button("🔗 Open Grafana in New Tab", use_container_width=True):
            st.markdown(
                f'<meta http-equiv="refresh" content="0;url={grafana_url}">',
                unsafe_allow_html=True,
            )

    # --- Setup instructions ---
    with st.expander("🛠️ First-time Grafana setup", expanded=False):
        st.markdown(
            """
            **If you haven't installed Grafana yet:**

            1. Run the setup script in PowerShell (as Administrator):
               ```powershell
               .\\setup_grafana.ps1
               ```
            2. This will download, install, and configure Grafana automatically
            3. Grafana will be available at `http://localhost:3000`
            4. Default login: `admin` / `admin`

            **After setup:**
            - Upload a CSV in the **Upload Dataset** page
            - Run cleaning/preprocessing as needed
            - Come back here and click **Sync Data to Grafana**
            """
        )


def page_download() -> None:
    """Download cleaned data, predictions, and saved models."""
    section_header("⬇️", "Download Results")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Cleaned dataset")
        if st.session_state.cleaned_df is not None:
            st.download_button(
                "Download CSV",
                data=to_csv_bytes(st.session_state.cleaned_df),
                file_name="cleaned_dataset.csv",
                mime="text/csv",
                use_container_width=True,
            )
        else:
            st.caption("Not available yet.")

    with col2:
        st.subheader("Predictions")
        if st.session_state.predictions_df is not None:
            st.caption(f"{len(st.session_state.predictions_df):,} rows ready")
            st.download_button(
                "Download CSV",
                data=to_csv_bytes(st.session_state.predictions_df),
                file_name="predictions.csv",
                mime="text/csv",
                use_container_width=True,
            )
        else:
            st.caption("Generate predictions on the Predictions page first.")

    with col3:
        st.subheader("Trained model")
        artifact_path = st.session_state.model_artifact_path
        if artifact_path and Path(artifact_path).exists():
            with open(artifact_path, "rb") as model_file:
                st.download_button(
                    "Download Model (.joblib)",
                    data=model_file.read(),
                    file_name="best_model.joblib",
                    mime="application/octet-stream",
                    use_container_width=True,
                )
        else:
            st.caption("Train models on the Machine Learning page first.")


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
PAGE_HANDLERS: dict[str, Callable[[], None]] = {
    "home": page_home,
    "upload": page_upload,
    "cleaning": page_cleaning,
    "eda": page_eda,
    "visualizations": page_visualizations,
    "grafana": page_grafana,
    "preprocessing": page_preprocessing,
    "ml": page_ml,
    "predictions": page_predictions,
    "download": page_download,
}


def main() -> None:
    """Application entry point."""
    init_session_state()
    render_theme_styles()
    render_sidebar_status()
    page_key = render_top_navigation()
    PAGE_HANDLERS[page_key]()


if __name__ == "__main__":
    main()
