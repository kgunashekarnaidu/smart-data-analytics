"""
Interactive and static visualization helpers for EDA.

Primary output is Plotly figures for use in Streamlit ``st.plotly_chart``.
Matplotlib/Seaborn are used sparingly where they add value.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from core.themes import apply_plotly_theme, get_plotly_template, get_theme, get_active_theme, style_bar_trace
from core.utils import classify_columns, get_logger

logger = get_logger("visualizer")

MAX_SCATTER_ROWS = 5_000
MAX_HEATMAP_COLUMNS = 30


def _apply_theme(fig: go.Figure, title: str | None = None) -> go.Figure:
    """Apply the active UI theme and optional title."""
    return apply_plotly_theme(fig, title=title)


def sample_dataframe(df: pd.DataFrame, max_rows: int = MAX_SCATTER_ROWS) -> pd.DataFrame:
    """Sample large datasets for responsive interactive charts."""
    if len(df) <= max_rows:
        return df
    return df.sample(n=max_rows, random_state=42)


def get_numeric_columns(df: pd.DataFrame) -> list[str]:
    """Return numeric column names."""
    return list(classify_columns(df).numeric)


def get_categorical_columns(df: pd.DataFrame) -> list[str]:
    """Return categorical column names."""
    return list(classify_columns(df).categorical)


def plot_histogram(df: pd.DataFrame, column: str, bins: int = 30) -> go.Figure:
    """Interactive histogram for a numeric column."""
    theme = get_theme(get_active_theme())
    display_name = column.replace("_", " ").title()
    fig = px.histogram(
        df,
        x=column,
        nbins=bins,
        title=f"Distribution of {display_name}",
        labels={column: display_name, "count": "Count"},
        color_discrete_sequence=[theme.chart_primary],
    )
    style_bar_trace(fig)
    return _apply_theme(fig)


def plot_box(df: pd.DataFrame, column: str, group_by: str | None = None) -> go.Figure:
    """Box plot for numeric column with optional grouping."""
    theme = get_theme(get_active_theme())
    display_name = column.replace("_", " ").title()
    fig = px.box(
        df,
        y=column,
        x=group_by,
        color=group_by,
        title=f"Box Plot — {display_name}" + (f" by {group_by.replace('_', ' ').title()}" if group_by else ""),
        labels={column: display_name},
        color_discrete_sequence=list(theme.chart_palette),
    )
    if group_by:
        fig.update_layout(showlegend=False)
    return _apply_theme(fig)


def plot_violin(df: pd.DataFrame, column: str, group_by: str | None = None) -> go.Figure:
    """Violin plot showing distribution shape."""
    fig = px.violin(
        df,
        y=column,
        x=group_by,
        color=group_by,
        box=True,
        points="outliers",
        title=f"Violin plot — {column}",
        color_discrete_sequence=px.colors.qualitative.Pastel,
    )
    if group_by:
        fig.update_layout(showlegend=False)
    return _apply_theme(fig)


def plot_scatter(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    color_col: str | None = None,
    show_trendline: bool = False,
) -> go.Figure:
    """Scatter plot with optional color grouping and OLS trendline."""
    theme = get_theme(get_active_theme())
    plot_df = sample_dataframe(df)
    fig = px.scatter(
        plot_df,
        x=x_col,
        y=y_col,
        color=color_col,
        opacity=0.65,
        trendline="ols" if show_trendline and color_col is None else None,
        title=f"{x_col.replace('_', ' ').title()} vs {y_col.replace('_', ' ').title()}",
        color_discrete_sequence=list(theme.chart_palette),
    )
    return _apply_theme(fig)


def plot_line(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    group_col: str | None = None,
) -> go.Figure:
    """Line chart for trends over time or sequence."""
    if group_col:
        grouped = (
            df.groupby([x_col, group_col], as_index=False)[y_col]
            .mean()
            .sort_values(x_col)
        )
        fig = px.line(
            grouped,
            x=x_col,
            y=y_col,
            color=group_col,
            markers=True,
            title=f"Trend of {y_col} over {x_col}",
        )
    else:
        grouped = df.groupby(x_col, as_index=False)[y_col].mean().sort_values(x_col)
        fig = px.line(
            grouped,
            x=x_col,
            y=y_col,
            markers=True,
            title=f"Trend of {y_col} over {x_col}",
        )
    return _apply_theme(fig)


def plot_bar(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    aggregation: str = "mean",
) -> go.Figure:
    """Bar chart with aggregation over a categorical column."""
    theme = get_theme(get_active_theme())
    aggregated = (
        df.groupby(x_col)[y_col]
        .agg(aggregation)
        .reset_index()
        .sort_values(y_col, ascending=False)
    )
    fig = px.bar(
        aggregated,
        x=x_col,
        y=y_col,
        title=f"{aggregation.title()} of {y_col.replace('_', ' ').title()} by {x_col.replace('_', ' ').title()}",
        color_discrete_sequence=[theme.chart_primary],
    )
    style_bar_trace(fig)
    fig.update_layout(showlegend=False, xaxis_tickangle=-30)
    return _apply_theme(fig)


def plot_count(df: pd.DataFrame, column: str, top_n: int = 20) -> go.Figure:
    """Count plot for categorical frequency."""
    theme = get_theme(get_active_theme())
    display_name = column.replace("_", " ").title()
    counts = df[column].value_counts().head(top_n).reset_index()
    counts.columns = [column, "Count"]
    fig = px.bar(
        counts,
        x=column,
        y="Count",
        title=f"Count of {display_name} (top {top_n})",
        labels={column: display_name, "Count": "Count"},
        color_discrete_sequence=[theme.chart_primary],
    )
    style_bar_trace(fig)
    fig.update_layout(showlegend=False, xaxis_tickangle=-30)
    return _apply_theme(fig)


def plot_pie(df: pd.DataFrame, column: str, value_col: str | None = None, top_n: int = 10) -> go.Figure:
    """Pie chart for category proportions."""
    if value_col:
        pie_df = df.groupby(column)[value_col].sum().reset_index()
        pie_df = pie_df.sort_values(value_col, ascending=False).head(top_n)
        fig = px.pie(
            pie_df,
            names=column,
            values=value_col,
            title=f"{value_col} share by {column}",
            color_discrete_sequence=px.colors.qualitative.Pastel,
        )
    else:
        counts = df[column].value_counts().head(top_n).reset_index()
        counts.columns = [column, "Count"]
        fig = px.pie(
            counts,
            names=column,
            values="Count",
            title=f"Distribution of {column}",
            color_discrete_sequence=px.colors.qualitative.Pastel,
        )
    return _apply_theme(fig)


def plot_correlation_heatmap(df: pd.DataFrame, columns: Iterable[str] | None = None) -> go.Figure:
    """Interactive correlation heatmap for numeric columns."""
    numeric_cols = list(columns) if columns else get_numeric_columns(df)
    numeric_cols = numeric_cols[:MAX_HEATMAP_COLUMNS]

    if len(numeric_cols) < 2:
        fig = go.Figure()
        fig.add_annotation(text="Need at least 2 numeric columns", showarrow=False)
        return _apply_theme(fig, "Correlation Heatmap")

    corr = df[numeric_cols].corr()
    fig = px.imshow(
        corr,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        title="Correlation Heatmap",
        zmin=-1,
        zmax=1,
    )
    return _apply_theme(fig)


def plot_missing_heatmap(df: pd.DataFrame, max_columns: int = 40) -> go.Figure:
    """Heatmap showing missing-value patterns across columns."""
    missing = df.isnull().astype(int)
    cols_with_missing = [c for c in missing.columns if missing[c].sum() > 0]

    if not cols_with_missing:
        fig = go.Figure()
        fig.add_annotation(text="No missing values in dataset", showarrow=False)
        return _apply_theme(fig, "Missing Value Heatmap")

    subset = missing[cols_with_missing[:max_columns]]
    plot_df = sample_dataframe(subset, max_rows=500)
    fig = px.imshow(
        plot_df.T,
        aspect="auto",
        color_continuous_scale=["#1a1f2e", "#f44336"],
        labels=dict(color="Missing"),
        title="Missing Value Heatmap (sampled rows)",
    )
    fig.update_layout(xaxis_title="Row sample", yaxis_title="Column")
    return _apply_theme(fig)


def plot_missing_bar(df: pd.DataFrame) -> go.Figure:
    """Bar chart of missing value counts per column."""
    theme = get_theme(get_active_theme())
    missing = (
        df.isnull()
        .sum()
        .reset_index(name="Missing Count")
        .rename(columns={"index": "Column"})
    )
    missing = missing[missing["Missing Count"] > 0].sort_values("Missing Count", ascending=False)
    if missing.empty:
        fig = go.Figure()
        fig.add_annotation(text="No missing values", showarrow=False)
        return _apply_theme(fig, "Missing Values by Column")

    fig = px.bar(
        missing,
        x="Column",
        y="Missing Count",
        title="Missing Values by Column",
        color_discrete_sequence=[theme.chart_secondary],
    )
    style_bar_trace(fig, color=theme.chart_secondary)
    fig.update_layout(showlegend=False, xaxis_tickangle=-45)
    return _apply_theme(fig)


def plot_pair_relationships(
    df: pd.DataFrame,
    columns: list[str],
    target: str | None = None,
) -> go.Figure:
    """Scatter matrix for selected numeric columns."""
    cols = [c for c in columns if c in df.columns][:6]
    if target and target in df.columns and target not in cols:
        cols = cols[:5] + [target]

    if len(cols) < 2:
        fig = go.Figure()
        fig.add_annotation(text="Select at least 2 numeric columns", showarrow=False)
        return _apply_theme(fig, "Pair Relationships")

    plot_df = sample_dataframe(df[cols])
    fig = px.scatter_matrix(
        plot_df,
        dimensions=cols,
        color=target if target in cols else None,
        opacity=0.6,
        title="Pair Relationships (scatter matrix)",
    )
    fig.update_traces(diagonal_visible=False)
    return _apply_theme(fig)


def plot_outlier_comparison(
    before: pd.Series,
    after: pd.Series,
    column: str,
) -> go.Figure:
    """Side-by-side histograms before and after outlier treatment."""
    theme = get_theme(get_active_theme())
    fig = make_subplots(rows=1, cols=2, subplot_titles=(f"Before — {column}", f"After — {column}"))

    fig.add_trace(
        go.Histogram(x=before, name="Before", marker=dict(color="#ef5350", line=dict(width=0.5, color="#c62828"))),
        row=1, col=1,
    )
    fig.add_trace(
        go.Histogram(x=after, name="After", marker=dict(color=theme.chart_secondary, line=dict(width=0.5, color="#1b5e20"))),
        row=1, col=2,
    )

    fig.update_layout(
        template=get_plotly_template(),
        title="Before vs After Outlier Treatment",
        showlegend=False,
        barmode="overlay",
        paper_bgcolor=theme.chart_paper,
        plot_bgcolor=theme.chart_plot,
        font=dict(family="Inter, Segoe UI, Roboto, Helvetica Neue, Arial, sans-serif", color=theme.chart_font),
        bargap=0.12,
    )
    return fig


def build_eda_figures(df: pd.DataFrame) -> dict[str, go.Figure]:
    """
    Generate a standard set of EDA figures for automatic dashboard display.

    Returns:
        Dictionary mapping chart name to Plotly figure.
    """
    figures: dict[str, go.Figure] = {}
    numeric = get_numeric_columns(df)
    categorical = get_categorical_columns(df)

    figures["missing_bar"] = plot_missing_bar(df)
    if df.isnull().sum().sum() > 0:
        figures["missing_heatmap"] = plot_missing_heatmap(df)

    if len(numeric) >= 2:
        figures["correlation"] = plot_correlation_heatmap(df, numeric)

    if numeric:
        figures["histogram"] = plot_histogram(df, numeric[0])
        figures["box"] = plot_box(df, numeric[0])

    if categorical:
        figures["count"] = plot_count(df, categorical[0])

    if len(numeric) >= 2:
        figures["scatter"] = plot_scatter(df, numeric[0], numeric[1])

    logger.info("Generated %s automatic EDA figures", len(figures))
    return figures
