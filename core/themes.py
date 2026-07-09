"""
UI theme tokens and helpers for dark / light appearance.

The active theme is stored in Streamlit session state (``ui_theme``) and
mirrored here so Plotly charts and CSS stay in sync.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import plotly.graph_objects as go

ThemeName = Literal["dark", "light"]

FONT_FAMILY = "Inter, Segoe UI, Roboto, Helvetica Neue, Arial, sans-serif"

_active_theme: ThemeName = "dark"


@dataclass(frozen=True)
class ThemeTokens:
    """Color and styling tokens for one UI theme."""

    name: ThemeName
    plotly_template: str
    main_bg: str
    sidebar_bg: str
    sidebar_border: str
    metric_bg: str
    metric_border: str
    metric_shadow: str
    card_bg: str
    card_border: str
    top_nav_bg: str
    top_nav_border: str
    text_primary: str
    text_muted: str
    accent: str
    accent_dark: str
    button_gradient: str
    hero_gradient: str
    hero_shadow: str
    table_highlight: str
    input_bg: str
    input_border: str
    input_text: str
    table_bg: str
    table_header_bg: str
    table_text: str
    table_border: str
    tab_active: str
    brand_title: str
    chart_primary: str
    chart_primary_dark: str
    chart_secondary: str
    chart_paper: str
    chart_plot: str
    chart_font: str
    chart_grid: str
    chart_palette: tuple[str, ...]


THEMES: dict[ThemeName, ThemeTokens] = {
    "dark": ThemeTokens(
        name="dark",
        plotly_template="plotly_dark",
        main_bg="#0f1117",
        sidebar_bg="linear-gradient(180deg, #1a1f2e 0%, #0f1117 100%)",
        sidebar_border="#2d3748",
        metric_bg="linear-gradient(135deg, #1e2a3a, #243447)",
        metric_border="#2d3748",
        metric_shadow="0 4px 15px rgba(0, 0, 0, 0.3)",
        card_bg="linear-gradient(135deg, #1e2a3a, #243447)",
        card_border="#1a73e8",
        top_nav_bg="linear-gradient(90deg, #1a1f2e, #243447)",
        top_nav_border="#2d3748",
        text_primary="#e8eaed",
        text_muted="#9aa0a6",
        accent="#4da3ff",
        accent_dark="#1a73e8",
        button_gradient="linear-gradient(90deg, #1a73e8, #0d47a1)",
        hero_gradient="linear-gradient(135deg, #0d47a1, #1a73e8, #00bcd4)",
        hero_shadow="0 8px 30px rgba(26, 115, 232, 0.4)",
        table_highlight="#1a5c2a",
        input_bg="#1e2a3a",
        input_border="#3d4f63",
        input_text="#e8eaed",
        table_bg="#1a1f2e",
        table_header_bg="#243447",
        table_text="#e8eaed",
        table_border="#2d3748",
        tab_active="#4da3ff",
        brand_title="#ffffff",
        chart_primary="#4da3ff",
        chart_primary_dark="#1a73e8",
        chart_secondary="#66bb6a",
        chart_paper="#1a1f2e",
        chart_plot="#12151c",
        chart_font="#e8eaed",
        chart_grid="#2d3748",
        chart_palette=("#4da3ff", "#66bb6a", "#ffb74d", "#ef5350", "#ab47bc", "#26c6da"),
    ),
    "light": ThemeTokens(
        name="light",
        plotly_template="plotly_white",
        main_bg="#f5f7fb",
        sidebar_bg="linear-gradient(180deg, #ffffff 0%, #eef2f7 100%)",
        sidebar_border="#d0d7e2",
        metric_bg="linear-gradient(135deg, #ffffff, #f0f4fa)",
        metric_border="#d0d7e2",
        metric_shadow="0 4px 14px rgba(15, 23, 42, 0.08)",
        card_bg="linear-gradient(135deg, #ffffff, #f8fafc)",
        card_border="#1a73e8",
        top_nav_bg="linear-gradient(90deg, #ffffff, #eef2f7)",
        top_nav_border="#d0d7e2",
        text_primary="#1f2937",
        text_muted="#6b7280",
        accent="#1a73e8",
        accent_dark="#0d47a1",
        button_gradient="linear-gradient(90deg, #1a73e8, #1565c0)",
        hero_gradient="linear-gradient(135deg, #1565c0, #1a73e8, #42a5f5)",
        hero_shadow="0 8px 24px rgba(26, 115, 232, 0.25)",
        table_highlight="#c8e6c9",
        input_bg="#ffffff",
        input_border="#d0d7e2",
        input_text="#1f2937",
        table_bg="#ffffff",
        table_header_bg="#f0f4fa",
        table_text="#1f2937",
        table_border="#e5e7eb",
        tab_active="#1a73e8",
        brand_title="#0d47a1",
        chart_primary="#1a73e8",
        chart_primary_dark="#0d47a1",
        chart_secondary="#2e7d32",
        chart_paper="#ffffff",
        chart_plot="#f8fafc",
        chart_font="#1f2937",
        chart_grid="#e5e7eb",
        chart_palette=("#1a73e8", "#2e7d32", "#ef6c00", "#c62828", "#6a1b9a", "#00838f"),
    ),
}


def get_theme(name: ThemeName | str = "dark") -> ThemeTokens:
    """Return token set for the requested theme."""
    return THEMES[name if name in THEMES else "dark"]


def set_active_theme(name: ThemeName | str) -> ThemeTokens:
    """Set module-level active theme for chart helpers."""
    global _active_theme
    theme_name: ThemeName = name if name in THEMES else "dark"
    _active_theme = theme_name
    return THEMES[theme_name]


def get_active_theme() -> ThemeName:
    """Return the currently active theme name."""
    return _active_theme


def get_plotly_template(name: ThemeName | str | None = None) -> str:
    """Return Plotly template string for a theme."""
    theme = get_theme(name or _active_theme)
    return theme.plotly_template


def apply_plotly_theme(fig: go.Figure, title: str | None = None) -> go.Figure:
    """Apply the active Plotly template without clobbering existing titles."""
    theme = get_theme(_active_theme)

    layout: dict = {
        "template": theme.plotly_template,
        "paper_bgcolor": theme.chart_paper,
        "plot_bgcolor": theme.chart_plot,
        "font": {"family": FONT_FAMILY, "color": theme.chart_font, "size": 13},
        "margin": dict(l=48, r=32, t=64, b=48),
        "legend": dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color=theme.chart_font, size=12),
        ),
        "hoverlabel": dict(
            bgcolor=theme.chart_paper,
            bordercolor=theme.chart_grid,
            font=dict(color=theme.chart_font, family=FONT_FAMILY, size=12),
        ),
        "xaxis": dict(
            gridcolor=theme.chart_grid,
            linecolor=theme.chart_grid,
            tickfont=dict(color=theme.chart_font, size=12),
            zerolinecolor=theme.chart_grid,
        ),
        "yaxis": dict(
            gridcolor=theme.chart_grid,
            linecolor=theme.chart_grid,
            tickfont=dict(color=theme.chart_font, size=12),
            zerolinecolor=theme.chart_grid,
        ),
    }

    existing_title = ""
    if fig.layout.title and fig.layout.title.text:
        existing_title = str(fig.layout.title.text)

    if title is not None:
        layout["title"] = dict(
            text=title,
            font=dict(size=17, color=theme.chart_font, family=FONT_FAMILY),
            x=0.02,
            xanchor="left",
        )
    elif existing_title:
        layout["title"] = dict(
            text=existing_title,
            font=dict(size=17, color=theme.chart_font, family=FONT_FAMILY),
            x=0.02,
            xanchor="left",
        )

    fig.update_layout(**layout)
    return fig


def style_bar_trace(fig: go.Figure, *, color: str | None = None, opacity: float = 0.88) -> go.Figure:
    """Apply consistent professional bar/histogram styling."""
    theme = get_theme(_active_theme)
    bar_color = color or theme.chart_primary
    fig.update_traces(
        marker=dict(
            color=bar_color,
            line=dict(color=theme.chart_primary_dark, width=0.6),
        ),
        opacity=opacity,
        selector=dict(type="bar"),
    )
    fig.update_traces(
        marker=dict(
            color=bar_color,
            line=dict(color=theme.chart_primary_dark, width=0.6),
        ),
        opacity=opacity,
        selector=dict(type="histogram"),
    )
    fig.update_layout(bargap=0.12, bargroupgap=0.08)
    return fig


def build_app_css(theme: ThemeTokens) -> str:
    """Generate Streamlit custom CSS for the selected theme."""
    return f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {{
        font-family: {FONT_FAMILY};
    }}

    .main {{ background-color: {theme.main_bg}; color: {theme.text_primary}; }}

    [data-testid="stAppViewContainer"],
    [data-testid="stMainBlockContainer"] {{
        background-color: {theme.main_bg};
    }}

    [data-testid="stSidebar"] {{
        background: {theme.sidebar_bg};
        border-right: 1px solid {theme.sidebar_border};
        min-width: 280px !important;
        visibility: visible !important;
    }}

    [data-testid="collapsedControl"] {{
        visibility: visible !important;
        display: flex !important;
    }}

    [data-testid="metric-container"] {{
        background: {theme.metric_bg};
        border: 1px solid {theme.metric_border};
        border-radius: 12px;
        padding: 16px;
        box-shadow: {theme.metric_shadow};
    }}

    [data-testid="metric-container"] label,
    [data-testid="metric-container"] [data-testid="stMetricValue"],
    [data-testid="metric-container"] [data-testid="stMetricLabel"],
    [data-testid="stMetricValue"],
    [data-testid="stMetricLabel"] {{
        color: {theme.text_primary} !important;
    }}

    .section-header {{
        background: linear-gradient(90deg, {theme.accent_dark}, {theme.accent});
        color: white;
        padding: 12px 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        font-size: 20px;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(26, 115, 232, 0.2);
    }}

    .info-card {{
        background: {theme.card_bg};
        border: 1px solid {theme.card_border};
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        color: {theme.text_primary};
    }}

    .info-card p, .info-card li {{
        color: {theme.text_primary};
    }}

    .success-banner {{
        background: linear-gradient(90deg, #0d6b3b, #1a8a4e);
        color: white;
        padding: 12px 20px;
        border-radius: 10px;
        margin: 10px 0;
        text-align: center;
        font-weight: 600;
    }}

    div.stButton > button,
    [data-testid="stDownloadButton"] button {{
        background: {theme.button_gradient} !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 10px 24px !important;
        font-weight: 600 !important;
        font-family: {FONT_FAMILY} !important;
    }}

    div.stButton > button:hover,
    [data-testid="stDownloadButton"] button:hover {{
        filter: brightness(1.06);
    }}

    [data-testid="stDownloadButton"] button p,
    [data-testid="stDownloadButton"] button span,
    [data-testid="stDownloadButton"] button div {{
        color: #ffffff !important;
    }}

    .top-nav-bar {{
        background: {theme.top_nav_bg};
        border: 1px solid {theme.top_nav_border};
        border-radius: 10px;
        padding: 8px 12px;
        margin-bottom: 16px;
    }}

    .pipeline-progress-container {{
        background: {theme.top_nav_bg};
        border: 1px solid {theme.top_nav_border};
        border-radius: 10px;
        padding: 12px 16px;
        margin-bottom: 12px;
    }}

    .pipeline-progress-header {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 8px;
        font-size: 13px;
    }}

    .pipeline-progress-track {{
        width: 100%;
        height: 10px;
        background: {theme.input_border};
        border-radius: 999px;
        overflow: hidden;
    }}

    .pipeline-progress-fill {{
        height: 100%;
        border-radius: 999px;
        background: {theme.button_gradient};
        transition: width 0.4s ease;
        min-width: 0;
    }}

    .pipeline-progress-steps {{
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-top: 8px;
        font-size: 12px;
    }}

    .pipeline-step-done {{
        color: {theme.accent} !important;
        font-weight: 600;
    }}

    .pipeline-step-pending {{
        color: {theme.text_muted} !important;
    }}

    [data-testid="stSidebar"] .theme-brand-title,
    [data-testid="stSidebar"] h1.theme-brand-title {{
        color: {theme.brand_title} !important;
    }}

    .theme-brand-title {{
        color: {theme.brand_title};
        font-size: 26px;
        font-weight: 700;
        margin: 0;
    }}

    .theme-brand-subtitle {{
        color: {theme.text_muted};
        font-size: 12px;
        margin: 0;
    }}

    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li,
    [data-testid="stMarkdownContainer"] span,
    [data-testid="stCaptionContainer"],
    label[data-testid="stWidgetLabel"],
    h1, h2, h3, h4 {{
        color: {theme.text_primary};
        font-family: {FONT_FAMILY};
    }}

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] label[data-testid="stWidgetLabel"] {{
        color: {theme.text_primary};
    }}

    /* Select boxes, inputs, number fields */
    [data-testid="stSelectbox"] [data-baseweb="select"] > div,
    [data-testid="stMultiSelect"] [data-baseweb="select"] > div,
    [data-testid="stNumberInput"] input,
    [data-testid="stTextInput"] input,
    [data-testid="stTextArea"] textarea {{
        background-color: {theme.input_bg} !important;
        color: {theme.input_text} !important;
        border-color: {theme.input_border} !important;
        border-radius: 8px !important;
    }}

    [data-testid="stSelectbox"] svg,
    [data-testid="stMultiSelect"] svg {{
        fill: {theme.text_muted} !important;
    }}

    /* File uploader */
    [data-testid="stFileUploader"] section {{
        background: {theme.input_bg} !important;
        border: 1.5px dashed {theme.accent} !important;
        border-radius: 12px !important;
        box-shadow: 0 0 0 3px rgba(77, 163, 255, 0.15);
        padding: 10px 12px;
    }}

    [data-testid="stFileUploader"] section *,
    [data-testid="stFileUploader"] small,
    [data-testid="stFileUploader"] svg,
    [data-testid="stFileUploader"] label {{
        color: {theme.input_text} !important;
        fill: {theme.input_text} !important;
    }}

    [data-testid="stFileUploader"] button {{
        background: {theme.button_gradient} !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        padding: 10px 18px !important;
    }}


    /* Data tables */
    [data-testid="stDataFrame"],
    [data-testid="stDataFrame"] > div,
    [data-testid="stTable"] {{
        background-color: {theme.table_bg} !important;
        border: 1px solid {theme.table_border} !important;
        border-radius: 8px !important;
    }}

    [data-testid="stDataFrame"] [data-testid="glideDataEditor"],
    [data-testid="stDataFrame"] canvas {{
        background-color: {theme.table_bg} !important;
    }}

    [data-testid="stDataFrame"] *,
    [data-testid="stTable"] * {{
        color: {theme.table_text} !important;
    }}

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
    }}

    .stTabs [data-baseweb="tab"] {{
        color: {theme.text_muted} !important;
        font-weight: 500;
    }}

    .stTabs [aria-selected="true"] {{
        color: {theme.tab_active} !important;
        border-bottom-color: {theme.tab_active} !important;
    }}

    /* Radio (theme picker) */
    [data-testid="stRadio"] label {{
        color: {theme.text_primary} !important;
    }}

    /* Alerts */
    [data-testid="stAlert"] {{
        border-radius: 8px;
    }}

    #MainMenu {{ visibility: hidden; }}
    footer {{ visibility: hidden; }}
</style>
"""


def sidebar_brand_html(theme: ThemeTokens) -> str:
    """HTML block for the sidebar logo area."""
    return f"""
    <div style='text-align:center; padding:10px 0;'>
        <h1 class='theme-brand-title'
            style='color:{theme.brand_title} !important; font-size:26px; font-weight:700; margin:0;'>
            📊 DataML Pro
        </h1>
        <p class='theme-brand-subtitle'
           style='color:{theme.text_muted}; font-size:12px; margin:0;'>
            Analytics & Machine Learning
        </p>
    </div>
    """


def pipeline_progress_html(
    theme: ThemeTokens,
    steps: list[tuple[str, bool]],
) -> str:
    """Render a fillable pipeline progress bar with step labels."""
    total = len(steps)
    completed = sum(1 for _, done in steps if done)
    pct = int((completed / total) * 100) if total else 0

    step_html = []
    for label, done in steps:
        css_class = "pipeline-step-done" if done else "pipeline-step-pending"
        icon = "✓" if done else "○"
        step_html.append(f'<span class="{css_class}">{icon} {label}</span>')

    return f"""
    <div class="pipeline-progress-container">
        <div class="pipeline-progress-header">
            <span style="color:{theme.text_primary}; font-weight:600;">Pipeline Progress</span>
            <span style="color:{theme.text_muted};">{completed}/{total} steps · {pct}%</span>
        </div>
        <div class="pipeline-progress-track">
            <div class="pipeline-progress-fill" style="width:{pct}%;"></div>
        </div>
        <div class="pipeline-progress-steps">
            {''.join(step_html)}
        </div>
    </div>
    """


def home_hero_html(theme: ThemeTokens) -> str:
    """HTML block for the home page hero banner."""
    return f"""
    <div style='background:{theme.hero_gradient};
                padding:36px;border-radius:16px;text-align:center;
                box-shadow:{theme.hero_shadow};margin-bottom:28px;'>
        <h1 style='color:white;font-size:40px;margin:0;font-weight:700;'>📊 DataML Pro</h1>
        <p style='color:rgba(255,255,255,0.9);font-size:17px;margin-top:10px;'>
            Upload any CSV · Clean · Explore · Train ML · Predict
        </p>
    </div>
    """


def home_info_card_html(theme: ThemeTokens, title: str, body_html: str) -> str:
    """HTML block for home page info cards."""
    return f"""
    <div class='info-card'>
        <h3 style='color:{theme.accent_dark}; font-weight:600;'>{title}</h3>
        {body_html}
    </div>
    """
