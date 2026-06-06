"""
analytics.py
============
Business Analytics Engine for Medical AI Platform.

Generates:
  - Disease rate analysis by demographics
  - Monthly/quarterly trend charts
  - Model performance analytics
  - Confidence distribution analysis
  - Hospital-level comparisons
  - Executive KPI summaries
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from src.database import (
    get_disease_rate_by_gender,
    get_disease_rate_by_age_group,
    get_disease_rate_by_hospital,
    get_monthly_trends,
    get_model_performance_trends,
    get_dashboard_kpis,
    get_recent_predictions
)
from src.logger import get_logger

logger = get_logger(__name__)
os.makedirs("./logs/analytics", exist_ok=True)

# ── Consistent color scheme ──────────────────────────────────────────────────
COLORS = {
    "PNEUMONIA": "#E74C3C",
    "NORMAL":    "#27AE60",
    "primary":   "#2C3E50",
    "secondary": "#3498DB",
    "accent":    "#E67E22",
    "background":"#ECF0F1"
}
PALETTE = [COLORS["primary"], COLORS["secondary"], COLORS["accent"],
           COLORS["PNEUMONIA"], COLORS["NORMAL"], "#9B59B6", "#1ABC9C"]


# ─────────────────────────────────────────────────────────────────────────────
# Disease Rate Analytics
# ─────────────────────────────────────────────────────────────────────────────
def plot_disease_by_gender(save_path: Optional[str] = None) -> go.Figure:
    """
    Interactive Plotly bar chart: pneumonia rate by gender.

    Business Insight: Identifies which gender has higher disease burden,
    enabling targeted screening programs.
    """
    df = get_disease_rate_by_gender()
    if df.empty:
        logger.warning("No data for gender analysis")
        return None

    fig = px.bar(
        df,
        x="gender",
        y="pneumonia_rate_pct",
        color="gender",
        title="Pneumonia Rate by Gender (%)",
        labels={"gender": "Gender", "pneumonia_rate_pct": "Pneumonia Rate (%)"},
        text="pneumonia_rate_pct",
        color_discrete_sequence=PALETTE
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(
        plot_bgcolor="white",
        showlegend=False,
        height=400,
        font=dict(size=13)
    )

    if save_path:
        fig.write_html(save_path)
        logger.info(f"Gender chart saved: {save_path}")

    return fig


def plot_disease_by_age_group(save_path: Optional[str] = None) -> go.Figure:
    """
    Interactive bar chart: pneumonia rate by age group.

    Business Insight: Age-stratified analysis helps prioritize
    high-risk age groups for early screening.
    """
    df = get_disease_rate_by_age_group()
    if df.empty:
        return None

    fig = px.bar(
        df,
        x="age_group",
        y="pneumonia_rate_pct",
        color="pneumonia_rate_pct",
        title="Pneumonia Rate by Age Group (%)",
        labels={"age_group": "Age Group", "pneumonia_rate_pct": "Pneumonia Rate (%)"},
        text="pneumonia_rate_pct",
        color_continuous_scale="RdYlGn_r"
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(plot_bgcolor="white", height=400)

    if save_path:
        fig.write_html(save_path)

    return fig


def plot_disease_by_hospital(save_path: Optional[str] = None) -> go.Figure:
    """
    Horizontal bar chart: disease rate by hospital.

    Business Insight: Hospital-level comparison reveals facility-specific
    patterns and helps resource allocation.
    """
    df = get_disease_rate_by_hospital()
    if df.empty:
        return None

    fig = px.bar(
        df,
        x="pneumonia_rate_pct",
        y="hospital",
        orientation="h",
        title="Pneumonia Rate by Hospital (%)",
        labels={"hospital": "Hospital", "pneumonia_rate_pct": "Pneumonia Rate (%)"},
        text="total_predictions",
        color="pneumonia_rate_pct",
        color_continuous_scale="Reds"
    )
    fig.update_traces(texttemplate="n=%{text}", textposition="outside")
    fig.update_layout(plot_bgcolor="white", height=500)

    if save_path:
        fig.write_html(save_path)

    return fig


def plot_monthly_trends(save_path: Optional[str] = None) -> go.Figure:
    """
    Line chart with dual Y-axes: monthly predictions + disease rate.

    Business Insight: Time-series trends help administrators plan
    capacity and detect seasonal disease outbreaks.
    """
    df = get_monthly_trends()
    if df.empty:
        return None

    fig = make_subplots(
        rows=1, cols=1,
        specs=[[{"secondary_y": True}]]
    )

    # Total predictions line
    fig.add_trace(
        go.Bar(
            x=df["month"],
            y=df["total_predictions"],
            name="Total Predictions",
            marker_color=COLORS["secondary"],
            opacity=0.7
        ),
        secondary_y=False
    )

    # Pneumonia count line
    fig.add_trace(
        go.Scatter(
            x=df["month"],
            y=df["pneumonia_count"],
            name="Pneumonia Cases",
            mode="lines+markers",
            line=dict(color=COLORS["PNEUMONIA"], width=3),
            marker=dict(size=8)
        ),
        secondary_y=True
    )

    fig.update_layout(
        title="Monthly Prediction & Disease Trends",
        plot_bgcolor="white",
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    fig.update_xaxes(title_text="Month")
    fig.update_yaxes(title_text="Total Predictions", secondary_y=False)
    fig.update_yaxes(title_text="Pneumonia Cases",   secondary_y=True)

    if save_path:
        fig.write_html(save_path)

    return fig


def plot_confidence_distribution(save_path: Optional[str] = None) -> go.Figure:
    """
    Histogram: prediction confidence distribution by class.

    Business Insight: Wide confidence distribution or clustering near 0.5
    indicates an uncertain model that needs retraining.
    """
    df = get_recent_predictions(limit=2000)
    if df.empty:
        return None

    fig = go.Figure()
    for label, color in [("NORMAL", COLORS["NORMAL"]),
                          ("PNEUMONIA", COLORS["PNEUMONIA"])]:
        subset = df[df["prediction_label"] == label]["confidence"]
        fig.add_trace(go.Histogram(
            x=subset,
            name=label,
            nbinsx=30,
            opacity=0.75,
            marker_color=color
        ))

    fig.update_layout(
        title="Prediction Confidence Distribution",
        xaxis_title="Confidence Score",
        yaxis_title="Count",
        barmode="overlay",
        plot_bgcolor="white",
        height=400
    )

    if save_path:
        fig.write_html(save_path)

    return fig


def plot_model_comparison(save_path: Optional[str] = None) -> go.Figure:
    """
    Radar/Spider chart comparing all model versions.

    Business Insight: Multi-dimensional model comparison helps
    stakeholders choose the right model for production.
    """
    df = get_model_performance_trends()
    if df.empty:
        return None

    metrics    = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    categories = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]

    fig = go.Figure()
    colors_radar = ["blue", "red", "green", "orange", "purple"]

    for i, (_, row) in enumerate(df.iterrows()):
        values = [row.get(m, 0) or 0 for m in metrics]
        values_closed = values + [values[0]]  # Close the polygon
        cats_closed   = categories + [categories[0]]

        fig.add_trace(go.Scatterpolar(
            r=values_closed,
            theta=cats_closed,
            fill="toself",
            name=f"{row['model_name']} {row['version']}",
            opacity=0.6,
            line=dict(color=colors_radar[i % len(colors_radar)])
        ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title="Model Performance Radar Chart",
        height=500
    )

    if save_path:
        fig.write_html(save_path)

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Summary Dashboard Report
# ─────────────────────────────────────────────────────────────────────────────
def generate_analytics_report() -> Dict:
    """
    Generate comprehensive analytics report with all visualizations.
    Saves all charts to ./logs/analytics/

    Returns:
        dict: Report metadata and file paths
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir  = "./logs/analytics"
    os.makedirs(base_dir, exist_ok=True)

    kpis = get_dashboard_kpis()
    logger.info(f"Generating analytics report | KPIs: {kpis}")

    report = {
        "generated_at": datetime.now().isoformat(),
        "kpis":         kpis,
        "charts":       {}
    }

    chart_functions = [
        ("gender_chart",     plot_disease_by_gender,     "disease_by_gender.html"),
        ("age_chart",        plot_disease_by_age_group,  "disease_by_age.html"),
        ("hospital_chart",   plot_disease_by_hospital,   "disease_by_hospital.html"),
        ("monthly_chart",    plot_monthly_trends,         "monthly_trends.html"),
        ("confidence_chart", plot_confidence_distribution,"confidence_dist.html"),
        ("model_chart",      plot_model_comparison,       "model_comparison.html"),
    ]

    for key, func, filename in chart_functions:
        try:
            path = os.path.join(base_dir, filename)
            fig  = func(save_path=path)
            if fig is not None:
                report["charts"][key] = path
                logger.info(f"  Chart generated: {filename}")
            else:
                logger.warning(f"  Skipped (no data): {filename}")
        except Exception as e:
            logger.error(f"  Chart failed {filename}: {e}")
            report["charts"][key] = None

    # Save report metadata
    meta_path = os.path.join(base_dir, f"report_meta_{timestamp}.json")
    import json
    with open(meta_path, "w") as f:
        json.dump({k: v for k, v in report.items() if k != "kpis"}, f, indent=2)

    logger.info(f"Analytics report complete: {len(report['charts'])} charts generated")
    return report


if __name__ == "__main__":
    report = generate_analytics_report()
    logger.info(f"Report: {report}")
