"""
mlops.py
========
MLOps Implementation for Medical AI Platform.

Features:
  - Experiment tracking with MLflow
  - Model versioning in database
  - Prediction monitoring
  - Performance drift detection
  - Model health checks
  - Automated reporting
"""

import os
import json
import time
import hashlib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import mlflow
import mlflow.keras
import yaml

from src.logger import get_logger
from src.database import (
    register_model_version, get_recent_predictions,
    get_model_performance_trends, get_dashboard_kpis,
    log_audit
)

logger = get_logger(__name__)

os.makedirs("./logs/mlruns",      exist_ok=True)
os.makedirs("./logs/experiments", exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# MLflow Setup
# ─────────────────────────────────────────────────────────────────────────────
def setup_mlflow(experiment_name: str = "medical_ai_experiments") -> str:
    """
    Initialize MLflow tracking.

    Args:
        experiment_name (str): MLflow experiment identifier

    Returns:
        str: Active experiment ID
    """
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "./logs/mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    experiment = mlflow.get_experiment_by_name(experiment_name)
    experiment_id = experiment.experiment_id if experiment else "0"

    logger.info(f"MLflow initialized | Experiment: {experiment_name}")
    logger.info(f"  Tracking URI: {tracking_uri}")
    logger.info(f"  Experiment ID: {experiment_id}")

    return experiment_id


# ─────────────────────────────────────────────────────────────────────────────
# Experiment Logging
# ─────────────────────────────────────────────────────────────────────────────
def log_experiment(
    model_name: str,
    architecture: str,
    hyperparameters: Dict,
    metrics: Dict,
    model,
    artifacts_dir: Optional[str] = None
) -> str:
    """
    Log a complete training experiment to MLflow.

    Logs:
    - All hyperparameters (batch_size, epochs, lr, etc.)
    - All evaluation metrics (accuracy, recall, AUC, etc.)
    - Model artifact (saved .h5)
    - Training plots
    - Confusion matrix

    Args:
        model_name (str): Model identifier
        architecture (str): Model architecture description
        hyperparameters (dict): Training hyperparameters
        metrics (dict): Evaluation metrics
        model: Trained Keras model
        artifacts_dir (str): Directory containing plots to log

    Returns:
        str: MLflow run ID
    """
    setup_mlflow()

    with mlflow.start_run(run_name=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M')}"):
        run_id = mlflow.active_run().info.run_id

        # Log tags
        mlflow.set_tags({
            "model_name":   model_name,
            "architecture": architecture,
            "framework":    "TensorFlow/Keras",
            "task":         "binary_classification",
            "domain":       "medical_imaging",
            "dataset":      "chest_xray_pneumonia"
        })

        # Log hyperparameters
        for key, value in hyperparameters.items():
            mlflow.log_param(key, value)

        # Log metrics
        metric_keys = [
            "accuracy", "precision", "recall", "f1_score",
            "roc_auc", "specificity", "sensitivity"
        ]
        for key in metric_keys:
            if key in metrics:
                mlflow.log_metric(key, metrics[key])

        # Log confusion matrix values
        for key in ["true_positive", "true_negative", "false_positive", "false_negative"]:
            if key in metrics:
                mlflow.log_metric(key, metrics[key])

        # Log model artifact
        try:
            model_path = f"./models/saved_models/{model_name}.h5"
            if os.path.exists(model_path):
                mlflow.log_artifact(model_path, artifact_path="model")
                logger.info(f"Model artifact logged: {model_path}")
        except Exception as e:
            logger.warning(f"Could not log model artifact: {e}")

        # Log plots
        if artifacts_dir and os.path.exists(artifacts_dir):
            for f in Path(artifacts_dir).glob("*.png"):
                if model_name in f.name or "comparison" in f.name:
                    try:
                        mlflow.log_artifact(str(f), artifact_path="plots")
                    except Exception:
                        pass

        logger.info(f"MLflow run logged | Run ID: {run_id}")
        return run_id


# ─────────────────────────────────────────────────────────────────────────────
# Model Versioning
# ─────────────────────────────────────────────────────────────────────────────
def register_and_version_model(
    model_name: str,
    architecture: str,
    metrics: Dict,
    hyperparameters: Dict,
    training_epochs: int,
    model=None
) -> Dict:
    """
    Register model in database and MLflow (dual tracking).

    Args:
        model_name (str): Model identifier
        architecture (str): Architecture description
        metrics (dict): Evaluation metrics
        hyperparameters (dict): Training settings
        training_epochs (int): Actual epochs trained
        model: Trained Keras model (for MLflow logging)

    Returns:
        dict: Registration result with run_id and version info
    """
    # Database registration
    model_v = register_model_version(
        model_name=model_name,
        architecture=architecture,
        metrics=metrics,
        file_path=f"./models/saved_models/{model_name}.h5",
        hyperparameters=hyperparameters,
        training_epochs=training_epochs,
        notes=f"Registered via MLOps pipeline on {datetime.now().isoformat()}"
    )

    # MLflow logging
    run_id = None
    try:
        run_id = log_experiment(
            model_name=model_name,
            architecture=architecture,
            hyperparameters=hyperparameters,
            metrics=metrics,
            model=model,
            artifacts_dir="./logs/plots"
        )
    except Exception as e:
        logger.warning(f"MLflow logging failed (non-critical): {e}")

    result = {
        "model_name":   model_name,
        "version":      model_v.version,
        "db_id":        model_v.id,
        "mlflow_run_id": run_id,
        "recall":       metrics.get("recall"),
        "accuracy":     metrics.get("accuracy"),
        "registered_at": datetime.now().isoformat()
    }

    logger.info(f"Model registered | {model_name} {model_v.version} | "
                f"Recall: {metrics.get('recall', 0):.4f}")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Prediction Monitoring
# ─────────────────────────────────────────────────────────────────────────────
def monitor_prediction_drift(
    window_days: int = 7,
    alert_threshold: float = 0.10
) -> Dict:
    """
    Detect distribution drift in recent predictions.

    Compares the pneumonia rate in the last window_days to the
    overall historical baseline. A significant shift may indicate:
    - Seasonal disease changes
    - Input data distribution change
    - Model degradation

    Args:
        window_days (int): Recent window in days
        alert_threshold (float): Alert if drift exceeds this fraction

    Returns:
        dict: Drift analysis results
    """
    df = get_recent_predictions(limit=5000)

    if df.empty:
        return {"status": "no_data", "message": "No predictions to analyze"}

    df["predicted_at"] = pd.to_datetime(df["predicted_at"])
    cutoff = datetime.utcnow() - timedelta(days=window_days)

    recent_df = df[df["predicted_at"] >= cutoff]
    historical_df = df[df["predicted_at"] < cutoff]

    if recent_df.empty:
        return {"status": "no_recent_data"}

    # Pneumonia rates
    recent_rate = (
        (recent_df["prediction_label"] == "PNEUMONIA").sum() / len(recent_df)
    )
    historical_rate = (
        (historical_df["prediction_label"] == "PNEUMONIA").sum() / len(historical_df)
        if len(historical_df) > 0 else recent_rate
    )

    drift = abs(recent_rate - historical_rate)
    alert = drift > alert_threshold

    # Confidence trends
    recent_conf = recent_df["confidence"].mean()
    hist_conf   = historical_df["confidence"].mean() if len(historical_df) > 0 else recent_conf
    conf_drift  = abs(recent_conf - hist_conf)

    result = {
        "status":               "alert" if alert else "normal",
        "window_days":          window_days,
        "recent_predictions":   len(recent_df),
        "historical_predictions": len(historical_df),
        "recent_pneumonia_rate":    round(float(recent_rate),     4),
        "historical_pneumonia_rate": round(float(historical_rate), 4),
        "prediction_drift":     round(float(drift),       4),
        "recent_avg_confidence":    round(float(recent_conf),  4),
        "historical_avg_confidence": round(float(hist_conf),   4),
        "confidence_drift":     round(float(conf_drift),   4),
        "alert_threshold":      alert_threshold,
        "alert_triggered":      alert,
        "checked_at":           datetime.now().isoformat()
    }

    if alert:
        logger.warning(
            f"DRIFT ALERT: Pneumonia rate changed by {drift:.2%} "
            f"(recent={recent_rate:.2%}, historical={historical_rate:.2%})"
        )
        log_audit("drift_alert", "monitoring", "prediction_drift",
                  f"Pneumonia rate drift: {drift:.4f}", result)
    else:
        logger.info(f"Drift check PASSED | drift={drift:.4f} < threshold={alert_threshold}")

    return result


def monitor_model_health(model_name: str) -> Dict:
    """
    Check model health: prediction volume, confidence distribution,
    error rates.

    Args:
        model_name (str): Model to check

    Returns:
        dict: Health metrics
    """
    df = get_recent_predictions(limit=1000)

    if df.empty:
        return {"status": "no_data"}

    if "model_name" in df.columns:
        model_df = df[df["model_name"] == model_name]
    else:
        model_df = df

    if model_df.empty:
        return {"status": "no_data_for_model", "model": model_name}

    # Confidence statistics
    conf_mean = model_df["confidence"].mean()
    conf_std  = model_df["confidence"].std()
    conf_min  = model_df["confidence"].min()
    conf_max  = model_df["confidence"].max()

    # Low-confidence predictions (< 0.6) — borderline cases
    low_conf_count = (model_df["confidence"] < 0.6).sum()
    low_conf_rate  = low_conf_count / len(model_df)

    # Prediction distribution
    pred_dist = model_df["prediction_label"].value_counts().to_dict()

    health = {
        "model_name":       model_name,
        "total_predictions": len(model_df),
        "confidence_mean":  round(float(conf_mean),       4),
        "confidence_std":   round(float(conf_std),        4),
        "confidence_min":   round(float(conf_min),        4),
        "confidence_max":   round(float(conf_max),        4),
        "low_confidence_count": int(low_conf_count),
        "low_confidence_rate":  round(float(low_conf_rate), 4),
        "prediction_distribution": pred_dist,
        "health_status":    "warning" if low_conf_rate > 0.15 else "healthy",
        "checked_at":       datetime.now().isoformat()
    }

    logger.info(f"Model health check: {model_name} | "
                f"Status: {health['health_status']} | "
                f"Avg confidence: {conf_mean:.4f}")

    return health


# ─────────────────────────────────────────────────────────────────────────────
# Automated Reporting
# ─────────────────────────────────────────────────────────────────────────────
def generate_mlops_report() -> str:
    """
    Generate a comprehensive MLOps monitoring report.
    Saves as Markdown file.

    Returns:
        str: Path to generated report
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f"./logs/mlops_report_{timestamp}.md"

    kpis  = get_dashboard_kpis()
    drift = monitor_prediction_drift()
    model_trends = get_model_performance_trends()

    lines = [
        "# MLOps Monitoring Report",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Dashboard KPIs",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total Patients | {kpis.get('total_patients', 0)} |",
        f"| Total Predictions | {kpis.get('total_predictions', 0)} |",
        f"| Pneumonia Cases | {kpis.get('pneumonia_cases', 0)} |",
        f"| Pneumonia Rate | {kpis.get('pneumonia_rate_pct', 0):.1f}% |",
        f"| Avg Confidence | {kpis.get('avg_confidence', 0):.4f} |",
        "",
        "## Prediction Drift Monitor",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Status | {drift.get('status', 'N/A')} |",
        f"| Recent Pneumonia Rate | {drift.get('recent_pneumonia_rate', 0):.4f} |",
        f"| Historical Rate | {drift.get('historical_pneumonia_rate', 0):.4f} |",
        f"| Drift | {drift.get('prediction_drift', 0):.4f} |",
        f"| Alert | {drift.get('alert_triggered', False)} |",
        "",
        "## Model Performance History",
    ]

    if not model_trends.empty:
        lines.append(model_trends.to_markdown(index=False))

    lines += [
        "",
        "---",
        "*Report auto-generated by Medical AI MLOps Pipeline*"
    ]

    with open(report_path, "w") as f:
        f.write("\n".join(lines))

    logger.info(f"MLOps report generated: {report_path}")
    return report_path


if __name__ == "__main__":
    setup_mlflow()
    drift_result = monitor_prediction_drift()
    logger.info(f"Drift monitoring: {drift_result}")
    report_path = generate_mlops_report()
    logger.info(f"Report: {report_path}")
    logger.info("MLOps pipeline test COMPLETE ✓")
