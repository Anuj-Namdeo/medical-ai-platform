"""
helpers.py
==========
Shared utility functions for the Streamlit Medical AI Platform.
Handles image preprocessing, prediction formatting, UI helpers, and session state.
"""

import io
import base64
import hashlib
import datetime
import numpy as np
import pandas as pd
import streamlit as st
import cv2
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, List
import plotly.graph_objects as go
import plotly.express as px


# ─────────────────────────────────────────────
# IMAGE HELPERS
# ─────────────────────────────────────────────

def read_image_bytes(uploaded_file) -> np.ndarray:
    """
    Convert a Streamlit UploadedFile into an OpenCV BGR image array.
    """
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return img


def preprocess_for_display(img: np.ndarray, max_size: int = 512) -> np.ndarray:
    """
    Resize image for display while preserving aspect ratio.
    Converts BGR to RGB for Streamlit rendering.
    """
    h, w = img.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return rgb


def preprocess_for_model(img: np.ndarray, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    Full preprocessing pipeline for model inference.
    Steps: convert to grayscale → CLAHE → denoise → resize → normalize → expand dims.
    Returns shape (1, H, W, 3) float32 array.
    """
    # Convert BGR to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # CLAHE enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    # Gaussian denoise
    denoised = cv2.GaussianBlur(enhanced, (3, 3), 0)
    # Resize
    resized = cv2.resize(denoised, target_size)
    # Stack to 3 channels
    rgb = np.stack([resized, resized, resized], axis=-1)
    # Normalize to [0, 1]
    normalized = rgb.astype(np.float32) / 255.0
    # Add batch dimension
    batched = np.expand_dims(normalized, axis=0)
    return batched


def compute_image_hash(img_bytes: bytes) -> str:
    """
    Compute SHA-256 hash of raw image bytes.
    Used to detect duplicate submissions.
    """
    return hashlib.sha256(img_bytes).hexdigest()


def image_to_base64(img: np.ndarray) -> str:
    """
    Convert a NumPy BGR image to a Base64-encoded JPEG string.
    Useful for embedding images in HTML or storing in DB.
    """
    _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buffer.tobytes()).decode('utf-8')


def base64_to_image(b64_str: str) -> np.ndarray:
    """
    Decode a Base64 JPEG string back to a NumPy BGR image.
    """
    img_bytes = base64.b64decode(b64_str)
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


# ─────────────────────────────────────────────
# PREDICTION HELPERS
# ─────────────────────────────────────────────

def format_prediction_result(prob: float, threshold: float = 0.5) -> Dict[str, Any]:
    """
    Convert raw model probability to a structured result dict.

    Args:
        prob: Probability of PNEUMONIA class (0..1).
        threshold: Decision threshold (default 0.5).

    Returns:
        Dict with keys: label, confidence, risk_level, recommendation
    """
    label = "PNEUMONIA" if prob >= threshold else "NORMAL"
    confidence = prob if label == "PNEUMONIA" else (1.0 - prob)

    if label == "PNEUMONIA":
        if confidence >= 0.85:
            risk_level = "HIGH"
            recommendation = "Immediate medical attention required. Consult a pulmonologist urgently."
        elif confidence >= 0.65:
            risk_level = "MODERATE"
            recommendation = "Medical evaluation recommended within 24 hours."
        else:
            risk_level = "LOW-MODERATE"
            recommendation = "Follow-up with a physician for confirmation."
    else:
        if confidence >= 0.85:
            risk_level = "LOW"
            recommendation = "Lungs appear normal. Continue routine health monitoring."
        else:
            risk_level = "BORDERLINE"
            recommendation = "Result is borderline normal. Physician review is advised."

    return {
        "label": label,
        "confidence": float(confidence),
        "probability_pneumonia": float(prob),
        "probability_normal": float(1.0 - prob),
        "risk_level": risk_level,
        "recommendation": recommendation,
        "threshold_used": threshold,
    }


def demo_predict(img: np.ndarray) -> Dict[str, Any]:
    """
    Demo prediction using image statistics (no model required).
    Used for portfolio showcasing when no trained model is available.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean_intensity = float(np.mean(gray))
    std_intensity = float(np.std(gray))
    # Heuristic: lower mean intensity + higher std → more likely pneumonia
    # (infiltrates create uneven brightness in X-rays)
    raw_score = (1.0 - (mean_intensity / 255.0)) * 0.6 + min(std_intensity / 80.0, 1.0) * 0.4
    # Add small noise for demo variety
    noise = np.random.uniform(-0.05, 0.05)
    prob = float(np.clip(raw_score + noise, 0.05, 0.95))
    result = format_prediction_result(prob)
    result["model_used"] = "Demo Mode (Heuristic)"
    result["demo_mode"] = True
    return result


# ─────────────────────────────────────────────
# SESSION STATE MANAGEMENT
# ─────────────────────────────────────────────

def init_session_state():
    """
    Initialize all required Streamlit session state keys.
    Call once at the top of the main app.
    """
    defaults = {
        "patient_id": None,
        "patient_name": "",
        "patient_age": 30,
        "patient_gender": "Male",
        "patient_hospital": "General Hospital",
        "last_prediction": None,
        "prediction_history": [],
        "total_predictions": 0,
        "model_loaded": False,
        "db_initialized": False,
        "current_page": "Home",
        "upload_count": 0,
        "demo_mode": False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def reset_prediction_state():
    """Clear prediction-related session state between uploads."""
    st.session_state.last_prediction = None
    st.session_state.patient_id = None


# ─────────────────────────────────────────────
# STREAMLIT UI HELPERS
# ─────────────────────────────────────────────

def get_risk_color(risk_level: str) -> str:
    """Return a hex color string matching the risk level."""
    mapping = {
        "HIGH":         "#EF4444",   # red
        "MODERATE":     "#F97316",   # orange
        "LOW-MODERATE": "#EAB308",   # yellow
        "BORDERLINE":   "#A78BFA",   # purple
        "LOW":          "#22C55E",   # green
    }
    return mapping.get(risk_level, "#94A3B8")


def render_confidence_gauge(confidence: float, label: str) -> go.Figure:
    """
    Create a Plotly gauge chart showing prediction confidence.

    Args:
        confidence: Float 0..1
        label: 'PNEUMONIA' or 'NORMAL'
    """
    pct = round(confidence * 100, 1)
    color = "#EF4444" if label == "PNEUMONIA" else "#22C55E"

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=pct,
        title={"text": f"Confidence: {label}", "font": {"size": 18, "color": "#F1F5F9"}},
        delta={"reference": 50, "increasing": {"color": color}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#94A3B8"},
            "bar": {"color": color},
            "bgcolor": "#1E293B",
            "borderwidth": 2,
            "bordercolor": "#334155",
            "steps": [
                {"range": [0,  50], "color": "#1E293B"},
                {"range": [50, 75], "color": "#292524"},
                {"range": [75, 100], "color": "#1C1917"},
            ],
            "threshold": {
                "line": {"color": "#FFFFFF", "width": 3},
                "thickness": 0.75,
                "value": pct,
            },
        },
        number={"suffix": "%", "font": {"size": 24, "color": "#F1F5F9"}},
    ))

    fig.update_layout(
        height=280,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="#0F172A",
        font={"color": "#F1F5F9"},
    )
    return fig


def render_probability_bar(prob_pneumonia: float) -> go.Figure:
    """
    Dual horizontal bar showing Normal vs Pneumonia probability.
    """
    prob_normal = 1.0 - prob_pneumonia

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="NORMAL",
        x=[round(prob_normal * 100, 1)],
        y=["Diagnosis"],
        orientation='h',
        marker_color="#22C55E",
        text=f"{round(prob_normal * 100, 1)}%",
        textposition='inside',
    ))
    fig.add_trace(go.Bar(
        name="PNEUMONIA",
        x=[round(prob_pneumonia * 100, 1)],
        y=["Diagnosis"],
        orientation='h',
        marker_color="#EF4444",
        text=f"{round(prob_pneumonia * 100, 1)}%",
        textposition='inside',
    ))

    fig.update_layout(
        barmode='stack',
        height=120,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="#0F172A",
        plot_bgcolor="#0F172A",
        font={"color": "#F1F5F9"},
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(range=[0, 100], title="Probability (%)", color="#94A3B8"),
        yaxis=dict(color="#94A3B8"),
    )
    return fig


def metric_card(title: str, value: str, delta: Optional[str] = None, color: str = "#2563EB") -> str:
    """
    Generate an HTML metric card string for use with st.markdown().
    """
    delta_html = f'<p style="color:#94A3B8;font-size:12px;margin:0">{delta}</p>' if delta else ""
    return f"""
    <div style="
        background: #1E293B;
        border-left: 4px solid {color};
        border-radius: 8px;
        padding: 16px 20px;
        margin: 4px 0;
    ">
        <p style="color:#94A3B8;font-size:12px;margin:0;text-transform:uppercase;letter-spacing:1px">{title}</p>
        <p style="color:#F1F5F9;font-size:28px;font-weight:700;margin:4px 0">{value}</p>
        {delta_html}
    </div>
    """


def info_card(title: str, body: str, icon: str = "ℹ️") -> str:
    """Generate an informational card HTML block."""
    return f"""
    <div style="
        background: #1E293B;
        border-radius: 8px;
        padding: 16px 20px;
        margin: 8px 0;
        border: 1px solid #334155;
    ">
        <h4 style="color:#F1F5F9;margin:0 0 8px 0">{icon} {title}</h4>
        <p style="color:#CBD5E1;margin:0;line-height:1.6">{body}</p>
    </div>
    """


def warning_banner(message: str) -> None:
    """Render a styled warning banner in Streamlit."""
    st.markdown(f"""
    <div style="
        background: #422006;
        border: 1px solid #F97316;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 8px 0;
    ">
        <span style="color:#FED7AA">⚠️ {message}</span>
    </div>
    """, unsafe_allow_html=True)


def success_banner(message: str) -> None:
    """Render a styled success banner in Streamlit."""
    st.markdown(f"""
    <div style="
        background: #052e16;
        border: 1px solid #22C55E;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 8px 0;
    ">
        <span style="color:#86EFAC">✅ {message}</span>
    </div>
    """, unsafe_allow_html=True)


def error_banner(message: str) -> None:
    """Render a styled error banner in Streamlit."""
    st.markdown(f"""
    <div style="
        background: #450a0a;
        border: 1px solid #EF4444;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 8px 0;
    ">
        <span style="color:#FCA5A5">❌ {message}</span>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# DATA FORMATTING HELPERS
# ─────────────────────────────────────────────

def format_timestamp(ts) -> str:
    """Convert a datetime or string timestamp to a human-readable format."""
    if isinstance(ts, str):
        try:
            ts = datetime.datetime.fromisoformat(ts)
        except ValueError:
            return str(ts)
    if isinstance(ts, datetime.datetime):
        return ts.strftime("%d %b %Y, %I:%M %p")
    return str(ts)


def calculate_age_group(age: int) -> str:
    """Bin age into demographic groups for analytics."""
    if age < 12:   return "Child (0-11)"
    if age < 18:   return "Adolescent (12-17)"
    if age < 35:   return "Young Adult (18-34)"
    if age < 55:   return "Middle Age (35-54)"
    if age < 70:   return "Senior (55-69)"
    return "Elderly (70+)"


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Division with zero-denominator protection."""
    return numerator / denominator if denominator != 0 else default


def round_pct(val: float, decimals: int = 1) -> str:
    """Format float as percentage string, e.g. 0.832 → '83.2%'"""
    return f"{round(val * 100, decimals)}%"


def df_to_download_csv(df: pd.DataFrame) -> bytes:
    """Serialize DataFrame to UTF-8 CSV bytes for st.download_button."""
    return df.to_csv(index=False).encode("utf-8")


def truncate_string(s: str, max_len: int = 40) -> str:
    """Truncate a string with ellipsis if too long."""
    return s if len(s) <= max_len else s[:max_len - 3] + "..."


# ─────────────────────────────────────────────
# FILE / PATH HELPERS
# ─────────────────────────────────────────────

def get_project_root() -> Path:
    """Return the absolute path to the project root directory."""
    return Path(__file__).resolve().parent.parent.parent


def ensure_dir(path: Path) -> Path:
    """Create directory (and parents) if it doesn't exist. Returns the path."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def list_saved_models(models_dir: Optional[Path] = None) -> List[Dict[str, str]]:
    """
    Scan the models/saved_models directory and return info dicts.
    Each dict has keys: name, path, size_mb, modified.
    """
    if models_dir is None:
        models_dir = get_project_root() / "models" / "saved_models"

    results = []
    if not models_dir.exists():
        return results

    for f in models_dir.glob("*.h5"):
        size_mb = round(f.stat().st_size / (1024 * 1024), 2)
        modified = datetime.datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        results.append({
            "name": f.stem,
            "path": str(f),
            "size_mb": size_mb,
            "modified": modified,
        })
    return sorted(results, key=lambda x: x["modified"], reverse=True)


def get_logs_summary(logs_dir: Optional[Path] = None) -> Dict[str, int]:
    """Return counts of log files by type."""
    if logs_dir is None:
        logs_dir = get_project_root() / "logs"
    if not logs_dir.exists():
        return {}
    return {f.suffix: sum(1 for _ in logs_dir.glob(f"*{f.suffix}")) for f in logs_dir.iterdir() if f.is_file()}


# ─────────────────────────────────────────────
# VALIDATION HELPERS
# ─────────────────────────────────────────────

def validate_image_upload(uploaded_file) -> Tuple[bool, str]:
    """
    Validate a Streamlit uploaded file for use as X-ray input.

    Returns:
        (is_valid: bool, error_message: str)
    """
    if uploaded_file is None:
        return False, "No file uploaded."

    allowed_types = ["image/jpeg", "image/jpg", "image/png", "image/bmp"]
    if uploaded_file.type not in allowed_types:
        return False, f"Unsupported file type: {uploaded_file.type}. Please upload JPG, PNG, or BMP."

    max_size_mb = 10
    if uploaded_file.size > max_size_mb * 1024 * 1024:
        return False, f"File too large ({uploaded_file.size / 1e6:.1f} MB). Maximum is {max_size_mb} MB."

    return True, ""


def validate_patient_form(name: str, age: int, gender: str, hospital: str) -> Tuple[bool, str]:
    """
    Validate patient registration form inputs.

    Returns:
        (is_valid: bool, error_message: str)
    """
    if not name or len(name.strip()) < 2:
        return False, "Patient name must be at least 2 characters."
    if not (0 < age < 130):
        return False, "Age must be between 1 and 129."
    if gender not in ("Male", "Female", "Other"):
        return False, "Gender must be Male, Female, or Other."
    if not hospital or len(hospital.strip()) < 2:
        return False, "Hospital name is required."
    return True, ""


# ─────────────────────────────────────────────
# EXPORT HELPERS
# ─────────────────────────────────────────────

def build_prediction_report(patient: Dict, result: Dict) -> str:
    """
    Generate a plain-text clinical prediction report.

    Args:
        patient: Dict with name, age, gender, hospital.
        result: Prediction result dict from format_prediction_result().

    Returns:
        Multi-line string report.
    """
    now = datetime.datetime.now().strftime("%d %B %Y, %I:%M %p")
    lines = [
        "=" * 60,
        "   MEDICAL AI PLATFORM — CHEST X-RAY ANALYSIS REPORT",
        "=" * 60,
        f"  Date / Time   : {now}",
        f"  Patient Name  : {patient.get('name', 'N/A')}",
        f"  Age           : {patient.get('age', 'N/A')} years",
        f"  Gender        : {patient.get('gender', 'N/A')}",
        f"  Hospital      : {patient.get('hospital', 'N/A')}",
        "-" * 60,
        f"  Diagnosis     : {result.get('label', 'N/A')}",
        f"  Confidence    : {round_pct(result.get('confidence', 0))}",
        f"  Risk Level    : {result.get('risk_level', 'N/A')}",
        f"  P(Pneumonia)  : {round_pct(result.get('probability_pneumonia', 0))}",
        f"  P(Normal)     : {round_pct(result.get('probability_normal', 0))}",
        "-" * 60,
        f"  Recommendation:",
        f"  {result.get('recommendation', '')}",
        "=" * 60,
        "  ⚠ DISCLAIMER: This AI tool is NOT a substitute for",
        "  professional medical diagnosis. Always consult a",
        "  qualified healthcare provider for clinical decisions.",
        "=" * 60,
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    # Quick smoke test
    print("Helpers module loaded OK")
    print(f"Project root: {get_project_root()}")
    result = format_prediction_result(0.82)
    print(f"Prediction: {result}")
    group = calculate_age_group(45)
    print(f"Age group: {group}")
