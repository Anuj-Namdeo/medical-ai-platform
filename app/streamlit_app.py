"""
streamlit_app.py
================
Production-ready Streamlit Web Application for Medical AI Platform.

Pages:
  1. 🏠 Home          — Overview and quick stats
  2. 🔬 Prediction    — Upload X-ray, get prediction
  3. 👥 Patients      — Patient management
  4. 📊 Analytics     — Business intelligence charts
  5. 🤖 Models        — Model registry and comparison
  6. ⚙️ MLOps         — Monitoring and drift detection

Run:
  streamlit run app/streamlit_app.py
"""

import os
import sys
import uuid
import hashlib
import json
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image
import cv2

# ── Add project root to Python path ─────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Page configuration (MUST be first Streamlit command) ────────────────────
st.set_page_config(
    page_title="Medical AI Platform",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/your-repo/medical-ai",
        "About":    "AI-Powered Medical Image Disease Detection Platform v1.0"
    }
)

# ── Lazy imports (only after set_page_config) ────────────────────────────────
from src.database import (
    initialize_database, create_patient, get_patient_by_id,
    get_all_patients, save_prediction, get_patient_predictions,
    get_recent_predictions, get_active_model_versions,
    get_dashboard_kpis, get_disease_rate_by_gender,
    get_disease_rate_by_age_group, get_disease_rate_by_hospital,
    get_monthly_trends, get_model_performance_trends,
    get_best_model, seed_sample_data
)
from src.mlops import monitor_prediction_drift, monitor_model_health
from src.logger import get_logger

logger = get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Global CSS Styling
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main color scheme */
    :root {
        --primary:   #2C3E50;
        --secondary: #3498DB;
        --danger:    #E74C3C;
        --success:   #27AE60;
        --warning:   #F39C12;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer    {visibility: hidden;}

    /* Metric cards */
    div[data-testid="metric-container"] {
        background-color: #F8F9FA;
        border:        1px solid #DEE2E6;
        border-radius: 8px;
        padding:       12px;
    }

    /* Prediction result boxes */
    .pneumonia-box {
        background: linear-gradient(135deg, #FF6B6B, #EE5A24);
        color:         white;
        padding:       20px;
        border-radius: 12px;
        text-align:    center;
        font-size:     24px;
        font-weight:   bold;
        margin:        10px 0;
    }
    .normal-box {
        background: linear-gradient(135deg, #6BCB77, #4D9B57);
        color:         white;
        padding:       20px;
        border-radius: 12px;
        text-align:    center;
        font-size:     24px;
        font-weight:   bold;
        margin:        10px 0;
    }

    /* Section headers */
    .section-header {
        background:    linear-gradient(90deg, #2C3E50, #3498DB);
        color:         white;
        padding:       10px 20px;
        border-radius: 8px;
        font-size:     18px;
        font-weight:   bold;
        margin:        15px 0 10px 0;
    }

    /* Sidebar */
    .css-1d391kg { background-color: #2C3E50; }

    /* Responsive table */
    .dataframe { font-size: 12px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Session State Initialization
# ─────────────────────────────────────────────────────────────────────────────
def init_session_state():
    defaults = {
        "current_patient_id": None,
        "prediction_result":  None,
        "uploaded_image":     None,
        "db_initialized":     False
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ─────────────────────────────────────────────────────────────────────────────
# Database Initialization (cached)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def init_db():
    """Initialize database once per session."""
    initialize_database()
    try:
        kpis = get_dashboard_kpis()
        if kpis["total_patients"] == 0:
            seed_sample_data(100)
            st.info("Demo data loaded. Connect real data for production use.")
    except Exception as e:
        logger.warning(f"DB init note: {e}")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Model Loading (cached)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_prediction_model(model_name: str = "densenet121"):
    """
    Load trained model for inference.
    Returns None if model file not found (demo mode).
    """
    model_path = f"./models/saved_models/{model_name}.h5"
    if not os.path.exists(model_path):
        # Try to find any available model
        for name in ["custom_cnn", "resnet50", "densenet121", "efficientnetb0"]:
            alt = f"./models/saved_models/{name}.h5"
            if os.path.exists(alt):
                model_path = alt
                model_name = name
                break
        else:
            logger.warning("No trained model found. Running in demo mode.")
            return None, None

    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(model_path)
        logger.info(f"Model loaded for inference: {model_path}")
        return model, model_name
    except Exception as e:
        logger.error(f"Model load error: {e}")
        return None, None


# ─────────────────────────────────────────────────────────────────────────────
# Prediction Engine
# ─────────────────────────────────────────────────────────────────────────────
def predict_image(
    image: np.ndarray,
    model,
    threshold: float = 0.5
) -> dict:
    """
    Run inference on a preprocessed image.

    Args:
        image (np.ndarray): RGB image array
        model: Loaded Keras model
        threshold (float): Decision threshold

    Returns:
        dict: Prediction result with label, confidence, probability
    """
    # Preprocess
    img_resized  = cv2.resize(image, (224, 224))
    img_norm     = img_resized.astype(np.float32) / 255.0
    img_batch    = np.expand_dims(img_norm, axis=0)

    # Predict
    raw_prob = float(model.predict(img_batch, verbose=0)[0][0])
    label    = "PNEUMONIA" if raw_prob >= threshold else "NORMAL"

    # Confidence = distance from decision boundary
    if label == "PNEUMONIA":
        confidence = raw_prob
    else:
        confidence = 1 - raw_prob

    return {
        "label":           label,
        "raw_probability": round(raw_prob,   4),
        "confidence":      round(confidence, 4),
        "threshold":       threshold,
        "is_pneumonia":    label == "PNEUMONIA"
    }


def demo_predict(image: np.ndarray) -> dict:
    """
    Demo prediction when no trained model is available.
    Uses image statistics to simulate a prediction.
    """
    import random
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    mean_intensity = gray.mean()

    # Simulate: darker images more likely PNEUMONIA (realistic heuristic)
    raw_prob = 1 - (mean_intensity / 255.0)
    raw_prob = max(0.1, min(0.95, raw_prob + random.gauss(0, 0.1)))
    label    = "PNEUMONIA" if raw_prob >= 0.5 else "NORMAL"
    conf     = raw_prob if label == "PNEUMONIA" else (1 - raw_prob)

    return {
        "label":           label,
        "raw_probability": round(raw_prob, 4),
        "confidence":      round(conf,     4),
        "threshold":       0.5,
        "is_pneumonia":    label == "PNEUMONIA",
        "demo_mode":       True
    }


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar Navigation
# ─────────────────────────────────────────────────────────────────────────────
def render_sidebar():
    """Render the sidebar with navigation and quick stats."""
    with st.sidebar:
        st.markdown("## 🏥 Medical AI Platform")
        st.markdown("---")

        page = st.radio(
            "Navigate",
            options=[
                "🏠 Home",
                "🔬 Prediction",
                "👥 Patients",
                "📊 Analytics",
                "🤖 Models",
                "⚙️ MLOps Monitor"
            ],
            label_visibility="collapsed"
        )

        st.markdown("---")

        # Quick KPIs in sidebar
        try:
            kpis = get_dashboard_kpis()
            st.markdown("### Quick Stats")
            st.metric("Total Patients",    kpis.get("total_patients", 0))
            st.metric("Total Predictions", kpis.get("total_predictions", 0))
            st.metric("Pneumonia Rate",
                      f"{kpis.get('pneumonia_rate_pct', 0):.1f}%")
            st.metric("Avg Confidence",
                      f"{kpis.get('avg_confidence', 0):.2%}")
        except Exception:
            pass

        st.markdown("---")
        st.markdown("**Model:** DenseNet121")
        st.markdown("**Version:** v1.0")
        st.markdown("**Status:** 🟢 Active")
        st.markdown("---")
        st.caption("© 2024 Medical AI Platform v1.0")

    return page.split(" ", 1)[1].strip()


# ─────────────────────────────────────────────────────────────────────────────
# Page 1: Home
# ─────────────────────────────────────────────────────────────────────────────
def page_home():
    st.title("🏥 Medical AI Disease Detection Platform")
    st.markdown(
        "AI-powered chest X-ray analysis using Deep Learning to detect "
        "**Pneumonia** from radiograph images."
    )

    st.markdown("---")

    # KPI Row
    kpis = get_dashboard_kpis()
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("👥 Patients",      kpis.get("total_patients", 0))
    c2.metric("🔬 Predictions",   kpis.get("total_predictions", 0))
    c3.metric("🦠 Pneumonia Cases", kpis.get("pneumonia_cases", 0))
    c4.metric("📈 Pneumonia Rate", f"{kpis.get('pneumonia_rate_pct', 0):.1f}%")
    c5.metric("🎯 Avg Confidence", f"{kpis.get('avg_confidence', 0):.2%}")

    st.markdown("---")
    st.markdown("### 📐 System Architecture")

    arch_text = """
    ```
    ┌─────────────────────────────────────────────────────────────────┐
    │               Medical AI Platform Architecture                   │
    ├─────────────────────────────────────────────────────────────────┤
    │                                                                  │
    │  [Chest X-Ray Input]                                             │
    │       │                                                          │
    │       ▼                                                          │
    │  [OpenCV Preprocessing]  ──►  Resize + CLAHE + Normalize        │
    │       │                                                          │
    │       ▼                                                          │
    │  [Deep Learning Models]                                          │
    │  ┌──────────────────────────────────────────┐                   │
    │  │  Custom CNN  │ ResNet50 │ DenseNet121     │                   │
    │  │              │          │ EfficientNetB0  │                   │
    │  └──────────────────────────────────────────┘                   │
    │       │                                                          │
    │       ▼                                                          │
    │  [Prediction]  ──►  NORMAL / PNEUMONIA + Confidence             │
    │       │                                                          │
    │       ▼                                                          │
    │  [SQL Database]  ──►  patients + predictions + model_versions   │
    │       │                                                          │
    │       ▼                                                          │
    │  [Analytics Engine]  ──►  Charts + Reports + Power BI           │
    │       │                                                          │
    │       ▼                                                          │
    │  [MLOps Monitor]  ──►  Drift Detection + Model Health           │
    └─────────────────────────────────────────────────────────────────┘
    ```
    """
    st.code(arch_text.strip(), language=None)

    # Monthly trends preview
    st.markdown("### 📈 Monthly Prediction Trends")
    df = get_monthly_trends()
    if not df.empty:
        fig = px.line(
            df, x="month",
            y=["total_predictions", "pneumonia_count"],
            title="Monthly Activity",
            labels={"value": "Count", "variable": "Metric"},
            color_discrete_map={
                "total_predictions": "#3498DB",
                "pneumonia_count":   "#E74C3C"
            }
        )
        fig.update_layout(plot_bgcolor="white", height=350)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No prediction data yet. Make predictions to see trends.")


# ─────────────────────────────────────────────────────────────────────────────
# Page 2: Prediction
# ─────────────────────────────────────────────────────────────────────────────
def page_prediction():
    st.title("🔬 X-Ray Disease Prediction")
    st.markdown("Upload a chest X-ray image to get an AI-powered diagnosis.")

    model, model_name = load_prediction_model()
    demo_mode = model is None

    if demo_mode:
        st.warning(
            "⚠️ **Demo Mode**: No trained model found. "
            "Train models first using `python src/cnn_model.py`. "
            "Demo predictions are shown for UI testing."
        )

    # Model selection
    st.sidebar.markdown("### Prediction Settings")
    threshold = st.sidebar.slider(
        "Decision Threshold",
        min_value=0.1, max_value=0.9,
        value=0.5, step=0.05,
        help="Lower threshold → more sensitive (higher recall, lower precision)"
    )

    # ── Patient Information ──────────────────────────────────────────────────
    st.markdown("#### 1. Patient Information")
    col1, col2, col3 = st.columns(3)

    with col1:
        patient_name = st.text_input("Patient Name *", placeholder="John Doe")
        age          = st.number_input("Age", min_value=0, max_value=120, value=35)

    with col2:
        gender   = st.selectbox("Gender", ["Male", "Female", "Other"])
        hospital = st.selectbox(
            "Hospital",
            ["City General Hospital", "St. Mary Medical Center",
             "Apollo Diagnostics", "Metro Health Clinic", "Other"]
        )

    with col3:
        department = st.selectbox(
            "Department",
            ["Radiology", "Emergency", "Pulmonology", "ICU", "OPD"]
        )
        doctor = st.text_input("Referring Doctor", placeholder="Dr. Smith")

    notes = st.text_area("Clinical Notes (optional)", height=80)

    # ── Image Upload ─────────────────────────────────────────────────────────
    st.markdown("#### 2. Upload Chest X-Ray")
    uploaded = st.file_uploader(
        "Choose an X-ray image",
        type=["jpg", "jpeg", "png"],
        help="Upload a chest X-ray in JPG or PNG format"
    )

    if uploaded is not None:
        img_bytes = uploaded.read()
        img_hash  = hashlib.sha256(img_bytes).hexdigest()

        # Decode image
        img_array = np.frombuffer(img_bytes, np.uint8)
        img_cv2   = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        img_rgb   = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)

        col_img, col_info = st.columns([1, 1])
        with col_img:
            st.image(img_rgb, caption="Uploaded X-Ray", use_column_width=True)
        with col_info:
            h, w, _ = img_rgb.shape
            st.markdown("**Image Details:**")
            st.markdown(f"- **Filename:** {uploaded.name}")
            st.markdown(f"- **Dimensions:** {w} × {h} px")
            st.markdown(f"- **File size:** {len(img_bytes)/1024:.1f} KB")
            st.markdown(f"- **SHA256:** `{img_hash[:16]}...`")

        # ── Run Prediction ───────────────────────────────────────────────────
        st.markdown("#### 3. Run Prediction")

        if st.button("🔬 Analyze X-Ray", type="primary", use_container_width=True):
            if not patient_name.strip():
                st.error("Please enter patient name.")
                return

            with st.spinner("Analyzing X-Ray with AI..."):
                # Get or create patient
                try:
                    patient = create_patient(
                        name=patient_name.strip(),
                        age=int(age),
                        gender=gender,
                        hospital=hospital,
                        department=department,
                        doctor=doctor if doctor else None
                    )

                    # Run prediction
                    if demo_mode:
                        result = demo_predict(img_rgb)
                    else:
                        result = predict_image(img_rgb, model, threshold)

                    # Get active model version ID
                    model_versions = get_active_model_versions()
                    model_v_id = model_versions[0]["id"] if model_versions else None

                    # Save to DB
                    pred_record = save_prediction(
                        patient_db_id=patient.id,
                        prediction_label=result["label"],
                        confidence=result["confidence"],
                        raw_probability=result["raw_probability"],
                        image_hash=img_hash,
                        model_version_id=model_v_id,
                        threshold_used=threshold,
                        notes=notes if notes else None
                    )

                    st.session_state.prediction_result = result
                    st.session_state.current_patient_id = patient.id

                except Exception as e:
                    st.error(f"Error during prediction: {e}")
                    logger.error(f"Prediction error: {e}")
                    return

            # ── Display Results ──────────────────────────────────────────────
            result = st.session_state.prediction_result
            st.markdown("---")
            st.markdown("### 🎯 Prediction Result")

            if result["is_pneumonia"]:
                st.markdown(
                    f"""<div class="pneumonia-box">
                    🦠 PNEUMONIA DETECTED<br>
                    <small>Confidence: {result['confidence']:.1%}</small>
                    </div>""",
                    unsafe_allow_html=True
                )
                st.error(
                    "⚠️ **Action Required**: High probability of pneumonia detected. "
                    "Please consult a radiologist for confirmation."
                )
            else:
                st.markdown(
                    f"""<div class="normal-box">
                    ✅ NORMAL CHEST X-RAY<br>
                    <small>Confidence: {result['confidence']:.1%}</small>
                    </div>""",
                    unsafe_allow_html=True
                )
                st.success(
                    "✅ No signs of pneumonia detected. "
                    "Regular follow-up recommended."
                )

            # Metrics
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Diagnosis",    result["label"])
            r2.metric("Confidence",   f"{result['confidence']:.2%}")
            r3.metric("Raw Prob",     f"{result['raw_probability']:.4f}")
            r4.metric("Threshold",    f"{result['threshold']:.2f}")

            # Confidence gauge
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=result["raw_probability"] * 100,
                title={"text": "Pneumonia Probability (%)"},
                gauge={
                    "axis":  {"range": [0, 100]},
                    "bar":   {"color": "#E74C3C"},
                    "steps": [
                        {"range": [0,  30],  "color": "#27AE60"},
                        {"range": [30, 50],  "color": "#F39C12"},
                        {"range": [50, 100], "color": "#E74C3C"},
                    ],
                    "threshold": {
                        "line":  {"color": "black", "width": 4},
                        "value": 50
                    }
                }
            ))
            fig_gauge.update_layout(height=280)
            st.plotly_chart(fig_gauge, use_container_width=True)

            if result.get("demo_mode"):
                st.caption("*Demo prediction — train models for real inference*")

            # Patient history
            st.markdown("---")
            st.markdown("### 📋 Patient Prediction History")
            try:
                history = get_patient_predictions(patient.id)
                if history:
                    df_hist = pd.DataFrame(history)
                    st.dataframe(
                        df_hist[["prediction_id", "prediction_label",
                                 "confidence", "predicted_at"]].head(10),
                        use_container_width=True
                    )
            except Exception:
                pass


# ─────────────────────────────────────────────────────────────────────────────
# Page 3: Patients
# ─────────────────────────────────────────────────────────────────────────────
def page_patients():
    st.title("👥 Patient Management")

    tab1, tab2 = st.tabs(["All Patients", "Add Patient"])

    with tab1:
        st.markdown("### Patient Records")
        try:
            patients = get_all_patients(limit=500)
            if patients:
                df = pd.DataFrame(patients)
                # Display filters
                col_f1, col_f2 = st.columns(2)
                with col_f1:
                    search = st.text_input("🔍 Search by name", "")
                with col_f2:
                    gender_filter = st.selectbox(
                        "Filter by gender",
                        ["All", "Male", "Female", "Other"]
                    )

                if search:
                    df = df[df["name"].str.contains(search, case=False, na=False)]
                if gender_filter != "All":
                    df = df[df["gender"] == gender_filter]

                st.markdown(f"**Showing {len(df)} patients**")
                st.dataframe(
                    df[["patient_id", "name", "age", "gender",
                         "hospital", "doctor", "created_at"]],
                    use_container_width=True, height=400
                )

                # Export
                csv = df.to_csv(index=False)
                st.download_button(
                    "📥 Export to CSV",
                    data=csv,
                    file_name="patients.csv",
                    mime="text/csv"
                )
            else:
                st.info("No patient records found.")
        except Exception as e:
            st.error(f"Error loading patients: {e}")

    with tab2:
        st.markdown("### Register New Patient")
        with st.form("add_patient_form"):
            col1, col2 = st.columns(2)
            with col1:
                name       = st.text_input("Full Name *")
                age        = st.number_input("Age", 0, 120, 30)
                gender     = st.selectbox("Gender", ["Male", "Female", "Other"])
            with col2:
                hospital   = st.text_input("Hospital")
                department = st.text_input("Department")
                doctor     = st.text_input("Referring Doctor")
            contact = st.text_input("Contact Number")

            submitted = st.form_submit_button("Register Patient", type="primary")
            if submitted:
                if not name.strip():
                    st.error("Name is required.")
                else:
                    try:
                        patient = create_patient(
                            name=name.strip(), age=int(age),
                            gender=gender, hospital=hospital,
                            department=department, doctor=doctor,
                            contact=contact
                        )
                        st.success(
                            f"✅ Patient registered | ID: **{patient.patient_id}**"
                        )
                    except Exception as e:
                        st.error(f"Registration failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Page 4: Analytics
# ─────────────────────────────────────────────────────────────────────────────
def page_analytics():
    st.title("📊 Business Analytics Dashboard")

    # KPI Row
    kpis = get_dashboard_kpis()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Patients",      kpis.get("total_patients", 0))
    c2.metric("Total Predictions",   kpis.get("total_predictions", 0))
    c3.metric("Pneumonia Rate",       f"{kpis.get('pneumonia_rate_pct', 0):.1f}%")
    c4.metric("Avg Model Confidence", f"{kpis.get('avg_confidence', 0):.2%}")

    st.markdown("---")

    # ── Row 1: Gender + Age ──────────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Disease Rate by Gender")
        df_gender = get_disease_rate_by_gender()
        if not df_gender.empty:
            fig = px.bar(
                df_gender, x="gender", y="pneumonia_rate_pct",
                color="gender", text="pneumonia_rate_pct",
                color_discrete_sequence=["#3498DB", "#E74C3C", "#27AE60"]
            )
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig.update_layout(plot_bgcolor="white", showlegend=False, height=350)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data available")

    with col2:
        st.markdown("#### Disease Rate by Age Group")
        df_age = get_disease_rate_by_age_group()
        if not df_age.empty:
            fig = px.bar(
                df_age, x="age_group", y="pneumonia_rate_pct",
                color="pneumonia_rate_pct",
                color_continuous_scale="RdYlGn_r",
                text="pneumonia_rate_pct"
            )
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig.update_layout(plot_bgcolor="white", height=350)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data available")

    # ── Row 2: Hospital + Monthly Trends ────────────────────────────────────
    col3, col4 = st.columns(2)

    with col3:
        st.markdown("#### Disease Rate by Hospital")
        df_hosp = get_disease_rate_by_hospital()
        if not df_hosp.empty:
            fig = px.bar(
                df_hosp, x="pneumonia_rate_pct", y="hospital",
                orientation="h",
                color="pneumonia_rate_pct",
                color_continuous_scale="Reds",
                text="total_predictions"
            )
            fig.update_traces(texttemplate="n=%{text}", textposition="outside")
            fig.update_layout(plot_bgcolor="white", height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data available")

    with col4:
        st.markdown("#### Monthly Prediction Trends")
        df_monthly = get_monthly_trends()
        if not df_monthly.empty:
            fig = px.line(
                df_monthly, x="month",
                y=["pneumonia_count", "normal_count"],
                title="",
                markers=True,
                color_discrete_map={
                    "pneumonia_count": "#E74C3C",
                    "normal_count":    "#27AE60"
                }
            )
            fig.update_layout(plot_bgcolor="white", height=350)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No monthly data yet")

    # ── Row 3: Confidence Distribution ──────────────────────────────────────
    st.markdown("#### Prediction Confidence Distribution")
    df_pred = get_recent_predictions(limit=500)
    if not df_pred.empty:
        fig = go.Figure()
        for label, color in [("NORMAL", "#27AE60"), ("PNEUMONIA", "#E74C3C")]:
            subset = df_pred[df_pred["prediction_label"] == label]["confidence"]
            if len(subset) > 0:
                fig.add_trace(go.Histogram(
                    x=subset, name=label,
                    nbinsx=25, opacity=0.75,
                    marker_color=color
                ))
        fig.update_layout(
            barmode="overlay",
            plot_bgcolor="white",
            xaxis_title="Confidence",
            yaxis_title="Count",
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Raw Data Table ───────────────────────────────────────────────────────
    st.markdown("#### Recent Predictions")
    if not df_pred.empty:
        display_cols = [c for c in
            ["prediction_id", "patient_name", "prediction_label",
             "confidence", "hospital", "predicted_at"]
            if c in df_pred.columns]
        st.dataframe(df_pred[display_cols].head(50), use_container_width=True)
        csv = df_pred.to_csv(index=False)
        st.download_button("📥 Export Predictions", csv,
                           "predictions.csv", "text/csv")


# ─────────────────────────────────────────────────────────────────────────────
# Page 5: Models
# ─────────────────────────────────────────────────────────────────────────────
def page_models():
    st.title("🤖 Model Registry & Performance")

    # Model versions table
    st.markdown("### Registered Model Versions")
    try:
        versions = get_active_model_versions()
        if versions:
            df_mv = pd.DataFrame(versions)
            st.dataframe(
                df_mv[["model_name", "version", "accuracy", "precision",
                        "recall", "f1_score", "roc_auc", "trained_at"]],
                use_container_width=True
            )

            # Highlight best
            best = get_best_model("recall")
            if best:
                st.success(
                    f"✅ **Best model by Recall:** {best['model_name']} "
                    f"{best['version']} | Recall = {best['recall']:.4f}"
                )
        else:
            st.warning(
                "No models registered yet. "
                "Train models using `python src/transfer_learning.py`"
            )
    except Exception as e:
        st.error(f"Error loading models: {e}")

    # Performance trends
    st.markdown("### Model Performance Trends")
    df_trends = get_model_performance_trends()
    if not df_trends.empty:
        fig = px.line(
            df_trends, x="trained_at",
            y=["accuracy", "recall", "f1_score", "roc_auc"],
            color_discrete_map={
                "accuracy": "#3498DB",
                "recall":   "#E74C3C",
                "f1_score": "#27AE60",
                "roc_auc":  "#9B59B6"
            },
            markers=True,
            title="Model Performance Over Versions"
        )
        fig.update_layout(plot_bgcolor="white", yaxis_range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Train models to see performance trends here.")

    # Architecture description
    st.markdown("### Model Architectures")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        **Custom CNN**
        - 4 Conv Blocks
        - BatchNorm
        - GlobalAvgPool
        - 25M params
        """)
    with col2:
        st.markdown("""
        **ResNet50**
        - Residual connections
        - 50 layers deep
        - 25.6M params
        - Skip connections
        """)
    with col3:
        st.markdown("""
        **DenseNet121**
        - Dense connections
        - 121 layers
        - 8M params
        - Best for X-rays
        """)
    with col4:
        st.markdown("""
        **EfficientNetB0**
        - Compound scaling
        - 5.3M params
        - State-of-the-art
        - Efficient inference
        """)


# ─────────────────────────────────────────────────────────────────────────────
# Page 6: MLOps Monitor
# ─────────────────────────────────────────────────────────────────────────────
def page_mlops():
    st.title("⚙️ MLOps Monitoring Dashboard")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📡 Prediction Drift Monitor")
        window = st.slider("Analysis window (days)", 1, 30, 7)
        threshold_drift = st.slider("Alert threshold", 0.05, 0.30, 0.10)

        if st.button("Run Drift Check", type="primary"):
            with st.spinner("Analyzing prediction distribution..."):
                result = monitor_prediction_drift(window, threshold_drift)

            status_color = "🔴" if result.get("alert_triggered") else "🟢"
            st.markdown(f"**Status:** {status_color} {result.get('status', 'N/A').upper()}")

            metrics = {
                "Recent Rate":     f"{result.get('recent_pneumonia_rate', 0):.2%}",
                "Historical Rate": f"{result.get('historical_pneumonia_rate', 0):.2%}",
                "Drift Amount":    f"{result.get('prediction_drift', 0):.4f}",
                "Threshold":       f"{result.get('alert_threshold', 0):.4f}",
                "Alert":           "YES" if result.get("alert_triggered") else "NO"
            }

            for k, v in metrics.items():
                st.markdown(f"- **{k}:** {v}")

            if result.get("alert_triggered"):
                st.error(
                    "⚠️ **DRIFT ALERT**: Prediction distribution has shifted "
                    "significantly. Consider model retraining."
                )
            else:
                st.success("✅ Prediction distribution is stable.")

    with col2:
        st.markdown("### 🏥 Model Health Check")
        model_name_input = st.text_input(
            "Model name to check", "densenet121"
        )
        if st.button("Check Model Health", type="primary"):
            with st.spinner("Checking model health..."):
                health = monitor_model_health(model_name_input)

            status_icon = "⚠️" if health.get("health_status") == "warning" else "✅"
            st.markdown(f"**Status:** {status_icon} {health.get('health_status', 'N/A').upper()}")

            health_metrics = {
                "Total Predictions":  health.get("total_predictions", 0),
                "Mean Confidence":    f"{health.get('confidence_mean', 0):.4f}",
                "Std Confidence":     f"{health.get('confidence_std', 0):.4f}",
                "Low-Conf Count":     health.get("low_confidence_count", 0),
                "Low-Conf Rate":      f"{health.get('low_confidence_rate', 0):.2%}"
            }
            for k, v in health_metrics.items():
                st.markdown(f"- **{k}:** {v}")

    # MLflow link
    st.markdown("---")
    st.markdown("### 📊 MLflow Experiment Tracker")
    st.info(
        "Start MLflow UI to view experiment runs:\n\n"
        "```\nmlflow ui --backend-store-uri ./logs/mlruns\n```\n"
        "Then open: http://localhost:5000"
    )

    # System info
    st.markdown("### 🖥️ System Information")
    col_s1, col_s2, col_s3 = st.columns(3)
    col_s1.info(f"**Python:** {sys.version.split()[0]}")
    col_s2.info(f"**Platform:** {sys.platform}")
    try:
        import tensorflow as tf
        col_s3.info(f"**TensorFlow:** {tf.__version__}")
    except Exception:
        col_s3.info("**TensorFlow:** Not loaded")


# ─────────────────────────────────────────────────────────────────────────────
# Main App Router
# ─────────────────────────────────────────────────────────────────────────────
def main():
    init_session_state()
    init_db()

    page = render_sidebar()

    page_map = {
        "Home":          page_home,
        "Prediction":    page_prediction,
        "Patients":      page_patients,
        "Analytics":     page_analytics,
        "Models":        page_models,
        "MLOps Monitor": page_mlops
    }

    # Route to correct page
    render_fn = page_map.get(page, page_home)
    render_fn()


if __name__ == "__main__":
    main()
