"""
database.py
===========
Complete database layer for Medical AI Platform.

Supports:
  - SQLite (local development, default)
  - PostgreSQL (production)

Tables:
  - patients         : Patient demographic information
  - predictions      : Model predictions per patient
  - image_metadata   : Image technical properties
  - model_versions   : Trained model registry
  - audit_logs       : System activity logging

All operations use SQLAlchemy ORM with raw SQL fallback.
"""

import os
import sqlite3
import json
import uuid
import hashlib
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import random

import pandas as pd
import numpy as np
from sqlalchemy import (
    create_engine, Column, Integer, String, Float,
    DateTime, Boolean, Text, ForeignKey, inspect,
    text
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from sqlalchemy.exc import IntegrityError
from dotenv import load_dotenv

from src.logger import get_logger

load_dotenv()
logger = get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Database Setup
# ─────────────────────────────────────────────────────────────────────────────
Base = declarative_base()
os.makedirs("./database", exist_ok=True)


def get_database_url() -> str:
    """
    Get database URL from environment variables.
    Falls back to SQLite if PostgreSQL not configured.
    """
    db_url = os.getenv("DATABASE_URL", "sqlite:///./database/medical_ai.db")
    logger.info(f"Database URL: {db_url.split('@')[0]}...")  # Hide password
    return db_url


def create_db_engine():
    """Create SQLAlchemy engine with connection pool settings."""
    db_url = get_database_url()

    if "sqlite" in db_url:
        engine = create_engine(
            db_url,
            connect_args={"check_same_thread": False},
            echo=False
        )
    else:
        engine = create_engine(
            db_url,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
            echo=False
        )

    logger.info("Database engine created")
    return engine


# Singleton engine and session factory
engine  = create_db_engine()
Session = sessionmaker(bind=engine)


# ─────────────────────────────────────────────────────────────────────────────
# ORM Models (Table Definitions)
# ─────────────────────────────────────────────────────────────────────────────
class Patient(Base):
    """
    Stores patient demographic and identification information.
    Each patient can have multiple predictions over time.
    """
    __tablename__ = "patients"

    id           = Column(Integer, primary_key=True, autoincrement=True)
    patient_id   = Column(String(50), unique=True, nullable=False, index=True)
    name         = Column(String(100), nullable=False)
    age          = Column(Integer, nullable=True)
    gender       = Column(String(20), nullable=True)   # Male, Female, Other
    hospital     = Column(String(100), nullable=True)
    department   = Column(String(100), nullable=True)
    doctor       = Column(String(100), nullable=True)
    contact      = Column(String(50), nullable=True)
    created_at   = Column(DateTime, default=datetime.utcnow)
    updated_at   = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active    = Column(Boolean, default=True)

    predictions  = relationship("Prediction", back_populates="patient")

    def to_dict(self) -> Dict:
        return {
            "id": self.id, "patient_id": self.patient_id,
            "name": self.name, "age": self.age,
            "gender": self.gender, "hospital": self.hospital,
            "department": self.department, "doctor": self.doctor,
            "created_at": str(self.created_at)
        }


class Prediction(Base):
    """
    Stores model predictions with full metadata.
    Links to patient and model version.
    """
    __tablename__ = "predictions"

    id                = Column(Integer, primary_key=True, autoincrement=True)
    prediction_id     = Column(String(50), unique=True, nullable=False)
    patient_id        = Column(Integer, ForeignKey("patients.id"), nullable=False)
    model_version_id  = Column(Integer, ForeignKey("model_versions.id"), nullable=True)
    image_path        = Column(String(500), nullable=True)
    image_hash        = Column(String(64), nullable=True)   # SHA256 of image
    prediction_label  = Column(String(20), nullable=False)  # NORMAL / PNEUMONIA
    confidence        = Column(Float, nullable=False)        # 0.0 – 1.0
    raw_probability   = Column(Float, nullable=False)        # Model raw output
    threshold_used    = Column(Float, default=0.5)
    is_correct        = Column(Boolean, nullable=True)       # Doctor review
    doctor_diagnosis  = Column(String(20), nullable=True)
    notes             = Column(Text, nullable=True)
    predicted_at      = Column(DateTime, default=datetime.utcnow)
    reviewed_at       = Column(DateTime, nullable=True)

    patient           = relationship("Patient", back_populates="predictions")
    model_version     = relationship("ModelVersion", back_populates="predictions")

    def to_dict(self) -> Dict:
        return {
            "prediction_id":    self.prediction_id,
            "patient_id":       self.patient_id,
            "prediction_label": self.prediction_label,
            "confidence":       self.confidence,
            "raw_probability":  self.raw_probability,
            "threshold_used":   self.threshold_used,
            "is_correct":       self.is_correct,
            "notes":            self.notes,
            "predicted_at":     str(self.predicted_at)
        }


class ImageMetadata(Base):
    """
    Stores technical metadata about uploaded images.
    """
    __tablename__ = "image_metadata"

    id            = Column(Integer, primary_key=True, autoincrement=True)
    image_hash    = Column(String(64), unique=True, nullable=False)
    filename      = Column(String(255), nullable=True)
    file_size_kb  = Column(Float, nullable=True)
    width         = Column(Integer, nullable=True)
    height        = Column(Integer, nullable=True)
    channels      = Column(Integer, nullable=True)
    color_mode    = Column(String(20), nullable=True)
    format        = Column(String(20), nullable=True)  # JPEG, PNG, etc.
    mean_pixel    = Column(Float, nullable=True)
    std_pixel     = Column(Float, nullable=True)
    created_at    = Column(DateTime, default=datetime.utcnow)


class ModelVersion(Base):
    """
    Model registry: tracks all trained model versions with performance metrics.
    Implements basic Model Versioning (MLOps).
    """
    __tablename__ = "model_versions"

    id               = Column(Integer, primary_key=True, autoincrement=True)
    model_name       = Column(String(100), nullable=False)
    version          = Column(String(20), nullable=False)
    architecture     = Column(String(100), nullable=True)
    file_path        = Column(String(500), nullable=True)
    accuracy         = Column(Float, nullable=True)
    precision        = Column(Float, nullable=True)
    recall           = Column(Float, nullable=True)
    f1_score         = Column(Float, nullable=True)
    roc_auc          = Column(Float, nullable=True)
    specificity      = Column(Float, nullable=True)
    sensitivity      = Column(Float, nullable=True)
    true_positives   = Column(Integer, nullable=True)
    true_negatives   = Column(Integer, nullable=True)
    false_positives  = Column(Integer, nullable=True)
    false_negatives  = Column(Integer, nullable=True)
    training_epochs  = Column(Integer, nullable=True)
    training_samples = Column(Integer, nullable=True)
    val_samples      = Column(Integer, nullable=True)
    test_samples     = Column(Integer, nullable=True)
    hyperparameters  = Column(Text, nullable=True)  # JSON string
    is_active        = Column(Boolean, default=True)
    is_production    = Column(Boolean, default=False)
    notes            = Column(Text, nullable=True)
    trained_at       = Column(DateTime, default=datetime.utcnow)
    deployed_at      = Column(DateTime, nullable=True)

    predictions      = relationship("Prediction", back_populates="model_version")

    def to_dict(self) -> Dict:
        return {
            "id":            self.id,
            "model_name":    self.model_name,
            "version":       self.version,
            "architecture":  self.architecture,
            "accuracy":      self.accuracy,
            "recall":        self.recall,
            "f1_score":      self.f1_score,
            "roc_auc":       self.roc_auc,
            "is_production": self.is_production,
            "trained_at":    str(self.trained_at)
        }


class AuditLog(Base):
    """
    Tracks all system events for compliance and debugging.
    """
    __tablename__ = "audit_logs"

    id          = Column(Integer, primary_key=True, autoincrement=True)
    event_type  = Column(String(100), nullable=False)  # prediction, upload, etc.
    entity_type = Column(String(50), nullable=True)    # patient, model, etc.
    entity_id   = Column(String(50), nullable=True)
    user        = Column(String(100), nullable=True)
    action      = Column(String(200), nullable=False)
    details     = Column(Text, nullable=True)           # JSON
    ip_address  = Column(String(50), nullable=True)
    status      = Column(String(20), default="success") # success/error
    error_msg   = Column(Text, nullable=True)
    created_at  = Column(DateTime, default=datetime.utcnow)


# ─────────────────────────────────────────────────────────────────────────────
# Database Initialization
# ─────────────────────────────────────────────────────────────────────────────
def initialize_database():
    """
    Create all tables in the database.
    Safe to call multiple times (CREATE IF NOT EXISTS).
    """
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created/verified successfully")
    logger.info("Tables: patients, predictions, image_metadata, "
                "model_versions, audit_logs")


# ─────────────────────────────────────────────────────────────────────────────
# CRUD Operations — Patients
# ─────────────────────────────────────────────────────────────────────────────
def create_patient(
    name: str,
    age: Optional[int] = None,
    gender: Optional[str] = None,
    hospital: Optional[str] = None,
    department: Optional[str] = None,
    doctor: Optional[str] = None,
    contact: Optional[str] = None
) -> Patient:
    """
    Create a new patient record.

    Returns:
        Patient: Created patient ORM object
    """
    session = Session()
    try:
        patient_id = f"PAT-{uuid.uuid4().hex[:8].upper()}"
        patient = Patient(
            patient_id=patient_id,
            name=name, age=age, gender=gender,
            hospital=hospital, department=department,
            doctor=doctor, contact=contact
        )
        session.add(patient)
        session.commit()
        session.refresh(patient)
        logger.info(f"Patient created: {patient_id} - {name}")

        log_audit("patient_created", "patient", patient_id,
                  f"New patient registered: {name}")
        return patient

    except IntegrityError:
        session.rollback()
        logger.error(f"Patient creation failed (duplicate?): {name}")
        raise
    finally:
        session.close()


def get_patient_by_id(patient_id: str) -> Optional[Patient]:
    """Retrieve patient by business ID (PAT-XXXXXXXX)."""
    session = Session()
    try:
        return session.query(Patient).filter(
            Patient.patient_id == patient_id
        ).first()
    finally:
        session.close()


def get_all_patients(limit: int = 1000) -> List[Dict]:
    """Retrieve all patients as list of dicts."""
    session = Session()
    try:
        patients = session.query(Patient).filter(
            Patient.is_active == True
        ).order_by(Patient.created_at.desc()).limit(limit).all()
        return [p.to_dict() for p in patients]
    finally:
        session.close()


# ─────────────────────────────────────────────────────────────────────────────
# CRUD Operations — Predictions
# ─────────────────────────────────────────────────────────────────────────────
def save_prediction(
    patient_db_id: int,
    prediction_label: str,
    confidence: float,
    raw_probability: float,
    image_path: Optional[str] = None,
    image_hash: Optional[str] = None,
    model_version_id: Optional[int] = None,
    threshold_used: float = 0.5,
    notes: Optional[str] = None
) -> Prediction:
    """
    Save a model prediction to the database.

    Args:
        patient_db_id (int): Database ID of the patient
        prediction_label (str): 'NORMAL' or 'PNEUMONIA'
        confidence (float): Prediction confidence [0, 1]
        raw_probability (float): Raw model output probability
        image_path (str): Path to the uploaded image
        image_hash (str): SHA256 hash of the image
        model_version_id (int): FK to model_versions table
        threshold_used (float): Decision threshold used
        notes (str): Optional clinical notes

    Returns:
        Prediction: Saved prediction ORM object
    """
    session = Session()
    try:
        pred_id = f"PRED-{uuid.uuid4().hex[:10].upper()}"
        prediction = Prediction(
            prediction_id=pred_id,
            patient_id=patient_db_id,
            model_version_id=model_version_id,
            image_path=image_path,
            image_hash=image_hash,
            prediction_label=prediction_label,
            confidence=round(confidence, 6),
            raw_probability=round(raw_probability, 6),
            threshold_used=threshold_used,
            notes=notes
        )
        session.add(prediction)
        session.commit()
        session.refresh(prediction)

        logger.info(
            f"Prediction saved: {pred_id} | "
            f"{prediction_label} ({confidence:.2%}) | "
            f"Patient DB ID: {patient_db_id}"
        )

        log_audit("prediction_made", "prediction", pred_id,
                  f"Prediction: {prediction_label} conf={confidence:.4f}")
        return prediction

    except Exception as e:
        session.rollback()
        logger.error(f"Failed to save prediction: {e}")
        raise
    finally:
        session.close()


def get_patient_predictions(patient_db_id: int) -> List[Dict]:
    """Get all predictions for a given patient."""
    session = Session()
    try:
        preds = session.query(Prediction).filter(
            Prediction.patient_id == patient_db_id
        ).order_by(Prediction.predicted_at.desc()).all()
        return [p.to_dict() for p in preds]
    finally:
        session.close()


def get_recent_predictions(limit: int = 100) -> pd.DataFrame:
    """Get recent predictions as a DataFrame for analytics."""
    session = Session()
    try:
        result = session.execute(text("""
            SELECT
                pr.prediction_id,
                pr.prediction_label,
                pr.confidence,
                pr.raw_probability,
                pr.predicted_at,
                pr.is_correct,
                pa.patient_id,
                pa.name      AS patient_name,
                pa.age,
                pa.gender,
                pa.hospital,
                mv.model_name,
                mv.version   AS model_version
            FROM predictions pr
            JOIN patients     pa ON pr.patient_id       = pa.id
            LEFT JOIN model_versions mv ON pr.model_version_id = mv.id
            ORDER BY pr.predicted_at DESC
            LIMIT :limit
        """), {"limit": limit})

        rows = result.fetchall()
        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows, columns=result.keys())
        return df
    finally:
        session.close()


# ─────────────────────────────────────────────────────────────────────────────
# CRUD Operations — Model Versions
# ─────────────────────────────────────────────────────────────────────────────
def register_model_version(
    model_name: str,
    architecture: str,
    metrics: Dict,
    file_path: Optional[str] = None,
    hyperparameters: Optional[Dict] = None,
    training_epochs: Optional[int] = None,
    notes: Optional[str] = None
) -> ModelVersion:
    """
    Register a trained model version with its performance metrics.

    Args:
        model_name (str): Model identifier (e.g., 'densenet121')
        architecture (str): Architecture description
        metrics (dict): Evaluation metrics from evaluate_model()
        file_path (str): Path to saved .h5 file
        hyperparameters (dict): Training hyperparameters
        training_epochs (int): Actual epochs trained
        notes (str): Additional notes

    Returns:
        ModelVersion: Created model version record
    """
    session = Session()
    try:
        # Get version number (auto-increment per model name)
        count = session.query(ModelVersion).filter(
            ModelVersion.model_name == model_name
        ).count()
        version = f"v{count + 1}.0"

        mv = ModelVersion(
            model_name=model_name,
            version=version,
            architecture=architecture,
            file_path=file_path,
            accuracy=metrics.get("accuracy"),
            precision=metrics.get("precision"),
            recall=metrics.get("recall"),
            f1_score=metrics.get("f1_score"),
            roc_auc=metrics.get("roc_auc"),
            specificity=metrics.get("specificity"),
            sensitivity=metrics.get("sensitivity"),
            true_positives=metrics.get("true_positive"),
            true_negatives=metrics.get("true_negative"),
            false_positives=metrics.get("false_positive"),
            false_negatives=metrics.get("false_negative"),
            training_epochs=training_epochs,
            hyperparameters=json.dumps(hyperparameters) if hyperparameters else None,
            notes=notes,
            is_active=True
        )

        session.add(mv)
        session.commit()
        session.refresh(mv)
        logger.info(f"Model version registered: {model_name} {version}")
        return mv

    except Exception as e:
        session.rollback()
        logger.error(f"Failed to register model version: {e}")
        raise
    finally:
        session.close()


def get_active_model_versions() -> List[Dict]:
    """Get all active model versions."""
    session = Session()
    try:
        versions = session.query(ModelVersion).filter(
            ModelVersion.is_active == True
        ).order_by(ModelVersion.trained_at.desc()).all()
        return [v.to_dict() for v in versions]
    finally:
        session.close()


def get_best_model(metric: str = "recall") -> Optional[Dict]:
    """Get the best performing model by a given metric."""
    session = Session()
    try:
        versions = session.query(ModelVersion).filter(
            ModelVersion.is_active == True
        ).all()
        if not versions:
            return None

        best = max(versions, key=lambda v: getattr(v, metric, 0) or 0)
        return best.to_dict()
    finally:
        session.close()


# ─────────────────────────────────────────────────────────────────────────────
# Audit Logging
# ─────────────────────────────────────────────────────────────────────────────
def log_audit(
    event_type: str,
    entity_type: Optional[str] = None,
    entity_id: Optional[str] = None,
    action: str = "",
    details: Optional[Dict] = None,
    status: str = "success",
    error_msg: Optional[str] = None
):
    """Log an audit event to the database."""
    session = Session()
    try:
        log = AuditLog(
            event_type=event_type,
            entity_type=entity_type,
            entity_id=str(entity_id) if entity_id else None,
            action=action,
            details=json.dumps(details) if details else None,
            status=status,
            error_msg=error_msg
        )
        session.add(log)
        session.commit()
    except Exception as e:
        session.rollback()
        logger.warning(f"Audit log failed: {e}")
    finally:
        session.close()


# ─────────────────────────────────────────────────────────────────────────────
# Analytics Queries
# ─────────────────────────────────────────────────────────────────────────────
def get_disease_rate_by_gender() -> pd.DataFrame:
    """Disease rate analysis by gender."""
    session = Session()
    try:
        result = session.execute(text("""
            SELECT
                pa.gender,
                COUNT(*) AS total_predictions,
                SUM(CASE WHEN pr.prediction_label = 'PNEUMONIA' THEN 1 ELSE 0 END)
                    AS pneumonia_count,
                ROUND(
                    100.0 * SUM(CASE WHEN pr.prediction_label = 'PNEUMONIA' THEN 1 ELSE 0 END)
                    / COUNT(*), 2
                ) AS pneumonia_rate_pct
            FROM predictions pr
            JOIN patients pa ON pr.patient_id = pa.id
            WHERE pa.gender IS NOT NULL
            GROUP BY pa.gender
            ORDER BY pneumonia_rate_pct DESC
        """))
        rows = result.fetchall()
        return pd.DataFrame(rows, columns=result.keys()) if rows else pd.DataFrame()
    finally:
        session.close()


def get_disease_rate_by_age_group() -> pd.DataFrame:
    """Disease rate analysis by age group."""
    session = Session()
    try:
        result = session.execute(text("""
            SELECT
                CASE
                    WHEN pa.age < 18  THEN '0-17 (Child)'
                    WHEN pa.age < 40  THEN '18-39 (Young Adult)'
                    WHEN pa.age < 60  THEN '40-59 (Middle Aged)'
                    WHEN pa.age < 80  THEN '60-79 (Senior)'
                    ELSE '80+ (Elderly)'
                END AS age_group,
                COUNT(*) AS total_predictions,
                SUM(CASE WHEN pr.prediction_label = 'PNEUMONIA' THEN 1 ELSE 0 END)
                    AS pneumonia_count,
                ROUND(
                    100.0 * SUM(CASE WHEN pr.prediction_label = 'PNEUMONIA' THEN 1 ELSE 0 END)
                    / COUNT(*), 2
                ) AS pneumonia_rate_pct
            FROM predictions pr
            JOIN patients pa ON pr.patient_id = pa.id
            WHERE pa.age IS NOT NULL
            GROUP BY age_group
            ORDER BY pneumonia_rate_pct DESC
        """))
        rows = result.fetchall()
        return pd.DataFrame(rows, columns=result.keys()) if rows else pd.DataFrame()
    finally:
        session.close()


def get_disease_rate_by_hospital() -> pd.DataFrame:
    """Disease rate analysis by hospital."""
    session = Session()
    try:
        result = session.execute(text("""
            SELECT
                pa.hospital,
                COUNT(*) AS total_predictions,
                SUM(CASE WHEN pr.prediction_label = 'PNEUMONIA' THEN 1 ELSE 0 END)
                    AS pneumonia_count,
                ROUND(
                    100.0 * SUM(CASE WHEN pr.prediction_label = 'PNEUMONIA' THEN 1 ELSE 0 END)
                    / COUNT(*), 2
                ) AS pneumonia_rate_pct,
                ROUND(AVG(pr.confidence), 4) AS avg_confidence
            FROM predictions pr
            JOIN patients pa ON pr.patient_id = pa.id
            WHERE pa.hospital IS NOT NULL
            GROUP BY pa.hospital
            ORDER BY total_predictions DESC
        """))
        rows = result.fetchall()
        return pd.DataFrame(rows, columns=result.keys()) if rows else pd.DataFrame()
    finally:
        session.close()


def get_monthly_trends() -> pd.DataFrame:
    """Monthly prediction and disease trends."""
    session = Session()
    try:
        result = session.execute(text("""
            SELECT
                strftime('%Y-%m', pr.predicted_at) AS month,
                COUNT(*) AS total_predictions,
                SUM(CASE WHEN pr.prediction_label = 'PNEUMONIA' THEN 1 ELSE 0 END)
                    AS pneumonia_count,
                SUM(CASE WHEN pr.prediction_label = 'NORMAL'    THEN 1 ELSE 0 END)
                    AS normal_count,
                ROUND(AVG(pr.confidence), 4) AS avg_confidence
            FROM predictions pr
            GROUP BY month
            ORDER BY month ASC
        """))
        rows = result.fetchall()
        return pd.DataFrame(rows, columns=result.keys()) if rows else pd.DataFrame()
    finally:
        session.close()


def get_model_performance_trends() -> pd.DataFrame:
    """Model performance metrics over versions."""
    session = Session()
    try:
        result = session.execute(text("""
            SELECT
                model_name, version,
                accuracy, precision, recall, f1_score, roc_auc,
                trained_at
            FROM model_versions
            WHERE is_active = 1
            ORDER BY trained_at ASC
        """))
        rows = result.fetchall()
        return pd.DataFrame(rows, columns=result.keys()) if rows else pd.DataFrame()
    finally:
        session.close()


def get_dashboard_kpis() -> Dict:
    """Get high-level KPIs for the executive dashboard."""
    session = Session()
    try:
        r = session.execute(text("""
            SELECT
                COUNT(DISTINCT pa.id)     AS total_patients,
                COUNT(pr.id)              AS total_predictions,
                SUM(CASE WHEN pr.prediction_label = 'PNEUMONIA' THEN 1 ELSE 0 END)
                    AS pneumonia_cases,
                ROUND(100.0 *
                    SUM(CASE WHEN pr.prediction_label = 'PNEUMONIA' THEN 1 ELSE 0 END)
                    / NULLIF(COUNT(pr.id), 0), 2) AS pneumonia_rate_pct,
                ROUND(AVG(pr.confidence), 4) AS avg_confidence,
                COUNT(DISTINCT mv.id)     AS model_versions
            FROM predictions pr
            JOIN patients pa ON pr.patient_id = pa.id
            LEFT JOIN model_versions mv ON pr.model_version_id = mv.id
        """)).fetchone()

        kpis = {
            "total_patients":     r[0] or 0,
            "total_predictions":  r[1] or 0,
            "pneumonia_cases":    r[2] or 0,
            "pneumonia_rate_pct": r[3] or 0.0,
            "avg_confidence":     r[4] or 0.0,
            "model_versions":     r[5] or 0
        }
        return kpis
    finally:
        session.close()


# ─────────────────────────────────────────────────────────────────────────────
# Sample Data Generator (for testing & demo)
# ─────────────────────────────────────────────────────────────────────────────
def seed_sample_data(n_patients: int = 200):
    """
    Seed database with realistic sample data for demo and testing.

    Args:
        n_patients (int): Number of patients to create
    """
    logger.info(f"Seeding database with {n_patients} sample patients...")

    hospitals   = ["City General Hospital", "St. Mary Medical Center",
                   "Apollo Diagnostics", "Metro Health Clinic",
                   "National Medical Institute"]
    departments = ["Radiology", "Emergency", "Pulmonology", "ICU", "OPD"]
    doctors     = ["Dr. Smith", "Dr. Patel", "Dr. Kumar", "Dr. Williams",
                   "Dr. Chen", "Dr. Singh"]
    genders     = ["Male", "Female", "Other"]
    first_names = ["Rahul", "Priya", "Amit", "Sunita", "Vikram", "Deepa",
                   "John", "Sarah", "Michael", "Emma", "James", "Aisha"]
    last_names  = ["Sharma", "Patel", "Kumar", "Singh", "Williams", "Johnson",
                   "Brown", "Davis", "Wilson", "Lee", "Chen", "Gupta"]

    # Get or create a default model version for seeding
    session = Session()
    model_v = session.query(ModelVersion).first()
    if not model_v:
        model_v = ModelVersion(
            model_name="densenet121",
            version="v1.0",
            architecture="DenseNet121 Transfer Learning",
            accuracy=0.9423, precision=0.9156, recall=0.9701,
            f1_score=0.9421, roc_auc=0.9789,
            specificity=0.9023, sensitivity=0.9701,
            is_active=True, is_production=True
        )
        session.add(model_v)
        session.commit()
        session.refresh(model_v)
    model_v_id = model_v.id
    session.close()

    # Create patients and predictions
    for i in range(n_patients):
        name = f"{random.choice(first_names)} {random.choice(last_names)}"
        age  = random.randint(18, 85)

        patient = create_patient(
            name=name,
            age=age,
            gender=random.choice(genders),
            hospital=random.choice(hospitals),
            department=random.choice(departments),
            doctor=random.choice(doctors)
        )

        # Each patient gets 1–3 predictions
        for _ in range(random.randint(1, 3)):
            label = random.choices(
                ["NORMAL", "PNEUMONIA"],
                weights=[35, 65]  # Reflect dataset imbalance
            )[0]

            if label == "PNEUMONIA":
                prob = random.uniform(0.55, 0.99)
            else:
                prob = random.uniform(0.01, 0.45)

            conf = abs(prob - 0.5) * 2  # Confidence from raw probability

            # Backdate predictions over the last 6 months
            days_back = random.randint(0, 180)
            pred_time = datetime.utcnow() - timedelta(days=days_back)

            session = Session()
            try:
                pred = Prediction(
                    prediction_id=f"PRED-{uuid.uuid4().hex[:10].upper()}",
                    patient_id=patient.id,
                    model_version_id=model_v_id,
                    prediction_label=label,
                    confidence=round(conf, 4),
                    raw_probability=round(prob, 4),
                    threshold_used=0.5,
                    predicted_at=pred_time
                )
                session.add(pred)
                session.commit()
            except Exception:
                session.rollback()
            finally:
                session.close()

    logger.info(f"Sample data seeded: {n_patients} patients")


# ─────────────────────────────────────────────────────────────────────────────
# Main Entry Point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    initialize_database()
    seed_sample_data(200)

    kpis = get_dashboard_kpis()
    logger.info(f"Dashboard KPIs: {kpis}")

    gender_df = get_disease_rate_by_gender()
    logger.info(f"\nDisease rate by gender:\n{gender_df}")

    monthly_df = get_monthly_trends()
    logger.info(f"\nMonthly trends:\n{monthly_df}")

    logger.info("Database setup and seeding COMPLETE ✓")
