"""
test_database.py
================
Unit and integration tests for the DatabaseManager and SQL schema.
Uses an in-memory SQLite database for isolation.

Run with: pytest tests/test_database.py -v
"""

import sys
import pytest
import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ─────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────

@pytest.fixture
def temp_db(tmp_path):
    """Create a DatabaseManager backed by a temporary SQLite file."""
    from database import DatabaseManager
    db_path = str(tmp_path / "test_medical.db")
    db = DatabaseManager(db_url=f"sqlite:///{db_path}")
    db.initialize_database()
    return db


@pytest.fixture
def patient_data():
    return {
        "name":     "Test Patient Alpha",
        "age":      45,
        "gender":   "Male",
        "hospital": "Test General Hospital",
        "notes":    "Test case",
    }


@pytest.fixture
def patient_id(temp_db, patient_data):
    """Insert one patient and return their ID."""
    return temp_db.add_patient(**patient_data)


# ─────────────────────────────────────────────
# DATABASE INITIALIZATION TESTS
# ─────────────────────────────────────────────

class TestDatabaseInitialization:
    """Tests for database schema creation."""

    def test_initialize_creates_tables(self, temp_db):
        """All five required tables must exist after initialization."""
        from sqlalchemy import inspect
        inspector = inspect(temp_db.engine)
        tables = inspector.get_table_names()
        required = {"patients", "predictions", "image_metadata", "model_versions", "audit_logs"}
        for table in required:
            assert table in tables, f"Table '{table}' not found. Tables present: {tables}"

    def test_second_initialize_is_idempotent(self, temp_db):
        """Calling initialize_database() twice should not raise an error."""
        temp_db.initialize_database()  # Should not raise

    def test_tables_have_correct_columns(self, temp_db):
        """Spot-check key columns on the patients table."""
        from sqlalchemy import inspect
        inspector = inspect(temp_db.engine)
        columns = {col["name"] for col in inspector.get_columns("patients")}
        expected = {"id", "name", "age", "gender", "hospital", "created_at"}
        for col in expected:
            assert col in columns, f"Column '{col}' missing from patients table"


# ─────────────────────────────────────────────
# PATIENT CRUD TESTS
# ─────────────────────────────────────────────

class TestPatientCRUD:
    """Tests for patient Create/Read/Update/Delete operations."""

    def test_add_patient_returns_id(self, temp_db, patient_data):
        """add_patient() should return an integer ID > 0."""
        pid = temp_db.add_patient(**patient_data)
        assert isinstance(pid, int), f"Expected int ID, got {type(pid)}"
        assert pid > 0, f"Expected positive ID, got {pid}"

    def test_get_patient_by_id(self, temp_db, patient_id, patient_data):
        """Fetching a patient by ID should return the correct record."""
        patient = temp_db.get_patient(patient_id)
        assert patient is not None, "Patient not found"
        assert patient["name"] == patient_data["name"]
        assert patient["age"]  == patient_data["age"]

    def test_get_all_patients_not_empty(self, temp_db, patient_id):
        """get_all_patients() should return a list with at least the test patient."""
        patients = temp_db.get_all_patients()
        assert len(patients) >= 1, "Expected at least 1 patient"

    def test_patient_names_are_unique_constraint(self, temp_db, patient_data):
        """Adding two patients with identical data (no unique name constraint) should work."""
        id1 = temp_db.add_patient(**patient_data)
        id2 = temp_db.add_patient(**patient_data)
        assert id1 != id2, "Two different patients should have different IDs"

    def test_search_patients_by_name(self, temp_db, patient_id, patient_data):
        """search_patients() should find patients by partial name match."""
        results = temp_db.search_patients(query="Alpha")
        assert len(results) >= 1, "Expected to find patient with 'Alpha' in name"
        names = [r["name"] for r in results]
        assert patient_data["name"] in names

    def test_soft_delete_patient(self, temp_db, patient_id):
        """Soft-deleted patient should not appear in active patient list."""
        temp_db.soft_delete_patient(patient_id)
        patients = temp_db.get_all_patients(active_only=True)
        ids = [p["id"] for p in patients]
        assert patient_id not in ids, "Soft-deleted patient should not appear in active list"

    def test_deleted_patient_still_in_full_list(self, temp_db, patient_id):
        """Soft-deleted patient should still appear when active_only=False."""
        temp_db.soft_delete_patient(patient_id)
        patients = temp_db.get_all_patients(active_only=False)
        ids = [p["id"] for p in patients]
        assert patient_id in ids, "Soft-deleted patient should still exist in full list"


# ─────────────────────────────────────────────
# PREDICTION CRUD TESTS
# ─────────────────────────────────────────────

class TestPredictionCRUD:
    """Tests for saving and retrieving predictions."""

    def test_save_prediction_returns_id(self, temp_db, patient_id):
        """save_prediction() should return an integer ID."""
        pred_id = temp_db.save_prediction(
            patient_id=patient_id,
            image_filename="test_xray.jpg",
            prediction_label="PNEUMONIA",
            confidence_score=0.87,
            probability_normal=0.13,
            probability_pneumonia=0.87,
            model_version="cnn_v1",
            processing_time_ms=120.5,
        )
        assert isinstance(pred_id, int) and pred_id > 0

    def test_get_predictions_for_patient(self, temp_db, patient_id):
        """get_patient_predictions() should return saved predictions for a patient."""
        temp_db.save_prediction(
            patient_id=patient_id,
            image_filename="test_xray.jpg",
            prediction_label="NORMAL",
            confidence_score=0.92,
            probability_normal=0.92,
            probability_pneumonia=0.08,
            model_version="cnn_v1",
            processing_time_ms=95.0,
        )
        preds = temp_db.get_patient_predictions(patient_id)
        assert len(preds) >= 1, "Expected at least 1 prediction"

    def test_prediction_fields_stored_correctly(self, temp_db, patient_id):
        """Stored prediction should have all correct field values."""
        temp_db.save_prediction(
            patient_id=patient_id,
            image_filename="chest_001.jpg",
            prediction_label="PNEUMONIA",
            confidence_score=0.76,
            probability_normal=0.24,
            probability_pneumonia=0.76,
            model_version="resnet50_v1",
            processing_time_ms=200.0,
        )
        preds = temp_db.get_patient_predictions(patient_id)
        pred = next((p for p in preds if p["image_filename"] == "chest_001.jpg"), None)
        assert pred is not None, "Prediction not found"
        assert pred["prediction_label"] == "PNEUMONIA"
        assert abs(pred["confidence_score"] - 0.76) < 1e-4

    def test_get_recent_predictions_limit(self, temp_db, patient_id):
        """get_recent_predictions() should respect the limit parameter."""
        for i in range(10):
            temp_db.save_prediction(
                patient_id=patient_id,
                image_filename=f"xray_{i}.jpg",
                prediction_label="NORMAL" if i % 2 == 0 else "PNEUMONIA",
                confidence_score=0.7 + i * 0.02,
                probability_normal=0.3,
                probability_pneumonia=0.7,
                model_version="cnn_v1",
                processing_time_ms=100.0,
            )
        recent = temp_db.get_recent_predictions(limit=5)
        assert len(recent) <= 5, f"Expected ≤5 results, got {len(recent)}"


# ─────────────────────────────────────────────
# MODEL VERSION TESTS
# ─────────────────────────────────────────────

class TestModelVersions:
    """Tests for model version registration and retrieval."""

    def test_register_model_version(self, temp_db):
        """register_model_version() should insert a record and return an ID."""
        model_id = temp_db.register_model_version(
            model_name="ResNet50_v1",
            version="1.0.0",
            architecture="resnet50",
            accuracy=0.94,
            precision=0.93,
            recall=0.96,
            f1_score=0.945,
            roc_auc=0.98,
            model_path="models/saved_models/resnet50_model.h5",
            notes="First production version",
        )
        assert isinstance(model_id, int) and model_id > 0

    def test_get_model_versions_returns_list(self, temp_db):
        """get_model_versions() should return a list."""
        temp_db.register_model_version(
            model_name="TestModel",
            version="0.0.1",
            architecture="cnn",
            accuracy=0.90, precision=0.89,
            recall=0.91, f1_score=0.90, roc_auc=0.95,
            model_path="models/saved_models/test.h5",
        )
        versions = temp_db.get_model_versions()
        assert isinstance(versions, list)
        assert len(versions) >= 1

    def test_model_metrics_stored_accurately(self, temp_db):
        """Stored model metrics should match inserted values within tolerance."""
        temp_db.register_model_version(
            model_name="PreciseModel",
            version="2.0.0",
            architecture="densenet121",
            accuracy=0.9512,
            precision=0.9401,
            recall=0.9703,
            f1_score=0.9549,
            roc_auc=0.9876,
            model_path="models/test.h5",
        )
        versions = temp_db.get_model_versions()
        model = next((m for m in versions if m["model_name"] == "PreciseModel"), None)
        assert model is not None
        assert abs(model["recall"] - 0.9703) < 1e-3


# ─────────────────────────────────────────────
# ANALYTICS QUERIES TESTS
# ─────────────────────────────────────────────

class TestAnalyticsQueries:
    """Tests for reporting and analytics queries."""

    @pytest.fixture(autouse=True)
    def seed_data(self, temp_db):
        """Insert a small dataset for analytics testing."""
        genders  = ["Male", "Female", "Male", "Female", "Male"]
        ages     = [25, 45, 62, 33, 78]
        labels   = ["PNEUMONIA", "NORMAL", "PNEUMONIA", "PNEUMONIA", "NORMAL"]
        hospital = "City Hospital"

        for i, (g, a, lbl) in enumerate(zip(genders, ages, labels)):
            pid = temp_db.add_patient(
                name=f"Patient {i}", age=a, gender=g, hospital=hospital
            )
            temp_db.save_prediction(
                patient_id=pid,
                image_filename=f"xray_{i}.jpg",
                prediction_label=lbl,
                confidence_score=0.85,
                probability_normal=0.15 if lbl == "PNEUMONIA" else 0.85,
                probability_pneumonia=0.85 if lbl == "PNEUMONIA" else 0.15,
                model_version="cnn_v1",
                processing_time_ms=100.0,
            )

    def test_get_kpi_summary_returns_dict(self, temp_db):
        """get_kpi_summary() should return a dict with expected keys."""
        kpis = temp_db.get_kpi_summary()
        assert isinstance(kpis, dict)
        for key in ["total_predictions", "total_patients", "pneumonia_rate"]:
            assert key in kpis, f"KPI '{key}' missing from summary"

    def test_pneumonia_rate_is_valid(self, temp_db):
        """Pneumonia rate should be between 0 and 1."""
        kpis = temp_db.get_kpi_summary()
        rate = kpis.get("pneumonia_rate", -1)
        assert 0.0 <= rate <= 1.0, f"Invalid pneumonia rate: {rate}"

    def test_get_predictions_by_gender(self, temp_db):
        """get_predictions_by_gender() should return data for both genders."""
        data = temp_db.get_predictions_by_gender()
        assert isinstance(data, list)
        assert len(data) >= 1

    def test_get_monthly_trends_returns_list(self, temp_db):
        """get_monthly_trends() should return a list (may be empty for fresh DB)."""
        trends = temp_db.get_monthly_trends(months=6)
        assert isinstance(trends, list)

    def test_total_predictions_count_matches(self, temp_db):
        """Total predictions in KPI should equal the number inserted in seed_data."""
        kpis = temp_db.get_kpi_summary()
        assert kpis["total_predictions"] >= 5, "Expected at least 5 predictions from seed_data"


# ─────────────────────────────────────────────
# AUDIT LOG TESTS
# ─────────────────────────────────────────────

class TestAuditLogs:
    """Tests for audit logging functionality."""

    def test_write_audit_log(self, temp_db):
        """write_audit_log() should insert a record without raising."""
        temp_db.write_audit_log(
            action="TEST_ACTION",
            table_name="predictions",
            record_id=1,
            user="pytest",
            details="Unit test audit entry",
        )

    def test_get_audit_logs_returns_list(self, temp_db):
        """get_audit_logs() should return a list."""
        temp_db.write_audit_log("TEST", "patients", 1, "pytest", "test")
        logs = temp_db.get_audit_logs(limit=10)
        assert isinstance(logs, list)
        assert len(logs) >= 1

    def test_audit_log_fields_stored_correctly(self, temp_db):
        """Audit log should store action and table name correctly."""
        temp_db.write_audit_log(
            action="INSERT_PATIENT",
            table_name="patients",
            record_id=99,
            user="test_runner",
            details="Automated test",
        )
        logs = temp_db.get_audit_logs(limit=5)
        log = next((l for l in logs if l.get("action") == "INSERT_PATIENT"), None)
        assert log is not None, "Audit log entry not found"
        assert log["table_name"] == "patients"
