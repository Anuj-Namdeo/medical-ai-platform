-- ============================================================
-- MEDICAL AI PLATFORM - DATABASE SCHEMA
-- SQLite Version (also compatible with PostgreSQL with minor changes)
-- ============================================================

-- ── Drop existing tables (for fresh setup) ──────────────────────────────────
DROP TABLE IF EXISTS audit_logs;
DROP TABLE IF EXISTS predictions;
DROP TABLE IF EXISTS image_metadata;
DROP TABLE IF EXISTS model_versions;
DROP TABLE IF EXISTS patients;


-- ── 1. PATIENTS TABLE ───────────────────────────────────────────────────────
-- Stores patient demographic and identification data
-- One patient can have many predictions (one-to-many)
CREATE TABLE patients (
    id           INTEGER  PRIMARY KEY AUTOINCREMENT,
    patient_id   TEXT     UNIQUE NOT NULL,        -- Business key: PAT-XXXXXXXX
    name         TEXT     NOT NULL,
    age          INTEGER,
    gender       TEXT     CHECK(gender IN ('Male','Female','Other')),
    hospital     TEXT,
    department   TEXT,
    doctor       TEXT,
    contact      TEXT,
    created_at   DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at   DATETIME DEFAULT CURRENT_TIMESTAMP,
    is_active    BOOLEAN  DEFAULT 1
);

CREATE INDEX idx_patients_patient_id ON patients(patient_id);
CREATE INDEX idx_patients_hospital   ON patients(hospital);
CREATE INDEX idx_patients_gender     ON patients(gender);


-- ── 2. MODEL VERSIONS TABLE ─────────────────────────────────────────────────
-- Tracks all trained model versions (Model Registry)
-- Enables MLOps: version control, performance comparison, rollback
CREATE TABLE model_versions (
    id                INTEGER  PRIMARY KEY AUTOINCREMENT,
    model_name        TEXT     NOT NULL,           -- custom_cnn, resnet50, etc.
    version           TEXT     NOT NULL,           -- v1.0, v2.0, etc.
    architecture      TEXT,                        -- Architecture description
    file_path         TEXT,                        -- Path to .h5 file
    accuracy          REAL,
    precision         REAL,
    recall            REAL,                        -- CRITICAL metric for medical AI
    f1_score          REAL,
    roc_auc           REAL,
    specificity       REAL,
    sensitivity       REAL,
    true_positives    INTEGER,
    true_negatives    INTEGER,
    false_positives   INTEGER,
    false_negatives   INTEGER,
    training_epochs   INTEGER,
    training_samples  INTEGER,
    val_samples       INTEGER,
    test_samples      INTEGER,
    hyperparameters   TEXT,                        -- JSON string
    is_active         BOOLEAN  DEFAULT 1,
    is_production     BOOLEAN  DEFAULT 0,          -- Currently deployed model
    notes             TEXT,
    trained_at        DATETIME DEFAULT CURRENT_TIMESTAMP,
    deployed_at       DATETIME
);

CREATE INDEX idx_model_versions_name       ON model_versions(model_name);
CREATE INDEX idx_model_versions_production ON model_versions(is_production);


-- ── 3. IMAGE METADATA TABLE ─────────────────────────────────────────────────
-- Stores technical properties of uploaded images
-- Enables analysis of image quality and dataset statistics
CREATE TABLE image_metadata (
    id           INTEGER  PRIMARY KEY AUTOINCREMENT,
    image_hash   TEXT     UNIQUE NOT NULL,         -- SHA256 for dedup
    filename     TEXT,
    file_size_kb REAL,
    width        INTEGER,
    height       INTEGER,
    channels     INTEGER,
    color_mode   TEXT,
    format       TEXT,                             -- JPEG, PNG, DICOM
    mean_pixel   REAL,
    std_pixel    REAL,
    created_at   DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_image_metadata_hash ON image_metadata(image_hash);


-- ── 4. PREDICTIONS TABLE ────────────────────────────────────────────────────
-- Stores every prediction made by the AI model
-- Central table linking patients, models, and images
CREATE TABLE predictions (
    id               INTEGER  PRIMARY KEY AUTOINCREMENT,
    prediction_id    TEXT     UNIQUE NOT NULL,     -- Business key: PRED-XXXXXXXXXX
    patient_id       INTEGER  NOT NULL REFERENCES patients(id),
    model_version_id INTEGER  REFERENCES model_versions(id),
    image_path       TEXT,
    image_hash       TEXT     REFERENCES image_metadata(image_hash),
    prediction_label TEXT     NOT NULL             -- NORMAL or PNEUMONIA
                     CHECK(prediction_label IN ('NORMAL','PNEUMONIA')),
    confidence       REAL     NOT NULL,            -- 0.0 to 1.0
    raw_probability  REAL     NOT NULL,            -- Model raw sigmoid output
    threshold_used   REAL     DEFAULT 0.5,
    is_correct       BOOLEAN,                      -- Doctor review result
    doctor_diagnosis TEXT,                         -- Actual diagnosis
    notes            TEXT,
    predicted_at     DATETIME DEFAULT CURRENT_TIMESTAMP,
    reviewed_at      DATETIME
);

CREATE INDEX idx_predictions_patient      ON predictions(patient_id);
CREATE INDEX idx_predictions_label        ON predictions(prediction_label);
CREATE INDEX idx_predictions_predicted_at ON predictions(predicted_at);
CREATE INDEX idx_predictions_model        ON predictions(model_version_id);


-- ── 5. AUDIT LOGS TABLE ─────────────────────────────────────────────────────
-- Tracks all system events for compliance, debugging, and monitoring
CREATE TABLE audit_logs (
    id          INTEGER  PRIMARY KEY AUTOINCREMENT,
    event_type  TEXT     NOT NULL,                 -- prediction, upload, login, etc.
    entity_type TEXT,                              -- patient, model, image
    entity_id   TEXT,
    user        TEXT,
    action      TEXT     NOT NULL,
    details     TEXT,                              -- JSON
    ip_address  TEXT,
    status      TEXT     DEFAULT 'success'
                CHECK(status IN ('success','error','warning')),
    error_msg   TEXT,
    created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_audit_logs_event_type ON audit_logs(event_type);
CREATE INDEX idx_audit_logs_created_at ON audit_logs(created_at);


-- ============================================================
-- REPORTING VIEWS
-- ============================================================

-- Executive Summary View
CREATE VIEW IF NOT EXISTS v_executive_summary AS
SELECT
    COUNT(DISTINCT p.id)                                AS total_patients,
    COUNT(pr.id)                                        AS total_predictions,
    SUM(CASE WHEN pr.prediction_label='PNEUMONIA' THEN 1 ELSE 0 END)
                                                        AS pneumonia_cases,
    SUM(CASE WHEN pr.prediction_label='NORMAL'    THEN 1 ELSE 0 END)
                                                        AS normal_cases,
    ROUND(
        100.0 * SUM(CASE WHEN pr.prediction_label='PNEUMONIA' THEN 1 ELSE 0 END)
        / NULLIF(COUNT(pr.id), 0), 2
    )                                                   AS pneumonia_rate_pct,
    ROUND(AVG(pr.confidence), 4)                        AS avg_confidence,
    COUNT(DISTINCT mv.id)                               AS model_versions
FROM predictions pr
JOIN patients     p  ON pr.patient_id       = p.id
LEFT JOIN model_versions mv ON pr.model_version_id = mv.id;


-- Patient Prediction Detail View
CREATE VIEW IF NOT EXISTS v_patient_prediction_detail AS
SELECT
    p.patient_id,
    p.name          AS patient_name,
    p.age,
    p.gender,
    p.hospital,
    p.department,
    p.doctor,
    pr.prediction_id,
    pr.prediction_label,
    ROUND(pr.confidence, 4)       AS confidence,
    ROUND(pr.raw_probability, 4)  AS raw_probability,
    pr.threshold_used,
    pr.is_correct,
    pr.doctor_diagnosis,
    pr.notes,
    pr.predicted_at,
    mv.model_name,
    mv.version AS model_version
FROM predictions  pr
JOIN patients     p  ON pr.patient_id       = p.id
LEFT JOIN model_versions mv ON pr.model_version_id = mv.id
ORDER BY pr.predicted_at DESC;


-- Monthly Trend View
CREATE VIEW IF NOT EXISTS v_monthly_trends AS
SELECT
    strftime('%Y-%m', pr.predicted_at)  AS month,
    COUNT(*)                             AS total_predictions,
    SUM(CASE WHEN pr.prediction_label='PNEUMONIA' THEN 1 ELSE 0 END)
                                         AS pneumonia_count,
    SUM(CASE WHEN pr.prediction_label='NORMAL'    THEN 1 ELSE 0 END)
                                         AS normal_count,
    ROUND(AVG(pr.confidence), 4)         AS avg_confidence,
    COUNT(DISTINCT pr.patient_id)        AS unique_patients
FROM predictions pr
GROUP BY month
ORDER BY month ASC;
