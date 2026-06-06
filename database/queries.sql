-- ============================================================
-- MEDICAL AI PLATFORM - REPORTING QUERIES
-- ============================================================
-- Collection of ready-to-run analytical SQL queries
-- Compatible with SQLite and PostgreSQL


-- ────────────────────────────────────────────────────────────
-- SECTION 1: BASIC CRUD OPERATIONS
-- ────────────────────────────────────────────────────────────

-- 1.1 Insert a new patient
INSERT INTO patients (patient_id, name, age, gender, hospital, department, doctor)
VALUES ('PAT-00000001', 'John Doe', 45, 'Male', 'City General Hospital', 'Radiology', 'Dr. Smith');

-- 1.2 Get all active patients
SELECT patient_id, name, age, gender, hospital, created_at
FROM patients
WHERE is_active = 1
ORDER BY created_at DESC
LIMIT 50;

-- 1.3 Search patient by name
SELECT * FROM patients
WHERE name LIKE '%Doe%' AND is_active = 1;

-- 1.4 Get patient with all predictions
SELECT
    p.patient_id, p.name, p.age, p.gender,
    pr.prediction_id, pr.prediction_label,
    pr.confidence, pr.predicted_at
FROM patients p
JOIN predictions pr ON p.id = pr.patient_id
WHERE p.patient_id = 'PAT-00000001'
ORDER BY pr.predicted_at DESC;

-- 1.5 Update patient info
UPDATE patients
SET hospital = 'New Hospital', updated_at = CURRENT_TIMESTAMP
WHERE patient_id = 'PAT-00000001';

-- 1.6 Soft delete patient (never hard delete in medical systems)
UPDATE patients SET is_active = 0 WHERE patient_id = 'PAT-00000001';


-- ────────────────────────────────────────────────────────────
-- SECTION 2: PREDICTION ANALYTICS
-- ────────────────────────────────────────────────────────────

-- 2.1 Overall prediction distribution
SELECT
    prediction_label,
    COUNT(*)                          AS count,
    ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM predictions), 2) AS percentage,
    ROUND(AVG(confidence), 4)         AS avg_confidence,
    ROUND(MIN(confidence), 4)         AS min_confidence,
    ROUND(MAX(confidence), 4)         AS max_confidence
FROM predictions
GROUP BY prediction_label;

-- 2.2 High confidence pneumonia cases (urgent review)
SELECT
    pr.prediction_id,
    p.name AS patient_name, p.age, p.gender, p.hospital,
    pr.confidence, pr.predicted_at
FROM predictions pr
JOIN patients p ON pr.patient_id = p.id
WHERE pr.prediction_label = 'PNEUMONIA'
  AND pr.confidence >= 0.90
ORDER BY pr.confidence DESC, pr.predicted_at DESC
LIMIT 20;

-- 2.3 Low confidence predictions (borderline cases requiring review)
SELECT
    pr.prediction_id,
    p.name, p.hospital,
    pr.prediction_label,
    pr.confidence,
    pr.raw_probability,
    pr.predicted_at
FROM predictions pr
JOIN patients p ON pr.patient_id = p.id
WHERE pr.confidence < 0.65
ORDER BY pr.confidence ASC
LIMIT 20;

-- 2.4 Recent predictions (last 24 hours)
SELECT
    p.name, p.hospital,
    pr.prediction_label, pr.confidence, pr.predicted_at
FROM predictions pr
JOIN patients p ON pr.patient_id = p.id
WHERE pr.predicted_at >= datetime('now', '-1 day')
ORDER BY pr.predicted_at DESC;


-- ────────────────────────────────────────────────────────────
-- SECTION 3: DEMOGRAPHIC ANALYTICS
-- ────────────────────────────────────────────────────────────

-- 3.1 Disease rate by gender
SELECT
    p.gender,
    COUNT(*)                                                             AS total,
    SUM(CASE WHEN pr.prediction_label='PNEUMONIA' THEN 1 ELSE 0 END)    AS pneumonia,
    ROUND(
        100.0 * SUM(CASE WHEN pr.prediction_label='PNEUMONIA' THEN 1 ELSE 0 END)
        / COUNT(*), 2
    )                                                                    AS pneumonia_rate_pct
FROM predictions pr
JOIN patients p ON pr.patient_id = p.id
WHERE p.gender IS NOT NULL
GROUP BY p.gender
ORDER BY pneumonia_rate_pct DESC;

-- 3.2 Disease rate by age group
SELECT
    CASE
        WHEN p.age < 18  THEN '0-17 Child'
        WHEN p.age < 40  THEN '18-39 Young Adult'
        WHEN p.age < 60  THEN '40-59 Middle Aged'
        WHEN p.age < 80  THEN '60-79 Senior'
        ELSE '80+ Elderly'
    END AS age_group,
    COUNT(*)                                                           AS total,
    SUM(CASE WHEN pr.prediction_label='PNEUMONIA' THEN 1 ELSE 0 END)  AS pneumonia,
    ROUND(
        100.0 * SUM(CASE WHEN pr.prediction_label='PNEUMONIA' THEN 1 ELSE 0 END)
        / COUNT(*), 2
    )                                                                  AS pneumonia_rate_pct,
    ROUND(AVG(p.age), 1)                                               AS avg_age
FROM predictions pr
JOIN patients p ON pr.patient_id = p.id
WHERE p.age IS NOT NULL
GROUP BY age_group
ORDER BY pneumonia_rate_pct DESC;

-- 3.3 Disease rate by hospital
SELECT
    p.hospital,
    COUNT(*)                                                           AS total_predictions,
    COUNT(DISTINCT p.id)                                               AS unique_patients,
    SUM(CASE WHEN pr.prediction_label='PNEUMONIA' THEN 1 ELSE 0 END)  AS pneumonia_cases,
    ROUND(
        100.0 * SUM(CASE WHEN pr.prediction_label='PNEUMONIA' THEN 1 ELSE 0 END)
        / COUNT(*), 2
    )                                                                  AS pneumonia_rate_pct,
    ROUND(AVG(pr.confidence), 4)                                       AS avg_confidence
FROM predictions pr
JOIN patients p ON pr.patient_id = p.id
WHERE p.hospital IS NOT NULL
GROUP BY p.hospital
ORDER BY total_predictions DESC;


-- ────────────────────────────────────────────────────────────
-- SECTION 4: TIME SERIES ANALYTICS
-- ────────────────────────────────────────────────────────────

-- 4.1 Monthly disease trends
SELECT
    strftime('%Y-%m', pr.predicted_at)                                 AS month,
    COUNT(*)                                                           AS total,
    SUM(CASE WHEN pr.prediction_label='PNEUMONIA' THEN 1 ELSE 0 END)  AS pneumonia,
    SUM(CASE WHEN pr.prediction_label='NORMAL'    THEN 1 ELSE 0 END)  AS normal,
    ROUND(
        100.0 * SUM(CASE WHEN pr.prediction_label='PNEUMONIA' THEN 1 ELSE 0 END)
        / COUNT(*), 2
    )                                                                  AS pneumonia_rate_pct,
    ROUND(AVG(pr.confidence), 4)                                       AS avg_confidence
FROM predictions pr
GROUP BY month
ORDER BY month ASC;

-- 4.2 Daily prediction count (last 30 days)
SELECT
    date(pr.predicted_at)                                              AS prediction_date,
    COUNT(*)                                                           AS total_predictions,
    SUM(CASE WHEN pr.prediction_label='PNEUMONIA' THEN 1 ELSE 0 END)  AS pneumonia_count
FROM predictions pr
WHERE pr.predicted_at >= date('now', '-30 days')
GROUP BY prediction_date
ORDER BY prediction_date ASC;

-- 4.3 Weekly summary
SELECT
    strftime('%Y-W%W', pr.predicted_at)                                AS week,
    COUNT(*)                                                           AS total,
    SUM(CASE WHEN pr.prediction_label='PNEUMONIA' THEN 1 ELSE 0 END)  AS pneumonia,
    COUNT(DISTINCT pr.patient_id)                                      AS unique_patients
FROM predictions pr
GROUP BY week
ORDER BY week ASC;


-- ────────────────────────────────────────────────────────────
-- SECTION 5: MODEL PERFORMANCE ANALYTICS
-- ────────────────────────────────────────────────────────────

-- 5.1 All model versions ranked by recall
SELECT
    model_name, version,
    ROUND(accuracy  * 100, 2) AS accuracy_pct,
    ROUND(precision * 100, 2) AS precision_pct,
    ROUND(recall    * 100, 2) AS recall_pct,      -- MOST IMPORTANT
    ROUND(f1_score  * 100, 2) AS f1_pct,
    ROUND(roc_auc   * 100, 2) AS roc_auc_pct,
    is_production,
    trained_at
FROM model_versions
WHERE is_active = 1
ORDER BY recall DESC;

-- 5.2 Best model by each metric
SELECT 'Best Accuracy'   AS metric, model_name, version, accuracy  AS value
FROM model_versions WHERE accuracy  = (SELECT MAX(accuracy)  FROM model_versions WHERE is_active=1)
UNION ALL
SELECT 'Best Recall',              model_name, version, recall
FROM model_versions WHERE recall   = (SELECT MAX(recall)    FROM model_versions WHERE is_active=1)
UNION ALL
SELECT 'Best F1 Score',            model_name, version, f1_score
FROM model_versions WHERE f1_score = (SELECT MAX(f1_score)  FROM model_versions WHERE is_active=1)
UNION ALL
SELECT 'Best ROC-AUC',             model_name, version, roc_auc
FROM model_versions WHERE roc_auc  = (SELECT MAX(roc_auc)   FROM model_versions WHERE is_active=1);

-- 5.3 Predictions per model version
SELECT
    mv.model_name, mv.version,
    COUNT(pr.id)    AS total_predictions,
    ROUND(AVG(pr.confidence), 4) AS avg_confidence,
    SUM(CASE WHEN pr.prediction_label='PNEUMONIA' THEN 1 ELSE 0 END) AS pneumonia_count
FROM model_versions mv
LEFT JOIN predictions pr ON mv.id = pr.model_version_id
GROUP BY mv.id, mv.model_name, mv.version
ORDER BY total_predictions DESC;


-- ────────────────────────────────────────────────────────────
-- SECTION 6: EXECUTIVE KPI QUERY
-- ────────────────────────────────────────────────────────────

-- Complete executive dashboard KPIs (single query)
SELECT
    (SELECT COUNT(*) FROM patients WHERE is_active=1)              AS total_patients,
    (SELECT COUNT(*) FROM predictions)                             AS total_predictions,
    (SELECT COUNT(*) FROM predictions WHERE prediction_label='PNEUMONIA')
                                                                   AS pneumonia_cases,
    (SELECT COUNT(*) FROM predictions WHERE prediction_label='NORMAL')
                                                                   AS normal_cases,
    ROUND(
        100.0 * (SELECT COUNT(*) FROM predictions WHERE prediction_label='PNEUMONIA')
        / NULLIF((SELECT COUNT(*) FROM predictions), 0), 2
    )                                                              AS pneumonia_rate_pct,
    ROUND((SELECT AVG(confidence) FROM predictions), 4)            AS avg_confidence,
    (SELECT COUNT(*) FROM model_versions WHERE is_active=1)        AS registered_models,
    (SELECT model_name FROM model_versions WHERE is_production=1 LIMIT 1)
                                                                   AS production_model;
