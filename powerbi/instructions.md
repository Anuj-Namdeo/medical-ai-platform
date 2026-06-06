# Power BI Implementation Guide
## Medical AI Platform — Chest X-Ray Analytics Dashboard

---

## OVERVIEW

This guide walks you through building a complete 5-page Power BI dashboard connected to the Medical AI Platform's SQLite/PostgreSQL database.

**Dashboard Pages:**
1. Executive Dashboard
2. Disease Analytics Dashboard  
3. Demographics Dashboard
4. Hospital Dashboard
5. Model Monitoring Dashboard

---

## PREREQUISITES

### Install Power BI Desktop
- Download: https://powerbi.microsoft.com/desktop
- Version required: 2.115 or later (supports SQLite)
- Operating System: Windows 10/11

### Install SQLite ODBC Driver
For SQLite connectivity:
1. Download: http://www.ch-werner.de/sqliteodbc/
2. Run `sqliteodbc_w64.exe`
3. Verify in ODBC Data Source Administrator

---

## STEP 1 — CONNECT TO DATABASE

### SQLite Connection
1. Open Power BI Desktop
2. Click **Get Data** → **ODBC**
3. In DSN, choose **SQLite3 Datasource**
4. Browse to: `<project_root>/database/medical_ai.db`
5. Click **Connect**

### PostgreSQL Connection (Production)
1. **Get Data** → **PostgreSQL database**
2. Server: `localhost` (or your server address)
3. Database: `medical_ai_db`
4. Enter credentials
5. Click **OK**

---

## STEP 2 — LOAD TABLES

In the Navigator window, select all 5 tables:
- ✅ `patients`
- ✅ `predictions`
- ✅ `image_metadata`
- ✅ `model_versions`
- ✅ `audit_logs`

Click **Transform Data** (do NOT click Load yet).

---

## STEP 3 — POWER QUERY TRANSFORMATIONS

### patients table
```
= Table.TransformColumnTypes(patients, {
    {"id", Int64.Type},
    {"age", Int64.Type},
    {"created_at", type datetime}
})
```

Add **Age Group** column:
```
= Table.AddColumn(patients, "AgeGroup", each
    if [age] < 12  then "Child (0-11)"
    else if [age] < 18  then "Adolescent (12-17)"
    else if [age] < 35  then "Young Adult (18-34)"
    else if [age] < 55  then "Middle Age (35-54)"
    else if [age] < 70  then "Senior (55-69)"
    else "Elderly (70+)"
)
```

### predictions table
```
= Table.TransformColumnTypes(predictions, {
    {"id", Int64.Type},
    {"patient_id", Int64.Type},
    {"confidence_score", type number},
    {"probability_pneumonia", type number},
    {"probability_normal", type number},
    {"created_at", type datetime}
})
```

Add **Month-Year** column:
```
= Table.AddColumn(predictions, "MonthYear",
    each Date.ToText(DateTime.Date([created_at]), "MMM yyyy"),
    type text
)
```

Add **Is Pneumonia** column (numeric for calculations):
```
= Table.AddColumn(predictions, "IsPneumonia",
    each if [prediction_label] = "PNEUMONIA" then 1 else 0,
    Int64.Type
)
```

Click **Close & Apply**.

---

## STEP 4 — DATA MODEL (RELATIONSHIPS)

In **Model view**, create these relationships:

| From Table | From Column | To Table | To Column | Cardinality |
|---|---|---|---|---|
| predictions | patient_id | patients | id | Many-to-One |
| image_metadata | prediction_id | predictions | id | Many-to-One |
| audit_logs | record_id | predictions | id | Many-to-One |

Set **Cross-filter direction** = Single for all relationships.

---

## STEP 5 — DAX MEASURES

Create a dedicated **Measures** table:  
Home → Enter Data → name it `_Measures` → Load

### Core KPI Measures

```dax
-- Total Predictions
Total Predictions = COUNTROWS(predictions)

-- Total Patients
Total Patients = DISTINCTCOUNT(predictions[patient_id])

-- Pneumonia Cases
Pneumonia Cases = CALCULATE(
    COUNTROWS(predictions),
    predictions[prediction_label] = "PNEUMONIA"
)

-- Normal Cases
Normal Cases = CALCULATE(
    COUNTROWS(predictions),
    predictions[prediction_label] = "NORMAL"
)

-- Pneumonia Rate
Pneumonia Rate = 
DIVIDE([Pneumonia Cases], [Total Predictions], 0)

-- Pneumonia Rate % (formatted)
Pneumonia Rate % = 
FORMAT([Pneumonia Rate], "0.0%")

-- Average Confidence
Avg Confidence = AVERAGE(predictions[confidence_score])

-- Average Confidence % (formatted)
Avg Confidence % = FORMAT([Avg Confidence], "0.0%")
```

### Time Intelligence Measures

```dax
-- Predictions Last 30 Days
Predictions Last 30 Days = 
CALCULATE(
    [Total Predictions],
    DATESINPERIOD(predictions[created_at], TODAY(), -30, DAY)
)

-- Predictions Last Month
Predictions Last Month = 
CALCULATE(
    [Total Predictions],
    PREVIOUSMONTH(predictions[created_at])
)

-- MoM Growth %
MoM Growth = 
DIVIDE(
    [Total Predictions] - [Predictions Last Month],
    [Predictions Last Month],
    0
)

-- 3-Month Rolling Average
Rolling 3M Predictions = 
AVERAGEX(
    DATESINPERIOD(
        predictions[created_at],
        LASTDATE(predictions[created_at]),
        -3, MONTH
    ),
    [Total Predictions]
)
```

### Model Performance Measures

```dax
-- Best Model Recall
Best Model Recall = 
MAXX(model_versions, model_versions[recall])

-- Best Model Name
Best Model Name = 
CALCULATE(
    FIRSTNONBLANK(model_versions[model_name], 1),
    model_versions[recall] = [Best Model Recall]
)

-- Average Model Accuracy
Avg Model Accuracy = AVERAGE(model_versions[accuracy])

-- Avg ROC-AUC
Avg ROC AUC = AVERAGE(model_versions[roc_auc])
```

### Demographic Measures

```dax
-- Male Pneumonia Rate
Male Pneumonia Rate = 
CALCULATE(
    [Pneumonia Rate],
    patients[gender] = "Male"
)

-- Female Pneumonia Rate
Female Pneumonia Rate = 
CALCULATE(
    [Pneumonia Rate],
    patients[gender] = "Female"
)

-- Senior Pneumonia Rate (Age 55+)
Senior Pneumonia Rate = 
CALCULATE(
    [Pneumonia Rate],
    patients[age] >= 55
)

-- Child Pneumonia Rate (Age < 12)
Child Pneumonia Rate = 
CALCULATE(
    [Pneumonia Rate],
    patients[age] < 12
)
```

### Confidence Bucketing

```dax
-- High Confidence Predictions (>= 85%)
High Confidence Count = 
CALCULATE(
    [Total Predictions],
    predictions[confidence_score] >= 0.85
)

-- Low Confidence Predictions (< 65%)
Low Confidence Count = 
CALCULATE(
    [Total Predictions],
    predictions[confidence_score] < 0.65
)

-- High Confidence Rate
High Confidence Rate = 
DIVIDE([High Confidence Count], [Total Predictions], 0)
```

---

## STEP 6 — CALCULATED COLUMNS

```dax
-- In predictions table:
Confidence Bucket = 
SWITCH(
    TRUE(),
    predictions[confidence_score] >= 0.90, "Very High (90%+)",
    predictions[confidence_score] >= 0.80, "High (80-90%)",
    predictions[confidence_score] >= 0.70, "Moderate (70-80%)",
    predictions[confidence_score] >= 0.60, "Low-Moderate (60-70%)",
    "Low (<60%)"
)

-- In patients table:
Risk Category = 
SWITCH(
    TRUE(),
    patients[age] >= 65, "High Risk (Elderly)",
    patients[age] < 5,   "High Risk (Infant)",
    "Standard Risk"
)
```

---

## PAGE 1 — EXECUTIVE DASHBOARD

### Layout (use blank canvas, 16:9)

**Row 1 — KPI Cards (5 cards)**  
Insert → Card visual for each:
- Total Predictions → [Total Predictions]
- Pneumonia Rate → [Pneumonia Rate %]
- Total Patients → [Total Patients]
- Avg Confidence → [Avg Confidence %]
- Predictions (30d) → [Predictions Last 30 Days]

**Card Formatting:**
- Background: #1E293B
- Font: Segoe UI, 28pt Bold
- Label: 10pt, #94A3B8
- No border

**Row 2 — Monthly Trends (Line Chart)**
- X Axis: predictions[MonthYear]
- Y Axis: [Total Predictions], [Pneumonia Cases]
- Legend: auto
- Title: "Monthly Prediction Volume & Pneumonia Cases"
- Colors: #2563EB (total), #EF4444 (pneumonia)

**Row 3 — Left: Donut Chart (Class Split)**
- Values: [Pneumonia Cases], [Normal Cases]
- Colors: #EF4444 (pneumonia), #22C55E (normal)
- Title: "Overall Diagnosis Split"

**Row 3 — Right: Bar Chart (Top 5 Hospitals)**
- Y Axis: patients[hospital]
- X Axis: [Total Predictions]
- Sort: Descending
- Title: "Top Hospitals by Case Volume"

**Page Filters:** Date slicer on predictions[created_at]

---

## PAGE 2 — DISEASE ANALYTICS DASHBOARD

**Visual 1: Stacked Bar Chart — Monthly Disease Breakdown**
- X: predictions[MonthYear]
- Y: Count of predictions
- Legend: predictions[prediction_label]
- Colors: #22C55E (NORMAL), #EF4444 (PNEUMONIA)

**Visual 2: Line Chart — Pneumonia Rate Over Time**
- X: predictions[MonthYear]
- Y: [Pneumonia Rate]
- Format Y as percentage
- Add constant line at 50% as reference

**Visual 3: Histogram — Confidence Distribution**
- Use clustered bar chart with Confidence Bucket column
- X: predictions[Confidence Bucket]
- Y: [Total Predictions]
- Title: "Prediction Confidence Distribution"

**Visual 4: Scatter Plot — Age vs Confidence**
- X: patients[age]
- Y: predictions[confidence_score]
- Legend: predictions[prediction_label]
- Title: "Age vs Prediction Confidence"

**Visual 5: Table — Recent Predictions**
Columns: patient name, age, label, confidence, date
Sort: descending by date
Conditional formatting: 
- Red background for PNEUMONIA rows (prediction_label = "PNEUMONIA")

---

## PAGE 3 — DEMOGRAPHICS DASHBOARD

**Visual 1: Bar Chart — Pneumonia Rate by Gender**
- X: patients[gender]
- Y: [Pneumonia Rate]
- Colors: #3B82F6 (Male), #EC4899 (Female), #A78BFA (Other)

**Visual 2: Bar Chart — Disease by Age Group**
- X: patients[AgeGroup]
- Y: [Total Predictions]
- Legend: predictions[prediction_label]
- Sort by age group order (add sort column 0-5 for each group)

**Visual 3: Donut — Gender Distribution**
- Values: DISTINCTCOUNT of patients[id] by gender

**Visual 4: KPI Cards (Demographic)**
- Senior Pneumonia Rate: [Senior Pneumonia Rate]
- Child Pneumonia Rate: [Child Pneumonia Rate]
- Male vs Female Rate: [Male Pneumonia Rate] vs [Female Pneumonia Rate]

**Visual 5: Matrix Table**
- Rows: patients[AgeGroup]
- Columns: patients[gender]
- Values: [Total Predictions], [Pneumonia Rate]
- Conditional formatting: background color scale (green→red)

---

## PAGE 4 — HOSPITAL DASHBOARD

**Visual 1: Bar Chart — Predictions by Hospital**
- X: [Total Predictions]
- Y: patients[hospital]
- Horizontal bars, sorted descending

**Visual 2: Bar Chart — Pneumonia Rate by Hospital**
- X: [Pneumonia Rate]
- Y: patients[hospital]
- Format X as percentage
- Conditional formatting: color by rate (red = high)

**Visual 3: Line Chart — Hospital Volume Trends**
- X: predictions[MonthYear]
- Y: [Total Predictions]
- Legend: patients[hospital]
- Small multiples optional

**Visual 4: KPI Table**
Columns: Hospital, Total Cases, Pneumonia Cases, Pneumonia Rate, Avg Confidence
Add conditional formatting to Pneumonia Rate column

**Slicer:** patients[hospital] (multi-select dropdown)

---

## PAGE 5 — MODEL MONITORING DASHBOARD

**Visual 1: Clustered Bar — Model Comparison**
- X: model_versions[model_name]
- Y: Accuracy, Precision, Recall, F1, ROC-AUC (all 5 series)
- Clustered bar
- Format Y axis 0-1
- Title: "Model Performance Comparison"

**Visual 2: Line Chart — Accuracy Over Model Versions**
- X: model_versions[version]
- Y: model_versions[accuracy], model_versions[recall]
- Title: "Model Performance Trend"

**Visual 3: KPI Cards (Model Metrics)**
- Best Recall: [Best Model Recall]
- Avg ROC-AUC: [Avg ROC AUC]
- Best Model: [Best Model Name]

**Visual 4: Table — All Model Versions**
Columns: name, version, accuracy, precision, recall, f1, roc_auc, created_at
Sort: recall descending
Conditional formatting: recall column (green = high recall)

**Visual 5: Gauge — Current Model Recall**
- Value: latest model recall
- Min: 0, Max: 1
- Target: 0.90
- Colors: green if > 0.90, amber if 0.80-0.90, red if < 0.80

---

## STEP 7 — PUBLISH TO POWER BI SERVICE

1. Sign in to Power BI Desktop with your Microsoft account
2. **Home** → **Publish** → Select workspace
3. Open Power BI Service: https://app.powerbi.com
4. Find your report under your workspace
5. Click **Schedule refresh** → set to Daily

### Share Dashboard
1. In Power BI Service, pin visuals to a new Dashboard
2. Dashboard → **Share** → enter email addresses
3. Or generate **Embed Link** for web embedding

---

## STEP 8 — CONNECTING TO LIVE DATA

For automatic refresh from SQLite (local only):
- Use Power BI Gateway (Personal mode)
- Download: https://powerbi.microsoft.com/gateway

For PostgreSQL (cloud):
- Direct Query mode for live data
- Or Import + Scheduled Refresh

---

## TROUBLESHOOTING

| Error | Cause | Fix |
|---|---|---|
| Cannot find SQLite driver | ODBC not installed | Install sqliteodbc_w64.exe |
| Relationship error | Duplicate keys | Check for orphaned records |
| Blank DAX measures | Wrong table reference | Verify table/column names |
| Refresh fails | File path changed | Update data source settings |
| Date format issues | Regional settings | Set locale in Power Query |

---

## EXPORT OPTIONS

- **PDF Export**: File → Export → PDF
- **PowerPoint**: File → Export → PowerPoint
- **CSV Data**: Right-click visual → Export data
- **Embed Code**: Power BI Service → File → Embed report

---

*Power BI Guide — Medical AI Platform v1.0*
