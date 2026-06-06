# COMPLETE PROJECT IMPLEMENTATION GUIDE
## AI-Powered Medical Image Disease Detection & Analytics Platform
### A Step-by-Step Deployment Handbook for Beginners

---

**Version:** 1.0  
**Platform:** Windows 11 + Anaconda + VS Code  
**Last Updated:** 2024  

---

## TABLE OF CONTENTS

- PART 1  — What You Need Before You Start
- PART 2  — Software Installation
- PART 3  — Project Folder Setup
- PART 4  — Environment Setup
- PART 5  — Kaggle Dataset Download
- PART 6  — EDA Execution Guide
- PART 7  — Image Processing Guide
- PART 8  — CNN Training Guide
- PART 9  — Transfer Learning Guide
- PART 10 — SQL Database Guide
- PART 11 — Streamlit App Guide
- PART 12 — Power BI Guide
- PART 13 — GitHub Guide
- PART 14 — Deployment Guide
- PART 15 — Testing Guide
- PART 16 — Troubleshooting Guide
- PART 17 — Common Errors & Fixes
- PART 18 — Final Execution Roadmap

---

# PART 1 — WHAT YOU NEED BEFORE YOU START

## 1.1 Objective
This section ensures you understand what the project does and that your machine meets the minimum requirements before any installation begins.

## 1.2 Project Overview
This platform performs automated pneumonia detection from chest X-ray images using deep learning. It:
- Trains custom CNN and pre-trained models (ResNet50, DenseNet121, EfficientNetB0)
- Stores all predictions in a SQL database
- Provides an interactive Streamlit web application
- Implements full MLOps tracking with MLflow
- Visualizes analytics in Power BI

## 1.3 System Requirements

| Component | Minimum | Recommended |
|---|---|---|
| OS | Windows 10 | Windows 11 |
| RAM | 8 GB | 16 GB |
| Storage | 10 GB free | 20 GB free |
| GPU | None (CPU works) | NVIDIA 8GB+ VRAM |
| CPU | 4 cores | 8 cores |
| Internet | Required | Required |

## 1.4 Accounts Required (All Free)
- [ ] **GitHub** — https://github.com (for version control and deployment)
- [ ] **Kaggle** — https://kaggle.com (for dataset download)
- [ ] **Streamlit Community Cloud** — https://share.streamlit.io (for deployment)
- [ ] **Microsoft** — https://microsoft.com (for Power BI)

## 1.5 Estimated Time to Complete
- Software installation: 30–60 minutes
- Dataset download: 10–20 minutes (1.2 GB)
- Full model training: 2–8 hours (CPU) / 30–60 minutes (GPU)
- App deployment: 15–30 minutes
- Total: 1 day for a beginner

---

# PART 2 — SOFTWARE INSTALLATION

## 2.1 Anaconda

**Objective:** Install Anaconda to manage Python environments and packages.

**What is Anaconda?**  
Anaconda is a distribution of Python that includes conda (a package and environment manager). It lets you create isolated Python environments so different projects don't conflict with each other.

**Download:**
1. Visit: https://www.anaconda.com/products/distribution
2. Click **Download** — choose **Windows 64-bit**
3. File will be named: `Anaconda3-YYYY.MM-Windows-x86_64.exe`

**Installation Steps:**
1. Run the installer as Administrator (right-click → Run as Administrator)
2. Click **Next** through the license agreement
3. Choose **Just Me** (not All Users)
4. Installation path: `C:\Users\YourName\anaconda3` (leave default)
5. ⚠️ Check **"Add Anaconda3 to my PATH environment variable"** (ignore warning)
6. Click **Install**
7. When complete, click **Finish**

**Verify Installation:**
```bash
# Open Anaconda Prompt (search in Start Menu)
conda --version
# Expected output: conda 23.x.x  (version number may differ)

python --version
# Expected output: Python 3.11.x
```

**Troubleshooting:**
- If `conda` is not recognized: Add `C:\Users\YourName\anaconda3\Scripts` to Windows PATH manually
- Path: System Properties → Environment Variables → System Variables → Path → Edit → New

---

## 2.2 VS Code

**Objective:** Install VS Code as your code editor.

**Download:**
1. Visit: https://code.visualstudio.com/download
2. Click **Windows** → download `.exe` installer

**Installation:**
1. Run the installer
2. Accept license
3. Check all boxes (especially "Add to PATH" and "Open with Code")
4. Click **Install**

**Required Extensions:**
After opening VS Code, press `Ctrl+Shift+X` and install:
- **Python** (by Microsoft) — Python language support
- **Pylance** — Type checking
- **Jupyter** — Notebook support
- **GitLens** — Enhanced Git UI
- **SQLite Viewer** — Browse SQLite databases

**Configure Python Interpreter:**
1. Open VS Code
2. Press `Ctrl+Shift+P`
3. Type: `Python: Select Interpreter`
4. Choose: `Python 3.11 (base) - Anaconda`

**Verify:**
- Bottom-left of VS Code shows: `Python 3.11.x ('base': conda)`

---

## 2.3 Git

**Objective:** Install Git for version control.

**Download:** https://git-scm.com/download/win

**Installation:**
1. Run the installer
2. All defaults are fine
3. Editor: choose **VS Code**
4. Default branch name: `main`
5. PATH: **Git from the command line and also from 3rd-party software**
6. Click **Install**

**Configure Git:**
```bash
# Open Anaconda Prompt or VS Code Terminal
git config --global user.name "Your Name"
git config --global user.email "your@email.com"
git config --global init.defaultBranch main

# Verify
git --version
# Expected: git version 2.43.x
```

---

## 2.4 SQLite Browser (Optional but Recommended)

**What it is:** A visual tool to inspect your SQLite database tables.

**Download:** https://sqlitebrowser.org/dl/
- Choose: **DB Browser for SQLite** → Windows installer

**Usage:**
1. Open DB Browser
2. File → Open Database → select `database/medical_ai.db`
3. Browse Tables tab → click any table to see data

---

## 2.5 Power BI Desktop

**Download:** https://powerbi.microsoft.com/desktop

**Installation:**
1. Run the `.msi` installer
2. Accept defaults
3. Sign in with a Microsoft account (free account works)

---

# PART 3 — PROJECT FOLDER SETUP

## 3.1 Objective
Create the exact folder structure the project expects.

## 3.2 Create Project Root

**Option A — Using VS Code:**
1. Open VS Code
2. File → Open Folder → navigate to `C:\Projects` (create if needed)
3. Terminal → New Terminal

**Option B — Using Anaconda Prompt:**
```cmd
cd C:\
mkdir Projects
cd Projects
```

## 3.3 Clone or Create Project

**If you have the files already:**
```cmd
cd C:\Projects
# Place all provided source files into: C:\Projects\medical_ai_platform\
```

**If starting fresh (clone from GitHub):**
```cmd
cd C:\Projects
git clone https://github.com/yourusername/medical-ai-platform.git medical_ai_platform
cd medical_ai_platform
```

## 3.4 Create All Directories

Open Anaconda Prompt and run:
```cmd
cd C:\Projects\medical_ai_platform

mkdir data\raw data\processed data\augmented
mkdir models\saved_models models\checkpoints models\model_weights
mkdir src app\pages app\utils
mkdir database\migrations
mkdir deployment docs config logs
mkdir notebooks tests powerbi
mkdir .streamlit
```

**Expected Structure:**
```
C:\Projects\medical_ai_platform\
├── app\
│   ├── pages\
│   └── utils\
├── config\
├── data\
│   ├── raw\
│   ├── processed\
│   └── augmented\
├── database\
├── deployment\
├── docs\
├── logs\
├── models\
│   ├── saved_models\
│   ├── checkpoints\
│   └── model_weights\
├── notebooks\
├── powerbi\
├── src\
└── tests\
```

**Verify:**
```cmd
dir /s /b | find "\"
# Should list all directories
```

---

# PART 4 — ENVIRONMENT SETUP

## 4.1 Objective
Create an isolated Python 3.11 environment with all required packages.

## 4.2 Create Conda Environment

```cmd
# Navigate to project root
cd C:\Projects\medical_ai_platform

# Create environment from environment.yml
conda env create -f environment.yml

# Expected output (takes 5-15 minutes):
# Collecting package metadata: done
# Solving environment: done
# Downloading and extracting packages: ▓▓▓▓▓▓▓▓▓▓ 100%
# ...
# done
```

**If conda env create fails:**
```cmd
# Create environment manually
conda create -n medical_ai python=3.11 -y
conda activate medical_ai
pip install -r requirements.txt
```

## 4.3 Activate Environment

```cmd
conda activate medical_ai

# Verify activation (prompt should show):
# (medical_ai) C:\Projects\medical_ai_platform>
```

**IMPORTANT:** Always activate the environment before running any Python scripts:
```cmd
conda activate medical_ai
```

## 4.4 Verify All Packages

```cmd
python -c "import tensorflow as tf; print('TF:', tf.__version__)"
python -c "import streamlit; print('Streamlit:', streamlit.__version__)"
python -c "import cv2; print('OpenCV:', cv2.__version__)"
python -c "import sqlalchemy; print('SQLAlchemy:', sqlalchemy.__version__)"
python -c "import mlflow; print('MLflow:', mlflow.__version__)"
```

**Expected outputs:**
```
TF: 2.13.x
Streamlit: 1.28.x
OpenCV: 4.8.x
SQLAlchemy: 2.0.x
MLflow: 2.8.x
```

## 4.5 Configure Environment Variables

```cmd
# Copy the template
copy .env.example .env

# Open .env in VS Code
code .env
```

Edit these values in `.env`:
```ini
# Change these to match your system:
DATABASE_URL=sqlite:///database/medical_ai.db
MODEL_DIR=models/saved_models
LOG_LEVEL=INFO
DEMO_MODE=true

# Kaggle credentials (fill in Part 5):
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_api_key
```

---

# PART 5 — KAGGLE DATASET DOWNLOAD

## 5.1 Create a Kaggle Account

1. Visit: https://www.kaggle.com
2. Click **Register** → use email or Google
3. Verify your email

## 5.2 Generate API Token

1. Log in to Kaggle
2. Click your profile picture (top-right) → **Settings**
3. Scroll to **API** section
4. Click **Create New Token**
5. File `kaggle.json` downloads automatically

**File contents look like:**
```json
{"username":"yourusername","key":"abc123...xyz"}
```

## 5.3 Place API Token

```cmd
# Create .kaggle directory in your home folder
mkdir C:\Users\YourName\.kaggle

# Copy the downloaded kaggle.json there
copy C:\Users\YourName\Downloads\kaggle.json C:\Users\YourName\.kaggle\kaggle.json
```

## 5.4 Install and Configure Kaggle CLI

```cmd
conda activate medical_ai
pip install kaggle

# Verify
kaggle --version
# Expected: Kaggle API 1.6.x
```

## 5.5 Download the Dataset

```cmd
cd C:\Projects\medical_ai_platform

kaggle datasets download -d paultimothymooney/chest-xray-pneumonia -p data/raw --unzip
```

**Expected output:**
```
Downloading chest-xray-pneumonia.zip to data/raw
100%|████████████████████| 1.15G/1.15G [03:42<00:00, 5.18MB/s]

Archive:  chest-xray-pneumonia.zip
   creating: data/raw/chest_xray/
   creating: data/raw/chest_xray/train/
   ...
```

**Verify download:**
```cmd
dir data\raw\chest_xray\train\NORMAL /s | find "File(s)"
# Expected: several hundred files

dir data\raw\chest_xray\train\PNEUMONIA /s | find "File(s)"
# Expected: several thousand files
```

## 5.6 Expected Dataset Structure

```
data/raw/chest_xray/
├── train/
│   ├── NORMAL/    (1,341 images)
│   └── PNEUMONIA/ (3,875 images)
├── test/
│   ├── NORMAL/    (234 images)
│   └── PNEUMONIA/ (390 images)
└── val/
    ├── NORMAL/    (8 images)
    └── PNEUMONIA/ (8 images)
```

**Note:** The validation set is very small. The training code uses test/ for primary evaluation.

---

# PART 6 — EDA EXECUTION GUIDE

## 6.1 Objective
Run exploratory data analysis to understand the dataset before training.

## 6.2 Launch Jupyter in VS Code

```cmd
conda activate medical_ai
cd C:\Projects\medical_ai_platform

# Convert eda.py to Jupyter notebook
pip install jupytext
jupytext --to notebook notebooks/eda.py -o notebooks/eda.ipynb

# Open in VS Code
code notebooks/eda.ipynb
```

## 6.3 Run the Notebook

1. Open `notebooks/eda.ipynb` in VS Code
2. Click **Select Kernel** → choose `medical_ai`
3. Click **Run All** (or press `Shift+Enter` for each cell)

## 6.4 What Each Section Shows

**Section 1 — Dataset Overview:**
- Count of images per class per split
- Confirms 1,341 NORMAL + 3,875 PNEUMONIA in training

**Section 2 — Class Distribution:**
- Bar charts showing imbalance
- Key insight: PNEUMONIA:NORMAL ≈ 2.9:1 ratio

**Section 3 — Sample Images:**
- Visual comparison of NORMAL vs PNEUMONIA X-rays
- PNEUMONIA shows white infiltrates/consolidation
- NORMAL shows clear lung fields

**Section 4 — Image Dimensions:**
- Images vary from ~400×400 to ~1500×1600 pixels
- Justifies resizing to standard 224×224

**Section 5 — Pixel Intensity:**
- PNEUMONIA tends to have lower mean brightness
- Higher std deviation due to infiltrates
- Justifies CLAHE preprocessing

**Section 6 — Preprocessing Pipeline:**
- Visual comparison: Original → Grayscale → CLAHE → Final
- Shows improvement in contrast

## 6.5 Expected Outputs
All chart images saved to `docs/`:
- `eda_class_distribution.png`
- `eda_sample_images.png`
- `eda_dimensions.png`
- `eda_pixel_intensity.png`

---

# PART 7 — IMAGE PROCESSING GUIDE

## 7.1 Objective
Understand and use the preprocessing pipeline that prepares images for model training.

## 7.2 What preprocessing.py Does

The file `src/preprocessing.py` contains the full pipeline:

1. **Load** — Read JPEG/PNG images using OpenCV
2. **Validate** — Check dimensions, channels, and pixel range
3. **Convert** — BGR to Grayscale
4. **CLAHE** — Contrast Limited Adaptive Histogram Equalization (improves local contrast in lung regions)
5. **Denoise** — Gaussian blur (reduces noise artifacts)
6. **Resize** — Standardize to 224×224 pixels
7. **Normalize** — Scale pixels to [0, 1]
8. **Stack** — Convert single-channel to 3-channel (required by pretrained models)

## 7.3 Run Preprocessing

```cmd
conda activate medical_ai
cd C:\Projects\medical_ai_platform

python -c "
from src.preprocessing import MedicalImagePreprocessor
p = MedicalImagePreprocessor()
train_gen, val_gen = p.create_data_generators(
    train_dir='data/raw/chest_xray/train',
    val_dir='data/raw/chest_xray/val'
)
print('Train batches:', len(train_gen))
print('Classes:', train_gen.class_indices)
"
```

**Expected output:**
```
Train batches: 165
Classes: {'NORMAL': 0, 'PNEUMONIA': 1}
```

## 7.4 Augmentation Strategy

The training generator applies these augmentations randomly:
- Rotation: ±15 degrees
- Horizontal flip (X-rays can be mirrored)
- Width/height shift: ±10%
- Brightness: 0.7–1.3×
- Zoom: ±10%

**Why these specific augmentations?**
- We do NOT apply vertical flips (upside-down lungs are not realistic)
- We do NOT apply shear (distorts anatomy)
- We DO apply brightness variation (simulates different X-ray exposures)

---

# PART 8 — CNN TRAINING GUIDE

## 8.1 Objective
Train a custom CNN model built from scratch on the chest X-ray dataset.

## 8.2 Model Architecture Summary

```
Input (224, 224, 3)
↓
Block 1: Conv2D(32) → BatchNorm → ReLU → MaxPool(2,2)
↓
Block 2: Conv2D(64) → BatchNorm → ReLU → MaxPool(2,2)
↓
Block 3: Conv2D(128) → BatchNorm → ReLU → MaxPool(2,2)
↓
Block 4: Conv2D(256) → BatchNorm → ReLU → MaxPool(2,2)
↓
GlobalAveragePooling2D
↓
Dense(256) + ReLU + Dropout(0.5)
↓
Dense(1) + Sigmoid → probability of PNEUMONIA
```

## 8.3 Train the CNN

```cmd
conda activate medical_ai
cd C:\Projects\medical_ai_platform

python src/run_pipeline.py --mode cnn-only
```

**Training takes approximately:**
- CPU only: 3–8 hours for 50 epochs
- GPU (RTX 3070+): 15–45 minutes

**Monitor training in real-time:**
- VS Code terminal shows: `Epoch 1/50 — loss: 0.45 — accuracy: 0.82 — val_recall: 0.87`
- Model checkpoints saved to `models/checkpoints/`

## 8.4 Training Callbacks Explained

| Callback | What It Does |
|---|---|
| EarlyStopping | Stops if val_recall doesn't improve for 10 epochs |
| ModelCheckpoint | Saves best model (by val_recall) to disk |
| ReduceLROnPlateau | Halves learning rate if stuck for 5 epochs |
| TensorBoard | Logs metrics for visualization |
| CSVLogger | Writes metrics to CSV for analysis |

## 8.5 View TensorBoard

```cmd
conda activate medical_ai
tensorboard --logdir logs/tensorboard
# Open: http://localhost:6006
```

## 8.6 Expected Results (Custom CNN)

| Metric | Expected Range |
|---|---|
| Accuracy | 85–92% |
| Recall | 88–96% |
| Precision | 80–90% |
| ROC-AUC | 0.92–0.97 |

---

# PART 9 — TRANSFER LEARNING GUIDE

## 9.1 Objective
Fine-tune pre-trained ImageNet models for pneumonia detection.

## 9.2 Why Transfer Learning?

Pre-trained models (ResNet50, DenseNet121, EfficientNetB0) have already learned:
- Edge detection (low-level features)
- Texture patterns (mid-level features)
- Object recognition (high-level features)

We replace only the classification head and adapt the learned representations to medical imaging.

## 9.3 Two-Phase Training Strategy

**Phase 1 — Head Only (5 epochs)**
- Freeze all base model layers
- Train only the custom head
- Learning rate: 0.001
- Goal: Initialize the new head to reasonable values

**Phase 2 — Fine-tuning (45 epochs)**
- Unfreeze the top 30% of base model layers
- Train everything (base + head)
- Learning rate: 0.0001 (10× smaller)
- Goal: Adapt pretrained features to X-ray domain

## 9.4 Train Transfer Learning Models

```cmd
# Train all three models
python src/run_pipeline.py --mode transfer-only

# Train individual model
python -c "
from src.transfer_learning import TransferLearningPipeline
p = TransferLearningPipeline()
p.train_model(
    architecture='densenet121',
    train_dir='data/raw/chest_xray/train',
    val_dir='data/raw/chest_xray/val',
    test_dir='data/raw/chest_xray/test',
    save_path='models/saved_models/densenet121_model.h5'
)
"
```

## 9.5 Model Comparison

```cmd
python src/run_pipeline.py --mode compare-only
```

Output: `models/comparison_report.csv`

Expected comparison:
| Model | Accuracy | Recall | ROC-AUC |
|---|---|---|---|
| Custom CNN | 88% | 91% | 0.94 |
| ResNet50 | 91% | 93% | 0.96 |
| DenseNet121 | **93%** | **95%** | **0.97** |
| EfficientNetB0 | 92% | 94% | 0.97 |

**Note:** DenseNet121 often performs best on medical imaging due to dense skip connections.

---

# PART 10 — SQL DATABASE GUIDE

## 10.1 Objective
Initialize the database and understand how predictions are stored and queried.

## 10.2 Initialize Database

```cmd
conda activate medical_ai
cd C:\Projects\medical_ai_platform

python src/database.py
```

**Expected output:**
```
INFO - Database initialized at: database/medical_ai.db
INFO - Created table: patients
INFO - Created table: predictions
INFO - Created table: image_metadata
INFO - Created table: model_versions
INFO - Created table: audit_logs
INFO - Seeded 200 patients
INFO - Seeded 1,187 predictions (6-month history)
```

## 10.3 Verify with DB Browser

1. Open **DB Browser for SQLite**
2. File → Open Database → `database/medical_ai.db`
3. Browse Data tab → select `predictions`
4. Should show ~1,000+ rows

## 10.4 Run Sample Queries

```cmd
# Open Python REPL
python

>>> from src.database import DatabaseManager
>>> db = DatabaseManager()
>>> kpis = db.get_kpi_summary()
>>> print(kpis)
```

**Or run SQL directly in DB Browser:**
```sql
-- Total predictions by month
SELECT 
    strftime('%Y-%m', created_at) AS month,
    COUNT(*) AS predictions,
    SUM(CASE WHEN prediction_label='PNEUMONIA' THEN 1 ELSE 0 END) AS pneumonia
FROM predictions
GROUP BY month
ORDER BY month;
```

## 10.5 Switching to PostgreSQL

For production deployment:

1. Install PostgreSQL: https://www.postgresql.org/download/windows/
2. Create database:
```sql
CREATE DATABASE medical_ai_db;
CREATE USER medical_ai_user WITH PASSWORD 'your_password';
GRANT ALL PRIVILEGES ON DATABASE medical_ai_db TO medical_ai_user;
```
3. Update `.env`:
```ini
DATABASE_URL=postgresql://medical_ai_user:your_password@localhost:5432/medical_ai_db
```
4. Run: `python src/database.py` (same command, auto-detects PostgreSQL)

---

# PART 11 — STREAMLIT APP GUIDE

## 11.1 Objective
Launch and use the Streamlit web application.

## 11.2 Launch the App

```cmd
conda activate medical_ai
cd C:\Projects\medical_ai_platform

streamlit run app/streamlit_app.py
```

**Expected output:**
```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501

  For better performance, install the Watchdog module:
  $ pip install watchdog
```

Browser opens automatically. If not: go to http://localhost:8501

## 11.3 App Pages Walkthrough

### 🏠 Home Page
- Shows KPI metrics cards (total predictions, pneumonia rate, avg confidence)
- Monthly trends line chart
- Project architecture description
- Works immediately with demo data (no model needed)

### 🔬 Prediction Page
**To test a prediction:**
1. Navigate to "Predict" in the sidebar
2. Fill in patient details (name, age, gender, hospital)
3. Click "Upload X-Ray Image"
4. Upload any `.jpg` chest X-ray image
5. Click "Analyze X-Ray"
6. See: diagnosis label, confidence gauge, probability bar, recommendation
7. Click "Save to Database" to persist the result

**Demo Mode (no model needed):**
If no `.h5` model file is found, the app automatically uses a heuristic algorithm that analyzes image statistics to simulate a prediction. Useful for portfolio demonstration.

### 👥 Patients Page
- Search patients by name
- Filter by hospital
- View complete prediction history per patient
- Register new patients

### 📈 Analytics Page
- Disease rate by gender (bar chart)
- Disease rate by age group (bar chart)
- Monthly trends (line chart)
- Confidence distribution (histogram)
- All charts are interactive (hover, zoom, download)
- Export to CSV button

### 🤖 Models Page
- Model registry table (all registered models)
- Performance comparison charts
- Architecture descriptions

### ⚙️ MLOps Page
- Prediction drift detection slider (simulate checking for data drift)
- Model health check button
- MLflow UI instructions

## 11.4 Stop the App

Press `Ctrl+C` in the terminal.

---

# PART 12 — POWER BI GUIDE

See: `powerbi/instructions.md` for the complete step-by-step Power BI guide including all DAX measures, calculated columns, and dashboard page configurations.

**Quick Start:**
1. Install Power BI Desktop
2. Open Power BI → Get Data → ODBC → connect to `database/medical_ai.db`
3. Load all 5 tables
4. Build relationships as described in `powerbi/instructions.md`
5. Add DAX measures from `powerbi/instructions.md`
6. Create 5 pages following the visual specifications

---

# PART 13 — GITHUB GUIDE

## 13.1 Objective
Push the project to GitHub for version control, collaboration, and deployment.

## 13.2 Create GitHub Repository

1. Visit: https://github.com → Sign in
2. Click **+** → **New repository**
3. Name: `medical-ai-platform`
4. Description: `AI-Powered Chest X-Ray Pneumonia Detection Platform`
5. Set to **Public** (required for free Streamlit Cloud deployment)
6. DO NOT initialize with README (we already have one)
7. Click **Create repository**

## 13.3 Initialize and Push

```cmd
cd C:\Projects\medical_ai_platform

git init
git add .
git commit -m "Initial commit: Complete Medical AI Platform"
git branch -M main
git remote add origin https://github.com/yourusername/medical-ai-platform.git
git push -u origin main
```

**If authentication fails:**
1. Go to GitHub → Settings → Developer Settings → Personal Access Tokens
2. Generate new token (classic) with `repo` scope
3. Use the token as your password when prompted

## 13.4 Important .gitignore Rules

Already configured in `.gitignore`:
- `data/` — Dataset is 1.2 GB, too large for GitHub
- `models/saved_models/*.h5` — Trained models are too large
- `*.db` — SQLite database
- `logs/` — Log files
- `.env` — Contains secrets

**What IS pushed to GitHub:**
- All Python source files
- Configuration files
- Documentation
- Tests
- Deployment files
- README

---

# PART 14 — DEPLOYMENT GUIDE

## 14.1 Option A — Streamlit Community Cloud (Recommended)

**Prerequisites:**
- GitHub repository is public
- `requirements.txt` is in root
- `app/streamlit_app.py` exists

**Steps:**
1. Visit: https://share.streamlit.io
2. Click **Sign in with GitHub**
3. Click **New app**
4. Repository: `yourusername/medical-ai-platform`
5. Branch: `main`
6. Main file path: `app/streamlit_app.py`
7. Click **Deploy**

**Deployment takes 3–10 minutes.**

**Environment Variables on Streamlit Cloud:**
1. In your app settings → **Secrets**
2. Add:
```toml
DATABASE_URL = "sqlite:///database/medical_ai.db"
DEMO_MODE = "true"
LOG_LEVEL = "INFO"
```

**App URL will be:** `https://yourusername-medical-ai-platform.streamlit.app`

**Common Issues:**
- If deployment fails: check `requirements.txt` has no version conflicts
- If app crashes: check logs in Streamlit Cloud dashboard
- TensorFlow too slow: reduce model size or use demo mode only

---

## 14.2 Option B — Render.com

1. Visit: https://dashboard.render.com
2. New → Web Service
3. Connect your GitHub repository
4. Settings:
   - **Name:** medical-ai-platform
   - **Runtime:** Python 3
   - **Build Command:** `pip install -r requirements.txt && python src/database.py`
   - **Start Command:** `streamlit run app/streamlit_app.py --server.port=$PORT --server.address=0.0.0.0`
5. Environment Variables → add same as Streamlit Cloud
6. Click **Create Web Service**

**Free tier limitations:** App sleeps after 15 minutes of inactivity (cold start ~30 seconds).

---

## 14.3 Option C — Docker (Local or Cloud)

**Build image:**
```cmd
cd C:\Projects\medical_ai_platform
docker build -f deployment/Dockerfile . -t medical-ai-platform
```

**Run container:**
```cmd
docker run -p 8501:8501 -v %CD%/data:/app/data medical-ai-platform
```

**With Docker Compose (includes MLflow):**
```cmd
cd deployment
docker-compose up --build
```

Streamlit: http://localhost:8501  
MLflow: http://localhost:5000

---

## 14.4 Option D — Hugging Face Spaces

1. Visit: https://huggingface.co/spaces
2. Create new Space → **Streamlit** framework
3. Clone the Space repository
4. Copy project files into it
5. Push — auto-deploys

Note: Hugging Face Spaces has a 50 GB storage limit and may time out large models.

---

# PART 15 — TESTING GUIDE

## 15.1 Run All Tests

```cmd
conda activate medical_ai
cd C:\Projects\medical_ai_platform

pytest tests/ -v --tb=short
```

**Expected output:**
```
tests/test_preprocessing.py::TestImageLoading::test_load_valid_image PASSED
tests/test_preprocessing.py::TestPreprocessing::test_resize_224x224 PASSED
...
tests/test_database.py::TestDatabaseInitialization::test_initialize_creates_tables PASSED
tests/test_database.py::TestPatientCRUD::test_add_patient_returns_id PASSED
...
tests/test_model.py::TestMetricsCalculation::test_recall_critical_for_medical PASSED
...
============== 28 passed in 12.34s ==============
```

## 15.2 Run with Coverage Report

```cmd
pytest tests/ --cov=src --cov-report=html -v
# Opens report at: htmlcov/index.html
```

## 15.3 Skip Slow Tests (No GPU)

```cmd
pytest tests/ -v -m "not slow"
```

## 15.4 Run Individual Test Files

```cmd
pytest tests/test_preprocessing.py -v  # Preprocessing only
pytest tests/test_database.py -v        # Database only
pytest tests/test_model.py -v           # Model tests only
```

**Note:** Model tests that require TensorFlow will auto-skip if TF is not installed.

---

# PART 16 — TROUBLESHOOTING GUIDE

## 16.1 Conda Environment Issues

**Problem:** `conda: command not found`
**Fix:**
```cmd
# Add to Windows PATH:
C:\Users\YourName\anaconda3
C:\Users\YourName\anaconda3\Scripts
C:\Users\YourName\anaconda3\Library\bin

# Or run from Anaconda Prompt instead of CMD
```

**Problem:** `conda env create` hangs
**Fix:**
```cmd
conda config --set channel_priority flexible
conda clean --all
conda env create -f environment.yml
```

## 16.2 TensorFlow Issues

**Problem:** TensorFlow import error
**Fix:**
```cmd
conda activate medical_ai
pip uninstall tensorflow
pip install tensorflow==2.13.0
```

**Problem:** GPU not detected by TensorFlow
**Fix:**
```cmd
# Check GPU
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# Install GPU drivers:
# NVIDIA: https://developer.nvidia.com/cuda-downloads
# Install CUDA 11.8 + cuDNN 8.6 for TF 2.13
```

## 16.3 OpenCV Issues

**Problem:** `cv2` module not found
**Fix:**
```cmd
pip uninstall opencv-python opencv-python-headless
pip install opencv-python-headless==4.8.1.78
```

## 16.4 Database Issues

**Problem:** `sqlite3.OperationalError: database is locked`
**Fix:**
- Close DB Browser for SQLite
- Kill any other Python processes accessing the DB
- Restart the Streamlit app

**Problem:** `Table not found`
**Fix:**
```cmd
python src/database.py  # Re-run initialization
```

## 16.5 Streamlit Issues

**Problem:** Port 8501 already in use
**Fix:**
```cmd
# Use a different port
streamlit run app/streamlit_app.py --server.port 8502
```

**Problem:** App doesn't load (blank screen)
**Fix:**
- Clear browser cache (Ctrl+Shift+Delete)
- Try incognito window
- Check terminal for error messages

## 16.6 Kaggle Download Issues

**Problem:** `401 Unauthorized`
**Fix:**
```cmd
# Verify kaggle.json is in the right location
type C:\Users\YourName\.kaggle\kaggle.json
# Should show: {"username":"...","key":"..."}

# Set permissions (Windows PowerShell)
icacls C:\Users\YourName\.kaggle\kaggle.json /inheritance:r /grant:r "%USERNAME%:R"
```

---

# PART 17 — COMMON ERRORS & FIXES

| Error Message | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'src'` | Wrong working directory | Run from project root: `cd C:\Projects\medical_ai_platform` |
| `FileNotFoundError: config.yaml` | Wrong working directory | Same as above |
| `OSError: [Errno 28] No space left on device` | Dataset too large | Free 5+ GB disk space |
| `ResourceExhaustedError: OOM` | GPU memory exhausted | Reduce `batch_size` in `config.yaml` from 32 to 16 |
| `ValueError: Input 0 is incompatible` | Wrong image shape | Verify `target_size=(224,224)` in preprocessing |
| `ConnectionRefusedError` (PostgreSQL) | DB server not running | Start PostgreSQL service in Windows Services |
| `streamlit: command not found` | Env not activated | `conda activate medical_ai` |
| `SSL: CERTIFICATE_VERIFY_FAILED` (Kaggle) | Corporate firewall | Try VPN or mobile hotspot |
| `KeyboardInterrupt` during training | Normal | Press Ctrl+C, model checkpoint was saved |
| `PermissionError: [WinError 5]` | File lock | Close VS Code/DB Browser holding the file |

---

# PART 18 — FINAL EXECUTION ROADMAP

## Complete Project Execution Order

```
Step  1: Install all software (Part 2)         ≈ 45 min
Step  2: Create folder structure (Part 3)       ≈  5 min
Step  3: Set up conda environment (Part 4)      ≈ 15 min
Step  4: Configure .env file (Part 4)           ≈  2 min
Step  5: Download Kaggle dataset (Part 5)       ≈ 20 min
Step  6: Run EDA notebook (Part 6)              ≈ 15 min
Step  7: Initialize database (Part 10)          ≈  2 min
Step  8: Train CNN model (Part 8)               ≈ 3-8 hrs
Step  9: Train transfer learning (Part 9)       ≈ 4-12 hrs
Step 10: Compare models (Part 9)                ≈  5 min
Step 11: Launch Streamlit app (Part 11)         ≈  1 min
Step 12: Build Power BI dashboard (Part 12)     ≈  2 hrs
Step 13: Push to GitHub (Part 13)               ≈ 10 min
Step 14: Deploy online (Part 14)                ≈ 20 min
Step 15: Run tests (Part 15)                    ≈  5 min
```

## Quick Demo (No GPU, No Training)

If you want to demo the project without training:
```cmd
# 1. Set up environment
conda env create -f environment.yml
conda activate medical_ai

# 2. Initialize database
python src/database.py

# 3. Launch app (demo mode auto-activates)
streamlit run app/streamlit_app.py
```

The app runs in demo mode with:
- 200 pre-seeded patients
- 6 months of synthetic prediction history
- Heuristic predictions on uploaded images

## Production Checklist

Before deploying to production:
- [ ] Replace SQLite with PostgreSQL
- [ ] Set `DEMO_MODE=false` in `.env`
- [ ] Train and verify model recall > 90%
- [ ] Run full test suite (all tests passing)
- [ ] Set `LOG_LEVEL=WARNING` for production
- [ ] Enable HTTPS on deployment platform
- [ ] Add authentication layer (not included in this demo)
- [ ] Review medical disclaimer with your organization

---

*This guide was generated as part of the Medical AI Platform portfolio project.*  
*Not intended for clinical use.*
