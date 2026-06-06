# 🫁 AI-Powered Medical Image Disease Detection & Analytics Platform

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13-FF6F00?logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![MLflow](https://img.shields.io/badge/MLflow-2.8-0194E2?logo=mlflow&logoColor=white)](https://mlflow.org)
[![SQLite](https://img.shields.io/badge/SQLite-3-003B57?logo=sqlite&logoColor=white)](https://sqlite.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> A production-ready end-to-end deep learning platform for automated chest X-ray pneumonia detection. Built with CNN, ResNet50, DenseNet121, EfficientNetB0, full SQL persistence, MLOps tracking, and an interactive Streamlit dashboard.

---

## 📸 Screenshots

| Home Dashboard | X-Ray Prediction | Analytics |
|:---:|:---:|:---:|
| KPI cards, trends | Upload & diagnose | Plotly charts |

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     DATA PIPELINE                                │
│  Kaggle Dataset → OpenCV Preprocessing → Augmentation           │
│  (CLAHE + Gaussian Denoising + Normalization + Resize 224×224)  │
└─────────────────────────┬───────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────────┐
│                     MODEL LAYER                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────────┐ │
│  │ Custom CNN   │  │  ResNet50    │  │ DenseNet121 / EffNet   │ │
│  │ (from scratch│  │ (TL Phase 1+2│  │ (Transfer Learning)    │ │
│  │  4-block CNN)│  │ fine-tuning) │  │                        │ │
│  └──────┬───────┘  └──────┬───────┘  └───────────┬────────────┘ │
│         └─────────────────┼──────────────────────┘              │
│                    Model Comparison                              │
│                  (Best by Recall Score)                          │
└─────────────────────────┬───────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────────┐
│                  PERSISTENCE LAYER                               │
│  SQLite / PostgreSQL                                             │
│  Tables: patients | predictions | image_metadata                │
│          model_versions | audit_logs                            │
│  MLflow: Experiment tracking + artifact storage                 │
└─────────────────────────┬───────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────────┐
│                  APPLICATION LAYER                               │
│  Streamlit Multi-Page App                                        │
│  Pages: Home | Predict | Patients | Analytics | Models | MLOps  │
│  Charts: Plotly interactive dashboards                          │
│  Export: CSV download, prediction reports                       │
└─────────────────────────┬───────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────────┐
│                  DEPLOYMENT                                      │
│  Streamlit Cloud | Render | Railway | Docker | HuggingFace      │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
medical_ai_platform/
│
├── 📂 app/                         # Streamlit application
│   ├── streamlit_app.py            # Main multi-page app
│   └── utils/
│       └── helpers.py              # Shared UI + prediction utilities
│
├── 📂 config/
│   └── config.yaml                 # All project configuration
│
├── 📂 data/                        # Dataset (gitignored)
│   ├── raw/chest_xray/             # Kaggle dataset
│   ├── processed/                  # Preprocessed tensors
│   └── augmented/                  # Augmented images
│
├── 📂 database/
│   ├── schema.sql                  # CREATE TABLE scripts
│   └── queries.sql                 # Reporting SQL library
│
├── 📂 deployment/
│   ├── Dockerfile                  # Container definition
│   ├── docker-compose.yml          # Multi-service stack
│   ├── Procfile                    # Render/Railway startup
│   ├── render.yaml                 # Render.com blueprint
│   └── runtime.txt                 # Python version pin
│
├── 📂 docs/                        # Documentation + EDA images
│   ├── COMPLETE_PROJECT_IMPLEMENTATION_GUIDE.md
│   └── interview_prep.md
│
├── 📂 models/
│   ├── saved_models/               # .h5 trained models
│   ├── checkpoints/                # Training checkpoints
│   └── model_weights/              # Layer weights
│
├── 📂 notebooks/
│   └── eda.py                      # EDA (convert to .ipynb)
│
├── 📂 powerbi/
│   └── instructions.md             # Complete Power BI guide
│
├── 📂 src/                         # Core Python modules
│   ├── analytics.py                # Business analytics + Plotly
│   ├── cnn_model.py                # Custom CNN architecture
│   ├── database.py                 # SQLAlchemy ORM + queries
│   ├── logger.py                   # Colored logging setup
│   ├── mlops.py                    # MLflow tracking + monitoring
│   ├── preprocessing.py            # Image preprocessing pipeline
│   ├── run_pipeline.py             # Master pipeline runner
│   └── transfer_learning.py        # ResNet50/DenseNet/EfficientNet
│
├── 📂 tests/
│   ├── test_database.py            # DB unit + integration tests
│   ├── test_model.py               # CNN + metrics tests
│   └── test_preprocessing.py       # Image processing tests
│
├── .env.example                    # Environment variable template
├── .gitignore
├── .streamlit/config.toml          # Streamlit dark theme
├── environment.yml                 # Conda environment
└── requirements.txt                # pip dependencies
```

---

## ⚡ Quick Start (5 Minutes)

### 1. Clone & Setup Environment

```bash
git clone https://github.com/yourusername/medical-ai-platform.git
cd medical-ai-platform

# Create conda environment
conda env create -f environment.yml
conda activate medical_ai

# Copy environment config
cp .env.example .env
```

### 2. Download Dataset

```bash
# Configure Kaggle API first (see docs/COMPLETE_PROJECT_IMPLEMENTATION_GUIDE.md)
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia -p data/raw --unzip
```

### 3. Initialize Database & Run App (Demo Mode)

```bash
# Initialize DB with 200 demo patients
python src/database.py

# Launch Streamlit app (works without trained models — demo mode)
streamlit run app/streamlit_app.py
```

App opens at: **http://localhost:8501**

### 4. Train Models (Optional)

```bash
# Full pipeline: DB + CNN + Transfer Learning + Comparison
python src/run_pipeline.py --mode full

# Train specific model only
python src/run_pipeline.py --mode cnn-only
python src/run_pipeline.py --mode transfer-only
```

---

## 🧠 Model Architecture

### Custom CNN
| Layer | Config |
|---|---|
| Conv Block 1 | Conv2D(32) + BN + ReLU + MaxPool |
| Conv Block 2 | Conv2D(64) + BN + ReLU + MaxPool |
| Conv Block 3 | Conv2D(128) + BN + ReLU + MaxPool |
| Conv Block 4 | Conv2D(256) + BN + ReLU + MaxPool |
| Head | GlobalAvgPool + Dense(256) + Dropout(0.5) + Dense(1, sigmoid) |

### Transfer Learning Models
| Model | Base | Trainable Params | Notes |
|---|---|---|---|
| ResNet50 | ImageNet | ~25M | Two-phase fine-tuning |
| DenseNet121 | ImageNet | ~8M | Dense skip connections |
| EfficientNetB0 | ImageNet | ~5M | Compound scaling |

### Why Recall over Accuracy?
> In medical AI, a **False Negative** (telling a sick patient they are healthy) is far more dangerous than a **False Positive** (extra follow-up for a healthy patient). We optimize for **Recall (Sensitivity)** and accept lower precision.

---

## 🗄 SQL Database

**SQLite** (default — zero setup) or **PostgreSQL** (production).

```sql
-- Key reporting query: monthly pneumonia trends
SELECT 
    strftime('%Y-%m', pr.created_at) AS month,
    COUNT(*) AS total_predictions,
    SUM(CASE WHEN pr.prediction_label = 'PNEUMONIA' THEN 1 ELSE 0 END) AS pneumonia_count,
    ROUND(100.0 * SUM(CASE WHEN pr.prediction_label = 'PNEUMONIA' THEN 1 ELSE 0 END) / COUNT(*), 2) AS pneumonia_rate_pct
FROM predictions pr
GROUP BY month
ORDER BY month;
```

---

## 📊 Streamlit App Pages

| Page | Features |
|---|---|
| 🏠 Home | KPIs, monthly trends, architecture overview |
| 🔬 Predict | Upload X-ray, patient form, AI diagnosis, confidence gauge |
| 👥 Patients | Search, register, view history |
| 📈 Analytics | Interactive Plotly charts: demographics, trends, distribution |
| 🤖 Models | Registry, performance comparison, architecture cards |
| ⚙️ MLOps | Drift detection, health monitoring, MLflow UI link |

---

## 🚀 Deployment

### Option 1 — Streamlit Community Cloud (Recommended, Free)
```
1. Push repo to GitHub
2. Visit: https://share.streamlit.io
3. Connect your GitHub repo
4. Main file: app/streamlit_app.py
5. Deploy (takes ~5 minutes)
```

### Option 2 — Docker
```bash
cd deployment
docker build -f Dockerfile .. -t medical-ai-platform
docker run -p 8501:8501 medical-ai-platform
```

### Option 3 — Render.com
```
1. Connect GitHub at https://dashboard.render.com
2. New Web Service → select repo
3. Build: pip install -r requirements.txt
4. Start: streamlit run app/streamlit_app.py --server.port=$PORT --server.address=0.0.0.0
5. Deploy
```

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=src --cov-report=html -v

# Run specific test file
pytest tests/test_database.py -v
pytest tests/test_model.py -v
pytest tests/test_preprocessing.py -v
```

---

## 📋 MLOps Features

- **Experiment Tracking**: MLflow logging of all training runs
- **Model Versioning**: Dual tracking (SQLite DB + MLflow registry)
- **Drift Detection**: Statistical tests on prediction confidence distribution
- **Health Monitoring**: Automated model health checks with alerts
- **Audit Logs**: Complete action history for compliance

```bash
# View MLflow UI
mlflow ui --host 0.0.0.0 --port 5000
# Open: http://localhost:5000
```

---

## 🔧 Configuration

All settings in `config/config.yaml`:

```yaml
model:
  architecture: "densenet121"    # Switch model here
  input_shape: [224, 224, 3]
  
training:
  batch_size: 32
  epochs: 50
  learning_rate: 0.001
  recall_weight: 2.0             # Emphasize recall

database:
  url: "sqlite:///database/medical_ai.db"
```

---

## 🛣 Future Improvements

- [ ] Multi-class detection (bacterial vs viral pneumonia)
- [ ] Grad-CAM visualization (highlight affected lung regions)
- [ ] DICOM file support
- [ ] REST API (FastAPI) for system integration
- [ ] Patient report PDF generation
- [ ] Mobile app (Flutter/React Native)
- [ ] Federated learning for multi-hospital collaboration
- [ ] Real-time alert system for high-risk predictions

---

## ⚠️ Medical Disclaimer

> This platform is developed for **educational and portfolio purposes only**. It is **NOT intended for actual clinical use**. All predictions must be validated by qualified healthcare professionals. Never use AI predictions as the sole basis for medical decisions.

---

## 📄 License

MIT License — see [LICENSE](LICENSE)

---

## 🙏 Acknowledgements

- Dataset: [Paul Timothy Mooney — Kaggle Chest X-Ray Images](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- Original research: Kermany et al., *Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning*, Cell 2018
