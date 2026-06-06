# ─────────────────────────────────────────────────────────────────────────────
# Medical AI Platform — Dockerfile
# ─────────────────────────────────────────────────────────────────────────────
# Build:   docker build -t medical-ai-platform .
# Run:     docker run -p 8501:8501 medical-ai-platform
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# System metadata
LABEL maintainer="Medical AI Platform"
LABEL version="1.0.0"
LABEL description="AI-Powered Chest X-Ray Pneumonia Detection Platform"

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    libgomp1 \
    wget \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first (Docker layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# Create required directories
RUN mkdir -p \
    data/raw data/processed data/augmented \
    models/saved_models models/checkpoints models/model_weights \
    logs \
    database \
    app/utils

# Initialize the SQLite database with seed data
RUN python src/database.py || true

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Environment variables (override at runtime)
ENV DATABASE_URL=sqlite:///database/medical_ai.db
ENV MODEL_DIR=models/saved_models
ENV LOG_LEVEL=INFO
ENV DEMO_MODE=true

# Streamlit configuration
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
ENV STREAMLIT_SERVER_ENABLE_CORS=false

# Start the Streamlit application
CMD ["streamlit", "run", "app/streamlit_app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
