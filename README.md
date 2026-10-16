# ⚡ AnomalyGuard: Real-Time Sensor Anomaly Detection System

> **Advanced Data Science Track** | Industrial IoT / Predictive Maintenance | Full-Stack ML Application

A production-grade, end-to-end anomaly detection system for multi-sensor IoT time-series data using a **PyTorch LSTM Autoencoder**, grounded in the paper *"LSTM-based Encoder-Decoder for Multi-sensor Anomaly Detection — ICML 2016"*.

⚠️ **Decision-support only** — Anomaly flags are for human review and action. This system does **not** autonomously trigger equipment shutdown or maintenance orders.

---

## 🏗️ Architecture

```
┌────────────────────────────────────────────────────────┐
│               React + Plotly Frontend                  │
│  - Real-Time Sensor Stream Dashboard (0-38 channels)   │
│  - Reconstruction Error vs Adaptive Val Threshold      │
│  - Root Cause Analysis (Top contributing sensors)      │
│  - Quantitative Model Evaluation & Baseline Comparison │
│  - Architecture Metadata & Hyperparameters Card        │
│  - Settings & Health Diagnostics                       │
└───────────────────────────▲────────────────────────────┘
                            │ REST / JSON (Axios)
┌───────────────────────────▼────────────────────────────┐
│                  FastAPI Backend                       │
│  - POST /api/v1/auth/login (JWT token generation)      │
│  - POST /api/v1/detect (Window inference + RCA)        │
│  - GET  /api/v1/stream (Real-time SMD data replay)     │
│  - GET  /api/v1/model (Model version & metadata)       │
│  - GET  /api/v1/evaluation (Quantitative F1 report)    │
│  - GET  /health (Readiness & model state probe)        │
└───────────────────────────▲────────────────────────────┘
                            │ In-Memory Inference
┌───────────────────────────▼────────────────────────────┐
│            PyTorch LSTM Autoencoder Pipeline           │
│  - 2-layer LSTM Encoder (38 sensors → 64 hidden dim)   │
│  - 2-layer LSTM Decoder (64 hidden dim → 38 sensors)   │
│  - Leakage-safe Standard Scaler (fit on train only)     │
│  - Validation-set threshold: Mean + 3.0*Std = 0.8281   │
└────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.10+
- Node.js 18+ and npm 9+

### 2. Backend Setup & Run

Use the dedicated backend virtual environment to isolate dependencies:

```bash
cd backend

# On Windows:
.venv\Scripts\python.exe -m uvicorn app.main:app --port 8000 --reload

# Or if creating fresh:
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python.exe -m uvicorn app.main:app --port 8000 --reload
```

- API Documentation: [http://localhost:8000/docs](http://localhost:8000/docs)
- Health Check: [http://localhost:8000/health](http://localhost:8000/health)

### 3. Frontend Setup & Run

```bash
cd frontend
npm install
npm run dev
```

- Web App: [http://localhost:5173](http://localhost:5173)

### 4. Credentials for Login (FR-007)

The application includes 1-click demo login buttons:
- **System Admin:** `admin` / `admin`
- **Plant Operator:** `operator` / `operator`

---

## 🧪 Testing

Run backend unit and integration test suite:

```bash
cd backend
.venv\Scripts\python.exe -m pytest tests/ -v
```

All 8 tests verify:
- Health check status probe (`GET /health`)
- Health response schema compliance
- Root documentation route (`GET /`)
- Unloaded model 503 guardrails
- JWT authentication (`POST /api/v1/auth/login`)
- Evaluation metrics report (`GET /api/v1/evaluation`)

---

## 📊 Evaluation & Baseline Comparison (FR-005 & FR-013)

Evaluated on held-out test data from the **Server Machine Dataset (machine-1-1, 28,479 timesteps)**:

| Metric | Moving Average Z-Score (Baseline) | LSTM Autoencoder (Ours) | Relative Improvement |
|:-------|:--------------------------------:|:-----------------------:|:-------------------:|
| **F1-Score** | 17.28% | **33.07%** | **+91.3%** 🚀 |
| **Recall** | 100.00% | **99.56%** | High coverage |
| **Precision** | 9.46% | **19.83%** | **+109.6%** |
| **Threshold** | Dynamic | **0.8281** (Val set: $\mu + 3\sigma$) | Leakage-free |

---

## 📁 Repository Structure

```text
ojt-project/
├── backend/
│   ├── .venv/                  # Dedicated Python environment
│   ├── app/
│   │   ├── api/v1/router.py    # All 5 API endpoints (detect, stream, model, eval, auth)
│   │   ├── config.py           # App settings
│   │   ├── main.py             # FastAPI entry & lifespan model loader
│   │   ├── models/schemas.py   # Pydantic schemas for requests/responses
│   │   └── services/           # ModelService, StreamService, LstmLoader
│   ├── tests/                  # Pytest test cases (8/8 passing)
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── components/Sidebar.jsx  # Navigation & user role / signout
│   │   ├── pages/
│   │   │   ├── Dashboard.jsx       # Real-time streaming & anomaly alert table
│   │   │   ├── EvaluationPage.jsx  # F1/Precision/Recall gauges & baseline chart
│   │   │   ├── LoginPage.jsx       # Auth login with 1-click demo buttons
│   │   │   ├── ModelInfoPage.jsx   # Model architecture & hyperparameters
│   │   │   ├── SensorsPage.jsx     # 38-channel monitor & Root Cause Analysis
│   │   │   └── SettingsPage.jsx    # System diagnostics & config preferences
│   │   ├── services/api.js         # Axios API client
│   │   └── App.jsx                 # Protected routing wrapper
│   └── package.json
├── training/
│   ├── artifacts/
│   │   ├── lstm_autoencoder_machine-1-1.pt  # PyTorch model weights
│   │   ├── scaler_machine-1-1.pkl           # Preprocessing scaler
│   │   ├── evaluation_results.json          # Test metrics
│   │   └── baseline_results.json            # Baseline metrics
│   ├── src/
│   │   ├── baseline.py         # Statistical benchmark
│   │   ├── evaluate.py         # Test evaluation script
│   │   ├── model.py            # LSTM Autoencoder PyTorch module
│   │   ├── preprocessing.py    # Leakage-safe windowing pipeline
│   │   └── train.py            # Training pipeline
│   └── configs/default.yaml
├── scripts/
│   └── download_dataset.py     # SMD dataset downloader
└── PRD.md                      # Product Requirements Document
```

---

## 🛡️ Product Principles & Governance

1. **Decision Support Only:** The system detects subtle precursor anomalies to assist plant engineers, not autonomously execute disruptive maintenance.
2. **Leakage-Free Validation:** Normalization scaler fit solely on training data. Anomaly threshold selected solely on the validation set. Held-out test set evaluated strictly once.
3. **Transparent Explainability:** Real-time Root Cause Analysis (FR-009) breaks down the reconstruction error across individual sensor channels so technicians know which components are anomalous.
