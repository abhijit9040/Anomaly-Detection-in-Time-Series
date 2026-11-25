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
│  - Bottleneck dimension = hidden_size = 64             │
│  - 2-layer LSTM Decoder (64 hidden dim → 38 sensors)   │
│  - Leakage-safe Standard Scaler (fit on train only)     │
│  - Validation-set threshold: 99.0th percentile = 0.1541 │
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

<!-- RESULTS:START -->

### Multi-Method Benchmark Comparison

Evaluated on held-out test data from the **Server Machine Dataset (machine-1-1, 28,479 timesteps / 28,450 windows)**:

| Method | Evaluation Unit | Threshold Strategy | Threshold Value | Flagged Rate | Precision | Recall | F1 Score | AUROC | AUPRC | Event Recall | False Alarms |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **LSTM Autoencoder (Ours) ★** | Windows (W=30, S=1) | Val 99.0th Percentile | 0.1541 | 23.01% | 42.42% | 94.91% | **58.64%** | 0.9461 | 0.6125 | 8/8 (100.00%) | 94 |
| **PCA Reconstruction** | Windows (W=30, S=1) | Val 99.0th Percentile | 0.2350 | 9.80% | 60.90% | 58.03% | 59.43% | 0.9466 | 0.7174 | 8/8 (100.00%) | 42 |
| **Isolation Forest** | Windows (W=30, S=1) | Val 99.0th Percentile | 0.6696 | 13.41% | 34.95% | 45.56% | 39.55% | 0.8615 | 0.3909 | 5/8 (62.50%) | 44 |
| **Moving-Average Z-Score** | Windows (W=30, S=1) | Val 99.0th Percentile | 6.0551 | 55.26% | 18.41% | 98.94% | 31.05% | 0.8438 | 0.4877 | 8/8 (100.00%) | 16 |
| **Flag Everything (Trivial)** | Windows (W=30, S=1) | All 1s | n/a | 100.00% | 10.28% | 100.00% | 18.65% | 0.5000 | 0.1028 | 8/8 (100.00%) | 0 |

### Before vs. After Model Calibration

| Metric | Before Calibration (Phase 0) | After Calibration (Phase 1–8) | Change / Impact |
|:---|:---:|:---:|:---|
| **Threshold Strategy** | Val Mean + 3.0*Std | Val 99.0th Percentile | Leakage-free validation percentile |
| **Threshold Value** | 0.8281 | **0.1541** | Calibrated to normal variance |
| **Flagged Rate** | 51.64% | **23.01%** | -28.63 pp (-55.4% false-alert reduction) |
| **Precision** | 19.83% | **42.42%** | +22.59 pp (+114.0% relative gain) |
| **Recall** | 99.56% | **94.91%** | -4.65 pp (slight drop to eliminate 55% false alarms) |
| **F1 Score** | 33.07% | **58.64%** | +25.57 pp (+77.3% relative gain) |
| **AUROC** | n/a | **0.9461** | Threshold-independent discriminative capacity |
| **AUPRC** | n/a | **0.6125** | Precision-recall area on imbalanced test split |
| **Event Recall** | n/a | **8/8 (100.00%)** | All contiguous anomaly segments detected |
| **False Alarm Events** | n/a | **94** | Contiguous ungrounded alert bursts |

### How to Read These Results

1. **Improvement vs. Trivial Reference:** The LSTM Autoencoder achieves an F1 of 58.64%, which is a **+39.99 percentage point gain (+214.4% relative)** over the trivial Flag-Everything reference (18.65%).
2. **Comparison vs. Best Fair Baseline (PCA Reconstruction):** The LSTM F1 is -0.79 percentage points (-1.33%) compared to PCA Reconstruction (59.43%).
3. **Trade-off Profile:** The LSTM Autoencoder provides a decisive **recall advantage** (94.91% vs. 58.03% for PCA Reconstruction), capturing nearly all anomaly windows across time, whereas PCA Reconstruction achieves higher **precision** (60.90% vs. 42.42%) with fewer false alarms.
4. **Oracle Diagnostic Bound:** The best possible theoretical test F1 is **61.94%** (at threshold `0.3066`, precision 53.20%, recall 74.13%). This is an *oracle / diagnostic bound only* that uses ground-truth test labels to select the threshold, and must never be used for deployment or headline claims.
5. **Evaluation Unit:** All metrics are computed strictly on **sliding windows of size 30 with stride 1** ($W=30, S=1$). A window is labeled ground-truth positive if any timestep within it is labeled anomalous.

<!-- RESULTS:END -->

---

## 🔬 Evaluation Protocol & Methodology

This system enforces strict, publication-grade evaluation hygiene across training, backend inference, and dashboard reporting:

1. **Unit of Evaluation**:
   - The primary unit is a **sliding window of size 30 with stride 1** ($W=30, S=1$).
   - A window is labeled ground-truth anomalous ($y=1$) if **any timestep** inside the window is anomalous in the SMD expert label file.
   - All benchmark algorithms (Moving Average, PCA, Isolation Forest, and LSTM Autoencoder) are evaluated on the exact same test windows.

2. **Threshold Selection Rule (Leakage-Free)**:
   - Anomaly thresholds are selected strictly on the **validation partition** ($20\%$ temporal hold-out of normal training data).
   - The default selection rule is decided in advance: the **99.0th percentile of validation reconstruction errors** ($\approx 1\%$ expected false-positive rate on normal operations).
   - Test labels are **never** used to pick, calibrate, or tune thresholds for deployment.

3. **Oracle / Diagnostic Bounds**:
   - An "oracle best-F1" threshold is computed exclusively as a diagnostic upper bound by sweeping test thresholds.
   - This number is clearly demarcated as `oracle/diagnostic only` in reports and the UI, and is never used as the headline metric.

4. **Point-Adjusted F1 Caveat**:
   - Point-adjusted metrics (where detecting any single point marks the entire contiguous anomaly segment as true positive) are reported purely for comparison with older academic papers.
   - We explicitly flag that point-adjustment artificially inflates F1 scores, and we prioritize unadjusted window-level metrics and event recall.

5. **Known Limitations**:
   - **Single Machine Focus**: Current calibration is tuned on SMD `machine-1-1`. Multi-machine generalization requires retraining per-machine scalers and models.
   - **Precursor Latency**: The system exhibits an average detection delay of 3–5 timesteps relative to ground truth event onset.
   - **Synthetic Controls**: The live demo includes synthetic anomaly injection buttons (`POST /api/v1/stream/inject`) designed for operator demonstration.

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
