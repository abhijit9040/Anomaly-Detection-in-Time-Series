# Product Requirements Document — Anomaly Detection in Time Series

**Project:** Anomaly Detection in Sensor/IoT Time Series Data
**Track:** Advanced Data Science | **Domain:** Industrial IoT / Manufacturing
**Duration:** 12 weeks | **Team:** 1 | **Level:** Advanced
**Stack:** Python, JavaScript | **Frameworks:** PyTorch, FastAPI, React
**Supporting Libraries:** pandas, Plotly, Tailwind CSS, Axios

---

## 1. Product Goal

Deliver a full-stack application that enables **Operations teams and predictive-maintenance engineers** in Industrial IoT / Manufacturing to **detect anomalies in multi-sensor time-series data in real time**, using an LSTM autoencoder model grounded in the paper *"LSTM-based Encoder-Decoder for Multi-sensor Anomaly Detection — ICML 2016"*. The system must produce a **trained anomaly detector**, a **real-time monitoring dashboard**, and a **quantitative F1 evaluation report**, all wrapped in a working React + FastAPI application.

---

## 2. Problem Statement

Industrial manufacturing environments generate continuous multi-sensor time-series data (temperature, vibration, pressure, etc.). Undetected anomalies in this data lead to:
- **Unplanned equipment downtime** costing thousands of dollars per hour.
- **Missed early-warning signals** that could have triggered preventive maintenance.
- **Manual monitoring** that is inconsistent, error-prone, and does not scale across hundreds of sensors.

This product provides an **automated, model-driven anomaly detection signal** displayed on a real-time dashboard for human review and action.

**Critical distinction — predictive, not reactive:** The LSTM autoencoder does not detect failures that have already happened. It detects *subtle deviations in sensor behavior patterns* that are precursors to failure. A rising reconstruction error on a vibration sensor, for example, may indicate bearing wear days before the bearing actually fails. The model gives operations teams a window to act *before* damage occurs — this is the core value of predictive maintenance over reactive repair.

---

## 3. Architecture Rationale

This section addresses three foundational design questions about the system architecture.

### 3.1 Why does this project need APIs?

The LSTM autoencoder model is a Python/PyTorch artifact. The monitoring dashboard is a React (JavaScript) frontend. These two cannot communicate directly — a backend API is the bridge between them. Specifically, FastAPI serves the following purposes:

- **Decoupling:** The frontend does not load or execute the PyTorch model. It sends a time-series window to `POST /api/v1/detect` and receives a structured JSON response (anomaly flag, reconstruction error, sensor breakdown). This separation means the model can be retrained, versioned, or swapped without changing a single line of frontend code.
- **Standardized interface:** The API defines a contract — input schema, output schema, error codes — that any client (dashboard, batch script, mobile app, third-party system) can consume. Without this, every consumer would need direct access to the PyTorch runtime.
- **Operational concerns:** Authentication (FR-007), health checks (FR-008), input validation (Pydantic), structured error responses, and request logging all live in the API layer. These cannot exist in a standalone model script.
- **Versioning & rollback:** The API response includes model version metadata. When a new model is deployed, the API serves it; if it underperforms, the previous version can be restored (FR-015) without frontend changes.

*Without the API layer, the trained model is a file on disk that no one outside the data science team can use.*

### 3.2 How does real-time monitoring work with the ML model?

The monitoring dashboard does not run the ML model. The data flow is:

```
Dataset (NAB/SMD)                    React Dashboard (Plotly)
     |                                        ^
     | replay at                               | poll every N seconds
     | configurable                            | (or WebSocket push)
     | interval                                |
     v                                        |
FastAPI Backend  ──── LSTM Model ────▶  JSON Response
(receives window)    (compute recon.     (anomaly flag,
                      error, compare      error score,
                      to threshold)       sensor breakdown)
```

Step by step:

1. **Data replay:** Since this project uses a static dataset (NAB/SMD), the backend replays sensor windows at a configurable interval to simulate a live data stream. In a production environment, this would be replaced by an MQTT/Kafka ingestion layer (currently out of scope).
2. **Inference:** For each window, the backend passes the normalized sensor data through the LSTM encoder-decoder. The reconstruction error per sensor is computed.
3. **Threshold comparison:** If the total reconstruction error exceeds the pre-selected threshold (chosen on the validation set), the window is flagged as anomalous.
4. **API response:** The backend returns: anomaly flag (boolean), total reconstruction error (float), per-sensor error breakdown (array), model version, and timestamp.
5. **Dashboard rendering:** The React frontend polls the API at a configurable interval (default: every few seconds), updates the Plotly charts (sensor streams, reconstruction error over time), and highlights anomaly markers with color + icon.

*The "real-time" aspect is the continuous polling loop between the frontend and backend — not a live sensor connection.*

### 3.3 If an issue has already happened, is there still a need for the model?

This is the most important question. The answer is: **the model does not detect issues that have already caused damage — it detects the early signals that precede them.**

- **Anomaly is not failure.** An anomaly is a deviation from normal sensor behavior. A vibration sensor reading that is 15% higher than its trained normal pattern is anomalous — but the machine is still running. Without the model, this deviation goes unnoticed until the bearing fails hours or days later.
- **The labeled anomalies in the NAB/SMD dataset** represent exactly these precursor windows — they mark periods where sensor behavior deviates from normal *before* or *during* an incident, not after the equipment is already destroyed.
- **Business value:** The cost of a detected anomaly (scheduled maintenance during a planned window) is orders of magnitude lower than the cost of an undetected anomaly (unplanned downtime, emergency repair, production loss, safety risk).
- **Even for post-incident analysis:** When an incident does occur, the model's historical reconstruction error timeline shows exactly when the deviation started and which sensors contributed — enabling root cause analysis and preventing recurrence.

*The model turns invisible sensor drift into a visible, actionable alert before damage occurs. That is the entire justification for this project.*

---

## 4. Target Users / Personas

| Persona | Role | Goal | Pain Point |
|---|---|---|---|
| Plant Operations Manager | Monitors overall equipment health | See real-time anomaly alerts across sensors to prevent downtime | Manual log review; anomalies discovered too late |
| Maintenance Technician | Inspects and repairs equipment | Know *which* sensor/component triggered an anomaly and *why* | No root-cause signal; reactive instead of proactive maintenance |
| Data Scientist / ML Engineer | Builds and validates the anomaly detection model | Evaluate model performance with rigorous metrics; reproduce results | Lack of standardized evaluation (F1, precision, recall) and experiment tracking |
| System Administrator | Manages app access and deployment | Ensure only authorized personnel access dashboards | No authentication on monitoring tools |

---

## 5. Research Grounding

- **Paper:** *LSTM-based Encoder-Decoder for Multi-sensor Anomaly Detection — ICML 2016* ([arXiv search link](https://arxiv.org/search/?query=LSTM-based+Encoder-Decoder+for+Multi-sensor+Anomaly+Detection&searchtype=all))
- **Core Idea:** An LSTM encoder compresses a window of multi-sensor time-series into a fixed-length representation; an LSTM decoder reconstructs the original window. High **reconstruction error** on unseen data indicates an anomaly.
- **Relationship to Implementation:** The paper provides the conceptual architecture. The PyTorch implementation may differ in specific layer sizes, training hyperparameters, or optimization strategy. Do not claim the paper specifies the exact implementation.

---

## 6. Dataset

- **Primary Dataset:** NAB (Numenta Anomaly Benchmark) and/or SMD (Server Machine Dataset)
- **Source:** [Kaggle search](https://www.kaggle.com/search?q=NAB/SMD)
- **Nature:** Multivariate time-series sensor readings with labeled anomaly windows.
- **Important Assumption:** The exact schema (number of sensors/features, row count, sampling rate, anomaly label format, and class distribution) must be documented from the actual downloaded dataset. Do not fabricate schema details.

---

## 7. Deliverables (Mandatory)

Every item below is a **required output** of the project:

| ID | Deliverable | Description |
|---|---|---|
| D-001 | **Trained Anomaly Detector** | A versioned PyTorch LSTM autoencoder model artifact, with documented training configuration, anomaly threshold, and reproducible pipeline. |
| D-002 | **Real-Time Monitoring Dashboard** | A polished React frontend with Plotly charts showing live/near-live sensor data, reconstruction error, and flagged anomalies. |
| D-003 | **F1 Evaluation Report** | A quantitative report containing F1-score, precision, recall, and standard NAB evaluation scores on the test set. |
| D-004 | **Full-Stack Application** | A working React frontend + FastAPI backend wrapping the above, deployable and demonstrable. |
| D-005 | **README** | Clear setup instructions, architecture diagram, and reproduction steps. |
| D-006 | **Unit & Integration Tests** | Tests covering API endpoints and the core ML pipeline. |
| D-007 | **Basic Auth / Session Handling** | Authentication on the web app so only authorized users access dashboards. |
| D-008 | **CI/CD Quality Gates** | A CI pipeline that enforces: schema validation, reproducibility check, artifact creation, metrics presence, threshold documentation, and model-card update before any deployment. |

---

## 8. Features (Prioritized)

| ID | Feature | Priority | Description | Acceptance Criteria |
|---|---|:---:|---|---|
| FR-001 | **Sensor Data Ingestion & Preprocessing** | Must | Ingest NAB/SMD dataset; normalize, window, and prepare multi-sensor time-series for LSTM input. | Dataset source/version documented; windowing parameters configurable; preprocessing is leakage-safe (fit on train only). |
| FR-002 | **LSTM Autoencoder Training Pipeline** | Must | Train a PyTorch LSTM encoder-decoder on normal data; compute reconstruction error; select anomaly threshold on validation set. | Versioned model artifact produced; training config (epochs, LR, hidden size, window size) logged; threshold selection procedure documented. |
| FR-003 | **Anomaly Detection Inference** | Must | Given a time-series window, compute reconstruction error and flag as anomaly if error exceeds threshold. | Returns anomaly flag (boolean), reconstruction error (float), sensor-level error breakdown, model version, and timestamp. |
| FR-004 | **Real-Time Monitoring Dashboard** | Must | React + Plotly dashboard showing: (a) live sensor time-series, (b) reconstruction error over time, (c) anomaly markers/alerts, (d) sensor health summary. | Dashboard loads in < 3 seconds; charts update at configurable interval; anomalies highlighted visually with color + icon (not color-only). |
| FR-005 | **F1 / Evaluation Report Page** | Must | Display F1-score, precision, recall, confusion matrix, and NAB score on a dedicated dashboard page or downloadable report. | Metrics computed on held-out test set; NAB evaluation protocol followed; report is reproducible. |
| FR-006 | **Model Performance & Metadata** | Must | API endpoint and UI page showing model version, training date, hyperparameters, dataset version, threshold, and evaluation metrics. | Matches the actual trained artifact; updates when model is retrained. |
| FR-007 | **Basic Authentication & Session Handling** | Must | Login screen with username/password; protected routes; session or JWT-based token management. | Unauthenticated requests to protected routes return 401; session expires after configurable timeout. |
| FR-008 | **Health Check Endpoint** | Must | `GET /health` returns service and model readiness status. | Returns 200 when model is loaded; returns 503 when model is unavailable. |
| FR-009 | **Root Cause Analysis** | Should | For each detected anomaly, show which sensors contributed most to the high reconstruction error. | Per-sensor error contribution displayed; top-N contributing sensors highlighted. |
| FR-010 | **Model Card** | Should | Document intended use, dataset, metrics, limitations, and ethical considerations. | Follows standard model card format; accessible via API or UI. |
| FR-011 | **Experiment Tracking** | Should | Log run ID, dataset version, hyperparameters, training metrics, and artifact version. | At least 3 tracked experiments with different configs; comparison possible. |
| FR-012 | **Batch Inference** | Could | Accept a CSV/JSON file of time-series data and return anomaly results for the entire batch. | Upload → process → downloadable results; handles files up to 50 MB. |
| FR-013 | **Baseline Anomaly Detector** | Must | Implement a simple baseline detector (e.g., statistical threshold / moving-average z-score) on the same dataset as a comparison benchmark. | Baseline F1, precision, and recall documented; LSTM autoencoder must outperform baseline to justify model complexity. |
| FR-014 | **Observability & Logging Dashboard** | Should | Log and surface: request ID, endpoint, latency, model version, anomaly-rate distribution, model-load failures, and p50/p95 inference latency. No raw sensor data in logs. | Logs are structured (JSON); at least request rate, error rate, and p95 latency are visible; health endpoint fails when model is unavailable. |
| FR-015 | **Artifact Rollback & Versioning** | Should | Package model artifact, preprocessing pipeline, anomaly threshold, and metadata together as a versioned unit. Maintain at least one previous version for rollback. | Previous version can be loaded without code changes; version is traceable in every API response. |

---

## 9. User Stories

### Operations & Monitoring

**US-001:** As a **plant operations manager**, I want to see a **real-time dashboard of all sensor streams** with anomaly alerts so I can **immediately identify equipment issues before they cause downtime**.
- *Acceptance:* Dashboard shows at least 5 sensor channels; anomalies are highlighted within 5 seconds of detection; I can filter by sensor/machine.

**US-002:** As a **maintenance technician**, I want to see **which specific sensors contributed to an anomaly** so I can **inspect the right component** instead of checking every sensor manually.
- *Acceptance:* Anomaly detail view shows per-sensor reconstruction error breakdown; top contributing sensors are ranked.

**US-003:** As an **operations manager**, I want to receive a **visual alert (color + icon) on the dashboard** when an anomaly is detected, so I can **take action even in a noisy control-room environment**.
- *Acceptance:* Alert is visible without scrolling; alert uses both color and icon (accessibility); alert disappears or changes state when acknowledged or when the anomaly window passes.

### Model Evaluation & Data Science

**US-004:** As a **data scientist**, I want to see **F1-score, precision, recall, and NAB score** on the test set so I can **quantitatively assess whether the LSTM autoencoder meets production standards**.
- *Acceptance:* Evaluation page shows all four metrics; confusion matrix is displayed; NAB evaluation protocol is documented.

**US-005:** As a **data scientist**, I want a **reproducible training configuration** (seed, hyperparameters, dataset version, split rationale) so I can **rerun experiments and get the same results**.
- *Acceptance:* Re-running training with the same config produces identical metrics (within floating-point tolerance).

**US-006:** As a **data scientist**, I want to **compare multiple model runs** (different architectures, hyperparameters) so I can **select the best-performing anomaly detector**.
- *Acceptance:* At least 3 experiment runs are logged; I can view a comparison table of metrics.

### Security & Administration

**US-007:** As a **system administrator**, I want **basic authentication on the dashboard** so that **only authorized plant personnel can view sensitive operational data**.
- *Acceptance:* Login page with username/password; unauthenticated requests redirected to login; session expires after 30 minutes of inactivity.

**US-008:** As a **new team member**, I want a **clear README with setup instructions and architecture diagram** so I can **run the entire application locally within 30 minutes**.
- *Acceptance:* README contains: prerequisites, install steps, environment variables, database/model setup, start commands for frontend and backend, and architecture diagram.

---

## 10. Evaluation & Success Metrics

| Metric | Target | Method |
|---|---|---|
| **F1-Score** | ≥ 0.80 on test set (document actual) | Standard sklearn F1 at chosen threshold |
| **Precision** | Document actual; discuss trade-offs | TP / (TP + FP) |
| **Recall** | Document actual; discuss trade-offs | TP / (TP + FN) |
| **NAB Score** | Compute and document using standard NAB evaluation protocol | NAB scoring function applied to anomaly detections |
| **Anomaly Threshold** | Selected on validation set, not test set | Reconstruction-error percentile or statistical method |
| **Dashboard Latency** | Chart update < 3 seconds | Manual + automated timing |
| **API Inference Latency** | p95 < 500 ms for single-window inference | Load test or manual measurement |

> **Important:** The anomaly threshold must be selected on the **training/validation** partition. The test set must remain untouched until final evaluation. This mirrors the split protocol from the reference documentation.

---

## 11. Non-Functional Requirements

| Category | Requirement |
|---|---|
| **Latency** | Single-window anomaly inference: p95 < 500 ms. Dashboard chart refresh: < 3 seconds. |
| **Authentication** | Basic auth (username/password) with session/JWT tokens. Protected routes return 401 for unauthenticated users. |
| **Security** | HTTPS in production; input validation via Pydantic; no raw sensor data in logs; dependency scanning; CORS restrictions. |
| **Scalability** | Keep FastAPI stateless. Do not introduce microservices or databases without demonstrated need. |
| **Reliability** | Load model at startup; expose `/health`; structured error responses; deterministic inference for fixed model + input. |
| **Accessibility** | Anomaly indicators must use both color AND icon/text (not color-only). Keyboard navigable. Semantic HTML. |
| **Data Volume** | Must handle at least the full NAB/SMD dataset without OOM. Document memory/GPU requirements. |
| **GPU** | LSTM training requires GPU (CUDA). Inference should work on CPU. Document requirements. |
| **Rollback / Versioning** | Model artifact, preprocessing pipeline, threshold, and metadata must be versioned together. At least one prior version must be retained for rollback without code changes. |
| **Responsible Use** | Anomaly detection output is decision-support only. The system must not autonomously trigger equipment shutdown or maintenance orders. A visible disclaimer must appear on every dashboard screen and in the model card. |

---

## 12. Scope Boundary

### In Scope
- LSTM autoencoder training and inference on NAB/SMD data.
- Full-stack React + FastAPI application with basic auth.
- Real-time monitoring dashboard with Plotly visualizations.
- F1, precision, recall, and NAB evaluation report.
- Root cause analysis (per-sensor contribution).
- Unit and integration tests for API and ML pipeline.
- README with architecture diagram and reproduction steps.

### Out of Scope
- Live connection to physical IoT sensors or MQTT brokers (data is replayed from dataset).
- Multi-tenant SaaS deployment.
- Mobile application.
- Advanced RBAC (role-based access control) beyond basic auth.
- Automated model retraining pipeline (CI/CD for ML ops).
- Integration with external alerting systems (PagerDuty, Slack, etc.).

---

## 13. Product Principles

1. **Detect first, act second** — the system surfaces anomalies for human review; it does not autonomously shut down equipment.
2. **Explainable anomalies** — every detection must show reconstruction error and sensor-level contribution; a bare "anomaly detected" flag is insufficient.
3. **Rigorous evaluation** — model quality is strictly measured by F1-score, precision, recall, and NAB protocol; do not over-report or cherry-pick metrics.
4. **Threshold transparency** — the anomaly threshold, its selection method, and its trade-offs (missed anomalies vs. false alarms) must be documented and visible.
5. **Version everything** — model artifact, preprocessing pipeline, threshold, and evaluation metrics must be versioned and traceable.
6. **Secure by default** — industrial sensor data is operationally sensitive; all dashboard access requires authentication.

---

## 14. Core Data / ML Lifecycle

```
Data Acquisition → Preprocessing & Windowing → EDA → Train/Val/Test Split →
LSTM Autoencoder Training → Threshold Selection (on Validation) →
Test Evaluation (F1, Precision, Recall, NAB) → Model Artifact Versioning →
FastAPI Serving → React Dashboard → Deployment
```

---

## 15. Dependencies on Other Documents

| Document | Dependency |
|---|---|
| BRD (`02_BRD.md`) | Business objectives, personas, and business requirements that this PRD translates into product features. |
| TRD (`05_TRD.md`) | Technical requirements (TR-IDs) that map to FR-IDs in this PRD. |
| UX Requirements (`04_UX_Requirements.md`) | Dashboard screens, states, and accessibility rules. |
| HLD (`06_HLD.md`) | Architecture diagram showing React → FastAPI → LSTM model flow. |
| API Specification (`08_API_Specification.md`) | Endpoint contracts consumed by FR-003, FR-006, FR-008. |
| Testing Strategy (`12_Testing_Strategy.md`) | Test cases validating acceptance criteria in this PRD. |
| Model Card (`21_Model_Card.md`) | Detailed model documentation referenced by FR-010. |

---

## 16. Phased Delivery (Aligned to Roadmap)

| Phase | Weeks | What Gets Delivered |
|---|:---:|---|
| **Phase 1 — Foundation** | 1–4 | Study paper; scaffold FastAPI + React shell; acquire & explore NAB/SMD dataset; build baseline pipeline; EDA complete. |
| **Phase 2 — Core Model & API** | 5–8 | Train & tune LSTM autoencoder; expose via FastAPI endpoints; run F1/precision/recall evaluation; build React frontend screens. |
| **Phase 3 — Integration & Polish** | 9–12 | Complete frontend-backend integration; deliver trained detector + dashboard + F1 report; add auth, root cause analysis; polish UI/UX; finalize evaluation, documentation, demo, and presentation. |

---

## 17. Traceability Summary

| Business Req | Feature | User Story | API Endpoint | Test Coverage |
|---|---|---|---|---|
| Detect anomalies in real time | FR-003, FR-004 | US-001, US-003 | POST /api/v1/detect | TEST-001, TEST-005 |
| Sensor-level root cause | FR-009, FR-013 | US-002 | POST /api/v1/detect | TEST-002 |
| F1/NAB evaluation | FR-005 | US-004 | GET /api/v1/model | TEST-003 |
| Reproducible experiments | FR-011, FR-013 | US-005, US-006 | GET /api/v1/model | TEST-007, TEST-008 |
| Secure access | FR-007 | US-007 | All protected routes | TEST-004 |
| Setup & documentation | D-005, D-008 | US-008 | N/A | README review |
| Rollback capability | FR-015 | US-005 | GET /api/v1/model | TEST-010 |
| Observability | FR-014 | US-004 | GET /api/v1/health | TEST-009 |

---

## 18. Distinction Criteria

To achieve a **distinction-level grade**, the project must satisfy all of the following (sourced from `26_Project_Quality_Score.md`):

| Criterion | What Is Required |
|---|---|
| **Clear problem formulation** | Problem statement references real IoT/manufacturing pain points with quantified impact (downtime cost, scale). |
| **Leakage-free methodology** | Preprocessing (normalization, windowing) fitted on training data only; threshold selected on validation set; test set untouched until final evaluation. |
| **Strong baseline comparison** | A simple statistical baseline detector must be built and its metrics (F1, precision, recall) compared against the LSTM autoencoder. LSTM must demonstrably outperform the baseline. |
| **Rigorous LSTM evaluation** | F1-score, precision, recall, NAB score, confusion matrix, and reconstruction-error distribution all reported on the held-out test set. |
| **Threshold & business trade-offs** | Anomaly threshold selection method documented; precision–recall trade-offs explained in plain language for operations users. |
| **Experiment reproducibility** | Fixed random seed, dataset version, split seed, and all hyperparameters logged per run; re-running produces same metrics. |
| **Clean model serving** | FastAPI stateless; model loaded once at startup; versioned artifact; `/health` endpoint; structured error responses. |
| **Tested API & UI** | Unit tests for preprocessing and inference; integration tests for all API endpoints; UI tests for loading/error/success states. |
| **Responsible use limitations** | Model card and dashboard both explicitly state the system is decision-support only; anomaly flag must not autonomously trigger actions. |
| **Polished dashboard** | Plotly charts render sensor data, reconstruction error, and anomaly markers; accessibility (color + icon); loads in under 3 seconds. |
