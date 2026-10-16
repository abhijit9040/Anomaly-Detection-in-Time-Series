# Traceability Matrix
| Business | Feature | User Story | Technical | API | Component | Artifact | Test |
|---|---|---|---|---|---|---|---|
| BR-001 | Dataset ingestion | US-004 | TR-001 | N/A | Training | Dataset | TEST-007 |
| BR-002 | Data validation | US-004 | TR-002 | N/A | Validation | Clean data | TEST-008 |
| BR-003 | EDA | US-003 | TR-004 | N/A | EDA | Plots/report | Review |
| BR-004 | Baseline | US-003 | TR-005 | GET /model | Training | Metrics | Model tests |
| BR-005 | AdaBoost | US-003 | TR-006 | GET /model | Model | Artifact | Model tests |
| BR-006 | Tuning | US-004 | TR-007/008 | GET /model | Experiments | Run metadata | Leakage tests |
| BR-007 | Test evaluation | US-003 | TR-009 | GET /model | Evaluation | Test metrics | TEST-008 |
| BR-008 | Risk tier | US-001 | TR-011 | POST /score | Tier mapper | Threshold | TEST-005 |
| BR-009 | API | US-001 | TR-012 | POST /score | FastAPI | Artifact | TEST-001..004 |
| BR-010 | UI | US-001 | TR-013 | POST /score | React | UI | E2E |
| BR-011 | Model card | US-002 | TR-014 | GET /model-card | Docs | Model card | Review |
