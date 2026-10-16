# Technical Requirements Document
## Architecture
Training is a Python data-science workflow producing a versioned artifact. Serving is a FastAPI service. React is the analyst UI.

## Requirements
| ID | Technical Requirement |
|---|---|
| TR-001 | Reproducible ingestion |
| TR-002 | Data/schema validation |
| TR-003 | Leakage-safe preprocessing pipeline |
| TR-004 | EDA artifacts |
| TR-005 | Baseline |
| TR-006 | AdaBoost |
| TR-007 | Hyperparameter optimization |
| TR-008 | CV only within training data |
| TR-009 | Held-out test evaluation |
| TR-010 | Model artifact/versioning |
| TR-011 | Risk-tier mapping |
| TR-012 | FastAPI scoring endpoint |
| TR-013 | React UI |
| TR-014 | Model card |
| TR-015 | Automated tests |

## Split Protocol
80% stratified training / 20% untouched test. Fit preprocessing on training data only. Use CV and tuning only within the 80% training partition. Select final model and threshold, then evaluate on the untouched test set.

## GPU Constraint
AdaBoost with tree-based weak learners is CPU-friendly; GPU should not become an artificial dependency.
