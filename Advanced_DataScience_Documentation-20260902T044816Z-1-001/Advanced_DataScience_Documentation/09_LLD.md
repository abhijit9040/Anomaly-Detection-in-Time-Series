# Low-Level Design
## Training
```text
training/
  data/ ingestion.py validation.py split.py
  eda/ distributions.py missingness.py target_analysis.py
  features/ preprocessing.py feature_engineering.py
  models/ baseline.py adaboost.py tuning.py evaluation.py threshold.py
  experiments/
  artifacts/
```

## Backend
```text
backend/app/
  main.py schemas.py
  routes/health.py scoring.py model.py
  services/predictor.py model_registry.py
  middleware/error_handler.py
```

## Frontend
```text
frontend/src/
  components/
  pages/
  services/
  hooks/
  types/
  utils/
```

## Predictor
load_model → validate_features → transform → predict → map_risk_tier → return metadata.

## Sequence
```mermaid
sequenceDiagram
participant U as Analyst
participant R as React
participant F as FastAPI
participant P as Predictor
participant M as AdaBoost
U->>R: Enter applicant data
R->>F: POST /api/v1/score
F->>P: Validate + transform
P->>M: Predict
M-->>P: Score
P->>P: Map tier
P-->>F: Score + tier + version
F-->>R: JSON
R-->>U: Display
```
