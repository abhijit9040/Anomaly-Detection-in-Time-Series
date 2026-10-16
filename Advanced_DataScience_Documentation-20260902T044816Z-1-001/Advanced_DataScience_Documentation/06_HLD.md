# High-Level Design
## Architecture
```mermaid
flowchart LR
U[Loan Officer / Credit Risk Analyst] --> FE[React + Tailwind]
FE -->|Axios HTTPS| API[FastAPI]
API --> V[Input Validation]
V --> P[Preprocessing Pipeline]
P --> M[AdaBoost Model Artifact]
M --> T[Risk-Tier Mapper]
T --> API --> FE
API --> MM[Model Metadata / Model Card]
DATA[(Give Me Some Credit Dataset)] --> TP[Training Pipeline]
TP --> EDA[EDA + Feature Engineering]
EDA --> BASE[Baseline + Candidate Models]
BASE --> TUNE[CV + Hyperparameter Search]
TUNE --> M
TUNE --> EV[Held-out Test Evaluation]
```
## Responsibilities
React: UI. FastAPI: validation/inference. Pipeline: transformations. AdaBoost: prediction. Tier mapper: operational category. Training pipeline: EDA, features, model selection, tuning and evaluation.

## Scalability
Keep FastAPI stateless. Scale horizontally only when traffic justifies it. Do not introduce microservices or databases without a demonstrated need.

## Reliability
Validate requests, load the artifact at startup, expose health checks, use stable error codes and make prediction deterministic for a fixed artifact/input.

## Security
HTTPS, validation, rate limiting when public, secret management, dependency scanning and no sensitive data in logs.
